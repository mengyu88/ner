import math
from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask
import torch
import torch.nn.functional as F
from .cnn import MaskCNN_1,MaskCNN_2,DilatedResRefiner
from .multi_head_biaffine3 import MultiHeadBiaffine

try:
    from torch_scatter import scatter_max as _scatter_max
except ImportError:
    _scatter_max = None

try:
    from torch_scatter import scatter_mean as _scatter_mean
except ImportError:
    _scatter_mean = None


def scatter_max(src, index, dim=1):
    if _scatter_max is not None:
        return _scatter_max(src, index=index, dim=dim)

    if index.dim() < src.dim():
        for _ in range(src.dim() - index.dim()):
            index = index.unsqueeze(-1)
        index = index.expand_as(src)

    index = index.long()
    out_shape = list(src.shape)
    out_shape[dim] = int(index.max().item()) + 1
    out = torch.full(out_shape, -float('inf'), dtype=src.dtype, device=src.device)
    out.scatter_reduce_(dim, index, src, reduce='amax', include_self=True)
    out = out.masked_fill(torch.isinf(out), 0)
    return out, None


def scatter_mean(src, index, dim=1):
    if _scatter_mean is not None:
        return _scatter_mean(src, index=index, dim=dim)

    if index.dim() < src.dim():
        for _ in range(src.dim() - index.dim()):
            index = index.unsqueeze(-1)
        index = index.expand_as(src)

    index = index.long()
    out_shape = list(src.shape)
    out_shape[dim] = int(index.max().item()) + 1

    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    count = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    out.scatter_add_(dim, index, src)
    count.scatter_add_(dim, index, torch.ones_like(src))
    out = out / count.clamp_min(1.0)
    return out


def apply_rotary_position(x):
    """
    x: [batch, seq_len, num_heads, head_dim]
    """
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError(f'RoPE requires even head_dim, got {head_dim}')

    seq_len = x.size(1)
    pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim))
    sinusoid = torch.einsum('l,d->ld', pos, inv_freq)
    sin = sinusoid.sin().unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
    cos = sinusoid.cos().unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)


class GeGLUProjector(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.4):
        super(GeGLUProjector, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, out_features * 2)
        torch.nn.init.xavier_normal_(self.proj.weight.data)
        if self.proj.bias is not None:
            torch.nn.init.zeros_(self.proj.bias.data)

    def forward(self, x):
        x = self.dropout(x)
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.gelu(gate)


class CNNNer(nn.Module):
    def __init__(self, model_name, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=4, cnn_depth=3,
                 n_layer=2,separateness_rate=0.1,theta=1,loss_theta=1,
                 size_feature_type='embed',
                 word_pooling='max',pool_gate_type='scalar',refiner_type='maskcnn',
                 pair_scorer='biaffine',gp_hidden=8,lowrank_dim=64,
                 mlp_type='mlp',fusion_type='sum',bfm_type='legacy',
                 sdm_mask_type='hard_gumbel', sdm_topk=2):
        super(CNNNer, self).__init__()
        self.mdim =(cnn_dim) 
        self.num_ner_tag = num_ner_tag
        self.cnn_dim = cnn_dim
        self.separateness_rate = separateness_rate
        self.cnn_dim = cnn_dim
        self.loss_theta = loss_theta
        self.n_layer = n_layer
        if size_feature_type not in ('embed', 'bias'):
            raise ValueError(f"size_feature_type must be one of ['embed', 'bias'], got {size_feature_type}")
        self.size_feature_type = size_feature_type
        if refiner_type not in ('maskcnn', 'dilated'):
            raise ValueError(f"refiner_type must be one of ['maskcnn', 'dilated'], got {refiner_type}")
        self.refiner_type = refiner_type
        if bfm_type not in ('legacy', 'pyramid', 'pyramid_gated'):
            raise ValueError(
                "bfm_type must be one of ['legacy', 'pyramid', 'pyramid_gated'], "
                f"got {bfm_type}"
            )
        self.bfm_type = bfm_type
        if sdm_mask_type not in ('hard_gumbel', 'soft_topk', 'softmax'):
            raise ValueError(
                "sdm_mask_type must be one of ['hard_gumbel', 'soft_topk', 'softmax'], "
                f"got {sdm_mask_type}"
            )
        if sdm_topk <= 0:
            raise ValueError(f"sdm_topk must be > 0, got {sdm_topk}")
        self.sdm_mask_type = sdm_mask_type
        self.sdm_topk = sdm_topk
        if pair_scorer not in ('biaffine', 'rope_gp', 'lowrank'):
            raise ValueError(f"pair_scorer must be one of ['biaffine', 'rope_gp', 'lowrank'], got {pair_scorer}")
        self.pair_scorer = pair_scorer
        if gp_hidden % 2 != 0:
            raise ValueError(f'gp_hidden must be even for RoPE, got {gp_hidden}')
        self.gp_hidden = gp_hidden
        if lowrank_dim <= 0:
            raise ValueError(f'lowrank_dim must be > 0, got {lowrank_dim}')
        self.lowrank_dim = lowrank_dim
        if mlp_type not in ('mlp', 'geglu'):
            raise ValueError(f"mlp_type must be one of ['mlp', 'geglu'], got {mlp_type}")
        self.mlp_type = mlp_type
        if fusion_type not in ('sum', 'adaptive'):
            raise ValueError(f"fusion_type must be one of ['sum', 'adaptive'], got {fusion_type}")
        self.fusion_type = fusion_type
        if word_pooling not in ('max', 'mean', 'mix'):
            raise ValueError(f"word_pooling must be one of ['max', 'mean', 'mix'], got {word_pooling}")
        self.word_pooling = word_pooling
        if pool_gate_type not in ('scalar', 'channel'):
            raise ValueError(f"pool_gate_type must be one of ['scalar', 'channel'], got {pool_gate_type}")
        self.pool_gate_type = pool_gate_type
        if self.fusion_type == 'adaptive':
            self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        # self.param_span= nn.Parameter(torch.randn(2,cnn_dim)/20,requires_grad=True)
        self.pretrain_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.pretrain_model.config.hidden_size
        if self.word_pooling == 'mix':
            if self.pool_gate_type == 'channel':
                self.pool_mix_logit = nn.Parameter(torch.zeros(1, 1, hidden_size))
            else:
                self.pool_mix_logit = nn.Parameter(torch.tensor(0.0))
        if size_embed_dim != 0:
            n_pos = 30
            if self.size_feature_type == 'embed':
                self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            else:
                self.size_bias_embedding = torch.nn.Embedding(n_pos, cnn_dim)
                self.size_bias_scale = nn.Parameter(torch.tensor(1.0))
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            if self.size_feature_type == 'embed':
                hsz = biaffine_size * 2 + size_embed_dim + 2
            else:
                hsz = biaffine_size * 2 + 2
        else:
            hsz = biaffine_size * 2 + 2
        biaffine_input_size = hidden_size
        self.dropout = nn.Dropout(logit_drop)
        self.dropout1 = nn.Dropout(logit_drop)
        if self.mlp_type == 'geglu':
            self.head_mlp = GeGLUProjector(biaffine_input_size, biaffine_size, dropout=0.4)
            self.tail_mlp = GeGLUProjector(biaffine_input_size, biaffine_size, dropout=0.4)
        else:
            self.head_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(biaffine_input_size, biaffine_size),
                nn.GELU(),
            )
            self.tail_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(biaffine_input_size, biaffine_size),
                nn.GELU(),
            )
        if self.pair_scorer == 'rope_gp':
            self.q_proj = nn.Linear(biaffine_size, cnn_dim * gp_hidden, bias=False)
            self.k_proj = nn.Linear(biaffine_size, cnn_dim * gp_hidden, bias=False)
            torch.nn.init.xavier_normal_(self.q_proj.weight.data)
            torch.nn.init.xavier_normal_(self.k_proj.weight.data)
        elif self.pair_scorer == 'lowrank':
            self.lowrank_head = nn.Linear(biaffine_size, lowrank_dim, bias=False)
            self.lowrank_tail = nn.Linear(biaffine_size, lowrank_dim, bias=False)
            self.lowrank_rel = nn.Parameter(torch.empty(cnn_dim, lowrank_dim))
            self.lowrank_bias = nn.Parameter(torch.zeros(cnn_dim))
            torch.nn.init.xavier_normal_(self.lowrank_head.weight.data)
            torch.nn.init.xavier_normal_(self.lowrank_tail.weight.data)
            torch.nn.init.xavier_normal_(self.lowrank_rel.data)
        elif n_head > 0:
            self.biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth > 0:
            if self.refiner_type == 'dilated':
                self.cnn1 = DilatedResRefiner(cnn_dim, depth=cnn_depth, dropout=logit_drop)
            elif self.n_layer == 1:
                self.cnn1 = MaskCNN_1(
                    cnn_dim,
                    cnn_dim,
                    kernel_size=kernel_size,
                    depth=cnn_depth,
                    theta=theta,
                    bfm_type=bfm_type,
                    sdm_mask_type=sdm_mask_type,
                    sdm_topk=sdm_topk,
                )
            elif self.n_layer == 2:
                self.cnn1 = MaskCNN_2(
                    cnn_dim,
                    cnn_dim,
                    kernel_size=kernel_size,
                    depth=cnn_depth,
                    theta=theta,
                    bfm_type=bfm_type,
                    sdm_mask_type=sdm_mask_type,
                    sdm_topk=sdm_topk,
                )
            else:
                raise ValueError(f"Unsupported n_layer={self.n_layer} for refiner_type=maskcnn")
        self.down_fc = nn.Linear(cnn_dim*3, num_ner_tag)
        torch.nn.init.xavier_normal_(self.down_fc.weight.data)
        self.logit_drop = logit_drop
    def forward(self, input_ids, bpe_len, indexes, matrix,raw_words):

        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        if self.word_pooling == 'mean':
            word_states = scatter_mean(last_hidden_states, index=indexes, dim=1)
        elif self.word_pooling == 'mix':
            mean_states = scatter_mean(last_hidden_states, index=indexes, dim=1)
            max_states = scatter_max(last_hidden_states, index=indexes, dim=1)[0]
            mix_weight = torch.sigmoid(self.pool_mix_logit)
            word_states = mix_weight * max_states + (1.0 - mix_weight) * mean_states
        else:
            word_states = scatter_max(last_hidden_states, index=indexes, dim=1)[0]
        state = word_states[:,1:]
        lengths, _ = indexes.max(dim=-1)
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if self.pair_scorer == 'rope_gp':
            bsz, word_len, _ = head_state.size()
            q = self.q_proj(head_state).reshape(bsz, word_len, self.cnn_dim, self.gp_hidden)
            k = self.k_proj(tail_state).reshape(bsz, word_len, self.cnn_dim, self.gp_hidden)
            q = apply_rotary_position(q)
            k = apply_rotary_position(k)
            scores1 = torch.einsum('blhd,bkhd->bhlk', q, k) / math.sqrt(self.gp_hidden)
        elif self.pair_scorer == 'lowrank':
            h_lr = self.lowrank_head(head_state)
            t_lr = self.lowrank_tail(tail_state)
            scores1 = torch.einsum('blr,or,bkr->bolk', h_lr, self.lowrank_rel, t_lr)
            scores1 = scores1 + self.lowrank_bias.view(1, -1, 1, 1)
        elif hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.biaffine(head_state, tail_state)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([head_state.unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 tail_state.unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)
        mask = seq_len_to_mask(lengths)  # bsz x length x length
        mask = mask[:, None] * mask.unsqueeze(-1)
        pad_mask = mask[:, None].eq(0)
        # pad_mask1 = pad_mask
        pad_mask1 = pad_mask * torch.tril(pad_mask).ne(0)
        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat(
                [self.dropout(affined_cat), self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)],
                dim=-1)

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
        if hasattr(self, 'size_bias_embedding'):
            size_bias = self.size_bias_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            size_bias = size_bias.permute(2, 0, 1).unsqueeze(0)  # [1, C, L, L]
            scores2 = scores2 + self.size_bias_scale * size_bias
        if self.fusion_type == 'adaptive':
            fusion_weight = torch.sigmoid(self.fusion_logit)
            scores = fusion_weight * scores1 + (1.0 - fusion_weight) * scores2
        else:
            scores = scores2 + scores1  # bsz x dim x L x L 
        
        if hasattr(self, 'cnn1'):
            if self.logit_drop != 0:
                scores = F.dropout(scores, p=self.logit_drop, training=self.training)
            u_scores1 = scores.masked_fill(pad_mask1, 0)
            u_score1= self.cnn1(u_scores1, pad_mask1,self.training)

        u_score = torch.concat([scores, u_score1],dim=1)
        final_score = self.down_fc(u_score.permute(0, 2, 3, 1))
        assert final_score.size(-1) == matrix.size(-1)
        if self.training:
            flat_scores = final_score.reshape(-1)
            mask = matrix.reshape(-1).ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, matrix.reshape(-1).float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()
            return {'loss':loss}
        return {'scores': final_score}
