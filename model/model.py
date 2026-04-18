from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask
import torch
import torch.nn.functional as F

from .cnn import MaskCNN_1, MaskCNN_2
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


class ResidualMLPScoreHead(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, hidden_ratio=1.5):
        super(ResidualMLPScoreHead, self).__init__()
        hidden_dim = max(int(in_features * hidden_ratio), out_features)
        self.base = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(in_features)
        self.up = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, out_features)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

        torch.nn.init.xavier_normal_(self.base.weight.data)
        torch.nn.init.xavier_normal_(self.up.weight.data)
        torch.nn.init.xavier_normal_(self.down.weight.data)
        if self.base.bias is not None:
            torch.nn.init.zeros_(self.base.bias.data)
        if self.up.bias is not None:
            torch.nn.init.zeros_(self.up.bias.data)
        if self.down.bias is not None:
            torch.nn.init.zeros_(self.down.bias.data)

    def forward(self, x):
        base = self.base(x)
        y = self.norm(x)
        y = self.up(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.down(y)
        return base + torch.tanh(self.res_scale) * y


class CNNNer(nn.Module):
    def __init__(
        self,
        model_name,
        num_ner_tag,
        cnn_dim=200,
        biaffine_size=200,
        size_embed_dim=0,
        logit_drop=0,
        kernel_size=3,
        n_head=4,
        cnn_depth=3,
        n_layer=2,
        separateness_rate=0.1,
        theta=1,
        loss_theta=1,
        size_feature_type='embed',
        word_pooling='max',
        pool_gate_type='scalar',
        refiner_type='maskcnn',
        span_encoder_type='none',
        pair_scorer='biaffine',
        gp_hidden=8,
        lowrank_dim=64,
        mlp_type='mlp',
        fusion_type='sum',
        bfm_type='pyramid_gated',
        sdm_mask_type='soft_topk_mix',
        sdm_topk=2,
        head_type='residual_mlp',
        sad_relation_bias=False,
        sad_dynamic_depthwise=False,
        sad_local_sparse_attn=False,
        sad_attn_topk=4,
        sad_attn_heads=1,
        sad_relation_rich_bias=False,
        sad_dual_path_fusion=False,
        sad_dual_path_gate_map=False,
        boundary_refine=False,
        boundary_scale=0.1,
        boundary_loss_weight=0.2,
        char_branch=False,
        char_vocab_size=2048,
        char_dim=32,
        char_max_len=16,
        char_dropout=0.1,
        class_pos_weights=None,
        short_span_pos_boost=1.0,
        short_span_neg_boost=1.0,
        short_span_max_len=2,
    ):
        super(CNNNer, self).__init__()
        self.mdim = cnn_dim
        self.num_ner_tag = num_ner_tag
        self.cnn_dim = cnn_dim
        self.separateness_rate = separateness_rate
        self.loss_theta = loss_theta
        self.n_layer = n_layer
        self.word_pooling = word_pooling
        self.pool_gate_type = pool_gate_type
        self.logit_drop = logit_drop
        self.sad_relation_bias = bool(sad_relation_bias)
        self.sad_dynamic_depthwise = bool(sad_dynamic_depthwise)
        self.sad_local_sparse_attn = bool(sad_local_sparse_attn)
        self.sdm_topk = int(sdm_topk)
        self.sad_attn_topk = int(sad_attn_topk)
        self.sad_attn_heads = int(sad_attn_heads)
        self.sad_relation_rich_bias = bool(sad_relation_rich_bias)
        self.sad_dual_path_fusion = bool(sad_dual_path_fusion)
        self.sad_dual_path_gate_map = bool(sad_dual_path_gate_map)
        self.boundary_refine = bool(boundary_refine)
        self.boundary_scale = float(boundary_scale)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.char_branch = bool(char_branch)
        self.char_vocab_size = int(char_vocab_size)
        self.char_max_len = int(char_max_len)
        self.short_span_pos_boost = float(short_span_pos_boost)
        self.short_span_neg_boost = float(short_span_neg_boost)
        self.short_span_max_len = int(short_span_max_len)

        if class_pos_weights is not None:
            class_pos_weights = torch.as_tensor(class_pos_weights, dtype=torch.float)
            if class_pos_weights.numel() != num_ner_tag:
                raise ValueError(
                    f'class_pos_weights length {class_pos_weights.numel()} != num_ner_tag {num_ner_tag}'
                )
            self.register_buffer('class_pos_weights', class_pos_weights.view(1, 1, 1, num_ner_tag))
        else:
            self.class_pos_weights = None

        if size_feature_type not in ('embed', 'bias'):
            raise ValueError(f"size_feature_type must be one of ['embed', 'bias'], got {size_feature_type}")
        if word_pooling not in ('max', 'mean', 'mix'):
            raise ValueError(f"word_pooling must be one of ['max', 'mean', 'mix'], got {word_pooling}")
        if pool_gate_type not in ('scalar', 'channel'):
            raise ValueError(f"pool_gate_type must be one of ['scalar', 'channel'], got {pool_gate_type}")

        fixed_options = {
            'refiner_type': ('maskcnn', refiner_type),
            'span_encoder_type': ('none', span_encoder_type),
            'pair_scorer': ('biaffine', pair_scorer),
            'mlp_type': ('mlp', mlp_type),
            'fusion_type': ('sum', fusion_type),
            'head_type': ('residual_mlp', head_type),
            'bfm_type': ('pyramid_gated', bfm_type),
            'sdm_mask_type': ('soft_topk_mix', sdm_mask_type),
        }
        for key, (expected, actual) in fixed_options.items():
            if actual != expected:
                raise ValueError(f'{key} is fixed to {expected}, got {actual}')
        if self.sdm_topk <= 0:
            raise ValueError(f'sdm_topk must be > 0, got {self.sdm_topk}')

        self.pretrain_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.pretrain_model.config.hidden_size

        if self.word_pooling == 'mix':
            if self.pool_gate_type == 'channel':
                self.pool_mix_logit = nn.Parameter(torch.zeros(1, 1, hidden_size))
            else:
                self.pool_mix_logit = nn.Parameter(torch.tensor(0.0))

        if size_embed_dim != 0:
            n_pos = 30
            if size_feature_type == 'embed':
                self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
                hsz = biaffine_size * 2 + size_embed_dim + 2
            else:
                self.size_bias_embedding = torch.nn.Embedding(n_pos, cnn_dim)
                self.size_bias_scale = nn.Parameter(torch.tensor(1.0))
                hsz = biaffine_size * 2 + 2
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', _span_size_ids.long())
        else:
            hsz = biaffine_size * 2 + 2

        self.dropout = nn.Dropout(logit_drop)
        self.dropout1 = nn.Dropout(logit_drop)
        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, biaffine_size),
            nn.GELU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, biaffine_size),
            nn.GELU(),
        )
        if n_head > 0:
            self.biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)

        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth > 0:
            if self.n_layer == 1:
                self.cnn1 = MaskCNN_1(
                    cnn_dim,
                    cnn_dim,
                    kernel_size=kernel_size,
                    depth=cnn_depth,
                    theta=theta,
                    sad_relation_bias=self.sad_relation_bias,
                    sad_dynamic_depthwise=self.sad_dynamic_depthwise,
                    sad_local_sparse_attn=self.sad_local_sparse_attn,
                    sdm_topk=self.sdm_topk,
                    sad_attn_topk=self.sad_attn_topk,
                    sad_attn_heads=self.sad_attn_heads,
                    sad_relation_rich_bias=self.sad_relation_rich_bias,
                    sad_dual_path_fusion=self.sad_dual_path_fusion,
                    sad_dual_path_gate_map=self.sad_dual_path_gate_map,
                )
            elif self.n_layer == 2:
                self.cnn1 = MaskCNN_2(
                    cnn_dim,
                    cnn_dim,
                    kernel_size=kernel_size,
                    depth=cnn_depth,
                    theta=theta,
                    sad_relation_bias=self.sad_relation_bias,
                    sad_dynamic_depthwise=self.sad_dynamic_depthwise,
                    sad_local_sparse_attn=self.sad_local_sparse_attn,
                    sdm_topk=self.sdm_topk,
                    sad_attn_topk=self.sad_attn_topk,
                    sad_attn_heads=self.sad_attn_heads,
                    sad_relation_rich_bias=self.sad_relation_rich_bias,
                    sad_dual_path_fusion=self.sad_dual_path_fusion,
                    sad_dual_path_gate_map=self.sad_dual_path_gate_map,
                )
            else:
                raise ValueError(f'Unsupported n_layer={self.n_layer} for refiner_type=maskcnn')
        self.score_head = ResidualMLPScoreHead(
            in_features=cnn_dim * 3,
            out_features=num_ner_tag,
            dropout=logit_drop,
            hidden_ratio=1.5,
        )
        if self.boundary_refine:
            self.boundary_start = nn.Linear(hidden_size, num_ner_tag)
            self.boundary_end = nn.Linear(hidden_size, num_ner_tag)
            torch.nn.init.xavier_normal_(self.boundary_start.weight.data)
            torch.nn.init.xavier_normal_(self.boundary_end.weight.data)
            if self.boundary_start.bias is not None:
                torch.nn.init.zeros_(self.boundary_start.bias.data)
            if self.boundary_end.bias is not None:
                torch.nn.init.zeros_(self.boundary_end.bias.data)

        if self.char_branch:
            self.char_embedding = nn.Embedding(self.char_vocab_size, char_dim)
            self.char_proj = nn.Linear(char_dim, hidden_size)
            self.char_gate = nn.Linear(hidden_size * 2, hidden_size)
            self.char_drop = nn.Dropout(char_dropout)
            torch.nn.init.xavier_normal_(self.char_embedding.weight.data)
            torch.nn.init.xavier_normal_(self.char_proj.weight.data)
            torch.nn.init.xavier_normal_(self.char_gate.weight.data)
            if self.char_proj.bias is not None:
                torch.nn.init.zeros_(self.char_proj.bias.data)
            if self.char_gate.bias is not None:
                torch.nn.init.zeros_(self.char_gate.bias.data)

    def _build_char_tensors(self, raw_words, batch_size, seq_len, device):
        # Build lightweight character ids directly from raw words in each batch.
        char_ids = torch.zeros(
            (batch_size, seq_len, self.char_max_len),
            dtype=torch.long,
            device=device,
        )
        char_mask = torch.zeros(
            (batch_size, seq_len, self.char_max_len),
            dtype=torch.float,
            device=device,
        )
        if raw_words is None:
            return char_ids, char_mask

        max_batch = min(batch_size, len(raw_words))
        for b in range(max_batch):
            words = raw_words[b]
            if words is None:
                continue
            if not isinstance(words, (list, tuple)):
                continue
            max_words = min(seq_len, len(words))
            for i in range(max_words):
                token = str(words[i])
                if len(token) == 0:
                    continue
                token = token[: self.char_max_len]
                for j, ch in enumerate(token):
                    char_ids[b, i, j] = ord(ch) % self.char_vocab_size
                    char_mask[b, i, j] = 1.0
        return char_ids, char_mask

    def forward(self, input_ids, bpe_len, indexes, matrix, raw_words):
        attention_mask = seq_len_to_mask(bpe_len)
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

        state = word_states[:, 1:]
        if self.char_branch:
            char_ids, char_mask = self._build_char_tensors(
                raw_words=raw_words,
                batch_size=state.size(0),
                seq_len=state.size(1),
                device=state.device,
            )
            char_embed = self.char_embedding(char_ids)
            char_mask = char_mask.unsqueeze(-1)
            char_state = (char_embed * char_mask).sum(dim=2) / char_mask.sum(dim=2).clamp_min(1.0)
            char_state = self.char_proj(self.char_drop(char_state))
            gate = torch.sigmoid(self.char_gate(torch.cat([state, char_state], dim=-1)))
            state = state + gate * char_state

        lengths, _ = indexes.max(dim=-1)
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.biaffine(head_state, tail_state)

        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat(
            [
                head_state.unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                tail_state.unsqueeze(1).expand(-1, head_state.size(1), -1, -1),
            ],
            dim=-1,
        )
        mask = seq_len_to_mask(lengths)
        mask = mask[:, None] * mask.unsqueeze(-1)
        pad_mask = mask[:, None].eq(0)
        pad_mask1 = pad_mask * torch.tril(pad_mask).ne(0)
        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[: state.size(1), : state.size(1)])
            affined_cat = torch.cat(
                [
                    self.dropout(affined_cat),
                    self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1),
                ],
                dim=-1,
            )

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)
        if hasattr(self, 'size_bias_embedding'):
            size_bias = self.size_bias_embedding(self.span_size_ids[: state.size(1), : state.size(1)])
            size_bias = size_bias.permute(2, 0, 1).unsqueeze(0)
            scores2 = scores2 + self.size_bias_scale * size_bias
        scores = scores2 + scores1

        if hasattr(self, 'cnn1'):
            if self.logit_drop != 0:
                scores = F.dropout(scores, p=self.logit_drop, training=self.training)
            u_scores1 = scores.masked_fill(pad_mask1, 0)
            u_score1 = self.cnn1(u_scores1, pad_mask1, self.training)

        u_score = torch.concat([scores, u_score1], dim=1)
        final_score = self.score_head(u_score.permute(0, 2, 3, 1))
        if self.boundary_refine:
            start_logits = self.boundary_start(state)  # [B, L, C]
            end_logits = self.boundary_end(state)      # [B, L, C]
            boundary_bias = 0.5 * (start_logits.unsqueeze(2) + end_logits.unsqueeze(1))
            final_score = final_score + self.boundary_scale * boundary_bias
        assert final_score.size(-1) == matrix.size(-1)
        if self.training:
            targets = matrix.float()
            valid_mask = targets.ne(-100).float()
            flat_scores = final_score.reshape(-1)
            flat_targets = targets.reshape(-1)
            mask = valid_mask.view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_targets, reduction='none')
            if self.loss_theta != 1:
                pos_w = torch.where(flat_targets > 0.5, self.loss_theta, 1.0)
                flat_loss = flat_loss * pos_w
            if self.class_pos_weights is not None:
                pos_cls_w = torch.where(
                    targets > 0.5,
                    self.class_pos_weights.expand_as(targets),
                    torch.ones_like(targets),
                )
                flat_loss = flat_loss * pos_cls_w.reshape(-1)
            if (
                self.short_span_max_len > 0
                and (self.short_span_pos_boost != 1.0 or self.short_span_neg_boost != 1.0)
            ):
                seq_len = targets.size(1)
                pos = torch.arange(seq_len, device=targets.device)
                span_len = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() + 1
                short_mask = span_len.le(self.short_span_max_len).view(1, seq_len, seq_len, 1).float()
                weight = torch.ones_like(targets)
                if self.short_span_pos_boost != 1.0:
                    weight = torch.where(
                        (targets > 0.5) & (short_mask > 0.5),
                        torch.full_like(weight, self.short_span_pos_boost),
                        weight,
                    )
                if self.short_span_neg_boost != 1.0:
                    weight = torch.where(
                        (targets <= 0.5) & (short_mask > 0.5),
                        torch.full_like(weight, self.short_span_neg_boost),
                        weight,
                    )
                flat_loss = flat_loss * weight.reshape(-1)
            span_loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

            if self.boundary_refine and self.boundary_loss_weight > 0:
                with torch.no_grad():
                    # Only use upper-triangle positives to avoid duplicate symmetric labels.
                    positive_pairs = matrix.gt(0.5)
                    tri = torch.triu(
                        torch.ones((state.size(1), state.size(1)), device=matrix.device, dtype=torch.bool)
                    )
                    tri = tri.unsqueeze(0).unsqueeze(-1)
                    positive_pairs = positive_pairs & tri
                    start_target = positive_pairs.any(dim=2).float()
                    end_target = positive_pairs.any(dim=1).float()
                    token_mask = seq_len_to_mask(lengths).float().unsqueeze(-1)

                start_loss = F.binary_cross_entropy_with_logits(start_logits, start_target, reduction='none')
                end_loss = F.binary_cross_entropy_with_logits(end_logits, end_target, reduction='none')
                start_loss = (start_loss * token_mask).sum() / token_mask.sum().clamp_min(1.0)
                end_loss = (end_loss * token_mask).sum() / token_mask.sum().clamp_min(1.0)
                loss = span_loss + self.boundary_loss_weight * 0.5 * (start_loss + end_loss)
            else:
                loss = span_loss
            return {'loss': loss}
        return {'scores': final_score}
