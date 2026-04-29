from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask
from torch_scatter import scatter_max
import torch
import torch.nn.functional as F
from .cnn import MaskCNN_1,MaskCNN_2
from .multi_head_biaffine3 import MultiHeadBiaffine


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
    def __init__(self, model_name, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=4, cnn_depth=3, n_layer=2,
                 separateness_rate=0.1, theta=1, loss_theta=1, sad_topk=2,
                 sad_attn_dim=None, sad_use_rel_bias=True, sad_gate=True,
                 head_type='linear'):
        super(CNNNer, self).__init__()
        self.mdim =(cnn_dim) 
        self.num_ner_tag = num_ner_tag
        self.cnn_dim = cnn_dim
        self.separateness_rate = separateness_rate
        self.cnn_dim = cnn_dim
        self.loss_theta = loss_theta
        self.n_layer = n_layer
        if head_type not in ('linear', 'residual_mlp'):
            raise ValueError(f"head_type must be one of ['linear', 'residual_mlp'], got {head_type}")
        self.head_type = head_type
        # self.param_span= nn.Parameter(torch.randn(2,cnn_dim)/20,requires_grad=True)
        self.pretrain_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.pretrain_model.config.hidden_size
        if size_embed_dim != 0:
            n_pos = 30
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size * 2 + size_embed_dim + 2 
        else:
            hsz = biaffine_size * 2 + 2
        biaffine_input_size = hidden_size
        self.dropout = nn.Dropout(logit_drop)
        self.dropout1 = nn.Dropout(logit_drop)
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
                    sad_topk=sad_topk,
                    sad_attn_dim=sad_attn_dim,
                    sad_use_rel_bias=sad_use_rel_bias,
                    sad_gate=sad_gate,
                )
            elif self.n_layer == 2:
                self.cnn1 = MaskCNN_2(
                    cnn_dim,
                    cnn_dim,
                    kernel_size=kernel_size,
                    depth=cnn_depth,
                    theta=theta,
                    sad_topk=sad_topk,
                    sad_attn_dim=sad_attn_dim,
                    sad_use_rel_bias=sad_use_rel_bias,
                    sad_gate=sad_gate,
                )
        if self.head_type == 'residual_mlp':
            self.score_head = ResidualMLPScoreHead(
                in_features=cnn_dim * 3,
                out_features=num_ner_tag,
                dropout=logit_drop,
                hidden_ratio=1.5,
            )
        else:
            self.score_head = nn.Linear(cnn_dim * 3, num_ner_tag)
            torch.nn.init.xavier_normal_(self.score_head.weight.data)
        self.logit_drop = logit_drop
    def forward(self, input_ids, bpe_len, indexes, matrix,raw_words):

        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        scat_max  = scatter_max(last_hidden_states, index=indexes, dim=1)[0]  # bsz x word_len x hidden_size
        state = scat_max[:,1:]
        lengths, _ = indexes.max(dim=-1)
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if hasattr(self, 'U'):
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
        scores = scores2 + scores1  # bsz x dim x L x L 
        
        if hasattr(self, 'cnn1'):
            if self.logit_drop != 0:
                scores = F.dropout(scores, p=self.logit_drop, training=self.training)
            u_scores1 = scores.masked_fill(pad_mask1, 0)
            u_score1= self.cnn1(u_scores1, pad_mask1,self.training)

        u_score = torch.concat([scores, u_score1],dim=1)
        final_score = self.score_head(u_score.permute(0, 2, 3, 1))
        assert final_score.size(-1) == matrix.size(-1)
        if self.training:
            flat_scores = final_score.reshape(-1)
            mask = matrix.reshape(-1).ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, matrix.reshape(-1).float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()
            return {'loss':loss}
        return {'scores': final_score}
