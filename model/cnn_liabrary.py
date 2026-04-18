from torch import nn
import torch
import torch.nn.functional as F


class Conv2d_selfAdapt(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=0,
        relation_bias=False,
        dynamic_depthwise=False,
        local_sparse_attn=False,
        sdm_topk=2,
        attn_topk=4,
        attn_heads=1,
        relation_rich_bias=False,
        dual_path_fusion=False,
        dual_path_gate_map=False,
    ):
        super(Conv2d_selfAdapt, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.t = theta
        self.relation_bias = bool(relation_bias)
        self.dynamic_depthwise = bool(dynamic_depthwise)
        self.local_sparse_attn = bool(local_sparse_attn)
        self.sdm_topk = int(sdm_topk)
        self.attn_topk = int(attn_topk)
        self.attn_heads = int(attn_heads)
        self.relation_rich_bias = bool(relation_rich_bias)
        self.dual_path_fusion = bool(dual_path_fusion)
        self.dual_path_gate_map = bool(dual_path_gate_map)
        if self.dynamic_depthwise and self.local_sparse_attn:
            raise ValueError('dynamic_depthwise and local_sparse_attn cannot be enabled together.')
        if self.sdm_topk <= 0:
            raise ValueError(f'sdm_topk must be > 0, got {self.sdm_topk}')
        if self.attn_heads <= 0:
            raise ValueError(f'attn_heads must be > 0, got {self.attn_heads}')
        if self.local_sparse_attn and in_channels % self.attn_heads != 0:
            raise ValueError(
                f'in_channels({in_channels}) must be divisible by attn_heads({self.attn_heads})'
            )
        self.center_idx = (kernel_size * kernel_size) // 2
        self.neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        self.mask_conv2d = nn.Conv2d(
            in_channels,
            8,
            kernel_size=kernel_size,
            stride=stride,
            padding='same',
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=1)
        self.softArgMax = Soft_argmax(t=self.t, topk=self.sdm_topk)
        self.layerNorm = LayerNorm((1, 8, 1, 1), dim_index=1)
        if self.relation_bias:
            rel_in_channels = 6 if self.relation_rich_bias else 2
            self.rel_bias_scale = nn.Parameter(torch.tensor(0.1))
            self.rel_bias_mlp = nn.Sequential(
                nn.Conv2d(rel_in_channels, 16, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(16, 8, kernel_size=1, bias=True),
            )
        if self.dynamic_depthwise:
            hidden_dim = max(16, in_channels // 8)
            self.dw_kernel_proj = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden_dim, kernel_size * kernel_size, kernel_size=1, bias=True),
            )
            self.dw_temperature = nn.Parameter(torch.tensor(1.0))
        if self.local_sparse_attn:
            self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            if self.attn_heads > 1:
                self.attn_out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            else:
                self.attn_out_proj = nn.Identity()
            self.attn_mix_logit = nn.Parameter(torch.tensor(0.0))
            if self.dual_path_fusion:
                if self.dual_path_gate_map:
                    self.dual_gate_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
                else:
                    self.dual_mix_logit = nn.Parameter(torch.tensor(0.5))
        base_weight = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]],
            requires_grad=False,
        )[None, None, :, :].repeat(in_channels, out_channels, 1, 1)
        self.register_buffer('diff_conv2d_weight', base_weight)
        self.unfold_mask01_x = None

    @staticmethod
    def _build_relation_feat(h, w, device, dtype, rich=False):
        row_idx = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
        col_idx = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
        span_len = col_idx - row_idx
        max_span = float(max(h - 1, 1))
        span_norm = span_len / max_span
        near_diag = (span_len.abs() <= 1).to(dtype)
        if not rich:
            return torch.cat([span_norm, near_diag], dim=1)
        abs_span_norm = span_len.abs() / max_span
        row_norm = (row_idx / max_span).expand(-1, -1, h, w)
        col_norm = (col_idx / max_span).expand(-1, -1, h, w)
        short_span = (span_len.abs() <= 3).to(dtype)
        return torch.cat(
            [span_norm, near_diag, row_norm, col_norm, abs_span_norm, short_span],
            dim=1,
        )

    def _dynamic_depthwise_response(self, x_unfold, x):
        # Input-conditioned local kernel: adaptive alternative to fixed difference kernel.
        raw_kernel = self.dw_kernel_proj(x)
        temperature = torch.clamp(self.dw_temperature.abs(), min=1e-4)
        raw_kernel = raw_kernel / temperature

        suppress_center = torch.zeros_like(raw_kernel)
        suppress_center[:, self.center_idx:self.center_idx + 1, :, :] = -1e4
        dyn_kernel = F.softmax(raw_kernel + suppress_center, dim=1)

        weighted_nb = (x_unfold * dyn_kernel.unsqueeze(1)).sum(dim=2)
        center_feat = x_unfold[:, :, self.center_idx, :, :]
        return center_feat - weighted_nb

    def _local_sparse_attention_response(self, x):
        # Local-window self-attention with sparse top-k selection, then center-minus-context enhancement.
        batch_size, hidden_size, h, w = x.shape
        kernel_size = self.kernel_size
        head_dim = hidden_size // self.attn_heads
        q = self.q_proj(x).view(batch_size, self.attn_heads, head_dim, h, w)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k_unfold = self.unfold(k).reshape(batch_size, hidden_size, kernel_size * kernel_size, h, w)
        k_unfold = k_unfold.view(batch_size, self.attn_heads, head_dim, kernel_size * kernel_size, h, w)
        v_unfold = self.unfold(v).reshape(batch_size, hidden_size, kernel_size * kernel_size, h, w)
        v_unfold = v_unfold.view(batch_size, self.attn_heads, head_dim, kernel_size * kernel_size, h, w)

        logits = (q.unsqueeze(3) * k_unfold).sum(dim=2) / (head_dim ** 0.5 + 1e-12)
        logits = logits[:, :, self.neighbor_indices, :, :]
        if self.relation_bias:
            rel_feat = self._build_relation_feat(
                h,
                w,
                x.device,
                logits.dtype,
                rich=self.relation_rich_bias,
            )
            rel_bias = self.rel_bias_mlp(rel_feat)
            logits = logits + self.rel_bias_scale * rel_bias.unsqueeze(1)

        topk = min(max(self.attn_topk, 1), logits.size(2))
        sparse = soft_topk_mask(logits, topk=topk, temperature=self.t, dim=2)
        temperature = max(float(self.t), 1e-6)
        dense = F.softmax(logits / temperature, dim=2)
        mix = torch.sigmoid(self.attn_mix_logit)
        attn = mix * sparse + (1.0 - mix) * dense

        value_nb = v_unfold[:, :, :, self.neighbor_indices, :, :]
        context = (value_nb * attn.unsqueeze(2)).sum(dim=3)
        center = v_unfold[:, :, :, self.center_idx, :, :]
        response = (center - context).reshape(batch_size, hidden_size, h, w)
        response = self.attn_out_proj(response)
        return response

    def _masked_difference_response(self, x, init_flag):
        batch_size, hidden_size, h, w = x.shape
        kernel_size = self.kernel_size
        x_unfold = self.unfold(x).reshape(batch_size, hidden_size, kernel_size * kernel_size, h, w)
        if init_flag or self.unfold_mask01_x is None:
            mask_x = self.mask_conv2d(x)
            if self.relation_bias:
                rel_feat = self._build_relation_feat(
                    h,
                    w,
                    x.device,
                    mask_x.dtype,
                    rich=self.relation_rich_bias,
                )
                rel_bias = self.rel_bias_mlp(rel_feat)
                mask_x = mask_x + self.rel_bias_scale * rel_bias
            mask_x = self.layerNorm(mask_x)
            mask01_x = self.softArgMax(mask_x)
            self.unfold_mask01_x = mask01_x.unsqueeze(1).reshape(batch_size, 1, 8, h, w)

        mask = self.unfold_mask01_x
        if mask.dtype != x_unfold.dtype:
            mask = mask.to(dtype=x_unfold.dtype)
        x_unfold = x_unfold.clone()
        x_unfold[:, :, [0, 1, 2, 3, 5, 6, 7, 8], :, :] = x_unfold[:, :, [0, 1, 2, 3, 5, 6, 7, 8], :, :] * mask
        if self.dynamic_depthwise:
            return self._dynamic_depthwise_response(x_unfold, x)
        x_with_mask = x_unfold.reshape(batch_size, -1, h * w)
        weight = self.diff_conv2d_weight.view(self.diff_conv2d_weight.size(0), -1).t()
        if weight.dtype != x_with_mask.dtype:
            weight = weight.to(dtype=x_with_mask.dtype)

        out_unf = x_with_mask.transpose(1, 2).matmul(weight).transpose(1, 2)
        output = F.fold(out_unf, (h, w), (1, 1))
        return output

    def forward(self, x, init_flag):
        if self.local_sparse_attn:
            local_response = self._local_sparse_attention_response(x)
            if self.dual_path_fusion:
                base_response = self._masked_difference_response(x, init_flag)
                if self.dual_path_gate_map:
                    mix = torch.sigmoid(self.dual_gate_proj(x))
                else:
                    mix = torch.sigmoid(self.dual_mix_logit)
                return mix * local_response + (1.0 - mix) * base_response
            return local_response
        return self._masked_difference_response(x, init_flag)


class Soft_argmax(nn.Module):
    def __init__(self, t, topk=2):
        super(Soft_argmax, self).__init__()
        self.t = t
        self.topk = topk
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        sparse = soft_topk_mask(x, topk=self.topk, temperature=self.t)
        temperature = max(float(self.t), 1e-6)
        dense = F.softmax(x / temperature, dim=1)
        mix = torch.sigmoid(self.mix_logit)
        return mix * sparse + (1.0 - mix) * dense


def soft_topk_mask(logits, topk=2, temperature=1.0, dim=1):
    temperature = max(float(temperature), 1e-6)
    scaled = logits / temperature
    if topk is None or topk <= 0 or topk >= scaled.size(dim):
        return F.softmax(scaled, dim=dim)
    topk_values, topk_indices = torch.topk(scaled, k=topk, dim=dim)
    topk_probs = F.softmax(topk_values, dim=dim)
    output = torch.zeros_like(scaled)
    output.scatter_(dim, topk_indices, topk_probs)
    return output


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x
