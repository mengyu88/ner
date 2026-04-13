from torch import nn
import torch
import torch.nn.functional as F

from .cnn_liabrary import Conv2d_selfAdapt


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


class MaskConv2d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        padding=1,
        groups=1,
        flag=1,
        dilation=1,
        stride=1,
        theta=1,
        sad_relation_bias=False,
        sad_dynamic_depthwise=False,
        sad_local_sparse_attn=False,
        sad_attn_topk=4,
    ):
        super(MaskConv2d, self).__init__()
        self.in_ch = in_ch
        self.flag = flag
        if flag in (1, 2):
            self.conv2d_3 = Conv2d_selfAdapt(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                groups=groups,
                dilation=dilation,
                stride=stride,
                theta=theta,
                relation_bias=sad_relation_bias,
                dynamic_depthwise=sad_dynamic_depthwise,
                local_sparse_attn=sad_local_sparse_attn,
                attn_topk=sad_attn_topk,
            )
        elif flag in (3, 4):
            self.conv2d_3 = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                groups=groups,
                dilation=dilation,
                stride=stride,
            )

    def forward(self, x, mask, init_flag):
        if self.flag != 4:
            x_1 = x.masked_fill(mask, 0)
        else:
            x_1 = x
        if self.flag in (1, 2):
            return self.conv2d_3(x_1, init_flag)
        return self.conv2d_3(x_1)


def _resize_mask(mask, x):
    if mask.size(-1) == x.size(-1) and mask.size(-2) == x.size(-2):
        return mask
    return F.interpolate(mask.float(), size=x.shape[-2:], mode='nearest').bool()


class _BFMConvBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super(_BFMConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.norm = LayerNorm((1, channels, 1, 1), dim_index=1)
        self.act = nn.GELU()

    def forward(self, x, mask):
        mask = _resize_mask(mask, x)
        y = x.masked_fill(mask, 0)
        y = self.conv(y)
        y = self.norm(y)
        y = self.act(y)
        return y


class PyramidGatedBFM(nn.Module):
    def __init__(self, channels):
        super(PyramidGatedBFM, self).__init__()
        self.enc0 = _BFMConvBlock(channels, stride=1)
        self.enc1 = _BFMConvBlock(channels, stride=2)
        self.bridge = _BFMConvBlock(channels, stride=1)
        self.dec0 = _BFMConvBlock(channels, stride=1)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, mask):
        m0 = _resize_mask(mask, x)
        e0 = self.enc0(x, m0)
        e1 = self.enc1(e0, _resize_mask(m0, e0))
        b = self.bridge(e1, _resize_mask(m0, e1))

        u0 = F.interpolate(b, size=e0.shape[-2:], mode='nearest') + e0
        u0 = self.dec0(u0, _resize_mask(m0, u0))

        gate_in = torch.cat([u0, e0], dim=1).masked_fill(m0, 0)
        gate = torch.sigmoid(self.gate(gate_in))
        fused = gate * u0 + (1.0 - gate) * e0
        out = self.out(fused.masked_fill(m0, 0))
        return out.masked_fill(m0, 0)


class MaskCNN_1(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        depth=3,
        theta=1,
        sad_relation_bias=False,
        sad_dynamic_depthwise=False,
        sad_local_sparse_attn=False,
        sad_attn_topk=4,
    ):
        super(MaskCNN_1, self).__init__()
        self.theta = theta
        self.inchannels = input_channels

        layers1 = []
        layers2 = []
        layers3 = []

        self.c1 = MaskConv2d(
            input_channels,
            input_channels,
            kernel_size=kernel_size,
            padding='same',
            flag=1,
            theta=self.theta,
            sad_relation_bias=sad_relation_bias,
            sad_dynamic_depthwise=sad_dynamic_depthwise,
            sad_local_sparse_attn=sad_local_sparse_attn,
            sad_attn_topk=sad_attn_topk,
        )
        self.f1 = MaskConv2d(input_channels, input_channels, kernel_size=1, padding='same', flag=3)
        layers1.extend(
            [
                self.c1,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
                self.c1,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
                self.f1,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
            ]
        )
        self.c2 = MaskConv2d(
            input_channels,
            input_channels,
            kernel_size=kernel_size,
            padding='same',
            flag=1,
            theta=self.theta,
            sad_relation_bias=sad_relation_bias,
            sad_dynamic_depthwise=sad_dynamic_depthwise,
            sad_local_sparse_attn=sad_local_sparse_attn,
            sad_attn_topk=sad_attn_topk,
        )
        self.f2 = MaskConv2d(input_channels, input_channels, kernel_size=1, padding='same', flag=3)
        layers2.extend(
            [
                self.c2,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
                self.c2,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
                self.f2,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
            ]
        )

        self.bfm = PyramidGatedBFM(input_channels)
        self.cnns1 = nn.ModuleList(layers1)
        self.cnns2 = nn.ModuleList(layers2)
        self.cnns3 = nn.ModuleList(layers3)
        self.lastLinear_dc = nn.Linear(in_features=input_channels * 2, out_features=input_channels)
        torch.nn.init.xavier_normal_(self.lastLinear_dc.weight.data)

    def forward(self, x, mask, train):
        x_1 = x.clone()
        x_2 = x.clone()
        x_3 = x.clone()
        x_4 = x.clone()
        _x1 = x_1
        _x2 = x_2
        _x3 = x_3

        t_hor = []
        t_ver = []
        t_real = []
        cnn_type = [0, 3]

        if 0 in cnn_type:
            direction_flag = True
            direction = 1
            for layer1 in self.cnns1:
                if isinstance(layer1, LayerNorm):
                    x_1 = layer1(x_1)
                elif isinstance(layer1, MaskConv2d):
                    _x1 = x_1
                    x_1 = layer1(x_1, mask, True)
                    if isinstance(layer1.conv2d_3, nn.Conv2d):
                        direction_flag = False
                        x_1 = x_1 + _x1
                    else:
                        layer1.conv2d_3.diff_conv2d_weight = layer1.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_1 = layer1(x_1)
                    if direction_flag:
                        x_1 = x_1 + _x1
                    t_ver.append(x_1)

        if 1 in cnn_type:
            direction_flag = True
            direction = 1
            for layer in self.cnns2:
                if isinstance(layer, LayerNorm):
                    x_2 = layer(x_2)
                elif isinstance(layer, MaskConv2d):
                    _x2 = x_2
                    x_2 = layer(x_2, mask, True)
                    if isinstance(layer.conv2d_3, nn.Conv2d):
                        direction_flag = False
                        x_2 = x_2 + _x2
                    else:
                        layer.conv2d_3.diff_conv2d_weight = layer.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_2 = layer(x_2)
                    if direction_flag:
                        x_2 = x_2 + _x2
                    t_hor.append(x_2)
        if 2 in cnn_type:
            direction_flag = True
            direction = 1
            for layer in self.cnns3:
                if isinstance(layer, LayerNorm):
                    x_3 = layer(x_3)
                elif isinstance(layer, MaskConv2d):
                    _x3 = x_3
                    x_3 = layer(x_3, mask, True)
                    if isinstance(layer.conv2d_3, nn.Conv2d):
                        direction_flag = False
                        x_3 = x_3 + _x3
                    else:
                        layer.conv2d_3.diff_conv2d_weight = layer.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_3 = layer(x_3)
                    if direction_flag:
                        x_3 = x_3 + _x3
                    t_real.append(x_3)

        bfm_feat = self.bfm(x_4, mask)
        linear_atts_dc1 = torch.cat([t_ver[2], bfm_feat], dim=1).masked_fill(mask, 0)
        return linear_atts_dc1


class MaskCNN_2(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        depth=3,
        theta=1,
        sad_relation_bias=False,
        sad_dynamic_depthwise=False,
        sad_local_sparse_attn=False,
        sad_attn_topk=4,
    ):
        super(MaskCNN_2, self).__init__()
        self.theta = theta
        self.inchannels = input_channels

        layers1 = []
        layers2 = []
        layers3 = []

        self.c1 = MaskConv2d(
            input_channels,
            input_channels,
            kernel_size=kernel_size,
            padding='same',
            flag=1,
            theta=self.theta,
            sad_relation_bias=sad_relation_bias,
            sad_dynamic_depthwise=sad_dynamic_depthwise,
            sad_local_sparse_attn=sad_local_sparse_attn,
            sad_attn_topk=sad_attn_topk,
        )
        self.f1 = MaskConv2d(input_channels, input_channels, kernel_size=1, padding='same', flag=3)
        layers1.extend(
            [
                self.c1,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
                self.c1,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
                self.f1,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
            ]
        )
        self.c2 = MaskConv2d(
            input_channels,
            input_channels,
            kernel_size=kernel_size,
            padding='same',
            flag=1,
            theta=self.theta,
            sad_relation_bias=sad_relation_bias,
            sad_dynamic_depthwise=sad_dynamic_depthwise,
            sad_local_sparse_attn=sad_local_sparse_attn,
            sad_attn_topk=sad_attn_topk,
        )
        self.f2 = MaskConv2d(input_channels, input_channels, kernel_size=1, padding='same', flag=3)
        layers2.extend(
            [
                self.c2,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
                self.c2,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
                self.f2,
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU(),
            ]
        )

        self.bfm = PyramidGatedBFM(input_channels)
        self.cnns1 = nn.ModuleList(layers1)
        self.cnns2 = nn.ModuleList(layers2)
        self.cnns3 = nn.ModuleList(layers3)
        self.lastLinear_dc = nn.Linear(in_features=input_channels * 2, out_features=input_channels)
        torch.nn.init.xavier_normal_(self.lastLinear_dc.weight.data)

    def forward(self, x, mask, train):
        x_1 = x.clone()
        x_2 = x.clone()
        x_3 = x.clone()
        x_4 = x.clone()
        _x1 = x_1
        _x2 = x_2
        _x3 = x_3

        t_hor = []
        t_ver = []
        t_real = []
        cnn_type = [0, 1, 3]

        if 0 in cnn_type:
            direction_flag = True
            direction = 1
            for layer1 in self.cnns1:
                if isinstance(layer1, LayerNorm):
                    x_1 = layer1(x_1)
                elif isinstance(layer1, MaskConv2d):
                    _x1 = x_1
                    x_1 = layer1(x_1, mask, True)
                    if isinstance(layer1.conv2d_3, nn.Conv2d):
                        direction_flag = False
                        x_1 = x_1 + _x1
                    else:
                        layer1.conv2d_3.diff_conv2d_weight = layer1.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_1 = layer1(x_1)
                    if direction_flag:
                        x_1 = x_1 + _x1
                    t_ver.append(x_1)

        if 1 in cnn_type:
            direction_flag = True
            direction = 1
            for layer in self.cnns2:
                if isinstance(layer, LayerNorm):
                    x_2 = layer(x_2)
                elif isinstance(layer, MaskConv2d):
                    _x2 = x_2
                    x_2 = layer(x_2, mask, True)
                    if isinstance(layer.conv2d_3, nn.Conv2d):
                        direction_flag = False
                        x_2 = x_2 + _x2
                    else:
                        layer.conv2d_3.diff_conv2d_weight = layer.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_2 = layer(x_2)
                    if direction_flag:
                        x_2 = x_2 + _x2
                    t_hor.append(x_2)
        if 2 in cnn_type:
            direction_flag = True
            direction = 1
            for layer in self.cnns3:
                if isinstance(layer, LayerNorm):
                    x_3 = layer(x_3)
                elif isinstance(layer, MaskConv2d):
                    _x3 = x_3
                    x_3 = layer(x_3, mask, True)
                    if isinstance(layer.conv2d_3, nn.Conv2d):
                        direction_flag = False
                        x_3 = x_3 + _x3
                    else:
                        layer.conv2d_3.diff_conv2d_weight = layer.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_3 = layer(x_3)
                    if direction_flag:
                        x_3 = x_3 + _x3
                    t_real.append(x_3)

        bfm_feat = self.bfm(x_4, mask)
        atts_dc1 = torch.cat([t_ver[2], t_hor[2]], dim=1)
        atts_dc1 = self.lastLinear_dc(atts_dc1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        linear_atts_dc1 = torch.cat([atts_dc1, bfm_feat], dim=1).masked_fill(mask, 0)
        return linear_atts_dc1
