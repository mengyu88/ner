import numpy
from torch import nn
import torch
import numpy as np
import copy
import math

import torch.nn.functional as F
from sparsemax import Sparsemax

class Conv2d_selfAdapt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0,
                 sdm_mask_type='hard_gumbel', sdm_topk=2):
        super(Conv2d_selfAdapt, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.t = theta
        self.mask_conv2d = nn.Conv2d(in_channels, 8, kernel_size=kernel_size, stride=stride, padding="same",
                              dilation=dilation, groups=groups, bias=bias)
        # self.filter_conv2d = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, stride=stride, padding="same",
        #                       dilation=dilation, groups=groups, bias=bias)
        self.unfold = nn.Unfold(kernel_size=kernel_size,dilation=dilation,padding=1)
        self.softArgMax = Soft_argmax(t=self.t, mask_type=sdm_mask_type, topk=sdm_topk)
        self.layerNorm = LayerNorm((1, 8, 1, 1), dim_index=1)
        base_weight = torch.tensor(
            [[-1., -1., -1.], [-1., 1., -1.], [-1., -1., -1.]],
            requires_grad=False,
        )[None, None, :, :].repeat(in_channels, out_channels, 1, 1)
        self.register_buffer('diff_conv2d_weight', base_weight)
        self.unfold_mask01_x = None
    def forward(self, x ,init_flag): 
        batch_size,hidden_size, h, w =x.shape
        kernel_size = self.kernel_size
        x_unfold = self.unfold(x).reshape(batch_size, hidden_size, kernel_size*kernel_size,h,w)
        if init_flag:
            mask_x = self.mask_conv2d(x)
            mask_x = self.layerNorm(mask_x)
            mask01_x = self.softArgMax(mask_x) # b kernel*kernel h w
            self.unfold_mask01_x = mask01_x.unsqueeze(1).reshape(batch_size,1, 8,h,w)

        x_unfold[:,:,[0,1,2,3,5,6,7,8],:,:] = x_unfold[:,:,[0,1,2,3,5,6,7,8],:,:] * self.unfold_mask01_x 
        x_with_mask = x_unfold.reshape(batch_size,-1,h*w)
        weight = self.diff_conv2d_weight.view(self.diff_conv2d_weight.size(0), -1).t()

        #  out_channel * kernel * kernel , in_channel
        out_unf = x_with_mask.transpose(1, 2).matmul(weight).transpose(1, 2) # bs,in_channel,h*w

        output = F.fold(out_unf,(h,w),(1,1))
        return output


class Soft_argmax(nn.Module):
    def __init__(self, t, mask_type='hard_gumbel', topk=2) -> None:
        super(Soft_argmax,self).__init__()
        self.t = t
        self.mask_type = mask_type
        self.topk = topk
        if self.mask_type == 'soft_topk_mix':
            # Learnable global mixture weight:
            # 1.0 -> sparse top-k mask, 0.0 -> dense softmax mask.
            self.mix_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('mix_logit', None)

    def forward(self,x):
        if self.mask_type == 'soft_topk':
            return soft_topk_mask(x, topk=self.topk, temperature=self.t)
        if self.mask_type == 'soft_topk_mix':
            sparse = soft_topk_mask(x, topk=self.topk, temperature=self.t)
            temperature = max(float(self.t), 1e-6)
            dense = F.softmax(x / temperature, dim=1)
            mix = torch.sigmoid(self.mix_logit)
            return mix * sparse + (1.0 - mix) * dense
        if self.mask_type == 'softmax':
            temperature = max(float(self.t), 1e-6)
            return F.softmax(x / temperature, dim=1)
        x_gumbel = gumbel_softmax_sample(x,self.t)
        argmax = torch.zeros_like(x)
        argmax = argmax.scatter_(1,torch.argmax(x_gumbel,1).unsqueeze(1),1)
        c = argmax - x_gumbel
        c = c.detach()
        output = c + x_gumbel
        return output


def soft_topk_mask(logits, topk=2, temperature=1.0):
    temperature = max(float(temperature), 1e-6)
    scaled = logits / temperature
    if topk is None or topk <= 0 or topk >= scaled.size(1):
        return F.softmax(scaled, dim=1)
    topk_values, topk_indices = torch.topk(scaled, k=topk, dim=1)
    topk_probs = F.softmax(topk_values, dim=1)
    output = torch.zeros_like(scaled)
    output.scatter_(1, topk_indices, topk_probs)
    return output


def sample_gumbel(shape, device, dtype, eps=1e-20):
    # Keep CPU-side RNG behavior close to the original implementation
    # (sample on CPU then move), so hard_gumbel remains comparable.
    U = torch.rand(shape, dtype=dtype).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    temperature = max(float(temperature), 1e-6)
    y = logits + sample_gumbel(logits.size(), device=logits.device, dtype=logits.dtype)
    return F.softmax(y / temperature, dim=1)

    
class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x
