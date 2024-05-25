import math
import random
from einops import rearrange
from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F





class DsAttention(nn.Module):
    def __init__(self, dim=48, n_heads=3, dropout=None, m_power=None):
        super(DsAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.l_q = nn.Linear(dim, dim)
        self.l_k = nn.Linear(dim, dim)
        self.l_v = nn.Linear(dim, dim)
        self.g_q = nn.Linear(dim, dim)
        self.g_k = nn.Linear(dim, dim)
        self.g_v = nn.Linear(dim, dim)
        self.m_power = m_power
        self.scale = torch.sqrt(torch.cuda.FloatTensor([self.dim // self.n_heads]))
        self.atten_drop = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = x[:, 0::2, :, :]
        x2 = x[:, 1::2, :, :]
        if self.m_power != None:
            x2 = nn.MaxPool2d(self.m_power)(x2)
        x1 = self.conv(x1)
        x1 = rearrange(x1, "b c h w -> b (h w) c").contiguous()
        x2 = rearrange(x2, "b c h w -> b (h w) c").contiguous()
        Batch = x1.shape[0]
        L_Q = self.l_q(x1)
        L_K = self.l_k(x1)
        L_V = self.l_v(x1)
        G_Q = self.g_q(x2)
        G_K = self.g_k(x2)
        G_V = self.g_v(x2)
        L_Q = L_Q.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        L_K = L_K.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        L_V = L_V.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        G_Q = G_Q.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        G_K = G_K.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        G_V = G_V.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)

        attentionLG = (L_Q @ G_K.transpose(-2, -1)) / self.scale
        attentionGL = (G_Q @ L_K.transpose(-2, -1)) / self.scale

        attentionLG = self.atten_drop(torch.softmax(attentionLG, dim=-1))
        attentionGL = self.atten_drop(torch.softmax(attentionGL, dim=-1))

        x1 = attentionLG @ G_V
        x2 = attentionGL @ L_V
        x1 = x1.permute(0, 2, 1, 3).contiguous()
        x2 = x2.permute(0, 2, 1, 3).contiguous()
        x1 = x1.view(Batch, -1, self.n_heads * (self.dim // self.n_heads))
        x2 = x2.view(Batch, -1, self.n_heads * (self.dim // self.n_heads))
        h1 = int(math.sqrt(x1.shape[1]))
        h2 = int(math.sqrt(x2.shape[1]))
        x1 = rearrange(x1, "b (h w) c -> b h w c", w=h1, h=h1).contiguous().permute(0, 3, 1, 2)
        x2 = rearrange(x2, "b (h w) c -> b h w c", w=h2, h=h2).contiguous().permute(0, 3, 1, 2)
        if self.m_power != None:
            x2 = nn.Upsample(scale_factor=self.m_power)(x2)
        x1 = torch.cat([x1, x2], dim=1)
        return x1


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)  # 返回的是方差和平均值
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):  # 3x3卷积
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):  # 1x1卷积
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):  # 是否有下采样参数
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):  # 这一段代码感觉没用，可能是作者引用backbone源论文时摘抄的
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, n_heads, dropout, ):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        self.dsAttention1 = DsAttention(dim=width // 2, n_heads=n_heads, dropout=dropout, m_power=8)
        self.dsAttention2 = DsAttention(dim=width * 2, n_heads=n_heads, dropout=dropout, m_power=4)
        self.dsAttention3 = DsAttention(dim=width * 4, n_heads=n_heads, dropout=dropout, m_power=2)
        self.dsAttention4 = DsAttention(dim=width * 8, n_heads=n_heads, dropout=dropout, m_power=None)
        self.body = nn.Sequential(OrderedDict([
            ('block1/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        attention = []
        attention.append(self.dsAttention1)
        attention.append(self.dsAttention2)
        attention.append(self.dsAttention3)
        attention.append(self.dsAttention4)
        b, c, in_size, _ = x.size()
        x = self.root(x)
        x = attention[0](x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            x = attention[i + 1](x)

            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        x = attention[-1](x)
        return x, features[::-1]


