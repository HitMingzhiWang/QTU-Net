import torch
import time
from torch import nn
from numpy.random import RandomState
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import Module
import math
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pdb
import collections
from scipy.stats import chi
from .core.quaternion_layers import *
from .mynonlocal import *


class eca_layer(nn.Module):
    """Constructs a ECA module.*
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        values, indices = torch.topk(y, 3, dim=1, largest=True, sorted=True)
        b, c, h, w = x.shape
        # output = x * y.expand_as(x)
        # output = torch.sum(output, dim=1).unsqueeze(1)
        out = []
        for i in range(b):
            m = x[i, :, :, :]
            j = indices[i]
            j = torch.squeeze(j)
            t = m.index_select(0, j)
            t = torch.unsqueeze(t, 0)
            out.append(t)
        out = torch.cat(out, dim=0)
        # z = torch.cat([output, out], dim=1)
        return out


class nonlocal_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(nonlocal_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = NLBlockND(in_channels=channel, mode='concatenate', dimension=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y)

        # Multi-scale information fusion
        y = self.sigmoid(y.squeeze(-1).squeeze(-1))
        values, indices = torch.topk(y, 3, dim=1, largest=True, sorted=True)
        b, c, h, w = x.shape
        # output = x * y.expand_as(x)
        # output = torch.sum(output, dim=1).unsqueeze(1)
        out = []
        for i in range(b):
            m = x[i, :, :, :]
            j = indices[i]
            j = torch.squeeze(j)
            t = m.index_select(0, j)
            t = torch.unsqueeze(t, 0)
            out.append(t)
        out = torch.cat(out, dim=0)
        # z = torch.cat([output, out], dim=1)
        return out


class eca_layer_one(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer_one, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)  # (b,c,1,1)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        x = torch.sum(x, dim=1)
        return x.unsqueeze(1)


class eca_layer_two(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer_two, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv2d(channel, 1, kernel_size=k_size, bias=False, groups=1)
        self.relu = torch.nn.GELU()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.conv(x)
        y = self.relu(y)
        return y


class generateq(nn.Module):
    def __init__(self, channel, k_size):
        super(generateq, self).__init__()
        # self.conv1 = eca_layer(channel, k_size=1)
        self.conv1 = nonlocal_layer(channel, k_size=1)
        self.conv2 = eca_layer_one(channel, k_size=1)
        self.conv3 = eca_layer_two(channel, k_size=1)

    def forward(self, x):
        # x = x.squeeze()
        x1 = self.conv1(x)  # (3,H,W)
        # x2 = self.conv2(x)
        x3 = self.conv3(x)
        x2 = torch.zeros_like(x1)
        # x2 = x2[:, 1, :, :]
        # x2 = x2.unsqueeze(1)
        # y1 = torch.cat([x2, x1], dim=1)
        # y1 = torch.cat([x3, x2], dim=1)
        y2 = torch.cat([x3, x1], dim=1)
        return y2
