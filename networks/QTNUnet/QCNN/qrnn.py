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

class eca_layer(nn.Module):
    """Constructs a ECA module.
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
        zeros = torch.zeros_like(y)
        ones = torch.ones_like(y)
        values, indices = torch.topk(y, 3, dim=1, largest=True, sorted=True)
        zeros.scatter_(1, torch.LongTensor(indices), ones)
        b, c, h, w = x.shape
        z1 = indices.expand(b,3,h,w)
        y = zeros
        z2 = x * y.expand_as(x)
        out =[]
        for i in range(b):
            m = z2[i, :, :, :]
            j = indices[i]
            j = torch.squeeze(j)
            t = m.index_select(0, j)
            t = torch.unsqueeze(t, 0)
            out.append(t)
        out = torch.cat(out, dim=0)
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
        y = self.avg_pool(x)
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
        self.conv1 = eca_layer(channel, k_size=1)
        self.conv2 = eca_layer_one(channel, k_size=1)
       # self.conv3 = eca_layer_two(channel, k_size=1)



    def forward(self, x):
        # x = x.squeeze()
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        #x3 = self.conv3(x)
        y1 = torch.cat([x2, x1], dim=1)
        #y2 = torch.cat([x3, x1], dim=1)
        return y1

class QrnnNet(nn.Module):  # Quaternion CNN

    def __init__(self, input_channel, n_classes, planes=15):
        super(QrnnNet, self).__init__()
        self.act_fn = F.relu
        self.input_channels = input_channel
        self.pre_press = generateq(channel=input_channel, k_size=1)
        self.features_sizes = self._get_sizes()
        self.fc1 = QuaternionLinear(self.features_sizes, n_classes)
        encoder_modules = []
        n = input_channel
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while n > 1:
                print("---------- {} ---------".format(n))
                if n == input_channel:
                    p1, p2 = 1, 2 * planes
                elif n == input_channel // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channel)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c * w

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pre_press(x)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        return x_classif, x

if __name__ == '__main__':
    img_size = 16
    x = torch.rand(2, 1, 200, img_size, img_size)
    model = QConvNet(input_channel=200, patch_size=15, n_classes=16)

    model.eval()

    # flops = model.flops()
    # print(f"number of GFLOPs: {flops / 1e9}")
    #
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of params: {n_parameters}")
    # out = model(x)
    # print(out)

    from thop import profile
    flops, params = profile(model, (x,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')