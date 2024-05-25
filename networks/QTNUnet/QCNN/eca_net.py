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
# from .core.quaternion_layers import *
# from .core.quaternion_layers import *
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
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Softmax()

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
        print(y.shape)
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        print(y.shape)
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
        #self.conv3 = eca_layer_two(channel, k_size=1)



    def forward(self, x):
        # x = x.squeeze()
        x1 = self.conv1(x)
        #h, w, _ = x1.shape
        x2 = torch.zeros_like(x1)
        x2 = x2[:,1,:,:]
        x2 = x2.unsqueeze(1)
        print(x2.shape)
        #x2 = self.conv2(x)
        #x3 = self.conv3(x)
        y1 = torch.cat([x2, x1], dim=1)
        #y2 = torch.cat([x3, x1], dim=1)
        return y1

class QConvEtAl(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(QConvEtAl, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('qconv',    QuaternionConv(4, 16, kernel_size=3, stride=1, padding=1)),
          #('bn',      nn.BatchNorm2d(64)),
          ('bn',      QuaternionBatchNorm2d(16, gamma_init=1.0, beta_param=True)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('qconv', QuaternionConv(16, 64, kernel_size=3, stride=1, padding=1)),
            # ('bn',      nn.BatchNorm2d(64)),
          ('bn', QuaternionBatchNorm2d(64, gamma_init=1.0, beta_param=True)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('qconv', QuaternionConv(64, 128, kernel_size=3, stride=1, padding=1)),
            # ('bn',      nn.BatchNorm2d(64)),
          ('bn', QuaternionBatchNorm2d(128, gamma_init=1.0, beta_param=True)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('qconv', QuaternionConv(128, 256, kernel_size=3, stride=1, padding=1)),
            # ('bn',      nn.BatchNorm2d(64)),
          ('bn', QuaternionBatchNorm2d(256, gamma_init=1.0, beta_param=True)),
          ('relu',    nn.ReLU()),
          #('avgpool', nn.AvgPool2d(kernel_size=4))
          ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.is_flatten = flatten
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
      #  h = self.layer4(h)
        #print(h.size())
        if(self.is_flatten): h = self.flatten(h)
        return h

class QConvNet(nn.Module):  # Quaternion CNN

    def __init__(self, input_channel, n_classes, patch_size):
        super(QConvNet, self).__init__()
        self.act_fn = F.relu
        self.input_channels = input_channel
        self.patch_size = patch_size
        self.pre_press = generateq(channel=input_channel, k_size=1)
        self.qconv = QConvEtAl(input_channels=4, flatten=True)
        self.features_sizes = self._get_sizes()
        self.fc1 = QuaternionLinear(self.features_sizes, n_classes)

    def _get_sizes(self):
        x = torch.zeros((1, 4, self.patch_size, self.patch_size))
        x = self.qconv(x)
        w, h = x.size()
        size0 = w * h
        return size0

    def forward(self, x):
        x = x.squeeze()
        x = self.pre_press(x)
        x = self.qconv(x)
        x = self.fc1(x)
        # x = self.act_fn(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = torch.reshape(x, (-1, 10, 4))
        # x = torch.sum(torch.abs(x), dim=2)
        # return F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    img_size = 15
    x = torch.rand(10, 1, 103, img_size, img_size)
    model = QConvNet(input_channel=103, patch_size=15, n_classes=16)

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
    # repetitions = 100
    # total_time = 0
    # optimal_batch_size = 2
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #         starter.record()
    #         _ = model(x)
    #         ender.record()
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender) / 1000
    #         total_time += curr_time
    # Throughput = (repetitions * optimal_batch_size) / total_time
    # print("FinalThroughput:", Throughput)

