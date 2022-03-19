import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import network.apam as APAM
import math
SRM_npy = np.load('network/SRM_Kernels.npy')


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels,
                       self.out_channels, 1) * -1.000)
        super(BayarConv2d, self).__init__()
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5),
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30),
                              requires_grad=True)
        self.SRM_us = torch.from_numpy((SRM_npy != 0).astype(int)).cuda()
        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def bayarConstraint(self):
        ctr = int(self.kernel_size / 2)
        self.weight.data[:, :, ctr, ctr] = 0
        real_kernel = self.weight.clone()
        real_kernel = real_kernel*self.SRM_us
        summ = real_kernel.sum(1).sum(1).sum(1).repeat(5, 5, 1)
        summ = summ.permute(2, 0, 1)
        summ = torch.unsqueeze(summ, 1)
        real_kernel = real_kernel/summ
        real_kernel[:, :, ctr, ctr] = -1
        self.weight.data = real_kernel
        return self.weight

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(),
                     stride=self.stride, padding=self.padding)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, with_bn=False, ap=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride)
        self.ap = ap
        if ap:
            self.relu = APAM.Apam(1, out_channels)
        else:
            self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        if self.ap:
            x = self.relu(x)
        else:
            x = self.relu(x)
        return self.norm(x)

    def reset_parameters(self):
        nn.init.xavier_uniform(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()


class Net(nn.Module):
    def __init__(self, with_bn=False, threshold=3):
        super(Net, self).__init__()
        self.with_bn = with_bn
        self.preprocessing = BayarConv2d(1, 30)
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        if with_bn:
            self.norm1 = nn.BatchNorm2d(30)
        else:
            self.norm1 = lambda x: x
        self.block2 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block3 = ConvBlock(30, 30, 3, with_bn=self.with_bn, ap=False)
        self.block4 = ConvBlock(30, 30, 3, with_bn=self.with_bn, ap=False)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.block5 = ConvBlock(30, 32, 5, with_bn=self.with_bn)
        self.pool2 = nn.AvgPool2d(3, 2)
        self.block6 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool3 = nn.AvgPool2d(3, 2)
        self.block7 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool4 = nn.AvgPool2d(3, 2)
        self.block8 = ConvBlock(32, 16, 3, with_bn=self.with_bn)
        self.block9 = ConvBlock(16, 16, 3, 3, with_bn=self.with_bn)
        self.ip1 = nn.Linear(3 * 3 * 16, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.preprocessing(x)
        x = self.TLU(x)
        x = x.abs()
        x = self.norm1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool1(x)
        x = self.block5(x)
        x = self.pool2(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.pool4(x)
        x = self.block8(x)
        x = self.block9(x)
        x = x.view(x.size(0), -1)
        fea = x
        x = self.ip1(x)
        return x, fea

    def reset_parameters(self):

        for mod in self.modules():
            if isinstance(mod, BayarConv2d) or \
                    isinstance(mod, nn.BatchNorm2d) or \
                    isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal(mod.weight, 0., 0.01)
                mod.bias.data.zero_()
