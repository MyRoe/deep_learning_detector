import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Apam(nn.Module):
    def __init__(self, gap_size, channel):
        super(Apam, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(2*channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x_Relu = self.relu(x)
        zeros = x_Relu - x_Relu
        x_min = torch.min(x, zeros)
        x_min = self.gap(x_min)
        x_Relu = self.gap(x_Relu)
        x_min = torch.flatten(x_min, 1)
        x_Relu = torch.flatten(x_Relu, 1)
        x = torch.cat([x_Relu, x_min], 1)
        x = self.fc(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(2)
        x_min = torch.min(x_raw, zeros)
        x_min = torch.max(x_min, -x)
        x_max = torch.max(x_raw, zeros)
        x = x_max+x_min
        return x
