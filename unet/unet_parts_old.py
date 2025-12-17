""" Parts of the U-Net model """
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, filter_size: int,
                 activation: int, use_dropout: int, use_batchnorm: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # 计算 padding 大小，确保输出特征图尺寸不变
        padding = filter_size // 2 if filter_size % 2 != 0 else (filter_size - 1) // 2

        layers = []

        # 第一个卷积层
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=filter_size, padding=padding, bias=False))

        if use_batchnorm == 1:
            layers.append(nn.BatchNorm2d(mid_channels))

        if activation == 0:
            layers.append(nn.ReLU(inplace=True))
        elif activation == 1:
            layers.append(nn.ELU(inplace=True))
        elif activation == 2:
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.RReLU(inplace=True))

        if use_dropout == 2:
            layers.append(nn.Dropout(p=0.3))
        elif activation == 3:
            layers.append(GaussianDropout(p=0.3))

        # 第二个卷积层
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=filter_size, padding=padding, bias=False))

        if use_batchnorm == 1:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == 0:
            layers.append(nn.ReLU(inplace=True))
        elif activation == 1:
            layers.append(nn.ELU(inplace=True))
        elif activation == 2:
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.RReLU(inplace=True))

        if use_dropout == 2:
            layers.append(nn.Dropout(p=0.3))
        elif activation == 3:
            layers.append(GaussianDropout(p=0.3))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, filter_size: int,
                 activation: int, pooling: int, use_dropout: int, use_batchnorm: int):
        super().__init__()
        if pooling == 0:
            p = nn.MaxPool2d(kernel_size=2)
        else:
            p = nn.AvgPool2d(kernel_size=2)

        self.maxpool_conv = nn.Sequential(p, DoubleConv(in_channels, out_channels, filter_size, activation, use_dropout, use_batchnorm))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, filter_size: int,
                 activation: int, use_dropout: int, use_batchnorm: int, bilinear: bool = True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, filter_size, activation, use_dropout, use_batchnorm, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, filter_size, activation, use_dropout, use_batchnorm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd


class GaussianDropout(nn.Module):

    def __init__(self, p: float = 0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        return x


class ConditionalDropout(nn.Module):

    def __init__(self, condition_fn: Callable[[torch.Tensor], torch.Tensor], p: float = 0.3):
        super(ConditionalDropout, self).__init__()
        self.condition_fn = condition_fn
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.ones_like(x)
            condition_mask = self.condition_fn(x)  # 条件判断
            dropout_mask = (torch.rand_like(mask) > self.p).float()
            mask = mask * dropout_mask * condition_mask.float()
            return x * mask
        return x


def condition_fn(x: torch.Tensor) -> torch.Tensor:
    # 定义条件：例如，只对值大于 0 的位置应用 dropout
    return x > 0

dropout_layer = ConditionalDropout(condition_fn, p=0.3)


class OutConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
