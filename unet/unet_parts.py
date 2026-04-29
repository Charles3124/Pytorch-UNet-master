"""
unet_parts.py

功能: Parts of the U-Net model
时间: 2026/03/20
版本: 1.0
"""

from typing import Optional, Dict, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        hparams: Dict[str, Any], mid_channels: Optional[int] = None
    ):
        super().__init__()

        filter_size = hparams["filter_size"]
        activation = hparams["activation"]
        use_dropout = hparams["use_dropout"]
        use_batchnorm = hparams["use_batchnorm"]

        if mid_channels is None:
            mid_channels = out_channels

        padding = filter_size // 2

        # 选择激活函数
        if activation == 0:
            act_layer = nn.ReLU(inplace=True)
        elif activation == 1:
            act_layer = nn.ELU(inplace=True)
        elif activation == 2:
            act_layer = nn.LeakyReLU(inplace=True)
        else:
            act_layer = nn.RReLU(inplace=True)

        layers = []

        # ----- Conv 1 -----
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=filter_size, padding=padding, bias=False))
        if use_batchnorm == 1:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(act_layer)

        # ----- Conv 2 -----
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=filter_size, padding=padding, bias=False))
        if use_batchnorm == 1:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_layer)

        # ----- Dropout -----
        if use_dropout == 2:
            layers.append(nn.Dropout2d(p=0.3))
        elif use_dropout == 3:
            layers.append(GaussianDropout(p=0.3))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DoubleConvOld(nn.Module):

    def __init__(
            self, in_channels: int, out_channels: int,
            hparams: Dict[str, Any], mid_channels: Optional[int] = None
    ):
        super().__init__()

        # 提取超参数
        filter_size = hparams["filter_size"]
        activation = hparams["activation"]
        use_dropout = hparams["use_dropout"]
        use_batchnorm = hparams["use_batchnorm"]

        if mid_channels is None:
            mid_channels = out_channels

        padding = filter_size // 2 if filter_size % 2 != 0 else (filter_size - 1) // 2

        # 主路径
        layers = []

        # 第一个卷积层
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=filter_size, padding=padding, bias=False))

        # 批量归一化
        if use_batchnorm == 1:
            layers.append(nn.BatchNorm2d(mid_channels))

        # 激活函数
        if activation == 0:
            layers.append(nn.ReLU(inplace=True))
        elif activation == 1:
            layers.append(nn.ELU(inplace=True))
        elif activation == 2:
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.RReLU(inplace=True))

        # dropout
        if use_dropout == 2:
            layers.append(nn.Dropout(p=0.3))
        elif use_dropout == 3:
            layers.append(GaussianDropout(p=0.3))

        # 第二个卷积层
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=filter_size, padding=padding, bias=False))

        # 批量归一化
        if use_batchnorm == 1:
            layers.append(nn.BatchNorm2d(out_channels))

        # dropout
        if use_dropout == 2:
            layers.append(nn.Dropout(p=0.3))
        elif use_dropout == 3:
            layers.append(GaussianDropout(p=0.3))

        self.double_conv = nn.Sequential(*layers)

        # 激活函数
        if activation == 0:
            self.act = nn.ReLU(inplace=True)
        elif activation == 1:
            self.act = nn.ELU(inplace=True)
        elif activation == 2:
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = nn.RReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.double_conv(x)
        out = self.act(out)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, hparams: Dict[str, Any]):
        super().__init__()

        if hparams["pooling"] == 0:
            p = nn.MaxPool2d(kernel_size=2)
        else:
            p = nn.AvgPool2d(kernel_size=2)

        self.maxpool_conv = nn.Sequential(p, DoubleConv(in_channels, out_channels, hparams))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
            self, in_channels: int, out_channels: int,
            hparams: Dict[str, Any], F_g: int, F_l: int, F_int: int,
            use_attention_depth: bool = True
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if hparams["bilinear"]:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, hparams, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, hparams)

        # Attention 模块
        self.use_attention = hparams["use_attention"] and use_attention_depth
        if self.use_attention:
            self.attention = AttentionBlock(hparams, F_g=F_g, F_l=F_l, F_int=F_int)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 上采样
        x1 = self.up(x1)

        # 对跳跃连接 x2 做注意力加权
        if self.use_attention:
            x2 = self.attention(g=x1, x=x2)

        # 调整尺寸，使拼接不出错
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # 拼接 + 双卷积
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GaussianDropout(nn.Module):

    def __init__(self, p: float = 0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev + 1
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
    """定义条件：例如，只对值大于 0 的位置应用 dropout"""
    return x > 0

dropout_layer = ConditionalDropout(condition_fn, p=0.3)


class OutConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionBlock(nn.Module):

    def __init__(self, hparams: Dict[str, Any], F_g: int, F_l: int, F_int: int):
        super(AttentionBlock, self).__init__()

        # 提取超参数
        attention_activation = hparams["attention_activation"]
        self.attention_fusion = hparams["attention_fusion"]

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        in_channels = F_int if self.attention_fusion == 0 else F_int * 2
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid() if attention_activation == 0 else nn.Hardsigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if self.attention_fusion == 0:  # 加法
            psi = self.relu(g1 + x1)
        else:                           # 拼接
            psi = self.relu(torch.cat([g1, x1], dim=1))

        psi = self.psi(psi)
        return x * (1 + psi)
