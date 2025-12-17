# unet_model5.py
""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet5(nn.Module):

    def __init__(
            self, n_channels: int, n_classes: int, filter_number: int, filter_size: int, activation: int,
            pooling: int, use_dropout: int, use_batchnorm: int, bilinear: bool = False, use_attention: bool = False
    ):
        super(UNet5, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.filter_number = filter_number
        self.filter_size = filter_size
        self.bilinear = bilinear
        self.activation = activation
        self.pooling = pooling
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, filter_number, filter_size, activation, use_dropout, use_batchnorm)
        self.down1 = Down(
            filter_number, filter_number * 2, filter_size, activation,
            pooling, use_dropout, use_batchnorm
        )
        self.down2 = Down(
            filter_number * 2, (filter_number * 4) // factor, filter_size, activation,
            pooling, use_dropout, use_batchnorm
        )

        self.up1 = Up(
            filter_number * 4, (filter_number * 2) // factor, filter_size, activation,
            use_dropout, use_batchnorm, filter_number * 2, filter_number * 2, filter_number,
            bilinear, use_attention=use_attention
        )
        self.up2 = Up(
            filter_number * 2, filter_number, filter_size, activation,
            use_dropout, use_batchnorm, filter_number, filter_number, filter_number // 2,
            bilinear, use_attention=use_attention
        )
        self.outc = OutConv(filter_number, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits
