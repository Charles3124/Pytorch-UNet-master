"""
unet_model5.py

功能: Full assembly of the parts to form the complete network
时间: 2026/03/20
版本: 1.0
"""

from .unet_parts import *


class UNet5(nn.Module):

    def __init__(self, hparams: Dict[str, Any]):
        super(UNet5, self).__init__()

        # 提取超参数
        n_channels = hparams["n_channels"]
        n_classes = hparams["n_classes"]
        filter_number = hparams["filter_number"]
        bilinear = hparams["bilinear"]

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, filter_number, hparams)
        self.down1 = Down(filter_number, filter_number * 2, hparams)
        self.down2 = Down(filter_number * 2, (filter_number * 4) // factor, hparams)

        self.up1 = Up(
            filter_number * 4, (filter_number * 2) // factor, hparams,
            filter_number * 2, filter_number * 2, filter_number
        )
        self.up2 = Up(
            filter_number * 2, filter_number, hparams,
            filter_number, filter_number, filter_number // 2
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
