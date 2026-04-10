"""
unet_model9.py

功能: Full assembly of the parts to form the complete network
时间: 2026/03/20
版本: 1.0
"""

from .unet_parts import *


class UNet9(nn.Module):

    def __init__(self, hparams: Dict[str, Any]):
        super(UNet9, self).__init__()

        # 提取超参数
        n_channels = hparams["n_channels"]
        n_classes = hparams["n_classes"]
        filters_number = hparams["filters_number"]
        bilinear = hparams["bilinear"]

        # Attention F_int 映射
        attention_ratio = hparams["attention_ratio"]
        F_l_list = [filters_number * 8, filters_number * 4, filters_number * 2, filters_number]
        F_int_list = [max(round(F_l * attention_ratio), 1) for F_l in F_l_list]

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, filters_number, hparams)
        self.down1 = Down(filters_number, filters_number * 2, hparams)
        self.down2 = Down(filters_number * 2, filters_number * 4, hparams)
        self.down3 = Down(filters_number * 4, filters_number * 8, hparams)
        self.down4 = Down(filters_number * 8, (filters_number * 16) // factor, hparams)

        self.up1 = Up(
            filters_number * 16, (filters_number * 8) // factor, hparams,
            F_g=filters_number * 8, F_l=filters_number * 8, F_int=F_int_list[0]
        )
        self.up2 = Up(
            filters_number * 8, (filters_number * 4) // factor, hparams,
            F_g=filters_number * 4, F_l=filters_number * 4, F_int=F_int_list[1]
        )
        self.up3 = Up(
            filters_number * 4, (filters_number * 2) // factor, hparams,
            F_g=filters_number * 2, F_l=filters_number * 2, F_int=F_int_list[2]
        )
        self.up4 = Up(
            filters_number * 2, filters_number, hparams,
            F_g=filters_number, F_l=filters_number, F_int=F_int_list[3]
        )
        self.outc = OutConv(filters_number, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
