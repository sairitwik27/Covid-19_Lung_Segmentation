import torch.nn.functional as F

from unet_att_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,reduction_ratio=8):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.att = CBAM(64,reduction_ratio=reduction_ratio)
        self.down1 = Down(64, 128)
        self.att1 = CBAM(128,reduction_ratio=reduction_ratio)
        self.down2 = Down(128, 256)
        self.att2 = CBAM(256,reduction_ratio=reduction_ratio)
        self.down3 = Down(256, 512)
        self.att3 = CBAM(512,reduction_ratio=reduction_ratio)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.att4 = CBAM(512,reduction_ratio=reduction_ratio)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_att = self.att(x1)
        x2 = self.down1(x1)
        x2_att = self.att1(x2)
        x3 = self.down2(x2)
        x3_att = self.att2(x3)
        x4 = self.down3(x3)
        x4_att = self.att3(x4)
        x5 = self.down4(x4)
        x5_att = self.att4(x5)
        x = self.up1(x5_att, x4_att)
        x = self.up2(x, x3_att)
        x = self.up3(x, x2_att)
        x = self.up4(x, x1_att)
        logits = self.outc(x)
        return logits