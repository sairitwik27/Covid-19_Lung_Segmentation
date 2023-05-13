import torch.nn.functional as F

from res_unetparts_dsc import *


class Res_UNet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=True,depth_multiplier=2):
        super(Res_UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        depth_multiplier=depth_multiplier
        self.inc = ConvBlock(n_channels, 64,depth_multiplier)
        self.down1 = Down(64, 128,depth_multiplier)
        self.down2 = Down(128, 256,depth_multiplier)
        self.down3 = Down(256, 512,depth_multiplier)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor,depth_multiplier)
        self.up1 = Up(1024, 512 // factor, bilinear,depth_multiplier)
        self.up2 = Up(512, 256 // factor, bilinear,depth_multiplier)
        self.up3 = Up(256, 128 // factor, bilinear,depth_multiplier)
        self.up4 = Up(128, 64, bilinear,depth_multiplier)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
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