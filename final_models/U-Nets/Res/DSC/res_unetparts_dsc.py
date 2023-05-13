import torch
import torch.nn as nn
import torch.nn.functional as F

class DWSC(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, depth_multiplier=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * depth_multiplier, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, inter_channels=None,depth_multiplier=1):
        super().__init__()
        if not inter_channels:
            inter_channels = out_channels
        self.conv_block = nn.Sequential(
            DWSC(in_channels, inter_channels, kernel_size=3, depth_multiplier=depth_multiplier, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            DWSC(inter_channels, out_channels, kernel_size=3, depth_multiplier=depth_multiplier, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,depth_multiplier=1,stride=1):
        super().__init__()
        self.expansion = 1
        if not mid_channels:
            mid_channels = out_channels
        self.convblock = nn.Sequential(
        DWSC(in_channels,mid_channels,kernel_size=1,depth_multiplier=depth_multiplier,padding=0),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(),
        DWSC(mid_channels,out_channels,kernel_size=3,depth_multiplier=depth_multiplier,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        DWSC(out_channels,out_channels,kernel_size=3,depth_multiplier=depth_multiplier,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        )
        self.shortcutconv = nn.Sequential(
        DWSC(in_channels,out_channels,kernel_size=1,depth_multiplier=depth_multiplier,padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )
        self.activate = nn.ReLU()
    
    def forward(self,x):
        identity = self.shortcutconv(x)
        x = self.convblock(x)
        x+= identity
        x = self.activate(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, depth_multiplier=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels,depth_multiplier),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,depth_multiplier=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.res = ResBlock(in_channels, out_channels, in_channels // 2,depth_multiplier=depth_multiplier)
            #self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResBlock(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.res(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)