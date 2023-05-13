import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

class NormalConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )
    def forward(self,x):
        return self.conv_block(x)

class DoubleConvEnc(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
        NormalConv(in_channels, out_channels),
        NormalConv(out_channels, out_channels),
        ) 
        self.shortcutconv = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1,padding=0)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
    def forward(self,x):
        identity = self.shortcutconv(x)
        x = self.double_conv(x)
        unpooled_shape = x.size()
        x+=identity
        x, ind = self.maxpool(x)
        return x, ind, unpooled_shape

class TripleConvEnc(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.triple_conv = nn.Sequential(
        NormalConv(in_channels, out_channels),
        NormalConv(out_channels, out_channels),
        NormalConv(out_channels, out_channels),
        ) 
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
    def forward(self,x):
        identity = x
        x = self.triple_conv(x)
        unpooled_shape = x.size()
        x+=identity
        x, ind = self.maxpool(x)
        return x, ind, unpooled_shape


class DoubleConvDec(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.double_conv = nn.Sequential(
        NormalConv(in_channels, out_channels),
        NormalConv(out_channels, out_channels),
        )
    def forward(self, x, ind, out_shape):
        x = self.unpool(input=x, indices=ind, output_size=out_shape)
        x = self.double_conv(x)
        return x


class TripleConvDec(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.unpool = nn.MaxUnPool2d(2, 2, return_indices=True)
        self.triple_conv = nn.Sequential(
        NormalConv(in_channels, out_channels),
        NormalConv(out_channels, out_channels),
        NormalConv(out_channels, out_channels),
        )
    def forward(self, x, ind, out_shape):
        x = self.unpool(input=x, indices=ind, output_size=out_shape)
        x = self.triple_conv(x)
        return x



