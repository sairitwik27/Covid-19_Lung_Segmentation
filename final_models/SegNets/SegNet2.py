import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from SegNet_Residual import *
#from Segnet_parts2 import *


class SegNet(nn.Module):
    """autoencoder definition
    """
    def __init__(self,in_channels=3,n_classes=1,is_unpooling=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.enc1 = DoubleConvEnc(self.in_channels, 64)
        self.enc2 = DoubleConvEnc(64, 128)
        self.enc3 = DoubleConvEnc(128, 256)
        self.dec3 = DoubleConvDec(256, 128)
        self.dec2 = DoubleConvDec(128, 64)
        self.dec1 = DoubleConvDec(64, n_classes)


    def forward(self, x):

        enc1, ind_1, unpool_shape1 = self.enc1(x)
        enc2, ind_2, unpool_shape2 = self.enc2(enc1)
        enc3, ind_3, unpool_shape3 = self.enc3(enc2)

        dec3 = self.dec3(enc3, ind_3, unpool_shape3)
        dec2 = self.dec2(dec3, ind_2, unpool_shape2)
        dec1 = self.dec1(dec2, ind_1, unpool_shape1)

        return dec1