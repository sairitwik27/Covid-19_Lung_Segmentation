import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from SegNet_Residual import *
from Segnet_parts1 import *


class SegNet(nn.Module):
    """autoencoder definition
    """
    def __init__(self,in_channels=3,n_classes=1,is_unpooling=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.enc1 = DoubleConvEnc(self.in_channels, 64)
        self.enc2 = DoubleConvEnc(64, 128)
        self.enc3 = TripleConvEnc(128, 256)
        self.enc4 = TripleConvEnc(256, 512)
        self.enc5 = TripleConvEnc(512, 512)

        self.dec5 = TripleConvDec(512, 512)
        self.dec4 = TripleConvDec(512, 256)
        self.dec3 = TripleConvDec(256, 128)
        self.dec2 = DoubleConvDec(128, 64)
        self.dec1 = DoubleConvDec(64, n_classes)


    def forward(self, x):

        enc1, ind_1, unpool_shape1 = self.enc1(x)
        enc2, ind_2, unpool_shape2 = self.enc2(enc1)
        enc3, ind_3, unpool_shape3 = self.enc3(enc2)
        enc4, ind_4, unpool_shape4 = self.enc4(enc3)
        enc5, ind_5, unpool_shape5 = self.enc5(enc4)

        dec5 = self.dec5(enc5, ind_5, unpool_shape5)
        dec4 = self.dec4(dec5, ind_4, unpool_shape4)
        dec3 = self.dec3(dec4, ind_3, unpool_shape3)
        dec2 = self.dec2(dec3, ind_2, unpool_shape2)
        dec1 = self.dec1(dec2, ind_1, unpool_shape1)

        return dec1