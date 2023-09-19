'''
    Text Extraction Module
'''

from .text_extra_parts import *
import torch.nn as nn


'''
    n_channels : input channel(int)
    n_classes : output channel(int)
'''
class text_extraction_module(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(text_extraction_module, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv1 = FirstConv(n_channels, 64)
        self.conv2 = Down(64, 128)
        self.conv3 = Down(128, 256)
        self.conv4 = Down(256, 512)
        self.conv5 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)



    def forward(self, x, y):
        x1 = self.conv1(x, y)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output