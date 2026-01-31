import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import  DoubleConv, Down, Up

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout=0.5):
        super(UNet, self).__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, dropout_p= 0.1)
        self.down2 = Down(128, 256, dropout_p=0.1)
        self.down3 = Down(256, 512, dropout_p=0.2)
        self.down4 = Down(512, 1024, dropout_p=0.3)

        self.up1 = Up(1024,512)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)


    def forward(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        y3 = self.up1(x4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up3(y2, x1)
        y = self.up4(y1, x)
        logits = self.outc(y)
        return logits
