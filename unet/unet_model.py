import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import  DoubleConv, Down, Up

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, input_res, kernel_flex=False):
        super(UNet, self).__init__()

        res1 = input_res           
        res2 = input_res // 2      
        res3 = input_res // 4      
        res4 = input_res // 8      
        res_bottleneck = input_res // 16
        
        self.inc = DoubleConv(n_channels, 64, input_res=res1, kernel_flex=kernel_flex)
        
        self.down1 = Down(64, 128, input_res=res2, dropout_p= 0.1, kernel_flex=kernel_flex)
        self.down2 = Down(128, 256, input_res=res3, dropout_p=0.1, kernel_flex=kernel_flex)
        self.down3 = Down(256, 512, input_res=res4, dropout_p=0.2, kernel_flex=kernel_flex)
        self.down4 = Down(512, 1024, input_res=res_bottleneck, dropout_p=0.3, kernel_flex=kernel_flex)

        self.up1 = Up(1024, 512, input_res=res4, kernel_flex=kernel_flex)
        self.up2 = Up(512, 256, input_res=res3, kernel_flex=kernel_flex)
        self.up3 = Up(256, 128, input_res=res2, kernel_flex=kernel_flex)
        self.up4 = Up(128, 64, input_res=res1, kernel_flex=kernel_flex)

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
