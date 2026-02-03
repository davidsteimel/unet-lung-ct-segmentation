import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0, input_res = 256):
        super(DoubleConv, self).__init__()

        kernel_size = int(3 * (input_res / 256))

        if kernel_size % 2 == 0: kernel_size += 1 
        padding = kernel_size // 2

        if input_res == 256:
            dilation = 1
        else:
            dilation = int(3 * (input_res / 256))

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))

        self.double_conv = nn.Sequential(*layers)
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
        