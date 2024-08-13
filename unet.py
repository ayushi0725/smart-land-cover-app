import torch 
from torch import nn
from torchvision import transforms

from modules import UpSample, DownSample, DoubleConv


class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__() 
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

    def forward(self, x):
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        x, x4 = self.down4(x)
        return x, x1, x2, x3, x4


class UpBlock(nn.Module):
    def __init__(self):
        super().__init__() 
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

    def forward(self, x, x1, x2, x3, x4):
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = DownBlock(in_channels)
        self.bottleneck = DoubleConv(512, 1024)
        self.up = UpBlock()
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x, x1, x2, x3, x4  = self.down(x)
        x = self.bottleneck(x)
        x = self.up(x, x1, x2, x3, x4)
        return self.output(x)


if __name__ == '__main__':
    x = torch.rand((1, 3, 512, 512))
    model = UNet(3, 10)
    output = model(x)
    print(output.shape)