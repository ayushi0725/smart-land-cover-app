import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(growth_rate),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

# class UpBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UpBlock, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
class UpBlock(nn.Module):
    def __init__(self, decoder_in_channels, encoder_in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(decoder_in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + encoder_in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ADUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(ADUNet, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            DenseBlock(in_channels, growth_rate=16, num_layers=4),
            SEBlock(3 + 4 * 16)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            DenseBlock(3 + 4 * 16, growth_rate=32, num_layers=4),
            SEBlock(3 + 4 * 16 + 4 * 32)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Sequential(
            DenseBlock(3 + 4 * 16 + 4 * 32, growth_rate=64, num_layers=4),
            SEBlock(3 + 4 * 16 + 4 * 32 + 4 * 64)
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DenseBlock(3 + 4 * 16 + 4 * 32 + 4 * 64, growth_rate=128, num_layers=4),
            SEBlock(3 + 4 * 16 + 4 * 32 + 4 * 64 + 4 * 128)
        )

        # Decoder (Corrected channel sizes)
        self.up3 = UpBlock(decoder_in_channels=963, encoder_in_channels=451, out_channels=256)
        self.up2 = UpBlock(decoder_in_channels=256, encoder_in_channels=195, out_channels=128)
        self.up1 = UpBlock(decoder_in_channels=128, encoder_in_channels=67, out_channels=64)


        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)            # 67
        enc2 = self.encoder2(self.pool1(enc1))  # 195
        enc3 = self.encoder3(self.pool2(enc2))  # 451

        bottleneck = self.bottleneck(self.pool3(enc3))  # 963

        dec3 = self.up3(bottleneck, enc3)  # -> 256
        dec2 = self.up2(dec3, enc2)        # -> 128
        dec1 = self.up1(dec2, enc1)        # -> 64

        return self.final(dec1)            # -> 6 class output
