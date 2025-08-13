# Gen Base
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, features=64):
        super(BasicGenerator, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)  # 6 -> 64
        self.enc2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.enc3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1)  # 128 -> 256

        # Bottleneck
        self.bottleneck = nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1)  # 256 -> 512

        # Decoder
        self.dec3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(features * 2, features, kernel_size=4, stride=2, padding=1)
        self.final = nn.ConvTranspose2d(features, out_channels, kernel_size=4, stride=2, padding=1)

        self.bn_enc2 = nn.BatchNorm2d(features * 2)
        self.bn_enc3 = nn.BatchNorm2d(features * 4)
        self.bn_bottleneck = nn.BatchNorm2d(features * 8)
        self.bn_dec3 = nn.BatchNorm2d(features * 4)
        self.bn_dec2 = nn.BatchNorm2d(features * 2)
        self.bn_dec1 = nn.BatchNorm2d(features)

    def forward(self, part1, part2):
        x = torch.cat([part1, part2], dim=1)  # concat canais, assumindo 3 canais cada = 6 total

        e1 = F.leaky_relu(self.enc1(x), 0.2)
        e2 = F.leaky_relu(self.bn_enc2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.bn_enc3(self.enc3(e2)), 0.2)
        b = F.leaky_relu(self.bn_bottleneck(self.bottleneck(e3)), 0.2)

        d3 = F.relu(self.bn_dec3(self.dec3(b)))
        d2 = F.relu(self.bn_dec2(self.dec2(d3)))
        d1 = F.relu(self.bn_dec1(self.dec1(d2)))
        out = torch.tanh(self.final(d1))  # saída entre -1 e 1

        return out
