import torch
import torch.nn as nn
from attention.cbam import CBAM
from attention.se_block import SEBlock

# ---- Blocos auxiliares ----
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, se=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(out_channels) if se else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x))


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, se=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.upconv = nn.Sequential(*layers)
        self.se = SEBlock(out_channels) if se else nn.Identity()

    def forward(self, x):
        return self.se(self.upconv(x))


# ---- DualEncoderUNetSkip6 com atenção ----
class DualEncoderUNetSkip6_Attention(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()

        # Encoder 1
        self.enc1_1 = ConvBlock(input_nc, ngf, use_bn=False, se=True)
        self.enc1_2 = ConvBlock(ngf, ngf*2, se=True)
        self.enc1_3 = ConvBlock(ngf*2, ngf*4, se=True)
        self.enc1_4 = ConvBlock(ngf*4, ngf*8, se=True)
        self.enc1_5 = ConvBlock(ngf*8, ngf*8, se=True)
        self.enc1_6 = ConvBlock(ngf*8, ngf*8, se=True)

        # Encoder 2
        self.enc2_1 = ConvBlock(input_nc, ngf, use_bn=False, se=True)
        self.enc2_2 = ConvBlock(ngf, ngf*2, se=True)
        self.enc2_3 = ConvBlock(ngf*2, ngf*4, se=True)
        self.enc2_4 = ConvBlock(ngf*4, ngf*8, se=True)
        self.enc2_5 = ConvBlock(ngf*8, ngf*8, se=True)
        self.enc2_6 = ConvBlock(ngf*8, ngf*8, se=True)

        # Bottleneck CBAM
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            CBAM(ngf*8)
        )

        # Decoder
        self.dec1 = UpConvBlock(ngf*8, ngf*8, dropout=True, se=True)   # + e1_6, e2_6
        self.dec2 = UpConvBlock(ngf*24, ngf*8, dropout=True, se=True)  # + e1_5, e2_5
        self.dec3 = UpConvBlock(ngf*24, ngf*8, dropout=True, se=True)  # + e1_4, e2_4
        self.dec4 = UpConvBlock(ngf*24, ngf*4, se=True)                # + e1_3, e2_3
        self.dec5 = UpConvBlock(ngf*12, ngf*2, se=True)                # + e1_2, e2_2
        self.dec6 = UpConvBlock(ngf*6, ngf, se=True)                   # + e1_1, e2_1

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*3, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, part1, part2):
        # Encoder
        e1_1 = self.enc1_1(part1); e1_2 = self.enc1_2(e1_1)
        e1_3 = self.enc1_3(e1_2); e1_4 = self.enc1_4(e1_3)
        e1_5 = self.enc1_5(e1_4); e1_6 = self.enc1_6(e1_5)

        e2_1 = self.enc2_1(part2); e2_2 = self.enc2_2(e2_1)
        e2_3 = self.enc2_3(e2_2); e2_4 = self.enc2_4(e2_3)
        e2_5 = self.enc2_5(e2_4); e2_6 = self.enc2_6(e2_5)

        # Bottleneck
        b = self.bottleneck(torch.cat([e1_6, e2_6], dim=1))

        # Decoder + skip direto
        d1 = self.dec1(b); d1 = torch.cat([d1, e1_6, e2_6], dim=1)
        d2 = self.dec2(d1); d2 = torch.cat([d2, e1_5, e2_5], dim=1)
        d3 = self.dec3(d2); d3 = torch.cat([d3, e1_4, e2_4], dim=1)
        d4 = self.dec4(d3); d4 = torch.cat([d4, e1_3, e2_3], dim=1)
        d5 = self.dec5(d4); d5 = torch.cat([d5, e1_2, e2_2], dim=1)
        d6 = self.dec6(d5); d6 = torch.cat([d6, e1_1, e2_1], dim=1)

        out = self.final(d6)
        return out
