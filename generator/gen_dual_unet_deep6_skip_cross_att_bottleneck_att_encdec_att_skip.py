import torch
import torch.nn as nn
from attention.cbam import CBAM
from attention.se_block import SEBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.up(x))


class DualEncoderUNetCrossSkip6_Attn(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, features=64):
        super().__init__()

        # Encoders
        self.encoder1 = nn.ModuleList([
            ConvBlock(input_nc, features, use_se=True),
            ConvBlock(features, features*2, use_se=True),
            ConvBlock(features*2, features*4, use_se=True),
            ConvBlock(features*4, features*8, use_se=True),
            ConvBlock(features*8, features*8, use_se=True),
            ConvBlock(features*8, features*8, use_se=True),
        ])
        self.encoder2 = nn.ModuleList([
            ConvBlock(input_nc, features, use_se=True),
            ConvBlock(features, features*2, use_se=True),
            ConvBlock(features*2, features*4, use_se=True),
            ConvBlock(features*4, features*8, use_se=True),
            ConvBlock(features*8, features*8, use_se=True),
            ConvBlock(features*8, features*8, use_se=True),
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*16, features*16, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            CBAM(features*16)
        )

        # Decoders
        self.decoder = nn.ModuleList([
            UpBlock(features*16, features*8, use_se=True),
            UpBlock(features*16, features*8, use_se=True),
            UpBlock(features*16, features*8, use_se=True),
            UpBlock(features*16, features*4, use_se=True),
            UpBlock(features*8, features*2, use_se=True),
            UpBlock(features*4, features, use_se=True),
        ])

        # Skip Attention
        self.skip_attn = nn.ModuleList([
            CBAM(f) for f in [features*8, features*8, features*8, features*4, features*2, features]
        ])

        # Output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, output_nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x1, x2):
        # Encoder streams
        skips1, skips2 = [], []
        for block in self.encoder1:
            x1 = block(x1)
            skips1.append(x1)
        for block in self.encoder2:
            x2 = block(x2)
            skips2.append(x2)

        # Bottleneck
        merged = torch.cat([x1, x2], dim=1)
        b = self.bottleneck(merged)

        # Decoder with cross-skips + attention
        d = b
        for i, block in enumerate(self.decoder):
            d = block(d)
            skip1 = self.skip_attn[i](skips1[-(i+1)])
            skip2 = self.skip_attn[i](skips2[-(i+1)])
            if i % 2 == 0:  # alterna: conecta de um encoder ou do outro
                d = torch.cat([d, skip1], dim=1)
            else:
                d = torch.cat([d, skip2], dim=1)

        return self.final(d)
