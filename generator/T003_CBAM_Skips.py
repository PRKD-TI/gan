# File: T003_CBAM_Skips.py
import torch
import torch.nn as nn
from .generator_blocks import ConvBlock, UpBlockCBAMSkip

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()
        self.enc1_1 = ConvBlock(input_nc, ngf)
        self.enc2_1 = ConvBlock(ngf, ngf*2)
        self.enc3_1 = ConvBlock(ngf*2, ngf*4)
        self.enc4_1 = ConvBlock(ngf*4, ngf*8)
        self.enc1_2 = ConvBlock(input_nc, ngf)
        self.enc2_2 = ConvBlock(ngf, ngf*2)
        self.enc3_2 = ConvBlock(ngf*2, ngf*4)
        self.enc4_2 = ConvBlock(ngf*4, ngf*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(ngf*8*2, ngf*16)
        # Use UpBlockCBAMSkip
        self.dec4 = UpBlockCBAMSkip(ngf*16, ngf*8*2, ngf*8, ConvBlock)
        self.dec3 = UpBlockCBAMSkip(ngf*8,  ngf*4*2, ngf*4, ConvBlock)
        self.dec2 = UpBlockCBAMSkip(ngf*4,  ngf*2*2, ngf*2, ConvBlock)
        self.dec1 = UpBlockCBAMSkip(ngf*2,  ngf*1*2, ngf,   ConvBlock)
        self.final = nn.Conv2d(ngf, output_nc, 1)

    def forward(self, part1, part2):
        e1_1 = self.enc1_1(part1)
        e2_1 = self.enc2_1(self.pool(e1_1))
        e3_1 = self.enc3_1(self.pool(e2_1))
        e4_1 = self.enc4_1(self.pool(e3_1))
        e1_2 = self.enc1_2(part2)
        e2_2 = self.enc2_2(self.pool(e1_2))
        e3_2 = self.enc3_2(self.pool(e2_2))
        e4_2 = self.enc4_2(self.pool(e3_2))
        bottleneck_input = torch.cat([self.pool(e4_1), self.pool(e4_2)], dim=1)
        bottleneck = self.bottleneck(bottleneck_input)
        d4 = self.dec4(bottleneck, torch.cat([e4_1, e4_2], dim=1))
        d3 = self.dec3(d4, torch.cat([e3_1, e3_2], dim=1))
        d2 = self.dec2(d3, torch.cat([e2_1, e2_2], dim=1))
        d1 = self.dec1(d2, torch.cat([e1_1, e1_2], dim=1))
        out = self.final(d1)
        return out, None