# File: T019_Residual_FullAttention_DS.py
# Descrição: Modelo "Máximo" com ResNet, DS, e Atenção em todos os estágios:
# 1. Encoder (CBAM)
# 2. Bottleneck (Self-Attention / Non-Local)
# 3. Skips (Cross-Attention)
# 4. Decoder (CBAM)

import torch
import torch.nn as nn
from .generator_blocks import (
    ResidualBlock, 
    CBAMBlock, 
    NonLocalBlock, 
    UpBlockCrossAttnSkip_CBAMDecoder
)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()
        
        # --- Encoder 1 (com Atenção CBAM) ---
        self.enc1_1 = ResidualBlock(input_nc, ngf)
        self.enc1_1_attn = CBAMBlock(ngf)
        self.enc2_1 = ResidualBlock(ngf, ngf*2)
        self.enc2_1_attn = CBAMBlock(ngf*2)
        self.enc3_1 = ResidualBlock(ngf*2, ngf*4)
        self.enc3_1_attn = CBAMBlock(ngf*4)
        self.enc4_1 = ResidualBlock(ngf*4, ngf*8)
        self.enc4_1_attn = CBAMBlock(ngf*8)

        # --- Encoder 2 (com Atenção CBAM) ---
        self.enc1_2 = ResidualBlock(input_nc, ngf)
        self.enc1_2_attn = CBAMBlock(ngf)
        self.enc2_2 = ResidualBlock(ngf, ngf*2)
        self.enc2_2_attn = CBAMBlock(ngf*2)
        self.enc3_2 = ResidualBlock(ngf*2, ngf*4)
        self.enc3_2_attn = CBAMBlock(ngf*4)
        self.enc4_2 = ResidualBlock(ngf*4, ngf*8)
        self.enc4_2_attn = CBAMBlock(ngf*8)

        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck (com Atenção Non-Local) ---
        self.bottleneck_conv = ResidualBlock(ngf*8*2, ngf*16)
        self.bottleneck_attn = NonLocalBlock(ngf*16)

        # --- Decoder (com CrossAttn-Skip e CBAM-Decoder) ---
        # (O novo bloco T019 cuida das duas atenções do decoder)
        self.dec4 = UpBlockCrossAttnSkip_CBAMDecoder(ngf*16, ngf*8*2, ngf*8, ResidualBlock)
        self.dec3 = UpBlockCrossAttnSkip_CBAMDecoder(ngf*8,  ngf*4*2, ngf*4, ResidualBlock)
        self.dec2 = UpBlockCrossAttnSkip_CBAMDecoder(ngf*4,  ngf*2*2, ngf*2, ResidualBlock)
        self.dec1 = UpBlockCrossAttnSkip_CBAMDecoder(ngf*2,  ngf*1*2, ngf,   ResidualBlock)

        # --- Saída Final e Deep Supervision ---
        self.final = nn.Conv2d(ngf, output_nc, 1)
        self.ds_out3 = nn.Conv2d(ngf*4, output_nc, 1)
        self.ds_out2 = nn.Conv2d(ngf*2, output_nc, 1)

    def forward(self, part1, part2):
        # --- Encoder 1 (com Atenção) ---
        e1_1 = self.enc1_1_attn(self.enc1_1(part1))
        e2_1 = self.enc2_1_attn(self.enc2_1(self.pool(e1_1)))
        e3_1 = self.enc3_1_attn(self.enc3_1(self.pool(e2_1)))
        e4_1 = self.enc4_1_attn(self.enc4_1(self.pool(e3_1)))

        # --- Encoder 2 (com Atenção) ---
        e1_2 = self.enc1_2_attn(self.enc1_2(part2))
        e2_2 = self.enc2_2_attn(self.enc2_2(self.pool(e1_2)))
        e3_2 = self.enc3_2_attn(self.enc3_2(self.pool(e2_2)))
        e4_2 = self.enc4_2_attn(self.enc4_2(self.pool(e3_2)))

        # --- Bottleneck (com Atenção) ---
        bottleneck_input = torch.cat([self.pool(e4_1), self.pool(e4_2)], dim=1)
        bottleneck = self.bottleneck_conv(bottleneck_input)
        bottleneck = self.bottleneck_attn(bottleneck)

        # --- Decoder (com Atenção) ---
        # (Os blocos 'dec' agora lidam com CrossAttn-Skip e CBAM-Decoder)
        d4 = self.dec4(bottleneck, torch.cat([e4_1, e4_2], dim=1))
        d3 = self.dec3(d4, torch.cat([e3_1, e3_2], dim=1))
        d2 = self.dec2(d3, torch.cat([e2_1, e2_2], dim=1))
        d1 = self.dec1(d2, torch.cat([e1_1, e1_2], dim=1))

        # --- Saídas ---
        out = self.final(d1)
        out_ds3 = self.ds_out3(d3)
        out_ds2 = self.ds_out2(d2)
        
        return out, [out_ds2, out_ds3]