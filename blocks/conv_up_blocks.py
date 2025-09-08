"""
ConvBlock e UpBlock Avançados
-----------------------------
Blocos básicos do encoder e decoder, agora com suporte a:
- ResNetBlock (blocos residuais)
- ASPP (dilatação múltipla)
- Atenção (CBAM ou SE)
Permite flexibilidade para criar redes profundas e estáveis.
"""

import torch
import torch.nn as nn
from .resnet_block import ResNetBlock
from attention.cbam import CBAM
from attention.se_block import SEBlock

# --- ASPP ---
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    -----------------------------
    Combina convoluções com diferentes rates de dilatação
    para capturar contexto em múltiplas escalas.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.project = nn.Conv2d(out_channels*4, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.bn(self.relu(self.project(x_cat)))

# --- ConvBlock para encoder ---
class ConvBlock(nn.Module):
    """
    Bloco de convolução do encoder
    -----------------------------
    Estrutura:
    Conv2d -> BN -> LeakyReLU -> (ResNet opcional) -> (ASPP opcional) -> (Atenção opcional)
    """
    def __init__(self, in_channels, out_channels, use_res=True, use_aspp=False, use_cbam=True, use_se=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.res = ResNetBlock(out_channels, out_channels) if use_res else nn.Identity()
        self.aspp = ASPP(out_channels, out_channels) if use_aspp else nn.Identity()
        self.attn = CBAM(out_channels) if use_cbam else SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        x = self.aspp(x)
        x = self.attn(x)
        return x

# --- UpBlock para decoder ---
class UpBlock(nn.Module):
    """
    Bloco de convolução transposta do decoder
    ----------------------------------------
    Estrutura:
    ConvTranspose2d -> BN -> ReLU -> (ResNet opcional) -> (ASPP opcional) -> (Atenção opcional)
    """
    def __init__(self, in_channels, out_channels, use_res=True, use_aspp=False, use_cbam=True, use_se=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res = ResNetBlock(out_channels, out_channels) if use_res else nn.Identity()
        self.aspp = ASPP(out_channels, out_channels) if use_aspp else nn.Identity()
        self.attn = CBAM(out_channels) if use_cbam else SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        x = self.res(x)
        x = self.aspp(x)
        x = self.attn(x)
        return x
