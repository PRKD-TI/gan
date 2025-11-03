import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Bloco Conv simples
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------------
# UpBlock SIMPLIFICADO (sem CBAM)
# ----------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch_up, in_ch_skip, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
        # O bloco de convolução agora recebe a soma dos canais upsampled + skip
        self.conv = ConvBlock(out_ch + in_ch_skip, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Concatenação direta da skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# ----------------------------
# Gerador BASELINE (Dual Encoder UNet - 4 Níveis)
# ----------------------------
class DualEncoderUNet_Baseline(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32):
        super().__init__()

        # Encoder parte1
        self.enc1_1 = ConvBlock(input_nc, ngf)
        self.enc2_1 = ConvBlock(ngf, ngf*2)
        self.enc3_1 = ConvBlock(ngf*2, ngf*4)
        self.enc4_1 = ConvBlock(ngf*4, ngf*8)

        # Encoder parte2
        self.enc1_2 = ConvBlock(input_nc, ngf)
        self.enc2_2 = ConvBlock(ngf, ngf*2)
        self.enc3_2 = ConvBlock(ngf*2, ngf*4)
        self.enc4_2 = ConvBlock(ngf*4, ngf*8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        # Recebe os canais concatenados dos dois encoders (ngf*8*2)
        self.bottleneck = ConvBlock(ngf*8*2, ngf*16)

        # Decoder (agora usando o UpBlock simplificado)
        # As skip connections também são concatenadas
        self.dec4 = UpBlock(in_ch_up=ngf*16, in_ch_skip=ngf*8*2, out_ch=ngf*8)
        self.dec3 = UpBlock(in_ch_up=ngf*8,  in_ch_skip=ngf*4*2, out_ch=ngf*4)
        self.dec2 = UpBlock(in_ch_up=ngf*4,  in_ch_skip=ngf*2*2, out_ch=ngf*2)
        self.dec1 = UpBlock(in_ch_up=ngf*2,  in_ch_skip=ngf*1*2, out_ch=ngf)

        # Saída
        self.final = nn.Conv2d(ngf, output_nc, 1)

    def forward(self, part1, part2):
        # Encoder parte1
        e1_1 = self.enc1_1(part1)
        e2_1 = self.enc2_1(self.pool(e1_1))
        e3_1 = self.enc3_1(self.pool(e2_1))
        e4_1 = self.enc4_1(self.pool(e3_1)) # Nível 4

        # Encoder parte2
        e1_2 = self.enc1_2(part2)
        e2_2 = self.enc2_2(self.pool(e1_2))
        e3_2 = self.enc3_2(self.pool(e2_2))
        e4_2 = self.enc4_2(self.pool(e3_2)) # Nível 4
        
        # O "fundo" do U (nível 5)
        # Pooling dos mapas do nível 4 e concatenação ANTES do bloco bottleneck
        bottleneck_input = torch.cat([self.pool(e4_1), self.pool(e4_2)], dim=1)
        bottleneck = self.bottleneck(bottleneck_input)

        # Decoder com skips concatenadas de ambos encoders
        # 
        d4 = self.dec4(bottleneck,  torch.cat([e4_1, e4_2], dim=1))
        d3 = self.dec3(d4,          torch.cat([e3_1, e3_2], dim=1))
        d2 = self.dec2(d3,          torch.cat([e2_1, e2_2], dim=1))
        d1 = self.dec1(d2,          torch.cat([e1_1, e1_2], dim=1))

        out = self.final(d1)

        # Retorna uma tupla para manter compatibilidade com seu train_loop
        return out, None