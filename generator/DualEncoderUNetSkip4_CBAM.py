import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Blocos de atenção: CBAM
# ----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3,7)
        padding = 3 if kernel_size==7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out,max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class CBAMBlock(nn.Module):
    """CBAM completo: Channel + Spatial Attention"""
    def __init__(self, in_ch, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_ch, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# ----------------------------
# Bloco Conv + CBAM opcional
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_cbam=False):
        super().__init__()
        self.use_cbam = use_cbam
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        if use_cbam:
            self.cbam = CBAMBlock(out_ch)

    def forward(self, x):
        x = self.conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        return x

# ----------------------------
# Bloco de Upsample + Conv + CBAM
# ----------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch_up, in_ch_skip, out_ch, use_cbam=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + in_ch_skip, out_ch, use_cbam)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# ----------------------------
# Dual Encoder UNet 4 níveis com CBAM
# ----------------------------
class DualEncoderUNetSkip4_CBAM(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, use_cbam=False):
        super().__init__()
        self.use_cbam = use_cbam
        # Encoder parte1
        self.enc1_1 = ConvBlock(input_nc, ngf, use_cbam)
        self.enc2_1 = ConvBlock(ngf, ngf*2, use_cbam)
        self.enc3_1 = ConvBlock(ngf*2, ngf*4, use_cbam)
        self.enc4_1 = ConvBlock(ngf*4, ngf*8, use_cbam)

        # Encoder parte2
        self.enc1_2 = ConvBlock(input_nc, ngf, use_cbam)
        self.enc2_2 = ConvBlock(ngf, ngf*2, use_cbam)
        self.enc3_2 = ConvBlock(ngf*2, ngf*4, use_cbam)
        self.enc4_2 = ConvBlock(ngf*4, ngf*8, use_cbam)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(ngf*8*2, ngf*16, use_cbam)

        # Decoder
        self.dec4 = UpBlock(in_ch_up=ngf*16, in_ch_skip=ngf*8*2, out_ch=ngf*8, use_cbam=use_cbam)
        self.dec3 = UpBlock(in_ch_up=ngf*8, in_ch_skip=ngf*4*2, out_ch=ngf*4, use_cbam=use_cbam)
        self.dec2 = UpBlock(in_ch_up=ngf*4, in_ch_skip=ngf*2*2, out_ch=ngf*2, use_cbam=use_cbam)
        self.dec1 = UpBlock(in_ch_up=ngf*2, in_ch_skip=ngf*1*2, out_ch=ngf, use_cbam=use_cbam)

        # Saída
        self.final = nn.Conv2d(ngf, output_nc, 1)

    def forward(self, part1, part2):
        # Encoder parte1
        e1_1 = self.enc1_1(part1)
        e2_1 = self.enc2_1(self.pool(e1_1))
        e3_1 = self.enc3_1(self.pool(e2_1))
        e4_1 = self.enc4_1(self.pool(e3_1))

        # Encoder parte2
        e1_2 = self.enc1_2(part2)
        e2_2 = self.enc2_2(self.pool(e1_2))
        e3_2 = self.enc3_2(self.pool(e2_2))
        e4_2 = self.enc4_2(self.pool(e3_2))

        # Bottleneck
        bottleneck = self.bottleneck(torch.cat([self.pool(e4_1), self.pool(e4_2)], dim=1))

        # Decoder com skips concatenados
        d4 = self.dec4(bottleneck, torch.cat([e4_1, e4_2], dim=1))
        d3 = self.dec3(d4, torch.cat([e3_1, e3_2], dim=1))
        d2 = self.dec2(d3, torch.cat([e2_1, e2_2], dim=1))
        d1 = self.dec1(d2, torch.cat([e1_1, e1_2], dim=1))

        out = self.final(d1)
        return out

# ----------------------------
# Exemplo de uso
# ----------------------------
if __name__ == "__main__":
    model = DualEncoderUNetSkip4_CBAM(input_nc=3, output_nc=3, ngf=32, use_cbam=True)
    part1 = torch.randn(4,3,256,384)
    part2 = torch.randn(4,3,256,384)
    out = model(part1, part2)
    print(out.shape)  # deve ser [4,3,256,384]
