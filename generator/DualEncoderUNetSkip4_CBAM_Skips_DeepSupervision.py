import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Bloco Residual (ResNetBlock)
# ----------------------------
# Bloco Residual (ResNetBlock)
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), # <--- Corrigido: Removido inplace=True
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU() # <--- Corrigido: Removido inplace=True

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual  # Adiciona a conexão residual
        out = self.relu(out)
        return out

# ----------------------------
# ASPP (Atrous Spatial Pyramid Pooling)
# ----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv_aspp1 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv_aspp2 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv_aspp3 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.conv_aspp4 = nn.Conv2d(in_channels, out_channels, 3, padding=24, dilation=24, bias=False)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv_aspp1(x)
        x3 = self.conv_aspp2(x)
        x4 = self.conv_aspp3(x)
        x5 = self.conv_aspp4(x)
        
        # Concatena todas as saídas e retorna
        return torch.cat((x1, x2, x3, x4, x5), dim=1)


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
        return self.sigmoid(avg_out + max_out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size==7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out,max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x

class CBAMBlock(nn.Module):
    def __init__(self, in_ch, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_ch, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# Bloco Conv com opcional ResNet
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_res=True):
        super().__init__()
        self.use_res = use_res
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(), # <--- Corrigido: Removido inplace=True
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        if use_res:
            self.res_block = ResNetBlock(out_ch, out_ch)
        self.relu = nn.ReLU() # <--- Corrigido: Removido inplace=True

    def forward(self, x):
        x = self.conv(x)
        if self.use_res:
            x = self.res_block(x)
        x = self.relu(x)
        return x

# ----------------------------
# UpBlock com opcional ResNet e CBAM nos skips
# ----------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch_up, in_ch_skip, out_ch, use_res=True, use_cbam=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam_skip = CBAMBlock(in_ch_skip)
        
        self.conv_block = ConvBlock(out_ch + in_ch_skip, out_ch, use_res=use_res)

    def forward(self, x, skip):
        x = self.up(x)
        if self.use_cbam:
            skip = self.cbam_skip(skip)  # atenção aplicada **antes** da concat
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

# ----------------------------
# Dual Encoder UNet 4 níveis com ResNet, ASPP, CBAM e Deep Supervision
# ----------------------------
class DualEncoderUNet_Res_ASPP_CBAM_DeepSupervision(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, use_cbam=True, use_res=True):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_res = use_res

        # Encoder parte1
        self.enc1_1 = ConvBlock(input_nc, ngf, use_res=use_res)
        self.enc2_1 = ConvBlock(ngf, ngf*2, use_res=use_res)
        self.enc3_1 = ConvBlock(ngf*2, ngf*4, use_res=use_res)
        self.enc4_1 = ConvBlock(ngf*4, ngf*8, use_res=use_res)

        # Encoder parte2
        self.enc1_2 = ConvBlock(input_nc, ngf, use_res=use_res)
        self.enc2_2 = ConvBlock(ngf, ngf*2, use_res=use_res)
        self.enc3_2 = ConvBlock(ngf*2, ngf*4, use_res=use_res)
        self.enc4_2 = ConvBlock(ngf*4, ngf*8, use_res=use_res)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck com ASPP e ResNet
        self.bottleneck_aspp = ASPP(ngf*8*2, ngf*16)
        self.bottleneck_res = ResNetBlock(ngf*16*5, ngf*16*5)
        self.bottleneck_conv = ConvBlock(ngf*16*5, ngf*16, use_res=use_res)

        # Decoder com ResNet e CBAM nos skips
        self.dec4 = UpBlock(in_ch_up=ngf*16, in_ch_skip=ngf*8*2, out_ch=ngf*8, use_res=use_res, use_cbam=use_cbam)
        self.dec3 = UpBlock(in_ch_up=ngf*8, in_ch_skip=ngf*4*2, out_ch=ngf*4, use_res=use_res, use_cbam=use_cbam)
        self.dec2 = UpBlock(in_ch_up=ngf*4, in_ch_skip=ngf*2*2, out_ch=ngf*2, use_res=use_res, use_cbam=use_cbam)
        self.dec1 = UpBlock(in_ch_up=ngf*2, in_ch_skip=ngf*1*2, out_ch=ngf, use_res=use_res, use_cbam=use_cbam)

        # Saída
        self.final = nn.Conv2d(ngf, output_nc, 1)

        # Camadas para Deep Supervision
        self.ds3 = nn.Conv2d(ngf*4, output_nc, kernel_size=1)
        self.ds2 = nn.Conv2d(ngf*2, output_nc, kernel_size=1)
        self.ds1 = nn.Conv2d(ngf, output_nc, kernel_size=1)

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
        bottleneck_input = torch.cat([self.pool(e4_1), self.pool(e4_2)], dim=1)
        bottleneck_output = self.bottleneck_aspp(bottleneck_input)
        bottleneck_output = self.bottleneck_res(bottleneck_output)
        bottleneck_output = self.bottleneck_conv(bottleneck_output)

        # Decoder
        d4 = self.dec4(bottleneck_output, torch.cat([e4_1, e4_2], dim=1))
        d3 = self.dec3(d4, torch.cat([e3_1, e3_2], dim=1))
        d2 = self.dec2(d3, torch.cat([e2_1, e2_2], dim=1))
        d1 = self.dec1(d2, torch.cat([e1_1, e1_2], dim=1))

        out = self.final(d1)

        # Gerando as saídas de Deep Supervision
        out_ds3 = F.interpolate(self.ds3(d3), size=out.shape[2:], mode='bilinear', align_corners=False)
        out_ds2 = F.interpolate(self.ds2(d2), size=out.shape[2:], mode='bilinear', align_corners=False)
        out_ds1 = F.interpolate(self.ds1(d1), size=out.shape[2:], mode='bilinear', align_corners=False)
        
        deep_outputs = [out_ds3, out_ds2, out_ds1]

        return out, deep_outputs

# ----------------------------
# Teste rápido
# ----------------------------
if __name__ == "__main__":
    model = DualEncoderUNet_Res_ASPP_CBAM_DeepSupervision(input_nc=3, output_nc=3, ngf=32, use_cbam=True, use_res=True)
    part1 = torch.randn(4, 3, 256, 384)
    part2 = torch.randn(4, 3, 256, 384)
    out, ds_outs = model(part1, part2)
    print(f"Saída final do gerador: {out.shape}")
    for i, ds_out in enumerate(ds_outs):
        print(f"Saída de Deep Supervision {i+1} shape: {ds_out.shape}")