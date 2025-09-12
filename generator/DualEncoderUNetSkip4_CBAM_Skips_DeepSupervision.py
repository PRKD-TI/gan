import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Bloco Residual (ResNetBlock)
# ----------------------------
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # <--- Adicionado bias=False explicitamente se usar InstanceNorm
            nn.InstanceNorm2d(out_channels), # <--- Alterado para InstanceNorm2d
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # <--- Adicionado bias=False
            nn.InstanceNorm2d(out_channels)  # <--- Alterado para InstanceNorm2d
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out

# ----------------------------
# ASPP (Atrous Spatial Pyramid Pooling)
# ----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Conv com bias=False para consistência com InstanceNorm se aplicada aqui
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
            nn.ReLU(), # <--- Corrigido: Removido inplace=True
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

# ----------------------------
# Bloco de Atenção Não-Local (NonLocalBlock)
# ----------------------------
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        # Conv com bias=False para consistência com InstanceNorm se aplicada aqui
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)

        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0) # <--- Removido, pois bias=False

        self.norm = nn.InstanceNorm2d(in_channels) # <--- Alterado para InstanceNorm2d

    def forward(self, x):
        batch_size, C, H, W = x.size()

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_N = f / N

        y = torch.matmul(f_div_N, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)

        W_y = self.W(y)
        out = self.norm(W_y + x) # Adição residual após InstanceNorm

        return out


# Bloco Conv com opcional ResNet
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_res=True):
        super().__init__()
        self.use_res = use_res
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), # <--- Adicionado bias=False
            nn.InstanceNorm2d(out_ch), # <--- Alterado para InstanceNorm2d
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), # <--- Adicionado bias=False
            nn.InstanceNorm2d(out_ch) # <--- Alterado para InstanceNorm2d
        )
        if use_res:
            self.res_block = ResNetBlock(out_ch, out_ch)
        self.relu = nn.ReLU()

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
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2, bias=False) # <--- Adicionado bias=False
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam_skip = CBAMBlock(in_ch_skip)
        
        self.conv_block = ConvBlock(out_ch + in_ch_skip, out_ch, use_res=use_res)

    def forward(self, x, skip):
        x = self.up(x)
        if self.use_cbam:
            skip = self.cbam_skip(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

# ----------------------------
# Dual Encoder UNet 4 níveis com ResNet, ASPP, CBAM e Deep Supervision
# ----------------------------
class DualEncoderUNet_Res_ASPP_CBAM_DeepSupervision(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, use_cbam=True, use_res=True, use_non_local=False):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_res = use_res
        self.use_non_local = use_non_local

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

        # Bottleneck com ASPP, ResNet e Non-Local (opcional)
        self.bottleneck_aspp = ASPP(ngf*8*2, ngf*16)
        self.bottleneck_res_1 = ResNetBlock(ngf*16*5, ngf*16*5) 
        self.bottleneck_conv = ConvBlock(ngf*16*5, ngf*16, use_res=use_res)

        if self.use_non_local:
            self.non_local_block = NonLocalBlock(ngf*16)

        # Decoder com ResNet e CBAM nos skips
        self.dec4 = UpBlock(in_ch_up=ngf*16, in_ch_skip=ngf*8*2, out_ch=ngf*8, use_res=use_res, use_cbam=use_cbam)
        self.dec3 = UpBlock(in_ch_up=ngf*8, in_ch_skip=ngf*4*2, out_ch=ngf*4, use_res=use_res, use_cbam=use_cbam)
        self.dec2 = UpBlock(in_ch_up=ngf*4, in_ch_skip=ngf*2*2, out_ch=ngf*2, use_res=use_res, use_cbam=use_cbam)
        self.dec1 = UpBlock(in_ch_up=ngf*2, in_ch_skip=ngf*1*2, out_ch=ngf, use_res=use_res, use_cbam=use_cbam)

        # Saída
        self.final = nn.Conv2d(ngf, output_nc, 1) # Geralmente a última conv não usa norm.

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
        bottleneck_input_enc = torch.cat([e4_1, e4_2], dim=1)
        bottleneck_input_pooled = self.pool(bottleneck_input_enc)

        bottleneck_output = self.bottleneck_aspp(bottleneck_input_pooled)
        bottleneck_output = self.bottleneck_res_1(bottleneck_output)
        bottleneck_output = self.bottleneck_conv(bottleneck_output)

        if self.use_non_local:
            bottleneck_output = self.non_local_block(bottleneck_output)

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
    print("Testando modelo COM InstanceNorm e SEM NonLocalBlock:")
    model_instancenorm_no_nl = DualEncoderUNet_Res_ASPP_CBAM_DeepSupervision(input_nc=3, output_nc=3, ngf=32, use_cbam=True, use_res=True, use_non_local=False)
    part1 = torch.randn(2, 3, 256, 384)
    part2 = torch.randn(2, 3, 256, 384)
    out_instancenorm_no_nl, ds_outs_instancenorm_no_nl = model_instancenorm_no_nl(part1, part2)
    print(f"Saída final do gerador (InstanceNorm, sem NL): {out_instancenorm_no_nl.shape}")
    for i, ds_out in enumerate(ds_outs_instancenorm_no_nl):
        print(f"Saída de Deep Supervision {i+1} shape (InstanceNorm, sem NL): {ds_out.shape}")

    print("\nTestando modelo COM InstanceNorm e COM NonLocalBlock:")
    model_instancenorm_with_nl = DualEncoderUNet_Res_ASPP_CBAM_DeepSupervision(input_nc=3, output_nc=3, ngf=32, use_cbam=True, use_res=True, use_non_local=True)
    out_instancenorm_with_nl, ds_outs_instancenorm_with_nl = model_instancenorm_with_nl(part1, part2)
    print(f"Saída final do gerador (InstanceNorm, com NL): {out_instancenorm_with_nl.shape}")
    for i, ds_out in enumerate(ds_outs_instancenorm_with_nl):
        print(f"Saída de Deep Supervision {i+1} shape (InstanceNorm, com NL): {ds_out.shape}")