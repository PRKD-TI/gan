# File: generator_blocks.py
# Descrição: Contém todos os blocos de construção reutilizáveis 
# para os geradores da dissertação.

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Bloco Conv Padrão (Baseline)
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
# Bloco Residual (para T002 e outros)
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        
        # Shortcut connection
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, padding=0)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv(x)
        return self.relu(x + shortcut)

# ----------------------------
# Blocos de atenção: CBAM (para T003, T004 e outros)
# (Este é o seu código original)
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

# ----------------------------
# Bloco Bottleneck: ASPP (para T006, T011, T013, T015)
# ----------------------------
class ASPP_Module(nn.Module):
    def __init__(self, in_ch, out_ch, rates=[1, 6, 12, 18]):
        super().__init__()
        self.convs = nn.ModuleList()
        # 1x1 conv
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        )
        # 3x3 atrous convs
        for rate in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
        # Image Pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # Concat conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_ch + out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        features = [conv(x) for conv in self.convs]
        img_pool = F.interpolate(self.image_pool(x), size=size, mode='bilinear', align_corners=False)
        features.append(img_pool)
        
        x = torch.cat(features, dim=1)
        return self.final_conv(x)

# ----------------------------
# Bloco Bottleneck: NonLocal (Self-Attention) (para T007, T012)
# ----------------------------
class NonLocalBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.inter_ch = in_ch // 2
        
        self.theta = nn.Conv2d(in_ch, self.inter_ch, 1)
        self.phi = nn.Conv2d(in_ch, self.inter_ch, 1)
        self.g = nn.Conv2d(in_ch, self.inter_ch, 1)
        
        self.out_conv = nn.Conv2d(self.inter_ch, in_ch, 1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        theta_x = self.theta(x).view(batch_size, self.inter_ch, -1).permute(0, 2, 1) # B, H*W, C//2
        phi_x = self.phi(x).view(batch_size, self.inter_ch, -1) # B, C//2, H*W
        g_x = self.g(x).view(batch_size, self.inter_ch, -1).permute(0, 2, 1) # B, H*W, C//2
        
        # Matmul e Softmax (Attention Map)
        attn = torch.bmm(theta_x, phi_x) # B, H*W, H*W
        attn = self.softmax(attn)
        
        # Matmul com 'g'
        y = torch.bmm(attn, g_x).permute(0, 2, 1) # B, C//2, H*W
        y = y.view(batch_size, self.inter_ch, H, W)
        
        # Conexão residual
        return x + self.out_conv(y)

# ----------------------------
# Blocos de Decoder (UpBlock) Variações
# ----------------------------

# 1. UpBlock Baseline (para T001, T002, T005, T006, T007, T009, T011, T012)
class UpBlockBaseline(nn.Module):
    def __init__(self, in_ch_up, in_ch_skip, out_ch, conv_block_type=ConvBlock):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
        self.conv = conv_block_type(out_ch + in_ch_skip, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# 2. UpBlock com CBAM na Skip (para T003, T008, T010, T013, T014, T015)
class UpBlockCBAMSkip(nn.Module):
    def __init__(self, in_ch_up, in_ch_skip, out_ch, conv_block_type=ConvBlock):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
        self.cbam_skip = CBAMBlock(in_ch_skip) # Aplica na skip
        self.conv = conv_block_type(out_ch + in_ch_skip, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.cbam_skip(skip) # Atenção aplicada antes da concat
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# 3. UpBlock com CBAM no Decoder (para T004)
class UpBlockCBAMDecoder(nn.Module):
    def __init__(self, in_ch_up, in_ch_skip, out_ch, conv_block_type=ConvBlock):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
        self.conv = conv_block_type(out_ch + in_ch_skip, out_ch)
        self.cbam_out = CBAMBlock(out_ch) # Aplica na saída do bloco
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.cbam_out(x) # Atenção aplicada depois da conv
        return x
    
    # File: generator_blocks.py (ADICIONAR ESTES BLOCOS AO FINAL)

# ----------------------------
# Bloco de Atenção Cruzada (Cross-Attention)
# Baseado no NonLocalBlock, mas Q, K, V são entradas separadas.
# ----------------------------
class CrossAttentionModule(nn.Module):
    """ Aplica atenção de Q (Query) para K (Key) e V (Value) """
    def __init__(self, in_ch_q, in_ch_kv):
        super().__init__()
        self.in_ch_q = in_ch_q
        self.in_ch_kv = in_ch_kv
        self.inter_ch = in_ch_q // 2
        
        # Q vem de x1, K e V vêm de x2
        self.theta = nn.Conv2d(self.in_ch_q, self.inter_ch, 1)
        self.phi = nn.Conv2d(self.in_ch_kv, self.inter_ch, 1)
        self.g = nn.Conv2d(self.in_ch_kv, self.inter_ch, 1)
        
        self.out_conv = nn.Conv2d(self.inter_ch, self.in_ch_q, 1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v):
        batch_size = q.size(0)
        H, W = q.shape[2:]
        
        theta_q = self.theta(q).view(batch_size, self.inter_ch, -1).permute(0, 2, 1) # B, H*W, C//2
        phi_k = self.phi(k).view(batch_size, self.inter_ch, -1) # B, C//2, H*W
        g_v = self.g(v).view(batch_size, self.inter_ch, -1).permute(0, 2, 1) # B, H*W, C//2
        
        # Matmul e Softmax (Attention Map)
        attn = torch.bmm(theta_q, phi_k) # B, H*W, H*W
        attn = self.softmax(attn)
        
        # Matmul com 'g' (Value)
        y = torch.bmm(attn, g_v).permute(0, 2, 1) # B, C//2, H*W
        y = y.view(batch_size, self.inter_ch, H, W)
        
        # Conexão residual
        return q + self.out_conv(y)

# ----------------------------
# Bloco de Fusão de Skips com Atenção Cruzada
# ----------------------------
class CrossAttentionSkipFusion(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.attn_1_on_2 = CrossAttentionModule(in_ch, in_ch)
        self.attn_2_on_1 = CrossAttentionModule(in_ch, in_ch)
        
    def forward(self, x1, x2):
        # Imagem 1 atende à Imagem 2
        x1_refined = self.attn_1_on_2(q=x1, k=x2, v=x2)
        # Imagem 2 atende à Imagem 1
        x2_refined = self.attn_2_on_1(q=x2, k=x1, v=x1)
        
        # Concatena os features refinados para o decoder
        return torch.cat([x1_refined, x2_refined], dim=1)

# ----------------------------
# Variação de UpBlock para Cross-Attention
# (Este bloco inteligentemente separa o tensor 'skip' concatenado)
# ----------------------------
class UpBlockCrossAttentionSkip(nn.Module):
    def __init__(self, in_ch_up, in_ch_skip, out_ch, conv_block_type=ConvBlock):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
        
        # in_ch_skip é o total (ex: 512*2). O canal de cada skip é in_ch_skip // 2
        self.fusion = CrossAttentionSkipFusion(in_ch_skip // 2)
        
        self.conv = conv_block_type(out_ch + in_ch_skip, out_ch)

    def forward(self, x, skip_cat):
        x = self.up(x)
        
        # Separa o tensor de skip concatenado (vindo do forward do gerador)
        channels = skip_cat.shape[1] // 2
        skip_1 = skip_cat[:, :channels, :, :]
        skip_2 = skip_cat[:, channels:, :, :]
        
        # Aplica a fusão com atenção cruzada
        skip_fused = self.fusion(skip_1, skip_2)
        
        x = torch.cat([x, skip_fused], dim=1)
        x = self.conv(x)
        return x
    
    # File: generator_blocks.py (ADICIONAR ESTA CLASSE AO FINAL)

# ----------------------------
# Variação de UpBlock para T019: Cross-Attn (Skips) + CBAM (Decoder)
# ----------------------------
class UpBlockCrossAttnSkip_CBAMDecoder(nn.Module):
    def __init__(self, in_ch_up, in_ch_skip, out_ch, conv_block_type=ConvBlock):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, out_ch, kernel_size=2, stride=2)
        
        # 1. Cross-Attention on Skips (T016)
        # (Assume que CrossAttentionSkipFusion está definido acima neste arquivo)
        self.fusion = CrossAttentionSkipFusion(in_ch_skip // 2) 
        
        self.conv = conv_block_type(out_ch + in_ch_skip, out_ch)
        
        # 2. CBAM on Decoder output (T004)
        # (Assume que CBAMBlock está definido acima neste arquivo)
        self.cbam_out = CBAMBlock(out_ch) 

    def forward(self, x, skip_cat):
        x = self.up(x)
        
        # Separa os skips concatenados
        channels = skip_cat.shape[1] // 2
        skip_1 = skip_cat[:, :channels, :, :]
        skip_2 = skip_cat[:, channels:, :, :]
        
        # Aplica a fusão com atenção cruzada
        skip_fused = self.fusion(skip_1, skip_2)
        
        x = torch.cat([x, skip_fused], dim=1)
        x = self.conv(x)
        
        # Aplica atenção na saída do decoder
        x = self.cbam_out(x)
        
        return x