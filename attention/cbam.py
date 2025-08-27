# Cbam
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        # Spatial Attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Channel attention
        avg = self.mlp(self.avg_pool(x).view(b, c))
        max = self.mlp(self.max_pool(x).view(b, c))
        ca = self.sigmoid(avg + max).view(b, c, 1, 1)
        x = x * ca
        # Spatial attention
        sa = self.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)))
        x = x * sa
        return x
