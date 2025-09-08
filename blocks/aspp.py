"""
ASPP (Atrous Spatial Pyramid Pooling)
-------------------------------------
Bloco ASPP inspirado em DeepLab, para aumentar o receptive field sem reduzir a resolução,
ideal para capturar contextos em múltiplas escalas em tarefas de imagem, como costura de imagens.

Características:
- Várias convoluções dilatadas em paralelo
- Pooling global para contexto de imagem inteira
- Concatenação das features extraídas
- Redução final de canais para integração no encoder, bottleneck ou decoder
- BatchNorm e ReLU após cada convolução
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18], use_bn=True):
        """
        Parâmetros:
        -----------
        in_channels : int
            Número de canais de entrada
        out_channels : int
            Número de canais de saída (após concatenação e redução)
        dilations : list[int]
            Taxas de dilatação para as convoluções paralelas
        use_bn : bool
            Usar BatchNorm após cada convolução
        """
        super(ASPP, self).__init__()

        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
                    nn.ReLU(inplace=True)
                )
            )

        # Pooling global (contexto de imagem inteira)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        # Redução após concatenação
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        res = []
        for branch in self.branches:
            res.append(branch(x))

        # Global pooling
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        res.append(gp)

        out = torch.cat(res, dim=1)
        out = self.project(out)
        return out


# Teste rápido quando executado diretamente
if __name__ == "__main__":
    x = torch.randn(1, 64, 128, 128)  # batch=1, 64 canais, 128x128
    aspp = ASPP(64, 128, dilations=[1, 6, 12, 18])
    y = aspp(x)
    print("Input:", x.shape, "Output:", y.shape)
