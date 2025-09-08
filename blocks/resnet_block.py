"""
ResNetBlock
-----------
Bloco residual padrão inspirado em ResNet, para uso em encoder, bottleneck ou decoder.

Características:
- Dois blocos convolucionais 3x3 com padding/dilatação ajustáveis
- Normalização configurável (BatchNorm ou InstanceNorm)
- Ativação configurável (ReLU ou LeakyReLU)
- Conexão residual com ajuste automático de canais
- Suporte a dilatação para uso em ASPP ou receptive field aumentado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm="batch", activation="relu", dilation=1):
        """
        Parâmetros:
        -----------
        in_channels : int
            Número de canais da entrada
        out_channels : int
            Número de canais da saída
        norm : str
            Tipo de normalização: 'batch' ou 'instance'
        activation : str
            Tipo de ativação: 'relu' ou 'leakyrelu'
        dilation : int
            Dilatação da convolução (para receptive field maior)
        """
        super(ResNetBlock, self).__init__()

        # Normalização
        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError("norm deve ser 'batch' ou 'instance'")

        # Ativação
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError("activation deve ser 'relu' ou 'leakyrelu'")

        # Convoluções principais
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.norm1 = norm_layer(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.norm2 = norm_layer(out_channels)

        # Ajuste do canal no atalho (caso in_channels != out_channels)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Passagem forward do bloco residual.
        """
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        out = self.activation(out)

        return out


# Teste rápido quando executado diretamente
if __name__ == "__main__":
    x = torch.randn(1, 64, 128, 128)  # batch=1, 64 canais, 128x128
    block = ResNetBlock(64, 128, norm="batch", activation="relu")
    y = block(x)
    print("Input:", x.shape, "Output:", y.shape)
