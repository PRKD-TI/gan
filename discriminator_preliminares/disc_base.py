# Disc Base
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicDiscriminator(nn.Module):
    def __init__(self, in_channels=9, features=64):
        super(BasicDiscriminator, self).__init__()
        # in_channels: part1(3) + part2(3) + imagem gerada ou real(3) = 9 canais

        self.layer1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1)
        self.final = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)

        self.bn2 = nn.BatchNorm2d(features * 2)
        self.bn3 = nn.BatchNorm2d(features * 4)
        self.bn4 = nn.BatchNorm2d(features * 8)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.layer2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.layer3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.layer4(x)), 0.2)
        out = self.final(x)  # saída sem ativação (logits)
        return out
