import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True):
        super().__init__()
        if down:
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        else:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True) if not down else nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNetStackGenerator(nn.Module):
    def __init__(self, input_nc=6, output_nc=3, ngf=64):
        super().__init__()
        input_nc = 6  # part1 (RGB) + part2 (RGB)

        # Encoder
        self.enc1 = UNetBlock(input_nc, ngf, down=True, use_bn=False)
        self.enc2 = UNetBlock(ngf, ngf*2)
        self.enc3 = UNetBlock(ngf*2, ngf*4)
        self.enc4 = UNetBlock(ngf*4, ngf*8)

        # Bottleneck
        self.bottleneck = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1)

        # Decoder
        self.dec4 = UNetBlock(ngf*8, ngf*8, down=False)
        self.dec3 = UNetBlock(ngf*16, ngf*4, down=False)  # skip connection concat
        self.dec2 = UNetBlock(ngf*8, ngf*2, down=False)
        self.dec1 = UNetBlock(ngf*4, ngf, down=False)

        self.final = nn.ConvTranspose2d(ngf*2, output_nc, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, part1, part2):
        # concatena internamente igual ao ResnetGenerator
        x = torch.cat([part1, part2], dim=1)  # [B, 6, H, W]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        out = self.final(d1)
        return self.tanh(out)
