import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout=False):
        super(ResnetBlock, self).__init__()
        block = [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            norm_layer(dim)
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, output_nc=3, ngf=64, n_blocks=6, norm_layer=nn.BatchNorm2d):
        """Recebe part1 (RGB) e part2 (RGB) e concatena internamente"""
        super(ResnetGenerator, self).__init__()
        input_nc = 6  # 3 canais part1 + 3 canais part2

        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        # downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2,
                          kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer)]

        # upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, part1, part2):
        # Concatena internamente
        x = torch.cat([part1, part2], dim=1)  # [B, 6, H, W]
        return self.model(x)
