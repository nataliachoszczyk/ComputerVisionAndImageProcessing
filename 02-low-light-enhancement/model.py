import torch
import torch.nn as nn
import torch.nn.functional as F

GLOBAL_SIZE = 64


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class GlobalBranch(nn.Module):
    """Make one vector that describes the brightness and the colors of the full image."""

    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in [16, 32, 64, 128]:
            layers += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(in_ch, GLOBAL_SIZE), nn.ReLU(inplace=True))

    def forward(self, thumb):
        return self.net(thumb)


class FiLM(nn.Module):
    """Scale and shift the feature channels with the global vector."""

    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(GLOBAL_SIZE, 2 * channels)
        # Zero init: at the start, the global vector does not change the features.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, vector):
        scale, shift = self.linear(vector)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class UNet(nn.Module):
    def __init__(self, c=32, global_branch=False):
        super().__init__()
        self.enc1 = conv_block(3, c)
        self.enc2 = conv_block(c, 2 * c)
        self.enc3 = conv_block(2 * c, 4 * c)
        self.bottleneck = conv_block(4 * c, 8 * c)

        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = conv_block(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = conv_block(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = conv_block(2 * c, c)

        self.out = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.global_branch = GlobalBranch() if global_branch else None
        if global_branch:
            self.film_b = FiLM(8 * c)
            self.film3 = FiLM(4 * c)
            self.film2 = FiLM(2 * c)
            self.film1 = FiLM(c)

    def forward(self, x, thumb=None):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        if self.global_branch is None:
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return x + self.out(d1)

        vector = self.global_branch(thumb)
        b = self.film_b(b, vector)
        d3 = self.film3(self.dec3(torch.cat([self.up3(b), e3], dim=1)), vector)
        d2 = self.film2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)), vector)
        d1 = self.film1(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), vector)
        return x + self.out(d1)


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.spatial = nn.Sequential(
            nn.Conv2d(c, 2 * c, 1),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c),
            SimpleGate(),
        )
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.spatial_out = nn.Conv2d(c, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.feed_forward = nn.Sequential(nn.Conv2d(c, 2 * c, 1), SimpleGate(), nn.Conv2d(c, c, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.spatial(self.norm1(x))
        y = self.spatial_out(y * self.channel_attention(y))
        x = x + y * self.beta
        return x + self.feed_forward(self.norm2(x)) * self.gamma


class NAFNet(nn.Module):
    def __init__(self, c=32, enc_blocks=(1, 1, 2, 4), middle_blocks=4, dec_blocks=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(3, c, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        channels = c
        for count in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(count)]))
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2
        self.middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blocks)])
        for count in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, 2 * channels, 1), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(count)]))

        self.out = nn.Conv2d(c, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, thumb=None):
        y = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            y = encoder(y)
            skips.append(y)
            y = down(y)
        y = self.middle(y)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            y = decoder(up(y) + skip)
        return x + self.out(y)


MODELS = {
    "unet": UNet,
    "unet-global": lambda: UNet(global_branch=True),
    "nafnet": NAFNet,
}
