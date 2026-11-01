import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, scale=1, c=32, upsample="bicubic"):
        super().__init__()
        self.scale = scale
        self.upsample = upsample

        # learned upsampling instead of bicubic: conv + pixel shuffle
        if scale > 1 and upsample == "learned":
            self.up_in = nn.Sequential(
                nn.Conv2d(3, 3 * scale ** 2, 3, padding=1),
                nn.PixelShuffle(scale),
            )

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

        # zero init: output starts as the upsampled input
        self.out = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        if self.scale > 1:
            if self.upsample == "learned":
                x = self.up_in(x)
            else:
                x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # encoder: 256 -> 128 -> 64 -> 32
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # decoder: 32 -> 64 -> 128 -> 256
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return x + self.out(d1)