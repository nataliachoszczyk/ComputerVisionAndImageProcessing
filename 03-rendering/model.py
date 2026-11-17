"""Conditional GAN that renders the Phong sphere from the scene parameters."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

IMAGE_SIZE = 128
# Far from the sphere the exact relative position has no effect, and large values make the training unstable.
RELATIVE_LIMIT = 4.0


def pixel_grid(size, device):
    """Normalized device coordinates of the pixel centers (2, size, size). The y axis points up, as in data.features."""
    t = (torch.arange(size, device=device) + 0.5) / size * 2 - 1
    y, x = torch.meshgrid(-t, t, indexing="ij")
    return torch.stack([x, y])


def condition_maps(inputs, size, relative):
    """The inputs as constant channels, the pixel coordinates and (optional) the coordinates relative to the sphere.

    The relative coordinates use the image position and the radius of the sphere from data.features (CoordConv).
    """
    grid = pixel_grid(size, inputs.device).expand(len(inputs), -1, -1, -1)
    maps = [inputs[:, :, None, None].expand(-1, -1, size, size), grid]
    if relative:
        radius = torch.exp(inputs[:, 2] - 2)[:, None, None, None]
        maps.append(((grid - inputs[:, :2, None, None]) / radius).clamp(-RELATIVE_LIMIT, RELATIVE_LIMIT))
    return torch.cat(maps, 1)


def n_condition_channels(n_inputs, relative):
    return n_inputs + (4 if relative else 2)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return F.leaky_relu(self.conv2(F.leaky_relu(self.conv1(x), 0.2)), 0.2)


class Generator(nn.Module):
    """The inputs go to a 4x4 map. Each block doubles the size up to 128x128 and gets the condition maps again."""

    CHANNELS = [256, 256, 128, 128, 64, 32]

    def __init__(self, n_inputs, relative):
        super().__init__()
        self.relative = relative
        n_condition = n_condition_channels(n_inputs, relative)
        self.start = nn.Linear(n_inputs, self.CHANNELS[0] * 16)
        in_channels = [self.CHANNELS[0], *self.CHANNELS[:-1]]
        self.blocks = nn.ModuleList(Block(c_in + n_condition, c_out) for c_in, c_out in zip(in_channels, self.CHANNELS))
        self.to_rgb = nn.Conv2d(self.CHANNELS[-1], 3, 1)

    def forward(self, inputs):
        h = F.leaky_relu(self.start(inputs), 0.2).view(len(inputs), -1, 4, 4)
        for index, block in enumerate(self.blocks):
            if index:
                h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            h = block(torch.cat([h, condition_maps(inputs, h.shape[-1], self.relative)], 1))
        # A sigmoid gives an exact black background without a clamp.
        return torch.sigmoid(self.to_rgb(h))


class Discriminator(nn.Module):
    """Conditional PatchGAN with spectral normalization. Each output value judges one patch of the image."""

    def __init__(self, n_inputs, relative):
        super().__init__()
        self.relative = relative
        channels = [3 + n_condition_channels(n_inputs, relative), 64, 128, 256]
        layers = []
        for c_in, c_out in zip(channels, channels[1:]):
            layers += [spectral_norm(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1)), nn.LeakyReLU(0.2)]
        layers += [spectral_norm(nn.Conv2d(256, 256, 3, padding=1)), nn.LeakyReLU(0.2),
                   spectral_norm(nn.Conv2d(256, 1, 3, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, images, inputs):
        maps = condition_maps(inputs, images.shape[-1], self.relative)
        return self.net(torch.cat([images * 2 - 1, maps], 1))


def save_generator(generator, input_kind, path):
    config = {"n_inputs": generator.start.in_features, "relative": generator.relative, "input_kind": input_kind}
    torch.save({"config": config, "state": generator.state_dict()}, path)


def load_generator(path, device="cpu"):
    """Returns the generator in eval mode and the kind of its inputs (a key of data.INPUTS)."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    generator = Generator(config["n_inputs"], config["relative"]).to(device)
    generator.load_state_dict(checkpoint["state"])
    return generator.eval(), config["input_kind"]
