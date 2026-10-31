from functools import partial
from pathlib import Path

import numpy as np
import torch
from skimage.util import random_noise
from torch.utils.data import Dataset

from data import degrade_patches, load_image

SIGMAS = [0.01, 0.03]


def add_noise(img, sigma, rng):
    # random_noise works on [0, 1] floats and takes the variance, not the std
    noisy = random_noise(img, mode="gaussian", var=sigma ** 2, rng=rng)
    return np.round(noisy * 255).astype(np.uint8)


def make_noisy_patches(patches_root, seed=42):
    rng = np.random.default_rng(seed)
    for sigma in SIGMAS:
        degrade_patches(patches_root, f"noise_{sigma}", partial(add_noise, sigma=sigma, rng=rng))


class OnlineNoiseDataset(Dataset):
    """Clean patches with fresh noise (and a random flip / 90° rotation) every time they are drawn."""

    def __init__(self, clean_dir, sigma):
        self.input_paths = sorted(Path(clean_dir).glob("*.png"))
        self.sigma = sigma

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        clean = load_image(self.input_paths[idx])
        clean = torch.rot90(clean, int(torch.randint(4, ())), dims=(1, 2))
        if torch.rand(()) < 0.5:
            clean = clean.flip(2)

        # same clipping and 8-bit quantization as add_noise
        noisy = clean + self.sigma * torch.randn_like(clean)
        noisy = (noisy.clamp(0, 1) * 255).round() / 255
        return noisy, clean
