from functools import partial

import numpy as np
from skimage.util import random_noise

from data import degrade_patches

SIGMAS = [0.01, 0.03]


def add_noise(img, sigma, rng):
    # random_noise works on [0, 1] floats and takes the variance, not the std
    noisy = random_noise(img, mode="gaussian", var=sigma ** 2, rng=rng)
    return np.round(noisy * 255).astype(np.uint8)


def make_noisy_patches(patches_root, seed=42):
    rng = np.random.default_rng(seed)
    for sigma in SIGMAS:
        degrade_patches(patches_root, f"noise_{sigma}", partial(add_noise, sigma=sigma, rng=rng))
