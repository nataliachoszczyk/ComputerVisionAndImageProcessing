from functools import partial

import cv2

from data import degrade_patches

SCALES = [4, 8]


def downscale(img, scale):
    size = (img.shape[1] // scale, img.shape[0] // scale)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def make_lr_patches(patches_root):
    for scale in SCALES:
        degrade_patches(patches_root, f"lr_x{scale}", partial(downscale, scale=scale))