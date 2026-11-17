"""Scene sampling for the Phong data set."""
from pathlib import Path

import numpy as np
from PIL import Image

N_IMAGES = 3000
N_TEST = 600
CAMERA_POSITION = np.array([5.0, 5.0, 15.0])
CAMERA_TARGET = np.array([0.0, 0.0, 0.0])
CAMERA_UP = np.array([0.0, 1.0, 0.0])
FOV_DEGREES = 45.0
NEAR = 0.1
SPHERE_RADIUS = 1.0
POSITION_RANGE = 20.0
SHININESS_RANGE = (3.0, 20.0)
# The light must stay outside the sphere, else the sphere is only ambient.
MIN_LIGHT_DISTANCE = 2.0 * SPHERE_RADIUS
# A small margin, so that the rasterized sphere does not touch the image border.
VIEW_MARGIN = 1.1 * SPHERE_RADIUS

PARAM_COLUMNS = ["object_x", "object_y", "object_z", "diffuse_r", "diffuse_g", "diffuse_b",
                 "shininess", "light_x", "light_y", "light_z"]


def camera_basis():
    forward = CAMERA_TARGET - CAMERA_POSITION
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, CAMERA_UP)
    right /= np.linalg.norm(right)
    return right, np.cross(right, forward), forward


def sphere_in_view(position):
    """True if the full sphere is in the camera frustum."""
    right, up, forward = camera_basis()
    offset = position - CAMERA_POSITION
    depth = offset @ forward
    if depth - SPHERE_RADIUS < NEAR:
        return False
    half = np.radians(FOV_DEGREES) / 2
    # The distance from the center to each side plane of the frustum must be at least the radius.
    for side in (offset @ right, offset @ up):
        if (depth * np.tan(half) - abs(side)) * np.cos(half) < VIEW_MARGIN:
            return False
    return True


def sample_scene(rng):
    """Sample one valid scene. Colors are integers in 0-255, as in the task text."""
    while True:
        position = rng.uniform(-POSITION_RANGE, POSITION_RANGE, 3)
        light = rng.uniform(-POSITION_RANGE, POSITION_RANGE, 3)
        if sphere_in_view(position) and np.linalg.norm(light - position) >= MIN_LIGHT_DISTANCE:
            break
    return {
        "object_position": position,
        "diffuse": rng.integers(0, 256, 3),
        "shininess": rng.uniform(*SHININESS_RANGE),
        "light_position": light,
    }


def scene_row(index, scene):
    split = "test" if index >= N_IMAGES - N_TEST else "train"
    return [index, split, *scene["object_position"], *scene["diffuse"], scene["shininess"], *scene["light_position"]]


def save_preview(image_dir, path, rows=4, cols=8):
    images = [np.asarray(Image.open(p)) for p in sorted(Path(image_dir).glob("image_*.png"))[:rows * cols]]
    grid = np.vstack([np.hstack(images[r * cols:(r + 1) * cols]) for r in range(rows)])
    Image.fromarray(grid).save(path)
