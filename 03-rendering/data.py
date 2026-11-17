"""Scene sampling, input features and loading of the Phong data set."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


FEATURE_NAMES = ["image_x", "image_y", "image_radius", "light_dir_x", "light_dir_y", "light_dir_z",
                 "light_distance", "diffuse_r", "diffuse_g", "diffuse_b", "shininess"]


def features(params):
    """Change the raw scene parameters to the input of the network.

    The camera does not move. Thus the image position and size of the sphere, and the light direction
    in camera coordinates, describe the image better than world positions. All values are approximately in [-1, 1].
    """
    right, up, forward = camera_basis()
    basis = np.stack([right, up, forward])
    obj = params[["object_x", "object_y", "object_z"]].to_numpy()
    light = params[["light_x", "light_y", "light_z"]].to_numpy()
    offset = (obj - CAMERA_POSITION) @ basis.T
    tan_half = np.tan(np.radians(FOV_DEGREES) / 2)
    image_xy = offset[:, :2] / (offset[:, 2:] * tan_half)  # Normalized device coordinates, y points up.
    image_radius = SPHERE_RADIUS / (offset[:, 2] * tan_half)
    to_light = (light - obj) @ basis.T
    distance = np.linalg.norm(to_light, axis=1, keepdims=True)
    diffuse = params[["diffuse_r", "diffuse_g", "diffuse_b"]].to_numpy() / 255.0
    shininess = (params[["shininess"]].to_numpy() - SHININESS_RANGE[0]) / (SHININESS_RANGE[1] - SHININESS_RANGE[0])
    max_distance = 2 * np.sqrt(3) * POSITION_RANGE
    # A log scale, because near spheres are much larger than far spheres.
    return np.hstack([image_xy, np.log(image_radius)[:, None] + 2, to_light / distance, distance / max_distance,
                      diffuse, shininess]).astype(np.float32)


def load_split(source, split):
    """Load one split from the data set folder or zip. Returns the features (N, F) and images (N, 3, H, W) in [0, 1]."""
    source = Path(source)
    if source.suffix == ".zip":
        archive = zipfile.ZipFile(source)
        read = archive.read
    else:
        read = lambda name: (source / name).read_bytes()
    params = pd.read_csv(io.BytesIO(read("params.csv")))
    params = params[params["split"] == split].reset_index(drop=True)
    images = np.stack([np.asarray(Image.open(io.BytesIO(read(f"image_{i:04}.png"))).convert("RGB"))
                       for i in params["index"]])
    x = torch.from_numpy(features(params))
    y = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    return x, y.float() / 255.0
