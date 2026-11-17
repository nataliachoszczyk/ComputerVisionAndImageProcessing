"""The four metrics of the task: FLIP, LPIPS, SSIM and the Hausdorff distance between Canny edges.

All functions take RGB images as float arrays of shape (H, W, 3) with values in [0, 1].
"""
import cv2
import lpips
import numpy as np
import pandas as pd
import torch
from flip_evaluator import evaluate as flip_evaluate
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity

CANNY_THRESHOLDS = (50, 150)
METRICS = ["FLIP", "LPIPS", "SSIM", "Hausdorff"]

_lpips_model = None


def flip(pred, target):
    return float(flip_evaluate(target.astype(np.float32), pred.astype(np.float32), "LDR",
                               applyMagma=False)[1])


def flip_map(pred, target):
    """The FLIP error of each pixel with the magma color map, as an RGB image."""
    return np.asarray(flip_evaluate(target.astype(np.float32), pred.astype(np.float32), "LDR")[0])


def lpips_score(pred, target):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    to_tensor = lambda img: torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        return _lpips_model(to_tensor(pred), to_tensor(target), normalize=True).item()


def ssim(pred, target):
    return float(structural_similarity(pred, target, channel_axis=2, data_range=1.0))


def edges(img):
    gray = cv2.cvtColor((img * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return np.argwhere(cv2.Canny(gray, *CANNY_THRESHOLDS) > 0)


def hausdorff(pred, target):
    """Symmetric Hausdorff distance in pixels between the edge pixels of the two images."""
    a, b = edges(pred), edges(target)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        # The distance is not defined. The image diagonal is the largest possible distance.
        return float(np.hypot(*pred.shape[:2]))
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def score(pred, target):
    return {"FLIP": flip(pred, target), "LPIPS": lpips_score(pred, target),
            "SSIM": ssim(pred, target), "Hausdorff": hausdorff(pred, target)}


def evaluate(preds, targets):
    """One row of metrics for each image pair."""
    return pd.DataFrame([score(p, t) for p, t in zip(preds, targets)])


CHANGES = {
    "same image": lambda img: img,
    "shift 2 px": lambda img: np.roll(img, 2, axis=1),
    "shift 10 px": lambda img: np.roll(img, 10, axis=1),
    "swap R and B": lambda img: img[..., ::-1],
    "50% darker": lambda img: img * 0.5,
    "black image": lambda img: np.zeros_like(img),
}


def change_test(targets):
    """Mean metrics of simple changes of the reference images. It shows which errors each metric sees."""
    return pd.DataFrame({name: evaluate([change(t) for t in targets], targets).mean()
                         for name, change in CHANGES.items()}).T
