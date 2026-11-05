import os

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from data import LowLightDataset

_lpips_model = None


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)


def ssim(pred, target):
    pred = pred.permute(0, 2, 3, 1).cpu().numpy()
    target = target.permute(0, 2, 3, 1).cpu().numpy()
    return torch.tensor([structural_similarity(p, t, channel_axis=2, data_range=1.0)
                         for p, t in zip(pred, target)])


def lpips_score(pred, target):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    model = _lpips_model.to(pred.device)
    return model(pred, target, normalize=True).flatten()


@torch.no_grad()
def evaluate(predict, dataset, device="cpu"):
    # The batch size is 1, because the images have two orientations.
    rows = []
    loader = DataLoader(dataset, batch_size=1)
    for name, (inputs, targets) in zip(dataset.names, loader):
        targets = targets.to(device)
        preds = predict(inputs.to(device)).to(device).clamp(0, 1)
        rows.append({
            "file": name,
            "PSNR": psnr(preds, targets).item(),
            "SSIM": ssim(preds, targets).item(),
            "LPIPS": lpips_score(preds, targets).item(),
        })
    return pd.DataFrame(rows)


def equalize_histogram(img):
    # Equalize only the luma, because a separate equalization of R, G and B changes the colors.
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def baseline_predict(inputs):
    images = (inputs * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    equalized = np.stack([equalize_histogram(img) for img in images])
    return torch.from_numpy(equalized).permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def plot_comparison(predict, dataset, scores, method, path, count=4, stride=6):
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset))).round().astype(int)
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3.1 * len(indices)), squeeze=False)
    for row, index in zip(axes, indices):
        inputs, target = dataset[index]
        pred = predict(inputs.unsqueeze(0)).clamp(0, 1)[0]
        score = scores.iloc[index]
        titles = [f"input: {score['file']}", f"{method}: PSNR {score['PSNR']:.2f}, SSIM {score['SSIM']:.3f}, "
                  f"LPIPS {score['LPIPS']:.3f}", "ground truth"]
        for ax, img, title in zip(row, [inputs, pred, target], titles):
            ax.imshow(img.permute(1, 2, 0).cpu().numpy()[::stride, ::stride])
            ax.set_title(title, loc="left", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def run_baseline(set_dir, names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dataset = LowLightDataset(set_dir, names)
    scores = evaluate(baseline_predict, dataset)
    scores.to_csv(f"{out_dir}/scores.csv", index=False)
    plot_comparison(baseline_predict, dataset, scores, "baseline", f"{out_dir}/comparison.png")
    summary = scores[["PSNR", "SSIM", "LPIPS"]].mean().to_frame("baseline").T
    summary.to_csv(f"{out_dir}/summary.csv", index_label="method")
    return summary
