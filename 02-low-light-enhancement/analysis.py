"""Analysis of the colors and the clipped highlights on the test set, for the baseline and the networks."""
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from skimage.color import rgb2lab

from data import METHOD_COLORS, LowLightDataset, _clean_axes, image_stats, to_tensor
from metrics import baseline_predict
from model import MODELS
from training import predict_tiles

CLIP_LEVEL = 250 / 255
REGIONS = ["shadows", "midtones", "highlights", "clipped"]
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
ERROR_CMAP = LinearSegmentedColormap.from_list("error", ["#ffffff", "#86b6ef", "#2a78d6", "#0d366b"])


def region_masks(gt):
    """Split the pixels by the ground truth: a clipped pixel has one channel >= 250."""
    luma = gt @ LUMA
    clipped = gt.max(axis=2) >= CLIP_LEVEL
    shadows = (luma < 0.1) & ~clipped
    highlights = (luma >= 0.7) & ~clipped
    return {"shadows": shadows, "midtones": ~(clipped | shadows | highlights), "highlights": highlights,
            "clipped": clipped}


def psnr_np(pred, gt):
    return 10 * np.log10(1 / max(((pred - gt) ** 2).mean(), 1e-10))


def gain_fit(pred, gt):
    # The best scale for each channel: it removes a global error of the exposure and of the white balance.
    scale = (pred * gt).sum(axis=(0, 1)) / np.maximum((pred * pred).sum(axis=(0, 1)), 1e-10)
    return np.clip(pred * scale, 0, 1)


def color_scores(pred, gt, gamma=2.2, stride=4):
    pred_linear, gt_linear = pred ** gamma, gt ** gamma
    a, b = pred_linear.reshape(-1, 3).mean(0), gt_linear.reshape(-1, 3).mean(0)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    # Lab of every 4th pixel: the mean is the same, and rgb2lab of the full image is slow.
    lab_pred, lab_gt = rgb2lab(pred[::stride, ::stride]), rgb2lab(gt[::stride, ::stride])
    chroma_pred = np.hypot(lab_pred[..., 1], lab_pred[..., 2])
    chroma_gt = np.hypot(lab_gt[..., 1], lab_gt[..., 2])
    return {
        "exposure_ratio": (pred_linear @ LUMA).mean() / (gt_linear @ LUMA).mean(),
        "cast_deg": np.degrees(np.arccos(np.clip(cos, -1, 1))),
        "delta_e": np.linalg.norm(lab_pred - lab_gt, axis=2).mean(),
        "chroma_ratio": chroma_pred.mean() / max(chroma_gt.mean(), 1e-6),
    }


def clip_scores(pred, masks):
    pred_clipped = pred.max(axis=2) >= CLIP_LEVEL
    clipped = masks["clipped"]
    return {
        # Part of the clipped ground truth that is also clipped in the prediction.
        "clip_recall": pred_clipped[clipped].mean() if clipped.any() else np.nan,
        # Part of the not clipped ground truth that is clipped in the prediction.
        "false_clip": pred_clipped[~clipped].mean(),
    }


def region_scores(pred, gt, masks):
    luma_error = (pred - gt) @ LUMA
    abs_error = np.abs(pred - gt).mean(axis=2)
    row = {}
    for region, mask in masks.items():
        row[f"frac_{region}"] = mask.mean()
        row[f"mae_{region}"] = abs_error[mask].mean() * 255 if mask.any() else np.nan
        # Positive: the prediction is brighter than the ground truth.
        row[f"bias_{region}"] = luma_error[mask].mean() * 255 if mask.any() else np.nan
    return row


def score_image(pred, gt):
    masks = region_masks(gt)
    return {"PSNR": psnr_np(pred, gt), "PSNR_gain_fit": psnr_np(gain_fit(pred, gt), gt),
            **color_scores(pred, gt), **clip_scores(pred, masks), **region_scores(pred, gt, masks)}


def load_network(model_name, weights, device):
    model = MODELS[model_name]().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.eval()


def best_window(score_map, size):
    """Top left corner of the size x size window with the highest mean of score_map."""
    mean = cv2.boxFilter(score_map.astype(np.float32), -1, (size, size), normalize=True, borderType=cv2.BORDER_CONSTANT)
    half = size // 2
    height, width = score_map.shape
    valid = mean[half:height - half, half:width - half]
    y, x = np.unravel_index(valid.argmax(), valid.shape)
    return int(y), int(x)


def plot_crops(cases, inputs, gts, preds, methods, path, size=320):
    main = methods[1]
    columns = ["input", *methods, "ground truth", f"|{main} - ground truth|"]
    fig, axes = plt.subplots(len(cases), len(columns), figsize=(2.3 * len(columns), 2.6 * len(cases)), squeeze=False)
    for row, (case, index, score_map) in zip(axes, cases):
        y, x = best_window(score_map, size)
        window = (slice(y, y + size), slice(x, x + size))
        gt = gts[index][window] / 255.0
        error = np.abs(preds[main][index][window] / 255.0 - gt).mean(axis=2)
        images = [inputs[index][window], *(preds[m][index][window] for m in methods), gts[index][window]]
        for ax, img in zip(row, images):
            ax.imshow(img)
        shown = row[-1].imshow(error, cmap=ERROR_CMAP, vmin=0, vmax=0.25)
        row[0].set_title(f"{case}\ninput", loc="left", fontsize=8)
        for ax, title in zip(row[1:], columns[1:]):
            ax.set_title(title, loc="left", fontsize=8)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    fig.colorbar(shown, cax=fig.add_axes((0.93, 0.3, 0.01, 0.4)), label="mean absolute error")
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def plot_regions(scores, methods, path):
    panels = [("mae", "Mean absolute error (8-bit levels)"), ("bias", "Mean luma error (8-bit levels, + = too bright)")]
    means = scores.groupby("method").mean(numeric_only=True)
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (prefix, title) in zip(axes, panels):
        for i, method in enumerate(methods):
            values = [means.loc[method, f"{prefix}_{region}"] for region in REGIONS]
            ax.bar(np.arange(len(REGIONS)) + (i - (len(methods) - 1) / 2) * width, values, width * 0.9,
                   color=METHOD_COLORS[method], label=method)
        ax.axhline(0, color="#52514e", linewidth=1)
        ax.set_xticks(range(len(REGIONS)), REGIONS)
        ax.set_title(title, loc="left", fontsize=10)
        _clean_axes(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def _log2_axis(axis):
    axis.set_major_locator(LogLocator(base=2))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def plot_exposure(scores, methods, path):
    panels = [("exposure_ratio", "Exposure: prediction / ground truth (linear)"), ("PSNR", "PSNR (dB)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (column, title) in zip(axes, panels):
        for method in methods:
            rows = scores[scores["method"] == method]
            ax.scatter(rows["gain"], rows[column], s=36, color=METHOD_COLORS[method], edgecolor="white",
                       linewidth=0.8, label=method)
        ax.set_xscale("log")
        _log2_axis(ax.xaxis)
        ax.set_xlabel("Linear gain of the ground truth over the input (8 = no auto brightness)")
        ax.set_title(title, loc="left", fontsize=10)
        _clean_axes(ax)
    axes[0].set_yscale("log")
    _log2_axis(axes[0].yaxis)
    axes[0].axhline(1, color="#52514e", linewidth=1, linestyle="--")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def plot_colors(scores, methods, path, seed=0):
    panels = [("cast_deg", "Color cast: angle of the mean color (deg)", None),
              ("delta_e", "Mean color difference (CIE76 ΔE)", None),
              ("chroma_ratio", "Saturation: prediction / ground truth", 1.0),
              ("clip_recall", "Clipped ground truth also clipped (%)", None)]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 0.6 * len(methods) + 1.6), sharey=True)
    for ax, (column, title, reference) in zip(axes, panels):
        scale = 100 if column == "clip_recall" else 1
        for y, method in enumerate(methods):
            values = scores.loc[scores["method"] == method, column].dropna() * scale
            ax.scatter(values, y + rng.uniform(-0.2, 0.2, len(values)), s=28, color=METHOD_COLORS[method],
                       edgecolor="white", linewidth=0.8)
            ax.plot([values.mean()] * 2, [y - 0.32, y + 0.32], color="#1f1f1e", linewidth=2)
        if reference is not None:
            ax.axvline(reference, color="#52514e", linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_yticks(range(len(methods)), methods)
        ax.set_ylim(len(methods) - 0.5, -0.5)
        _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def run_analysis(set_dir, names, networks, out_dir, device, pred_dir=None):
    """networks: {codename: (model name, path of the weights)}. The first network is the main one."""
    os.makedirs(out_dir, exist_ok=True)
    dataset = LowLightDataset(set_dir, names)
    predictors = {"baseline": baseline_predict}
    for codename, (model_name, weights) in networks.items():
        model = load_network(model_name, weights, device)
        predictors[codename] = lambda inputs, model=model: predict_tiles(model, inputs, device)
    methods = list(predictors)

    rows = []
    inputs, gts = [], []
    preds = {method: [] for method in methods}
    for name, (gt8, dark8) in zip(names, dataset.pairs):
        inputs.append(dark8)
        gts.append(gt8)
        gt = gt8.astype(np.float32) / 255.0
        gain = image_stats(gt8, dark8)["gain"]
        for method, predict in predictors.items():
            with torch.no_grad():
                pred = predict(to_tensor(dark8)[None]).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            rows.append({"file": name, "method": method, "gain": gain, **score_image(pred, gt)})
            pred8 = np.round(pred * 255).astype(np.uint8)
            preds[method].append(pred8)
            if pred_dir is not None and method != "baseline":
                os.makedirs(f"{pred_dir}/{method}", exist_ok=True)
                imageio.imwrite(f"{pred_dir}/{method}/{name}.png", pred8)
        print(f"{name}: " + ", ".join(f"{r['method']} {r['PSNR']:.2f}" for r in rows[-len(methods):]), flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(f"{out_dir}/scores.csv", index=False)
    summary = scores.drop(columns=["file"]).groupby("method", sort=False).mean()
    summary.to_csv(f"{out_dir}/summary.csv")

    plot_regions(scores, methods, f"{out_dir}/regions.png")
    plot_exposure(scores, methods, f"{out_dir}/exposure.png")
    plot_colors(scores, methods, f"{out_dir}/colors.png")

    # Crops: the two images with the most clipped ground truth, and the two with the largest color error of the
    # main network. A crop shows the window with the most clipped pixels, or with the largest color error.
    main = methods[1]
    main_rows = scores[scores["method"] == main].reset_index(drop=True)
    clipped_order = main_rows["frac_clipped"].sort_values(ascending=False).index[:2]
    color_order = main_rows.drop(index=clipped_order)["delta_e"].sort_values(ascending=False).index[:2]
    cases = []
    for index in clipped_order:
        mask = region_masks(gts[index] / 255.0)["clipped"]
        cases.append((f"most clipped: {names[index]}", index, mask))
    for index in color_order:
        error = np.linalg.norm(rgb2lab(preds[main][index]) - rgb2lab(gts[index]), axis=2)
        cases.append((f"largest ΔE of {main}: {names[index]}", index, error))
    plot_crops(cases, inputs, gts, preds, methods, f"{out_dir}/crops.png")
    return summary
