import csv
import glob
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rawpy
import requests

SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}


def download_raise_nef(csv_path, output_dir, limit=None):
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, mode="r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):
            if limit is not None and i >= limit:
                break
            nef_url = row["NEF"]
            filename = row["File"]
            if not filename.upper().endswith(".NEF"):
                filename += ".NEF"
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f"Skipping {filename} - already exists.")
                continue
            print(f"Downloading {filename}...")
            response = requests.get(nef_url, stream=True)
            with open(filepath, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)


def get_ground_truth(raw):
    return raw.postprocess(use_camera_wb=True, half_size=True)


def add_physical_noise(linear_img, alpha=0.001, sigma=0.005):
    noisy_img = np.random.poisson(linear_img / alpha) * alpha
    noisy_img += np.random.normal(0, sigma, linear_img.shape)
    return np.clip(noisy_img, 0.0, 1.0)


def get_linear_image(raw):
    linear_16bit = raw.postprocess(
        use_camera_wb=True,
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
        half_size=True,
    )
    return linear_16bit.astype(np.float32) / 65535.0


def reduce_exposure(linear_img, ev_shift):
    return linear_img * 2.0 ** ev_shift


def apply_gamma_correction(img, gamma=2.2):
    return img ** (1 / gamma)


def convert_to_8bit(img):
    return np.round(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def get_low_light(raw, ev_shift=-3.0):
    linear_img = get_linear_image(raw)
    dark_linear = reduce_exposure(linear_img, ev_shift)
    noisy_dark_linear = add_physical_noise(dark_linear)
    dark_srgb = apply_gamma_correction(noisy_dark_linear)
    return convert_to_8bit(dark_srgb)


def process_single_file(file_path, output_gt_dir, output_dark_dir, ev_shift):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    gt_path = os.path.join(output_gt_dir, f"{base_name}_gt.png")
    dark_path = os.path.join(output_dark_dir, f"{base_name}_input.png")

    if os.path.exists(gt_path) and os.path.exists(dark_path):
        print(f"Skipping {base_name} - already processed.")
        return

    print(f"Processing {base_name}...")
    try:
        with rawpy.imread(file_path) as raw:
            img_gt = get_ground_truth(raw)
            img_dark = get_low_light(raw, ev_shift)
        imageio.imsave(gt_path, img_gt)
        imageio.imsave(dark_path, img_dark)
    except Exception as e:
        print(f"Error processing {base_name}: {e}")


def generate_dataset(nef_dir, output_gt_dir, output_dark_dir, ev_shift=-3.0):
    os.makedirs(output_gt_dir, exist_ok=True)
    os.makedirs(output_dark_dir, exist_ok=True)
    for file_path in sorted(glob.glob(os.path.join(nef_dir, "*.NEF"))):
        process_single_file(file_path, output_gt_dir, output_dark_dir, ev_shift)


def split_name(index):
    # The photos of each split come from different sessions, so similar photos stay in one split.
    return "train" if index < 70 else "val" if index < 80 else "test"


def read_file_names(csv_path, limit=100):
    with open(csv_path, mode="r", encoding="utf-8") as file:
        return [row["File"] for _, row in zip(range(limit), csv.DictReader(file))]


def read_pair(set_dir, name):
    return imageio.imread(f"{set_dir}/gt/{name}_gt.png"), imageio.imread(f"{set_dir}/input/{name}_input.png")


def image_stats(gt, dark, gamma=2.2):
    # Compare the images in linear light: without auto brightness in the ground truth, the gain is 8.
    gt_linear = (gt / 255.0) ** gamma
    dark_linear = (dark / 255.0) ** gamma
    gains = gt_linear.mean(axis=(0, 1)) / dark_linear.mean(axis=(0, 1))
    return {
        "height": gt.shape[0],
        "width": gt.shape[1],
        "gt_mean": gt.mean(),
        "input_mean": dark.mean(),
        "gain": gt_linear.mean() / dark_linear.mean(),
        "gain_r": gains[0],
        "gain_g": gains[1],
        "gain_b": gains[2],
        "gt_clipped": (gt >= 250).mean(),
        "input_black": (dark == 0).mean(),
    }


def collect_stats(csv_path, set_dir, limit=100):
    rows = []
    histograms = {}
    for index, name in enumerate(read_file_names(csv_path, limit)):
        gt, dark = read_pair(set_dir, name)
        split = split_name(index)
        rows.append({"file": name, "split": split, **image_stats(gt, dark)})
        split_hist = histograms.setdefault(split, {"gt": np.zeros(256), "input": np.zeros(256)})
        split_hist["gt"] += np.bincount(gt.ravel(), minlength=256)
        split_hist["input"] += np.bincount(dark.ravel(), minlength=256)
    return pd.DataFrame(rows), histograms


def _clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_histograms(histograms, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kind, title in zip(axes, ["gt", "input"], ["Ground truth", "Input"]):
        for split, hist in histograms.items():
            ax.plot(hist[kind] / hist[kind].sum(), color=SPLIT_COLORS[split], linewidth=2, label=split)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Pixel value (8-bit)")
        ax.set_xlim(0, 255)
        _clean_axes(ax)
    axes[0].set_ylabel("Fraction of values (log scale)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def plot_per_image(stats, path, seed=0):
    panels = [
        ("gain", "Linear gain: ground truth / input", 1),
        ("gt_clipped", "Ground truth values >= 250 (%)", 100),
        ("input_black", "Input values = 0 (%)", 100),
    ]
    splits = [s for s in SPLIT_COLORS if s in set(stats["split"])]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, (column, title, scale) in zip(axes, panels):
        for y, split in enumerate(splits):
            values = stats.loc[stats["split"] == split, column] * scale
            jitter = rng.uniform(-0.2, 0.2, len(values))
            ax.scatter(values, y + jitter, s=32, color=SPLIT_COLORS[split], edgecolor="white", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_yticks(range(len(splits)), splits)
        ax.set_ylim(len(splits) - 0.5, -0.5)
        _clean_axes(ax)
    axes[0].axvline(8, color="#52514e", linewidth=1, linestyle="--")
    axes[0].annotate("8 = no auto brightness", (8, 1), xycoords=("data", "axes fraction"),
                     xytext=(4, -2), textcoords="offset points", fontsize=8, color="#52514e", va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def plot_samples(stats, set_dir, path, stride=6):
    cases = [
        ("median gain", stats.iloc[(stats["gain"] - stats["gain"].median()).abs().argmin()]),
        ("highest gain", stats.loc[stats["gain"].idxmax()]),
        ("most clipped ground truth", stats.loc[stats["gt_clipped"].idxmax()]),
        ("most black input", stats.loc[stats["input_black"].idxmax()]),
    ]
    fig, axes = plt.subplots(len(cases), 2, figsize=(9, 3.1 * len(cases)))
    for row, (case, stat) in zip(axes, cases):
        gt, dark = read_pair(set_dir, stat["file"])
        row[0].imshow(dark[::stride, ::stride])
        row[1].imshow(gt[::stride, ::stride])
        row[0].set_title(f"{case}: {stat['file']}, gain {stat['gain']:.1f}\ninput", loc="left", fontsize=9)
        row[1].set_title("ground truth", loc="left", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def run_eda(csv_path, set_dir, out_dir, limit=100):
    os.makedirs(out_dir, exist_ok=True)
    stats, histograms = collect_stats(csv_path, set_dir, limit)
    stats.to_csv(f"{out_dir}/stats.csv", index=False)
    plot_histograms(histograms, f"{out_dir}/histograms.png")
    plot_per_image(stats, f"{out_dir}/per_image.png")
    plot_samples(stats, set_dir, f"{out_dir}/samples.png")
    return stats
