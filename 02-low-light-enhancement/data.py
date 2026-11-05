import csv
import glob
import os

import imageio
import numpy as np
import rawpy
import requests


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
