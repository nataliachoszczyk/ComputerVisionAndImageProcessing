"""Run the renderer with the generator for the 3000 scenes of the data set (seed 0) and score the test scenes."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro3"
PROJECT = "03-rendering"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "gan-v2a"
PAIRS = Path("/kaggle/temp/render-v1")
OUT = Path("/kaggle/working/render-v1")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "moderngl==5.12.0", "moderngl-window==3.1.1",
                "pyrr==0.10.3", "PyWavefront==1.3.3", "lpips", "flip-evaluator"], check=True)
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from data import N_IMAGES, N_TEST
from metrics import METRICS, change_test, evaluate
from training import crop

inputs = Path("/kaggle/input")
generator = next(p for p in inputs.rglob("generator.pt") if p.parent.name == CODENAME)
data_zip = zipfile.ZipFile(next(inputs.rglob("data-v1-set.zip")))
predictions = np.load(next(p for p in inputs.rglob("predictions.npz") if p.parent.name == CODENAME))
print("generator", generator, flush=True)

# Kaggle has no display. EGL gives an OpenGL context without a display.
log = subprocess.run([sys.executable, f"{CLONE}/{PROJECT}/renderer/src/main.py", "--window", "headless", "--backend", "egl",
                      "--n_images", str(N_IMAGES), "--seed", "0", "--generator", str(generator),
                      "--output_path", str(PAIRS)], check=True, capture_output=True, text=True).stdout
print(log, flush=True)

OUT.mkdir(parents=True, exist_ok=True)
phong, network, same = [], [], 0
for i in range(N_IMAGES):
    pair = np.asarray(Image.open(PAIRS / f"compare_{i:04}.png"))
    reference = np.asarray(Image.open(data_zip.open(f"image_{i:04}.png")).convert("RGB"))
    same += np.array_equal(pair[:, :128], reference)
    if i >= N_IMAGES - N_TEST:
        phong.append(pair[:, :128] / 255.0)
        network.append(pair[:, 128:] / 255.0)
print(f"Phong images equal to the data set: {same} of {N_IMAGES}", flush=True)

change_test(phong).to_csv(OUT / "change-test.csv")
scores = evaluate(network, phong)
scores.insert(0, "index", np.arange(N_IMAGES - N_TEST, N_IMAGES))
scores.to_csv(OUT / "scores.csv", index=False)
# The same metrics on a 64x64 crop around the sphere. Most pixels of a full image are background.
centers = [np.argwhere(p.max(2) > 0).mean(0) for p in phong]
crop_scores = evaluate([crop(n, c, 64) for n, c in zip(network, centers)], [crop(p, c, 64) for p, c in zip(phong, centers)])
crop_scores.insert(0, "index", scores["index"])
crop_scores.to_csv(OUT / "scores-crop.csv", index=False)
# The renderer runs the generator on the CPU, the training on the GPU. The difference must be small.
difference = np.abs(np.stack(network) * 255 - predictions["preds"]).max()
summary = {"codename": CODENAME, **scores[METRICS].mean().round(5).to_dict(),
           **{f"{k} (crop)": v for k, v in crop_scores[METRICS].mean().round(5).to_dict().items()},
           "phong_equal_to_data_set": int(same),
           "max_difference_to_training_predictions": float(difference),
           "ms_per_image_cpu": float(log.split("network: ")[1].split(" ms")[0])}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(summary, flush=True)

# A preview: 8 test pairs as full images and as crops around the sphere.
rows = []
for i in range(N_IMAGES - N_TEST, N_IMAGES - N_TEST + 8):
    pair = np.asarray(Image.open(PAIRS / f"compare_{i:04}.png"))
    center = np.argwhere(pair[:, :128].max(2) > 0).mean(0)
    zoom = lambda img: np.kron(crop(img, center, 32), np.ones((4, 4, 1), dtype=np.uint8))
    rows.append(np.hstack([pair, zoom(pair[:, :128]), zoom(pair[:, 128:])]))
Image.fromarray(np.vstack(rows)).save(OUT / "preview.png")
