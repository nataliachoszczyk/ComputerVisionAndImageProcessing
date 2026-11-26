"""Draw the figure of the test examples of gan-v2a again, with the titles of the columns and a zoom of each sphere."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro3"
PROJECT = "03-rendering"
CLONE = "/tmp/repo"
CODENAME = "gan-v2a"
OUT = Path("/kaggle/working/figures-v1")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips", "flip-evaluator"], check=True)
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from data import load_split
from training import plot_examples, to_numpy

inputs = Path("/kaggle/input")
run_dir = next(p.parent for p in inputs.rglob("predictions.npz") if p.parent.name == CODENAME)
predictions = np.load(run_dir / "predictions.npz")
scores = pd.read_csv(run_dir / "scores.csv")
params, images = load_split(next(inputs.rglob("data-v1-set.zip")), "test")
assert (params["index"].to_numpy() == predictions["index"]).all() and (scores["index"] == params["index"]).all()

OUT.mkdir(parents=True, exist_ok=True)
fig = plot_examples(predictions["preds"] / 255.0, to_numpy(images), scores)
fig.savefig(OUT / f"examples-{CODENAME}.png", dpi=110, facecolor="white")
print("done", flush=True)
