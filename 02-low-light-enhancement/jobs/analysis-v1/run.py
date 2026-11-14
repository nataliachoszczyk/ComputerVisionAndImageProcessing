"""Analyze the colors and the clipped highlights of the baseline, train-v2a and train-v2b on the test set."""
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro2"
PROJECT = "02-low-light-enhancement"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
OUT = "/kaggle/working"
# The first network is the main one. The weights come from the output of the job train-v2ab.
NETWORKS = {"train-v2a": "unet-global", "train-v2b": "nafnet-global"}

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rawpy", "imageio", "lpips"], check=True)  # The Kaggle image does not have them.
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

import pandas as pd
import torch

from analysis import run_analysis
from data import split_files

assert torch.cuda.is_available(), "no GPU: check the machine of the job"
weights = {}
for codename in NETWORKS:
    # Accept the output of a kernel (<codename>/best.pt) and a dataset (<codename>.pt).
    found = [p for p in Path("/kaggle/input").rglob("*.pt") if codename in str(p)]
    assert len(found) == 1, f"weights of {codename}: {found}"
    weights[codename] = (NETWORKS[codename], found[0])
print("weights", weights)

set_dir = next(Path("/kaggle/input").rglob("gt")).parent
names = split_files(f"{CLONE}/{PROJECT}/RAISE_258.csv", "test")
summary = run_analysis(set_dir, names, weights, f"{OUT}/analysis-v1", "cuda:0", pred_dir=f"{OUT}/preds")
print(summary.T.to_string())

# The PSNR must be the same as in the training job, otherwise the weights are wrong.
for codename in NETWORKS:
    expected = pd.read_csv(f"{CLONE}/{PROJECT}/results/{codename}/summary.csv")["PSNR"].iloc[0]
    print(f"{codename}: PSNR {summary.loc[codename, 'PSNR']:.3f}, training job {expected:.3f}")
