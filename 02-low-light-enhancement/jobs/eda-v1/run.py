"""Compute the statistics and the figures of the 100 image pairs."""
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro2"
PROJECT = "02-low-light-enhancement"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rawpy", "imageio"], check=True)  # The Kaggle image has no rawpy.
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from data import run_eda

set_dir = next(Path("/kaggle/input").rglob("gt")).parent
print("set", set_dir)
stats = run_eda(f"{CLONE}/{PROJECT}/RAISE_258.csv", set_dir, "/kaggle/working/eda-v1")
print(stats.groupby("split").describe().T.to_string())
