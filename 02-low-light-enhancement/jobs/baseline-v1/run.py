"""Score the histogram equalization baseline on the test set, and repeat the analysis with the final split."""
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro2"
PROJECT = "02-low-light-enhancement"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rawpy", "imageio", "lpips"], check=True)  # The Kaggle image does not have them.
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from data import run_eda, split_files
from metrics import run_baseline

csv_path = f"{CLONE}/{PROJECT}/RAISE_258.csv"
set_dir = next(Path("/kaggle/input").rglob("gt")).parent
print("set", set_dir)

stats = run_eda(csv_path, set_dir, "/kaggle/working/eda-v2")
print(stats.groupby("split")["gain"].describe().to_string())

summary = run_baseline(set_dir, split_files(csv_path, "test"), "/kaggle/working/baseline-v1")
print(summary.to_string())
