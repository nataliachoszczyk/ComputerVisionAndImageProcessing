"""Download the first 100 NEF files of RAISE and make the zip of the ground truth and input images."""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro2"
PROJECT = "02-low-light-enhancement"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
WORK = "/kaggle/temp/raise100"  # Only the zip goes to the output.

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from data import download_raise_nef, generate_dataset

download_raise_nef(f"{CLONE}/{PROJECT}/RAISE_258.csv", f"{WORK}/nef", limit=100)
generate_dataset(f"{WORK}/nef", f"{WORK}/set/gt", f"{WORK}/set/input")

print("nef", len(os.listdir(f"{WORK}/nef")))
print("gt", len(os.listdir(f"{WORK}/set/gt")))
print("input", len(os.listdir(f"{WORK}/set/input")))

shutil.make_archive("/kaggle/working/raise100-set", "zip", f"{WORK}/set")
print("zip MB", os.path.getsize("/kaggle/working/raise100-set.zip") / 1e6)
