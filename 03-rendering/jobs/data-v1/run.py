"""Render the data set of 3000 Phong images with the renderer of the task."""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro3"
PROJECT = "03-rendering"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
IMAGES = Path("/kaggle/temp/data-v1")
OUT = Path("/kaggle/working/data-v1")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "moderngl==5.12.0", "moderngl-window==3.1.1",
                "pyrr==0.10.3", "PyWavefront==1.3.3"], check=True)
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from data import N_IMAGES, save_preview

# Kaggle has no display. EGL gives an OpenGL context without a display.
subprocess.run([sys.executable, f"{CLONE}/{PROJECT}/renderer/src/main.py", "--window", "headless", "--backend", "egl",
                "--n_images", str(N_IMAGES), "--seed", "0", "--output_path", str(IMAGES)], check=True)

OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(IMAGES / "params.csv", OUT / "params.csv")
save_preview(IMAGES, OUT / "preview.png")
shutil.make_archive("/kaggle/working/data-v1-set", "zip", IMAGES)
print("images", len(list(IMAGES.glob("image_*.png"))))
