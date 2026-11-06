"""Train U-Net and U-Net with a global branch, one on each GPU, and score them on the test set."""
import subprocess
import sys
import traceback
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro2"
PROJECT = "02-low-light-enhancement"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
OUT = "/kaggle/working"
# One list of (codename, model) for each GPU.
PLAN = [[("train-v1a", "unet")], [("train-v1b", "unet-global")]]
SETTINGS = dict(steps=15000, val_every=1000)


def run(experiments, gpu):
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from training import run_experiment

    set_dir = next(Path("/kaggle/input").rglob("gt")).parent
    for codename, model_name in experiments:
        # An error in one experiment must not stop the job: Kaggle deletes all output of a failed job.
        try:
            run_experiment(codename, model_name, set_dir, f"{CLONE}/{PROJECT}/RAISE_258.csv", OUT, f"cuda:{gpu}", **SETTINGS)
        except Exception:
            print(f"{codename} failed", flush=True)
            traceback.print_exc()
            Path(f"{OUT}/{codename}").mkdir(exist_ok=True)
            Path(f"{OUT}/{codename}/error.txt").write_text(traceback.format_exc())


# A new process imports this file again, so the clone and the start must be only in the main process.
if __name__ == "__main__":
    import torch

    assert torch.cuda.device_count() >= len(PLAN), "no GPU: check the machine of the job"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rawpy", "imageio", "lpips"], check=True)  # The Kaggle image does not have them.
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    # Not a Pool: a Pool waits without end if the system stops one of its processes.
    context = torch.multiprocessing.get_context("spawn")
    processes = [context.Process(target=run, args=(experiments, gpu)) for gpu, experiments in enumerate(PLAN)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    print("exit codes", [process.exitcode for process in processes])
