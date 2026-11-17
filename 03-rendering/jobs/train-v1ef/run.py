"""Train the main model with the GAN weight 1 and with a second seed, one on each GPU."""
import subprocess
import sys
import traceback
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/ComputerVisionAndImageProcessing.git"
BRANCH = "pro3"
PROJECT = "03-rendering"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
OUT = "/kaggle/working"
VARIANTS = {
    "gan-v1e": dict(adversarial=1.0),  # the GAN weight 1 instead of 5
    "gan-v1f": dict(seed=1),  # gan-v1a with a different seed: the random difference between two runs
}
PLAN = [["gan-v1e"], ["gan-v1f"]]
SETTINGS = dict(steps=40000, eval_every=2000, deadline_hours=1.6)


def find_data():
    found = [*Path("/kaggle/input").rglob("data-v1-set.zip"), *Path("/kaggle/input").rglob("params.csv")]
    print("data", found, flush=True)
    return found[0] if found[0].suffix == ".zip" else found[0].parent


def run(codenames, gpu):
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from training import run_experiment

    for codename in codenames:
        # An error in one experiment must not stop the job: Kaggle deletes all output of a failed job.
        try:
            run_experiment(codename, find_data(), OUT, f"cuda:{gpu}", **VARIANTS[codename], **SETTINGS)
        except Exception:
            print(f"{codename} failed", flush=True)
            traceback.print_exc()
            Path(f"{OUT}/{codename}").mkdir(exist_ok=True)
            Path(f"{OUT}/{codename}/error.txt").write_text(traceback.format_exc())


# A new process imports this file again, so the clone and the start must be only in the main process.
if __name__ == "__main__":
    import torch

    assert torch.cuda.device_count() >= len(PLAN), "no GPU: check the machine of the job"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips", "flip-evaluator"], check=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    # Not a Pool: a Pool waits without end if the system stops one of its processes.
    context = torch.multiprocessing.get_context("spawn")
    processes = [context.Process(target=run, args=(codenames, gpu)) for gpu, codenames in enumerate(PLAN)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    print("exit codes", [process.exitcode for process in processes])
