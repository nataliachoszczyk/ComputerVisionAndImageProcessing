# Experiments on Kaggle

This document describes a method to run experiments with models on Kaggle from a local repository. It is a collection of rules and lessons from real projects. A project can use only the parts that it needs, and change them for its own work.

The local machine has the code and the records. Kaggle does the heavy computation. Git connects the two.

## Values for this repository

| Item | Value |
|---|---|
| `<user>` | `nataliachoszczyk` |
| `<owner>/<repository>` | `nataliachoszczyk/ComputerVisionAndImageProcessing` (public) |
| `<project branch>` | `pro2` for project 2 |
| `<project folder>` | `02-low-light-enhancement` for project 2 |
| `<project>` in kernel names | `sigk-p2` for project 2 |

## 1. Local setup

The local machine needs these items:

| Item | Where | What to do |
|---|---|---|
| Kaggle account | kaggle.com | Do the phone verification (Settings). Without it, a kernel gets no GPU and no internet. |
| Python environment | `.venv` in the root folder of the repository | Make it with uv (see the README or the instructions of the repository). |
| Kaggle CLI | the package `kaggle` in `requirements.txt` | Install it into `.venv`. Version 2.2.5 works, and it also needs the package `six`. The command is `.venv/Scripts/kaggle` on Windows and `.venv/bin/kaggle` on Linux. |
| API token | `~/.kaggle/kaggle.json` | On kaggle.com, open Settings, API, and generate a new token. Keep the downloaded file in this location. |
| Access to the data | the private Kaggle dataset of the project | The owner of the dataset shares it with the account. |
| Git remote | a public repository with the project branch | The kernels clone the code from it (section 4). |

Do these steps:

1. Install the environment:

   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv -r requirements.txt --index-strategy unsafe-best-match
   ```

2. Put the token file `kaggle.json` into the folder `~/.kaggle/`.

3. Do a test of the connection. The command shows the kernels of the account:

   ```bash
   .venv/Scripts/kaggle kernels list --mine
   ```

Do not commit the token. It gives full access to the account.

## 2. Platform facts

| Item | Value |
|---|---|
| GPU machine | `"machine_shape": "NvidiaTeslaT4"` gives 2x Tesla T4, 15 GB of memory each. |
| GPU sessions at the same time | 2 for each account. A third push fails with "Maximum batch GPU session count of 2 reached". |
| GPU time | A weekly quota (approximately 30 hours). A session counts its hours, not the number of its GPUs. |
| CPU sessions | 4 cores and approximately 30 GB of RAM. They do not use the GPU quota. |
| Session limit | 12 hours. |
| Output folder | `/kaggle/working/`, approximately 20 GB. Only this folder is kept after the session. |
| Temporary folder | `/kaggle/temp/`. It is deleted after the session. |
| Input data | `/kaggle/input/<slug>/...`, read-only. |
| Python | 3.13 on Kaggle and 3.12 in the local `.venv`. Do not use features of one version only. |
| Account | GPU and internet need a phone verification of the account. Without it, a GPU kernel runs on the CPU and no error occurs. |

Important behavior:

- If a kernel fails (status ERROR), Kaggle deletes its complete output. Hours of results can be lost.
- The CLI cannot stop a kernel. Only the web page can cancel it.
- The CLI does not give the version ID of a run. Thus use one kernel (notebook) for each job, not many versions of one kernel.
- `kaggle kernels logs -f <user>/<project>-<job name>` shows the live log. The output of child processes is sometimes not in it.
- Use `kaggle kernels status <user>/<project>-<job name>` for the state: QUEUED, RUNNING, COMPLETE, ERROR or CANCEL.

## 3. Repository structure

The structure is similar to project 1. Keep it as simple as possible.

```
<project folder>/
  data.py, model.py, ...  all logic (Python modules)
  experiments.ipynb       shows the results
  jobs/
    <job name>/           one folder for each Kaggle kernel
      run.py              the script that Kaggle runs
      kernel-metadata.json
  results/
    <codename>/           small results: metrics, logs of the training, images
  WORKLOG.md              the list of all jobs and the work log
  SUMMARY.md              the report
```

### Names

| Name | Where it occurs |
|---|---|
| Job name, for example `unet-v1ab` | The folder in `jobs/` and the kernel id (`<user>/<project>-unet-v1ab`). |
| Codename, for example `unet-v1a` | The argument of the experiment function and the folder in `results/`. |

- Each name is unique in the project.
- A job has one or more codenames.
- Variants of one idea get letters: `unet-v1a`, `unet-v1b`.
- If a job has only one experiment, its codename is the job name.
- Use only lowercase letters, digits and hyphens. Kaggle makes the kernel slug from the title, and other characters change.

### Rules

- A job is one Kaggle kernel. A job can run more than one experiment, for example one on each GPU.
- `run.py` holds the settings of its experiments. There are no separate config files.
- The folder of a job does not change after the start. A new variant gets a new folder.
- `results/` keeps only small files. Data, model weights and caches stay on Kaggle.

## 4. Code and data on Kaggle

### Code

The kernel clones the project branch from a public repository. Thus every run uses code from git, and the commit hash goes into the log. Push the code before the push of the kernel.

If the repository has more than one project, clone only the project folder with a sparse checkout. The templates below do this. If the project is the complete repository, clone the complete repository and remove the `sparse-checkout` line:

```python
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, CLONE], check=True)
sys.path.insert(0, CLONE)
```

Clone into `/tmp`, not into `/kaggle/working`. Otherwise the clone becomes a part of the output.

### Data

- Keep the data in a private Kaggle dataset. Upload the original zip as one file, not thousands of small files. Share the dataset with the team.
- The output of one kernel can be the input of another kernel: `"kernel_sources": ["<user>/<project>-<job name>"]`. Thus a data set or model weights from one job go to the next job without a download.
- The path under `/kaggle/input` can change. Find files with a recursive search, for example `Path("/kaggle/input").rglob("val.npy")`.
- To make a dataset from a kernel output, use the page of the kernel: Output, New Dataset. Kaggle extracts a zip to a folder with the name of the zip. If the output also has a folder with this name, the creation fails ("Directory already exists"). Thus give a zip a name like `<codename>-set.zip`.

### `kernel-metadata.json`

```json
{
  "id": "<user>/<project>-<job name>",
  "title": "<project>-<job name>",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["<user>/<dataset>"],
  "competition_sources": [],
  "kernel_sources": []
}
```

For a CPU job, use `"enable_gpu": false` and remove `machine_shape`.

### `run.py` for one experiment on one GPU

```python
"""<What the job does, in one or two sentences.>"""
import subprocess
import sys

REPO = "https://github.com/<owner>/<repository>.git"
BRANCH = "<project branch>"
PROJECT = "<project folder>"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAMES = ["<codename>"]

import torch

# A long job on the CPU wastes hours, so stop immediately without a GPU.
assert torch.cuda.is_available(), "no GPU: check the phone verification of the account"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from training import run_experiment

for codename in CODENAMES:
    run_experiment(codename)
```

## 5. Use of the two GPUs

A T4 x2 session has two GPUs, and the quota counts the session. A job that uses one GPU wastes half of its cost.

One process on each GPU uses both GPUs. Each process does its own work: one experiment, or one half of the data (shard).

### `run.py` for two experiments, one on each GPU

```python
"""Run two experiments at the same time, one on each GPU."""
import subprocess
import sys

REPO = "https://github.com/<owner>/<repository>.git"
BRANCH = "<project branch>"
PROJECT = "<project folder>"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAMES = ["<codename a>", "<codename b>"]


def run(codename: str, gpu: int) -> None:
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from training import run_experiment
    run_experiment(codename, f"cuda:{gpu}")


# A new process imports this file again, so the clone and the start must be only in the main process.
if __name__ == "__main__":
    import torch

    assert torch.cuda.device_count() == 2, "this job needs 2 GPUs (T4 x2)"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    with torch.multiprocessing.get_context("spawn").Pool(2) as pool:
        pool.starmap(run, [(codename, gpu) for gpu, codename in enumerate(CODENAMES)])
```

Rules for this pattern:

- Use the `spawn` context. CUDA does not work in a process from `fork`.
- Put the main code under `if __name__ == "__main__":`. Each new process imports `run.py` again.
- The experiment function gets the device as an argument, and it calls `torch.cuda.set_device(device)`.
- Each process writes its results into its own folder, `/kaggle/working/<codename>/`.
- The main process must not keep GPU memory. If it used the GPU before the start of the pool, call `torch.cuda.empty_cache()`. In one job, the main process kept 12 GB of cache on GPU 0, and the process for GPU 0 failed with "CUDA out of memory".
- If one process fails, the complete job fails, and Kaggle deletes the output of both experiments. Thus do a smoke test first (section 7).
- Two jobs with two experiments each give four experiments at the same time.
- Two processes share the RAM of the session. Each process must stay below approximately half of it.

For a CPU job, use `multiprocessing.Pool(4)` for the 4 cores.

## 6. Start and follow a job

1. Commit and push the code.
2. Add a row to the table of jobs in `WORKLOG.md` (section 9).
3. Push the kernel from the project folder:

   ```bash
   kaggle kernels push -p jobs/<job name>
   ```

Output if the push is correct:

```
Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/<user>/<project>-<job name>
```

The first push makes the kernel. A second push of the same folder makes version 2 of the same kernel.

If two GPU sessions are active, the push fails. Try the push again when a session stops.

Show the state of a job with this command:

```bash
kaggle kernels status <user>/<project>-<job name>
```

When a job stops, write its final status and its main results into `WORKLOG.md`.

## 7. Protect long runs

- **Smoke test.** Before a run of many hours, run the same code path on a small part of the data. Use a CPU kernel, or the local machine if it has enough memory. The test must do all steps, also the final steps (selection, reports, final files). An error at the end of a long run loses all its results.
- **Early checks.** Put cheap checks at the start. If a check fails, stop the job. Example: after a small part of a data generation, compare a statistic of the new data with the expected value. If it is wrong, the job stops before the long part.
- **Deadline.** A long loop gets a deadline (for example 9 hours). After it, no new batch starts, and the job writes its results before the 12-hour limit.
- **No heavy work on the local machine.** The local machine does only small checks. It can stop a process if the memory is low.

## 8. Outputs

1. List the files and their sizes: `kaggle kernels files <user>/<project>-<job name>`.
2. Download only the small files into `results/`:

   ```bash
   kaggle kernels output <user>/<project>-<job name> -p results --file-pattern '(json|csv|png)$'
   ```

   The pattern is a regular expression for the complete path. Do not start it with `^`.
3. Delete the downloaded `.log` file if it is not necessary.
4. Commit the small results.

Write large intermediate data only to `/kaggle/temp/` or to `/kaggle/working/cache/`. Write small results as CSV or JSON.

## 9. Records

Keep one table of jobs at the start of `WORKLOG.md`:

```markdown
| Date | Job | Link | Comment | Status |
|---|---|---|---|---|
| YYYY-MM-DD | <job name> | [<project>-<job name>](https://www.kaggle.com/code/<user>/<project>-<job name>) | <what the job does; the main results after the end> | complete |
```

- Status values: to run, queued, running, complete, failed, cancelled.
- One job has one row. A new version of the job uses the same row and gets a note in the comment.
- Each row has the job name and the codenames. Then the settings and the results of a row are easy to find.

## 10. Experiment method

- **Codenames.** Each experiment has a codename with a version, for example `unet-v7a`. Variants of one idea get letters (section 3).
- **One change for each variant.** A group of variants has a common base, and each variant changes one thing. Then the difference shows the effect of this change.
- **Split.** Keep a fixed validation split in a file. Keep a small tune part of the train data for thresholds and for the selection of the checkpoint. Use the validation set only for decisions.
- **Checkpoints.** Train longer and keep checkpoints (for example after each 5 epochs, in CPU memory). At the end, score each checkpoint with the real metric on the tune part, and keep the best. The loss does not always follow the metric.
- **Noise.** Two runs with the same settings and a different seed show the random difference. A difference smaller than this is not a result. Keep a separate `train_seed` if the main seed also selects the data split.
