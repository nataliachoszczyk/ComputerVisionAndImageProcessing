# Work log: project 2

## Jobs

| Date | Job | Link | Comment | Status |
|---|---|---|---|---|
| 2026-11-05 | data-v1 | [sigk-p2-data-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p2-data-v1) | Download 100 NEF files and make `raise100-set.zip` (CPU). | running |

## Log

### 2026-11-05

- Add the root `requirements.txt` and make the `.venv`.
- Change `KAGGLE.md` and the local rules for this repository and for a simple structure.
- Test the Kaggle connection with a private test kernel, `sigk-p2-smoke`.
- Results of the test:
  - The kernel has 2 Tesla T4 GPUs, Python 3.13 and internet access.
  - The kernel can clone the repository.
  - The kernel can download a NEF file from the RAISE server.
  - The kernel can decode the NEF file with rawpy.
- Add `data.py` with the data preparation script from the task text.
- Do a local test of `data.py` on one NEF file. The ground truth image and the input image have the same size.
- Add the job `data-v1`.
