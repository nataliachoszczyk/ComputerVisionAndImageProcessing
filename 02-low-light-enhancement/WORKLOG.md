# Work log: project 2

## Jobs

| Date | Job | Link | Comment | Status |
|---|---|---|---|---|
| 2026-11-05 | data-v1 | [sigk-p2-data-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p2-data-v1) | Stop after 4 s: the Kaggle image has no rawpy. | failed |
| 2026-11-05 | data-v2 | [sigk-p2-data-v2](https://www.kaggle.com/code/nataliachoszczyk/sigk-p2-data-v2) | Same as data-v1, and install rawpy. Result: 100 NEF files, 100 ground truth images, 100 input images, `raise100-set.zip` (1585 MB, CPU, 17 min). | complete |
| 2026-11-05 | eda-v1 | [sigk-p2-eda-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p2-eda-v1) | Compute the statistics and the figures of the 100 image pairs (CPU). Result: median linear gain is 7.5 (train), 21.4 (val), 8.5 (test). | complete |
| 2026-11-05 | baseline-v1 | [sigk-p2-baseline-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p2-baseline-v1) | Codenames `eda-v2`, `baseline-v1`. Repeat the analysis with the final split. Score histogram equalization on the test set (CPU). Result: PSNR 11.32, SSIM 0.152, LPIPS 1.077. | complete |

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
- Add the job `data-v2`, because `data-v1` failed without rawpy.
- Make the private dataset `sigk-p2-raise100` from the output of `data-v2`, and share it with the team.
- Add the split, the statistics and the figures to `data.py`. Add the job `eda-v1`.
- Add the results of `eda-v1` and `experiments.ipynb`.
- Change the val split to the images 43-53, because the median gain of the old val split is 21.4. The results of `eda-v1` use the old split.
- Add `LowLightDataset` to `data.py`.
- Add `metrics.py` with PSNR, SSIM, LPIPS and the histogram equalization baseline.
- Add the job `baseline-v1`.
- Add the results of `eda-v2` and `baseline-v1` to `experiments.ipynb`.
