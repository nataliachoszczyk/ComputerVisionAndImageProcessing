# Work log: project 3

## Jobs

| Date | Job | Link | Comment | Status |
|---|---|---|---|---|
| 2026-11-17 | gl-smoke | [sigk-p3-gl-smoke](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-gl-smoke) | Test OpenGL without a display (CPU). Result: the default backend fails (no X display). The `egl` backend works with Mesa llvmpipe, without apt. 3000 frames of 128x128 in 0.78 s. The headless window of `moderngl_window` also works. | complete |
| 2026-11-17 | data-v1 | [sigk-p3-data-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-data-v1) | Render 3000 images of 128x128 with seed 0 (CPU, EGL). Result: `data-v1-set.zip` with the images and `params.csv`, 2400 train and 600 test images. Rendering takes 10 s, the full job 24 s. | complete |

## Log

### 2026-11-17

- Make the branch `pro3` and the folder `03-rendering`.
- Move the renderer code of the task into `renderer/`.
- Add the packages of the renderer and the FLIP metric to the root `requirements.txt`.
- Add the job `gl-smoke`. Remove its code after the test.
- Add `data.py` with the scene sampling:
  - The full sphere must be in the camera view. Thus the sphere is not always at the center of the image.
  - The light must be at a distance of 2 radii or more from the sphere center. Else the light can be in the sphere.
  - The last 600 images are the test split.
- Change the renderer for the data generation:
  - `PhongWindow` samples a new scene for each frame, and writes `params.csv`.
  - The window closes after `--n_images` frames.
  - `main.py` uses absolute paths and accepts arguments from the command line.
  - The image size is 128x128.
  - The ambient colors in `phong.frag` are the values of the task text (76/255 and 25/255).
- Do a local test with 3000 images (10 s). The sphere center agrees with the parameters. No sphere touches the image border.
- Add the job `data-v1`.
- Add the results of `data-v1` in `results/data-v1/`: the parameters and a preview of 32 images.
- Add `metrics.py` with FLIP, LPIPS, SSIM and the Hausdorff distance between Canny edges.
  - The Canny thresholds are 50 and 150. With these thresholds, all 600 test images have edges.
  - If only one image has edges, the Hausdorff distance is the image diagonal (181 px).
- Do a local test of the metrics on 100 test images (mean values):

  | Change | FLIP | LPIPS | SSIM | Hausdorff |
  |---|---|---|---|---|
  | same image | 0.0000 | 0.0000 | 1.0000 | 0.00 |
  | shift 2 px | 0.0055 | 0.0164 | 0.9894 | 2.00 |
  | shift 10 px | 0.0122 | 0.0772 | 0.9730 | 14.04 |
  | swap R and B | 0.0030 | 0.0203 | 0.9971 | 0.97 |
  | 50% darker | 0.0060 | 0.0297 | 0.9944 | 1.63 |
  | black image | 0.0073 | 0.1960 | 0.9822 | 181.02 |

- A black image gets a better FLIP than a shift of 10 px, and an SSIM of 0.98. The cause: most pixels of the images are black.
- For the report: use this test to answer the question of the task "Do all metrics show the quality well?". FLIP and SSIM give a mean over all pixels, and most pixels are background.
- Decisions: the model is a GAN. The results table has no baseline row.
