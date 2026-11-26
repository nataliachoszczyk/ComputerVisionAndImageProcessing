# Work log: project 3

## Jobs

| Date | Job | Link | Comment | Status |
|---|---|---|---|---|
| 2026-11-17 | gl-smoke | [sigk-p3-gl-smoke](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-gl-smoke) | Test OpenGL without a display (CPU). Result: the default backend fails (no X display). The `egl` backend works with Mesa llvmpipe, without apt. 3000 frames of 128x128 in 0.78 s. The headless window of `moderngl_window` also works. | complete |
| 2026-11-17 | data-v1 | [sigk-p3-data-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-data-v1) | Render 3000 images of 128x128 with seed 0 (CPU, EGL). Result: `data-v1-set.zip` with the images and `params.csv`, 2400 train and 600 test images. Rendering takes 10 s, the full job 24 s. | complete |
| 2026-11-17 | train-v1 | [sigk-p3-train-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-train-v1) | Full training, 40000 steps: `gan-v1a` (GAN weight 5, main), `gan-v1b` (only L1), `gan-v1c` (raw parameters), `gan-v1d` (no relative coordinates). Result (600 test images, FLIP / LPIPS / SSIM / Hausdorff): `a` 0.00156 / 0.00117 / 0.99927 / 1.60; `b` 0.00131 / 0.00197 / 0.99940 / 2.09; `c` 0.00255 / 0.00393 / 0.99772 / 2.38; `d` 0.00186 / 0.00143 / 0.99905 / 1.65. | complete |
| 2026-11-17 | train-v1ef | [sigk-p3-train-v1ef](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-train-v1ef) | Full training, 40000 steps: `gan-v1e` (GAN weight 1), `gan-v1f` (`gan-v1a` with seed 1). Result (600 test images): `gan-v1e` FLIP 0.00136, LPIPS 0.00134, SSIM 0.99934, Hausdorff 1.58; `gan-v1f` FLIP 0.00131, LPIPS 0.00103, SSIM 0.99936, Hausdorff 1.50. The best checkpoint is the last step. 72 min for each run. | complete |
| 2026-11-18 | train-v2ab | [sigk-p3-train-v2ab](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-train-v2ab) | `gan-v2a`: the main model with 100000 steps. `gan-v2b`: the main model without the mirror augmentation (40000 steps). Result: `gan-v2a` 0.00115 / 0.00099 / 0.99949 / 1.50 (best checkpoint at step 80000, 168 min); `gan-v2b` 0.00204 / 0.00234 / 0.99881 / 2.32. | complete |
| 2026-11-18 | train-v3ab | [sigk-p3-train-v3ab](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-train-v3ab) | 100000 steps, as `gan-v2a`: `gan-v3a` (only L1), `gan-v3b` (`gan-v2a` with seed 1). Result: `gan-v3a` 0.00118 / 0.00156 / 0.99944 / 1.80 (best step 100000, 116 min); `gan-v3b` 0.00142 / 0.00109 / 0.99930 / 1.59 (best step 65000, 173 min). | complete |
| 2026-11-18 | render-v1 | [sigk-p3-render-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-render-v1) | The renderer with the generator of `gan-v2a` for the 3000 scenes (seed 0, CPU, EGL). Check that the Phong images are equal to `data-v1`. Score the 600 test scenes on the full image and on a 64x64 crop. Change test of the metrics. Result: all 3000 Phong images are equal to `data-v1`. The network image differs from the training predictions by 1/255 or less. Full image: FLIP 0.00074, LPIPS 0.00099, SSIM 0.99950, Hausdorff 1.50. Crop 64x64: FLIP 0.00295, LPIPS 0.00191, SSIM 0.99782, Hausdorff 1.50. The network takes 54 ms for one image on the CPU. | complete |
| 2026-11-18 | figures-v1 | [sigk-p3-figures-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-figures-v1) | Draw the test examples of `gan-v2a` again, with titles of the columns and the zoom areas (CPU). The zoom is too small for large spheres. Replaced by `figures-v2`. | complete |
| 2026-11-18 | figures-v2 | [sigk-p3-figures-v2](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-figures-v2) | As `figures-v1`, but the zoom holds the full sphere (40 px or more). Result: `results/figures-v2/examples-gan-v2a.png`. | complete |

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
- Add the input features of the network to `data.py` (11 values):
  - the position of the sphere in the image (normalized device coordinates),
  - the log of the image radius of the sphere,
  - the direction from the sphere to the light in camera coordinates,
  - the distance from the sphere to the light,
  - the diffuse color divided by 255,
  - the shininess scaled to [0, 1].
- Add `load_split` to `data.py`. It reads one split from the folder or the zip of the data set.
- Test the features locally on 3000 images. The sphere center agrees with the image to 0.9 px. The measured radius is 0.97-1.07 times the computed radius.

### 2026-11-17 (night)

- Add `raw_features` to `data.py`. It is the direct input for the comparison with hint 1 of the task.
- Add the mirror augmentation to `data.py`. A mirror in a plane through the camera axis flips the image and gives a correct new scene. The test shows that the features change as expected (the x or y values change sign).
- Change `load_split`. It returns the parameters and the uint8 images, so the training can make the inputs of each variant.
- Add `model.py`:
  - The generator starts from a 4x4 map. Each block gets the inputs as constant channels, the pixel coordinates and the coordinates relative to the sphere.
  - The generator has a linear output in the training. A sigmoid output saturates at black and stops the gradient. In the evaluation, the output is clamped to [0, 1].
  - The discriminator is a conditional PatchGAN with spectral normalization (16x16 patches).
- Add `training.py`:
  - The L1 loss gives the same weight to the sphere region and to the background. Most pixels are background.
  - The GAN loss is least squares. It uses only the patches that touch the sphere region. Background patches are equal in real and fake images, and without this limit the discriminator gives 0.5 everywhere.
  - Mixed precision (AMP), Adam with a cosine decay of the learning rate to 0.
  - The generator for the scores is the mean of the weights (EMA, decay 0.999, smaller at the start).
  - The last 200 train images are the tune set. The checkpoint with the best tune LPIPS is kept.
  - At the end: the scores on the 600 test images, a figure with examples and FLIP error maps, and the time of one image.
- Add `flip_map` to `metrics.py` for the figures.
- Change `PhongWindow`: the methods `draw_scene` and `read_image` hold the drawing and the reading of one scene. The data generation does not change.
- Add `renderer/src/neural_window.py` (answer to "the network must be connected to the renderer"):
  - The window samples a scene. It shows the Phong image (left) and the image of the generator (right) for the same scene.
  - `main.py` starts this window when the command has `--generator <path>`.
  - With `--output_path`, the window saves the pairs (`compare_NNNN.png`) and `params.csv`.
  - With `--scene_seconds`, a visible window keeps each scene for some seconds.
- Local test with random weights (5 frames, headless): the window image is equal to the saved pair. The scene 0 is equal to the scene 0 of `data-v1`.
- Add the job `render-v1` (not started). It runs the renderer with the generator for the 3000 scenes (seed 0), and scores the 600 test scenes.
- Plan of the experiments (one change for each variant):
  - `gan-v1a`: GAN + L1, features, relative coordinates (main model),
  - `gan-v1b`: only L1, no GAN,
  - `gan-v1c`: raw parameters, no relative coordinates (hint 1),
  - `gan-v1d`: features, no relative coordinates,
  - `gan-v1e`: the GAN weight 1 instead of 5,
  - `gan-v1f`: `gan-v1a` with a different seed.
- Speed on a T4: a GAN step takes 0.1 s, an L1 step 0.07 s.
- Add the jobs `train-v1` and `train-v1ef`.

### 2026-11-18

- Add the results of `train-v1ef` in `results/gan-v1e/` and `results/gan-v1f/`.
- The tune metrics decrease until the last step (40000). A longer training can give a small improvement.
- The images of `gan-v1f`:
  - The best images have no visible difference.
  - The worst images: the light is behind the sphere. The network draws the bright edge, but the edge is smooth and has a weak color. The edge of the reference has steps from the mesh.
- Results of `train-v1` and `train-v2ab`: `gan-v2a` (100000 steps) is the best in all four metrics. Only L1 (`gan-v1b`) has the best FLIP and SSIM of the 40000-step runs.
- Add the job `train-v3ab`. Only L1 had 40000 steps and `gan-v2a` 100000 steps. The comparison of the GAN with only L1 needs the same number of steps. The second seed shows the random difference at 100000 steps.

- Add the results of `render-v1` in `results/render-v1/`.
- The FLIP of the renderer (0.00074) is smaller than the FLIP of the training (0.00115) for all 600 images. The other metrics do not change.
  - The training scores the float output of the network. The renderer saves 8-bit PNG images.
  - The rounding to 8 bits sets very small background values to 0. FLIP sees these values, the other metrics do not.
  - The row `neural_renderer` uses the scores of the renderer, because the renderer gives the final images.

- Add the results of `train-v3ab` in `results/gan-v3a/` and `results/gan-v3b/`.
  - With the same number of steps, the GAN (`gan-v2a`) is better than only L1 (`gan-v3a`) in all four metrics.
  - The second seed (`gan-v3b`) is worse than `gan-v2a`. The random difference between seeds is approximately 0.0003 in FLIP and 0.0001 in LPIPS.
  - `gan-v2a` stays the final model. Thus `render-v1` gives the row `neural_renderer`, and the job `render-v2` is not necessary.

- Add `experiments.ipynb`. It shows the data, the row `neural_renderer`, the table of all variants, the learning curves, the test images, the network in the renderer and the analysis of the metrics.
- Change `plot_histories` in `metrics.py`: a log scale, and 1 - SSIM instead of SSIM. On a linear scale, the early peaks hide the end of the training.

- Make the figures of the notebook easier to read:
  - Fix `plot_examples`: the column titles were missing, because `row is axes[0]` is always false for numpy rows.
  - Add `plot_scenes` and `plot_render_pairs` to `training.py`. All figures show a zoom of the sphere and the zoom area.
  - Add the job `figures-v1` for the new example figure, because the predictions are only on Kaggle.
- Make the text of `experiments.ipynb` shorter.

- Add the job `figures-v2`. The zoom size of `plot_examples` changes with the sphere size.

- Write `SUMMARY.md` (Polish) on the request of the user.
- Add `plot_change_examples` to `metrics.py`, and the figures of the report in `results/summary/`.

## State and next steps (2026-11-18)

### State

- All training jobs and `render-v1` are complete. Their small results are in `results/`.
- `gan-v2a` is the final model. `render-v1` gives the row `neural_renderer`: FLIP 0.00074, LPIPS 0.00099, SSIM 0.99950, Hausdorff 1.50.
- `experiments.ipynb` shows all results and the answer to the question "Do all metrics show the quality well?".

### Decisions

- The model is a conditional GAN with the GAN weight 5, the features and the relative coordinates, 100000 steps.
- `SUMMARY.md` has the report.
- The data set is the output of the kernel `sigk-p3-data-v1`.

### Next steps

1. At the end, merge `pro3` into `main` and change the README files (after the approval of the user).

## 2026-11-27

- Add project 3 to the table in the root `README.md`.
- Merge `pro3` into `main`.
