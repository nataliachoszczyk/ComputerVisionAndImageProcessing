# Work log: project 3

## Jobs

| Date | Job | Link | Comment | Status |
|---|---|---|---|---|
| 2026-11-17 | gl-smoke | [sigk-p3-gl-smoke](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-gl-smoke) | Test OpenGL without a display (CPU). Result: the default backend fails (no X display). The `egl` backend works with Mesa llvmpipe, without apt. 3000 frames of 128x128 in 0.78 s. The headless window of `moderngl_window` also works. | complete |
| 2026-11-17 | data-v1 | [sigk-p3-data-v1](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-data-v1) | Render 3000 images of 128x128 with seed 0 (CPU, EGL). Result: `data-v1-set.zip` with the images and `params.csv`, 2400 train and 600 test images. Rendering takes 10 s, the full job 24 s. | complete |
| 2026-11-17 | train-smoke | [sigk-p3-train-smoke](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-train-smoke) | All steps of the four variants (`smoke-a` to `smoke-d`) with 300 steps each. Measure the speed. Result: all steps work. The four variants have the same L1 loss (0.0975) after 300 steps: the generator gives a black image. The sigmoid output saturates and stops the gradient. A GAN step takes 0.19 s (30000 steps: 1.6 h). | complete |
| 2026-11-17 | train-smoke2 | [sigk-p3-train-smoke2](https://www.kaggle.com/code/nataliachoszczyk/sigk-p3-train-smoke2) | The four variants (`smoke2-a` to `smoke2-d`) with 3000 steps, after the changes below. Does the generator leave the black image? Speed with AMP. | running |

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
  - The output has a sigmoid, so the background can be exact black. A noisy background gives false Canny edges.
  - The discriminator is a conditional PatchGAN with spectral normalization (16x16 patches).
- Add `training.py`:
  - The L1 loss gives the same weight to the sphere region and to the background. Most pixels are background.
  - The GAN loss is least squares. The generator for the scores is the mean of the weights (EMA, decay 0.999).
  - The last 200 train images are the tune set. The checkpoint with the best tune LPIPS is kept.
  - At the end: the scores on the 600 test images, a figure with examples and FLIP error maps, and the time of one image.
- Add `flip_map` to `metrics.py` for the figures.
- Plan of the experiments (one change for each variant):
  - `a`: GAN + L1, features, relative coordinates (main model),
  - `b`: only L1, no GAN,
  - `c`: raw parameters, no relative coordinates (hint 1),
  - `d`: features, no relative coordinates.
- Add the job `train-smoke`.
- Change `PhongWindow`: the methods `draw_scene` and `read_image` hold the drawing and the reading of one scene. The data generation does not change.
- Add `renderer/src/neural_window.py` (answer to "the network must be connected to the renderer"):
  - The window samples a scene. It shows the Phong image (left) and the image of the generator (right) for the same scene.
  - `main.py` starts this window when the command has `--generator <path>`.
  - With `--output_path`, the window saves the pairs (`compare_NNNN.png`) and `params.csv`.
  - With `--scene_seconds`, a visible window keeps each scene for some seconds.
- Local test with random weights (5 frames, headless): the window image is equal to the saved pair. The scene 0 is equal to the scene 0 of `data-v1`.
- Add the job `render-v1` (not started). It runs the renderer with the generator for the 3000 scenes (seed 0), and scores the 600 test scenes.
- Changes after `train-smoke`:
  - The generator has a linear output in the training. In the evaluation, it clamps the output to [0, 1].
  - Mixed precision (AMP) for the speed.
  - The EMA decay is small at the start, so the EMA generator is not the random start generator.
- Add the job `train-smoke2`.

## State and next steps (2026-11-17)

### State

- The branch `pro3` is on GitHub. It has the renderer changes, `data.py`, `metrics.py` and the job `data-v1`.
- The data set is the output of the kernel `sigk-p3-data-v1` (`data-v1-set.zip`). There is no separate Kaggle dataset, because only one person does the work. A training job uses `"kernel_sources": ["nataliachoszczyk/sigk-p3-data-v1"]`.
- To make the same data set locally (seed 0, 10 s), use this command in the root folder:

  ```bash
  .venv/Scripts/python 03-rendering/renderer/src/main.py --window headless --n_images 3000 --output_path <folder>
  ```

  On Kaggle, also use `--backend egl`.

### Decisions

- The model is a conditional GAN. Do a diffusion model only if time remains.
- The results table has no baseline row.
- The position ranges are the ranges of the task text. Two limits apply (answer to hint 2): the full sphere is in the view, and the light is not in the sphere.
- Most spheres are small (median diameter approximately 10 px).

### Plan of the GAN

- `model.py`:
  - The generator changes the 11 features to an image. It starts from a 4x4 map and doubles the size to 128x128.
  - Each block of the generator gets two coordinate channels (CoordConv), because the sphere position must be exact.
  - The discriminator is a conditional PatchGAN. It gets the image and the 11 features as constant channels.
- `training.py`: the GAN loss plus an L1 loss (as in pix2pix). Save the weights and calculate the metrics on the 600 test images.
- Estimate for the full training on a T4: 20-40 min (approximately 15000 steps, batch size 32).

### Next steps

1. Write `model.py` and `training.py`. Do a short test on the CPU.
2. Do a smoke job on the Kaggle GPU. Measure the speed.
3. Do the full training on Kaggle. Look at the images and the metrics. Change the model if necessary.
4. Connect the network to the renderer: for the same scene, show the Phong image and the network image side by side.
5. Calculate the metrics on the test set. Show good and bad cases and the FLIP error maps in `experiments.ipynb`.
6. The user writes `SUMMARY.md`. Do not write it.
7. At the end, merge `pro3` into `main` and change the README files.
