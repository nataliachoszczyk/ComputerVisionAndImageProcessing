"""Training of the conditional GAN, selection of the checkpoint and the scores on the test set."""
import copy
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import INPUTS, augment, load_split
from metrics import METRICS, evaluate, flip_map
from model import Discriminator, Generator, load_generator, save_generator

LINE_COLOR = "#2a78d6"


def sphere_region(images, margin=3):
    """The sphere pixels and a small border around them. Only the sphere is not black in the reference images."""
    sphere = (images.amax(1, keepdim=True) > 0).float()
    return F.max_pool2d(sphere, 2 * margin + 1, stride=1, padding=margin)


def reconstruction_loss(pred, target):
    # Most pixels are background. Without equal weights for the two regions, the network can ignore small spheres.
    region = sphere_region(target)
    error = (pred - target).abs().mean(1, keepdim=True)
    inside = (error * region).sum() / region.sum().clamp(min=1)
    outside = (error * (1 - region)).sum() / (1 - region).sum().clamp(min=1)
    return inside + outside


@torch.no_grad()
def predict(generator, inputs, device, batch_size=100):
    """Images (N, H, W, 3) as float arrays in [0, 1]."""
    generator.eval()
    preds = [generator(inputs[i:i + batch_size].to(device)).float().cpu() for i in range(0, len(inputs), batch_size)]
    return torch.cat(preds).permute(0, 2, 3, 1).numpy()


def to_numpy(images):
    return images.permute(0, 2, 3, 1).float().numpy() / 255.0


def train(generator, discriminator, train_inputs, train_images, tune_inputs, tune_images, out_dir, input_kind, device,
          steps, batch_size, lr, adversarial, l1_weight, eval_every, ema_decay, deadline_hours):
    """With adversarial = 0, the discriminator does not train and the generator gets only the L1 loss."""
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    # The mean of the weights over the last steps gives smoother images than the last weights of a GAN.
    ema = copy.deepcopy(generator).eval().requires_grad_(False)
    train_inputs, train_images = train_inputs.to(device), train_images.to(device)
    tune_targets = to_numpy(tune_images)

    start = time.time()
    history = []
    totals = {"loss_d": 0.0, "loss_adv": 0.0, "loss_rec": 0.0}
    counted = 0
    best_lpips = float("inf")
    for step in range(1, steps + 1):
        generator.train()
        index = torch.randint(len(train_inputs), (batch_size,), device=device)
        inputs, targets = train_inputs[index], train_images[index].float() / 255.0
        fakes = generator(inputs)

        if adversarial > 0:
            # Least-squares GAN loss: it is more stable than the cross entropy for a small data set.
            loss_d = ((discriminator(targets, inputs) - 1) ** 2).mean() + (discriminator(fakes.detach(), inputs) ** 2).mean()
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            loss_adv = ((discriminator(fakes, inputs) - 1) ** 2).mean()
        else:
            loss_d = loss_adv = torch.zeros((), device=device)
        loss_rec = reconstruction_loss(fakes, targets)
        loss_g = adversarial * loss_adv + l1_weight * loss_rec
        if not torch.isfinite(loss_g):
            raise RuntimeError(f"The loss is not finite at step {step}.")
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), generator.parameters()):
                p_ema.lerp_(p, 1 - ema_decay)

        for name, value in (("loss_d", loss_d), ("loss_adv", loss_adv), ("loss_rec", loss_rec)):
            totals[name] += value.item()
        counted += 1

        # A training that stops at the deadline still gives a model and all results.
        past_deadline = time.time() - start > deadline_hours * 3600
        if step % eval_every == 0 or step == steps or past_deadline:
            scores = evaluate(predict(ema, tune_inputs, device), tune_targets).mean()
            row = {"step": step, **{k: v / counted for k, v in totals.items()}, **scores.to_dict(),
                   "seconds": time.time() - start}
            history.append(row)
            pd.DataFrame(history).to_csv(f"{out_dir}/history.csv", index=False)
            print(f"{os.path.basename(out_dir)}: {row}", flush=True)
            totals = dict.fromkeys(totals, 0.0)
            counted = 0
            if scores["LPIPS"] < best_lpips:
                best_lpips = scores["LPIPS"]
                save_generator(ema, input_kind, f"{out_dir}/generator.pt")
        if past_deadline:
            break
    return pd.DataFrame(history)


def plot_history(history, codename, path):
    panels = [("loss_rec", "Train loss (L1)"), ("loss_adv", "Train loss (generator, GAN)"),
              ("LPIPS", "Tune LPIPS"), ("Hausdorff", "Tune Hausdorff (px)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for ax, (column, title) in zip(axes, panels):
        ax.plot(history["step"], history[column], color=LINE_COLOR, linewidth=2)
        ax.set_title(f"{codename}: {title}", loc="left", fontsize=10)
        ax.set_xlabel("Step")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def crop(image, center, size=40):
    """A square crop around the sphere, because most spheres are small."""
    half = size // 2
    y, x = (int(np.clip(c, half, image.shape[0] - half)) for c in center)
    return image[y - half:y + half, x - half:x + half]


def plot_examples(preds, targets, scores, codename, path):
    """Two best, four middle and two worst test images (LPIPS). Full images, then crops around the sphere."""
    order = scores["LPIPS"].sort_values().index.to_numpy()
    picks = [*order[:2], *order[np.linspace(0, len(order) - 1, 6).astype(int)[1:5]], *order[-2:]]
    titles = ["Reference", "Network", "Reference (crop)", "Network (crop)", "FLIP error (crop)"]
    fig, axes = plt.subplots(len(picks), len(titles), figsize=(2.2 * len(titles), 2.3 * len(picks)))
    for row, i in zip(axes, picks):
        center = np.argwhere(targets[i].max(2) > 0).mean(0)
        panels = [targets[i], preds[i], crop(targets[i], center), crop(preds[i], center),
                  crop(flip_map(preds[i], targets[i]), center)]
        for ax, panel, title in zip(row, panels, titles):
            ax.imshow(np.clip(panel, 0, 1), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row is axes[0]:
                ax.set_title(title, fontsize=9)
        s = scores.loc[i]
        row[0].set_ylabel(f"#{i}\nLPIPS {s['LPIPS']:.3f}\nHausdorff {s['Hausdorff']:.1f}", fontsize=8)
    fig.suptitle(codename, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def run_experiment(codename, source, out_root, device, input_kind="features", relative=True, adversarial=1.0,
                   steps=30000, batch_size=32, lr=2e-4, l1_weight=100.0, eval_every=2000, ema_decay=0.999,
                   n_tune=200, mirror=True, deadline_hours=3.0, limit=None, seed=0):
    """Train one variant and write its results into out_root/codename. limit makes a small smoke test."""
    out_dir = f"{out_root}/{codename}"
    os.makedirs(out_dir, exist_ok=True)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    torch.manual_seed(seed)

    params, images = load_split(source, "train")
    # The last train images are a tune set for the selection of the checkpoint. The test set is only for the scores.
    train_params, train_images = params.iloc[:-n_tune].reset_index(drop=True), images[:-n_tune]
    tune_params, tune_images = params.iloc[-n_tune:].reset_index(drop=True), images[-n_tune:]
    if limit:
        train_params, train_images = train_params.iloc[:limit], train_images[:limit]
        tune_params, tune_images = tune_params.iloc[:limit], tune_images[:limit]
    if mirror:
        train_params, train_images = augment(train_params, train_images)
    to_inputs = lambda p: torch.from_numpy(INPUTS[input_kind](p))
    train_inputs, tune_inputs = to_inputs(train_params), to_inputs(tune_params)

    generator = Generator(train_inputs.shape[1], relative).to(device)
    discriminator = Discriminator(train_inputs.shape[1], relative).to(device)
    params_g = sum(p.numel() for p in generator.parameters())
    print(f"{codename}: {len(train_inputs)} train images, {params_g} generator parameters, device {device}", flush=True)

    history = train(generator, discriminator, train_inputs, train_images, tune_inputs, tune_images, out_dir, input_kind,
                    device, steps, batch_size, lr, adversarial, l1_weight, eval_every, ema_decay, deadline_hours)
    plot_history(history, codename, f"{out_dir}/history.png")

    generator, _ = load_generator(f"{out_dir}/generator.pt", device)
    test_params, test_images = load_split(source, "test")
    if limit:
        test_params, test_images = test_params.iloc[:limit], test_images[:limit]
    test_inputs = to_inputs(test_params)
    targets = to_numpy(test_images)
    preds = predict(generator, test_inputs, device)
    scores = evaluate(preds, targets)
    scores.insert(0, "index", test_params["index"].to_numpy())
    scores.to_csv(f"{out_dir}/scores.csv", index=False)
    plot_examples(preds, targets, scores, codename, f"{out_dir}/examples.png")
    # The predictions stay on Kaggle for a later analysis. They are too large for git.
    np.savez_compressed(f"{out_dir}/predictions.npz", preds=(preds * 255).round().astype(np.uint8),
                        index=test_params["index"].to_numpy())

    # Time of one image without the first call, which includes the start of CUDA.
    one = test_inputs[:1].to(device)
    with torch.no_grad():
        generator(one)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        begin = time.time()
        for _ in range(50):
            generator(one)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    best = history.loc[history["LPIPS"].idxmin()]
    summary = {"codename": codename, **scores[METRICS].mean().round(5).to_dict(),
               "input_kind": input_kind, "relative": relative, "adversarial": adversarial, "mirror": mirror,
               "steps": int(history["step"].iloc[-1]), "best_step": int(best["step"]), "generator_params": params_g,
               "train_minutes": round(float(history["seconds"].iloc[-1]) / 60, 1),
               "ms_per_image": round((time.time() - begin) / 50 * 1000, 2)}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{codename}: {summary}", flush=True)
    return summary
