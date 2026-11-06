import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import LowLightDataset, _clean_axes, make_thumbnail, split_files
from metrics import evaluate, plot_comparison, psnr
from model import MODELS

LINE_COLOR = "#2a78d6"


def tile_starts(size, tile, overlap):
    last = size - tile
    return [*range(0, last, tile - overlap), last]


@torch.no_grad()
def predict_tiles(model, inputs, device, tile=512, overlap=64, batch_size=8):
    # A full image can be too large for the GPU. All tiles use one thumbnail, so they see the same scene.
    _, height, width = inputs.shape[1:]
    thumb = make_thumbnail(inputs).to(device)
    ramp = torch.minimum(torch.arange(1, tile + 1), torch.arange(tile, 0, -1)).clamp(max=overlap).float() / overlap
    window = ramp[:, None] * ramp[None, :]
    output = torch.zeros(3, height, width)
    weight = torch.zeros(height, width)
    corners = [(y, x) for y in tile_starts(height, tile, overlap) for x in tile_starts(width, tile, overlap)]

    for first in range(0, len(corners), batch_size):
        batch = corners[first:first + batch_size]
        tiles = torch.cat([inputs[:, :, y:y + tile, x:x + tile] for y, x in batch]).to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
            preds = model(tiles, thumb.expand(len(batch), -1, -1, -1))
        for (y, x), pred in zip(batch, preds.float().cpu()):
            output[:, y:y + tile, x:x + tile] += pred * window
            weight[y:y + tile, x:x + tile] += window
    return (output / weight)[None]


def validate(model, dataset, device):
    scores = []
    for index in range(len(dataset)):
        inputs, target = dataset[index]
        pred = predict_tiles(model, inputs[None], device).clamp(0, 1)
        scores.append(psnr(pred, target[None]).item())
    return sum(scores) / len(scores)


def endless(loader):
    while True:
        yield from loader


def train(model, train_set, val_set, out_dir, device, steps, batch_size, lr, val_every, deadline_hours, num_workers):
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
                        persistent_workers=num_workers > 0, pin_memory=device.startswith("cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    loss_fn = nn.L1Loss()
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start = time.time()
    history = []
    best_psnr = -1.0
    total_loss = 0.0
    counted = 0
    for step, (inputs, targets, thumbs) in enumerate(endless(loader), start=1):
        model.train()
        inputs, targets, thumbs = inputs.to(device), targets.to(device), thumbs.to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=use_amp):
            loss = loss_fn(model(inputs, thumbs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"The loss is not finite at step {step}.")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        counted += 1

        # A training that stops at the deadline still gives a model and all results.
        past_deadline = time.time() - start > deadline_hours * 3600
        if step % val_every == 0 or step == steps or past_deadline:
            model.eval()
            val_psnr = validate(model, val_set, device)
            history.append({"step": step, "train_loss": total_loss / counted, "val_psnr": val_psnr,
                            "seconds": time.time() - start})
            pd.DataFrame(history).to_csv(f"{out_dir}/history.csv", index=False)
            print(f"{os.path.basename(out_dir)}: {history[-1]}", flush=True)
            total_loss, counted = 0.0, 0
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), f"{out_dir}/best.pt")
        if step == steps or past_deadline:
            return pd.DataFrame(history)


def plot_history(history, codename, path):
    panels = [("train_loss", "Train loss (L1)"), ("val_psnr", "Validation PSNR (dB)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, (column, title) in zip(axes, panels):
        ax.plot(history["step"], history[column], color=LINE_COLOR, linewidth=2)
        ax.set_title(f"{codename}: {title}", loc="left", fontsize=10)
        ax.set_xlabel("Step")
        _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def run_experiment(codename, model_name, set_dir, csv_path, out_root, device, steps, batch_size=16, patch_size=256,
                   lr=2e-4, val_every=2000, deadline_hours=3.4, num_workers=2, limit=None):
    out_dir = f"{out_root}/{codename}"
    os.makedirs(out_dir, exist_ok=True)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    torch.manual_seed(0)

    train_names, val_names, test_names = (split_files(csv_path, split)[:limit] for split in ("train", "val", "test"))
    train_set = LowLightDataset(set_dir, train_names, patch_size=patch_size, patches_per_image=100)
    val_set = LowLightDataset(set_dir, val_names)
    model = MODELS[model_name]().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{codename}: model {model_name}, {params} parameters, device {device}", flush=True)

    history = train(model, train_set, val_set, out_dir, device, steps, batch_size, lr, val_every, deadline_hours,
                    num_workers)
    plot_history(history, codename, f"{out_dir}/history.png")

    model.load_state_dict(torch.load(f"{out_dir}/best.pt", map_location=device))
    model.eval()
    test_set = LowLightDataset(set_dir, test_names)
    predict = lambda inputs: predict_tiles(model, inputs, device)
    start = time.time()
    scores = evaluate(predict, test_set, device)
    scores.to_csv(f"{out_dir}/scores.csv", index=False)
    plot_comparison(predict, test_set, scores, codename, f"{out_dir}/comparison.png")

    summary = scores[["PSNR", "SSIM", "LPIPS"]].mean().to_frame(codename).T
    summary["model"] = model_name
    summary["params"] = params
    summary["steps"] = int(history["step"].iloc[-1])
    summary["best_val_psnr"] = history["val_psnr"].max()
    summary["train_hours"] = history["seconds"].iloc[-1] / 3600
    summary["test_seconds"] = time.time() - start
    summary.to_csv(f"{out_dir}/summary.csv", index_label="method")
    print(summary.to_string(), flush=True)
    return summary
