import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from metrics import psnr


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = []
    for inputs, targets in loader:
        preds = model(inputs.to(device)).clamp(0, 1)
        scores += psnr(preds, targets.to(device)).tolist()
    return sum(scores) / len(scores)


def train(model, train_set, val_set, checkpoint, epochs=30, lr=2e-4, batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.L1Loss()

    history = []
    best_psnr = 0.0

    for epoch in range(1, epochs + 1):
        start = time.time()

        # train
        model.train()
        total_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            loss = loss_fn(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # validate + save best
        train_loss = total_loss / len(train_loader)
        val_psnr = validate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_psnr": val_psnr})

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), checkpoint)

        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val PSNR {val_psnr:.2f} dB | {time.time() - start:.0f}s")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return pd.DataFrame(history)