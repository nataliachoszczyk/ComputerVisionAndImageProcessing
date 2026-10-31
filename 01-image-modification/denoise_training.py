"""Trains a denoising UNet on online noise; run as a script so each sigma can use its own GPU."""
import argparse
from pathlib import Path

from data import PatchDataset
from denoise_data_preparation import OnlineNoiseDataset
from model import UNet
from training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patches")
    parser.add_argument("sigma", type=float)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    out = Path(args.out)
    train_set = OnlineNoiseDataset(f"{args.patches}/train/original", args.sigma)
    val_set = PatchDataset(f"{args.patches}/val/noise_{args.sigma}", f"{args.patches}/val/original")
    history = train(UNet(c=args.width), train_set, val_set, out / f"unet_wide_sigma_{args.sigma}.pt",
                    epochs=args.epochs)
    history.to_csv(out / f"history_wide_sigma_{args.sigma}.csv", index=False)
