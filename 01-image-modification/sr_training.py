"""Trains one super-resolution UNet; run as a script so x4 and x8 can train on separate GPUs."""
import argparse
from pathlib import Path

from data import PatchDataset
from model import UNet
from training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patches")
    parser.add_argument("scale", type=int)
    parser.add_argument("--upsample", default="bicubic", choices=["bicubic", "learned"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    # file prefix: "" / "learned_" / "wide_" / "learned_wide_"
    prefix = ("learned_" if args.upsample == "learned" else "") + ("wide_" if args.width != 32 else "")
    out = Path(args.out)

    train_set = PatchDataset(f"{args.patches}/train/lr_x{args.scale}", f"{args.patches}/train/original")
    val_set = PatchDataset(f"{args.patches}/val/lr_x{args.scale}", f"{args.patches}/val/original")
    model = UNet(scale=args.scale, c=args.width, upsample=args.upsample)
    history = train(model, train_set, val_set, out / f"unet_{prefix}x{args.scale}.pt", epochs=args.epochs)
    history.to_csv(out / f"history_{prefix}x{args.scale}.csv", index=False)
