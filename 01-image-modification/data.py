import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

PATCH_SIZE = 256

# DIV2K image ids per split
SPLITS = {
    "train": range(1, 751),
    "val": range(751, 801),
    "test": range(801, 901),
}


def find_image_dir(root, image_id):
    matches = sorted(Path(root).rglob(f"{image_id:04d}.png"))
    if not matches:
        raise FileNotFoundError(f"{image_id:04d}.png not found under {root}")
    return matches[0].parent


def cut_patches(div2k_root, out_root, patches_per_image=40, seed=42):
    rng = random.Random(seed)
    for split, ids in SPLITS.items():
        src = find_image_dir(div2k_root, ids[0])
        dst = Path(out_root) / split / "original"
        dst.mkdir(parents=True, exist_ok=True)

        for i in tqdm(ids, desc=split):
            img = cv2.imread(str(src / f"{i:04d}.png"))
            h, w = img.shape[:2]

            # train: random positions, val/test: grid over whole image
            if split == "train":
                corners = [(rng.randint(0, h - PATCH_SIZE), rng.randint(0, w - PATCH_SIZE))
                           for _ in range(patches_per_image)]
            else:
                corners = [(y, x)
                           for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE)
                           for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE)]

            for k, (y, x) in enumerate(corners):
                patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                if split == "train" and rng.random() < 0.5:
                    patch = cv2.flip(patch, 1)
                cv2.imwrite(str(dst / f"{i:04d}_{k:03d}.png"), patch)


def degrade_patches(out_root, name, degrade):
    for split in SPLITS:
        src = Path(out_root) / split / "original"
        dst = Path(out_root) / split / name
        dst.mkdir(parents=True, exist_ok=True)
        for path in tqdm(sorted(src.glob("*.png")), desc=f"{split}/{name}"):
            cv2.imwrite(str(dst / path.name), degrade(cv2.imread(str(path))))


def load_image(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class PatchDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_paths = sorted(Path(input_dir).glob("*.png"))
        self.target_dir = Path(target_dir)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        path = self.input_paths[idx]
        return load_image(path), load_image(self.target_dir / path.name)