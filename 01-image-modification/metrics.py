import lpips
import pandas as pd
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_lpips_model = None


def sne(pred, target):
    return ((pred - target) ** 2).sum(dim=(1, 2, 3))


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)


def ssim(pred, target):
    pred = pred.permute(0, 2, 3, 1).cpu().numpy()
    target = target.permute(0, 2, 3, 1).cpu().numpy()
    return torch.tensor([structural_similarity(p, t, channel_axis=2, data_range=1.0)
                         for p, t in zip(pred, target)])


def lpips_score(pred, target):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
    model = _lpips_model.to(pred.device)
    return model(pred, target, normalize=True).flatten()


@torch.no_grad()
def evaluate(predict, dataset, device="cpu", batch_size=16):
    scores = {"PSNR": [], "SSIM": [], "LPIPS": [], "SNE": []}
    for inputs, targets in tqdm(DataLoader(dataset, batch_size=batch_size)):
        targets = targets.to(device)
        preds = predict(inputs.to(device)).to(device).clamp(0, 1)
        scores["PSNR"] += psnr(preds, targets).tolist()
        scores["SSIM"] += ssim(preds, targets).tolist()
        scores["LPIPS"] += lpips_score(preds, targets).tolist()
        scores["SNE"] += sne(preds, targets).tolist()
    return pd.DataFrame(scores, index=[p.name for p in dataset.input_paths])