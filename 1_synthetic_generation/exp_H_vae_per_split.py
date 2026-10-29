"""Experiment H (step 5): leakage-free synthetic images. One VAE per patient-disjoint split, trained on that
split's 20 training patients only, then the same generation, rescaling and audit as the paper.

Everything copies the original pipeline (1_synthetic_generation/):
  * VAE: models.vae_v8.V8VAE, latent 256; loss L1 + 2e-4 KL + 0.1 V8PerceptualLoss (Eq. 2); AdamW 1e-4,
    weight decay 1e-4, batch 16, 100 epochs, gradient clipping 1.0, best epoch by a 15% hold-out of the
    TRAINING images (never validation patients).
  * Generation: within-class latent interpolation, w ~ U(0, 1), jitter N(0, 0.2^2), 500 AD + 500 CO, seed 43,
    source images = that split's training images only; saved with min-max normalisation (as save_image).
  * Rescaling (Eq. 7): per class, to the mean and SD of that split's real training images (native size).
  * Audit: nearest-neighbour SSIM and L2 against the split's real training images of the same class,
    flag SSIM > 0.85 and L2 < 2.0; label check (class of the most similar real image).
Output per split k:  ~/Gem Synthetic code/synthetic_leakfree_split{k}_rescaled/{AD,CO}/*.png
                     ~/Gem Synthetic code/synthetic_leakfree_split{k}/{AD,CO}/*.png (unrescaled)
                     vae_leakfree_split{k}.pth, audit in vae_leakfree_audit.csv

Run from "~/Gem Synthetic code" (the folder with models/ and losses/):
    python reset_v2_eyeclip/moore_scripts/exp_H_vae_per_split.py --splits 0 --epochs 2 --quick   # check
    python reset_v2_eyeclip/moore_scripts/exp_H_vae_per_split.py                                  # all
Then, from reset_v2_eyeclip/:
    python moore_scripts/final_synthetic_all_methods.py \
        --pool "$HOME/Gem Synthetic code/synthetic_leakfree_split{split}_rescaled" --out predictions_synth_leakfree.csv
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, os.getcwd())
from models.vae_v8 import V8VAE  # noqa: E402
from losses.perceptual_ssim import V8PerceptualLoss  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.expanduser("~/Gem Synthetic code")
SPLIT_DIR = os.path.join(ROOT, "OCT_PNG_rebuilt_ADCO_slices_cropped_CLEAN_SPLIT_seed{k}")
TF = transforms.Compose([transforms.Grayscale(1), transforms.Resize((224, 224)), transforms.ToTensor()])


class Folder(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return TF(Image.open(self.files[i]).convert("L")), 0


def train_files(k, cls=None):
    classes = [cls] if cls else ["AD", "CO"]
    fs = []
    for c in classes:
        fs += sorted(glob.glob(os.path.join(SPLIT_DIR.format(k=k), "train", c, "*.png")))
    return fs


def check_disjoint(k):
    pid = lambda f: os.path.basename(f).split("_")[0]
    tr = {pid(f) for f in train_files(k)}
    va = {pid(f) for f in glob.glob(os.path.join(SPLIT_DIR.format(k=k), "val", "*", "*.png"))}
    assert tr and va and not (tr & va), f"split {k}: train/val patients overlap or folders missing"
    return len(tr), len(va)


def encode_mu(model, x):
    if hasattr(model, "encode"):
        out = model.encode(x)
        return out[0] if isinstance(out, (tuple, list)) else out
    _, mu, _ = model(x)
    return mu


def train_vae(k, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    ds = Folder(train_files(k))
    n_val = max(1, int(len(ds) * 0.15))
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=16, shuffle=True, num_workers=2)
    vl = DataLoader(va, batch_size=16, shuffle=False, num_workers=2)
    model = V8VAE(latent_dim=256).to(DEVICE)
    perc = V8PerceptualLoss().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    path, best = f"vae_leakfree_split{k}.pth", float("inf")

    def loss_of(x):
        r, mu, lv = model(x)
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        return F.l1_loss(r, x) + 2e-4 * kl + 0.1 * perc(r, x)

    for ep in range(1, epochs + 1):
        model.train()
        for x, _ in tl:
            x = x.to(DEVICE)
            opt.zero_grad()
            l = loss_of(x)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([loss_of(x.to(DEVICE)).item() for x, _ in vl])
        if v < best:
            best = v
            torch.save(model.state_dict(), path)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  split {k} epoch {ep:3d}/{epochs} val loss {v:.4f}", flush=True)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.eval()


@torch.no_grad()
def generate(model, k, n_per_class, seed, out_dir):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    for cls in ("AD", "CO"):
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
        dl = DataLoader(Folder(train_files(k, cls)), batch_size=32, shuffle=True)
        g = 0
        while g < n_per_class:
            for x, _ in dl:
                if g >= n_per_class:
                    break
                z = encode_mu(model, x.to(DEVICE))
                b = z.size(0)
                i1, i2 = torch.randperm(b), torch.randperm(b)
                w = torch.rand(b, 1, device=DEVICE)
                zs = w * z[i1] + (1 - w) * z[i2] + torch.randn_like(z) * 0.20
                imgs = model.decode(zs)
                for im in imgs:
                    if g >= n_per_class:
                        break
                    save_image(im, os.path.join(out_dir, cls, f"synth_leakfree_s{k}_{cls}_{g + 1:04d}.png"), normalize=True)
                    g += 1


def rescale(k, src, dst):
    stats = []
    for cls in ("AD", "CO"):
        real = np.concatenate([np.asarray(Image.open(f).convert("L"), np.float64).ravel() for f in train_files(k, cls)])
        sf = sorted(glob.glob(os.path.join(src, cls, "*.png")))
        syn = np.concatenate([np.asarray(Image.open(f).convert("L"), np.float64).ravel() for f in sf])
        mr, sr, ms, ss = real.mean(), real.std(), syn.mean(), syn.std()
        os.makedirs(os.path.join(dst, cls), exist_ok=True)
        for f in sf:
            a = np.asarray(Image.open(f).convert("L"), np.float64)
            a = np.clip((a - ms) / ss * sr + mr, 0, 255).astype(np.uint8)
            Image.fromarray(a).save(os.path.join(dst, cls, os.path.basename(f)))
        stats.append(dict(cls=cls, real_mean=mr, real_sd=sr, synth_mean_before=ms, synth_sd_before=ss))
    return stats


def audit(k, syn_dir, n_check):
    from skimage.metrics import structural_similarity as ssim
    load = lambda fs: np.stack([np.asarray(Image.open(f).convert("L").resize((224, 224)), np.float64) / 255 for f in fs])
    R = {c: load(train_files(k, c)) for c in ("AD", "CO")}
    rows = []
    for cls in ("AD", "CO"):
        S = load(sorted(glob.glob(os.path.join(syn_dir, cls, "*.png")))[:n_check])
        nn_ssim, nn_l2, same, flag = [], [], 0, 0
        for s in S:
            best = {c: max(ssim(s, r, data_range=1.0) for r in R[c]) for c in R}
            l2 = min(np.linalg.norm(s - r) for r in R[cls])
            nn_ssim.append(best[cls]); nn_l2.append(l2)
            same += best[cls] >= best["CO" if cls == "AD" else "AD"]
            flag += best[cls] > 0.85 and l2 < 2.0
        rows.append(dict(split=k, cls=cls, n=len(S), nn_ssim=np.mean(nn_ssim), nn_ssim_sd=np.std(nn_ssim),
                         nn_l2=np.mean(nn_l2), label_same_class=same / len(S), flagged=flag))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n", type=int, default=500, help="synthetic images per class")
    ap.add_argument("--audit-n", type=int, default=500, help="synthetic images per class to audit")
    ap.add_argument("--quick", action="store_true", help="20 images per class, short audit")
    a = ap.parse_args()
    if a.quick:
        a.n, a.audit_n = 20, 10
    rows, stats = [], []
    for k in a.splits:
        ntr, nva = check_disjoint(k)
        print(f"== split {k}: VAE trained on {len(train_files(k))} images from {ntr} training patients "
              f"({nva} validation patients excluded)", flush=True)
        model = train_vae(k, a.epochs, seed=42 + k)
        raw_dir = os.path.join(ROOT, f"synthetic_leakfree_split{k}{'_quick' if a.quick else ''}")
        res_dir = raw_dir + "_rescaled"
        generate(model, k, a.n, seed=43, out_dir=raw_dir)
        for s in rescale(k, raw_dir, res_dir):
            stats.append(dict(split=k, **s))
        rows += audit(k, raw_dir, a.audit_n)
        print(pd.DataFrame([r for r in rows if r["split"] == k]).round(3).to_string(index=False), flush=True)
    sfx = "_quick" if a.quick else ""
    pd.DataFrame(rows).to_csv(f"vae_leakfree_audit{sfx}.csv", index=False)
    pd.DataFrame(stats).to_csv(f"vae_leakfree_rescale_stats{sfx}.csv", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
