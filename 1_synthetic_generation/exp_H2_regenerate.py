"""Experiment H2: regenerate leakage-free synthetic images that are NEW scans, not near-copies.

Why: the exp_H images (per-split VAE, w ~ U(0, 1), jitter 0.2) were near-reconstructions of single
training scans (NN-SSIM 0.85 to 0.90; top Pearson pairs r = 0.94 to 0.995). This script reuses the
trained per-split VAEs (vae_leakfree_split{k}.pth, no retraining) and changes only the sampling:
  * each pair mixes two images from DIFFERENT training patients of the same class,
  * w ~ U(0.3, 0.7), so no output sits on one source image,
  * latent jitter N(0, sigma^2), with sigma chosen by calibration.
Everything else (min-max save, per-class rescaling to the split's real training images, seed 43,
500 per class) is unchanged. Validation patients are never used.

Run from "~/Gem Synthetic code" (needs exp_H_vae_per_split.py and audit_existing_synthetic.py in
reset_v2_eyeclip/moore_scripts):
  1) calibrate on split 0 (about 5 min): prints NN-SSIM for several sigma values
       python reset_v2_eyeclip/moore_scripts/exp_H2_regenerate.py --calibrate
       python reset_v2_eyeclip/moore_scripts/exp_H2_regenerate.py --calibrate --relative --sigmas 0.25 0.5 1 1.5 2
  2) generate all splits with the chosen sigma (aim: NN-SSIM about 0.55 to 0.65, as in the paper)
       python reset_v2_eyeclip/moore_scripts/exp_H2_regenerate.py --sigma 0.8
Output: synthetic_leakfree_v2_split{k}/{AD,CO} and synthetic_leakfree_v2_split{k}_rescaled/{AD,CO}
Then audit:   python reset_v2_eyeclip/moore_scripts/audit_existing_synthetic.py --leakfree --lf-name synthetic_leakfree_v2
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_H_vae_per_split as H  # noqa: E402  (also puts the VAE code on the path)
import audit_existing_synthetic as aud  # noqa: E402

PID = lambda f: os.path.basename(f).split("_")[0]


def load_vae(k):
    path = os.path.join(H.ROOT, f"vae_leakfree_split{k}.pth")
    if not os.path.exists(path):
        path = f"vae_leakfree_split{k}.pth"
    m = H.V8VAE(latent_dim=256).to(H.DEVICE)
    m.load_state_dict(torch.load(path, map_location=H.DEVICE))
    return m.eval()


@torch.no_grad()
def class_latents(model, k, cls):
    files = H.train_files(k, cls)
    dl = DataLoader(H.Folder(files), batch_size=32, shuffle=False)
    z = torch.cat([H.encode_mu(model, x.to(H.DEVICE)) for x, _ in dl])
    return z, np.array([PID(f) for f in files])


@torch.no_grad()
def sample(model, z, pids, n, sigma, rng, wlo=0.3, whi=0.7, relative=False):
    """n decoded images (tensor, N x 1 x H x W), each mixing two different patients."""
    scale = z.std(0, keepdim=True) * sigma if relative else sigma
    out = []
    while sum(len(o) for o in out) < n:
        b = min(64, n - sum(len(o) for o in out))
        i = rng.integers(0, len(z), b)
        j = rng.integers(0, len(z), b)
        ok = pids[i] != pids[j]
        i, j = i[ok], j[ok]
        if len(i) == 0:
            continue
        w = torch.tensor(rng.uniform(wlo, whi, len(i)), dtype=z.dtype, device=z.device)[:, None]
        zs = w * z[i] + (1 - w) * z[j] + torch.randn(len(i), z.shape[1], device=z.device) * scale
        out.append(model.decode(zs).cpu())
    return torch.cat(out)[:n]


def minmax(x):
    lo = x.amin(dim=(1, 2, 3), keepdim=True)
    hi = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


def nn_ssim(imgs, k, cls):
    R, _ = aud.load(H.train_files(k, cls))
    S = torch.nn.functional.interpolate(minmax(imgs), size=(aud.SIZE, aud.SIZE), mode="bilinear",
                                        align_corners=False)[:, 0].numpy().astype(np.float32)
    s, l2 = aud.memorisation(S, R)
    return float(s.mean()), float((s > 0.85).mean()), float(l2.mean())


def real_reference(k, cls):
    """NN-SSIM of REAL validation images (other patients) against the real training images."""
    import glob
    va = sorted(glob.glob(os.path.join(H.SPLIT_DIR.format(k=k), "val", cls, "*.png")))
    V, _ = aud.load(va)
    R, _ = aud.load(H.train_files(k, cls))
    s, _ = aud.memorisation(V, R)
    return float(s.mean()), float((s > 0.85).mean()), len(va)


def calibrate(a):
    k = 0
    model = load_vae(k)
    lat = {c: class_latents(model, k, c) for c in ("AD", "CO")}
    for c in ("AD", "CO"):
        z = lat[c][0]
        print(f"{c} latent codes: per-dimension SD mean {z.std(0).mean():.3f}, |z| mean {z.norm(dim=1).mean():.2f}")
        m, frac, n = real_reference(k, c)
        print(f"{c} REAL reference: {n} validation images vs training images: NN-SSIM {m:.3f}, share >0.85 {frac:.2f}")
    print(f"split {k}: NN-SSIM of {a.calib_n} synthetic images per class (paper's images: 0.554 AD, 0.566 CO)")
    print("sigma = multiple of the latent SD when --relative is set")
    print("sigma   AD nn_ssim  AD >0.85   CO nn_ssim  CO >0.85")
    for sigma in a.sigmas:
        torch.manual_seed(43)
        rng = np.random.default_rng(43)
        row = []
        for c in ("AD", "CO"):
            z, p = lat[c]
            m, frac, _ = nn_ssim(sample(model, z, p, a.calib_n, sigma, rng, relative=a.relative), k, c)
            row += [m, frac]
        print(f"{sigma:5.2f}   {row[0]:9.3f}  {row[1]:8.2f}   {row[2]:9.3f}  {row[3]:8.2f}", flush=True)


def generate(a):
    rows = []
    for k in a.splits:
        H.check_disjoint(k)
        model = load_vae(k)
        raw_dir = os.path.join(H.ROOT, f"{a.name}_split{k}")
        res_dir = raw_dir + "_rescaled"
        random.seed(43); np.random.seed(43); torch.manual_seed(43)
        rng = np.random.default_rng(43 + k)
        for c in ("AD", "CO"):
            z, p = class_latents(model, k, c)
            imgs = sample(model, z, p, a.n, a.sigma, rng, relative=a.relative)
            os.makedirs(os.path.join(raw_dir, c), exist_ok=True)
            for g, im in enumerate(imgs):
                save_image(im, os.path.join(raw_dir, c, f"synth_lfv2_s{k}_{c}_{g + 1:04d}.png"), normalize=True)
            m, frac, l2 = nn_ssim(imgs[:a.audit_n], k, c)
            rows.append(dict(split=k, cls=c, sigma=a.sigma, relative=a.relative, nn_ssim=m, share_over_085=frac, nn_l2=l2))
            print(f"split {k} {c}: saved {len(imgs)} | NN-SSIM {m:.3f} | share >0.85 {frac:.2f} | L2 {l2:.2f}", flush=True)
        H.rescale(k, raw_dir, res_dir)
    import pandas as pd
    pd.DataFrame(rows).to_csv(f"{a.name}_quick_audit.csv", index=False)
    print("DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--sigmas", nargs="*", type=float, default=[0.2, 0.5, 0.8, 1.2, 1.6])
    ap.add_argument("--calib-n", type=int, default=100)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--splits", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--audit-n", type=int, default=200)
    ap.add_argument("--name", default="synthetic_leakfree_v2")
    ap.add_argument("--relative", action="store_true", help="sigma is a multiple of each latent dimension's SD")
    a = ap.parse_args()
    calibrate(a) if a.calibrate else generate(a)


if __name__ == "__main__":
    main()
