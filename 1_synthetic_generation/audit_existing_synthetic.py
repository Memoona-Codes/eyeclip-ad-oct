#!/usr/bin/env python
"""
audit_existing_synthetic.py -- quality audit of the synthetic images ALREADY on disk. Generates nothing.

Pools read (per split k = 0..4):
    synthetic_dataset_v8_clean_split{k}/{AD,CO}             original
    synthetic_dataset_v8_clean_split{k}_rescaled/{AD,CO}    rescaled
Real reference: OCT_PNG_rebuilt_ADCO_slices_cropped_CLEAN_SPLIT_seed{k}/{train,val}/{AD,CO}
Optional (--include-batch2): the old synthetic_dataset_v8_batch2 and _rescaled pools, against split 0.

Per split, pool and class it reports:
    pixel mean / SD                       (raw 0-255)
    min-max stretch signature             share of images whose min == 0 and max == 255
    FID vs real train                     Inception-v3 pool features (torchvision weights)
    FID floor                             real train vs real val (different patients, same class)
    memorisation                          nearest-neighbour SSIM and L2 to real TRAIN images
                                          (count with NN-SSIM > 0.85; L2 on [0,1] pixels at 224x224)
    diversity                             mean SSIM of random synthetic pairs vs random real pairs (same class)
    best Pearson correlation              top-2 synthetic/real pairs (for Figure 3)
Figures (split 0): fig4_pixel_histograms, fig3_top_pearson_pairs, fig2_real_vs_synthetic (PNG + PDF)

Run from ~/Gem Synthetic code:
    python reset_v2_eyeclip/moore_scripts/audit_existing_synthetic.py
    python reset_v2_eyeclip/moore_scripts/audit_existing_synthetic.py --splits 0 --quick   # fast check
Outputs: synth_audit/  (audit_per_split.csv, audit_summary.md, figures)
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT = "OCT_PNG_rebuilt_ADCO_slices_cropped_CLEAN_SPLIT_seed{k}"
POOLS = {"original": "synthetic_dataset_v8_clean_split{k}", "rescaled": "synthetic_dataset_v8_clean_split{k}_rescaled"}
OLD = {"batch2 original": "synthetic_dataset_v8_batch2", "batch2 rescaled": "synthetic_dataset_v8_batch2_rescaled"}
CLASSES = ["AD", "CO"]
SIZE = 224


def files(d):
    return sorted(glob.glob(os.path.join(d, "*.png")))


def load(paths, size=SIZE):
    """Grayscale, resized, float in [0,1]; also returns raw 8-bit min/max/mean/sd per image."""
    arr, raw = [], []
    for p in paths:
        im = Image.open(p).convert("L")
        a = np.asarray(im, dtype=np.uint8)
        raw.append((a.min(), a.max()))
        arr.append(np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0)
    return np.stack(arr), np.array(raw)


def pixel_stats(paths):
    v = np.concatenate([np.asarray(Image.open(p).convert("L"), dtype=np.float64).ravel() for p in paths])
    return v.mean(), v.std()


# ---------------------------------------------------------------- SSIM (Wang et al. 2004), 7x7 uniform window
def _win(x, k=7):
    return F.avg_pool2d(x, k, stride=1)


def ssim_one_vs_many(x, Y, C1=0.01 ** 2, C2=0.03 ** 2):
    """x: (1,1,H,W), Y: (N,1,H,W) -> (N,) SSIM of x against each image in Y."""
    X = x.expand_as(Y)
    mx, my = _win(X), _win(Y)
    sx = _win(X * X) - mx ** 2
    sy = _win(Y * Y) - my ** 2
    sxy = _win(X * Y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))
    return s.mean(dim=(1, 2, 3))


def ssim_pairs(A, B):
    """elementwise SSIM of paired stacks A[i], B[i]."""
    out = []
    for i in range(0, len(A), 64):
        a, b = A[i:i + 64], B[i:i + 64]
        ma, mb = _win(a), _win(b)
        sa = _win(a * a) - ma ** 2; sb = _win(b * b) - mb ** 2; sab = _win(a * b) - ma * mb
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        s = ((2 * ma * mb + C1) * (2 * sab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (sa + sb + C2))
        out.append(s.mean(dim=(1, 2, 3)))
    return torch.cat(out)


def memorisation(S, R):
    """For each synthetic image: max SSIM and min L2 against all real train images."""
    St = torch.tensor(S[:, None], device=DEV); Rt = torch.tensor(R[:, None], device=DEV)
    nn_ssim = torch.stack([ssim_one_vs_many(St[i:i + 1], Rt).max() for i in range(len(St))]).cpu().numpy()
    l2 = torch.cdist(St.flatten(1), Rt.flatten(1)).min(dim=1).values.cpu().numpy()
    return nn_ssim, l2


def diversity(X, n_pairs, rng):
    if len(X) < 2:
        return np.nan
    i = rng.integers(0, len(X), n_pairs); j = rng.integers(0, len(X), n_pairs)
    keep = i != j
    Xt = torch.tensor(X[:, None], device=DEV)
    return ssim_pairs(Xt[i[keep]], Xt[j[keep]]).mean().item()


def pearson_top(S, R, Spaths, Rpaths, k=2):
    s = torch.tensor(S.reshape(len(S), -1), device=DEV); r = torch.tensor(R.reshape(len(R), -1), device=DEV)
    s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-8)
    r = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-8)
    C = (s @ r.T) / s.shape[1]
    best, idx = C.max(dim=1)
    order = torch.argsort(best, descending=True)[:k].cpu().numpy()
    return [(Spaths[o], Rpaths[idx[o].item()], best[o].item()) for o in order], best.cpu().numpy()


# ---------------------------------------------------------------- FID (Heusel et al. 2017)
_INC = None


def inception():
    global _INC
    if _INC is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.fc = torch.nn.Identity()
        _INC = m.eval().to(DEV)
    return _INC


@torch.no_grad()
def inc_feats(X):
    m = inception()
    out = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    for i in range(0, len(X), 32):
        x = torch.tensor(X[i:i + 32][:, None], device=DEV).repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        out.append(m((x - mean) / std).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def fid(f1, f2):
    from scipy import linalg
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(c1 @ c2, disp=False)
    covmean = covmean.real
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))


# ---------------------------------------------------------------- figures
def figures(out, k, real_tr, pools_k, top_pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    C_REAL, C_ORIG, C_RESC = "#2a78d6", "#eb6834", "#1baf7a"

    # Figure 4: histograms
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True)
    bins = np.arange(0, 257, 4)
    for r, cls in enumerate(CLASSES):
        real = np.concatenate([np.asarray(Image.open(p).convert("L")).ravel() for p in real_tr[cls]])
        for c, pool, col in ((0, "original", C_ORIG), (1, "rescaled", C_RESC)):
            syn = np.concatenate([np.asarray(Image.open(p).convert("L")).ravel() for p in pools_k[pool][cls][:300]])
            ax = axs[r, c]
            ax.hist(real, bins=bins, density=True, alpha=0.55, color=C_REAL,
                    label=f"Real (mean {real.mean():.1f}, SD {real.std():.1f})")
            ax.hist(syn, bins=bins, density=True, alpha=0.55, color=col,
                    label=f"Synthetic, {pool} (mean {syn.mean():.1f}, SD {syn.std():.1f})")
            ax.set_title(f"{cls}, {'before' if pool == 'original' else 'after'} correction", loc="left")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 1:
                ax.set_xlabel("Pixel intensity (0 to 255)")
            if c == 0:
                ax.set_ylabel("Density")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig4_pixel_histograms_split{k}.{ext}"), dpi=300)
    plt.close(fig)

    # Figure 3: top-2 Pearson pairs per class (rescaled pool)
    fig, axs = plt.subplots(4, 3, figsize=(7.2, 9.2))
    row = 0
    for cls in CLASSES:
        for sp, rp, r_ in top_pairs[cls]:
            s = np.asarray(Image.open(sp).convert("L").resize((SIZE, SIZE)), float) / 255
            rr = np.asarray(Image.open(rp).convert("L").resize((SIZE, SIZE)), float) / 255
            axs[row, 0].imshow(s, cmap="gray", vmin=0, vmax=1); axs[row, 0].set_title(f"Synthetic ({cls})")
            axs[row, 1].scatter(s.ravel()[::7], rr.ravel()[::7], s=0.3, alpha=0.3, color=C_REAL)
            axs[row, 1].set_xlim(0, 1); axs[row, 1].set_ylim(0, 1)
            axs[row, 1].set_title(f"Pearson r = {r_:.3f}")
            axs[row, 1].set_xlabel("synthetic pixel"); axs[row, 1].set_ylabel("real pixel")
            axs[row, 2].imshow(rr, cmap="gray", vmin=0, vmax=1); axs[row, 2].set_title(f"Real ({cls})")
            for a in (axs[row, 0], axs[row, 2]):
                a.axis("off")
            row += 1
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig3_top_pearson_pairs_split{k}.{ext}"), dpi=300)
    plt.close(fig)

    # Figure 2 (no reconstructions, since nothing is decoded): real / original / rescaled, 4 per class
    fig, axs = plt.subplots(3, 8, figsize=(12, 4.8))
    rng = np.random.default_rng(0)
    for c, cls in enumerate(CLASSES):
        for r, (name, lst) in enumerate((("Real", real_tr[cls]), ("Synthetic, original", pools_k["original"][cls]),
                                         ("Synthetic, rescaled", pools_k["rescaled"][cls]))):
            pick = rng.choice(len(lst), 4, replace=False)
            for j, pi in enumerate(pick):
                ax = axs[r, c * 4 + j]
                ax.imshow(np.asarray(Image.open(lst[pi]).convert("L").resize((SIZE, SIZE))), cmap="gray", vmin=0, vmax=255)
                ax.axis("off")
                if j == 0:
                    ax.set_title(f"{name} ({cls})", loc="left", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig2_real_vs_synthetic_split{k}.{ext}"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--include-batch2", action="store_true")
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--quick", action="store_true", help="100 synthetic images per pool/class, 200 diversity pairs")
    ap.add_argument("--out", default="synth_audit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_syn = 100 if a.quick else 500
    n_pairs = 200 if a.quick else 1000
    rng = np.random.default_rng(123)
    rows = []
    do_fid = not a.no_fid
    if do_fid:
        try:
            inception()
        except Exception as e:
            print(f"FID skipped (Inception weights not available: {type(e).__name__}: {e})")
            do_fid = False

    jobs = []
    for k in a.splits:
        jobs += [(k, name, pat.format(k=k)) for name, pat in POOLS.items()]
    if a.include_batch2:
        jobs += [(0, name, d) for name, d in OLD.items()]

    real_cache = {}
    for k, pool, pdir in jobs:
        sdir = SPLIT.format(k=k)
        for cls in CLASSES:
            key = (k, cls)
            if key not in real_cache:
                tr = files(os.path.join(sdir, "train", cls)); va = files(os.path.join(sdir, "val", cls))
                Rtr, rawtr = load(tr); Rva, _ = load(va)
                ent = {"paths": tr, "X": Rtr, "raw": rawtr, "mean_sd": pixel_stats(tr),
                       "div": diversity(Rtr, n_pairs, rng)}
                if do_fid:
                    ent["f_tr"] = inc_feats(Rtr); ent["f_va"] = inc_feats(Rva)
                    ent["fid_floor"] = fid(ent["f_tr"], ent["f_va"])
                real_cache[key] = ent
            R = real_cache[key]
            sp = files(os.path.join(pdir, cls))[:n_syn]
            if not sp:
                print(f"  missing: {pdir}/{cls}"); continue
            S, raws = load(sp)
            m, sd = pixel_stats(sp)
            stretch = float(np.mean((raws[:, 0] == 0) & (raws[:, 1] == 255)))
            stretch_real = float(np.mean((R["raw"][:, 0] == 0) & (R["raw"][:, 1] == 255)))
            nn_ssim, l2 = memorisation(S, R["X"])
            flagged = int(np.sum(nn_ssim > 0.85))
            top, best = pearson_top(S, R["X"], sp, R["paths"])
            row = dict(split=k, pool=pool, cls=cls, n_synth=len(sp), n_real=len(R["paths"]),
                       real_mean=R["mean_sd"][0], real_sd=R["mean_sd"][1], synth_mean=m, synth_sd=sd,
                       minmax_stretched_synth=stretch, minmax_stretched_real=stretch_real,
                       nn_ssim_mean=float(nn_ssim.mean()), nn_ssim_sd=float(nn_ssim.std()), nn_ssim_max=float(nn_ssim.max()),
                       nn_ssim_over_085=flagged, nn_l2_min=float(l2.min()), nn_l2_mean=float(l2.mean()),
                       diversity_ssim_synth=diversity(S, n_pairs, rng), diversity_ssim_real=R["div"],
                       pearson_best=float(best.max()), pearson_top2=";".join(f"{t[2]:.3f}" for t in top))
            if do_fid:
                fs = inc_feats(S)
                row["fid"] = fid(R["f_tr"], fs)
                row["fid_floor_real_train_vs_val"] = R["fid_floor"]
                row["fid_ratio"] = row["fid"] / R["fid_floor"]
            rows.append(row)
            print(f"split {k} {pool:16s} {cls}: mean {m:6.1f} sd {sd:5.1f} | stretched {stretch:.2f} (real {stretch_real:.2f}) | "
                  f"NN-SSIM {row['nn_ssim_mean']:.3f} max {row['nn_ssim_max']:.3f} over0.85 {flagged} | "
                  f"diversity SSIM {row['diversity_ssim_synth']:.3f} (real {R['div']:.3f})"
                  + (f" | FID {row['fid']:.1f} (floor {row['fid_floor_real_train_vs_val']:.1f})" if do_fid else ""), flush=True)
            if pool == "rescaled" and k == a.splits[0]:
                real_cache[key].setdefault("top", {})
                real_cache[key]["top"] = top

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "audit_per_split.csv"), index=False)
    num = [c for c in df.columns if c not in ("split", "pool", "cls", "pearson_top2")]
    summ = df.groupby(["pool", "cls"])[num].agg(["min", "max"])
    with open(os.path.join(a.out, "audit_summary.md"), "w") as f:
        f.write("# Synthetic audit (existing images only)\n\nRange over splits (min, max).\n\n")
        try:
            f.write(summ.round(3).to_markdown())
        except ImportError:
            f.write(summ.round(3).to_string())
        f.write("\n")
    print(f"\nsaved {a.out}/audit_per_split.csv and audit_summary.md")

    # figures for the first split
    k = a.splits[0]
    real_tr = {c: real_cache[(k, c)]["paths"] for c in CLASSES}
    pools_k = {p: {c: files(os.path.join(POOLS[p].format(k=k), c)) for c in CLASSES} for p in POOLS}
    top_pairs = {c: real_cache[(k, c)].get("top", []) for c in CLASSES}
    figures(a.out, k, real_tr, pools_k, top_pairs)
    print(f"saved figures for split {k} in {a.out}/")


if __name__ == "__main__":
    main()
