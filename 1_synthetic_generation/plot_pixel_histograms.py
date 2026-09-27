"""Supplementary Figure S2: pixel intensity histograms, real vs synthetic, before and after rescaling.
Run from the data root: python plot_pixel_histograms.py"""
import glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
REAL = "OCT_PNG_rebuilt_ADCO_slices_cropped_CLEAN_SPLIT/train/"
POOLS = [("unrescaled", "synthetic_dataset_v8_batch2", "#eb6834"), ("rescaled", "synthetic_dataset_v8_batch2_rescaled", "#1baf7a")]
px = lambda d: np.concatenate([np.asarray(Image.open(f).convert("L")).ravel() for f in sorted(glob.glob(d + "/*.png"))])
fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True); bins = np.arange(0, 257, 4)
for r, c in enumerate(["AD", "CO"]):
    real = px(REAL + c)
    for k, (name, d, col) in enumerate(POOLS):
        syn = px(d + "/" + c); ax = axs[r, k]
        ax.hist(real, bins=bins, density=True, alpha=0.55, color="#2a78d6", label=f"Real (mean {real.mean():.1f}, SD {real.std():.1f})")
        ax.hist(syn, bins=bins, density=True, alpha=0.55, color=col, label=f"Synthetic, {name} (mean {syn.mean():.1f}, SD {syn.std():.1f})")
        ax.set_title(f"{c}, {'before' if k == 0 else 'after'} rescaling", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if r == 1: ax.set_xlabel("Pixel intensity (0 to 255)")
        if k == 0: ax.set_ylabel("Density")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figS2_pixel_histograms.{e}", dpi=300)
