"""Figure 2: real B-scans (VAE training images) and synthetic B-scans before rescaling, 4 per class, random with a fixed seed.
Run from the data root:  python fig2_real_vs_synthetic.py"""
import glob, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
REAL = "OCT_PNG_rebuilt_ADCO_slices/train"
SYN = "synthetic_dataset_v8_batch2"
rng = np.random.default_rng(2026)
pick = lambda d, c: sorted(rng.choice(sorted(glob.glob(f"{d}/{c}/*.png")), 4, replace=False))
real = {c: pick(REAL, c) for c in ("AD", "CO")}
syn = {c: pick(SYN, c) for c in ("AD", "CO")}
rows = [("(a) Real B-scans (VAE training images)", real), ("(b) Synthetic B-scans: VAE latent interpolation within class, before rescaling", syn)]
fig = plt.figure(figsize=(7.5, 2.55))
gs = fig.add_gridspec(2, 8, hspace=0.3, wspace=0.05, left=0.01, right=0.99, top=0.9, bottom=0.01)
for r, (title, files) in enumerate(rows):
    for k, (c, p) in enumerate([(c, p) for c in ("AD", "CO") for p in files[c]]):
        ax = fig.add_subplot(gs[r, k])
        if k == 0: first = ax
        ax.imshow(np.asarray(Image.open(p).convert("L").resize((224, 224))), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor("#eb6834" if c == "AD" else "#2a78d6"); s.set_linewidth(1.6)
        if k in (0, 4):
            ax.text(0.03, 0.97, c, transform=ax.transAxes, va="top", fontsize=7, weight="bold", color="white",
                    bbox=dict(facecolor="#eb6834" if c == "AD" else "#2a78d6", edgecolor="none", pad=1.2))
    fig.text(0.01, first.get_position().y1 + 0.012, title, fontsize=8, weight="bold", ha="left", va="bottom")
for e in ("png", "pdf"):
    fig.savefig(f"fig2_real_vs_synthetic.{e}", dpi=400)
print("real:", {c: [os.path.basename(p) for p in real[c]] for c in real})
print("synthetic:", {c: [os.path.basename(p) for p in syn[c]] for c in syn})
print("saved fig2_real_vs_synthetic.png and .pdf")
