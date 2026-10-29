"""Regenerate Figure 3 (raw vs CLAHE) from an original 8-bit B-scan.
Usage: python make_clahe_figure.py path/to/AD_bscan.png fig3_clahe.png"""
import sys, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
raw = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw)  # same settings as the paper
c = raw.shape[1] // 2
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1.25]})
for a, img, t in [(ax[0], raw, "(a) Raw B-scan"), (ax[1], cl, "(b) After CLAHE")]:
    a.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto"); a.set_title(t, loc="left"); a.axis("off")
    a.axvline(c, color="#F2C14E", lw=0.8, ls="--")
ax[2].plot(raw[:, c], color="#E8622A", lw=1.4, label="Raw"); ax[2].plot(cl[:, c], color="#2A7BE8", lw=1.4, label="CLAHE")
ax[2].set_title("(c) Grey level along the dashed A-scan", loc="left"); ax[2].set_xlabel("Depth (pixels)"); ax[2].set_ylabel("Grey level")
ax[2].legend(frameon=False, loc="lower left"); ax[2].spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=300, facecolor="white")
