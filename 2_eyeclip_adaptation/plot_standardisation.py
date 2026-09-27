"""Figure 4: validation score distributions relative to the transferred threshold,
pooled vs real-only score standardisation (TaskRes+SAM, CLAHE)."""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "..")
FILES = {"Pooled standardisation": f"{REPO}/data/predictions/predictions_dosage_batch2_pooled.csv",
         "Real only standardisation": f"{REPO}/data/predictions/predictions_dosage_batch2.csv"}
OUT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "fig4_standardisation")
AD, CO = "#eb6834", "#2a78d6"

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); pos, neg = s[y == 1], s[y == 0]
    return (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))

D = {}
for rule, f in FILES.items():
    d = pd.read_csv(f)
    d = d[d.preproc.str.lower() == "clahe"].copy()
    d["margin"] = d.score - d.threshold
    D[rule] = d
    print(f"{rule}: {len(d)} rows; methods {sorted(d.method.unique())}; doses {sorted(d.dosage.unique())}")

rows = []
for rule, d in D.items():
    for (m, dose), g in d.groupby(["method", "dosage"]):
        runs = [(r.y_true.values, r.score.values, r.margin.values) for _, r in g.groupby(["split", "seed"])]
        called = [((mg >= 0) & (y == 1)).sum() / (y == 1).sum() for y, s, mg in runs]
        called_co = [((mg >= 0) & (y == 0)).sum() / (y == 0).sum() for y, s, mg in runs]
        coll = sum(((mg >= 0).all() or (mg < 0).all()) for y, s, mg in runs)
        rows.append(dict(rule=rule.split()[0], method=m, dose=dose, runs=len(runs),
                         AUC=np.mean([auc(y, s) for y, s, mg in runs]),
                         AD_called_AD=np.mean(called), CO_called_AD=np.mean(called_co), collapsed=coll,
                         median_margin_AD=np.median(g.margin[g.y_true == 1]),
                         median_margin_CO=np.median(g.margin[g.y_true == 0])))
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T.round(3).to_string(index=False))
T.round(4).to_csv(OUT + "_summary.csv", index=False)

panels = [("No synthetic", "original", 0), ("+500 unrescaled", "original", 500), ("+500 rescaled", "rescaled", 500)]
fig, axs = plt.subplots(2, 3, figsize=(10, 5.4), sharey=True)
allm = np.concatenate([d.margin.values for d in D.values()])
lo, hi = np.percentile(allm, [0.5, 97.5]); lo, hi = min(lo, -0.2), max(hi, 0.3)
bins = np.linspace(lo, hi, 46)
print(f"x-axis {lo:.2f} to {hi:.2f}; {np.mean((allm < lo) | (allm > hi)) * 100:.1f}% of scores placed in the edge bins")
letters = iter("abcdef")
for i, (rule, d) in enumerate(D.items()):
    for j, (title, pool, dose) in enumerate(panels):
        ax = axs[i, j]
        g = d[(d.method == f"TaskRes+SAM [{pool} pool]") & (d.dosage == dose)]
        if g.empty:
            ax.text(0.5, 0.5, "no rows", ha="center", transform=ax.transAxes); continue
        for y, col, lab in ((0, CO, "CO"), (1, AD, "AD")):
            v = np.clip(g.margin[g.y_true == y], lo, hi)
            ax.hist(v, bins=bins, density=True, color=col, alpha=0.35, edgecolor="none")
            ax.hist(v, bins=bins, density=True, histtype="step", color=col, linewidth=1.6, label=lab)
        ax.axvline(0, color="#1b1b1b", linewidth=1.2, linestyle="--"); ax.set_xlim(lo, hi)
        r = T[(T.rule == rule.split()[0]) & (T.method == f"TaskRes+SAM [{pool} pool]") & (T.dose == dose)].iloc[0]
        ax.text(0.98, 0.97, f"AUC {r.AUC:.3f}\nAD called AD {100*r.AD_called_AD:.0f}%\nCO called AD {100*r.CO_called_AD:.0f}%\n"
                f"collapsed {int(r.collapsed)}/{int(r.runs)}", transform=ax.transAxes, va="top", ha="right", fontsize=7.5, color="#333333", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))
        ax.set_title(f"({next(letters)}) {rule}, {title}", loc="left", fontsize=8.5)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#999999"); ax.spines["bottom"].set_color("#999999")
        ax.tick_params(colors="#555555", labelsize=7.5)
        if i == 1: ax.set_xlabel("Standardised score minus threshold", fontsize=8)
        if j == 0: ax.set_ylabel("Density", fontsize=8)
axs[0, 0].legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"{OUT}.{e}", dpi=300)
print(f"saved {OUT}.png, {OUT}.pdf and {OUT}_summary.csv")
