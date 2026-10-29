"""Experiment G: label efficiency. Does the CLAHE effect depend on how many labelled patients are available?

OCTDL, normal against any disease. The test sets are the same five patient-grouped folds as exp_C (seed 42).
For each fold, n patients per class are drawn at random from the training patients of that fold
(n = 5, 10, 20, 40, 80; 5 draws each), plus the full training set ("all"). Each head is fitted on the
drawn patients only and tested on the whole test fold, raw and CLAHE, with the same draw for both inputs.

Heads (no tuning on test data):
  EyeCLIP  Zero-Shot          (needs no labels: one value per fold, shown as a reference line)
  EyeCLIP  Nearest Centroid   (blocks 7 to 10, as in the paper)
  EyeCLIP  Linear Probe       (projected embedding; C by patient-grouped inner CV, exp_D protocol)
  RETFound Nearest Centroid   (mean of final patch tokens, RETFound's own eval transform)
  RETFound Linear Probe

Run from reset_v2_eyeclip/ after exp_C and the RETFound native run (needs octdl_index_used.csv and
retfound_feats_octdl_native.npz):
    python moore_scripts/exp_G_label_efficiency.py > expG.log 2>&1; tail -40 expG.log
EyeCLIP features are extracted once and cached in eyeclip_feats_octdl.npz.
Outputs: labeleff_runs.csv (every fit), labeleff_summary.csv, labeleff.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import clahe_accuracy_full_metrics as cam  # noqa: E402
from clahe_accuracy_full_metrics import nc_probs  # noqa: E402
from exp_D_linear_probe import linear_probe_probs  # noqa: E402

SIZES = [5, 10, 20, 40, 80, "all"]


def eyeclip_features(paths, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        assert list(z["paths"]) == list(paths), "cached EyeCLIP features belong to another image list"
        return {k: z[k] for k in z.files if k != "paths"}
    from exp_C_octdl_clahe import text_embeddings, zeroshot
    model, preprocess, inter = cam.build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = text_embeddings(model)
    d = {}
    for cond, use in [("raw", False), ("clahe", True)]:
        v = cam.extract_zeroshot(model, preprocess, paths, use_clahe=use)
        d[f"v|{cond}"] = v
        d[f"f|{cond}"] = cam.extract_layers7_10(model, preprocess, inter, paths, use_clahe=use)
        d[f"zs|{cond}"] = zeroshot(text, v)
    np.savez(cache, paths=np.array(paths), **d)
    return d


def draw(pid_tr, y_tr, n, rng):
    """Indices of the training images of n random patients per class (all patients if n == 'all')."""
    if n == "all":
        return np.arange(len(y_tr))
    keep = []
    for c in (0, 1):
        pats = np.unique(pid_tr[y_tr == c])
        if len(pats) < n:
            return None
        keep += list(rng.choice(pats, n, replace=False))
    return np.where(np.isin(pid_tr, keep))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="octdl_index_used.csv")
    ap.add_argument("--rf-cache", default="retfound_feats_octdl_native.npz")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.index)
    paths, y, pid = df["path"].tolist(), df["label"].to_numpy(), df["patient_id"].astype(str).to_numpy()
    F = eyeclip_features(paths, "eyeclip_feats_octdl.npz")
    feats = {("EyeCLIP", "Nearest Centroid"): "f", ("EyeCLIP", "Linear Probe"): "v"}
    if os.path.exists(a.rf_cache):
        R = np.load(a.rf_cache)
        assert len(R["mean|raw"]) == len(y), "RETFound cache and index differ in length"
        F["rf|raw"], F["rf|clahe"] = R["mean|raw"], R["mean|clahe"]
        feats[("RETFound", "Nearest Centroid")] = "rf"
        feats[("RETFound", "Linear Probe")] = "rf"
    else:
        print(f"WARNING: {a.rf_cache} not found, RETFound skipped")
    heads = {"Nearest Centroid": nc_probs, "Linear Probe": linear_probe_probs}

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.seed)
                 .split(np.zeros(len(y)), y, groups=pid))
    sizes = SIZES[:2] + ["all"] if a.quick else SIZES
    rows = []
    for k, (tr, te) in enumerate(folds):
        assert not (set(pid[tr]) & set(pid[te])), "patient leakage"
        for cond in ("raw", "clahe"):
            rows.append(dict(fold=k, n=0, draw=0, encoder="EyeCLIP", method="Zero-Shot", input=cond,
                             auc=roc_auc_score(y[te], F[f"zs|{cond}"][te])))
        for n in sizes:
            for r in range(1 if n == "all" else a.draws):
                rng = np.random.default_rng(1000 * k + 10 * r + (0 if n == "all" else n))
                idx = draw(pid[tr], y[tr], n, rng)
                if idx is None:
                    continue
                sub = tr[idx]
                n_pat = len(set(pid[sub]))
                for (enc, head), key in feats.items():
                    for cond in ("raw", "clahe"):
                        X = F[f"{key}|{cond}"]
                        s = heads[head](X[sub], y[sub], X[te]) if head == "Nearest Centroid" else \
                            heads[head](X[sub], y[sub], X[te], groups=pid[sub])
                        rows.append(dict(fold=k, n=n, draw=r, n_patients=n_pat, encoder=enc, method=head,
                                         input=cond, auc=roc_auc_score(y[te], s)))
                print(f"fold {k} n={n} draw {r} done ({n_pat} patients)", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv("labeleff_runs.csv", index=False)

    w = runs.pivot_table(index=["fold", "n", "draw", "encoder", "method"], columns="input", values="auc").reset_index()
    w["gain"] = w["clahe"] - w["raw"]
    w["n"] = w["n"].astype(str)
    summ = (w.groupby(["encoder", "method", "n"])
            .agg(auc_raw=("raw", "mean"), auc_clahe=("clahe", "mean"), gain=("gain", "mean"),
                 gain_sd=("gain", "std"), clahe_better=("gain", lambda g: f"{(g > 0).sum()}/{len(g)}"))
            .reset_index())
    order = {str(s): i for i, s in enumerate([0] + SIZES)}
    summ = summ.sort_values(["encoder", "method", "n"], key=lambda c: c.map(order) if c.name == "n" else c)
    summ.to_csv("labeleff_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        xs = [s for s in SIZES if s != "all"]
        for ax, enc in zip(axes, ["EyeCLIP", "RETFound"]):
            for head, mk in [("Nearest Centroid", "o"), ("Linear Probe", "s")]:
                g = summ[(summ.encoder == enc) & (summ.method == head)]
                if g.empty:
                    continue
                g = g.set_index("n")
                xv = [x for x in xs if str(x) in g.index]
                ax.errorbar(xv, [g.loc[str(x), "gain"] for x in xv], yerr=[g.loc[str(x), "gain_sd"] for x in xv],
                            marker=mk, capsize=3, label=head)
                if "all" in g.index:
                    ax.axhline(g.loc["all", "gain"], ls=":", lw=1, color=ax.lines[-1].get_color())
            zs = summ[(summ.encoder == enc) & (summ.method == "Zero-Shot")]
            if not zs.empty:
                ax.axhline(zs.gain.iloc[0], ls="--", color="grey", lw=1, label="Zero-Shot (no labels)")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            ax.set_xlabel("Labelled training patients per class")
            ax.set_title(enc)
        axes[0].set_ylabel("AUC gain from CLAHE (CLAHE minus raw)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("labeleff.png", dpi=200)
        print("saved labeleff.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
