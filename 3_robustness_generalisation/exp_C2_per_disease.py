"""Experiment C2: which OCTDL diseases gain most from CLAHE? (CPU only, a few minutes)

Uses the scores saved by exp_C_octdl_clahe.py (octdl_scores.npz) and its image list
(octdl_index_used.csv), so nothing is re-run. For each disease class, the task is
normal (NO) images against images of that disease only. For every method we report
AUC raw, AUC CLAHE, the gain and a 95% CI from a bootstrap over patients (2,000 resamples).

Run from the folder where exp_C_octdl_clahe.py was run:
    python moore_scripts/exp_C2_per_disease.py --out octdl_per_disease.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def boot_diff(y, a, b, pid, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(y))).groupby(pid).apply(list).to_list()
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    return np.percentile(diffs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="octdl_scores.npz")
    ap.add_argument("--index", default="octdl_index_used.csv")
    ap.add_argument("--normal", default="NO")
    ap.add_argument("--out", default="octdl_per_disease.csv")
    a = ap.parse_args()

    z = np.load(a.scores, allow_pickle=True)
    idx = pd.read_csv(a.index)
    y_all, pid_all = z["y"], z["pid"].astype(str)
    assert len(idx) == len(y_all), "index and scores differ in length; use the files from the same run"
    assert (idx["label"].to_numpy() == y_all).all(), "labels differ between index and scores"
    idx["cls"] = idx["path"].map(lambda p: os.path.basename(os.path.dirname(p)))
    methods = sorted({k.split("|")[0] for k in z.files if "|" in k})

    rows = []
    for cls in sorted(idx["cls"].unique()):
        if cls.upper() == a.normal.upper():
            continue
        keep = idx["cls"].isin([cls, a.normal]).to_numpy()
        y, pid = y_all[keep], pid_all[keep]
        for m in methods:
            r, c = z[f"{m}|raw"][keep], z[f"{m}|clahe"][keep]
            lo, hi = boot_diff(y, r, c, pid)
            rows.append(dict(disease=cls, n_images=int(y.sum()), n_patients=len(set(pid[y == 1])), method=m,
                             auc_raw=roc_auc_score(y, r), auc_clahe=roc_auc_score(y, c),
                             gain=roc_auc_score(y, c) - roc_auc_score(y, r), ci_low=lo, ci_high=hi,
                             significant=bool(lo > 0 or hi < 0)))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print("\nGain by disease (rows) and method (columns):")
    print(out.pivot(index="disease", columns="method", values="gain").round(3).to_string())


if __name__ == "__main__":
    main()
