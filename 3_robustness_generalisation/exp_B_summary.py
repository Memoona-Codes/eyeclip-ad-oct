"""Summarise Experiment B (repeated patient-grouped cross-validation). CPU only.

For each method and input, the out-of-fold scores of all 7 folds of a repeat are pooled,
so every one of the 28 patients contributes once per repeat. Scores are already standardised
per fitted model (real-anchored), so pooling across folds is on a common scale.

Reported per method and input:
  slice AUC and patient AUC: mean and SD over the 5 repeats (each AUC over all 275 B-scans / 28 patients)
  slice and patient accuracy: each fold's own transferred threshold, pooled over folds
  patient AUC 95% CI: patient scores averaged over the 5 repeats, bootstrap over patients
                      (resampled within class, 2,000 resamples)

    python exp_B_summary.py --pred predictions_grouped_cv.csv --out grouped_cv_summary.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions_grouped_cv.csv")
    ap.add_argument("--out", default="grouped_cv_summary.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.pred)
    d["repeat"] = d["split"] // 10
    rows = []
    for (m, pp), g in d.groupby(["method", "preproc"]):
        s_auc, p_auc, s_acc, p_acc, pat_scores = [], [], [], [], []
        for r, gr in g.groupby("repeat"):
            n_pat = gr["patient_id"].nunique()
            assert n_pat == 28, f"{m} {pp} repeat {r}: {n_pat} patients (incomplete run?)"
            s_auc.append(roc_auc_score(gr["y_true"], gr["score"]))
            s_acc.append(((gr["score"] >= gr["threshold"]).astype(int) == gr["y_true"]).mean())
            pg = gr.groupby("patient_id").agg(y=("y_true", "first"), s=("score", "mean"),
                                              tau=("patient_threshold", "first"))
            p_auc.append(roc_auc_score(pg["y"], pg["s"]))
            p_acc.append(((pg["s"] >= pg["tau"]).astype(int) == pg["y"]).mean())
            pat_scores.append(pg[["y", "s"]].assign(repeat=r))
        ps = pd.concat(pat_scores).groupby(level=0).agg(y=("y", "first"), s=("s", "mean"))
        lo, hi = boot_ci(ps["y"].to_numpy(), ps["s"].to_numpy())
        rows.append(dict(method=m, input=pp, repeats=len(s_auc),
                         slice_auc=np.mean(s_auc), slice_auc_sd=np.std(s_auc, ddof=1) if len(s_auc) > 1 else np.nan,
                         slice_acc=np.mean(s_acc),
                         patient_auc=np.mean(p_auc), patient_auc_sd=np.std(p_auc, ddof=1) if len(p_auc) > 1 else np.nan,
                         patient_acc=np.mean(p_acc),
                         patient_auc_avg_scores=roc_auc_score(ps["y"], ps["s"]), ci_low=lo, ci_high=hi))
    out = pd.DataFrame(rows).sort_values(["input", "slice_auc"], ascending=[True, False])
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    # CLAHE minus raw, paired by method
    w = out.pivot(index="method", columns="input", values=["slice_auc", "patient_auc"])
    if ("slice_auc", "clahe") in w and ("slice_auc", "raw") in w:
        print("\nCLAHE minus raw:")
        print(pd.DataFrame({"slice": w["slice_auc"]["clahe"] - w["slice_auc"]["raw"],
                            "patient": w["patient_auc"]["clahe"] - w["patient_auc"]["raw"]}).round(3))


if __name__ == "__main__":
    main()
