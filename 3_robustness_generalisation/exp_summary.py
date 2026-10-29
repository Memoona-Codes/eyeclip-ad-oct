"""Summarise prediction files per method and input (CPU only). Used for Experiment A.

Each run is one (split, seed). Slice AUC uses all validation B-scans of the run; patient AUC uses
the mean standardised score of each validation patient. Mean and SD are taken over runs, as in
the paper (Eq. 1). Pass predictions_final.csv as well to get the raw and paper-CLAHE rows for
comparison and the gain of every CLAHE setting over raw.

    python exp_summary.py --pred predictions_clahe_sensitivity.csv predictions_final.csv \
                          --out clahe_sensitivity_summary.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--out", default="summary.csv")
    ap.add_argument("--dosage", type=int, default=0, help="keep rows with this dosage (0 = real only)")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(p) for p in a.pred], ignore_index=True)
    d = d[d["dosage"] == a.dosage]
    rows = []
    for (m, pp), g in d.groupby(["method", "preproc"]):
        s_auc, p_auc, s_acc, n_runs = [], [], [], 0
        for (_, _), gr in g.groupby(["split", "seed"]):
            n_runs += 1
            s_auc.append(roc_auc_score(gr["y_true"], gr["score"]))
            s_acc.append(((gr["score"] >= gr["threshold"]).astype(int) == gr["y_true"]).mean())
            pg = gr.groupby("patient_id").agg(y=("y_true", "first"), s=("score", "mean"))
            p_auc.append(roc_auc_score(pg["y"], pg["s"]))
        rows.append(dict(method=m, input=pp, runs=n_runs,
                         slice_auc=np.mean(s_auc), slice_auc_sd=np.std(s_auc, ddof=1) if n_runs > 1 else np.nan,
                         slice_acc=np.mean(s_acc),
                         patient_auc=np.mean(p_auc), patient_auc_sd=np.std(p_auc, ddof=1) if n_runs > 1 else np.nan))
    out = pd.DataFrame(rows)
    raw = out[out["input"] == "raw"].set_index("method")
    out["slice_gain_vs_raw"] = out.apply(lambda r: r["slice_auc"] - raw["slice_auc"].get(r["method"], np.nan), axis=1)
    out["patient_gain_vs_raw"] = out.apply(lambda r: r["patient_auc"] - raw["patient_auc"].get(r["method"], np.nan), axis=1)
    out = out.sort_values(["method", "input"])
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    # compact view: slice AUC gain over raw, method x setting
    piv = out[out["input"] != "raw"].pivot(index="method", columns="input", values="slice_gain_vs_raw")
    print("\nSlice AUC gain over raw (positive = CLAHE setting helps):")
    print(piv.round(3).to_string())
    print("\nMethods improved per setting:", (piv > 0).sum().to_dict())


if __name__ == "__main__":
    main()
