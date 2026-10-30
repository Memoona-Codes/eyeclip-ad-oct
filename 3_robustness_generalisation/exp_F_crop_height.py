"""Experiment F: is crop height a shortcut? (manuscript Section 3.2, results/robustness/expF_height_check.csv)

Part A (needs the cropped Bissig images): the height of each cropped B-scan, the slice AUC of height alone,
and the patients with any crop taller than --max-height pixels.
Part B (needs only predictions_final.csv): the main real-image analysis repeated with those patients removed
from validation. The models are not refitted; their validation B-scans are dropped before AUC is computed.

Run from reset_v2_eyeclip/ (or pass --pred and --split-dir):
    python moore_scripts/exp_F_crop_height.py --pred predictions_final.csv
Without the images, give the patients directly:
    python exp_F_crop_height.py --pred ../data/predictions/predictions_final.csv --exclude AD414 AD727 AD807 CO015
"""
import argparse
import glob
import os

import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

SPLIT_DIR = os.path.expanduser("~/Gem Synthetic code/OCT_PNG_rebuilt_ADCO_slices_cropped_CLEAN_SPLIT_seed0")


def heights(split_dir):
    """Split 0 holds every patient once (train + val)."""
    rows = []
    for sub in ("train", "val"):
        for cls, y in (("AD", 1), ("CO", 0)):
            for f in sorted(glob.glob(os.path.join(split_dir, sub, cls, "*.png"))):
                w, h = Image.open(f).size
                rows.append(dict(file=os.path.basename(f), patient_id=os.path.basename(f).split("_")[0], y=y, height=h))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions_final.csv")
    ap.add_argument("--split-dir", default=SPLIT_DIR)
    ap.add_argument("--max-height", type=int, default=110)
    ap.add_argument("--exclude", nargs="*", default=None, help="patients to remove (skips Part A)")
    ap.add_argument("--out", default="expF_height_check.csv")
    a = ap.parse_args()
    out = []

    if a.exclude is None:
        h = heights(a.split_dir)
        assert h.patient_id.nunique() == 28, "expected 28 patients in split 0 (train + val)"
        tall = h[h.height > a.max_height].groupby("patient_id").size()
        exclude = sorted(tall.index)
        out += [("n_images", len(h)),
                ("AD_mean_height_px", round(h[h.y == 1].height.mean(), 1)),
                ("CO_mean_height_px", round(h[h.y == 0].height.mean(), 1)),
                ("slice_AUC_of_height_alone", round(roc_auc_score(h.y, h.height), 3)),
                (f"patients_with_any_crop_over_{a.max_height}px", " ".join(f"{p} ({n})" for p, n in tall.items()))]
    else:
        exclude = sorted(a.exclude)

    d = pd.read_csv(a.pred)
    if "run_time" in d:
        d = d.sort_values("run_time")
    d = d.drop_duplicates(["method", "preproc", "split", "seed", "slice_id"], keep="last")
    run = ["method", "preproc", "split", "seed"]
    auc = lambda g: roc_auc_score(g.y_true, g.score)
    full = d.groupby(run).apply(auc).groupby(["method", "preproc"]).mean().unstack()
    kept = d[~d.patient_id.isin(exclude)].groupby(run).apply(auc).groupby(["method", "preproc"]).mean().unstack()
    gain = kept["clahe"] - kept["raw"]
    best_all, best_kept = full["clahe"].idxmax(), kept["clahe"].idxmax()
    out += [("excluded_patients", " ".join(exclude)),
            ("best_slice_AUC_CLAHE_all_patients", f"{full['clahe'].max():.3f} ({best_all})"),
            (f"best_slice_AUC_CLAHE_excluding_{len(exclude)}_patients", f"{kept['clahe'].max():.3f} ({best_kept})"),
            (f"CLAHE_gain_range_excluding_{len(exclude)}_patients",
             f"{gain.min():.3f} to {gain.max():.3f} ({int((gain > 0).sum())} of {len(gain)} methods positive)")]

    res = pd.DataFrame(out, columns=["check", "value"])
    res.to_csv(a.out, index=False)
    print(res.to_string(index=False))
    print("\nPer method, slice AUC with the excluded patients removed:")
    print(pd.DataFrame({"raw": kept["raw"], "clahe": kept["clahe"], "gain": gain}).round(3).to_string())


if __name__ == "__main__":
    main()
