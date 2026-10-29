"""Experiment C: does the CLAHE gain generalise to a second public OCT dataset?

Task: normal vs any retinal disease on OCTDL (or any dataset given as a CSV), with the same frozen
EyeCLIP encoder and the same CLAHE settings as the paper (OpenCV, clipLimit 2.0, 8x8 tiles).

  1. Zero-Shot (no training at all): text prompts for "normal" vs "disease".
  Linear Probe (logistic regression on the frozen embedding) is included as the standard simplest baseline.
  2. Nearest Centroid and TaskRes+SAM: 5-fold cross-validation grouped by patient, out-of-fold scores
     pooled over the whole dataset.
For each method: AUC on raw and on CLAHE, and the CLAHE minus raw difference with a 95% CI from a
bootstrap that resamples patients (clusters), 2,000 resamples. Raw and CLAHE use the same folds.

Before running, check in the EyeCLIP paper that OCTDL was NOT in its pretraining data.

Input, option 1: a CSV with columns  path,label,patient_id   (label 1 = disease, 0 = normal)
    python moore_scripts/exp_C_octdl_clahe.py --csv octdl_index.csv
Input, option 2: build the CSV from the OCTDL folders (one sub-folder per class) and its label file
    python moore_scripts/exp_C_octdl_clahe.py --root /path/OCTDL --labels /path/OCTDL_labels.csv \
        --normal NO --file-col file_name --pid-col patient_id
Check the column names in the label file first (head -2 OCTDL_labels.csv) and pass them if different.
Add --quick to run on 300 images only.
"""
import argparse
import glob
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

import clip  # noqa: E402
import clahe_accuracy_full_metrics as cam  # noqa: E402
from clahe_accuracy_full_metrics import nc_probs, taskres_probs  # noqa: E402
from exp_D_linear_probe import linear_probe_probs, register_groups  # noqa: E402

PROMPTS = {
    0: ["optical coherence tomography, normal retina",
        "a retinal OCT scan of a healthy eye",
        "OCT scan showing normal retinal layers",
        "retinal cross-section from a healthy individual",
        "optical coherence tomography scan of an eye without retinal disease"],
    1: ["optical coherence tomography, retinal disease",
        "a retinal OCT scan of an eye with retinal disease",
        "OCT scan showing abnormal retinal layers",
        "retinal cross-section with retinal pathology",
        "optical coherence tomography scan showing retinal pathology"],
}
IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def build_index(root, labels_csv, normal, file_col, pid_col):
    rows = []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for ext in IMG_EXT:
            for p in glob.glob(os.path.join(d, "**", ext), recursive=True):
                rows.append(dict(path=p, cls=cls, label=int(cls.upper() != normal.upper())))
    df = pd.DataFrame(rows)
    assert len(df), f"no images found under {root}"
    df["stem"] = df["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    if labels_csv:
        lab = pd.read_csv(labels_csv)
        assert file_col in lab and pid_col in lab, f"label file columns are {list(lab.columns)}"
        lab["stem"] = lab[file_col].astype(str).map(lambda f: os.path.splitext(os.path.basename(f))[0])
        df = df.merge(lab[["stem", pid_col]].drop_duplicates("stem"), on="stem", how="left")
        missing = df[pid_col].isna().sum()
        assert missing == 0, f"{missing} images have no patient id in the label file"
        df["patient_id"] = df[pid_col].astype(str)
    else:
        print("WARNING: no label file, patient id taken from the file name prefix. Check this is valid.")
        df["patient_id"] = df["stem"].map(lambda s: s.split("_")[0])
    print(df.groupby("cls").agg(images=("path", "size"), patients=("patient_id", "nunique")))
    return df[["path", "label", "patient_id"]]


@torch.no_grad()
def text_embeddings(model):
    rows = []
    for c in (0, 1):
        e = model.encode_text(clip.tokenize(PROMPTS[c]).to(cam.DEVICE)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        m = e.mean(0)
        rows.append(m / m.norm())
    return torch.stack(rows)                      # row 0 normal, row 1 disease


def zeroshot(text, X):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=cam.DEVICE)
        return torch.softmax(100 * (Xt @ text.T), dim=1)[:, 1].cpu().numpy()


def fit_std(fn, Xa, ya, Xb):
    """Train on (Xa, ya) and score Xb, standardised by the model's own training scores
    (real-anchored, as in the paper; monotone, so AUC within a fold is unchanged)."""
    out = fn(Xa, ya, np.concatenate([Xa, Xb]))
    s_in, s_out = out[:len(Xa)], out[len(Xa):]
    return (s_out - s_in.mean()) / (s_in.std() + 1e-12)


def boot_diff(y, a, b, pid, n=2000, seed=0):
    """Bootstrap over patients (clusters) of AUC(b) - AUC(a)."""
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
    ap.add_argument("--csv")
    ap.add_argument("--root"); ap.add_argument("--labels")
    ap.add_argument("--normal", default="NO")
    ap.add_argument("--file-col", default="file_name"); ap.add_argument("--pid-col", default="patient_id")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="octdl_clahe_results.csv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv) if a.csv else build_index(a.root, a.labels, a.normal, a.file_col, a.pid_col)
    if a.quick:
        df = df.sample(n=min(300, len(df)), random_state=0)
    df.to_csv("octdl_index_used.csv", index=False)
    paths, y, pid = df["path"].tolist(), df["label"].to_numpy(), df["patient_id"].astype(str).to_numpy()
    print(f"{len(y)} images, {len(set(pid))} patients, {y.mean():.1%} disease")

    model, preprocess, inter = cam.build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = text_embeddings(model)
    folds = list(StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
                 .split(np.zeros(len(y)), y, groups=pid))
    for tr, te in folds:
        assert not (set(pid[tr]) & set(pid[te])), "patient leakage"

    scores = {}
    for cond, use in [("raw", False), ("clahe", True)]:
        zs = cam.extract_zeroshot(model, preprocess, paths, use_clahe=use)
        ly = cam.extract_layers7_10(model, preprocess, inter, paths, use_clahe=use)
        scores[("Zero-Shot", cond)] = zeroshot(text, zs)
        register_groups(zs, pid)
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        nc = np.zeros(len(y)); tr_ = np.zeros(len(y)); lp = np.zeros(len(y))
        for tr, te in folds:
            nc[te] = fit_std(nc_probs, ly[tr], y[tr], ly[te])
            lp[te] = fit_std(linear_probe_probs, zs[tr], y[tr], zs[te])
            tr_[te] = fit_std(lambda a_, b_, c_: taskres_probs(text, a_, b_, c_, use_sam=True, model_seed=a.seed),
                              zs[tr], y[tr], zs[te])
        scores[("Nearest Centroid", cond)] = nc
        scores[("TaskRes+SAM", cond)] = tr_
        scores[("Linear Probe", cond)] = lp

    rows = []
    for m in ["Zero-Shot", "Nearest Centroid", "Linear Probe", "TaskRes+SAM"]:
        r, c = scores[(m, "raw")], scores[(m, "clahe")]
        auc_r, auc_c = roc_auc_score(y, r), roc_auc_score(y, c)
        lo, hi = boot_diff(y, r, c, pid)
        rows.append(dict(method=m, auc_raw=auc_r, auc_clahe=auc_c, diff=auc_c - auc_r, ci_low=lo, ci_high=hi,
                         significant=bool(lo > 0 or hi < 0)))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    np.savez("octdl_scores.npz", y=y, pid=pid, **{f"{m}|{c}": v for (m, c), v in scores.items()})
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
