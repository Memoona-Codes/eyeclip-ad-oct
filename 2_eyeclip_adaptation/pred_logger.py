"""
pred_logger.py -- save per-slice predictions from your existing evaluation scripts.

Every table and figure in the paper (ROC curves, confusion matrices, 95% CIs, raw-vs-CLAHE
tests, McNemar, failure analysis, dosage) is computed by analyze_predictions.py from the CSV
this writes. Nothing needs retraining afterwards.

How to use (3 lines in your script)
-----------------------------------
    from pred_logger import PredLogger
    LOG = PredLogger("predictions_final.csv")          # once, at the top

    # inside the loop, right after you compute validation scores for one run:
    LOG.log(method="TaskRes+SAM", preproc="clahe", split=split_seed, seed=model_seed,
            paths=val_paths, y_true=y_val, scores=val_scores, threshold=thr,
            patient_threshold=thr_patient)            # patient_threshold optional

Arguments
---------
method            str   e.g. "Zero-Shot", "TaskRes+SAM" -- keep names identical across raw/CLAHE
preproc           str   "raw" or "clahe"
split             int   patient-disjoint split seed (0-4)
seed              int   model seed (42, 100, 2024, 777, 999); use 0 for deterministic methods
paths             list  validation image paths (or file names), one per slice
y_true            array 1 = AD, 0 = CO
scores            array higher = more AD (probability or logit/cosine difference, any scale)
threshold         float slice-level decision threshold chosen on TRAIN (score >= threshold -> AD)
patient_threshold float optional; threshold for the mean-per-patient score. Default: same as threshold
dosage            int   number of synthetic images in train (0 for real only)
patient_ids       list  optional; otherwise parsed from paths with the regex (AD|CO)\\d+

The file is appended to, so several scripts can write to the same CSV. Re-running a
script adds duplicate rows; analyze_predictions.py keeps only the newest copy of each
(method, preproc, dosage, split, seed, slice).
"""
import csv
import os
import re
import time
from pathlib import Path

import numpy as np

PATIENT_RE = re.compile(r"(AD|CO|YA|AQ)[_-]?(\d+)", re.IGNORECASE)
CONDITION_RE = re.compile(r"(DARK|LIGHT)", re.IGNORECASE)

FIELDS = ["run_time", "method", "preproc", "dosage", "split", "seed", "slice_id",
          "patient_id", "condition", "y_true", "score", "threshold", "patient_threshold"]


def parse_patient(path):
    m = PATIENT_RE.search(Path(str(path)).name) or PATIENT_RE.search(str(path))
    if not m:
        raise ValueError(f"Cannot find a patient ID like AD057/CO123 in: {path}. "
                         f"Pass patient_ids= explicitly.")
    return f"{m.group(1).upper()}{m.group(2)}"


def parse_condition(path):
    m = CONDITION_RE.search(str(path))
    return m.group(1).upper() if m else ""


class PredLogger:
    def __init__(self, out_csv="predictions_final.csv"):
        self.out = Path(out_csv)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        if not self.out.exists():
            with open(self.out, "w", newline="") as f:
                csv.writer(f).writerow(FIELDS)

    def log(self, method, preproc, split, seed, paths, y_true, scores, threshold,
            patient_threshold=None, dosage=0, patient_ids=None):
        y_true = np.asarray(y_true).astype(int).ravel()
        scores = np.asarray(scores, dtype=float).ravel()
        n = len(paths)
        if not (len(y_true) == len(scores) == n):
            raise ValueError(f"length mismatch: paths={n} y_true={len(y_true)} scores={len(scores)}")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores contain NaN/inf")
        if set(np.unique(y_true)) - {0, 1}:
            raise ValueError("y_true must be 0 (CO) / 1 (AD)")
        pids = list(patient_ids) if patient_ids is not None else [parse_patient(p) for p in paths]
        # sanity: label must be constant within a patient and match the ID prefix
        for pid, y in zip(pids, y_true):
            if pid.startswith("AD") and y != 1 or pid.startswith("CO") and y != 0:
                raise ValueError(f"label {y} does not match patient {pid}")
        pthr = threshold if patient_threshold is None else patient_threshold
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.out, "a", newline="") as f:
            w = csv.writer(f)
            for p, pid, y, s in zip(paths, pids, y_true, scores):
                w.writerow([stamp, method, str(preproc).lower(), int(dosage), int(split), int(seed),
                            Path(str(p)).name, pid, parse_condition(p), int(y), f"{s:.8g}",
                            f"{float(threshold):.8g}", f"{float(pthr):.8g}"])
        return n


if __name__ == "__main__":
    # self-test
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "t.csv")
    L = PredLogger(tmp)
    L.log("Demo", "clahe", 0, 42, ["AD057_DARK_s1.png", "CO123_LIGHT_s2.png"], [1, 0], [0.7, 0.2], 0.5)
    print(open(tmp).read())
