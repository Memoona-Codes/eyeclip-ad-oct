#!/usr/bin/env python
"""
final_unified_eval.py -- the final benchmark run for the paper.

10 methods x {CLAHE, RAW} x 5 patient-disjoint splits x 5 model seeds, with
per-slice predictions saved for analyze_predictions.py.

It reuses YOUR method code unchanged, imported from:
    clahe_accuracy_full_metrics.py        (features, NC, Proto-Adapter, TaskRes, SAM)
    unified_benchmark_clahe_vs_raw.py     (Zero-Shot, Tip-Adapter, Tip-Adapter-F, CLIP-Adapter, CoOp)
    threshold_and_patient_level_eval.py   (Tip-Adapter-F+SAM, balanced_tau, to_patient)
and the exact threshold procedure of threshold_and_patient_level_eval.py:
    - scores z-scored with each model's own in-sample train scores
    - slice threshold  = balanced (sens = spec) on OUT-OF-FOLD train slice scores
      (5 patient-grouped, class-stratified inner folds)
    - patient threshold = balanced on out-of-fold train PATIENT scores (mean of slices)
So with --seeds 42 the CLAHE AUCs reproduce your existing table exactly.

What is new compared with the existing scripts
    - 5 model seeds per split (42, 100, 2024, 777, 999) instead of seed 42 only.
      Deterministic methods (Zero-Shot, Tip-Adapter, Nearest Centroid) run once (logged seed 0).
    - RAW images get the same out-of-fold / patient-level treatment as CLAHE.
    - every validation slice score is written to predictions_final.csv.

Run from reset_v2_eyeclip/ (where your scripts, eyeclip_visual.pt and the text embeddings live):
    python moore_scripts/final_unified_eval.py --quick          # split 0, seed 42, CLAHE: sanity check
    nohup python -u moore_scripts/final_unified_eval.py > final_unified_log.txt 2>&1 &
It resumes: combinations already in the CSV are skipped, so a crashed run can be restarted.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # pred_logger
sys.path.insert(0, os.getcwd())                # your scripts (run from reset_v2_eyeclip/)
sys.path.insert(1, os.path.dirname(HERE))

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from clahe_accuracy_full_metrics import (  # noqa: E402
    DEVICE, SPLIT_SEEDS, MODEL_SEED,
    load_split, build_eyeclip, extract_layers7_10, extract_zeroshot,
    get_zeroshot_text_embeddings, nc_probs, proto_adapter_probs, taskres_probs,
)
from unified_benchmark_clahe_vs_raw import (  # noqa: E402
    zeroshot_probs, tip_probs, tip_f_probs, clip_adapter_probs, coop_probs,
)
from threshold_and_patient_level_eval import (  # noqa: E402
    tip_f_sam_probs, patient_id, balanced_tau, to_patient, N_INNER_FOLDS,
)
from pred_logger import PredLogger  # noqa: E402

MODEL_SEEDS = [42, 100, 2024, 777, 999]
DETERMINISTIC = {"Zero-Shot", "Tip-Adapter", "Nearest Centroid"}
CONDITIONS = {"clahe": True, "raw": False}

# Existing results (seed 42, CLAHE, 5 splits) used for the sanity check
EXPECTED_CLAHE_SEED42 = {"CLIP-Adapter": 0.7684, "CoOp": 0.7681, "TaskRes+SAM": 0.7677,
                         "TaskRes": 0.7669, "Tip-Adapter": 0.7658, "Nearest Centroid": 0.7657,
                         "Proto-Adapter": 0.7654, "Zero-Shot": 0.7615, "Tip-Adapter-F": 0.7614}


def registry(model, text):
    """name -> (feature set, fn(Xtr, ytr, Xeval, seed) -> scores on Xeval). Same calls as your scripts."""
    return {
        "Zero-Shot":         ("zs", lambda a, b, c, s: zeroshot_probs(text, a, c)[1]),
        "Tip-Adapter":       ("zs", lambda a, b, c, s: tip_probs(text, a, b, c)[1]),
        "Tip-Adapter-F":     ("zs", lambda a, b, c, s: tip_f_probs(text, a, b, c, seed=s)[1]),
        "Tip-Adapter-F+SAM": ("zs", lambda a, b, c, s: tip_f_sam_probs(text, a, b, c, seed=s)[1]),
        "CLIP-Adapter":      ("zs", lambda a, b, c, s: clip_adapter_probs(a, b, c, seed=s)[1]),
        "CoOp":              ("zs", lambda a, b, c, s: coop_probs(model, a, b, c, seed=s)[1]),
        "Nearest Centroid":  ("l",  lambda a, b, c, s: nc_probs(a, b, c)),
        "Proto-Adapter":     ("l",  lambda a, b, c, s: proto_adapter_probs(a, b, c, model_seed=s)),
        "TaskRes":           ("zs", lambda a, b, c, s: taskres_probs(text, a, b, c, use_sam=False, model_seed=s)),
        "TaskRes+SAM":       ("zs", lambda a, b, c, s: taskres_probs(text, a, b, c, use_sam=True, model_seed=s)),
    }


def fit_score(fn, Xa, ya, Xb, seed):
    """Identical to threshold_and_patient_level_eval.fit_score: train on (Xa, ya), return scores for
    Xa and Xb standardised by this model's own in-sample mean/SD (monotone; AUC unchanged)."""
    s = fn(Xa, ya, np.concatenate([Xa, Xb]), seed)
    s_in, s_out = s[:len(Xa)], s[len(Xa):]
    mu, sd = s_in.mean(), s_in.std() + 1e-12
    return (s_in - mu) / sd, (s_out - mu) / sd


def evaluate(fn, Xtr, ytr, Xva, pid_tr, folds, seed):
    p_tr_in, p_va = fit_score(fn, Xtr, ytr, Xva, seed)
    p_oof = np.zeros(len(Xtr))
    for tr_i, ho_i in folds:
        p_oof[ho_i] = fit_score(fn, Xtr[tr_i], ytr[tr_i], Xtr[ho_i], seed)[1]
    tau_slice = balanced_tau(ytr, p_oof)
    ytr_p, ptr_p = to_patient(pid_tr, ytr, p_oof)
    tau_pat = balanced_tau(ytr_p, ptr_p)
    return p_va, tau_slice, tau_pat


def done_keys(csv_path):
    keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                keys.add((r["method"], r["preproc"], int(r["dosage"]), int(r["split"]), int(r["seed"])))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="predictions_final.csv")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=["clahe", "raw"], choices=["clahe", "raw"])
    ap.add_argument("--splits", nargs="*", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    ap.add_argument("--quick", action="store_true", help="split 0, seed 42, CLAHE only: sanity check")
    a = ap.parse_args()
    if a.quick:
        a.splits, a.seeds, a.conditions = [0], [MODEL_SEED], ["clahe"]
        a.out = "predictions_quick.csv"

    print(f"device={DEVICE} | splits={a.splits} | seeds={a.seeds} | conditions={a.conditions} | out={a.out}")
    model, preprocess, inter = build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = get_zeroshot_text_embeddings().to(DEVICE)
    methods = registry(model, text)
    if a.methods:
        methods = {k: v for k, v in methods.items() if k in a.methods}
    log = PredLogger(a.out)
    done = done_keys(a.out)
    auc_seen = {}
    t0 = time.time()

    for split in a.splits:
        tr_p, ytr, va_p, yva = load_split(split)
        pid_tr = np.array([patient_id(p) for p in tr_p])
        pid_va = np.array([patient_id(p) for p in va_p])
        assert not (set(pid_tr) & set(pid_va)), "patient leakage"
        folds = list(StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=split)
                     .split(np.zeros(len(ytr)), ytr, groups=pid_tr))
        print(f"\nSplit {split}: {len(tr_p)} train / {len(set(pid_tr))} pts ({int(ytr.sum())} AD slices), "
              f"{len(va_p)} val / {len(set(pid_va))} pts")

        for cond in a.conditions:
            use_clahe = CONDITIONS[cond]
            feats = {
                "l": (extract_layers7_10(model, preprocess, inter, tr_p, use_clahe=use_clahe),
                      extract_layers7_10(model, preprocess, inter, va_p, use_clahe=use_clahe)),
                "zs": (extract_zeroshot(model, preprocess, tr_p, use_clahe=use_clahe),
                       extract_zeroshot(model, preprocess, va_p, use_clahe=use_clahe)),
            }
            for name, (fs, fn) in methods.items():
                Xtr, Xva = feats[fs]
                seeds = [0] if name in DETERMINISTIC else a.seeds
                for seed in seeds:
                    if (name, cond, 0, split, seed) in done:
                        continue
                    run_seed = MODEL_SEED if seed == 0 else seed
                    torch.manual_seed(run_seed); np.random.seed(run_seed)
                    p_va, tau_s, tau_p = evaluate(fn, Xtr, ytr, Xva, pid_tr, folds, run_seed)
                    log.log(method=name, preproc=cond, split=split, seed=seed, paths=va_p,
                            y_true=yva, scores=p_va, threshold=tau_s, patient_threshold=tau_p,
                            patient_ids=pid_va)
                    auc = roc_auc_score(yva, p_va)
                    auc_seen.setdefault((name, cond, seed), []).append(auc)
                    print(f"  [{cond:5s}] {name:<18} seed {seed:<5} AUC={auc:.4f}   "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    # sanity check 1: per-split values saved by unified_benchmark_clahe_vs_raw.py (seed 42)
    ref = "unified_benchmark_per_split.csv"
    if os.path.exists(ref):
        print(f"\n=== Sanity check vs {ref} (seed 42, per split) ===")
        with open(ref) as f:
            old = {(r["method"], r["condition"].lower(), int(r["split"])): float(r["auc"]) for r in csv.DictReader(f)}
        n_ok = n_bad = 0
        for (m, cond, s), v in sorted(auc_seen.items()):
            if s not in (0, MODEL_SEED):
                continue
            for i, sp in enumerate(a.splits):
                if i < len(v) and (m, cond, sp) in old:
                    ok = abs(v[i] - old[(m, cond, sp)]) < 1e-3
                    n_ok += ok; n_bad += (not ok)
                    print(f"  {m:<18} {cond:5s} split {sp}: new {v[i]:.4f}  old {old[(m, cond, sp)]:.4f}  "
                          f"{'OK' if ok else 'MISMATCH'}")
        print(f"  -> {n_ok} match, {n_bad} mismatch")

    # sanity check 2: against the existing seed-42 CLAHE table (only meaningful with all 5 splits)
    print("\n=== Sanity check: CLAHE, seed 42 (0 for deterministic), mean over splits run ===")
    for m, exp in EXPECTED_CLAHE_SEED42.items():
        s = 0 if m in DETERMINISTIC else MODEL_SEED
        v = auc_seen.get((m, "clahe", s))
        if v:
            flag = "" if len(v) < 5 else ("OK" if abs(np.mean(v) - exp) < 0.001 else "MISMATCH")
            print(f"  {m:<18} {np.mean(v):.4f}  (existing 5-split value {exp:.4f}) {flag}")
    print(f"\nSaved: {os.path.abspath(a.out)}")
    print("Next:  python moore_scripts/analyze_predictions.py --pred", a.out, "--out results_final")


if __name__ == "__main__":
    main()
