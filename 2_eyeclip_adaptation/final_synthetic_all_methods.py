#!/usr/bin/env python
"""
final_synthetic_all_methods.py -- all 10 EyeCLIP methods, raw and CLAHE, with synthetic images added.

Uses the synthetic images ALREADY on disk (nothing is generated):
    ~/Gem Synthetic code/synthetic_dataset_v8_clean_split{k}_rescaled/{AD,CO}   (default)
Doses: +500 and +1000 (250 or 500 per class, same nested order as final_synthetic_dosage.py).
The real-only results (+0) are NOT recomputed; they are already in predictions_final.csv.

Everything else is identical to final_unified_eval.py: same 10 methods (your code), same 5 splits,
same 5 model seeds, same out-of-fold balanced thresholds over REAL training patients only.
Synthetic images are added to every training fit and never scored; validation is real only.
Synthetic images get the same preprocessing as real ones (CLAHE in the CLAHE condition, none in raw).

Run from reset_v2_eyeclip/:
    python moore_scripts/final_synthetic_all_methods.py --quick        # split 0, seed 42, a few minutes
    nohup python -u moore_scripts/final_synthetic_all_methods.py > synth_all_log.txt 2>&1 &
Output: predictions_synth_all.csv (resumes if interrupted). Then:
    python moore_scripts/analyze_predictions.py --pred predictions_final.csv predictions_synth_all.csv --out results_synth
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from clahe_accuracy_full_metrics import (  # noqa: E402
    DEVICE, SPLIT_SEEDS, MODEL_SEED, load_split, build_eyeclip,
    extract_layers7_10, extract_zeroshot, get_zeroshot_text_embeddings,
)
from threshold_and_patient_level_eval import patient_id, N_INNER_FOLDS  # noqa: E402
from final_unified_eval import registry, DETERMINISTIC, MODEL_SEEDS, done_keys  # noqa: E402
from final_synthetic_dosage import list_pool, evaluate_aug  # noqa: E402
from pred_logger import PredLogger  # noqa: E402

DEFAULT_POOL = os.path.expanduser("~/Gem Synthetic code/synthetic_dataset_v8_clean_split{split}_rescaled")
CONDITIONS = {"clahe": True, "raw": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=DEFAULT_POOL, help="folder with AD/ and CO/; {split} is filled in")
    ap.add_argument("--doses", nargs="*", type=int, default=[500, 1000])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=["clahe", "raw"], choices=["clahe", "raw"])
    ap.add_argument("--splits", nargs="*", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    ap.add_argument("--out", default="predictions_synth_all_v2.csv")
    ap.add_argument("--quick", action="store_true", help="split 0, seed 42, both conditions, +1000 only")
    a = ap.parse_args()
    if a.quick:
        a.splits, a.seeds, a.doses, a.out = [0], [MODEL_SEED], [1000], "predictions_synth_all_v2_quick.csv"

    need = max(a.doses) // 2
    for sp in a.splits:
        p = list_pool(a.pool.format(split=sp))
        if min(len(p["AD"]), len(p["CO"])) < need:
            sys.exit(f"{a.pool.format(split=sp)}: fewer than {need} images per class")
    print(f"pool: {a.pool} | doses {a.doses} | conditions {a.conditions} | splits {a.splits} | seeds {a.seeds}")

    model, preprocess, inter = build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = get_zeroshot_text_embeddings().to(DEVICE)
    methods = registry(model, text)
    if a.methods:
        methods = {k: v for k, v in methods.items() if k in a.methods}

    log = PredLogger(a.out)
    done = done_keys(a.out)
    t0 = time.time()
    summary = {}
    for split in a.splits:
        tr_p, ytr, va_p, yva = load_split(split)
        pid_tr = np.array([patient_id(p) for p in tr_p]); pid_va = np.array([patient_id(p) for p in va_p])
        assert not (set(pid_tr) & set(pid_va)), "patient leakage"
        folds = list(StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=split)
                     .split(np.zeros(len(ytr)), ytr, groups=pid_tr))
        pool = list_pool(a.pool.format(split=split))
        syn_files = pool["AD"][:need] + pool["CO"][:need]
        print(f"\nSplit {split}: {len(tr_p)} real train, {len(va_p)} val, {len(syn_files)} synthetic available", flush=True)

        for cond in a.conditions:
            uc = CONDITIONS[cond]
            real = {"l": (extract_layers7_10(model, preprocess, inter, tr_p, use_clahe=uc),
                          extract_layers7_10(model, preprocess, inter, va_p, use_clahe=uc)),
                    "zs": (extract_zeroshot(model, preprocess, tr_p, use_clahe=uc),
                           extract_zeroshot(model, preprocess, va_p, use_clahe=uc))}
            Sl = extract_layers7_10(model, preprocess, inter, syn_files, use_clahe=uc)
            Sz = extract_zeroshot(model, preprocess, syn_files, use_clahe=uc)
            syn = {"l": (Sl[:need], Sl[need:]), "zs": (Sz[:need], Sz[need:])}

            for name, (fs, fn) in methods.items():
                Xtr, Xva = real[fs]
                ad, co = syn[fs]
                for seed in ([0] if name in DETERMINISTIC else a.seeds):
                    run_seed = MODEL_SEED if seed == 0 else seed
                    for n in a.doses:
                        if (name, cond, n, split, seed) in done:
                            continue
                        k = n // 2
                        Xs = np.concatenate([ad[:k], co[:k]])
                        ys = np.concatenate([np.ones(k, int), np.zeros(k, int)])
                        torch.manual_seed(run_seed); np.random.seed(run_seed)
                        p_va, tau_s, tau_p = evaluate_aug(fn, Xtr, ytr, Xs, ys, Xva, pid_tr, folds, run_seed)
                        log.log(method=name, preproc=cond, dosage=n, split=split, seed=seed, paths=va_p,
                                y_true=yva, scores=p_va, threshold=tau_s, patient_threshold=tau_p,
                                patient_ids=pid_va)
                        pred = (p_va >= tau_s).astype(int)
                        sens = pred[yva == 1].mean(); spec = 1 - pred[yva == 0].mean()
                        auc = roc_auc_score(yva, p_va)
                        summary.setdefault((name, cond, n), []).append((auc, sens, spec))
                        print(f"  [{cond:5s}] {name:<18} +{n:<5} seed {seed:<5} AUC={auc:.4f} Sens={sens:.3f} "
                              f"Spec={spec:.3f}{'  COLLAPSED' if sens == 0 or spec == 0 else ''}   "
                              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    print("\n=== Summary (mean over runs) ===")
    for (name, cond, n), v in sorted(summary.items()):
        v = np.array(v)
        col = int(((v[:, 1] == 0) | (v[:, 2] == 0)).sum())
        print(f"  {name:<18} {cond:5s} +{n:<5} AUC={v[:, 0].mean():.4f} Sens={v[:, 1].mean():.3f} "
              f"Spec={v[:, 2].mean():.3f} collapsed {col}/{len(v)}")
    print(f"\nSaved: {os.path.abspath(a.out)}")
    print("Next:  python moore_scripts/analyze_predictions.py --pred predictions_final.csv", a.out, "--out results_synth")


if __name__ == "__main__":
    main()
