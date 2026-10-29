"""Experiment A: CLAHE sensitivity analysis (robustness, not tuning).

Re-runs the paper pipeline (same splits, seeds, methods, inner folds, threshold rule and
real-anchored standardisation) with pre-specified CLAHE settings. Only the CLAHE clip limit
and tile grid change. The paper setting (clip 2.0, 8x8) stays the primary result.

Copy this file next to final_unified_eval.py and run it from the same folder as before
(reset_v2_eyeclip/, where eyeclip_visual.pt lives):

    python moore_scripts/exp_A_clahe_sensitivity.py --quick          # 1 split, 1 seed, check it runs
    python moore_scripts/exp_A_clahe_sensitivity.py                  # full run
    python moore_scripts/exp_summary.py --pred predictions_clahe_sensitivity.csv predictions_final.csv

Output preproc labels look like "clahe_c1.0_t8" so they never clash with "raw" or "clahe".
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import clahe_accuracy_full_metrics as cam  # noqa: E402
from final_unified_eval import registry, evaluate, done_keys, MODEL_SEEDS, DETERMINISTIC  # noqa: E402
from threshold_and_patient_level_eval import patient_id, N_INNER_FOLDS  # noqa: E402
from pred_logger import PredLogger  # noqa: E402

# Pre-specified before running. Do not add settings after seeing results.
DEFAULT_SETTINGS = ["1.0x8", "2.0x8", "4.0x8", "2.0x4", "2.0x16"]


def parse_setting(s):
    clip, tile = s.split("x")
    return float(clip), int(tile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS,
                    help="clipLimit x tileGrid, e.g. 1.0x8 means clipLimit=1.0, tileGridSize=(8,8)")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", type=int, default=list(cam.SPLIT_SEEDS))
    ap.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    ap.add_argument("--out", default="predictions_clahe_sensitivity.csv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.splits, a.seeds, a.out = [0], [cam.MODEL_SEED], "predictions_clahe_sensitivity_quick.csv"

    model, preprocess, inter = cam.build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = cam.get_zeroshot_text_embeddings().to(cam.DEVICE)
    methods = registry(model, text)
    if a.methods:
        methods = {k: v for k, v in methods.items() if k in a.methods}
    log, done, t0 = PredLogger(a.out), done_keys(a.out), time.time()

    for split in a.splits:
        tr_p, ytr, va_p, yva = cam.load_split(split)
        pid_tr = np.array([patient_id(p) for p in tr_p])
        pid_va = np.array([patient_id(p) for p in va_p])
        assert not (set(pid_tr) & set(pid_va)), "patient leakage"
        # identical inner folds to final_unified_eval.py
        folds = list(StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=split)
                     .split(np.zeros(len(ytr)), ytr, groups=pid_tr))
        for setting in a.settings:
            clip_limit, tile = parse_setting(setting)
            cam.CLAHE_CLIP_LIMIT, cam.CLAHE_TILE_GRID = clip_limit, (tile, tile)  # read by cam.apply_clahe
            cond = f"clahe_c{clip_limit:.1f}_t{tile}"
            feats = {
                "l": (cam.extract_layers7_10(model, preprocess, inter, tr_p, use_clahe=True),
                      cam.extract_layers7_10(model, preprocess, inter, va_p, use_clahe=True)),
                "zs": (cam.extract_zeroshot(model, preprocess, tr_p, use_clahe=True),
                       cam.extract_zeroshot(model, preprocess, va_p, use_clahe=True)),
            }
            for name, (fs, fn) in methods.items():
                Xtr, Xva = feats[fs]
                for seed in ([0] if name in DETERMINISTIC else a.seeds):
                    if (name, cond, 0, split, seed) in done:
                        continue
                    run_seed = cam.MODEL_SEED if seed == 0 else seed
                    torch.manual_seed(run_seed); np.random.seed(run_seed)
                    p_va, tau_s, tau_p = evaluate(fn, Xtr, ytr, Xva, pid_tr, folds, run_seed)
                    log.log(method=name, preproc=cond, split=split, seed=seed, paths=va_p, y_true=yva,
                            scores=p_va, threshold=tau_s, patient_threshold=tau_p, patient_ids=pid_va)
                    print(f"split {split} {cond:<16} {name:<18} seed {seed:<5} "
                          f"AUC={roc_auc_score(yva, p_va):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    cam.CLAHE_CLIP_LIMIT, cam.CLAHE_TILE_GRID = 2.0, (8, 8)
    print("Done. Next: python moore_scripts/exp_summary.py --pred", a.out, "predictions_final.csv")


if __name__ == "__main__":
    main()
