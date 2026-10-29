"""Experiment B: repeated patient-grouped cross-validation (every patient tested).

All 28 patients (275 B-scans) are split into 7 outer folds by patient, stratified by class,
so each fold tests 2 AD and 2 CO patients and trains on the other 24. This is repeated with
5 different fold assignments, so every patient is tested once per repeat (5 times in total).
Repeat r uses model seed MODEL_SEEDS[r]. Everything else is identical to the paper: fixed
settings, inner folds grouped by patient for the threshold, real-anchored standardisation.

Copy next to final_unified_eval.py and run from reset_v2_eyeclip/:

    python moore_scripts/exp_B_grouped_cv.py --quick      # repeat 0, 2 methods, check it runs
    python moore_scripts/exp_B_grouped_cv.py              # full run
    python moore_scripts/exp_B_summary.py --pred predictions_grouped_cv.csv

The "split" column stores 10 * repeat + fold (for example 23 = repeat 2, fold 3).
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
from final_unified_eval import registry, evaluate, done_keys, MODEL_SEEDS, CONDITIONS  # noqa: E402
from threshold_and_patient_level_eval import patient_id, N_INNER_FOLDS  # noqa: E402
from pred_logger import PredLogger  # noqa: E402

N_OUTER = 7
N_REPEATS = 5
DEFAULT_METHODS = ["Zero-Shot", "Nearest Centroid", "Tip-Adapter", "TaskRes", "TaskRes+SAM",
                   "Proto-Adapter", "CoOp", "CLIP-Adapter", "Tip-Adapter-F", "Tip-Adapter-F+SAM"]


def load_all():
    """All 275 B-scans: the train and validation folders of split 0 together hold every patient once."""
    tr_p, ytr, va_p, yva = cam.load_split(0)
    paths = list(tr_p) + list(va_p)
    y = np.concatenate([ytr, yva])
    pid = np.array([patient_id(p) for p in paths])
    assert len(set(paths)) == len(paths), "duplicate files"
    n_pat = len(set(pid))
    print(f"{len(paths)} B-scans, {n_pat} patients ({len(set(pid[y == 1]))} AD, {len(set(pid[y == 0]))} CO)")
    assert n_pat == 28, "expected 28 patients; check split 0 folders"
    return paths, y, pid


def outer_folds(y, pid, repeat):
    sgkf = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=repeat)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups=pid))
    tested = []
    for tr_i, te_i in folds:
        assert not (set(pid[tr_i]) & set(pid[te_i])), "patient leakage"
        tested += sorted(set(pid[te_i]))
    assert len(tested) == len(set(tested)) == 28, "each patient must be tested exactly once per repeat"
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--conditions", nargs="*", default=["clahe", "raw"], choices=["clahe", "raw"])
    ap.add_argument("--repeats", nargs="*", type=int, default=list(range(N_REPEATS)))
    ap.add_argument("--out", default="predictions_grouped_cv.csv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.repeats, a.methods, a.out = [0], ["Zero-Shot", "TaskRes+SAM"], "predictions_grouped_cv_quick.csv"

    model, preprocess, inter = cam.build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = cam.get_zeroshot_text_embeddings().to(cam.DEVICE)
    methods = {k: v for k, v in registry(model, text).items() if k in a.methods}
    log, done, t0 = PredLogger(a.out), done_keys(a.out), time.time()

    paths, y, pid = load_all()
    paths_arr = np.array(paths)
    for cond in a.conditions:
        use_clahe = CONDITIONS[cond]
        # the encoder is frozen, so features are extracted once for all 275 B-scans
        F = {"l": cam.extract_layers7_10(model, preprocess, inter, paths, use_clahe=use_clahe),
             "zs": cam.extract_zeroshot(model, preprocess, paths, use_clahe=use_clahe)}
        for r in a.repeats:
            run_seed = MODEL_SEEDS[r % len(MODEL_SEEDS)]
            for k, (tr_i, te_i) in enumerate(outer_folds(y, pid, r)):
                split_code = 10 * r + k
                ytr, yte, pid_tr = y[tr_i], y[te_i], pid[tr_i]
                inner = list(StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                                  random_state=1000 + split_code)
                             .split(np.zeros(len(ytr)), ytr, groups=pid_tr))
                for name, (fs, fn) in methods.items():
                    if (name, cond, 0, split_code, run_seed) in done:
                        continue
                    torch.manual_seed(run_seed); np.random.seed(run_seed)
                    p_te, tau_s, tau_p = evaluate(fn, F[fs][tr_i], ytr, F[fs][te_i], pid_tr, inner, run_seed)
                    log.log(method=name, preproc=cond, split=split_code, seed=run_seed,
                            paths=list(paths_arr[te_i]), y_true=yte, scores=p_te, threshold=tau_s,
                            patient_threshold=tau_p, patient_ids=pid[te_i])
                    auc = roc_auc_score(yte, p_te) if len(set(yte)) == 2 else float("nan")
                    print(f"[{cond}] rep {r} fold {k} {name:<18} AUC={auc:.3f} "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("Done. Next: python moore_scripts/exp_B_summary.py --pred", a.out)


if __name__ == "__main__":
    main()
