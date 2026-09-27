#!/usr/bin/env python
"""
final_synthetic_dosage.py -- synthetic dosage on the FINAL pipeline (CLAHE + TaskRes+SAM, and
Nearest Centroid), both synthetic pools, 5 patient-disjoint splits x 5 model seeds.

For each pool (original, rescaled), dose n in {0, 250, 500, 750, 1000}:
    train = real train slices of the split  +  first n/2 AD and n/2 CO synthetic images
            (each class shuffled once with seed 123, so doses are nested: +500 contains +250)
    validation = real validation slices only
Thresholds follow threshold_and_patient_level_eval.py exactly (balanced, out-of-fold over REAL
train patients; synthetic images are added to every inner training fold but never scored).

Both pools pass through CLAHE like the real images (synthetic -> CLAHE -> CLIP preprocessing).
The rescaled pool must already be intensity-rescaled on disk (Equation 7).

Each pool folder must contain AD/ and CO/ subfolders of .png files. Put {split} in the folder name
to give every split its own pool (made by generate_clean_split_pools.py), e.g.
    --orig-dir "~/Gem Synthetic code/synthetic_dataset_v8_clean_split{split}"
    --rescaled-dir "~/Gem Synthetic code/synthetic_dataset_v8_clean_split{split}_rescaled"
A folder without {split} is used for every split, e.g.
    ~/Gem Synthetic code/synthetic_dataset_v8_2000/AD/*.png
Find candidates with:
    find ~ -maxdepth 3 -type d -iname "*synth*" | grep -v -i png

Run from reset_v2_eyeclip/:
    python moore_scripts/final_synthetic_dosage.py --orig-dir <original pool> --rescaled-dir <rescaled pool> --quick
    nohup python -u moore_scripts/final_synthetic_dosage.py --orig-dir <...> --rescaled-dir <...> > dosage_log.txt 2>&1 &
Output rows go to predictions_dosage.csv; method names are "<method> [original pool]" and
"<method> [rescaled pool]". Dose 0 is computed once and logged under both names.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(HERE))

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from clahe_accuracy_full_metrics import (  # noqa: E402
    DEVICE, SPLIT_SEEDS, MODEL_SEED,
    load_split, build_eyeclip, extract_layers7_10, extract_zeroshot,
    get_zeroshot_text_embeddings, nc_probs, taskres_probs,
)
from threshold_and_patient_level_eval import (  # noqa: E402
    patient_id, balanced_tau, to_patient, N_INNER_FOLDS,
)
from pred_logger import PredLogger  # noqa: E402
from final_unified_eval import done_keys  # noqa: E402

MODEL_SEEDS = [42, 100, 2024, 777, 999]
SHUFFLE_SEED = 123


def list_pool(pool_dir):
    out = {}
    for cls in ("AD", "CO"):
        files = sorted(glob.glob(os.path.join(os.path.expanduser(pool_dir), cls, "*.png")))
        if not files:
            sys.exit(f"no PNGs in {pool_dir}/{cls}")
        rs = np.random.RandomState(SHUFFLE_SEED)
        out[cls] = [files[i] for i in rs.permutation(len(files))]
    return out


def fit_score(fn, Xa, ya, Xb, seed):
    s = fn(Xa, ya, np.concatenate([Xa, Xb]), seed)
    s_in, s_out = s[:len(Xa)], s[len(Xa):]
    mu, sd = s_in.mean(), s_in.std() + 1e-12
    return (s_in - mu) / sd, (s_out - mu) / sd


def evaluate_aug(fn, Xtr, ytr, Xs, ys, Xva, pid_tr, folds, seed):
    """Real train + synthetic (Xs, ys) for every fit; OOF scores only for real train slices."""
    aug = lambda X, y: (np.concatenate([X, Xs]) if len(Xs) else X, np.concatenate([y, ys]) if len(ys) else y)
    Xa, ya = aug(Xtr, ytr)
    _, p_va = fit_score(fn, Xa, ya, Xva, seed)
    p_oof = np.zeros(len(Xtr))
    for tr_i, ho_i in folds:
        Xf, yf = aug(Xtr[tr_i], ytr[tr_i])
        p_oof[ho_i] = fit_score(fn, Xf, yf, Xtr[ho_i], seed)[1]
    tau_s = balanced_tau(ytr, p_oof)
    ytr_p, ptr_p = to_patient(pid_tr, ytr, p_oof)
    return p_va, tau_s, balanced_tau(ytr_p, ptr_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True, help="original (uncorrected) synthetic pool with AD/ and CO/")
    ap.add_argument("--rescaled-dir", required=True, help="intensity-rescaled synthetic pool with AD/ and CO/")
    ap.add_argument("--doses", nargs="*", type=int, default=[0, 250, 500, 750, 1000])
    ap.add_argument("--methods", nargs="*", default=["TaskRes+SAM", "Nearest Centroid"])
    ap.add_argument("--splits", nargs="*", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    ap.add_argument("--out", default="predictions_dosage.csv")
    ap.add_argument("--quick", action="store_true", help="split 0, seed 42, doses 0 and 1000")
    a = ap.parse_args()
    if a.quick:
        a.splits, a.seeds, a.doses, a.out = [0], [MODEL_SEED], [0, 1000], "predictions_dosage_quick.csv"

    dirs = {"original": a.orig_dir, "rescaled": a.rescaled_dir}
    per_split = any("{split}" in d for d in dirs.values())
    need = max(a.doses) // 2

    def pool_for(name, split):
        p = list_pool(dirs[name].format(split=split))
        if min(len(p["AD"]), len(p["CO"])) < need:
            sys.exit(f"{dirs[name].format(split=split)} has fewer than {need} images per class")
        return p

    for name in dirs:
        for sp in (a.splits if per_split else [a.splits[0]]):
            p = pool_for(name, sp)
            print(f"{name} pool{f' (split {sp})' if per_split else ''}: "
                  f"{dirs[name].format(split=sp)}  AD {len(p['AD'])}, CO {len(p['CO'])}")

    model, preprocess, inter = build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = get_zeroshot_text_embeddings().to(DEVICE)
    fns = {
        "TaskRes+SAM": ("zs", lambda a_, b, c, s: taskres_probs(text, a_, b, c, use_sam=True, model_seed=s)),
        "TaskRes": ("zs", lambda a_, b, c, s: taskres_probs(text, a_, b, c, use_sam=False, model_seed=s)),
        "Nearest Centroid": ("l", lambda a_, b, c, s: nc_probs(a_, b, c)),
    }
    deterministic = {"Nearest Centroid"}

    # synthetic features (CLAHE applied, like real images); once per pool, or once per pool and split
    syn_cache = {}

    def syn_feats(name, split):
        key = (name, split if per_split else None)
        if key not in syn_cache:
            p = pool_for(name, split)
            files = p["AD"][:need] + p["CO"][:need]
            print(f"extracting features: {name} pool, split {split} ({len(files)} images, CLAHE)")
            Fl = extract_layers7_10(model, preprocess, inter, files, use_clahe=True)
            Fz = extract_zeroshot(model, preprocess, files, use_clahe=True)
            syn_cache[key] = {"l": (Fl[:need], Fl[need:]), "zs": (Fz[:need], Fz[need:])}
        return syn_cache[key]

    log = PredLogger(a.out)
    done = done_keys(a.out)
    t0 = time.time()
    summary = {}
    for split in a.splits:
        tr_p, ytr, va_p, yva = load_split(split)
        pid_tr = np.array([patient_id(p) for p in tr_p]); pid_va = np.array([patient_id(p) for p in va_p])
        folds = list(StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=split)
                     .split(np.zeros(len(ytr)), ytr, groups=pid_tr))
        feats = {"l": (extract_layers7_10(model, preprocess, inter, tr_p, use_clahe=True),
                       extract_layers7_10(model, preprocess, inter, va_p, use_clahe=True)),
                 "zs": (extract_zeroshot(model, preprocess, tr_p, use_clahe=True),
                        extract_zeroshot(model, preprocess, va_p, use_clahe=True))}
        print(f"\nSplit {split}: {len(tr_p)} real train, {len(va_p)} val")
        for m in a.methods:
            fs, fn = fns[m]
            Xtr, Xva = feats[fs]
            for seed in ([0] if m in deterministic else a.seeds):
                run_seed = MODEL_SEED if seed == 0 else seed
                for n in a.doses:
                    pools_here = ["original", "rescaled"]
                    keys = [(f"{m} [{pl} pool]", "clahe", n, split, seed) for pl in pools_here]
                    if all(k in done for k in keys):
                        continue
                    results = {}
                    if n == 0:
                        r = evaluate_aug(fn, Xtr, ytr, np.empty((0, Xtr.shape[1])), np.empty(0, int),
                                         Xva, pid_tr, folds, run_seed)
                        results = {pl: r for pl in pools_here}
                    else:
                        k = n // 2
                        for pl in pools_here:
                            ad, co = syn_feats(pl, split)[fs]
                            Xs = np.concatenate([ad[:k], co[:k]])
                            ys = np.concatenate([np.ones(k, int), np.zeros(k, int)])
                            results[pl] = evaluate_aug(fn, Xtr, ytr, Xs, ys, Xva, pid_tr, folds, run_seed)
                    for pl, (p_va, tau_s, tau_p) in results.items():
                        name = f"{m} [{pl} pool]"
                        log.log(method=name, preproc="clahe", dosage=n, split=split, seed=seed, paths=va_p,
                                y_true=yva, scores=p_va, threshold=tau_s, patient_threshold=tau_p,
                                patient_ids=pid_va)
                        pred = (p_va >= tau_s).astype(int)
                        tn, fp, fn_, tp = confusion_matrix(yva, pred, labels=[0, 1]).ravel()
                        sens = tp / (tp + fn_) if tp + fn_ else 0.0
                        spec = tn / (tn + fp) if tn + fp else 0.0
                        auc = roc_auc_score(yva, p_va)
                        summary.setdefault((name, n), []).append((auc, sens, spec))
                        print(f"  {name:<32} +{n:<5} seed {seed:<5} AUC={auc:.4f} Sens={sens:.3f} "
                              f"Spec={spec:.3f}{'  COLLAPSED' if sens == 0 or spec == 0 else ''}   "
                              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    print("\n=== Summary (mean over runs; collapsed = Sens or Spec exactly 0) ===")
    for (name, n), v in sorted(summary.items()):
        v = np.array(v)
        col = int(((v[:, 1] == 0) | (v[:, 2] == 0)).sum())
        print(f"  {name:<32} +{n:<5} AUC={v[:, 0].mean():.4f}  Sens={v[:, 1].mean():.3f}  "
              f"Spec={v[:, 2].mean():.3f}  collapsed {col}/{len(v)}")
    print(f"\nSaved: {os.path.abspath(a.out)}")
    print("Next:  python moore_scripts/analyze_predictions.py --pred predictions_final.csv", a.out,
          "--out results_final")


if __name__ == "__main__":
    main()
