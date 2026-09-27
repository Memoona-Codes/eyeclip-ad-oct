"""
Better decision thresholds + patient-level evaluation, CLAHE images only.

Three evaluations of the SAME methods on the SAME 5 CLEAN_SPLIT_seed{0-4} splits:
  (A) slice level, threshold from IN-SAMPLE train scores   <- current method, should
      reproduce the unified-benchmark CLAHE table exactly
  (B) slice level, threshold from OUT-OF-FOLD train scores  <- new
      (train patients split into 5 patient-grouped, class-stratified folds; each fold
       scored by a model trained on the other 4; threshold chosen on those scores;
       the final model is then trained on ALL train patients and applied to val)
  (C) patient level: each patient's slice scores averaged into one score,
      threshold from out-of-fold train PATIENT scores

AUC is identical in (A) and (B) (the final model is the same); only the
threshold changes. (C) has its own AUC, computed over ~8 val patients per split.

Also adds Tip-Adapter-F + SAM (rho=0.05) as a 10th method.

Needs, in the same folder: clahe_accuracy_full_metrics.py, unified_benchmark_clahe_vs_raw.py
Run:
  python -u threshold_and_patient_level_eval.py 2>&1 | tee threshold_patient_log.txt
"""

import os
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from clahe_accuracy_full_metrics import (
    DEVICE, SPLIT_SEEDS, MODEL_SEED, SAM,
    load_split, build_eyeclip, extract_layers7_10, extract_zeroshot,
    get_zeroshot_text_embeddings, nc_probs, proto_adapter_probs, taskres_probs,
)
from unified_benchmark_clahe_vs_raw import (
    t, _tip_setup, _tip_logits, TIPF_LR, TIPF_WD, TIPF_STEPS,
    zeroshot_probs, tip_probs, tip_f_probs, clip_adapter_probs, coop_probs,
)

N_INNER_FOLDS = 5
TIPF_SAM_RHO = 0.05
METRICS = ["auc", "acc", "f1", "sens", "spec"]


# ---------------- new method: Tip-Adapter-F + SAM ----------------
def tip_f_sam_probs(text, Xtr, ytr, Xva, seed=MODEL_SEED):
    torch.manual_seed(seed)
    y, cv, cc = _tip_setup(ytr)
    Xtr_t = t(Xtr)
    keys = nn.Parameter(Xtr_t.clone())
    opt = SAM([keys], torch.optim.AdamW, rho=TIPF_SAM_RHO, lr=TIPF_LR, weight_decay=TIPF_WD)
    ce = nn.functional.cross_entropy
    for _ in range(TIPF_STEPS):
        ce(_tip_logits(Xtr_t, keys, text, cv, cc), y).backward()
        opt.first_step(zero_grad=True)
        ce(_tip_logits(Xtr_t, keys, text, cv, cc), y).backward()
        opt.second_step(zero_grad=True)
    with torch.no_grad():
        f = lambda X: torch.softmax(_tip_logits(t(X), keys, text, cv, cc), 1)[:, 1].cpu().numpy()
        return f(Xtr), f(Xva)


# ---------------- helpers ----------------
def patient_id(path):
    return os.path.basename(path).split("_")[0]


def balanced_tau(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmin(np.abs(tpr - (1 - fpr)))]


def metrics_at(y, p, tau):
    pred = (p >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p), acc=accuracy_score(y, pred),
                f1=f1_score(y, pred, zero_division=0),
                sens=tp / (tp + fn) if tp + fn else 0.0,
                spec=tn / (tn + fp) if tn + fp else 0.0)


def to_patient(pids, y, p):
    """Average slice scores per patient -> (patient labels, patient scores)."""
    s, lab = defaultdict(list), {}
    for pid, yi, pi in zip(pids, y, p):
        s[pid].append(pi)
        assert lab.setdefault(pid, yi) == yi, f"patient {pid} has mixed labels"
    keys = sorted(s)
    return np.array([lab[k] for k in keys]), np.array([np.mean(s[k]) for k in keys])


# ---------------- method registry ----------------
def build_methods(model, text):
    """name -> (feature set, fn(Xtr, ytr, Xeval) -> scores on Xeval)."""
    pair = lambda fn: (lambda Xtr, ytr, Xev: fn(Xtr, ytr, Xev)[1])
    return {
        "Zero-Shot":        ("zs", pair(lambda a, b, c: zeroshot_probs(text, a, c))),
        "Tip-Adapter":      ("zs", pair(lambda a, b, c: tip_probs(text, a, b, c))),
        "Tip-Adapter-F":    ("zs", pair(lambda a, b, c: tip_f_probs(text, a, b, c))),
        "Tip-Adapter-F+SAM": ("zs", pair(lambda a, b, c: tip_f_sam_probs(text, a, b, c))),
        "CLIP-Adapter":     ("zs", pair(clip_adapter_probs)),
        "CoOp":             ("zs", pair(lambda a, b, c: coop_probs(model, a, b, c))),
        "Nearest Centroid": ("l",  nc_probs),
        "Proto-Adapter":    ("l",  proto_adapter_probs),
        "TaskRes":          ("zs", lambda a, b, c: taskres_probs(text, a, b, c, use_sam=False)),
        "TaskRes+SAM":      ("zs", lambda a, b, c: taskres_probs(text, a, b, c, use_sam=True)),
    }


def main():
    print("Loading EyeCLIP...")
    model, preprocess, inter = build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = get_zeroshot_text_embeddings().to(DEVICE)
    methods = build_methods(model, text)

    evals = ["A_slice_insample", "B_slice_oof", "C_patient_oof"]
    res = {e: {m: {k: [] for k in METRICS} for m in methods} for e in evals}
    rows = []

    for seed in SPLIT_SEEDS:
        tr_p, ytr, va_p, yva = load_split(seed)
        pid_tr = np.array([patient_id(p) for p in tr_p])
        pid_va = np.array([patient_id(p) for p in va_p])
        overlap = set(pid_tr) & set(pid_va)
        print(f"\nSplit {seed}: {len(tr_p)} train slices / {len(set(pid_tr))} patients, "
              f"{len(va_p)} val slices / {len(set(pid_va))} patients, overlap={len(overlap)}")
        assert not overlap, "patient leakage"

        feats = {
            "l": (extract_layers7_10(model, preprocess, inter, tr_p, use_clahe=True),
                  extract_layers7_10(model, preprocess, inter, va_p, use_clahe=True)),
            "zs": (extract_zeroshot(model, preprocess, tr_p, use_clahe=True),
                   extract_zeroshot(model, preprocess, va_p, use_clahe=True)),
        }
        folds = list(StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
                     .split(np.zeros(len(ytr)), ytr, groups=pid_tr))

        for name, (fs, fn) in methods.items():
            Xtr, Xva = feats[fs]

            def fit_score(Xa, ya, Xb):
                """Train on (Xa, ya); return z-scored scores for Xa (in-sample) and Xb,
                standardised with THIS model's own in-sample score mean/SD, so thresholds
                are comparable between different models (fold models vs final model).
                A monotone transform: AUC and in-sample-threshold results are unchanged."""
                s = fn(Xa, ya, np.concatenate([Xa, Xb]))
                s_in, s_out = s[:len(Xa)], s[len(Xa):]
                mu, sd = s_in.mean(), s_in.std() + 1e-12
                return (s_in - mu) / sd, (s_out - mu) / sd

            # final model on ALL train patients
            p_tr_in, p_va = fit_score(Xtr, ytr, Xva)

            # out-of-fold train scores, each on its own fold model's standardised scale
            p_oof = np.zeros(len(Xtr))
            for tr_i, ho_i in folds:
                p_oof[ho_i] = fit_score(Xtr[tr_i], ytr[tr_i], Xtr[ho_i])[1]

            out = {
                "A_slice_insample": metrics_at(yva, p_va, balanced_tau(ytr, p_tr_in)),
                "B_slice_oof": metrics_at(yva, p_va, balanced_tau(ytr, p_oof)),
            }
            ytr_p, ptr_p = to_patient(pid_tr, ytr, p_oof)
            yva_p, pva_p = to_patient(pid_va, yva, p_va)
            out["C_patient_oof"] = metrics_at(yva_p, pva_p, balanced_tau(ytr_p, ptr_p))

            for e in evals:
                for k in METRICS:
                    res[e][name][k].append(out[e][k])
                rows.append(dict(split=seed, method=name, evaluation=e, **out[e]))
            a, b, c = out["A_slice_insample"], out["B_slice_oof"], out["C_patient_oof"]
            print(f"  {name:<18} AUC={a['auc']:.4f} | Acc in-sample={a['acc']:.4f} "
                  f"OOF={b['acc']:.4f} | patient AUC={c['auc']:.4f} Acc={c['acc']:.4f}")

    with open("threshold_patient_per_split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fmt = lambda v: f"{np.mean(v):.4f}+-{np.std(v):.4f}"
    titles = {
        "A_slice_insample": "(A) SLICE level, threshold from IN-SAMPLE train scores (current method)",
        "B_slice_oof": "(B) SLICE level, threshold from OUT-OF-FOLD train scores",
        "C_patient_oof": "(C) PATIENT level (mean of slice scores), out-of-fold threshold",
    }
    for e in evals:
        print(f"\n{'=' * 100}\n {titles[e]}\n{'=' * 100}")
        print(f"{'Method':<20}{'AUC':<18}{'Accuracy':<18}{'F1':<18}{'Sensitivity':<18}Specificity")
        for m in sorted(methods, key=lambda m: -np.mean(res[e][m]["auc"])):
            r = res[e][m]
            print(f"{m:<20}{fmt(r['auc']):<18}{fmt(r['acc']):<18}{fmt(r['f1']):<18}"
                  f"{fmt(r['sens']):<18}{fmt(r['spec'])}")

    print(f"\n{'=' * 70}\n ACCURACY CHANGE: in-sample threshold -> out-of-fold threshold\n{'=' * 70}")
    for m in methods:
        a = np.mean(res["A_slice_insample"][m]["acc"]); b = np.mean(res["B_slice_oof"][m]["acc"])
        sa = np.std(res["A_slice_insample"][m]["acc"]); sb = np.std(res["B_slice_oof"][m]["acc"])
        print(f"{m:<20} {a:.4f}+-{sa:.4f} -> {b:.4f}+-{sb:.4f}  ({b - a:+.4f})")

    print("\nSanity check: table (A) should match the unified-benchmark CLAHE table "
          "(e.g. CLIP-Adapter 0.7684, TaskRes+SAM 0.7677, Acc 0.6526 / 0.6475).")
    print("Note: patient-level results use ~8 val patients per split, so each patient "
          "is ~12.5% of accuracy - interpret with care.")
    print("Saved per-split results to threshold_patient_per_split.csv")


if __name__ == "__main__":
    main()
