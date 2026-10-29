"""Experiment D: linear-probe baseline (logistic regression on the frozen EyeCLIP embedding).

The linear probe is the simplest standard baseline for a frozen foundation model. It uses the
512-dimensional projected embedding v (the same input as Zero-Shot, Tip-Adapter and TaskRes),
L2-normalised and then standardised per feature on the training images, with L2-regularised
logistic regression (balanced class weights, lbfgs). C is chosen from a grid fixed in advance
(1e-4 to 10) by patient-grouped inner cross-validation on the training images only. It has no random
elements, so it is run once per split. Everything else (splits, inner folds, threshold rule,
real-anchored standardisation) is the paper pipeline.

Copy next to final_unified_eval.py and run from reset_v2_eyeclip/:

  # 1) five patient-disjoint splits, raw and CLAHE (minutes)
  python moore_scripts/exp_D_linear_probe.py --design main
  python moore_scripts/exp_summary.py --pred predictions_linear_probe.csv predictions_final.csv --out linear_probe_summary.csv

  # 2) repeated patient-grouped cross-validation (minutes)
  python moore_scripts/exp_D_linear_probe.py --design grouped
  python moore_scripts/exp_B_summary.py --pred predictions_grouped_cv_lp.csv --out grouped_cv_lp_summary.csv

  # 3) OCTDL: rerun exp_C_octdl_clahe.py (it now includes the linear probe)
"""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

# Two protocols, both with C chosen by patient-grouped inner CV on training images only:
#   "std"  : standardise each feature, C in 1e-4 .. 10
#   "clip" : no standardisation (as in the CLIP paper), C in 1e-2 .. 1e6
# Select with the environment variable LP_PROTOCOL (default "std").
LP_PROTOCOL = os.environ.get("LP_PROTOCOL", "std")
LP_GRID = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0] if LP_PROTOCOL == "std"
           else [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
_GROUPS = {}                                    # feature row (bytes) -> patient id


def register_groups(X, pid):
    """Remember the patient of every feature row, so the probe can tune C with patient-grouped folds."""
    for row, p in zip(np.asarray(X), pid):
        _GROUPS[row.tobytes()] = str(p)


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


class _Identity:
    def fit(self, X):
        return self

    def transform(self, X):
        return X


def _fit(Xtr, ytr, C):
    scaler = (StandardScaler() if LP_PROTOCOL == "std" else _Identity()).fit(Xtr)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=20000, tol=1e-10).fit(scaler.transform(Xtr), ytr)
    return scaler, clf


def choose_C(X, y, groups):
    """Pick C by patient-grouped inner cross-validation on the training images only (pooled out-of-fold AUC).
    Ties go to the smaller C (stronger regularisation)."""
    n_pat = min(len(set(groups[y == c])) for c in (0, 1))
    k = min(5, n_pat)
    folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0).split(X, y, groups))
    best_C, best_auc = LP_GRID[0], -1.0
    for C in LP_GRID:
        oof = np.zeros(len(y))
        for tr, te in folds:
            sc, clf = _fit(X[tr], y[tr], C)
            oof[te] = clf.decision_function(sc.transform(X[te]))
        auc = roc_auc_score(y, oof)
        if auc > best_auc + 1e-9:
            best_C, best_auc = C, auc
    return best_C


def linear_probe_probs(X_train, y_train, X_eval, groups=None):
    """L2-normalise, standardise each feature on the training images, then L2-regularised logistic regression
    with balanced class weights. C is chosen from LP_GRID by patient-grouped inner cross-validation on the
    training images (the standard linear-probe protocol, without patient leakage).
    History: a first version used fixed C = 1.0 without standardisation; L-BFGS then stopped after 0 or 1
    iteration with near-zero weights (constant scores in one split). Found and fixed 25 Oct 2026."""
    X_train, X_eval, y_train = np.asarray(X_train), np.asarray(X_eval), np.asarray(y_train)
    if groups is None:
        groups = [_GROUPS.get(r.tobytes()) for r in X_train]
        assert None not in groups, "patient ids unknown for the linear probe: call register_groups first"
    groups = np.asarray(groups)
    Xtr, Xev = _norm(X_train), _norm(X_eval)
    C = choose_C(Xtr, y_train, groups)
    scaler, clf = _fit(Xtr, y_train, C)
    p = clf.predict_proba(scaler.transform(Xev))[:, 1]
    assert np.ptp(p) > 0, "linear probe gave constant scores"
    return p


def patch_registry():
    """Add 'Linear Probe' to the method registry of final_unified_eval (run once per split)."""
    import final_unified_eval as fue
    original = fue.registry

    def registry(model, text):
        d = original(model, text)
        d["Linear Probe"] = ("zs", lambda a, b, c, s: linear_probe_probs(a, b, c))
        return d
    fue.registry = registry
    fue.DETERMINISTIC.add("Linear Probe")
    original_eval = fue.evaluate

    def evaluate(fn, Xtr, ytr, Xva, pid_tr, folds, seed):
        register_groups(Xtr, pid_tr)              # lets the probe tune C with patient-grouped folds
        return original_eval(fn, Xtr, ytr, Xva, pid_tr, folds, seed)
    fue.evaluate = evaluate
    return fue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["main", "grouped"], required=True)
    a = ap.parse_args()
    fue = patch_registry()
    if a.design == "main":
        sys.argv = ["final_unified_eval.py", "--methods", "Linear Probe", "--out", "predictions_linear_probe.csv"]
        fue.main()
    else:
        import exp_B_grouped_cv as expb          # imported after the patch, so it sees the new method
        expb.registry = fue.registry
        expb.evaluate = fue.evaluate
        sys.argv = ["exp_B_grouped_cv.py", "--methods", "Linear Probe", "--out", "predictions_grouped_cv_lp.csv"]
        expb.main()


if __name__ == "__main__":
    main()
