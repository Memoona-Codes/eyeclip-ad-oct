#!/usr/bin/env python
"""
analyze_predictions.py -- every results table and figure for the paper, from one CSV.

Input : predictions CSV(s) written by pred_logger.py (one row per validation slice per run)
Output: results_final/
    per_run_metrics.csv            every run, slice and patient level, all metrics
    table_slice_level.{csv,md,tex} mean±SD (+ 95% CI for AUC) per method x preproc
    table_patient_level.{...}      same at patient level (mean slice score per patient)
    table_raw_vs_clahe.{...}       paired raw-vs-CLAHE test per method, slice and patient
    table_top_methods.{...}        top methods, slice vs patient (paper Table 4 layout)
    table_mcnemar.{...}            pooled patient-level McNemar among top methods, Bonferroni
    table_dosage.{...}             only if the CSV has dosage > 0 rows
    confusion_matrices.csv         best method, raw and CLAHE, slice and patient
    failure_analysis.csv           per-patient error rate for the best method
    fig_roc_confusion.{pdf,png}    (a) slice ROC (b) patient ROC (c,d) CLAHE confusion matrices
    fig_confusion_raw_vs_clahe.{pdf,png}
    summary.md                     everything above in one readable file

Usage
-----
python analyze_predictions.py --pred predictions_final.csv --out results_final
python analyze_predictions.py --pred predictions_final.csv --best "TaskRes+SAM" --top "TaskRes+SAM" "CLIP-Adapter" "Proto-Adapter"
python analyze_predictions.py --demo            # runs on simulated data to check the pipeline

Statistics used
---------------
* 95% CI: patient-cluster bootstrap. Within each split, validation patients are resampled with
  replacement (stratified by class); all seeds of that split use the same resample; the metric is
  averaged over all runs. Repeated n_boot times; percentile interval.
* Raw vs CLAHE: paired over matched (split, seed) runs.
  - corrected resampled t-test (Nadeau & Bengio 2003): var * (1/n + n_val/n_train), df = n-1
  - Wilcoxon signed-rank on the 5 per-split means (smallest possible two-sided p = 0.0625)
* McNemar: exact binomial on discordant patients, pooled over splits. Each method's patient call
  is the majority over model seeds (ties -> seed-mean score vs seed-mean threshold).
"""
import argparse
import itertools
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------ colours (validated pair)
C_CLAHE = "#2a78d6"   # categorical slot 1
C_RAW = "#eb6834"     # categorical slot 2
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

METRICS_SLICE = ["AUC", "AUPR", "Acc", "BalAcc", "F1", "Sens", "Spec", "PPV", "NPV", "MCC", "ECE"]
METRICS_PAT = ["AUC", "AUPR", "Acc", "BalAcc", "F1", "Sens", "Spec", "PPV", "NPV", "MCC"]


# ================================================================== metrics
def fast_auc(y, s):
    """Mann-Whitney AUC with tie handling. y in {0,1}."""
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def aupr(y, s):
    """Average precision (step-wise area under precision-recall)."""
    y = np.asarray(y); s = np.asarray(s, float)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    prec = tp / k
    return float((prec * y).sum() / y.sum())


def ece(y, p, bins=10):
    p = np.asarray(p, float)
    if p.min() < 0 or p.max() > 1:
        return np.nan  # scores are not probabilities
    y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def threshold_metrics(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return dict(Acc=(tp + tn) / n, BalAcc=np.nanmean([sens, spec]), F1=f1, Sens=sens, Spec=spec,
                PPV=ppv, NPV=npv, MCC=mcc, TP=tp, FP=fp, TN=tn, FN=fn)


def all_metrics(y, s, thr, with_ece):
    pred = (np.asarray(s) >= thr).astype(int)
    m = dict(AUC=fast_auc(y, s), AUPR=aupr(y, s))
    m.update(threshold_metrics(y, pred))
    if with_ece:
        m["ECE"] = ece(y, s)
    m["n"] = len(y)
    m["collapsed"] = int(m["TP"] + m["FP"] == 0 or m["TN"] + m["FN"] == 0)  # one class predicted
    return m


# ================================================================== loading
REQUIRED = ["method", "preproc", "split", "seed", "slice_id", "patient_id", "y_true", "score", "threshold"]


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"missing columns: {miss}")
    if "dosage" not in df:
        df["dosage"] = 0
    if "patient_threshold" not in df:
        df["patient_threshold"] = df["threshold"]
    df["patient_threshold"] = df["patient_threshold"].fillna(df["threshold"])
    df["preproc"] = df["preproc"].str.lower()
    key = ["method", "preproc", "dosage", "split", "seed", "slice_id"]
    if "run_time" in df:
        df = df.sort_values("run_time")
    before = len(df)
    df = df.drop_duplicates(key, keep="last").reset_index(drop=True)
    if len(df) < before:
        print(f"  dropped {before - len(df)} duplicate rows (kept newest)")
    # label consistency per patient
    bad = df.groupby("patient_id")["y_true"].nunique()
    if (bad > 1).any():
        sys.exit(f"patients with mixed labels: {list(bad[bad > 1].index)}")
    return df


RUN = ["method", "preproc", "dosage", "split", "seed"]
GROUP = ["method", "preproc", "dosage"]


def patient_table(run_df):
    g = run_df.groupby("patient_id").agg(y=("y_true", "first"), s=("score", "mean"),
                                         thr=("patient_threshold", "first"))
    return g


def per_run(df):
    rows = []
    for key, r in df.groupby(RUN, sort=False):
        base = dict(zip(RUN, key))
        ms = all_metrics(r.y_true.values, r.score.values, r.threshold.iloc[0], True)
        rows.append({**base, "level": "slice", **ms})
        p = patient_table(r)
        mp = all_metrics(p.y.values, p.s.values, p.thr.iloc[0], False)
        rows.append({**base, "level": "patient", **mp})
    return pd.DataFrame(rows)


# ================================================================== bootstrap CI
def bootstrap_ci(df, level, n_boot, rng, metric="AUC"):
    """Patient-cluster bootstrap of the grand-mean metric for each (method, preproc, dosage)."""
    out = {}
    for gkey, g in df.groupby(GROUP, sort=False):
        # per split: patients by class, per run arrays
        split_info = []
        for sp, gs in g.groupby("split"):
            pats = gs.groupby("patient_id")["y_true"].first()
            ad = pats[pats == 1].index.values; co = pats[pats == 0].index.values
            runs = []
            for seed, gr in gs.groupby("seed"):
                if level == "slice":
                    by_p = {pid: (sub.y_true.values, sub.score.values) for pid, sub in gr.groupby("patient_id")}
                    runs.append(("slice", by_p, gr.threshold.iloc[0]))
                else:
                    pt = patient_table(gr)
                    runs.append(("patient", pt, pt.thr.iloc[0]))
            split_info.append((ad, co, runs))
        vals = np.empty(n_boot)
        for b in range(n_boot):
            acc = []
            for ad, co, runs in split_info:
                samp = np.concatenate([rng.choice(ad, len(ad)), rng.choice(co, len(co))])
                for kind, data, thr in runs:
                    if kind == "slice":
                        y = np.concatenate([data[p][0] for p in samp])
                        s = np.concatenate([data[p][1] for p in samp])
                    else:
                        y = data.loc[samp, "y"].values; s = data.loc[samp, "s"].values
                    if metric == "AUC":
                        acc.append(fast_auc(y, s))
                    else:
                        acc.append(((s >= thr).astype(int) == y).mean())
            vals[b] = np.nanmean(acc)
        out[gkey] = (np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5))
    return out


# ================================================================== summary tables
def fmt(m, s, d=3):
    if pd.isna(m):
        return "–"
    return f"{m:.{d}f} ± {s:.{d}f}" if not pd.isna(s) else f"{m:.{d}f}"


def summary_table(pr, level, metrics, ci):
    rows = []
    sub = pr[pr.level == level]
    for gkey, g in sub.groupby(GROUP, sort=False):
        row = dict(zip(GROUP, gkey))
        row["runs"] = len(g)
        for m in metrics:
            if m in g and g[m].notna().any():
                row[m] = fmt(g[m].mean(), g[m].std(ddof=1))
                row[m + "_mean"] = g[m].mean()
        if gkey in ci:
            lo, hi = ci[gkey]
            row["AUC 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        row["collapsed runs"] = int(g["collapsed"].sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    t = t.sort_values(["dosage", "preproc", "AUC_mean"], ascending=[True, True, False])
    return t


def corrected_ttest(d, n_val, n_train):
    d = np.asarray(d, float); n = len(d)
    if n < 2 or np.var(d, ddof=1) == 0:
        return np.nan, np.nan
    v = np.var(d, ddof=1) * (1 / n + n_val / n_train)
    t = d.mean() / math.sqrt(v)
    p = 2 * stats.t.sf(abs(t), n - 1)
    return t, p


def raw_vs_clahe(pr, df):
    rows = []
    n_val = df.groupby("split").patient_id.nunique().mean()
    n_train = df.patient_id.nunique() - n_val
    sub = pr[pr.dosage == 0]
    for (method, level), g in sub.groupby(["method", "level"]):
        a = g[g.preproc == "raw"].set_index(["split", "seed"])["AUC"]
        b = g[g.preproc == "clahe"].set_index(["split", "seed"])["AUC"]
        idx = a.index.intersection(b.index)
        if len(idx) < 2:
            continue
        d = (b.loc[idx] - a.loc[idx]).values
        t, p = corrected_ttest(d, n_val, n_train)
        per_split = (b.loc[idx] - a.loc[idx]).groupby(level="split").mean()
        try:
            wp = stats.wilcoxon(per_split.values).pvalue if len(per_split) >= 2 and np.any(per_split != 0) else np.nan
        except ValueError:
            wp = np.nan
        rows.append({"method": method, "level": level,
                     "AUC raw": fmt(a.loc[idx].mean(), a.loc[idx].std(ddof=1)),
                     "AUC CLAHE": fmt(b.loc[idx].mean(), b.loc[idx].std(ddof=1)),
                     "ΔAUC": f"{d.mean():+.4f}",
                     "splits improved": f"{int((per_split > 0).sum())}/{len(per_split)}",
                     "corrected t p": ("<0.0001" if p < 1e-4 else f"{p:.4f}") if not np.isnan(p) else "–",
                     "Wilcoxon p (splits)": f"{wp:.4f}" if not np.isnan(wp) else "–",
                     "matched runs": len(idx), "_d": d.mean()})
    t = pd.DataFrame(rows)
    return t.sort_values(["level", "_d"], ascending=[False, False]).drop(columns="_d") if len(t) else t


def patient_calls(df, method, preproc, dosage=0):
    """Seed-majority patient call per (split, patient)."""
    sub = df[(df.method == method) & (df.preproc == preproc) & (df.dosage == dosage)]
    rows = []
    for (sp, seed), r in sub.groupby(["split", "seed"]):
        pt = patient_table(r)
        pt["pred"] = (pt.s >= pt.thr).astype(int)
        pt["split"] = sp; pt["seed"] = seed
        rows.append(pt.reset_index())
    a = pd.concat(rows)
    agg = a.groupby(["split", "patient_id"]).agg(y=("y", "first"), votes=("pred", "mean"),
                                                 s=("s", "mean"), thr=("thr", "mean"))
    agg["pred"] = np.where(agg.votes > 0.5, 1, np.where(agg.votes < 0.5, 0, (agg.s >= agg.thr).astype(int)))
    return agg


def slice_calls(df, method, preproc, dosage=0):
    sub = df[(df.method == method) & (df.preproc == preproc) & (df.dosage == dosage)].copy()
    sub["pred"] = (sub.score >= sub.threshold).astype(int)
    agg = sub.groupby(["split", "slice_id"]).agg(y=("y_true", "first"), votes=("pred", "mean"),
                                                 s=("score", "mean"), thr=("threshold", "mean"))
    agg["pred"] = np.where(agg.votes > 0.5, 1, np.where(agg.votes < 0.5, 0, (agg.s >= agg.thr).astype(int)))
    return agg


def mcnemar_table(df, methods, preproc="clahe"):
    calls = {m: patient_calls(df, m, preproc) for m in methods}
    pairs = list(itertools.combinations(methods, 2))
    K = len(pairs)
    rows = []
    for a, b in pairs:
        j = calls[a][["y", "pred"]].join(calls[b][["pred"]], rsuffix="_b", how="inner")
        ca = j.pred == j.y; cb = j.pred_b == j.y
        n11 = int((ca & cb).sum()); n00 = int((~ca & ~cb).sum())
        b_ = int((ca & ~cb).sum()); c_ = int((~ca & cb).sum())
        nd = b_ + c_
        p = stats.binomtest(b_, nd, 0.5).pvalue if nd else 1.0
        rows.append({"Method A": a, "Method B": b, "both correct": n11, "A only": b_, "B only": c_,
                     "both wrong": n00, "discordant": nd, "p raw": f"{p:.4f}",
                     "p Bonferroni": f"{min(1, K * p):.4f}", "significant (0.05)": "yes" if K * p < 0.05 else "no"})
    return pd.DataFrame(rows)


def confusion(calls):
    y = calls.y.values; p = calls.pred.values
    return dict(TP=int(((p == 1) & (y == 1)).sum()), FN=int(((p == 0) & (y == 1)).sum()),
                FP=int(((p == 1) & (y == 0)).sum()), TN=int(((p == 0) & (y == 0)).sum()))


def failure_analysis(df, method, preproc="clahe"):
    sub = df[(df.method == method) & (df.preproc == preproc) & (df.dosage == 0)]
    rows = []
    for (sp, seed), r in sub.groupby(["split", "seed"]):
        pt = patient_table(r)
        pt["wrong"] = ((pt.s >= pt.thr).astype(int) != pt.y).astype(int)
        rows.append(pt.reset_index()[["patient_id", "y", "wrong"]])
    a = pd.concat(rows)
    t = a.groupby("patient_id").agg(label=("y", lambda v: "AD" if v.iloc[0] == 1 else "CO"),
                                    evaluations=("wrong", "size"), errors=("wrong", "sum"))
    t["error rate"] = (t.errors / t.evaluations).round(3)
    return t.sort_values(["error rate", "evaluations"], ascending=[False, False]).reset_index()


# ================================================================== output helpers
def to_latex(t, caption, label):
    cols = [c for c in t.columns if not c.endswith("_mean")]
    esc = lambda x: str(x).replace("±", r"$\pm$").replace("%", r"\%").replace("_", r"\_").replace("Δ", r"$\Delta$")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in t[cols].iterrows():
        lines.append(" & ".join(esc(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def save_table(t, out, name, caption):
    if t is None or len(t) == 0:
        return ""
    show = t[[c for c in t.columns if not c.endswith("_mean")]]
    show.to_csv(out / f"{name}.csv", index=False)
    try:
        md = show.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        cols = list(show.columns)
        md = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
            "| " + " | ".join(map(str, r)) + " |" for r in show.values)
    (out / f"{name}.md").write_text(md + "\n")
    (out / f"{name}.tex").write_text(to_latex(show, caption, "tab:" + name) + "\n")
    return f"## {caption}\n\n{md}\n\n"


# ================================================================== figures
def mean_roc(df, method, preproc, level, grid=np.linspace(0, 1, 201), dosage=0):
    from sklearn.metrics import roc_curve
    sub = df[(df.method == method) & (df.preproc == preproc) & (df.dosage == dosage)]
    tprs, aucs = [], []
    for _, r in sub.groupby(["split", "seed"]):
        if level == "slice":
            y, s = r.y_true.values, r.score.values
        else:
            pt = patient_table(r); y, s = pt.y.values, pt.s.values
        fpr, tpr, _ = roc_curve(y, s)
        t = np.interp(grid, fpr, tpr); t[0] = 0.0
        tprs.append(t); aucs.append(fast_auc(y, s))
    tprs = np.array(tprs)
    m = tprs.mean(0); m[-1] = 1.0
    return grid, m, tprs.std(0), np.mean(aucs), np.std(aucs, ddof=1)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def plot_roc(ax, df, method, level, title):
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 3)), label="Chance")
    for pre, col, name in (("raw", C_RAW, "Raw"), ("clahe", C_CLAHE, "CLAHE")):
        if not ((df.method == method) & (df.preproc == pre)).any():
            continue
        g, m, sd, a, asd = mean_roc(df, method, pre, level)
        ax.fill_between(g, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), color=col, alpha=0.15, lw=0)
        ax.plot(g, m, color=col, lw=2, label=f"{name}  AUC {a:.3f} ± {asd:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate (1 − specificity)", fontsize=9, color=INK)
    ax.set_ylabel("True-positive rate (sensitivity)", fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK)


def plot_cm(ax, cm, title):
    import matplotlib.colors as mcolors
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    row = mat.sum(1, keepdims=True)
    pct = np.divide(mat, row, out=np.zeros_like(mat, float), where=row > 0)
    cmap = mcolors.LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc0ee", C_CLAHE, "#10407a"])
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CO", "AD"], fontsize=8, color=INK2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["CO", "AD"], fontsize=8, color=INK2)
    ax.set_xlabel("Predicted", fontsize=9, color=INK); ax.set_ylabel("True", fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=2)


def make_figures(df, best, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 8.2))
    plot_roc(axs[0, 0], df, best, "slice", f"(a) Slice level: {best}")
    plot_roc(axs[0, 1], df, best, "patient", f"(b) Patient level: {best}")
    n_splits = df.split.nunique()
    plot_cm(axs[1, 0], confusion(slice_calls(df, best, "clahe")), f"(c) Slices, CLAHE ({n_splits} splits)")
    plot_cm(axs[1, 1], confusion(patient_calls(df, best, "clahe")),
            f"(d) Patients, CLAHE ({len(patient_calls(df, best, 'clahe'))} evaluations)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_roc_confusion.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if (df.preproc == "raw").any():
        fig, axs = plt.subplots(2, 2, figsize=(7, 6.6))
        for i, lvl in enumerate(("slice", "patient")):
            for j, pre in enumerate(("raw", "clahe")):
                f = slice_calls if lvl == "slice" else patient_calls
                plot_cm(axs[i, j], confusion(f(df, best, pre)), f"{lvl.title()}, {'Raw' if pre == 'raw' else 'CLAHE'}")
        fig.suptitle(best, fontsize=10, color=INK, x=0.02, ha="left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_confusion_raw_vs_clahe.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # dose figure: real, +500, +1000 for the best method, one figure per preprocessing
    doses = sorted(df[df.method == best].dosage.unique())
    if len(doses) > 1:
        cmaps = {"clahe": "Blues", "raw": "Oranges"}
        for pre in sorted(df[(df.method == best) & (df.dosage > 0)].preproc.unique()):
            fig, axs = plt.subplots(3, len(doses), figsize=(3.2 * len(doses), 9.6))
            for j, lvl in enumerate(("slice", "patient")):
                ax = axs[0, j]; style_axes(ax)
                ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 3)), label="Chance")
                for dose, col in zip(doses, [plt.get_cmap(cmaps[pre])(v) for v in np.linspace(0.95, 0.4, len(doses))]):
                    if not ((df.method == best) & (df.preproc == pre) & (df.dosage == dose)).any():
                        continue
                    g, m, sd, a, asd = mean_roc(df, best, pre, lvl, dosage=dose)
                    ax.plot(g, m, color=col, lw=2, label=f"{'Real' if dose == 0 else f'Real + {dose}'}  AUC {a:.3f}")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
                ax.set_xlabel("1 − specificity", fontsize=9, color=INK)
                ax.set_ylabel("Sensitivity", fontsize=9, color=INK)
                ax.set_title(f"({'ab'[j]}) ROC, {lvl} level", fontsize=10, color=INK, loc="left")
                ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=INK)
            for j in range(2, len(doses)):
                axs[0, j].axis("off")
            letters = iter("cdefghijklmnopqrstuvwxyz")
            for i, (lvl, f) in enumerate((("slice", slice_calls), ("patient", patient_calls)), start=1):
                for j, dose in enumerate(doses):
                    name = "Real" if dose == 0 else f"Real + {dose}"
                    plot_cm(axs[i, j], confusion(f(df, best, pre, dose)), f"({next(letters)}) {lvl.title()}, {name}")
            fig.suptitle(f"{best}, {'CLAHE' if pre == 'clahe' else 'Raw'}", fontsize=10, color=INK, x=0.02, ha="left")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out / f"fig_dosage_roc_confusion_{pre}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)


# ================================================================== demo data
def make_demo(path, rng):
    """28 patients (14 AD/14 CO), ~10 slices each, 5 splits x 5 seeds, 4 methods, raw/CLAHE, dosage."""
    pats = [f"AD{100 + i}" for i in range(14)] + [f"CO{200 + i}" for i in range(14)]
    ylab = {p: int(p.startswith("AD")) for p in pats}
    nsl = {p: int(rng.integers(9, 11)) for p in pats}
    effect = {p: rng.normal(0, 0.9) for p in pats}  # patient-level difficulty
    methods = {"TaskRes+SAM": 1.05, "CLIP-Adapter": 1.07, "Proto-Adapter": 1.02, "Zero-Shot": 0.95}
    rows = []
    for sp in range(5):
        r = np.random.default_rng(sp)
        ad = r.permutation([p for p in pats if ylab[p]]); co = r.permutation([p for p in pats if not ylab[p]])
        val = list(ad[:4]) + list(co[:4])
        for m, strength in methods.items():
            for pre, boost in (("raw", 0.0), ("clahe", 0.25)):
                seeds = [0] if m == "Zero-Shot" else [42, 100, 2024, 777, 999]
                for dosage in ([0, 500, 1000] if m == "TaskRes+SAM" else [0]):
                    for sd in seeds:
                        rs = np.random.default_rng(zlib.crc32(f"{sp}|{m}|{pre}|{sd}|{dosage}".encode()))
                        for p in val:
                            for k in range(nsl[p]):
                                mu = (strength + boost - dosage / 5000) * (2 * ylab[p] - 1) * 0.6 + effect[p] * 0.5
                                s = 1 / (1 + np.exp(-(mu + rs.normal(0, 1.0))))
                                cond = "DARK" if k < nsl[p] // 2 else "LIGHT"
                                rows.append(["demo", m, pre, dosage, sp, sd, f"{p}_{cond}_s{k}.png", p, cond,
                                             ylab[p], s, 0.5, 0.5])
    cols = ["run_time", "method", "preproc", "dosage", "split", "seed", "slice_id", "patient_id",
            "condition", "y_true", "score", "threshold", "patient_threshold"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", help="prediction CSV(s) from pred_logger.py")
    ap.add_argument("--out", default="results_final")
    ap.add_argument("--best", default=None, help="method for ROC/confusion/failure analysis (default: top CLAHE slice AUC)")
    ap.add_argument("--top", nargs="*", default=None, help="methods for McNemar (default: top 3 CLAHE slice AUC)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--demo", action="store_true", help="simulate data and run the whole pipeline")
    a = ap.parse_args()
    rng = np.random.default_rng(12345)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.demo:
        a.pred = [str(make_demo(out / "demo_predictions.csv", rng))]
        print(f"demo data written to {a.pred[0]}")
    if not a.pred:
        sys.exit("give --pred <csv> (or --demo)")

    print("loading ...")
    df = load(a.pred)
    print(f"  {len(df):,} rows | methods: {sorted(df.method.unique())} | preproc: {sorted(df.preproc.unique())} | "
          f"dosage: {sorted(df.dosage.unique())} | splits: {sorted(df.split.unique())}")
    chk = df.groupby(["split"]).patient_id.nunique()
    print(f"  validation patients per split: {dict(chk)}")

    print("per-run metrics ...")
    pr = per_run(df)
    pr.to_csv(out / "per_run_metrics.csv", index=False)

    print(f"bootstrap CIs ({a.n_boot} resamples, patient-cluster) ...")
    ci_s = bootstrap_ci(df, "slice", a.n_boot, rng)
    ci_p = bootstrap_ci(df, "patient", a.n_boot, rng)

    t_slice = summary_table(pr, "slice", METRICS_SLICE, ci_s)
    t_pat = summary_table(pr, "patient", METRICS_PAT, ci_p)

    base = t_slice[(t_slice.preproc == "clahe") & (t_slice.dosage == 0)]
    if base.empty:
        base = t_slice[t_slice.dosage == 0]
    ranked = [m for m in base.sort_values("AUC_mean", ascending=False).method.tolist() if "pool]" not in m]
    best = a.best or ranked[0]
    top = a.top or ranked[:3]
    print(f"  best method: {best} | McNemar set: {top}")

    md = [f"# Results summary\n\nSource: {', '.join(a.pred)}  \nRows: {len(df):,}  \n"
          f"Best method (figures, failure analysis): **{best}**\n\n"]
    bench = lambda t: t[(t.dosage == 0) & ~t.method.str.contains("pool]", regex=False)]
    main_s = bench(t_slice); main_p = bench(t_pat)
    md.append(save_table(main_s, out, "table_slice_level", "Slice-level results (mean ± SD over runs; 95% CI by patient-cluster bootstrap)"))
    md.append(save_table(main_p, out, "table_patient_level", "Patient-level results (mean slice score per patient)"))

    rvc = raw_vs_clahe(pr, df)
    md.append(save_table(rvc, out, "table_raw_vs_clahe", "Raw vs CLAHE, paired over matched (split, seed) runs"))

    # paper Table 4 layout
    rows = []
    for m in top:
        for lvl, t in (("Slice", main_s), ("Patient", main_p)):
            r = t[(t.method == m) & (t.preproc == "clahe")]
            if len(r):
                r = r.iloc[0]
                rows.append({"Method": m, "Level": lvl, **{k: r.get(k, "–") for k in ["AUC", "AUC 95% CI", "Acc", "BalAcc", "F1", "Sens", "Spec", "MCC"]}})
    md.append(save_table(pd.DataFrame(rows), out, "table_top_methods", "Top methods after CLAHE, slice vs patient level"))

    if len(top) >= 2:
        md.append(save_table(mcnemar_table(df, top), out, "table_mcnemar",
                             "Pooled patient-level McNemar tests (seed-majority calls), Bonferroni-corrected"))

    if df.dosage.nunique() > 1:
        dz = []
        for lvl, t in (("slice", t_slice), ("patient", t_pat)):
            combos = df[df.dosage > 0][["method", "preproc"]].drop_duplicates()
            d = t.merge(combos, on=["method", "preproc"]).copy()
            d.insert(0, "level", lvl)
            dz.append(d[["level", "method", "preproc", "dosage", "runs", "AUC", "AUC 95% CI", "Acc", "BalAcc",
                         "F1", "Sens", "Spec", "MCC", "collapsed runs"]])
        dz = pd.concat(dz).sort_values(["level", "method", "preproc", "dosage"], ascending=[False, True, True, True])
        md.append(save_table(dz, out, "table_dosage", "Synthetic dosage (real validation only)"))

        # confusion matrices for every method, preprocessing and dose (seed-majority calls summed over splits)
        rows = []
        for (m, pre) in df[df.dosage > 0][["method", "preproc"]].drop_duplicates().itertuples(index=False):
            for dose in sorted(df[(df.method == m) & (df.preproc == pre)].dosage.unique()):
                for lvl, f in (("slice", slice_calls), ("patient", patient_calls)):
                    rows.append({"method": m, "preproc": pre, "dosage": dose, "level": lvl,
                                 **confusion(f(df, m, pre, dose))})
        cmd = pd.DataFrame(rows).sort_values(["level", "method", "preproc", "dosage"], ascending=[False, True, True, True])
        md.append(save_table(cmd, out, "confusion_dosage", "Confusion matrices at each dose (TN, FP, FN, TP)"))

    cms = []
    for pre in sorted(df[df.method == best].preproc.unique()):
        for lvl, f in (("slice", slice_calls), ("patient", patient_calls)):
            cms.append({"method": best, "preproc": pre, "level": lvl, **confusion(f(df, best, pre))})
    cm = pd.DataFrame(cms)
    md.append(save_table(cm, out, "confusion_matrices", f"Confusion matrices, {best} (seed-majority calls summed over splits)"))

    fa = failure_analysis(df, best)
    fa.to_csv(out / "failure_analysis.csv", index=False)
    hard = fa[fa["error rate"] > 0.5]
    md.append(f"## Failure analysis ({best}, CLAHE, patient level)\n\n"
              f"{len(hard)} of {len(fa)} patients misclassified in more than half of their evaluations"
              f"{': ' + ', '.join(f'{r.patient_id} ({r.label}, {r.errors}/{r.evaluations})' for r in hard.itertuples()) if len(hard) else ''}.\n\n")

    (out / "summary.md").write_text("".join(md))
    print("figures ...")
    try:
        make_figures(df, best, out)
    except Exception as e:
        print(f"  figure step failed ({type(e).__name__}: {e}); tables and summary.md are saved")
    md.append("## Figures\n\n- fig_roc_confusion.pdf / .png\n- fig_confusion_raw_vs_clahe.pdf / .png\n"
              + ("- fig_dosage_roc_confusion_clahe / _raw .pdf / .png (real, +500, +1000)\n" if df.dosage.nunique() > 1 else ""))
    (out / "summary.md").write_text("".join(md))
    print(f"done -> {out.resolve()}")
    for f in sorted(out.iterdir()):
        print("   ", f.name)


if __name__ == "__main__":
    main()
