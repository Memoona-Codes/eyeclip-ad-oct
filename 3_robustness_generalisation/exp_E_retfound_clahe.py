"""Experiment E: does the CLAHE gain hold for a second retinal foundation model (RETFound, OCT weights)?

Same data, same splits and same CLAHE settings as the paper (OpenCV, clipLimit 2.0, 8x8 tiles).
RETFound is used frozen. Two simple heads, both fixed in advance (no tuning):
  * Nearest Centroid (cosine), the paper's training-free method
  * Linear Probe (L2-normalised features, logistic regression C = 1.0, balanced), as in exp_D
Three feature read-outs are reported. The PRIMARY read-out is fixed in advance:
  * "mean"  : mean of final-layer patch tokens after the final norm (RETFound's own global pool)  <- primary
  * "cls"   : final-layer CLS token after the final norm
  * "block2": CLS token after transformer block 2 (best block in our earlier probing)

Designs:
  bissig : the paper's five patient-disjoint splits (cam.load_split, seeds 0 to 4).
           Slice AUC per split (mean and SD), patient AUC (mean slice score per patient, pooled over
           the 5 splits, 40 patient evaluations) and the number of splits where CLAHE beats raw.
  octdl  : the image list and 5 patient-grouped folds of exp_C (octdl_index_used.csv, seed 42).
           Pooled out-of-fold AUC, raw and CLAHE, with a patient-bootstrap CI of the difference.

Run from reset_v2_eyeclip/ (where exp_C was run, so octdl_index_used.csv is there):
    python moore_scripts/exp_E_retfound_clahe.py --ckpt /path/RETFound_oct_weights.pth --design bissig
    python moore_scripts/exp_E_retfound_clahe.py --ckpt /path/RETFound_oct_weights.pth --design octdl
Features are cached in retfound_feats_<design>.npz, so a rerun only redoes the heads.
Add --quick to use 40 images per split / 300 OCTDL images (pipeline check only).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import clahe_accuracy_full_metrics as cam  # noqa: E402
from clahe_accuracy_full_metrics import nc_probs  # noqa: E402
from exp_D_linear_probe import linear_probe_probs, register_groups  # noqa: E402

READOUTS = ["mean", "cls", "block2"]
HEADS = {"Nearest Centroid": nc_probs, "Linear Probe": linear_probe_probs}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])   # RETFound uses ImageNet stats


def build_retfound(ckpt, arch):
    import timm
    model = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="token")
    import argparse as _ap
    torch.serialization.add_safe_globals([_ap.Namespace])     # RETFound checkpoints store their training args
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith(("decoder", "mask_token", "head"))}
    if "pos_embed" in sd and sd["pos_embed"].shape != model.pos_embed.shape:
        raise SystemExit(f"pos_embed shape {tuple(sd['pos_embed'].shape)} does not match {arch}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"RETFound loaded from {ckpt}: {len(sd)} tensors, missing {missing}, unexpected {unexpected}")
    assert not [m for m in missing if m.startswith(("blocks", "patch_embed", "cls_token"))], \
        "backbone weights missing; check the checkpoint and --arch"
    return model.eval().to(cam.DEVICE)


TRANSFORM = "native"   # set in main(); "native" = RETFound's own eval transform, "stretch" = full image to 224 x 224
SUFFIX = ""


def load_img(path, use_clahe):
    img = Image.open(path)
    img = cam.apply_clahe(img) if use_clahe else img.convert("RGB")
    if TRANSFORM == "stretch":
        img = img.resize((224, 224), Image.BICUBIC)
    else:                                             # RETFound/MAE eval: shorter side to 256, centre crop 224
        w, h = img.size
        s = 256 / min(w, h)
        img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = img.size
        l, t = (w - 224) // 2, (h - 224) // 2
        img = img.crop((l, t, l + 224, t + 224))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32)


@torch.no_grad()
def extract(model, paths, use_clahe, bs=32):
    keep = {}
    h = model.blocks[2].register_forward_hook(lambda m, i, o: keep.__setitem__("b2", o[:, 0]))
    out = {r: [] for r in READOUTS}
    for i in range(0, len(paths), bs):
        x = torch.stack([load_img(p, use_clahe) for p in paths[i:i + bs]]).to(cam.DEVICE)
        tok = model.forward_features(x)                       # (B, 1 + patches, D), after final norm
        out["cls"].append(tok[:, 0].float().cpu())
        out["mean"].append(tok[:, model.num_prefix_tokens:].mean(1).float().cpu())
        out["block2"].append(keep["b2"].float().cpu())
    h.remove()
    return {r: torch.cat(v).numpy() for r, v in out.items()}


def cached(path, fn):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    d = fn()
    np.savez(path, **d)
    return d


def boot_diff(y, a, b, pid, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(y))).groupby(pid).apply(list).to_list()
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    return np.percentile(diffs, [2.5, 97.5])


def pid_of(path):
    return os.path.basename(path).split("_")[0]


def run_bissig(model, quick):
    splits = []
    for s in cam.SPLIT_SEEDS:
        trp, ytr, vap, yva = cam.load_split(s)
        assert not ({pid_of(p) for p in trp} & {pid_of(p) for p in vap}), f"patient leakage in split {s}"
        if quick:
            trp, ytr, vap, yva = trp[:40], ytr[:40], vap[:40], yva[:40]
        splits.append((s, trp, np.asarray(ytr), vap, np.asarray(yva)))
    paths = sorted({p for _, a, _, b, _ in splits for p in a + b})
    pos = {p: i for i, p in enumerate(paths)}
    print(f"Bissig: {len(paths)} unique images across {len(splits)} splits")

    def feats():
        d = {}
        for cond, use in [("raw", False), ("clahe", True)]:
            for r, X in extract(model, paths, use).items():
                d[f"{r}|{cond}"] = X
        return d
    F = cached(f"retfound_feats_bissig{SUFFIX}{'_quick' if quick else ''}.npz", feats)

    for k in F:
        register_groups(F[k], [pid_of(p) for p in paths])
    rows, pat_rows = [], []
    for r in READOUTS:
        for head, fn in HEADS.items():
            for cond in ("raw", "clahe"):
                X = F[f"{r}|{cond}"]
                for s, trp, ytr, vap, yva in splits:
                    sc = fn(X[[pos[p] for p in trp]], ytr, X[[pos[p] for p in vap]])
                    rows.append(dict(readout=r, method=head, preproc=cond, split=s, slice_auc=roc_auc_score(yva, sc)))
                    df = pd.DataFrame(dict(pid=[pid_of(p) for p in vap], y=yva, s=sc)).groupby("pid").mean()
                    for pid, row in df.iterrows():
                        pat_rows.append(dict(readout=r, method=head, preproc=cond, split=s, pid=pid, y=row.y, s=row.s))
    per = pd.DataFrame(rows)
    per.to_csv(f"retfound_bissig_per_split{SUFFIX}.csv", index=False)
    pat = pd.DataFrame(pat_rows)

    out = []
    for (r, head), g in per.groupby(["readout", "method"], sort=False):
        w = g.pivot(index="split", columns="preproc", values="slice_auc")
        pa = {c: roc_auc_score(p.y, p.s) for c, p in pat[(pat.readout == r) & (pat.method == head)].groupby("preproc")}
        out.append(dict(readout=r, method=head,
                        slice_raw=w.raw.mean(), slice_raw_sd=w.raw.std(),
                        slice_clahe=w.clahe.mean(), slice_clahe_sd=w.clahe.std(),
                        slice_gain=(w.clahe - w.raw).mean(), splits_improved=int((w.clahe > w.raw).sum()),
                        patient_raw=pa["raw"], patient_clahe=pa["clahe"]))
    return pd.DataFrame(out)


def run_octdl(model, index, folds_n, seed, quick):
    df = pd.read_csv(index)
    if quick:
        df = df.sample(n=min(300, len(df)), random_state=0)
    paths, y, pid = df["path"].tolist(), df["label"].to_numpy(), df["patient_id"].astype(str).to_numpy()
    print(f"OCTDL: {len(y)} images, {len(set(pid))} patients, {y.mean():.1%} disease")
    folds = list(StratifiedGroupKFold(n_splits=folds_n, shuffle=True, random_state=seed)
                 .split(np.zeros(len(y)), y, groups=pid))
    for tr, te in folds:
        assert not (set(pid[tr]) & set(pid[te])), "patient leakage"

    def feats():
        d = {}
        for cond, use in [("raw", False), ("clahe", True)]:
            for r, X in extract(model, paths, use).items():
                d[f"{r}|{cond}"] = X
        return d
    F = cached(f"retfound_feats_octdl{SUFFIX}{'_quick' if quick else ''}.npz", feats)

    for k in F:
        register_groups(F[k], pid)
    out, scores = [], {}
    for r in READOUTS:
        for head, fn in HEADS.items():
            for cond in ("raw", "clahe"):
                X, s = F[f"{r}|{cond}"], np.zeros(len(y))
                for tr, te in folds:
                    s[te] = fn(X[tr], y[tr], X[te])
                scores[(r, head, cond)] = s
            a, b = scores[(r, head, "raw")], scores[(r, head, "clahe")]
            lo, hi = boot_diff(y, a, b, pid)
            out.append(dict(readout=r, method=head, auc_raw=roc_auc_score(y, a), auc_clahe=roc_auc_score(y, b),
                            diff=roc_auc_score(y, b) - roc_auc_score(y, a), ci_low=lo, ci_high=hi,
                            significant=bool(lo > 0 or hi < 0)))
    np.savez(f"retfound_octdl_scores{SUFFIX}.npz", y=y, pid=pid, **{"|".join(k): v for k, v in scores.items()})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="vit_large_patch16_224")
    ap.add_argument("--design", choices=["bissig", "octdl"], required=True)
    ap.add_argument("--index", default="octdl_index_used.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--transform", choices=["native", "stretch"], default="native",
                    help="native: RETFound eval transform (resize 256, centre crop 224); stretch: first runs")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = build_retfound(a.ckpt, a.arch)
    global TRANSFORM, SUFFIX
    TRANSFORM, SUFFIX = a.transform, ("" if a.transform == "stretch" else "_native")
    res = run_bissig(model, a.quick) if a.design == "bissig" else run_octdl(model, a.index, a.folds, a.seed, a.quick)
    name = f"retfound_{a.design}_summary{SUFFIX}{'_quick' if a.quick else ''}.csv"
    res.to_csv(name, index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"saved {name}  (primary read-out = mean, transform = {TRANSFORM})")


if __name__ == "__main__":
    main()
