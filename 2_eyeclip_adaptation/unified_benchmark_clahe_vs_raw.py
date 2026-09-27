"""
Unified benchmark: ALL 9 methods on the SAME splits, SAME features, SAME
threshold rule -- run twice, once on CLAHE images and once on raw images.

Why: the older Zero-Shot / Tip-Adapter / Tip-Adapter-F / CLIP-Adapter / CoOp
results used different data (some uncropped), different in-memory splits
(seeds 1-5 or 0-4, rebuilt on the fly) and a Youden threshold, so they can't
be compared with the CLAHE table. Here every method uses:
  - splits: OCT_PNG_rebuilt_ADCO_slices_cropped_CLEAN_SPLIT_seed{0-4} (on disk)
  - features: identical extraction functions imported from
    clahe_accuracy_full_metrics.py (so the 4 existing methods reproduce exactly)
  - threshold: balanced (sens = spec) chosen on TRAIN only
  - one training run per split, model seed 42

Methods re-implemented from the original scripts (same hyperparameters):
  Zero-Shot, Tip-Adapter (beta=20, alpha=1)  <- eyeclip_zeroshot_tipadapter_clean_repeated.py
  Tip-Adapter-F (AdamW lr=0.01, wd=1e-4, 150 full-batch steps) <- same script
  CLIP-Adapter (ratio 0.2, 512->128->512, AdamW 1e-4, 50 ep, bs 16, 5-ep warmup)
      <- eyeclip_clip_adapter_repeated_splits.py
      (trained on cached frozen features: identical to the original because the
       backbone is frozen and in eval mode)
  CoOp (16 ctx tokens, AdamW 2e-3, 100 full-batch epochs)
      <- eyeclip_coop_clean_repeated_CORRECTED_SPLITS.py

Run from reset_v2_eyeclip/ (it imports clahe_accuracy_full_metrics.py):
  python -u unified_benchmark_clahe_vs_raw.py 2>&1 | tee unified_benchmark_log.txt
"""

import csv
import numpy as np
import torch
import torch.nn as nn
import clip
from sklearn.metrics import roc_auc_score

from clahe_accuracy_full_metrics import (
    DEVICE, SPLIT_SEEDS, EYECLIP_CKPT, MODEL_SEED,
    load_split, build_eyeclip, extract_layers7_10, extract_zeroshot,
    get_zeroshot_text_embeddings, balanced_threshold_eval,
    nc_probs, proto_adapter_probs, taskres_probs,
)

CONDITIONS = [("CLAHE", True), ("RAW", False)]

# Tip-Adapter / Tip-Adapter-F
TIP_BETA, TIP_ALPHA = 20, 1.0
TIPF_LR, TIPF_WD, TIPF_STEPS = 0.01, 1e-4, 150
# CLIP-Adapter
CA_RATIO, CA_BOTTLENECK, CA_LR, CA_EPOCHS, CA_BS, CA_WARMUP = 0.2, 128, 1e-4, 50, 16, 5
# CoOp
COOP_NCTX, COOP_LR, COOP_EPOCHS = 16, 2e-3, 100
COOP_CLASS_NAMES = ["healthy control", "Alzheimer's disease"]   # index 0=CO, 1=AD

METHODS = ["Zero-Shot", "Tip-Adapter", "Tip-Adapter-F", "CLIP-Adapter", "CoOp",
           "Nearest Centroid", "Proto-Adapter", "TaskRes", "TaskRes+SAM"]


def t(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


# ---------------- Zero-Shot ----------------
def zeroshot_probs(text, Xtr, Xva):
    with torch.no_grad():
        f = lambda X: torch.softmax(100 * (t(X) @ text.T), dim=1)[:, 1].cpu().numpy()
        return f(Xtr), f(Xva)


# ---------------- Tip-Adapter / Tip-Adapter-F ----------------
def _tip_logits(feats, keys, text, cache_values, class_counts):
    zs = feats @ text.T
    aff = torch.exp(-TIP_BETA * (1 - feats @ keys.T))
    return zs + TIP_ALPHA * (aff @ cache_values) / class_counts


def _tip_setup(ytr):
    y = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    cache_values = torch.zeros(len(ytr), 2, device=DEVICE)
    cache_values.scatter_(1, y.unsqueeze(1), 1.0)
    class_counts = torch.tensor([(y == 0).sum().item(), (y == 1).sum().item()],
                                dtype=torch.float32, device=DEVICE)
    return y, cache_values, class_counts


def tip_probs(text, Xtr, ytr, Xva):
    _, cv, cc = _tip_setup(ytr)
    keys = t(Xtr)
    with torch.no_grad():
        f = lambda X: torch.softmax(_tip_logits(t(X), keys, text, cv, cc), 1)[:, 1].cpu().numpy()
        return f(Xtr), f(Xva)


def tip_f_probs(text, Xtr, ytr, Xva, seed=MODEL_SEED):
    torch.manual_seed(seed)
    y, cv, cc = _tip_setup(ytr)
    Xtr_t = t(Xtr)
    keys = nn.Parameter(Xtr_t.clone())
    opt = torch.optim.AdamW([keys], lr=TIPF_LR, weight_decay=TIPF_WD)
    for _ in range(TIPF_STEPS):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(_tip_logits(Xtr_t, keys, text, cv, cc), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        f = lambda X: torch.softmax(_tip_logits(t(X), keys, text, cv, cc), 1)[:, 1].cpu().numpy()
        return f(Xtr), f(Xva)


# ---------------- CLIP-Adapter ----------------
class CLIPAdapterHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(512, CA_BOTTLENECK), nn.ReLU(inplace=True),
                                     nn.Linear(CA_BOTTLENECK, 512))
        self.head = nn.Linear(512, 2)

    def forward(self, feats):                      # feats already L2-normalised
        x = CA_RATIO * self.adapter(feats) + (1 - CA_RATIO) * feats
        return self.head(x / x.norm(dim=-1, keepdim=True))


def clip_adapter_probs(Xtr, ytr, Xva, seed=MODEL_SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    net = CLIPAdapterHead().to(DEVICE)
    Xtr_t, y = t(Xtr), torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=CA_LR)
    n = len(Xtr_t)
    warmup_steps = max(1, n // CA_BS) * CA_WARMUP
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup_steps) if s < warmup_steps else 1.0)
    net.train()
    for _ in range(CA_EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - CA_BS + 1, CA_BS):
            idx = perm[i:i + CA_BS]
            opt.zero_grad()
            nn.functional.cross_entropy(net(Xtr_t[idx]), y[idx]).backward()
            opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        f = lambda X: torch.softmax(net(t(X)), 1)[:, 1].cpu().numpy()
        return f(Xtr), f(Xva)


# ---------------- CoOp ----------------
class PromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx = torch.empty(COOP_NCTX, ctx_dim)
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)
        prompts = [" ".join(["X"] * COOP_NCTX) + " " + n + "." for n in COOP_CLASS_NAMES]
        tok = clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.token_embedding(tok).float()
        self.register_buffer("prefix", emb[:, :1, :])
        self.register_buffer("suffix", emb[:, 1 + COOP_NCTX:, :])
        self.tok = tok

    def encode(self, m):
        ctx = self.ctx.unsqueeze(0).expand(len(COOP_CLASS_NAMES), -1, -1)
        x = torch.cat([self.prefix, ctx, self.suffix], dim=1) + m.positional_embedding.float()
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = m.ln_final(x).float()
        x = x[torch.arange(x.shape[0]), self.tok.argmax(-1)] @ m.text_projection.float()
        return x / x.norm(dim=-1, keepdim=True)


def coop_probs(clip_model, Xtr, ytr, Xva, seed=MODEL_SEED):
    torch.manual_seed(seed)
    pl = PromptLearner(clip_model).to(DEVICE)
    opt = torch.optim.AdamW([pl.ctx], lr=COOP_LR)
    Xtr_t, y = t(Xtr), torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    scale = clip_model.logit_scale.exp().item()
    for _ in range(COOP_EPOCHS):
        opt.zero_grad()
        nn.functional.cross_entropy(scale * Xtr_t @ pl.encode(clip_model).T, y).backward()
        opt.step()
    with torch.no_grad():
        text = pl.encode(clip_model)
        f = lambda X: torch.softmax(scale * t(X) @ text.T, 1)[:, 1].cpu().numpy()
        return f(Xtr), f(Xva)


# ---------------- main ----------------
def checkpoint_report():
    sd = torch.load(EYECLIP_CKPT, map_location="cpu")["model_state_dict"]
    n_vis = sum(k.startswith("visual.") for k in sd)
    n_txt = sum(k.startswith(("transformer.", "token_embedding", "ln_final", "text_projection"))
                for k in sd)
    print(f"Checkpoint keys: visual={n_vis}, text-encoder={n_txt}")
    if n_txt == 0:
        print("  NOTE: no text-encoder weights in checkpoint -> CoOp uses the ORIGINAL "
              "OpenAI CLIP text encoder, not an EyeCLIP one. Keep this in mind when reading CoOp.")


def main():
    checkpoint_report()
    print("Loading EyeCLIP...")
    model, preprocess, inter = build_eyeclip()
    for p in model.parameters():
        p.requires_grad = False
    text = get_zeroshot_text_embeddings().to(DEVICE)          # [2,512], 0=CO, 1=AD

    results = {c: {m: {k: [] for k in ["auc", "acc", "f1", "sens", "spec"]} for m in METHODS}
               for c, _ in CONDITIONS}
    rows = []

    for seed in SPLIT_SEEDS:
        tr_p, ytr, va_p, yva = load_split(seed)
        for cond, use_clahe in CONDITIONS:
            print(f"\nSplit {seed} [{cond}]: {len(tr_p)} train, {len(va_p)} val")
            Xtr_l = extract_layers7_10(model, preprocess, inter, tr_p, use_clahe=use_clahe)
            Xva_l = extract_layers7_10(model, preprocess, inter, va_p, use_clahe=use_clahe)
            Xtr_z = extract_zeroshot(model, preprocess, tr_p, use_clahe=use_clahe)
            Xva_z = extract_zeroshot(model, preprocess, va_p, use_clahe=use_clahe)

            probs = {
                "Zero-Shot": zeroshot_probs(text, Xtr_z, Xva_z),
                "Tip-Adapter": tip_probs(text, Xtr_z, ytr, Xva_z),
                "Tip-Adapter-F": tip_f_probs(text, Xtr_z, ytr, Xva_z),
                "CLIP-Adapter": clip_adapter_probs(Xtr_z, ytr, Xva_z),
                "CoOp": coop_probs(model, Xtr_z, ytr, Xva_z),
                # the 4 existing methods, called exactly as in clahe_accuracy_full_metrics.py
                "Nearest Centroid": (nc_probs(Xtr_l, ytr, Xtr_l), nc_probs(Xtr_l, ytr, Xva_l)),
                "Proto-Adapter": (proto_adapter_probs(Xtr_l, ytr, Xtr_l),
                                  proto_adapter_probs(Xtr_l, ytr, Xva_l)),
                "TaskRes": (taskres_probs(text, Xtr_z, ytr, Xtr_z, use_sam=False),
                            taskres_probs(text, Xtr_z, ytr, Xva_z, use_sam=False)),
                "TaskRes+SAM": (taskres_probs(text, Xtr_z, ytr, Xtr_z, use_sam=True),
                                taskres_probs(text, Xtr_z, ytr, Xva_z, use_sam=True)),
            }
            for m in METHODS:
                ptr, pva = probs[m]
                auc = roc_auc_score(yva, pva)
                acc, f1, sens, spec, tau = balanced_threshold_eval(ptr, ytr, pva, yva)
                for k, v in zip(["auc", "acc", "f1", "sens", "spec"], [auc, acc, f1, sens, spec]):
                    results[cond][m][k].append(v)
                rows.append(dict(condition=cond, method=m, split=seed, auc=auc, acc=acc,
                                 f1=f1, sens=sens, spec=spec, tau=tau))
                print(f"  {m:<17} AUC={auc:.4f} Acc={acc:.4f} F1={f1:.4f} "
                      f"Sens={sens:.4f} Spec={spec:.4f}")

    with open("unified_benchmark_per_split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fmt = lambda v: f"{np.mean(v):.4f}+-{np.std(v):.4f}"
    for cond, _ in CONDITIONS:
        print(f"\n{'=' * 100}\n {cond} IMAGES (5 splits, balanced threshold from train)\n{'=' * 100}")
        print(f"{'Method':<18}{'AUC':<18}{'Accuracy':<18}{'F1':<18}{'Sensitivity':<18}Specificity")
        for m in sorted(METHODS, key=lambda m: -np.mean(results[cond][m]["auc"])):
            r = results[cond][m]
            print(f"{m:<18}{fmt(r['auc']):<18}{fmt(r['acc']):<18}{fmt(r['f1']):<18}"
                  f"{fmt(r['sens']):<18}{fmt(r['spec'])}")

    print(f"\n{'=' * 60}\n CLAHE EFFECT (AUC, same splits)\n{'=' * 60}")
    for m in METHODS:
        c, r = np.mean(results["CLAHE"][m]["auc"]), np.mean(results["RAW"][m]["auc"])
        print(f"{m:<18} raw {r:.4f} -> CLAHE {c:.4f}  ({c - r:+.4f})")

    print("\nSanity check - CLAHE rows should match: NC 0.7657, Proto 0.7654, "
          "TaskRes 0.7669, TaskRes+SAM 0.7677, Zero-Shot ~0.7615")
    print("Saved per-split results to unified_benchmark_per_split.csv")


if __name__ == "__main__":
    main()
