"""
Full metrics (AUC, Accuracy, Sensitivity, Specificity, F1) for the top
CLAHE-based methods from the final leaderboard: Nearest Centroid,
Proto-Adapter, TaskRes, TaskRes+SAM (champion).

Threshold: "balanced" strategy established early in this project (the
CLIP-Adapter phase found fixed-0.5 and F1-max thresholds both
degenerate to predicting one class always, since probabilities are
tightly compressed near 0.50-0.51). Balanced threshold is calibrated
on TRAIN only (the point where sens=spec on train), then applied
as-is to val -- no leakage, matches the established methodology.
"""

import os
import glob
import cv2
import clip
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPLITS_ROOT = os.path.expanduser("~/Gem Synthetic code")
SPLIT_SEEDS = [0, 1, 2, 3, 4]
EYECLIP_CKPT = "eyeclip_visual.pt"
EMBEDDINGS_PATH = "ensembled_clinical_text_embeddings.pt"
LAYER_INDICES = [6, 7, 8, 9]
MARGIN = 1.0
FT_EPOCHS = 50
FT_LR = 1e-2
TASKRES_EPOCHS = 50
TASKRES_LR = 1e-2
ALPHA_INIT = 0.5
SAM_RHO = 0.05
MODEL_SEED = 42

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def split_dir(seed):
    return os.path.join(SPLITS_ROOT, f"OCT_PNG_rebuilt_ADCO_slices_cropped_CLEAN_SPLIT_seed{seed}")


def load_split(seed):
    def load_subset(subset):
        paths, labels = [], []
        for cls, label in [("AD", 1), ("CO", 0)]:
            for fpath in glob.glob(os.path.join(split_dir(seed), subset, cls, "*.png")):
                paths.append(fpath)
                labels.append(label)
        return paths, np.array(labels)
    train_paths, train_labels = load_subset("train")
    val_paths, val_labels = load_subset("val")
    return train_paths, train_labels, val_paths, val_labels


def apply_clahe(pil_img):
    img_array = np.array(pil_img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(img_array)
    return Image.fromarray(enhanced).convert('RGB')


def build_eyeclip():
    model, preprocess = clip.load('ViT-B/32', device=DEVICE, jit=False)
    sd = torch.load(EYECLIP_CKPT, map_location='cpu')
    model.load_state_dict(sd['model_state_dict'], strict=False)
    model.eval()
    model = model.float()
    intermediate_outputs = {}
    def make_hook(layer_idx):
        def hook(module, inp, output):
            intermediate_outputs[layer_idx] = output.detach()
        return hook
    for idx in LAYER_INDICES:
        model.visual.transformer.resblocks[idx].register_forward_hook(make_hook(idx))
    return model, preprocess, intermediate_outputs


@torch.no_grad()
def extract_layers7_10(model, preprocess, intermediate_outputs, paths, use_clahe=True):
    per_layer_feats = {idx: [] for idx in LAYER_INDICES}
    for p in paths:
        img = Image.open(p).convert('RGB')
        if use_clahe:
            img = apply_clahe(img)
        img_t = preprocess(img).unsqueeze(0).to(DEVICE)
        _ = model.encode_image(img_t)
        for idx in LAYER_INDICES:
            tok_seq = intermediate_outputs[idx]
            cls_token = tok_seq[0, :, :]
            cls_token = model.visual.ln_post(cls_token)
            if model.visual.proj is not None:
                cls_token = cls_token @ model.visual.proj
            cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
            per_layer_feats[idx].append(cls_token.squeeze(0).cpu().numpy())
    arrs = [np.stack(per_layer_feats[idx]) for idx in LAYER_INDICES]
    return np.concatenate(arrs, axis=1)


@torch.no_grad()
def extract_zeroshot(model, preprocess, paths, use_clahe=True):
    feats = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        if use_clahe:
            img = apply_clahe(img)
        img_t = preprocess(img).unsqueeze(0).to(DEVICE)
        f = model.encode_image(img_t).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.squeeze(0).cpu().numpy())
    return np.stack(feats)


def get_zeroshot_text_embeddings():
    d = torch.load(EMBEDDINGS_PATH, map_location=DEVICE)
    return d['embeddings'].float()


# --- Balanced threshold (train-calibrated) ---
def balanced_threshold_eval(train_probs, train_labels, val_probs, val_labels):
    fpr, tpr, thresholds = roc_curve(train_labels, train_probs)
    balanced_idx = np.argmin(np.abs(tpr - (1 - fpr)))
    tau = thresholds[balanced_idx]
    preds = (val_probs >= tau).astype(int)
    acc = accuracy_score(val_labels, preds)
    f1 = f1_score(val_labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(val_labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return acc, f1, sens, spec, tau


# --- Nearest Centroid ---
def nc_probs(X_train, y_train, X_val):
    proto_co = X_train[y_train == 0].mean(axis=0)
    proto_ad = X_train[y_train == 1].mean(axis=0)
    proto_co = proto_co / (np.linalg.norm(proto_co) + 1e-8)
    proto_ad = proto_ad / (np.linalg.norm(proto_ad) + 1e-8)
    X_val_norm = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)
    sim_co = X_val_norm @ proto_co
    sim_ad = X_val_norm @ proto_ad
    return sim_ad / (sim_ad + sim_co + 1e-8)


# --- Proto-Adapter ---
class ProtoAdapter(nn.Module):
    def __init__(self, init_protos):
        super().__init__()
        self.protos = nn.Parameter(torch.tensor(init_protos, dtype=torch.float32))

    def forward(self, x):
        return torch.cdist(x, self.protos)


def margin_loss(dists, labels, margin):
    d_own = dists.gather(1, labels.unsqueeze(1)).squeeze(1)
    d_other = dists.gather(1, (1 - labels).unsqueeze(1)).squeeze(1)
    return torch.clamp(margin - (d_other - d_own), min=0).mean()


def proto_adapter_probs(X_train, y_train, X_eval, model_seed=MODEL_SEED):
    torch.manual_seed(model_seed)
    proto_co = X_train[y_train == 0].mean(axis=0)
    proto_ad = X_train[y_train == 1].mean(axis=0)
    init_protos = np.stack([proto_co, proto_ad])
    model = ProtoAdapter(init_protos).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=FT_LR)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    model.train()
    for epoch in range(FT_EPOCHS):
        optimizer.zero_grad()
        dists = model(X_train_t)
        loss = margin_loss(dists, y_train_t, MARGIN)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=DEVICE)
        dists = model(X_eval_t).cpu().numpy()
    d_co, d_ad = dists[:, 0], dists[:, 1]
    return d_co / (d_co + d_ad + 1e-8)


# --- TaskRes / TaskRes+SAM ---
class TaskRes(nn.Module):
    def __init__(self, base_text_embeddings, alpha_init=ALPHA_INIT):
        super().__init__()
        self.register_buffer('base_embeddings', base_text_embeddings)
        self.residual = nn.Parameter(torch.zeros_like(base_text_embeddings))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, image_features):
        text_weights = self.base_embeddings + self.alpha * self.residual
        text_weights = text_weights / text_weights.norm(dim=-1, keepdim=True)
        return image_features @ text_weights.T


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None])
        )
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.state[p]:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()


def taskres_probs(base_embeddings, X_train, y_train, X_eval, use_sam=False, model_seed=MODEL_SEED):
    torch.manual_seed(model_seed)
    model = TaskRes(base_embeddings).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    if use_sam:
        optimizer = SAM(model.parameters(), torch.optim.Adam, rho=SAM_RHO, lr=TASKRES_LR)
        model.train()
        for epoch in range(TASKRES_EPOCHS):
            logits = model(X_train_t)
            loss = loss_fn(logits, y_train_t)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            logits2 = model(X_train_t)
            loss2 = loss_fn(logits2, y_train_t)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=TASKRES_LR)
        model.train()
        for epoch in range(TASKRES_EPOCHS):
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = loss_fn(logits, y_train_t)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=DEVICE)
        probs = torch.softmax(model(X_eval_t), dim=1)[:, 1].cpu().numpy()
    return probs


def main():
    print("Loading EyeCLIP...")
    model, preprocess, intermediate_outputs = build_eyeclip()
    base_embeddings = get_zeroshot_text_embeddings()

    methods = ["Nearest Centroid", "Proto-Adapter", "TaskRes", "TaskRes+SAM"]
    all_metrics = {m: {"auc": [], "acc": [], "f1": [], "sens": [], "spec": []} for m in methods}

    for seed in SPLIT_SEEDS:
        train_paths, train_labels, val_paths, val_labels = load_split(seed)
        print(f"\nSplit {seed}: {len(train_paths)} train, {len(val_paths)} val")

        X_train_l = extract_layers7_10(model, preprocess, intermediate_outputs, train_paths, use_clahe=True)
        X_val_l = extract_layers7_10(model, preprocess, intermediate_outputs, val_paths, use_clahe=True)
        X_train_zs = extract_zeroshot(model, preprocess, train_paths, use_clahe=True)
        X_val_zs = extract_zeroshot(model, preprocess, val_paths, use_clahe=True)

        method_probs = {
            "Nearest Centroid": (nc_probs(X_train_l, train_labels, X_train_l), nc_probs(X_train_l, train_labels, X_val_l)),
            "Proto-Adapter": (proto_adapter_probs(X_train_l, train_labels, X_train_l), proto_adapter_probs(X_train_l, train_labels, X_val_l)),
            "TaskRes": (taskres_probs(base_embeddings, X_train_zs, train_labels, X_train_zs, use_sam=False),
                        taskres_probs(base_embeddings, X_train_zs, train_labels, X_val_zs, use_sam=False)),
            "TaskRes+SAM": (taskres_probs(base_embeddings, X_train_zs, train_labels, X_train_zs, use_sam=True),
                            taskres_probs(base_embeddings, X_train_zs, train_labels, X_val_zs, use_sam=True)),
        }

        for name, (train_probs, val_probs) in method_probs.items():
            auc = roc_auc_score(val_labels, val_probs)
            acc, f1, sens, spec, tau = balanced_threshold_eval(train_probs, train_labels, val_probs, val_labels)
            all_metrics[name]["auc"].append(auc)
            all_metrics[name]["acc"].append(acc)
            all_metrics[name]["f1"].append(f1)
            all_metrics[name]["sens"].append(sens)
            all_metrics[name]["spec"].append(spec)
            print(f"  {name}: AUC={auc:.4f} Acc={acc:.4f} F1={f1:.4f} Sens={sens:.4f} Spec={spec:.4f} (tau={tau:.4f})")

    print(f"\n{'='*90}")
    print(f"{'Method':<20}{'AUC':<18}{'Accuracy':<18}{'F1':<18}{'Sensitivity':<18}{'Specificity'}")
    for name in methods:
        m = all_metrics[name]
        print(f"{name:<20}"
              f"{np.mean(m['auc']):.4f}+-{np.std(m['auc']):.4f}   "
              f"{np.mean(m['acc']):.4f}+-{np.std(m['acc']):.4f}   "
              f"{np.mean(m['f1']):.4f}+-{np.std(m['f1']):.4f}   "
              f"{np.mean(m['sens']):.4f}+-{np.std(m['sens']):.4f}   "
              f"{np.mean(m['spec']):.4f}+-{np.std(m['spec']):.4f}")

    print(f"\nFor comparison, established AUCs (should match rows above):")
    print(f"  Nearest Centroid: 0.7657+-0.0548")
    print(f"  Proto-Adapter:    0.7654+-0.0547")
    print(f"  TaskRes:          0.7669+-0.0522")
    print(f"  TaskRes+SAM:      0.7677+-0.0523")


if __name__ == "__main__":
    main()
