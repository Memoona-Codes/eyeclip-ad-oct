"""Compare the EyeCLIP checkpoint used in the paper with the updated release (EyeCLIP README, 2026-02:
"Fixed an issue where the text encoder weights were corrupted in the previous checkpoint").

    python moore_scripts/compare_eyeclip_ckpt.py --old eyeclip_visual.pt --new eyeclip_visual_2026.pt

Reports, for vision and text tensors separately: how many differ between old and new, and how many
differ from OpenAI CLIP ViT-B/32. Also recomputes the AD/CO prompt embeddings with each text encoder and
the zero-shot slice AUC on split 0 (CPU, about 3 minutes), so we know whether the paper must be rerun.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import clip  # noqa: E402

CO = ["optical coherence tomography, healthy control", "a retinal OCT scan of a healthy control subject",
      "OCT scan showing normal retinal layers with no signs of neurodegeneration",
      "retinal cross-section from a cognitively normal individual",
      "optical coherence tomography scan of an eye without Alzheimer disease"]
AD = ["optical coherence tomography, Alzheimer disease", "a retinal OCT scan of a patient with Alzheimer's disease",
      "OCT scan showing retinal changes consistent with Alzheimer's disease",
      "retinal cross-section from an Alzheimer's disease patient",
      "optical coherence tomography scan showing neurodegenerative retinal thinning"]


def load_sd(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(d, dict) and k in d:
            d = d[k]
            break
    return {k.replace("module.", ""): v for k, v in d.items()}


def text_emb(model):
    with torch.no_grad():
        rows = []
        for P in (CO, AD):
            e = model.encode_text(clip.tokenize(P)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            m = e.mean(0)
            rows.append(m / m.norm())
    return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="eyeclip_visual.pt")
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    base, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    oa = {k: v.float() for k, v in base.state_dict().items()}
    old, new = load_sd(a.old), load_sd(a.new)
    keys = [k for k in oa if k in old and k in new]
    print("tensors compared", len(keys), "| missing in old", len([k for k in oa if k not in old]),
          "| missing in new", len([k for k in oa if k not in new]))
    for part, sel in [("vision", lambda k: k.startswith("visual.")),
                      ("text", lambda k: not k.startswith("visual.") and k != "logit_scale")]:
        ks = [k for k in keys if sel(k)]
        d_on = [k for k in ks if not torch.equal(old[k].float(), new[k].float())]
        d_oa_old = [k for k in ks if not torch.equal(old[k].float(), oa[k])]
        d_oa_new = [k for k in ks if not torch.equal(new[k].float(), oa[k])]
        print(f"{part}: {len(ks)} tensors | old vs new differ {len(d_on)} | old vs OpenAI differ {len(d_oa_old)} "
              f"{d_oa_old[:3] if len(d_oa_old) < 5 else ''} | new vs OpenAI differ {len(d_oa_new)}")
    for k in ("logit_scale",):
        print(k, "OpenAI", float(oa[k]), "old", float(old.get(k, torch.tensor(float('nan')))),
              "new", float(new.get(k, torch.tensor(float('nan')))))

    import clahe_accuracy_full_metrics as cam
    from sklearn.metrics import roc_auc_score
    res = {}
    for name, sd in [("old", old), ("new", new)]:
        m, pre = clip.load("ViT-B/32", device="cpu", jit=False)
        m.load_state_dict(sd, strict=False)
        m = m.float().eval()
        T = text_emb(m)
        res[name] = (m, pre, T)
    print("cosine between old and new prompt embeddings (CO, AD):",
          (res["old"][2] * res["new"][2]).sum(-1).numpy().round(4))
    saved = torch.load(cam.EMBEDDINGS_PATH, map_location="cpu", weights_only=False)["embeddings"].float()
    print("cosine between saved file and new prompt embeddings (CO, AD):",
          (saved / saved.norm(dim=-1, keepdim=True) * res["new"][2]).sum(-1).numpy().round(4))

    tr, ytr, va, yva = cam.load_split(0)
    cam.DEVICE = torch.device("cpu")
    for name in ("old", "new"):
        m, pre, T = res[name]
        for use in (False, True):
            X = torch.tensor(cam.extract_zeroshot(m, pre, va, use_clahe=use))
            s = torch.softmax(100 * X @ T.T, dim=1)[:, 1].numpy()
            print(f"zero-shot split 0, {name} checkpoint, {'CLAHE' if use else 'raw'}: slice AUC {roc_auc_score(yva, s):.4f}")
    print("Paper (old checkpoint) Zero-Shot split 0 seed 42 CLAHE AUC: 0.7604")


if __name__ == "__main__":
    main()
