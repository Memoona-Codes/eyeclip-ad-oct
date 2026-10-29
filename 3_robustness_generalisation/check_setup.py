"""Audit of the set-up behind the LEAD paper (read-only; CPU; about 2 minutes).
Run from reset_v2_eyeclip/:   python moore_scripts/check_setup.py > check_setup.txt 2>&1; cat check_setup.txt

Checks
 1. Package versions (for the paper's reproducibility statement and requirements.txt).
 2. EyeCLIP checkpoint: does it really load into the CLIP ViT-B/32 (load_state_dict uses strict=False,
    so a key mismatch would silently leave OpenAI CLIP weights)? Which keys are missing / unexpected?
 3. Text embeddings file used by Zero-Shot / TaskRes: shape, contents, and agreement with the text
    encoder inside the checkpoint (EyeCLIP fixed a corrupted text encoder in Feb 2026).
 4. Bissig data: images per split, image size and mode, identical bytes for the same file across
    split folders, patient-disjointness, agreement with data/splits/all_splits.csv if present.
 5. OCTDL data: image sizes and modes per class (from octdl_index_used.csv).
"""
import collections
import glob
import hashlib
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())


def section(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8, flush=True)


section("1. Packages")
print("python", sys.version.split()[0])
for m in ["torch", "torchvision", "numpy", "scipy", "sklearn", "pandas", "cv2", "PIL", "timm", "clip", "matplotlib"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} {getattr(mod, '__version__', 'installed (no version attribute)')}")
    except Exception as e:
        print(f"{m:12s} NOT AVAILABLE ({e.__class__.__name__})")
import torch  # noqa: E402
print("cuda available", torch.cuda.is_available(), "| device",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

import clahe_accuracy_full_metrics as cam  # noqa: E402
section("2. EyeCLIP checkpoint")
try:
    import clip  # noqa: E402
    ck = cam.EYECLIP_CKPT
    print("file", os.path.abspath(ck), "| size MB", round(os.path.getsize(ck) / 1e6, 1),
          "| modified", __import__("time").ctime(os.path.getmtime(ck)))
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    print("top-level keys", list(raw.keys()) if isinstance(raw, dict) else type(raw))
    sd = raw["model_state_dict"]
    base, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    res = base.load_state_dict(sd, strict=False)
    print("checkpoint tensors", len(sd), "| model tensors", len(base_sd))
    print("missing keys", len(res.missing_keys), res.missing_keys[:10])
    print("unexpected keys", len(res.unexpected_keys), res.unexpected_keys[:10])
    vis = [k for k in base_sd if k.startswith("visual.")]
    txt = [k for k in base_sd if not k.startswith("visual.") and k != "logit_scale"]
    changed_v = sum(not torch.equal(base_sd[k], base.state_dict()[k]) for k in vis)
    changed_t = sum(not torch.equal(base_sd[k], base.state_dict()[k]) for k in txt)
    print(f"visual tensors changed from OpenAI CLIP: {changed_v} of {len(vis)}")
    print(f"text tensors changed from OpenAI CLIP:   {changed_t} of {len(txt)}")
    print("VERDICT:", "EyeCLIP vision weights loaded" if changed_v > 0.9 * len(vis) else "PROBLEM: vision weights NOT loaded")
except Exception as e:
    import traceback; traceback.print_exc(); print("SECTION FAILED:", e)

section("3. Text embeddings file")
try:
    d = torch.load(cam.EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
    print("file", os.path.abspath(cam.EMBEDDINGS_PATH), "| modified", __import__("time").ctime(os.path.getmtime(cam.EMBEDDINGS_PATH)))
    print("keys", list(d.keys()) if isinstance(d, dict) else type(d))
    E = d["embeddings"].float()
    print("embeddings shape", tuple(E.shape), "| row norms", E.norm(dim=-1).numpy().round(4))
    for k, v in d.items():
        if k != "embeddings":
            print(" ", k, ":", v if not torch.is_tensor(v) else tuple(v.shape))
    prompts = {0: ["optical coherence tomography, healthy control", "a retinal OCT scan of a healthy control subject",
                   "OCT scan showing normal retinal layers with no signs of neurodegeneration",
                   "retinal cross-section from a cognitively normal individual",
                   "optical coherence tomography scan of an eye without Alzheimer disease"],
               1: ["optical coherence tomography, Alzheimer disease", "a retinal OCT scan of a patient with Alzheimer's disease",
                   "OCT scan showing retinal changes consistent with Alzheimer's disease",
                   "retinal cross-section from an Alzheimer's disease patient",
                   "optical coherence tomography scan showing neurodegenerative retinal thinning"]}
    with torch.no_grad():
        rows = []
        for c in (0, 1):
            e = base.float().encode_text(clip.tokenize(prompts[c])).float()
            e = e / e.norm(dim=-1, keepdim=True)
            m = e.mean(0)
            rows.append(m / m.norm())
        T = torch.stack(rows)
        En = E / E.norm(dim=-1, keepdim=True)
        print("cosine(file row, recomputed with checkpoint text encoder) CO, AD:",
              (En * T).sum(-1).numpy().round(4), " (about 1.0 = same encoder and prompts)")
except Exception as e:
    import traceback; traceback.print_exc(); print("SECTION FAILED:", e)

section("4. Bissig data")
try:
    from PIL import Image  # noqa: E402
    by_split, sizes, modes, hashes = {}, collections.Counter(), collections.Counter(), collections.defaultdict(set)
    for s in cam.SPLIT_SEEDS:
        tr, ytr, va, yva = cam.load_split(s)
        pt = {os.path.basename(p).split("_")[0] for p in tr}
        pv = {os.path.basename(p).split("_")[0] for p in va}
        by_split[s] = (len(tr), len(va), len(pt), len(pv), len(pt & pv), int(np.sum(ytr)), int(np.sum(yva)))
        for p in tr + va:
            im = Image.open(p)
            sizes[im.size] += 1
            modes[im.mode] += 1
            hashes[os.path.basename(p)].add(hashlib.md5(open(p, "rb").read()).hexdigest())
    for s, v in by_split.items():
        print(f"split {s}: train {v[0]} imgs / {v[2]} pts ({v[5]} AD imgs), val {v[1]} imgs / {v[3]} pts ({v[6]} AD imgs), overlap {v[4]}")
    print("unique file names", len(hashes), "| names with differing bytes across splits",
          sum(len(h) > 1 for h in hashes.values()))
    print("image sizes (w, h):", dict(sizes))
    print("image modes:", dict(modes))
    w, h = sizes.most_common(1)[0][0]
    print(f"CLIP preprocessing keeps the central {min(w, h) / max(w, h):.0%} of the {'width' if w > h else 'height'}"
          f" (resize shorter side to 224, centre crop 224)")
    csvp = next((p for p in ["data/splits/all_splits.csv", "all_splits.csv", "../all_splits.csv"] if os.path.exists(p)), None)
    if csvp:
        import pandas as pd
        a = pd.read_csv(csvp)
        ok = all(set(a[(a.split_seed == s) & (a.set == "val")].patient_id) ==
                 {os.path.basename(p).split("_")[0] for p in cam.load_split(s)[2]} for s in cam.SPLIT_SEEDS)
        print(f"{csvp}: validation patients match the split folders: {ok}")
    else:
        print("all_splits.csv not found here (optional check skipped)")
except Exception as e:
    import traceback; traceback.print_exc(); print("SECTION FAILED:", e)

section("5. OCTDL data")
try:
    if os.path.exists("octdl_index_used.csv"):
        import pandas as pd
        df = pd.read_csv("octdl_index_used.csv")
        df["cls"] = df.path.map(lambda p: os.path.basename(os.path.dirname(p)))
        print(df.groupby("cls").agg(images=("path", "size"), patients=("patient_id", "nunique")))
        sz, md = collections.Counter(), collections.Counter()
        for p in df.path:
            im = Image.open(p)
            sz[im.size] += 1
            md[im.mode] += 1
        print("most common sizes:", sz.most_common(6), "| distinct sizes", len(sz))
        print("modes:", dict(md))
        print("patients in both classes NO and disease:",
              len(set(df[df.label == 0].patient_id) & set(df[df.label == 1].patient_id)))
    else:
        print("octdl_index_used.csv not found (run exp_C first)")
except Exception as e:
    import traceback; traceback.print_exc(); print("SECTION FAILED:", e)
print("\nCHECK COMPLETE")
