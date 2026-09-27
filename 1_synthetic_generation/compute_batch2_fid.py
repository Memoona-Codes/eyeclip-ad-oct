"""
Fréchet Inception Distance (FID) audit for batch 2, matching the exact
protocol used in the original Step 1 audit (which was run on batch 1 only):

1. Per-class FID between the real training set and the corresponding
   synthetic images (ImageNet-pretrained InceptionV3 pool features, 2048-dim).
2. A real-vs-real split-half FID noise floor per class, to isolate the
   contribution of small-sample-size bias alone.
3. Explicit N stated for both real and synthetic sets, per class.

Run from: the data root folder  (so relative paths resolve)
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from scipy import linalg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REAL_TRAIN_ROOT = "OCT_PNG_rebuilt_ADCO_slices/train"
BATCH2_ROOT = "synthetic_dataset_v8_batch2"
SPLIT_SEED = 123

# ------------------------------------------------------------------
# InceptionV3 pool-features extractor (2048-dim, matches standard FID)
# ------------------------------------------------------------------
print("Loading InceptionV3 (ImageNet pretrained) for FID feature extraction...")
inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
inception.fc = nn.Identity()  # strip final classifier, keep 2048-dim pool features
inception.eval().to(DEVICE)

fid_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # InceptionV3 expects 3-channel
    transforms.Resize((299, 299)),                 # InceptionV3's native input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_inception_features(file_list, batch_size=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i + batch_size]
            imgs = torch.stack([fid_transform(Image.open(f).convert('L')) for f in batch_files]).to(DEVICE)
            out = inception(imgs)
            # inception_v3 in eval mode with aux_logits=True still returns a single tensor
            if isinstance(out, tuple):
                out = out[0]
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)

def compute_fid(feats1, feats2):
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def list_files(root, cls):
    return sorted(glob.glob(os.path.join(root, cls, '*.png')))

results = []

for cls in ['AD', 'CO']:
    print(f"\n{'='*70}\n Class: {cls}\n{'='*70}")

    real_files = list_files(REAL_TRAIN_ROOT, cls)
    synth_files = list_files(BATCH2_ROOT, cls)
    n_real = len(real_files)
    n_synth = len(synth_files)
    print(f"N_real (train) = {n_real}, N_synth (batch2) = {n_synth}")

    print("Extracting real-image features...")
    real_feats = get_inception_features(real_files)
    print("Extracting batch-2 synthetic-image features...")
    synth_feats = get_inception_features(synth_files)

    # Real-vs-synthetic FID
    fid_real_vs_synth = compute_fid(real_feats, synth_feats)

    # Real-vs-real split-half noise floor
    rng = np.random.RandomState(SPLIT_SEED)
    idx = np.arange(n_real)
    rng.shuffle(idx)
    half = n_real // 2
    half1_feats = real_feats[idx[:half]]
    half2_feats = real_feats[idx[half:]]
    fid_noise_floor = compute_fid(half1_feats, half2_feats)

    ratio = fid_real_vs_synth / fid_noise_floor if fid_noise_floor > 0 else float('inf')

    print(f"Real-vs-real split-half FID (noise floor, {half}/{n_real-half}): {fid_noise_floor:.2f}")
    print(f"Real-vs-batch2-synthetic FID: {fid_real_vs_synth:.2f}")
    print(f"Ratio (synth FID / noise floor): {ratio:.2f}x")

    results.append({
        'class': cls, 'n_real': n_real, 'n_synth': n_synth,
        'fid_noise_floor': fid_noise_floor, 'fid_real_vs_synth': fid_real_vs_synth,
        'ratio': ratio,
    })

print(f"\n{'='*70}\n SUMMARY -- Batch 2 FID vs. Batch 1 (from original Step 1 audit)\n{'='*70}")
print(f"{'Class':<8}{'N_real':<8}{'N_synth':<9}{'Noise Floor':<14}{'Batch2 FID':<13}{'Ratio':<8}")
for r in results:
    print(f"{r['class']:<8}{r['n_real']:<8}{r['n_synth']:<9}{r['fid_noise_floor']:<14.2f}"
          f"{r['fid_real_vs_synth']:<13.2f}{r['ratio']:<8.2f}x")

print(f"\nFor comparison, batch 1's original Step 1 audit found:")
print(f"  AD: FID=427.86, noise floor=73.20, ratio=5.8x")
print(f"  CO: FID=443.60, noise floor=64.51, ratio=6.9x")

import csv
with open('batch2_fid_audit.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['class', 'n_real', 'n_synth', 'fid_noise_floor', 'fid_real_vs_synth', 'ratio'])
    for r in results:
        writer.writerow([r['class'], r['n_real'], r['n_synth'], r['fid_noise_floor'],
                          r['fid_real_vs_synth'], r['ratio']])
print("\nSaved to batch2_fid_audit.csv")
