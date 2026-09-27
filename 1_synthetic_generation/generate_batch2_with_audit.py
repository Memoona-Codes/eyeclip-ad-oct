"""
Generate a SECOND batch of 1000 synthetic OCT images (500 AD / 500 CO),
distinct from the existing synthetic_dataset_v8 pool, using the same
pairwise-latent-interpolation-with-jitter method as generate_bulk_synthetic.py
-- but with a built-in memorization/leakage audit run BEFORE the batch is
accepted, following the same protocol as the Step 1 audit already on record
(explicit real-N and synth-N stated, SSIM nearest-neighbor + L2 distance,
flagged near-duplicate threshold: SSIM>0.85 AND L2<2.0).

ADDITIONAL CHECK not in the original Step 1 audit: this batch is also
checked against the EXISTING 1000 synthetic images, to catch the case where
the interpolation/jitter process regenerates near-duplicates of images
already in the pool (a distinct failure mode from real-image memorization).

*** ARCHITECTURE PLACEHOLDER ***
The class below (V8VAE) is copied from vae_v8.py, our current best guess
for the architecture matching v8_vae_grayscale_best.pth. If the uploaded
checkpoint turns out to match a DIFFERENT architecture (e.g. the
Encoder/Decoder classes in vae_gan_train_v8_cond.py), replace the class
definition below accordingly before running -- everything else in this
script (generation loop, audit logic) is architecture-independent.
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints_grayscale/v8_vae_grayscale_best.pth"
REAL_TRAIN_ROOT = "OCT_PNG_rebuilt_ADCO_slices/train"   # explicit N stated below, per Step 1 protocol
EXISTING_SYNTH_ROOT = "synthetic_dataset_v8_cond"        # the existing 1000-image pool
OUTPUT_DIR = "synthetic_dataset_v8_batch2"                # NEW folder -- does not overwrite batch 1
TOTAL_IMAGES = 1000
IMAGES_PER_CLASS = TOTAL_IMAGES // 2  # 500 AD, 500 CO
SEED = 43  # distinct from batch 1's implicit seeding, so batch 2 is a genuinely independent draw

MEMORIZATION_SSIM_THRESHOLD = 0.85
MEMORIZATION_L2_THRESHOLD = 2.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# ARCHITECTURE (placeholder -- confirm/replace before running)
# ============================================================
class V8VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(512 * 14 * 14, latent_dim)
        self.fc_var = nn.Linear(512 * 14 * 14, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 512 * 14 * 14)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        features = torch.flatten(self.encoder(x), start_dim=1)
        return self.fc_mu(features)

    def decode(self, z):
        dec_features = self.decoder_input(z)
        dec_features = dec_features.view(-1, 512, 14, 14)
        return self.decoder(dec_features)


class ClassFolderDataset(Dataset):
    """Loads images from a class-labeled subfolder, e.g. root/AD/*.png"""
    def __init__(self, folder):
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        self.files = sorted([f for f in glob.glob(os.path.join(folder, '*')) if f.lower().endswith(valid_exts)])
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]).convert('L'))


def load_images_as_array(folder, max_images=None):
    """Load a folder of images as a numpy array (N, H, W) in [0,1], for SSIM/L2 comparison."""
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
    files = sorted([f for f in glob.glob(os.path.join(folder, '*')) if f.lower().endswith(valid_exts)])
    if max_images:
        files = files[:max_images]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
    ])
    arr = np.stack([np.array(transform(Image.open(f).convert('L'))) / 255.0 for f in files])
    return arr, files


def memorization_audit(new_synth_dir, reference_dir, reference_label, n_new_max=None):
    """
    For every new synthetic image, find its nearest neighbor (max SSIM, min L2)
    in the reference set. Flags candidates where SSIM > threshold AND L2 < threshold.
    Prints explicit N for both sets, per Step 1 protocol.
    """
    new_imgs, new_files = load_images_as_array(new_synth_dir, max_images=n_new_max)
    ref_imgs, ref_files = load_images_as_array(reference_dir)

    print(f"\n  Comparing against: {reference_label}")
    print(f"    N_new = {len(new_imgs)}, N_reference = {len(ref_imgs)}")

    max_ssim_per_new = []
    min_l2_per_new = []
    flagged = []

    for i, new_img in enumerate(new_imgs):
        best_ssim = -1.0
        best_l2 = float('inf')
        for ref_img in ref_imgs:
            s = ssim(new_img, ref_img, data_range=1.0)
            l2 = np.linalg.norm(new_img - ref_img)
            if s > best_ssim:
                best_ssim = s
            if l2 < best_l2:
                best_l2 = l2
        max_ssim_per_new.append(best_ssim)
        min_l2_per_new.append(best_l2)
        if best_ssim > MEMORIZATION_SSIM_THRESHOLD and best_l2 < MEMORIZATION_L2_THRESHOLD:
            flagged.append((new_files[i], best_ssim, best_l2))

    max_ssim_per_new = np.array(max_ssim_per_new)
    min_l2_per_new = np.array(min_l2_per_new)

    print(f"    NN-SSIM: {max_ssim_per_new.mean():.4f} +/- {max_ssim_per_new.std():.4f}")
    print(f"    Min L2 distance (mean): {min_l2_per_new.mean():.2f}")
    print(f"    Flagged near-duplicates: {len(flagged)} / {len(new_imgs)} "
          f"({100*len(flagged)/len(new_imgs):.1f}%)")
    if flagged:
        print(f"    First few flagged files:")
        for f, s, l2 in flagged[:5]:
            print(f"      {f}: SSIM={s:.4f}, L2={l2:.2f}")

    return {
        'reference_label': reference_label,
        'n_new': len(new_imgs), 'n_reference': len(ref_imgs),
        'mean_ssim': float(max_ssim_per_new.mean()), 'std_ssim': float(max_ssim_per_new.std()),
        'mean_l2': float(min_l2_per_new.mean()),
        'n_flagged': len(flagged), 'pct_flagged': 100 * len(flagged) / len(new_imgs),
    }


def generate_batch2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls in ['AD', 'CO']:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    model = V8VAE(latent_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    for cls in ['AD', 'CO']:
        real_folder = os.path.join(REAL_TRAIN_ROOT, cls)
        dataset = ClassFolderDataset(real_folder)
        n_real = len(dataset)
        print(f"\n{cls}: found {n_real} real training images in '{real_folder}'")
        loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

        generated = 0
        print(f"Generating {IMAGES_PER_CLASS} class-conditional synthetic {cls} images (within-class interpolation only)...")

        with torch.no_grad():
            while generated < IMAGES_PER_CLASS:
                for real_imgs in loader:
                    if generated >= IMAGES_PER_CLASS:
                        break
                    real_imgs = real_imgs.to(DEVICE)
                    b = real_imgs.size(0)
                    z_real = model.encode(real_imgs)

                    idx1 = torch.randperm(b)
                    idx2 = torch.randperm(b)
                    alpha = torch.rand(b, 1, device=DEVICE)
                    z_blend = alpha * z_real[idx1] + (1 - alpha) * z_real[idx2]
                    z_synthetic = z_blend + torch.randn_like(z_blend) * 0.20
                    synthetic_imgs = model.decode(z_synthetic)

                    for i in range(synthetic_imgs.size(0)):
                        if generated >= IMAGES_PER_CLASS:
                            break
                        img_path = os.path.join(OUTPUT_DIR, cls, f"synth_batch2_{cls}_{generated+1:04d}.png")
                        save_image(synthetic_imgs[i], img_path, normalize=True)
                        generated += 1
                    print(f"  {generated}/{IMAGES_PER_CLASS}...", end="\r")

        print(f"\n  Done: {generated} {cls} images saved to {os.path.join(OUTPUT_DIR, cls)}")

    print(f"\nBatch 2 generation complete: {TOTAL_IMAGES} images ({IMAGES_PER_CLASS} AD, {IMAGES_PER_CLASS} CO)")


def run_full_audit():
    print("\n" + "=" * 70)
    print(" MEMORIZATION / LEAKAGE AUDIT -- BATCH 2")
    print(" (protocol matches Step 1: explicit N, SSIM+L2 nearest-neighbor,")
    print(f"  flagged if SSIM>{MEMORIZATION_SSIM_THRESHOLD} AND L2<{MEMORIZATION_L2_THRESHOLD})")
    print("=" * 70)

    results = []
    for cls in ['AD', 'CO']:
        print(f"\n--- Class: {cls} ---")
        new_dir = os.path.join(OUTPUT_DIR, cls)
        real_dir = os.path.join(REAL_TRAIN_ROOT, cls)
        existing_synth_dir = os.path.join(EXISTING_SYNTH_ROOT, cls)

        r1 = memorization_audit(new_dir, real_dir, reference_label=f"real training set ({cls})")
        r1['class'] = cls
        results.append(r1)

        if os.path.isdir(existing_synth_dir):
            r2 = memorization_audit(new_dir, existing_synth_dir, reference_label=f"existing synthetic batch 1 ({cls})")
            r2['class'] = cls
            results.append(r2)
        else:
            print(f"\n  (skipping batch1-vs-batch2 check: '{existing_synth_dir}' not found -- "
                  f"confirm EXISTING_SYNTH_ROOT path if this is unexpected)")

    print("\n" + "=" * 70)
    print(" AUDIT SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r['class']} vs {r['reference_label']}: "
              f"N_new={r['n_new']}, N_ref={r['n_reference']}, "
              f"NN-SSIM={r['mean_ssim']:.4f}+/-{r['std_ssim']:.4f}, "
              f"flagged={r['n_flagged']}/{r['n_new']} ({r['pct_flagged']:.1f}%)")

    any_flagged = any(r['n_flagged'] > 0 for r in results)
    if any_flagged:
        print("\n*** WARNING: some batch-2 images flagged as candidate memorization/duplicates. ***")
        print("*** Review flagged files above before using this batch for classification experiments. ***")
    else:
        print("\nNo memorization or duplication flagged, against either real data or the existing synthetic pool.")

    import csv
    with open('batch2_memorization_audit.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'reference_label', 'n_new', 'n_reference', 'mean_ssim', 'std_ssim',
                          'mean_l2', 'n_flagged', 'pct_flagged'])
        for r in results:
            writer.writerow([r['class'], r['reference_label'], r['n_new'], r['n_reference'],
                              r['mean_ssim'], r['std_ssim'], r['mean_l2'], r['n_flagged'], r['pct_flagged']])
    print("\nSaved to batch2_memorization_audit.csv")


if __name__ == "__main__":
    generate_batch2()
    run_full_audit()
