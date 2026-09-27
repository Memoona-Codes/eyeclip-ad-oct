"""
Diagnostic: does batch 2's AD/ vs CO/ folder split actually correlate with
real AD/CO content? (Companion check to check_batch1_class_correlation.py --
batch 1 was tested and confirmed to carry NO class signal, ~50/50 chance.)

generate_batch2_with_audit.py's code interpolates strictly WITHIN class
(AD-with-AD, CO-with-CO only) -- this script independently verifies that
expectation empirically, rather than trusting the code alone. Expectation:
strong majority (well above 50%) nearest-real-neighbor agreement with the
folder's own class label, in contrast to batch 1's near-chance result.
"""

import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

REAL_TRAIN_ROOT = "OCT_PNG_rebuilt_ADCO_slices/train"
BATCH2_ROOT = "synthetic_dataset_v8_batch2"

def load_images_as_array(folder, max_images=None):
    valid_exts = ('.png', '.jpg', '.jpeg')
    files = sorted([f for f in glob.glob(os.path.join(folder, '*')) if f.lower().endswith(valid_exts)])
    if max_images:
        files = files[:max_images]
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
    ])
    arr = np.stack([np.array(tfm(Image.open(f).convert('L'))) / 255.0 for f in files])
    return arr, files

print("Loading real reference images...")
real_ad, real_ad_files = load_images_as_array(os.path.join(REAL_TRAIN_ROOT, "AD"))
real_co, real_co_files = load_images_as_array(os.path.join(REAL_TRAIN_ROOT, "CO"))
print(f"  Real AD: {len(real_ad)}, Real CO: {len(real_co)}")

def nearest_real_class(img):
    """Returns 'AD' or 'CO' depending on which real set has the higher-SSIM match."""
    best_ad_ssim = max(ssim(img, r, data_range=1.0) for r in real_ad)
    best_co_ssim = max(ssim(img, r, data_range=1.0) for r in real_co)
    return 'AD' if best_ad_ssim > best_co_ssim else 'CO', best_ad_ssim, best_co_ssim

results = {}
for folder_label in ['AD', 'CO']:
    print(f"\nChecking batch2/{folder_label}/ ...")
    imgs, files = load_images_as_array(os.path.join(BATCH2_ROOT, folder_label))
    print(f"  N = {len(imgs)}")

    votes_ad, votes_co = 0, 0
    for img in imgs:
        nearest_class, ad_score, co_score = nearest_real_class(img)
        if nearest_class == 'AD':
            votes_ad += 1
        else:
            votes_co += 1

    pct_ad = 100 * votes_ad / len(imgs)
    pct_co = 100 * votes_co / len(imgs)
    print(f"  Nearest-real-neighbor class breakdown: {votes_ad} closer to AD ({pct_ad:.1f}%), "
          f"{votes_co} closer to CO ({pct_co:.1f}%)")
    results[folder_label] = (votes_ad, votes_co, pct_ad, pct_co)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"batch2/AD/  -> {results['AD'][2]:.1f}% nearest-real-AD, {results['AD'][3]:.1f}% nearest-real-CO")
print(f"batch2/CO/  -> {results['CO'][2]:.1f}% nearest-real-AD, {results['CO'][3]:.1f}% nearest-real-CO")

ad_folder_agreement = results['AD'][2]   # % of batch2/AD/ images whose nearest real neighbor is real AD
co_folder_agreement = results['CO'][3]   # % of batch2/CO/ images whose nearest real neighbor is real CO
print(f"\nAD-folder self-class agreement: {ad_folder_agreement:.1f}% "
      f"(batch2/AD/ images whose nearest real match is real AD)")
print(f"CO-folder self-class agreement: {co_folder_agreement:.1f}% "
      f"(batch2/CO/ images whose nearest real match is real CO)")

if ad_folder_agreement > 65 and co_folder_agreement > 65:
    print("\nCONCLUSION: strong self-class agreement in both folders -- confirms batch 2's")
    print("AD/CO labels carry real class signal, consistent with its class-conditional")
    print("generation code. Safe to treat as the reliable, trusted synthetic pool.")
elif ad_folder_agreement > 55 and co_folder_agreement > 55:
    print("\nCONCLUSION: modest self-class agreement -- some real signal present, but weaker")
    print("than expected for genuinely within-class-only interpolation. Worth a closer look")
    print("before fully trusting these labels for Phase 2.")
else:
    print("\nCONCLUSION: close to or below chance -- batch 2's AD/CO split does NOT show the")
    print("expected class signal. This is a SERIOUS finding: it would mean either the")
    print("class-conditional generation code did not work as intended, or something else")
    print("is wrong. Do NOT proceed to trust batch 2 as the reliable pool without investigating.")

import csv
with open('batch2_class_correlation_check.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['folder', 'n', 'votes_nearest_AD', 'votes_nearest_CO', 'pct_nearest_AD', 'pct_nearest_CO'])
    for label, (vad, vco, pad, pco) in results.items():
        writer.writerow([label, vad + vco, vad, vco, pad, pco])
print("\nSaved to batch2_class_correlation_check.csv")
