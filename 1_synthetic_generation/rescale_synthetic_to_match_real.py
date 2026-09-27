"""
Fixes the real-vs-synthetic pixel-statistics mismatch confirmed by
synth_diagnostic.py this session:

    Real AD/CO (post-CLIP-preprocess):  mean~1.5-1.6, std~0.17-0.28
    Synth AD/CO (post-CLIP-preprocess): mean~0.75-0.98, std~0.80-0.97

Synthetic images are darker and 3-5x higher variance than real images.
This is very likely why LoRA/CLIP-Adapter/CLIP-LoRA collapsed to
degenerate Sens=0/Spec=1 predictions the moment synthetic images entered
train -- the model can trivially separate real-vs-synthetic by brightness
alone, and that shortcut swamps the much subtler AD-vs-CO signal.

FIX: simple per-class linear rescaling in RAW PIXEL SPACE (before any
CLIP preprocessing), matching synthetic images' mean and std to real
images' mean and std, per class (AD matched to real AD stats, CO matched
to real CO stats separately -- keeps class-conditional differences
intact rather than washing them out).

    pixel_new = (pixel_old - synth_mean) / synth_std * real_std + real_mean

Clipped to [0, 255] and saved as new PNGs in a separate output directory
-- the original synthetic_dataset_v8_batch2 is left untouched.

Run this ONCE to produce the corrected pool, then point any future
synthetic-dosage experiment at the corrected directory instead of the
original. Re-run synth_diagnostic.py afterward (pointed at the new
directory) to confirm the fix worked before spending GPU time on a full
retraining sweep.
"""

import os
import glob
import numpy as np
from PIL import Image

REAL_ROOT = "OCT_PNG_rebuilt_ADCO_slices_cropped_CLEAN_SPLIT/train"
SYNTH_ROOT = "synthetic_dataset_v8_batch2"
OUTPUT_ROOT = "synthetic_dataset_v8_batch2_rescaled"
CLASSES = ["AD", "CO"]


def list_files(root, cls):
    return sorted(glob.glob(os.path.join(root, cls, "*.png")))


def compute_pixel_stats(files):
    """Mean and std across ALL pixels of ALL images in this list (grayscale-equivalent,
    computed per RGB channel then averaged, since these are effectively grayscale
    OCT images saved as RGB)."""
    all_pixels = []
    for f in files:
        img = np.array(Image.open(f).convert("RGB"), dtype=np.float64)
        all_pixels.append(img.flatten())
    all_pixels = np.concatenate(all_pixels)
    return all_pixels.mean(), all_pixels.std()


def rescale_image(path, synth_mean, synth_std, real_mean, real_std):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float64)
    if synth_std < 1e-6:
        synth_std = 1e-6
    img_new = (img - synth_mean) / synth_std * real_std + real_mean
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)
    return Image.fromarray(img_new)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    stats_summary = {}

    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

        real_files = list_files(REAL_ROOT, cls)
        synth_files = list_files(SYNTH_ROOT, cls)
        print(f"\n{'='*70}\n Class: {cls}\n{'='*70}")
        print(f"Real train files: {len(real_files)}   Synthetic files: {len(synth_files)}")

        print("Computing real pixel statistics...")
        real_mean, real_std = compute_pixel_stats(real_files)
        print(f"  Real:      mean={real_mean:.2f}  std={real_std:.2f}")

        print("Computing synthetic pixel statistics (before correction)...")
        synth_mean, synth_std = compute_pixel_stats(synth_files)
        print(f"  Synthetic: mean={synth_mean:.2f}  std={synth_std:.2f}  (before correction)")

        print(f"Rescaling {len(synth_files)} synthetic images...")
        for f in synth_files:
            rescaled = rescale_image(f, synth_mean, synth_std, real_mean, real_std)
            out_path = os.path.join(OUTPUT_ROOT, cls, os.path.basename(f))
            rescaled.save(out_path)

        # Verify: recompute stats on the newly-saved rescaled images
        rescaled_files = list_files(OUTPUT_ROOT, cls)
        new_mean, new_std = compute_pixel_stats(rescaled_files)
        print(f"  Synthetic: mean={new_mean:.2f}  std={new_std:.2f}  (AFTER correction -- "
              f"should now be close to real mean={real_mean:.2f}, std={real_std:.2f})")

        stats_summary[cls] = {
            "real_mean": real_mean, "real_std": real_std,
            "synth_mean_before": synth_mean, "synth_std_before": synth_std,
            "synth_mean_after": new_mean, "synth_std_after": new_std,
            "n_real": len(real_files), "n_synth": len(synth_files),
        }

    print(f"\n{'='*70}\n SUMMARY\n{'='*70}")
    for cls, s in stats_summary.items():
        print(f"{cls}: real(mean={s['real_mean']:.2f},std={s['real_std']:.2f})  "
              f"synth before(mean={s['synth_mean_before']:.2f},std={s['synth_std_before']:.2f})  "
              f"synth after(mean={s['synth_mean_after']:.2f},std={s['synth_std_after']:.2f})")

    print(f"\nRescaled synthetic pool saved to: {OUTPUT_ROOT}")
    print("\nNEXT STEP: re-run synth_diagnostic.py, pointed at this new directory instead of")
    print("the original synthetic_dataset_v8_batch2, to confirm the post-CLIP-preprocess")
    print("tensor stats (mean/std) now land close to the real image stats, before spending")
    print("GPU time re-running any synthetic-dosage training sweep.")

    import csv
    with open("synthetic_rescale_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "n_real", "n_synth", "real_mean", "real_std",
                          "synth_mean_before", "synth_std_before", "synth_mean_after", "synth_std_after"])
        for cls, s in stats_summary.items():
            writer.writerow([cls, s["n_real"], s["n_synth"], s["real_mean"], s["real_std"],
                              s["synth_mean_before"], s["synth_std_before"],
                              s["synth_mean_after"], s["synth_std_after"]])
    print("Saved stats to synthetic_rescale_stats.csv")


if __name__ == "__main__":
    main()
