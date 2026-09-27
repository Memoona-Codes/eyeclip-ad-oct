"""
Reproduces the "highest-correlation synthetic/real pair + pixel scatter plot"
figure style from Slootweg et al. (2024), applied to batch 2.

For each class (AD, CO): find the batch-2 synthetic image with the highest
Pearson correlation to any real training image, and display the pair
side-by-side with a pixel-value scatter plot in between (synthetic pixel
value vs. real pixel value, one point per pixel).

Run from: the data root folder
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

REAL_TRAIN_ROOT = "OCT_PNG_rebuilt_ADCO_slices/train"
BATCH2_ROOT = "synthetic_dataset_v8_batch2"
N_TOP_PAIRS = 3  # show top-3 highest-correlation pairs per class, matching the paper's layout

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

def find_top_correlated_pairs(synth_imgs, synth_files, real_imgs, real_files, n_top):
    """For every synth image, find its max-Pearson-correlation real match. Return the n_top
    synth images with the HIGHEST such correlation (i.e. the closest matches overall)."""
    best_corr = np.zeros(len(synth_imgs))
    best_real_idx = np.zeros(len(synth_imgs), dtype=int)
    for i, s in enumerate(synth_imgs):
        s_flat = s.flatten()
        corrs = np.array([np.corrcoef(s_flat, r.flatten())[0, 1] for r in real_imgs])
        best_real_idx[i] = np.argmax(corrs)
        best_corr[i] = corrs[best_real_idx[i]]
    top_idx = np.argsort(-best_corr)[:n_top]
    return [(synth_imgs[i], synth_files[i], real_imgs[best_real_idx[i]], real_files[best_real_idx[i]], best_corr[i])
            for i in top_idx]

fig, axes = plt.subplots(N_TOP_PAIRS * 2, 3, figsize=(9, N_TOP_PAIRS * 2 * 3))

row = 0
for cls in ['AD', 'CO']:
    print(f"Loading {cls} images...")
    real_imgs, real_files = load_images_as_array(os.path.join(REAL_TRAIN_ROOT, cls))
    synth_imgs, synth_files = load_images_as_array(os.path.join(BATCH2_ROOT, cls))
    print(f"  Real: {len(real_imgs)}, Synth (batch2): {len(synth_imgs)}")

    print(f"  Finding top-{N_TOP_PAIRS} highest-correlation pairs for {cls}...")
    top_pairs = find_top_correlated_pairs(synth_imgs, synth_files, real_imgs, real_files, N_TOP_PAIRS)

    for synth_img, synth_file, real_img, real_file, corr in top_pairs:
        axes[row, 0].imshow(synth_img, cmap='gray')
        axes[row, 0].set_title(f'synthetic ({cls})', fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].scatter(synth_img.flatten(), real_img.flatten(), s=0.3, alpha=0.15, c='#4477AA')
        axes[row, 1].set_title(f'pearsonr correlation: {corr:.3f}', fontsize=9)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].set_xlabel('synthetic pixel value', fontsize=7)
        axes[row, 1].set_ylabel('real pixel value', fontsize=7)

        axes[row, 2].imshow(real_img, cmap='gray')
        axes[row, 2].set_title(f'real ({cls})', fontsize=9)
        axes[row, 2].axis('off')

        row += 1

plt.suptitle("Batch 2: Synthetic vs. Real Image Pairs with Highest Pearson Correlation\n"
              "(matches comparison-paper Figure 2 style)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("batch2_pearsonr_scatter_pairs.png", dpi=150, bbox_inches='tight')
print("\nSaved: batch2_pearsonr_scatter_pairs.png")
