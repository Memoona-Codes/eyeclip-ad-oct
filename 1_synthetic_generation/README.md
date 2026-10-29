# 1. Synthetic image generation and quality audit
Run from the data root (the folder that holds the image folders).

## Original pool (generator trained on all 28 patients)
| Step | Script | Output |
|---|---|---|
| 1 | `train_v8_vae.py` | VAE checkpoint (Supplementary Table S1) |
| 2 | `generate_batch2_with_audit.py` | 500 AD + 500 CO images, seed 43 |
| 3 | `rescale_synthetic_to_match_real.py` | Rescaled pool (Equation 4) |
| 4 | `compute_batch2_fid.py`, `check_batch2_class_correlation.py`, `audit_existing_synthetic.py` | FID, nearest-neighbour SSIM and L2, label check (Table 4) |
| 5 | `fig2_real_vs_synthetic.py`, `batch2_pearsonr_scatter_pairs.py`, `plot_pixel_histograms.py` | Supplementary Figure S1 and audit figures |

## Leakage-free pools (one generator per split, training patients only)
| Step | Script | Output |
|---|---|---|
| 1 | `exp_H_vae_per_split.py` | One VAE per split trained on its 20 training patients (15% held out to choose the epoch); near-copy pool (original sampling), rescaled to the split's training images |
| 2 | `exp_H2_regenerate.py --relative --sigma 1` | New-image pool: reuses the per-split VAEs, mixes two training patients of the same class with w ~ U(0.3, 0.7) and latent noise equal to each dimension's SD |
| 3 | `audit_existing_synthetic.py --leakfree --lf-name <pool>` | Audit against each split's training images |
| 4 | `../2_eyeclip_adaptation/final_synthetic_all_methods.py --pool "<pool>_split{split}_rescaled"` | Prediction files `predictions_synth_leakfree_*.csv` |

Validation patients are never read by any script in this section.
