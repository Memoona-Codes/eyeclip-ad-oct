# 1. Synthetic image generation and quality audit
Run from the data root (the folder that holds the image folders).

| Step | Script | Output |
|---|---|---|
| 1 | `train_v8_vae.py` | VAE checkpoint (Supplementary Table S1) |
| 2 | `generate_batch2_with_audit.py` | 500 AD + 500 CO images, seed 43 (Supplementary Table S2) |
| 3 | `rescale_synthetic_to_match_real.py` | Rescaled pool (Equation 7) |
| 4 | `compute_batch2_fid.py`, `check_batch2_class_correlation.py`, `audit_existing_synthetic.py` | FID, nearest neighbour SSIM and L2, label check (Table 3) |
| 5 | `batch2_pearsonr_scatter_pairs.py` | Supplementary Figure S1 |
| 6 | `plot_pixel_histograms.py` | Supplementary Figure S2 |
