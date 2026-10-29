# Robustness and generalisation analyses (manuscript Section 2.7, Phase III)

All scripts reuse the code, splits, CLAHE settings, threshold rule and real-anchored standardisation of
`2_eyeclip_adaptation/`. Copy them next to `final_unified_eval.py` (we used a `moore_scripts/` folder)
and run them from the folder that holds the EyeCLIP weights.

| Script | Analysis | Paper item |
|---|---|---|
| `exp_A_clahe_sensitivity.py`, `exp_summary.py` | CLAHE settings: clip limits 1.0, 2.0, 4.0 (8 × 8 tiles) and tiles 4 × 4, 16 × 16 (clip 2.0) | Section 3.2 |
| `exp_B_grouped_cv.py`, `exp_B_summary.py` | Repeated patient-grouped cross-validation, 7 folds × 5 repeats, all 28 patients | Table 2 (right) |
| `exp_C_octdl_clahe.py` | OCTDL, normal against any disease, raw vs CLAHE, patient-grouped 5-fold CV, patient bootstrap | Table 3 (OCTDL) |
| `exp_C2_per_disease.py` | OCTDL, normal against each disease (uses the scores saved by exp_C) | Section 3.3 |
| `exp_D_linear_probe.py` | Linear-probe baseline (C chosen by patient-grouped inner CV) on the five splits and in grouped CV | Tables 2 and 3 |
| `exp_E_retfound_clahe.py` | RETFound OCT weights (frozen), Nearest Centroid and linear probe, Bissig and OCTDL | Table 3 (RETFound) |
| `exp_G_label_efficiency.py` | OCTDL with 5 to 80 labelled patients per class and all training patients | Figure 4 |
| `check_setup.py`, `compare_eyeclip_ckpt.py` | Read-only audit: package versions, EyeCLIP checkpoint loading, text embeddings, data integrity | Methods |
| `run_lp_fix.sh` | Batch script that reruns every linear-probe and RETFound analysis | — |

Result tables are in `../results/robustness/`.

## Data
* Bissig OCT: https://doi.org/10.5061/dryad.msbcc2ftc (CC0), arranged by `data/splits/`.
* OCTDL: Kulyabin et al., Scientific Data 11:365 (2024), Mendeley Data (CC BY 4.0). Pass the class folders
  and the label file: `--root OCTDL --labels OCTDL_labels.csv` (normal class `NO`).
* RETFound OCT weights: Hugging Face `YukunZhou/RETFound_mae_natureOCT`.

## Linear probe protocol
L2-normalised embedding, standardised with the training statistics, then logistic regression with balanced class
weights. C is chosen from 1e-4 to 10 by patient-grouped inner cross-validation on the training images only
(`LP_PROTOCOL=std`, the default and the version reported). A first version (fixed C = 1.0, no standardisation) did
not converge in one split and was discarded.

## RETFound input transform
`--transform native` (default) uses RETFound's own evaluation transform: resize the shorter side to 256 and
crop the centre 224 × 224, ImageNet normalisation. `--transform stretch` resizes the whole B-scan to
224 × 224 (first runs, kept for comparison).
