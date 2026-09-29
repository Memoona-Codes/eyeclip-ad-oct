# Patient disjoint evaluation of EyeCLIP adaptation for Alzheimer's disease detection from retinal OCT

Code, patient disjoint split definitions and prediction files for the manuscript of the same title.

## Folders
| Folder | Contents |
|---|---|
| `1_synthetic_generation/` | VAE training, synthetic image generation, intensity rescaling and the quality audit (FID, SSIM, L2, label check, Pearson pairs, histograms) |
| `2_eyeclip_adaptation/` | EyeCLIP features, CLAHE, the ten adaptation methods, threshold choice, slice and patient level evaluation, statistics and figures |
| `data/splits/` | `split_seed0.csv` to `split_seed4.csv`: split_seed, set (train/val), class (AD/CO), patient_id, file. `all_splits.csv` joins them |
| `data/predictions/` | Validation scores and thresholds for every method, input, split and seed (see below) |
| `results/` | Metric tables produced from the prediction files |

## Prediction files (`data/predictions/`)
| File | Contents | Paper item |
|---|---|---|
| `predictions_final.csv` | Real images only, 10 methods, raw and CLAHE | Tables 4, 5; Figure 3 |
| `predictions_synth_batch2.csv` | Real + 500 or 1,000 rescaled synthetic images | Table 6; Supplementary Tables S3, S4; Supplementary Figure S3 |
| `predictions_dosage_batch2.csv` | Standardisation study, real only rule | Table 7; Figure 4 |
| `predictions_dosage_batch2_pooled.csv` | Standardisation study, pooled rule | Table 7; Figure 4 |

Columns: run_time, method, preproc, dosage, split, seed, slice_id, patient_id, condition (DARK/LIGHT), y_true (1 = AD), score (standardised), threshold, patient_threshold.
## Paper items and scripts
| Paper item | Script |
|---|---|
| Table 1 | `data/splits/` |
| Supplementary Table S1, Equation 2 | `1_synthetic_generation/train_v8_vae.py` |
| Supplementary Table S2 | `1_synthetic_generation/generate_batch2_with_audit.py` |
| Equation 7 | `1_synthetic_generation/rescale_synthetic_to_match_real.py` |
| Table 3 | `compute_batch2_fid.py`, `check_batch2_class_correlation.py`, `audit_existing_synthetic.py` |
| Supplementary Figure S1 | `1_synthetic_generation/batch2_pearsonr_scatter_pairs.py` |
| Supplementary Figure S2 | `1_synthetic_generation/plot_pixel_histograms.py` |
| Table 2, Section 3.3 | `2_eyeclip_adaptation/clahe_accuracy_full_metrics.py`, `unified_benchmark_clahe_vs_raw.py`, `threshold_and_patient_level_eval.py` |
| Tables 4, 5 | `final_unified_eval.py`, then `analyze_predictions.py` |
| Table 6, Supplementary Tables S3, S4 | `final_synthetic_all_methods.py`, then `analyze_predictions.py` |
| Table 7 | `final_synthetic_dosage.py` (real only rule) and `final_synthetic_dosage_pooled_rule.py` |
| Figure 3, Supplementary Figure S3 | `analyze_predictions.py` |
| Figure 4 | `2_eyeclip_adaptation/plot_standardisation.py` |

## Reproduce the tables and figures from the prediction files (no GPU needed)
    cd 2_eyeclip_adaptation
    python analyze_predictions.py --pred ../data/predictions/predictions_final.csv ../data/predictions/predictions_synth_batch2.csv --out ../results_check --best "TaskRes+SAM"
    python analyze_predictions.py --pred ../data/predictions/predictions_dosage_batch2.csv --out ../results_check_dosage --best "TaskRes+SAM"
    python plot_standardisation.py

## Rerun the experiments (GPU)
1. Download the OCT images from Dryad (https://doi.org/10.5061/dryad.msbcc2ftc, CC0) and use the raw acquisitions (275 B-scans, 14 AD, 14 CO). Arrange them by the files listed in `data/splits/`.
2. Obtain the EyeCLIP ViT-B/32 weights (image and text encoders) from the EyeCLIP authors. Images and weights are not redistributed here.
3. Run the scripts in `1_synthetic_generation/` from the data root, then the scripts in `2_eyeclip_adaptation/` from inside that folder.
Experiments ran on one NVIDIA A100 (40 GB). `requirements.txt` lists the main packages; `requirements_full.txt` is the full environment.

## Note on the synthetic images
The VAE was trained on images from all 28 patients, and the synthetic source images included every validation patient.
The augmentation results are therefore an optimistic test.

## Citation
[confirm: add the paper reference and Zenodo DOI after acceptance]
