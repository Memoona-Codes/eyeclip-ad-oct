# 2. EyeCLIP adaptation, CLAHE, slice and patient level evaluation
Run every script from inside this folder, so the shared modules import correctly.

| Step | Script | Output |
|---|---|---|
| 1 | `final_unified_eval.py` | `predictions_final.csv`: 10 methods, raw and CLAHE, 5 splits x 5 seeds |
| 2 | `final_synthetic_all_methods.py --pool <synthetic_dataset_v8_batch2_rescaled>` | `predictions_synth_batch2.csv` |
| 3 | `final_synthetic_dosage.py` | `predictions_dosage_batch2.csv` (real only standardisation) |
| 4 | `final_synthetic_dosage_pooled_rule.py` | pooled standardisation, for comparison |
| 5 | `analyze_predictions.py` | Tables 4 to 7, Supplementary Tables S3, S4, Figure 3, Supplementary Figure S3 |
| 6 | `plot_standardisation.py` | Figure 4 |

Shared modules: `clahe_accuracy_full_metrics.py` (EyeCLIP loading, features, Zero-Shot, Nearest Centroid, Proto-Adapter,
TaskRes), `unified_benchmark_clahe_vs_raw.py` (Tip-Adapter, Tip-Adapter-F, CLIP-Adapter, CoOp),
`threshold_and_patient_level_eval.py` (threshold, patient scores, Tip-Adapter-F+SAM), `pred_logger.py` (prediction files).
