#!/bin/bash
# Re-runs every linear-probe result after the 25 Oct fix (feature standardisation before logistic regression).
# Run from reset_v2_eyeclip/ inside tmux:   bash moore_scripts/run_lp_fix.sh
# Old prediction files are kept with the suffix _OLD. All results go to lpfix_*.txt.
set -e
CK=~/.cache/huggingface/hub/models--YukunZhou--RETFound_mae_natureOCT/snapshots/8e551e2f547d3e27aedf43c9cf1c5bcea1c17171/RETFound_mae_natureOCT.pth
PERSPLIT="import pandas as pd;from sklearn.metrics import roc_auc_score as A;d=pd.read_csv('predictions_linear_probe.csv');r=d.groupby(['preproc','split']).apply(lambda g:A(g.y_true,g.score)).unstack(0);print(r.round(4));print('improved',(r.clahe>r.raw).sum());print(d.groupby(['preproc','split']).score.agg(['std','nunique']))"

echo "== 3a five splits"; date
[ -f predictions_linear_probe.csv ] && mv predictions_linear_probe.csv predictions_linear_probe_OLD.csv
python moore_scripts/exp_D_linear_probe.py --design main > lpfix_3a.log 2>&1
python moore_scripts/exp_summary.py --pred predictions_linear_probe.csv predictions_final.csv --out linear_probe_summary.csv | grep "Linear Probe" > lpfix_3a.txt
python -c "$PERSPLIT" >> lpfix_3a.txt 2>/dev/null
cat lpfix_3a.txt

echo "== 3b grouped CV"; date
[ -f predictions_grouped_cv_lp.csv ] && mv predictions_grouped_cv_lp.csv predictions_grouped_cv_lp_OLD.csv
python moore_scripts/exp_D_linear_probe.py --design grouped > lpfix_3b.log 2>&1
python moore_scripts/exp_B_summary.py --pred predictions_grouped_cv_lp.csv --out grouped_cv_lp_summary.csv > lpfix_3b.txt
tail -6 lpfix_3b.txt

echo "== RETFound (cached features)"; date
python moore_scripts/exp_E_retfound_clahe.py --ckpt $CK --design bissig > lpfix_rf_bissig.txt 2>&1
python moore_scripts/exp_E_retfound_clahe.py --ckpt $CK --design octdl > lpfix_rf_octdl.txt 2>&1
tail -9 lpfix_rf_bissig.txt; tail -9 lpfix_rf_octdl.txt

echo "== 3c OCTDL (about 30 min)"; date
python moore_scripts/exp_C_octdl_clahe.py --root ~/OCTDL_data/octdl_raw/OCTDL --labels ~/OCTDL_data/OCTDL_labels.csv > lpfix_3c.log 2>&1
tail -6 lpfix_3c.log > lpfix_3c.txt
python moore_scripts/exp_C2_per_disease.py --out octdl_per_disease.csv > lpfix_3c_disease.txt
cat lpfix_3c.txt; grep "Linear" octdl_per_disease.csv
echo "== ALL DONE"; date
