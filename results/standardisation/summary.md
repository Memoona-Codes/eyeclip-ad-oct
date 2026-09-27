# Results summary

Source: predictions_dosage_batch2.csv  
Rows: 23,520  
Best method (figures, failure analysis): **TaskRes+SAM [rescaled pool]**

## Pooled patient-level McNemar tests (seed-majority calls), Bonferroni-corrected

| Method A | Method B | both correct | A only | B only | both wrong | discordant | p raw | p Bonferroni | significant (0.05) |
|---|---|---|---|---|---|---|---|---|---|
| TaskRes+SAM [rescaled pool] | TaskRes+SAM [original pool] | 31 | 0 | 0 | 9 | 0 | 1.0000 | 1.0000 | no |

## Synthetic dosage (real validation only)

| level | method | preproc | dosage | runs | AUC | AUC 95% CI | Acc | BalAcc | F1 | Sens | Spec | MCC | collapsed runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| slice | Nearest Centroid [original pool] | clahe | 0 | 5 | 0.766 ± 0.061 | [0.664, 0.854] | 0.653 ± 0.062 | 0.652 ± 0.062 | 0.656 ± 0.081 | 0.671 ± 0.149 | 0.634 ± 0.141 | 0.313 ± 0.122 | 0 |
| slice | Nearest Centroid [original pool] | clahe | 250 | 5 | 0.766 ± 0.055 | [0.661, 0.855] | 0.653 ± 0.054 | 0.652 ± 0.055 | 0.656 ± 0.075 | 0.671 ± 0.141 | 0.634 ± 0.134 | 0.312 ± 0.107 | 0 |
| slice | Nearest Centroid [original pool] | clahe | 500 | 5 | 0.765 ± 0.053 | [0.663, 0.857] | 0.660 ± 0.062 | 0.660 ± 0.063 | 0.666 ± 0.074 | 0.686 ± 0.144 | 0.634 ± 0.146 | 0.328 ± 0.126 | 0 |
| slice | Nearest Centroid [original pool] | clahe | 750 | 5 | 0.762 ± 0.052 | [0.649, 0.851] | 0.660 ± 0.061 | 0.660 ± 0.062 | 0.665 ± 0.081 | 0.686 ± 0.158 | 0.634 ± 0.146 | 0.330 ± 0.123 | 0 |
| slice | Nearest Centroid [original pool] | clahe | 1000 | 5 | 0.763 ± 0.053 | [0.659, 0.853] | 0.653 ± 0.063 | 0.652 ± 0.064 | 0.657 ± 0.080 | 0.675 ± 0.155 | 0.629 ± 0.149 | 0.314 ± 0.129 | 0 |
| slice | Nearest Centroid [rescaled pool] | clahe | 0 | 5 | 0.766 ± 0.061 | [0.662, 0.855] | 0.653 ± 0.062 | 0.652 ± 0.062 | 0.656 ± 0.081 | 0.671 ± 0.149 | 0.634 ± 0.141 | 0.313 ± 0.122 | 0 |
| slice | Nearest Centroid [rescaled pool] | clahe | 250 | 5 | 0.764 ± 0.059 | [0.651, 0.858] | 0.650 ± 0.060 | 0.650 ± 0.061 | 0.655 ± 0.072 | 0.671 ± 0.134 | 0.629 ± 0.138 | 0.306 ± 0.121 | 0 |
| slice | Nearest Centroid [rescaled pool] | clahe | 500 | 5 | 0.765 ± 0.059 | [0.663, 0.853] | 0.650 ± 0.060 | 0.650 ± 0.061 | 0.655 ± 0.072 | 0.671 ± 0.134 | 0.629 ± 0.138 | 0.306 ± 0.121 | 0 |
| slice | Nearest Centroid [rescaled pool] | clahe | 750 | 5 | 0.765 ± 0.060 | [0.670, 0.857] | 0.650 ± 0.060 | 0.650 ± 0.061 | 0.655 ± 0.072 | 0.671 ± 0.134 | 0.629 ± 0.138 | 0.306 ± 0.121 | 0 |
| slice | Nearest Centroid [rescaled pool] | clahe | 1000 | 5 | 0.765 ± 0.060 | [0.661, 0.848] | 0.650 ± 0.060 | 0.650 ± 0.061 | 0.655 ± 0.072 | 0.671 ± 0.134 | 0.629 ± 0.138 | 0.306 ± 0.121 | 0 |
| slice | TaskRes+SAM [original pool] | clahe | 0 | 25 | 0.768 ± 0.053 | [0.673, 0.861] | 0.655 ± 0.048 | 0.655 ± 0.048 | 0.655 ± 0.066 | 0.665 ± 0.140 | 0.644 ± 0.132 | 0.320 ± 0.097 | 0 |
| slice | TaskRes+SAM [original pool] | clahe | 250 | 25 | 0.767 ± 0.049 | [0.660, 0.858] | 0.648 ± 0.059 | 0.647 ± 0.060 | 0.646 ± 0.079 | 0.655 ± 0.146 | 0.639 ± 0.135 | 0.303 ± 0.118 | 0 |
| slice | TaskRes+SAM [original pool] | clahe | 500 | 25 | 0.766 ± 0.048 | [0.663, 0.853] | 0.663 ± 0.049 | 0.662 ± 0.050 | 0.668 ± 0.063 | 0.686 ± 0.126 | 0.639 ± 0.112 | 0.333 ± 0.101 | 0 |
| slice | TaskRes+SAM [original pool] | clahe | 750 | 25 | 0.766 ± 0.047 | [0.664, 0.854] | 0.665 ± 0.049 | 0.665 ± 0.050 | 0.670 ± 0.062 | 0.686 ± 0.126 | 0.644 ± 0.116 | 0.338 ± 0.102 | 0 |
| slice | TaskRes+SAM [original pool] | clahe | 1000 | 25 | 0.766 ± 0.048 | [0.657, 0.852] | 0.665 ± 0.049 | 0.665 ± 0.050 | 0.670 ± 0.062 | 0.686 ± 0.126 | 0.644 ± 0.116 | 0.338 ± 0.102 | 0 |
| slice | TaskRes+SAM [rescaled pool] | clahe | 0 | 25 | 0.768 ± 0.053 | [0.671, 0.853] | 0.655 ± 0.048 | 0.655 ± 0.048 | 0.655 ± 0.066 | 0.665 ± 0.140 | 0.644 ± 0.132 | 0.320 ± 0.097 | 0 |
| slice | TaskRes+SAM [rescaled pool] | clahe | 250 | 25 | 0.766 ± 0.054 | [0.668, 0.852] | 0.653 ± 0.051 | 0.652 ± 0.051 | 0.656 ± 0.059 | 0.666 ± 0.118 | 0.639 ± 0.125 | 0.312 ± 0.101 | 0 |
| slice | TaskRes+SAM [rescaled pool] | clahe | 500 | 25 | 0.765 ± 0.054 | [0.671, 0.856] | 0.650 ± 0.054 | 0.650 ± 0.054 | 0.654 ± 0.064 | 0.666 ± 0.123 | 0.634 ± 0.134 | 0.307 ± 0.106 | 0 |
| slice | TaskRes+SAM [rescaled pool] | clahe | 750 | 25 | 0.765 ± 0.055 | [0.668, 0.858] | 0.650 ± 0.054 | 0.650 ± 0.054 | 0.654 ± 0.064 | 0.666 ± 0.123 | 0.634 ± 0.134 | 0.307 ± 0.106 | 0 |
| slice | TaskRes+SAM [rescaled pool] | clahe | 1000 | 25 | 0.765 ± 0.054 | [0.668, 0.851] | 0.650 ± 0.054 | 0.650 ± 0.054 | 0.654 ± 0.064 | 0.666 ± 0.123 | 0.634 ± 0.134 | 0.307 ± 0.106 | 0 |
| patient | Nearest Centroid [original pool] | clahe | 0 | 5 | 0.812 ± 0.140 | [0.662, 0.938] | 0.775 ± 0.056 | 0.775 ± 0.056 | 0.775 ± 0.071 | 0.800 ± 0.209 | 0.750 ± 0.250 | 0.601 ± 0.102 | 0 |
| patient | Nearest Centroid [original pool] | clahe | 250 | 5 | 0.838 ± 0.122 | [0.700, 0.950] | 0.750 ± 0.088 | 0.750 ± 0.088 | 0.721 ± 0.184 | 0.750 ± 0.306 | 0.750 ± 0.250 | 0.561 ± 0.144 | 0 |
| patient | Nearest Centroid [original pool] | clahe | 500 | 5 | 0.875 ± 0.117 | [0.750, 0.975] | 0.750 ± 0.088 | 0.750 ± 0.088 | 0.721 ± 0.184 | 0.750 ± 0.306 | 0.750 ± 0.250 | 0.561 ± 0.144 | 0 |
| patient | Nearest Centroid [original pool] | clahe | 750 | 5 | 0.850 ± 0.095 | [0.700, 0.963] | 0.750 ± 0.088 | 0.750 ± 0.088 | 0.721 ± 0.184 | 0.750 ± 0.306 | 0.750 ± 0.250 | 0.561 ± 0.144 | 0 |
| patient | Nearest Centroid [original pool] | clahe | 1000 | 5 | 0.850 ± 0.095 | [0.713, 0.963] | 0.750 ± 0.088 | 0.750 ± 0.088 | 0.721 ± 0.184 | 0.750 ± 0.306 | 0.750 ± 0.250 | 0.561 ± 0.144 | 0 |
| patient | Nearest Centroid [rescaled pool] | clahe | 0 | 5 | 0.812 ± 0.140 | [0.662, 0.950] | 0.775 ± 0.056 | 0.775 ± 0.056 | 0.775 ± 0.071 | 0.800 ± 0.209 | 0.750 ± 0.250 | 0.601 ± 0.102 | 0 |
| patient | Nearest Centroid [rescaled pool] | clahe | 250 | 5 | 0.800 ± 0.149 | [0.637, 0.938] | 0.775 ± 0.056 | 0.775 ± 0.056 | 0.775 ± 0.071 | 0.800 ± 0.209 | 0.750 ± 0.250 | 0.601 ± 0.102 | 0 |
| patient | Nearest Centroid [rescaled pool] | clahe | 500 | 5 | 0.800 ± 0.149 | [0.637, 0.938] | 0.775 ± 0.056 | 0.775 ± 0.056 | 0.775 ± 0.071 | 0.800 ± 0.209 | 0.750 ± 0.250 | 0.601 ± 0.102 | 0 |
| patient | Nearest Centroid [rescaled pool] | clahe | 750 | 5 | 0.800 ± 0.149 | [0.650, 0.938] | 0.775 ± 0.056 | 0.775 ± 0.056 | 0.775 ± 0.071 | 0.800 ± 0.209 | 0.750 ± 0.250 | 0.601 ± 0.102 | 0 |
| patient | Nearest Centroid [rescaled pool] | clahe | 1000 | 5 | 0.800 ± 0.149 | [0.650, 0.938] | 0.775 ± 0.056 | 0.775 ± 0.056 | 0.775 ± 0.071 | 0.800 ± 0.209 | 0.750 ± 0.250 | 0.601 ± 0.102 | 0 |
| patient | TaskRes+SAM [original pool] | clahe | 0 | 25 | 0.825 ± 0.124 | [0.662, 0.950] | 0.775 ± 0.051 | 0.775 ± 0.051 | 0.775 ± 0.065 | 0.800 ± 0.191 | 0.750 ± 0.228 | 0.601 ± 0.094 | 0 |
| patient | TaskRes+SAM [original pool] | clahe | 250 | 25 | 0.850 ± 0.095 | [0.713, 0.975] | 0.750 ± 0.081 | 0.750 ± 0.081 | 0.721 ± 0.168 | 0.750 ± 0.280 | 0.750 ± 0.228 | 0.561 ± 0.132 | 0 |
| patient | TaskRes+SAM [original pool] | clahe | 500 | 25 | 0.875 ± 0.107 | [0.738, 0.975] | 0.750 ± 0.081 | 0.750 ± 0.081 | 0.721 ± 0.168 | 0.750 ± 0.280 | 0.750 ± 0.228 | 0.561 ± 0.132 | 0 |
| patient | TaskRes+SAM [original pool] | clahe | 750 | 25 | 0.875 ± 0.107 | [0.750, 0.975] | 0.750 ± 0.081 | 0.750 ± 0.081 | 0.721 ± 0.168 | 0.750 ± 0.280 | 0.750 ± 0.228 | 0.561 ± 0.132 | 0 |
| patient | TaskRes+SAM [original pool] | clahe | 1000 | 25 | 0.875 ± 0.107 | [0.750, 0.975] | 0.750 ± 0.081 | 0.750 ± 0.081 | 0.721 ± 0.168 | 0.750 ± 0.280 | 0.750 ± 0.228 | 0.561 ± 0.132 | 0 |
| patient | TaskRes+SAM [rescaled pool] | clahe | 0 | 25 | 0.825 ± 0.124 | [0.675, 0.950] | 0.775 ± 0.051 | 0.775 ± 0.051 | 0.775 ± 0.065 | 0.800 ± 0.191 | 0.750 ± 0.228 | 0.601 ± 0.094 | 0 |
| patient | TaskRes+SAM [rescaled pool] | clahe | 250 | 25 | 0.825 ± 0.124 | [0.662, 0.950] | 0.775 ± 0.051 | 0.775 ± 0.051 | 0.775 ± 0.065 | 0.800 ± 0.191 | 0.750 ± 0.228 | 0.601 ± 0.094 | 0 |
| patient | TaskRes+SAM [rescaled pool] | clahe | 500 | 25 | 0.825 ± 0.124 | [0.675, 0.950] | 0.775 ± 0.051 | 0.775 ± 0.051 | 0.775 ± 0.065 | 0.800 ± 0.191 | 0.750 ± 0.228 | 0.601 ± 0.094 | 0 |
| patient | TaskRes+SAM [rescaled pool] | clahe | 750 | 25 | 0.825 ± 0.124 | [0.675, 0.950] | 0.775 ± 0.051 | 0.775 ± 0.051 | 0.775 ± 0.065 | 0.800 ± 0.191 | 0.750 ± 0.228 | 0.601 ± 0.094 | 0 |
| patient | TaskRes+SAM [rescaled pool] | clahe | 1000 | 25 | 0.825 ± 0.124 | [0.662, 0.950] | 0.775 ± 0.051 | 0.775 ± 0.051 | 0.775 ± 0.065 | 0.800 ± 0.191 | 0.750 ± 0.228 | 0.601 ± 0.094 | 0 |

## Confusion matrices at each dose (TN, FP, FN, TP)

| method | preproc | dosage | level | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|
| Nearest Centroid [original pool] | clahe | 0 | slice | 133 | 65 | 71 | 123 |
| Nearest Centroid [original pool] | clahe | 250 | slice | 133 | 65 | 71 | 123 |
| Nearest Centroid [original pool] | clahe | 500 | slice | 136 | 62 | 71 | 123 |
| Nearest Centroid [original pool] | clahe | 750 | slice | 136 | 62 | 71 | 123 |
| Nearest Centroid [original pool] | clahe | 1000 | slice | 134 | 64 | 72 | 122 |
| Nearest Centroid [rescaled pool] | clahe | 0 | slice | 133 | 65 | 71 | 123 |
| Nearest Centroid [rescaled pool] | clahe | 250 | slice | 133 | 65 | 72 | 122 |
| Nearest Centroid [rescaled pool] | clahe | 500 | slice | 133 | 65 | 72 | 122 |
| Nearest Centroid [rescaled pool] | clahe | 750 | slice | 133 | 65 | 72 | 122 |
| Nearest Centroid [rescaled pool] | clahe | 1000 | slice | 133 | 65 | 72 | 122 |
| TaskRes+SAM [original pool] | clahe | 0 | slice | 132 | 66 | 69 | 125 |
| TaskRes+SAM [original pool] | clahe | 250 | slice | 130 | 68 | 70 | 124 |
| TaskRes+SAM [original pool] | clahe | 500 | slice | 136 | 62 | 70 | 124 |
| TaskRes+SAM [original pool] | clahe | 750 | slice | 136 | 62 | 69 | 125 |
| TaskRes+SAM [original pool] | clahe | 1000 | slice | 136 | 62 | 69 | 125 |
| TaskRes+SAM [rescaled pool] | clahe | 0 | slice | 132 | 66 | 69 | 125 |
| TaskRes+SAM [rescaled pool] | clahe | 250 | slice | 132 | 66 | 70 | 124 |
| TaskRes+SAM [rescaled pool] | clahe | 500 | slice | 132 | 66 | 71 | 123 |
| TaskRes+SAM [rescaled pool] | clahe | 750 | slice | 132 | 66 | 71 | 123 |
| TaskRes+SAM [rescaled pool] | clahe | 1000 | slice | 132 | 66 | 71 | 123 |
| Nearest Centroid [original pool] | clahe | 0 | patient | 16 | 4 | 5 | 15 |
| Nearest Centroid [original pool] | clahe | 250 | patient | 15 | 5 | 5 | 15 |
| Nearest Centroid [original pool] | clahe | 500 | patient | 15 | 5 | 5 | 15 |
| Nearest Centroid [original pool] | clahe | 750 | patient | 15 | 5 | 5 | 15 |
| Nearest Centroid [original pool] | clahe | 1000 | patient | 15 | 5 | 5 | 15 |
| Nearest Centroid [rescaled pool] | clahe | 0 | patient | 16 | 4 | 5 | 15 |
| Nearest Centroid [rescaled pool] | clahe | 250 | patient | 16 | 4 | 5 | 15 |
| Nearest Centroid [rescaled pool] | clahe | 500 | patient | 16 | 4 | 5 | 15 |
| Nearest Centroid [rescaled pool] | clahe | 750 | patient | 16 | 4 | 5 | 15 |
| Nearest Centroid [rescaled pool] | clahe | 1000 | patient | 16 | 4 | 5 | 15 |
| TaskRes+SAM [original pool] | clahe | 0 | patient | 16 | 4 | 5 | 15 |
| TaskRes+SAM [original pool] | clahe | 250 | patient | 15 | 5 | 5 | 15 |
| TaskRes+SAM [original pool] | clahe | 500 | patient | 15 | 5 | 5 | 15 |
| TaskRes+SAM [original pool] | clahe | 750 | patient | 15 | 5 | 5 | 15 |
| TaskRes+SAM [original pool] | clahe | 1000 | patient | 15 | 5 | 5 | 15 |
| TaskRes+SAM [rescaled pool] | clahe | 0 | patient | 16 | 4 | 5 | 15 |
| TaskRes+SAM [rescaled pool] | clahe | 250 | patient | 16 | 4 | 5 | 15 |
| TaskRes+SAM [rescaled pool] | clahe | 500 | patient | 16 | 4 | 5 | 15 |
| TaskRes+SAM [rescaled pool] | clahe | 750 | patient | 16 | 4 | 5 | 15 |
| TaskRes+SAM [rescaled pool] | clahe | 1000 | patient | 16 | 4 | 5 | 15 |

## Confusion matrices, TaskRes+SAM [rescaled pool] (seed-majority calls summed over splits)

| method | preproc | level | TP | FN | FP | TN |
|---|---|---|---|---|---|---|
| TaskRes+SAM [rescaled pool] | clahe | slice | 132 | 66 | 69 | 125 |
| TaskRes+SAM [rescaled pool] | clahe | patient | 16 | 4 | 5 | 15 |

## Failure analysis (TaskRes+SAM [rescaled pool], CLAHE, patient level)

6 of 22 patients misclassified in more than half of their evaluations: AD443 (AD, 10/10), AD057 (AD, 5/5), AD362 (AD, 5/5), CO015 (CO, 5/5), CO091 (CO, 5/5), CO928 (CO, 5/5).

## Figures

- fig_roc_confusion.pdf / .png
- fig_confusion_raw_vs_clahe.pdf / .png
- fig_dosage_roc_confusion_clahe / _raw .pdf / .png (real, +500, +1000)
