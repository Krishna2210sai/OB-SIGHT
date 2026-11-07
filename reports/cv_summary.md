# Five-Fold Cross-Validation Summary

Stratified 5-fold cross-validation on all 552 records. Pooled metrics use
out-of-fold predictions. Fold mean ± SD shows how much the result depends on
the particular split. These are exploratory estimates; the fixed held-out
test set in `model_evaluation_report.md` is the primary untouched test.

| Model | Pooled ROC-AUC | Fold mean AUC ± SD | Pooled PR-AUC | Fold mean PR-AUC ± SD |
|---|---:|---:|---:|---:|
| Multimodal CNN + clinical MLP | 0.767 | 0.774 ± 0.053 | 0.368 | 0.416 ± 0.039 |
| CTG-only 1D CNN | 0.763 | 0.770 ± 0.039 | 0.327 | 0.385 ± 0.128 |
| CTG-only CNN-BiLSTM | 0.810 | 0.800 ± 0.096 | 0.399 | 0.419 ± 0.131 |
| Clinical-only MLP | 0.561 | 0.552 ± 0.091 | 0.100 | 0.121 ± 0.031 |
| Logistic regression (clinical) | 0.577 | 0.586 ± 0.053 | 0.112 | 0.139 ± 0.023 |
| Logistic regression (CTG descriptors) | 0.744 | 0.744 ± 0.083 | 0.251 | 0.352 ± 0.163 |
| Logistic regression (clinical + CTG) | 0.726 | 0.725 ± 0.065 | 0.305 | 0.380 ± 0.207 |
| Gradient boosting (clinical + CTG) | 0.702 | 0.704 ± 0.071 | 0.217 | 0.263 ± 0.082 |

See `reports/figures/fig_cv_roc_pr.png`.

Cross-validation reuses every record once as validation data, so it is not a
replacement for external validation. The confidence intervals in the held-out
test report remain wide because only 46 cesarean deliveries exist.
