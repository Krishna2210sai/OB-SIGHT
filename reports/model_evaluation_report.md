# Model Evaluation Report

Generated: 2026-11-05T20:24:29+00:00

## Task and data

- **Prediction task:** Binary classification of delivery mode from intrapartum CTG and maternal data
- **Target:** Cesarean delivery (Deliv. type = 2) versus vaginal delivery (Deliv. type = 1)
- **Held-out test set:** 83 records (7 cesarean, 76 vaginal)
- **Test prevalence:** 8.4%

The test set was fixed before training and was never used for model selection, early stopping, calibration, or threshold selection.

## Results at threshold 0.5

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Accuracy | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multimodal CNN + clinical MLP | 0.733 | 0.402 | 0.286 | 0.987 | 0.400 | 0.928 | 0.068 |
| CTG-only 1D CNN | 0.797 | 0.305 | 0.000 | 1.000 | 0.000 | 0.916 | 0.069 |
| CTG-only CNN-BiLSTM | 0.720 | 0.450 | 0.286 | 0.987 | 0.400 | 0.928 | 0.066 |
| Clinical-only MLP | 0.498 | 0.105 | 0.000 | 1.000 | 0.000 | 0.916 | 0.078 |
| Logistic regression (clinical) | 0.468 | 0.103 | 0.571 | 0.487 | 0.160 | 0.494 | 0.265 |
| Logistic regression (CTG descriptors) | 0.701 | 0.204 | 0.714 | 0.763 | 0.333 | 0.759 | 0.180 |
| Logistic regression (clinical + CTG) | 0.611 | 0.184 | 0.286 | 0.763 | 0.148 | 0.723 | 0.184 |
| Gradient boosting (clinical + CTG) | 0.675 | 0.194 | 0.143 | 0.934 | 0.154 | 0.867 | 0.100 |

## Threshold-free uncertainty

| Model | ROC-AUC (95% bootstrap CI) | PR-AUC (95% bootstrap CI) |
|---|---|---|
| Multimodal CNN + clinical MLP | 0.733 (0.551–0.910) | 0.402 (0.129–0.757) |
| CTG-only 1D CNN | 0.797 (0.660–0.914) | 0.305 (0.156–0.672) |
| CTG-only CNN-BiLSTM | 0.720 (0.453–0.938) | 0.450 (0.161–0.808) |
| Clinical-only MLP | 0.498 (0.244–0.727) | 0.105 (0.071–0.272) |
| Logistic regression (clinical) | 0.468 (0.238–0.716) | 0.103 (0.070–0.260) |
| Logistic regression (CTG descriptors) | 0.701 (0.451–0.904) | 0.204 (0.115–0.451) |
| Logistic regression (clinical + CTG) | 0.611 (0.397–0.810) | 0.184 (0.091–0.515) |
| Gradient boosting (clinical + CTG) | 0.675 (0.425–0.882) | 0.194 (0.105–0.467) |

Bootstrap intervals are stratified and resample the 7 cesarean and 76 vaginal test records separately. They describe uncertainty *on this small test set*, not clinical uncertainty in a new hospital.

## Final comparison at validation-selected threshold

| Model | ROC-AUC (95% CI) | PR-AUC | Sensitivity | Specificity | Threshold |
|---|---:|---:|---:|---:|---:|
| Multimodal CNN + clinical MLP | 0.733 (0.551–0.910) | 0.402 | 0.429 | 0.842 | 0.140 |
| CTG-only 1D CNN | 0.797 (0.660–0.914) | 0.305 | 0.571 | 0.684 | 0.080 |
| CTG-only CNN-BiLSTM | 0.720 (0.453–0.938) | 0.450 | 0.571 | 0.750 | 0.080 |
| Clinical-only MLP | 0.498 (0.244–0.727) | 0.105 | 0.143 | 0.829 | 0.110 |
| Logistic regression (clinical) | 0.468 (0.238–0.716) | 0.103 | 0.286 | 0.697 | 0.610 |
| Logistic regression (CTG descriptors) | 0.701 (0.451–0.904) | 0.204 | 0.714 | 0.618 | 0.330 |
| Logistic regression (clinical + CTG) | 0.611 (0.397–0.810) | 0.184 | 0.286 | 0.855 | 0.640 |
| Gradient boosting (clinical + CTG) | 0.675 (0.425–0.882) | 0.194 | 0.286 | 0.895 | 0.370 |

Thresholds were selected on the validation split by maximising Youden's J statistic. The test split was not used for threshold selection.

## Paired AUC comparison

| Comparison | AUC difference | 95% paired bootstrap CI | Interpretation |
|---|---:|---|---|
| CTG-only CNN-BiLSTM minus CTG-only 1D CNN | -0.077 | -0.331 to +0.167 | The paired bootstrap interval includes zero, so this experiment does not establish a statistically meaningful AUC difference between the CNN-BiLSTM and the 1D CNN. |

The paired bootstrap resamples the same test records for both models, preserving the correlation between predictions. With seven test positives the interval is wide and should be read as an uncertainty statement.

## Interpretation

- The best ROC-AUC on this test split was obtained by **CTG-only 1D CNN** (0.797); with only 46 events in the full dataset this ranking is unstable.
- The multimodal model achieved ROC-AUC 0.733 and PR-AUC 0.402 on 7 cesarean cases. At 0.5 it detected 2 of 7 cesareans with 1 false positives among 76 vaginal deliveries.
- The CNN-BiLSTM achieved ROC-AUC 0.720 and PR-AUC 0.450. Its AUC difference versus the 1D CNN was -0.077 (95% paired bootstrap CI -0.331 to +0.167). The paired bootstrap interval includes zero, so this experiment does not establish a statistically meaningful AUC difference between the CNN-BiLSTM and the 1D CNN.
- A classifier that always predicts vaginal delivery would reach 91.6% accuracy while detecting zero cesareans. Accuracy is therefore not a meaningful headline metric here.
- Hand-crafted CTG descriptors and the CNN perform similarly, which suggests the signal information is modest and easily confounded by recording quality.
- No model is suitable for clinical use. The system is a research prototype and a demonstration of an end-to-end pipeline.

## Figures

- `reports/figures/fig_roc_pr_test.png`
- `reports/figures/fig_confusion_matrices_test.png`
- `reports/figures/fig_calibration_test.png`
- `reports/figures/fig_model_comparison_test.png`
- `reports/figures/fig_training_history_multimodal.png`
- `reports/figures/fig_training_history_bilstm.png`
- `reports/figures/fig_class_distribution.png`

## Important limitations

- Only 46 cesarean deliveries exist in the entire public dataset; the test set contains 7.
- The cohort is a selected, homogeneous subset of 9164 recordings from one Czech hospital.
- Cesarean recordings were collected as a separate 46-case subset and differ in signal quality and duration.
- No external validation has been performed.
- These numbers do not establish clinical validity, safety, or effectiveness.
