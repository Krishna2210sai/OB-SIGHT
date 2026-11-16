# Phase B — Baseline reproduction and imbalance-aware learning

**Project:** OB-SIGHT intrapartum cardiotocography delivery-mode prediction
**Scope:** Stage B0 (baseline reproduction) and Stage B1 (single-strategy imbalance experiments)
**Data:** development set only — 469 records, 39 cesareans
**Preprocessing:** 30-minute CTG window, 1.0 Hz
**Validation:** nested patient/record-level cross-validation, outer 5 folds, inner 3 folds, seed 42
**Primary threshold:** 0.5; PR-AUC and cesarean sensitivity emphasised
**Held-out test split:** not loaded and not evaluated

> Every value below is read from generated experiment artifacts. Runs that did not
> complete are marked unavailable rather than filled in.

## 1. Baseline reproduction

Stage B0 reruns the six existing model families on the fixed Phase A outer folds with the same 30-minute / 1 Hz preprocessing, seed 42 and threshold 0.5. Deep models use the production weighted binary cross-entropy with a fixed learning rate of 1e-3 and a **25-epoch budget with no outer-validation early stopping**; this is a common reproduction protocol, not the longer production schedule (80–120 epochs with early stopping), so small differences from the production numbers are expected. Classical models use the production `class_weight='balanced'` settings. No imbalance strategy is applied in B0.

| Model | Family | ROC-AUC | 95% CI | PR-AUC | 95% CI | Sensitivity | Specificity | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| CTG-only 1D CNN | deep | 0.751 | 0.646–0.848 | 0.335 | 0.217–0.498 | 0.256 | 0.965 | 0.312 | 0.272 | 0.081 |
| CNN-BiLSTM | deep | 0.801 | 0.709–0.882 | 0.391 | 0.264–0.550 | 0.487 | 0.937 | 0.447 | 0.394 | 0.082 |
| Multimodal CNN + Clinical MLP | deep | 0.761 | 0.660–0.846 | 0.355 | 0.236–0.501 | 0.359 | 0.940 | 0.354 | 0.295 | 0.091 |
| Clinical-only MLP | deep | 0.574 | 0.480–0.676 | 0.154 | 0.096–0.263 | 0.436 | 0.644 | 0.163 | 0.046 | 0.211 |
| Logistic regression | classical | 0.745 | 0.657–0.825 | 0.338 | 0.225–0.509 | 0.564 | 0.798 | 0.297 | 0.237 | 0.158 |
| Histogram gradient boosting | classical | 0.708 | 0.602–0.805 | 0.296 | 0.185–0.454 | 0.385 | 0.949 | 0.395 | 0.342 | 0.083 |

Historical reference values. These were produced with different protocols and are **not directly comparable** to B0; they are shown only to confirm that B0 is in the expected range.

| Source | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Phase A deep screening (multimodal, 30 min/1 Hz, 35 epochs) | 0.761 | 0.357 | 0.308 | 0.960 |
| Phase A window study (logistic, 30 min/1 Hz) | 0.745 | 0.338 | 0.564 | 0.798 |
| Phase A window study (gradient_boosting, 30 min/1 Hz) | 0.714 | 0.298 | 0.385 | 0.949 |
| Production record-level 5-fold CV (signal) | 0.763 | 0.327 | 0.348 | 0.937 |
| Production record-level 5-fold CV (bilstm) | 0.810 | 0.399 | 0.522 | 0.931 |
| Production record-level 5-fold CV (multimodal) | 0.767 | 0.368 | 0.348 | 0.935 |
| Production record-level 5-fold CV (clinical) | 0.561 | 0.100 | 0.348 | 0.753 |

The Phase A classical window study used the same folds and tabular features as B0, so its logistic regression result should reproduce almost exactly; the deep and production values use different epoch budgets, splits or thresholding and are context only.

## 2. Class imbalance analysis

- Development records: **469**.
- Cesarean deliveries (minority): **39** (8.3%).
- Vaginal deliveries (majority): **430** (91.7%).
- Negative-to-positive ratio: **11.0:1**.
- Positive class prevalence: **0.0832**.
- A no-skill classifier that always predicts vaginal delivery scores accuracy 0.917, balanced accuracy 0.500, cesarean sensitivity 0.000, PR-AUC 0.083 and MCC 0.000.

Outer fold composition (fixed Phase A patient/record-level folds):

| Fold | Records | Cesareans | Vaginal | Cesarean % |
|---|---|---|---|---|
| 1 | 94 | 8 | 86 | 8.5% |
| 2 | 94 | 8 | 86 | 8.5% |
| 3 | 94 | 8 | 86 | 8.5% |
| 4 | 94 | 8 | 86 | 8.5% |
| 5 | 93 | 7 | 86 | 7.5% |

With 7–8 cesareans per outer validation fold, a single misclassified cesarean moves fold sensitivity by 12.5–14.3 percentage points. Fold-level sensitivity is therefore reported for transparency but should not be interpreted as a stable estimate.

The minority class is small enough that loss reweighting can trade a large amount of specificity for a small absolute gain in true positives. For that reason PR-AUC, cesarean sensitivity at a fixed threshold, and calibration are analysed jointly rather than optimising accuracy.

## 3. Experimental methodology

### 3.1 Data and preprocessing

- Development set only: **469 records** (39 cesareans).
  The 83-record held-out test split is not loaded anywhere in `src/phase_b/`; the loader
  raises an exception if a `test` record is present.
- Phase A production preprocessing recommendation: **30-minute CTG window at 1 Hz**
  (1800 samples per channel), existing CTG feature extraction and existing clinical
  feature extraction (`32` clinical inputs after encoding, `40` tabular features).
- Signal standardisation and clinical imputation/scaling are fitted **inside each
  training partition** and applied unchanged to the corresponding validation partition.
- Patient-level separation follows the fixed Phase A outer folds (`experiments/phase_a/folds.csv`),
  which are preserved unchanged for historical comparison.

### 3.2 Nested cross-validation

- **Outer**: 5 stratified folds, fixed Phase A assignments, seed 42.
- **Inner**: 3 stratified folds *inside each outer training partition*, used only for
  model/hyperparameter selection.
- Classical models: inner folds select `C` for logistic regression or
  `max_depth`/`learning_rate` for gradient boosting, using mean inner-fold ROC-AUC
  (PR-AUC tie-break).
- Deep models: inner folds select the learning rate from `(0.0005, 0.001)` and an epoch
  budget (mean best epoch across inner folds) for the **weighted-BCE** configuration.
  Those settings are then reused for all five strategies of that architecture, so the
  strategy comparison is not confounded by a different tuning budget per strategy.
- The outer model is refit on the full outer-training partition for the selected epoch
  count with **no outer-validation early stopping**. The outer validation fold is
  scored exactly once per fold.
- Class weights, sampling weights, oversampling and SMOTE are computed from the
  outer-training labels only. The weighted sampler is created over the training dataset
  only; validation data are never resampled.

### 3.3 Deep imbalance strategies (one at a time)

| Strategy | Loss | Sampling | Notes |
|---|---|---|---|
| Plain BCE | `BCEWithLogitsLoss` | shuffle | Neutral reference |
| Weighted BCE | `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` | shuffle | Production loss |
| Focal loss (gamma=1) | `FocalLoss(gamma=1, alpha=n_neg/n)` | shuffle | Focal, moderate |
| Focal loss (gamma=2) | `FocalLoss(gamma=2, alpha=n_neg/n)` | shuffle | Focal, aggressive |
| Weighted random sampling | `BCEWithLogitsLoss` | WeightedRandomSampler(inverse class frequency) | Sampling, no loss reweighting |


Classical strategies:

| Strategy | Implementation | Constraint |
|---|---|---|
| No rebalancing | plain fit | reference |
| Class weights | `class_weight='balanced'` in the estimator | production classical baseline |
| Random oversampling | minority rows sampled with replacement to parity | training partition only |
| SMOTE | custom tabular SMOTE, k=5, interpolation to parity | training partition only; never raw CTG tensors |


### 3.4 Metrics and thresholds

- Primary operating point: **threshold 0.5**, fixed before evaluation.
- Threshold-free metrics (ROC-AUC, PR-AUC, Brier, ECE) are the primary imbalance
  comparison; they do not depend on the 0.5 cut-off.
- Inner-fold-selected thresholds are stored and reported **secondarily**; they are
  never fitted on the outer validation fold.
- Uncertainty: stratified bootstrap (1000 resamples, seed 42) for pooled 95% CIs, and
  paired stratified bootstrap for strategy differences against the weighted-BCE
  reference (deep) or class-weighted reference (classical).
- Every run records fold assignments, configuration, losses, class weights, sampler
  configuration, seed, checkpoints, predictions, metrics, histories and logs under
  `experiments/phase_b/`.

## 4. Comparison of imbalance strategies

Deep strategies, mean over the four architectures (pooled out-of-fold metrics):

| Strategy | Mean ROC-AUC | Mean PR-AUC | Mean sensitivity | Mean specificity | Mean MCC | Mean Brier | Mean ECE | Models |
|---|---|---|---|---|---|---|---|---|
| Plain BCE | 0.720 | 0.261 | 0.019 | 0.996 | 0.054 | 0.072 | 0.041 | 4 |
| Weighted BCE | 0.716 | 0.258 | 0.494 | 0.813 | 0.239 | 0.138 | 0.178 | 4 |
| Focal loss (gamma=1) | 0.715 | 0.268 | 0.538 | 0.791 | 0.234 | 0.153 | 0.247 | 4 |
| Focal loss (gamma=2) | 0.711 | 0.272 | 0.583 | 0.760 | 0.231 | 0.171 | 0.298 | 4 |
| Weighted random sampling | 0.728 | 0.290 | 0.449 | 0.872 | 0.287 | 0.119 | 0.139 | 4 |

Differences relative to weighted BCE (positive PR-AUC/sensitivity means an improvement; positive Brier means worse calibration):

| Strategy | Δ PR-AUC | Δ sensitivity | Δ specificity | Δ Brier |
|---|---|---|---|---|
| Plain BCE | +0.004 | -0.474 | +0.183 | -0.066 |
| Weighted BCE | +0.000 | +0.000 | +0.000 | +0.000 |
| Focal loss (gamma=1) | +0.011 | +0.045 | -0.022 | +0.015 |
| Focal loss (gamma=2) | +0.014 | +0.090 | -0.052 | +0.033 |
| Weighted random sampling | +0.032 | -0.045 | +0.059 | -0.019 |

Classical models by tabular strategy:

| Model | Strategy | ROC-AUC | PR-AUC | Sensitivity | Specificity | MCC | Brier |
|---|---|---|---|---|---|---|---|
| Logistic regression | No rebalancing | 0.760 | 0.342 | 0.179 | 0.991 | 0.310 | 0.065 |
| Logistic regression | Class weights | 0.727 | 0.286 | 0.564 | 0.807 | 0.246 | 0.158 |
| Logistic regression | Random oversampling | 0.730 | 0.284 | 0.564 | 0.812 | 0.251 | 0.157 |
| Logistic regression | SMOTE (tabular) | 0.736 | 0.300 | 0.564 | 0.809 | 0.248 | 0.152 |
| Histogram gradient boosting | No rebalancing | 0.672 | 0.269 | 0.128 | 0.988 | 0.223 | 0.071 |
| Histogram gradient boosting | Class weights | 0.704 | 0.337 | 0.359 | 0.951 | 0.326 | 0.079 |
| Histogram gradient boosting | Random oversampling | 0.703 | 0.260 | 0.282 | 0.958 | 0.275 | 0.079 |
| Histogram gradient boosting | SMOTE (tabular) | 0.736 | 0.292 | 0.333 | 0.935 | 0.262 | 0.088 |

SMOTE and random oversampling are applied to the standardised tabular matrix only. The raw `(4, 1800)` CTG tensors are never interpolated or oversampled.

Paired bootstrap differences against weighted BCE whose 95% CI excludes zero (top 15 by p-value). These are unadjusted for multiple comparisons and should be read as exploratory:

| Model | Strategy | Metric | Difference | 95% CI | p |
|---|---|---|---|---|---|
| CTG-only 1D CNN | Plain BCE | Cesarean sensitivity | -0.436 | -0.590 to -0.282 | 0.000 |
| CTG-only 1D CNN | Focal loss (gamma=2) | Cesarean sensitivity | +0.179 | +0.077 to +0.308 | 0.000 |
| CNN-BiLSTM | Plain BCE | Cesarean sensitivity | -0.641 | -0.795 to -0.487 | 0.000 |
| Multimodal CNN + Clinical MLP | Plain BCE | Cesarean sensitivity | -0.333 | -0.462 to -0.179 | 0.000 |
| Clinical-only MLP | Plain BCE | Cesarean sensitivity | -0.487 | -0.641 to -0.333 | 0.000 |
| Multimodal CNN + Clinical MLP | Focal loss (gamma=1) | Cesarean sensitivity | +0.154 | +0.051 to +0.282 | 0.000 |
| Multimodal CNN + Clinical MLP | Focal loss (gamma=2) | Cesarean sensitivity | +0.179 | +0.051 to +0.333 | 0.020 |
| CTG-only 1D CNN | Weighted random sampling | PR-AUC | +0.092 | +0.016 to +0.160 | 0.024 |

**Recommendation rule.** Highest mean PR-AUC with mean ROC-AUC within 0.01 of the best. Selected strategy: **Weighted random sampling** (mean PR-AUC 0.290, mean ROC-AUC 0.728, mean sensitivity 0.449, mean specificity 0.872).

## 5. Model-wise results

### CTG-only 1D CNN (`signal`)

| Strategy | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|---|
| Plain BCE | 0.731 | 0.260 | 0.026 | 0.998 | 0.500 | 0.049 | 0.099 | 0.071 |
| Weighted BCE | 0.715 | 0.223 | 0.462 | 0.865 | 0.237 | 0.313 | 0.245 | 0.120 |
| Focal loss (gamma=1) | 0.727 | 0.226 | 0.487 | 0.821 | 0.198 | 0.281 | 0.211 | 0.143 |
| Focal loss (gamma=2) | 0.723 | 0.230 | 0.641 | 0.721 | 0.172 | 0.272 | 0.216 | 0.179 |
| Weighted random sampling | 0.754 | 0.315 | 0.385 | 0.958 | 0.455 | 0.417 | 0.370 | 0.080 |

- Best PR-AUC: **Weighted random sampling** (0.315) with sensitivity 0.385 and specificity 0.958.
- Highest sensitivity: **Focal loss (gamma=2)** (0.641) with PR-AUC 0.230 and specificity 0.721.

### CNN-BiLSTM (`bilstm`)

| Strategy | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|---|
| Plain BCE | 0.813 | 0.314 | 0.026 | 0.991 | 0.200 | 0.045 | 0.044 | 0.067 |
| Weighted BCE | 0.825 | 0.323 | 0.667 | 0.888 | 0.351 | 0.460 | 0.420 | 0.107 |
| Focal loss (gamma=1) | 0.824 | 0.361 | 0.667 | 0.844 | 0.280 | 0.394 | 0.354 | 0.123 |
| Focal loss (gamma=2) | 0.821 | 0.374 | 0.667 | 0.830 | 0.263 | 0.377 | 0.336 | 0.130 |
| Weighted random sampling | 0.818 | 0.404 | 0.615 | 0.895 | 0.348 | 0.444 | 0.398 | 0.101 |

- Best PR-AUC: **Weighted random sampling** (0.404) with sensitivity 0.615 and specificity 0.895.
- Highest sensitivity: **Weighted BCE** (0.667) with PR-AUC 0.323 and specificity 0.888.

### Multimodal CNN + Clinical MLP (`multimodal`)

| Strategy | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|---|
| Plain BCE | 0.763 | 0.334 | 0.026 | 0.995 | 0.333 | 0.048 | 0.073 | 0.066 |
| Weighted BCE | 0.752 | 0.340 | 0.359 | 0.921 | 0.292 | 0.322 | 0.255 | 0.090 |
| Focal loss (gamma=1) | 0.740 | 0.342 | 0.513 | 0.891 | 0.299 | 0.377 | 0.318 | 0.116 |
| Focal loss (gamma=2) | 0.733 | 0.341 | 0.538 | 0.877 | 0.284 | 0.372 | 0.314 | 0.145 |
| Weighted random sampling | 0.749 | 0.294 | 0.282 | 0.956 | 0.367 | 0.319 | 0.268 | 0.086 |

- Best PR-AUC: **Focal loss (gamma=1)** (0.342) with sensitivity 0.513 and specificity 0.891.
- Highest sensitivity: **Focal loss (gamma=2)** (0.538) with PR-AUC 0.341 and specificity 0.877.

### Clinical-only MLP (`clinical`)

| Strategy | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|---|
| Plain BCE | 0.573 | 0.136 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.086 |
| Weighted BCE | 0.571 | 0.144 | 0.487 | 0.577 | 0.095 | 0.158 | 0.036 | 0.234 |
| Focal loss (gamma=1) | 0.567 | 0.144 | 0.487 | 0.607 | 0.101 | 0.167 | 0.053 | 0.231 |
| Focal loss (gamma=2) | 0.565 | 0.141 | 0.487 | 0.614 | 0.103 | 0.170 | 0.057 | 0.231 |
| Weighted random sampling | 0.592 | 0.146 | 0.513 | 0.677 | 0.126 | 0.202 | 0.111 | 0.210 |

- Best PR-AUC: **Weighted random sampling** (0.146) with sensitivity 0.513 and specificity 0.677.
- Highest sensitivity: **Weighted random sampling** (0.513) with PR-AUC 0.146 and specificity 0.677.

## 6. Minority-class detection performance

Cesarean detection receives particular attention. PR-AUC is the primary ranking metric because prevalence is only 8.3%; a model that ranks cesareans poorly cannot achieve high PR-AUC regardless of threshold. Sensitivity is reported at the fixed 0.5 threshold and, secondarily, at the inner-fold-selected threshold.

| Model | Strategy | PR-AUC | Sensitivity @0.5 | Cesareans detected | Specificity @0.5 | Sensitivity @inner threshold | Specificity @inner threshold |
|---|---|---|---|---|---|---|---|
| CTG-only 1D CNN | Plain BCE | 0.260 | 0.026 | 1/39 | 0.998 | 0.462 | 0.874 |
| CTG-only 1D CNN | Weighted BCE | 0.223 | 0.462 | 18/39 | 0.865 | 0.718 | 0.712 |
| CTG-only 1D CNN | Focal loss (gamma=1) | 0.226 | 0.487 | 19/39 | 0.821 | 0.846 | 0.319 |
| CTG-only 1D CNN | Focal loss (gamma=2) | 0.230 | 0.641 | 25/39 | 0.721 | 0.923 | 0.116 |
| CTG-only 1D CNN | Weighted random sampling | 0.315 | 0.385 | 15/39 | 0.958 | 0.538 | 0.888 |
| CNN-BiLSTM | Plain BCE | 0.314 | 0.026 | 1/39 | 0.991 | 0.308 | 0.937 |
| CNN-BiLSTM | Weighted BCE | 0.323 | 0.667 | 26/39 | 0.888 | 0.769 | 0.830 |
| CNN-BiLSTM | Focal loss (gamma=1) | 0.361 | 0.667 | 26/39 | 0.844 | 0.872 | 0.719 |
| CNN-BiLSTM | Focal loss (gamma=2) | 0.374 | 0.667 | 26/39 | 0.830 | 0.846 | 0.647 |
| CNN-BiLSTM | Weighted random sampling | 0.404 | 0.615 | 24/39 | 0.895 | 0.667 | 0.851 |
| Multimodal CNN + Clinical MLP | Plain BCE | 0.334 | 0.026 | 1/39 | 0.995 | 0.564 | 0.865 |
| Multimodal CNN + Clinical MLP | Weighted BCE | 0.340 | 0.359 | 14/39 | 0.921 | 0.641 | 0.767 |
| Multimodal CNN + Clinical MLP | Focal loss (gamma=1) | 0.342 | 0.513 | 20/39 | 0.891 | 0.846 | 0.244 |
| Multimodal CNN + Clinical MLP | Focal loss (gamma=2) | 0.341 | 0.538 | 21/39 | 0.877 | 0.974 | 0.016 |
| Multimodal CNN + Clinical MLP | Weighted random sampling | 0.294 | 0.282 | 11/39 | 0.956 | 0.487 | 0.872 |
| Clinical-only MLP | Plain BCE | 0.136 | 0.000 | 0/39 | 1.000 | 0.000 | 1.000 |
| Clinical-only MLP | Weighted BCE | 0.144 | 0.487 | 19/39 | 0.577 | 0.487 | 0.600 |
| Clinical-only MLP | Focal loss (gamma=1) | 0.144 | 0.487 | 19/39 | 0.607 | 0.513 | 0.572 |
| Clinical-only MLP | Focal loss (gamma=2) | 0.141 | 0.487 | 19/39 | 0.614 | 0.513 | 0.540 |
| Clinical-only MLP | Weighted random sampling | 0.146 | 0.513 | 20/39 | 0.677 | 0.410 | 0.700 |

The inner-threshold columns show what the operating point chosen without outer validation data would have produced. Where sensitivity rises sharply at the inner threshold, the ranking information was present but the 0.5 cut-off was miscalibrated. Conversely, a very high inner-threshold sensitivity paired with near-zero specificity shows that a threshold transferred from inner folds is not a substitute for outer-fold calibration; it can over-predict the minority class when the score distribution shifts.

### Per-fold sensitivity, recommended strategy (Weighted random sampling)

| Model | Fold | Val cesareans | Sensitivity | Specificity | Fold PR-AUC |
|---|---|---|---|---|---|
| CTG-only 1D CNN | 1 | 8 | 0.375 | 0.907 | 0.165 |
| CTG-only 1D CNN | 2 | 8 | 0.375 | 0.965 | 0.265 |
| CTG-only 1D CNN | 3 | 8 | 0.375 | 0.988 | 0.530 |
| CTG-only 1D CNN | 4 | 8 | 0.375 | 0.965 | 0.499 |
| CTG-only 1D CNN | 5 | 7 | 0.429 | 0.965 | 0.388 |
| CNN-BiLSTM | 1 | 8 | 0.625 | 0.860 | 0.370 |
| CNN-BiLSTM | 2 | 8 | 0.875 | 0.907 | 0.572 |
| CNN-BiLSTM | 3 | 8 | 0.625 | 0.895 | 0.329 |
| CNN-BiLSTM | 4 | 8 | 0.375 | 0.919 | 0.471 |
| CNN-BiLSTM | 5 | 7 | 0.571 | 0.895 | 0.492 |
| Multimodal CNN + Clinical MLP | 1 | 8 | 0.125 | 0.953 | 0.170 |
| Multimodal CNN + Clinical MLP | 2 | 8 | 0.625 | 0.930 | 0.470 |
| Multimodal CNN + Clinical MLP | 3 | 8 | 0.250 | 1.000 | 0.651 |
| Multimodal CNN + Clinical MLP | 4 | 8 | 0.125 | 0.977 | 0.343 |
| Multimodal CNN + Clinical MLP | 5 | 7 | 0.286 | 0.919 | 0.437 |
| Clinical-only MLP | 1 | 8 | 0.625 | 0.709 | 0.218 |
| Clinical-only MLP | 2 | 8 | 0.625 | 0.593 | 0.202 |
| Clinical-only MLP | 3 | 8 | 0.500 | 0.663 | 0.139 |
| Clinical-only MLP | 4 | 8 | 0.375 | 0.628 | 0.280 |
| Clinical-only MLP | 5 | 7 | 0.429 | 0.791 | 0.146 |

## 7. Statistical uncertainty

Pooled out-of-fold 95% confidence intervals from a stratified bootstrap (1000 resamples, seed 42). Bootstrap intervals for sensitivity are wide because only 39 cesareans exist in the development set.

| Configuration | ROC-AUC | 95% CI | PR-AUC | 95% CI | Sensitivity | 95% CI |
|---|---|---|---|---|---|---|
| B0 CTG-only 1D CNN | 0.751 | 0.646–0.848 | 0.335 | 0.217–0.498 | 0.256 | 0.128–0.385 |
| CTG-only 1D CNN — Plain BCE | 0.731 | 0.637–0.813 | 0.260 | 0.169–0.389 | 0.026 | 0.000–0.077 |
| CTG-only 1D CNN — Weighted BCE | 0.715 | 0.617–0.802 | 0.223 | 0.154–0.350 | 0.462 | 0.308–0.615 |
| CTG-only 1D CNN — Focal loss (gamma=1) | 0.727 | 0.634–0.808 | 0.226 | 0.160–0.350 | 0.487 | 0.333–0.641 |
| CTG-only 1D CNN — Focal loss (gamma=2) | 0.723 | 0.626–0.809 | 0.230 | 0.159–0.369 | 0.641 | 0.487–0.769 |
| CTG-only 1D CNN — Weighted random sampling | 0.754 | 0.649–0.843 | 0.315 | 0.211–0.459 | 0.385 | 0.231–0.538 |
| B0 CNN-BiLSTM | 0.801 | 0.709–0.882 | 0.391 | 0.264–0.550 | 0.487 | 0.333–0.641 |
| CNN-BiLSTM — Plain BCE | 0.813 | 0.727–0.881 | 0.314 | 0.222–0.467 | 0.026 | 0.000–0.077 |
| CNN-BiLSTM — Weighted BCE | 0.825 | 0.735–0.895 | 0.323 | 0.236–0.475 | 0.667 | 0.513–0.821 |
| CNN-BiLSTM — Focal loss (gamma=1) | 0.824 | 0.749–0.888 | 0.361 | 0.249–0.493 | 0.667 | 0.513–0.821 |
| CNN-BiLSTM — Focal loss (gamma=2) | 0.821 | 0.740–0.887 | 0.374 | 0.259–0.505 | 0.667 | 0.513–0.821 |
| CNN-BiLSTM — Weighted random sampling | 0.818 | 0.731–0.891 | 0.404 | 0.281–0.559 | 0.615 | 0.462–0.769 |
| B0 Multimodal CNN + Clinical MLP | 0.761 | 0.660–0.846 | 0.355 | 0.236–0.501 | 0.359 | 0.205–0.513 |
| Multimodal CNN + Clinical MLP — Plain BCE | 0.763 | 0.662–0.854 | 0.334 | 0.217–0.494 | 0.026 | 0.000–0.077 |
| Multimodal CNN + Clinical MLP — Weighted BCE | 0.752 | 0.651–0.839 | 0.340 | 0.222–0.496 | 0.359 | 0.205–0.513 |
| Multimodal CNN + Clinical MLP — Focal loss (gamma=1) | 0.740 | 0.629–0.835 | 0.342 | 0.220–0.500 | 0.513 | 0.359–0.667 |
| Multimodal CNN + Clinical MLP — Focal loss (gamma=2) | 0.733 | 0.626–0.831 | 0.341 | 0.223–0.513 | 0.538 | 0.385–0.692 |
| Multimodal CNN + Clinical MLP — Weighted random sampling | 0.749 | 0.653–0.836 | 0.294 | 0.192–0.436 | 0.282 | 0.154–0.436 |
| B0 Clinical-only MLP | 0.574 | 0.480–0.676 | 0.154 | 0.096–0.263 | 0.436 | 0.282–0.590 |
| Clinical-only MLP — Plain BCE | 0.573 | 0.486–0.668 | 0.136 | 0.087–0.227 | 0.000 | 0.000–0.000 |
| Clinical-only MLP — Weighted BCE | 0.571 | 0.477–0.676 | 0.144 | 0.095–0.262 | 0.487 | 0.333–0.641 |
| Clinical-only MLP — Focal loss (gamma=1) | 0.567 | 0.470–0.673 | 0.144 | 0.094–0.261 | 0.487 | 0.333–0.641 |
| Clinical-only MLP — Focal loss (gamma=2) | 0.565 | 0.467–0.669 | 0.141 | 0.093–0.267 | 0.487 | 0.333–0.641 |
| Clinical-only MLP — Weighted random sampling | 0.592 | 0.492–0.699 | 0.146 | 0.098–0.258 | 0.513 | 0.359–0.667 |
| Logistic regression — B0 production | 0.745 | 0.657–0.825 | 0.338 | 0.225–0.509 | 0.564 | 0.410–0.718 |
| Logistic regression — B1 no rebalancing | 0.760 | 0.672–0.838 | 0.342 | 0.230–0.511 | 0.179 | 0.077–0.308 |
| Logistic regression — B1 class weights | 0.727 | 0.629–0.813 | 0.286 | 0.193–0.448 | 0.564 | 0.410–0.718 |
| Logistic regression — B1 random oversampling | 0.730 | 0.637–0.816 | 0.284 | 0.191–0.446 | 0.564 | 0.410–0.718 |
| Logistic regression — B1 SMOTE | 0.736 | 0.644–0.821 | 0.300 | 0.193–0.463 | 0.564 | 0.410–0.718 |
| Histogram gradient boosting — B0 production | 0.708 | 0.602–0.805 | 0.296 | 0.185–0.454 | 0.385 | 0.231–0.538 |
| Histogram gradient boosting — B1 no rebalancing | 0.672 | 0.566–0.773 | 0.269 | 0.165–0.433 | 0.128 | 0.026–0.231 |
| Histogram gradient boosting — B1 class weights | 0.704 | 0.598–0.802 | 0.337 | 0.213–0.498 | 0.359 | 0.205–0.513 |
| Histogram gradient boosting — B1 random oversampling | 0.703 | 0.605–0.794 | 0.260 | 0.163–0.411 | 0.282 | 0.154–0.436 |
| Histogram gradient boosting — B1 SMOTE | 0.736 | 0.644–0.822 | 0.292 | 0.191–0.457 | 0.333 | 0.179–0.487 |

Fold-wise variability (mean ± standard deviation across the five outer folds):

| Configuration | Metric | Mean ± SD | Range |
|---|---|---|---|
| CTG-only 1D CNN — Plain BCE | PR-AUC | 0.344 ± 0.145 | 0.170–0.505 |
| CTG-only 1D CNN — Plain BCE | Cesarean sensitivity | 0.025 ± 0.056 | 0.000–0.125 |
| CTG-only 1D CNN — Weighted BCE | PR-AUC | 0.364 ± 0.184 | 0.168–0.554 |
| CTG-only 1D CNN — Weighted BCE | Cesarean sensitivity | 0.468 ± 0.231 | 0.125–0.714 |
| CTG-only 1D CNN — Focal loss (gamma=1) | PR-AUC | 0.359 ± 0.177 | 0.165–0.501 |
| CTG-only 1D CNN — Focal loss (gamma=1) | Cesarean sensitivity | 0.493 ± 0.225 | 0.125–0.714 |
| CTG-only 1D CNN — Focal loss (gamma=2) | PR-AUC | 0.379 ± 0.199 | 0.161–0.577 |
| CTG-only 1D CNN — Focal loss (gamma=2) | Cesarean sensitivity | 0.646 ± 0.219 | 0.375–0.875 |
| CTG-only 1D CNN — Weighted random sampling | PR-AUC | 0.370 ± 0.154 | 0.165–0.530 |
| CTG-only 1D CNN — Weighted random sampling | Cesarean sensitivity | 0.386 ± 0.024 | 0.375–0.429 |
| CNN-BiLSTM — Plain BCE | PR-AUC | 0.412 ± 0.161 | 0.203–0.586 |
| CNN-BiLSTM — Plain BCE | Cesarean sensitivity | 0.025 ± 0.056 | 0.000–0.125 |
| CNN-BiLSTM — Weighted BCE | PR-AUC | 0.416 ± 0.143 | 0.225–0.554 |
| CNN-BiLSTM — Weighted BCE | Cesarean sensitivity | 0.668 ± 0.164 | 0.500–0.875 |
| CNN-BiLSTM — Focal loss (gamma=1) | PR-AUC | 0.430 ± 0.111 | 0.303–0.558 |
| CNN-BiLSTM — Focal loss (gamma=1) | Cesarean sensitivity | 0.671 ± 0.185 | 0.500–0.875 |
| CNN-BiLSTM — Focal loss (gamma=2) | PR-AUC | 0.419 ± 0.103 | 0.305–0.536 |
| CNN-BiLSTM — Focal loss (gamma=2) | Cesarean sensitivity | 0.668 ± 0.060 | 0.625–0.750 |
| CNN-BiLSTM — Weighted random sampling | PR-AUC | 0.447 ± 0.098 | 0.329–0.572 |
| CNN-BiLSTM — Weighted random sampling | Cesarean sensitivity | 0.614 ± 0.178 | 0.375–0.875 |
| Multimodal CNN + Clinical MLP — Plain BCE | PR-AUC | 0.407 ± 0.149 | 0.172–0.561 |
| Multimodal CNN + Clinical MLP — Plain BCE | Cesarean sensitivity | 0.025 ± 0.056 | 0.000–0.125 |
| Multimodal CNN + Clinical MLP — Weighted BCE | PR-AUC | 0.434 ± 0.153 | 0.207–0.629 |
| Multimodal CNN + Clinical MLP — Weighted BCE | Cesarean sensitivity | 0.371 ± 0.286 | 0.125–0.857 |
| Multimodal CNN + Clinical MLP — Focal loss (gamma=1) | PR-AUC | 0.421 ± 0.164 | 0.167–0.593 |
| Multimodal CNN + Clinical MLP — Focal loss (gamma=1) | Cesarean sensitivity | 0.521 ± 0.217 | 0.375–0.857 |
| Multimodal CNN + Clinical MLP — Focal loss (gamma=2) | PR-AUC | 0.401 ± 0.152 | 0.152–0.562 |
| Multimodal CNN + Clinical MLP — Focal loss (gamma=2) | Cesarean sensitivity | 0.546 ± 0.238 | 0.250–0.857 |
| Multimodal CNN + Clinical MLP — Weighted random sampling | PR-AUC | 0.414 ± 0.176 | 0.170–0.651 |
| Multimodal CNN + Clinical MLP — Weighted random sampling | Cesarean sensitivity | 0.282 ± 0.205 | 0.125–0.625 |
| Clinical-only MLP — Plain BCE | PR-AUC | 0.165 ± 0.069 | 0.107–0.256 |
| Clinical-only MLP — Plain BCE | Cesarean sensitivity | 0.000 ± 0.000 | 0.000–0.000 |
| Clinical-only MLP — Weighted BCE | PR-AUC | 0.210 ± 0.067 | 0.155–0.324 |
| Clinical-only MLP — Weighted BCE | Cesarean sensitivity | 0.489 ± 0.169 | 0.250–0.625 |
| Clinical-only MLP — Focal loss (gamma=1) | PR-AUC | 0.208 ± 0.069 | 0.143–0.324 |
| Clinical-only MLP — Focal loss (gamma=1) | Cesarean sensitivity | 0.489 ± 0.169 | 0.250–0.625 |
| Clinical-only MLP — Focal loss (gamma=2) | PR-AUC | 0.208 ± 0.073 | 0.137–0.329 |
| Clinical-only MLP — Focal loss (gamma=2) | Cesarean sensitivity | 0.489 ± 0.169 | 0.250–0.625 |
| Clinical-only MLP — Weighted random sampling | PR-AUC | 0.197 ± 0.057 | 0.139–0.280 |
| Clinical-only MLP — Weighted random sampling | Cesarean sensitivity | 0.511 ± 0.113 | 0.375–0.625 |

Fold-level dispersion is large relative to the differences between strategies. Strategy rankings within ±0.02 PR-AUC should therefore be treated as ties.

## 8. Calibration findings

| Model | Strategy | Brier | ECE | Precision @0.5 | Sensitivity @0.5 |
|---|---|---|---|---|---|
| CTG-only 1D CNN | Plain BCE | 0.0707 | 0.0249 | 0.500 | 0.026 |
| CTG-only 1D CNN | Weighted BCE | 0.1205 | 0.1284 | 0.237 | 0.462 |
| CTG-only 1D CNN | Focal loss (gamma=1) | 0.1432 | 0.2174 | 0.198 | 0.487 |
| CTG-only 1D CNN | Focal loss (gamma=2) | 0.1785 | 0.3057 | 0.172 | 0.641 |
| CTG-only 1D CNN | Weighted random sampling | 0.0796 | 0.0645 | 0.455 | 0.385 |
| CNN-BiLSTM | Plain BCE | 0.0666 | 0.0293 | 0.200 | 0.026 |
| CNN-BiLSTM | Weighted BCE | 0.1070 | 0.1268 | 0.351 | 0.667 |
| CNN-BiLSTM | Focal loss (gamma=1) | 0.1234 | 0.2007 | 0.280 | 0.667 |
| CNN-BiLSTM | Focal loss (gamma=2) | 0.1301 | 0.2319 | 0.263 | 0.667 |
| CNN-BiLSTM | Weighted random sampling | 0.1010 | 0.0966 | 0.348 | 0.615 |
| Multimodal CNN + Clinical MLP | Plain BCE | 0.0659 | 0.0360 | 0.333 | 0.026 |
| Multimodal CNN + Clinical MLP | Weighted BCE | 0.0903 | 0.0786 | 0.292 | 0.359 |
| Multimodal CNN + Clinical MLP | Focal loss (gamma=1) | 0.1160 | 0.1831 | 0.299 | 0.513 |
| Multimodal CNN + Clinical MLP | Focal loss (gamma=2) | 0.1445 | 0.2652 | 0.284 | 0.538 |
| Multimodal CNN + Clinical MLP | Weighted random sampling | 0.0861 | 0.0754 | 0.367 | 0.282 |
| Clinical-only MLP | Plain BCE | 0.0862 | 0.0748 | 0.000 | 0.000 |
| Clinical-only MLP | Weighted BCE | 0.2344 | 0.3797 | 0.095 | 0.487 |
| Clinical-only MLP | Focal loss (gamma=1) | 0.2313 | 0.3866 | 0.101 | 0.487 |
| Clinical-only MLP | Focal loss (gamma=2) | 0.2312 | 0.3899 | 0.103 | 0.487 |
| Clinical-only MLP | Weighted random sampling | 0.2103 | 0.3206 | 0.126 | 0.513 |

Classical calibration for reference:

| Model | Strategy | Brier | ECE |
|---|---|---|---|
| Logistic regression | No rebalancing | 0.0654 | 0.0134 |
| Logistic regression | Class weights | 0.1584 | 0.2211 |
| Logistic regression | Random oversampling | 0.1566 | 0.2072 |
| Logistic regression | SMOTE (tabular) | 0.1517 | 0.1952 |
| Histogram gradient boosting | No rebalancing | 0.0711 | 0.0531 |
| Histogram gradient boosting | Class weights | 0.0794 | 0.0658 |
| Histogram gradient boosting | Random oversampling | 0.0792 | 0.0678 |
| Histogram gradient boosting | SMOTE (tabular) | 0.0876 | 0.0795 |

Unweighted and focal losses usually keep better-calibrated probabilities than reweighted losses or resampling, which deliberately shift the predicted probability scale towards the minority class. Brier/ECE therefore often worsen exactly when sensitivity improves. Because the operating threshold is fixed at 0.5, this miscalibration directly reduces specificity and can also suppress sensitivity even when ranking (ROC-AUC/PR-AUC) improves. Calibration curves are saved as `../comparisons/figures/calibration_<model>.png`.

## 9. Limitations

- **Small minority class.** Only 39 development-set cesareans (7–8 per outer fold) support
  all conclusions. Confidence intervals for sensitivity span tens of percentage points and
  a single fold can change the pooled estimate materially.
- **Single seed.** Deep runs use seed 42 with deterministic settings, but no multi-seed
  variance study was run. Deep training variance may be comparable to the strategy effects.
- **Inner selection transferred across strategies.** Learning rate and epoch budget were
  selected once per outer fold with weighted BCE and reused for every strategy. This keeps
  the comparison fair but may disadvantage strategies that would prefer a different budget.
- **Fixed 0.5 threshold.** Primary sensitivity/specificity use threshold 0.5. Inner-selected
  thresholds are reported secondarily, but no outer-fold threshold calibration was performed,
  so clinical operating points are not yet optimised.
- **No multiplicity control.** Paired bootstrap p-values are unadjusted; with many model ×
  strategy comparisons, some "significant" differences are expected by chance.
- **CPU-only training.** Epoch budgets were limited by runtime, not by convergence analysis.
  Deep runs are bitwise reproducible only when `PHASE_B_THREADS` is fixed, because CPU
  floating-point reduction order depends on the thread count; all Phase B artifacts were
  generated with `PHASE_B_THREADS=3`. Repeating the multimodal `none` augmentation control
  at a different thread count changed pooled PR-AUC by about 0.016, which is a useful
  reminder that single-run deep differences of that size are not meaningful.
- **Tabular SMOTE is a local implementation.** `imbalanced-learn` is not installed, so the
  SMOTE variant uses a tested but minimal implementation; results may differ slightly from
  the reference library.
- **No held-out test evaluation.** By design, the 83-record test split is never loaded in
  Phase B. All reported numbers are development-set out-of-fold estimates and are expected
  to be optimistic relative to external performance.
- **No claim of clinical utility.** These are research-prototype results on a single-centre
  retrospective database, not a validated clinical decision tool.

## 10. Recommended imbalance strategy

**Deep models: Weighted random sampling (`weighted_sampling`).** Selection rule: Highest mean PR-AUC with mean ROC-AUC within 0.01 of the best.

- Mean PR-AUC: **0.290** (best PR-AUC strategy: 0.290).
- Mean ROC-AUC: **0.728** (best ROC-AUC strategy: 0.728).
- Mean cesarean sensitivity: **0.449**; mean specificity: **0.872**; mean Brier: **0.119**.

**Classical models.** Tabular resampling should remain a secondary tool. The production `class_weight='balanced'` configuration is the most defensible default because it needs no synthetic data, and SMOTE results are reported only as an exploratory tabular comparison.

Per-architecture best strategy by PR-AUC:

| Model | Best PR-AUC strategy | PR-AUC | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| CTG-only 1D CNN | Weighted random sampling | 0.315 | 0.754 | 0.385 | 0.958 |
| CNN-BiLSTM | Weighted random sampling | 0.404 | 0.818 | 0.615 | 0.895 |
| Multimodal CNN + Clinical MLP | Focal loss (gamma=1) | 0.342 | 0.740 | 0.513 | 0.891 |
| Clinical-only MLP | Weighted random sampling | 0.146 | 0.592 | 0.513 | 0.677 |

Architecture-specific winners differ, so the pooled recommendation should be applied as a default, not as a claim that one loss dominates for every model.

**Operational recommendation.** Keep PR-AUC and calibration as the primary decision criteria, select the operating threshold on inner folds or a dedicated calibration split, and treat any strategy whose PR-AUC gain is below ~0.02 as equivalent to the baseline. Do not deploy a reweighted model solely because its sensitivity at 0.5 increased; check the accompanying specificity and Brier change.

## Appendix A. Controlled CTG augmentation (exploratory)

Augmentation is deliberately kept outside the B1 strategy comparison and applied only to the multimodal architecture. Noise is added to the two physiological channels only; masks are untouched. The `mix` mode combines Gaussian noise, a ±5% amplitude scale and a ±30-second time shift.

| Mode | ROC-AUC | PR-AUC | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|
| none | 0.761 | 0.355 | 0.359 | 0.940 | 0.091 |
| gaussian | 0.761 | 0.336 | 0.333 | 0.940 | 0.094 |
| mix | 0.756 | 0.358 | 0.487 | 0.905 | 0.103 |

As a reproducibility check, the `none` augmentation mode uses the same configuration and seeds as B0 multimodal: the pooled metrics are **identical**, confirming that the staged runner is deterministic for an unchanged configuration.

## Appendix B. Inner-selection and training-fold class weights

Selected learning rate and epoch budget per outer fold, taken from the inner 3-fold selection that only sees the outer-training records. Class weights are computed from the same outer-training labels and stored in each fold artifact.

| Model | Outer fold | Selected LR | Epochs | Inner AUROC | Inner threshold | Cesarean weight | pos_weight |
|---|---|---|---|---|---|---|---|
| CTG-only 1D CNN | 1 | 0.0005 | 12 | 0.767 | 0.142 | 6.048 | 11.097 |
| CTG-only 1D CNN | 2 | 0.0010 | 11 | 0.766 | 0.165 | 6.048 | 11.097 |
| CTG-only 1D CNN | 3 | 0.0005 | 11 | 0.743 | 0.274 | 6.048 | 11.097 |
| CTG-only 1D CNN | 4 | 0.0010 | 10 | 0.713 | 0.063 | 6.048 | 11.097 |
| CTG-only 1D CNN | 5 | 0.0005 | 14 | 0.654 | 0.106 | 5.875 | 10.750 |
| CNN-BiLSTM | 1 | 0.0005 | 13 | 0.887 | 0.206 | 6.048 | 11.097 |
| CNN-BiLSTM | 2 | 0.0005 | 15 | 0.767 | 0.378 | 6.048 | 11.097 |
| CNN-BiLSTM | 3 | 0.0005 | 13 | 0.817 | 0.250 | 6.048 | 11.097 |
| CNN-BiLSTM | 4 | 0.0010 | 10 | 0.828 | 0.264 | 6.048 | 11.097 |
| CNN-BiLSTM | 5 | 0.0005 | 14 | 0.847 | 0.503 | 5.875 | 10.750 |
| Multimodal CNN + Clinical MLP | 1 | 0.0005 | 12 | 0.763 | 0.102 | 6.048 | 11.097 |
| Multimodal CNN + Clinical MLP | 2 | 0.0010 | 10 | 0.784 | 0.106 | 6.048 | 11.097 |
| Multimodal CNN + Clinical MLP | 3 | 0.0010 | 10 | 0.753 | 0.081 | 6.048 | 11.097 |
| Multimodal CNN + Clinical MLP | 4 | 0.0010 | 11 | 0.727 | 0.179 | 6.048 | 11.097 |
| Multimodal CNN + Clinical MLP | 5 | 0.0010 | 12 | 0.703 | 0.054 | 5.875 | 10.750 |
| Clinical-only MLP | 1 | 0.0010 | 6 | 0.388 | 0.647 | 6.048 | 11.097 |
| Clinical-only MLP | 2 | 0.0010 | 15 | 0.660 | 0.445 | 6.048 | 11.097 |
| Clinical-only MLP | 3 | 0.0010 | 10 | 0.602 | 0.496 | 6.048 | 11.097 |
| Clinical-only MLP | 4 | 0.0010 | 14 | 0.587 | 0.495 | 6.048 | 11.097 |
| Clinical-only MLP | 5 | 0.0010 | 10 | 0.598 | 0.439 | 5.875 | 10.750 |

## Generated artifacts

| Path | Contents |
|---|---|
| `experiments/phase_b/config.json` | Phase B configuration and environment |
| `experiments/phase_b/folds.csv` | Outer fold assignments (Phase A folds) |
| `experiments/phase_b/dataset_manifest.json` | Dataset counts, hashes, fold composition |
| `experiments/phase_b/b0/<model>/result.json` | B0 baseline metrics, predictions, folds |
| `experiments/phase_b/b1/deep/<model>/<strategy>/` | Deep checkpoints, predictions, histories, metrics |
| `experiments/phase_b/b1/classical/<model>/<strategy>/` | Classical predictions, selections, metrics |
| `experiments/phase_b/b1/augmentation/<mode>/` | Exploratory multimodal augmentation runs |
| `experiments/phase_b/comparisons/` | Comparison tables, paired differences, recommendation |
| `experiments/phase_b/comparisons/figures/` | ROC, PR, calibration, confusion and fold figures |
| `experiments/phase_b/logs/` | Stage logs |

Figures were written only for configurations that completed; missing figure files indicate that the corresponding stage had not run when the report was generated.

## Reproducibility

```bash
python -m src.phase_b.run --stage prepare
python -m src.phase_b.run --stage b0 --force
python -m src.phase_b.run --stage classical --force
python -m src.phase_b.run --stage deep --force
python -m src.phase_b.run --stage augment --force
python -m src.phase_b.run --stage compare --force
python -m src.phase_b.run --stage report --force
python -m pytest -q
```

All runs use `PYTHONHASHSEED`-independent deterministic seeding through `set_seed`, and `PHASE_B_THREADS` can cap CPU threads per process. Deep checkpoints include the fold, loss strategy, class weights and preprocessing metadata needed to reproduce predictions.
