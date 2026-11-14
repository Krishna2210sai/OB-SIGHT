# Phase A — Data Handling and Preprocessing Optimisation

**Project:** AI-Based Clinical Decision Support System for Predicting Delivery Outcomes Using CTG and Maternal Clinical Data  
**Repository:** https://github.com/Krishna2210sai/OB-SIGHT  
**Scope:** data quality, window duration, sampling frequency and feature selection for the existing delivery-mode models.  
**Random seed:** 42  
**Held-out test set:** not used or evaluated anywhere in Phase A.

## Executive summary

- All assumptions about the CTU-UHB database were verified: 552 records, two channels (FHR and UC), native 4 Hz, 552 usable 30-minute windows and 46 cesarean deliveries.
- The existing production pipeline uses the last **30 minutes** at **1 Hz** and 25 hand-crafted CTG descriptors plus 15 clinical features.
- Exact zeros are missing values, not physiology: the smallest positive UC value is 0.5 and 0 positive UC samples fall below 0.1.
- Recording length is strongly confounded with outcome. Cesarean recordings are much shorter, so demanding a 60-minute window removes **41 of 552 records**, disproportionately cesarean cases. This is a selection-bias risk, not merely a preprocessing choice.
- Window and sampling frequency do not change the signal dramatically around the production setting; the dominant effects are missingness and class/fold composition. Feature selection changes the feature count much more than it changes out-of-fold ROC-AUC.
- In the focused deep screen the highest fold-mean AUROC was 0.790 at w30_fs0p5:multimodal (SD 0.136). The 1 Hz production result was 0.758; the intervals overlap, so this is not evidence that a lower rate is superior.
- No record is deleted from the main experiments. Quality labels are descriptive and every window exclusion is recorded explicitly.

## 1. Existing preprocessing pipeline (preserved)

Phase A is additive. The production preprocessing, model architectures, API and frontend are unchanged; all Phase A code lives in `src/phase_a/` and writes to `experiments/phase_a/`.

| Component | Existing behaviour |
|:---|:---|
| Raw signals | WFDB `.hea/.dat`, channels FHR and UC, 4 Hz, gain 100 |
| Zero handling | samples `<= 0` treated as missing |
| Window | last 30 minutes of the trimmed recording |
| Resampling | deterministic decimation to 1 Hz with short-gap interpolation |
| Tensors | `fhr`, `uc`, `fhr_mask`, `uc_mask`, shape `(4, 1800)` |
| Clinical | 4 numeric + 7 binary + presentation one-hot (15 features) |
| CTG descriptors | 25 hand-crafted features (baseline, variability, accelerations, decelerations, UC) |
| Models | multimodal CNN+MLP, CTG-only 1D CNN, CNN-BiLSTM, clinical-only MLP |
| Baselines | logistic regression and histogram gradient boosting |
| Split | fixed record-level train/validation/test; only train+validation used here |

Phase A differs only where the experiment requires it: anti-aliased resampling, explicit inclusion policies for short recordings, and fold-specific feature selection.

## 2. Verification of dataset assumptions

| Assumption | Expected | Observed | Status |
|:---|:---|:---|:---|
| Number of records | 552 | 552 | verified |
| Signal channels | FHR + UC | 2 interleaved channels | verified |
| Native sampling frequency | 4 Hz | header states 4 Hz; 14 400 samples/hour | verified |
| Current downsampling | 1 Hz | production pipeline | verified |
| Current window | last 30 min | production pipeline | verified |
| Delivery-mode labels | 46 cesareans / 506 vaginal | {'0': 506, '1': 46} | verified |
| Missing target labels | 0 | 0 | verified |
| Clinical feature encoding | 15 features | 4 numeric + 7 binary + presentation | verified |

The development set used for every Phase A comparison contains **469 records** (`split` in train/validation); the 83 held-out test records are excluded from all code paths.

## 3. Dataset quality findings

### 3.1 Recording duration

| Trimmed duration | Value |
|:---|---:|
| Mean | 70.2 min |
| Median | 69.9 min |
| Minimum | 41.2 min |
| Maximum | 90.1 min |

The shortest trimmed recording still exceeds 30 minutes, so the production window never requires padding. Longer windows do:

| Window | Records with sufficient duration |
|:---|---:|
| 30m | 552 |
| 45m | 548 |
| 60m | 511 |


Duration is not missing at random with respect to outcome. Cesarean recordings are systematically shorter (many are truncated at 60 minutes). Any window that removes short recordings therefore removes cesarean cases preferentially. Phase A quantifies this instead of silently accepting it.

### 3.2 Missing-value analysis

#### Target and clinical variables

| Clinical variable | Missing fraction |
|:---|---:|
| gravidity | 0.72% |
| presentation | 0.54% |
| age | 0.00% |
| parity | 0.00% |
| gest_weeks | 0.00% |
| diabetes | 0.00% |
| hypertension | 0.00% |
| preeclampsia | 0.00% |
| liquor_praecox | 0.00% |
| pyrexia | 0.00% |
| meconium | 0.00% |
| induced | 0.00% |

Target labels: 0 missing; class counts {'0': 506, '1': 46}; prevalence 8.33%.

#### FHR and UC

| Metric | Mean | Median | p90 | Max |
|:---|---:|---:|---:|---:|
| fhr_missing_fraction_30m | 18.39% | 14.68% | 41.09% | 92.36% |
| uc_missing_fraction_30m | 20.95% | 11.88% | 52.77% | 100.00% |
| fhr_zero_fraction_recording | 14.24% | 12.74% | 28.60% | 49.72% |
| uc_zero_fraction_recording | 15.14% | 11.80% | 33.52% | 100.00% |
| longest_fhr_gap_min_30m | 1.79 min | 0.70 min | 5.18 min | 22.93 min |
| longest_uc_gap_min_30m | 3.85 min | 1.32 min | 11.42 min | 30.00 min |


| Mean by outcome | Vaginal | Cesarean |
|:---|---:|---:|
| fhr_missing_fraction_30m | 0.1921 | 0.0929 |
| uc_missing_fraction_30m | 0.2151 | 0.1478 |
| implausible_fhr_fraction_30m | 0.0000 | 0.0000 |
| jump_fraction_30m | 0.0004 | 0.0005 |
| flatline_fraction_30m | 0.0022 | 0.0092 |


In the 30-minute window, 0 records have an entirely missing FHR channel and 2 have an entirely missing UC channel. More than half of the window is missing for 25 records in FHR and 66 records in UC. These records are retained in the main experiments and flagged rather than deleted.

#### Why exact zeros are treated as missing

Three pieces of evidence support the missing-value interpretation; none of them assumes that every zero is missing by definition:

1. The smallest positive UC value in the entire database is **0.5**; 0 positive samples are below 0.1. A genuine uterine quiescence reading of 0.0 would produce a continuous distribution near zero, which is not observed.
2. FHR = 0 bpm is not physiological during labour.
3. Zero runs are long and contiguous (median longest UC run is several minutes, maximum 75 minutes) and partly coincide across channels, which is characteristic of signal loss.
The audit nevertheless retains the raw zero fractions and the count of positive values below 0.1 so the assumption can be re-checked against the source data.

## 4. CTG signal quality

| Problem | Definition | Reported metric |
|:---|:---|:---|
| Missing segments | `value <= 0`; contiguous runs measured in minutes | fhr/uc missing fraction, longest gap, number of gaps >10 s and >60 s |
| Zero-value segments | longest zero run per channel; explicit zero-encoding evidence | longest zero run |
| Implausible FHR | outside 30–240 bpm | implausible fraction and count |
| Discontinuities | sample-to-sample change >25 bpm | jump count, rate per minute, maximum jump |
| Flatlines | 5-minute rolling SD < 2 bpm, assessable windows only | flatline fraction and longest run |
| Noise/artefacts | captured indirectly by jump and flatline statistics | documented as a limitation |
| Insufficient duration | trimmed duration below the requested window | support flags for 30/45/60 min |


Each of the 552 records receives a row in `data_quality/record_quality.csv` with all metrics, the assigned category and the human-readable reason list.

### 4.1 Quality categories

Thresholds are configurable in `src/phase_a/config.py::QualityThresholds`. ``good`` means all metrics are inside the strict limits, ``acceptable`` means at least one metric is outside the strict limit but none exceeds the permissive limit, and ``poor`` means at least one permissive limit is exceeded or the record cannot provide the window.

| Category | Records |
|:---|---:|
| good | 89 |
| acceptable | 267 |
| poor | 196 |


Most frequent reasons among records not labelled ``good``:

| Reason | Records |
|:---|---:|
| UC missing fraction | 132 |
| FHR missing fraction | 111 |
| longest UC gap (min) | 71 |
| longest FHR gap (min) | 24 |
| FHR jump fraction | 2 |
| FHR flatline fraction | 2 |


### 4.2 Selection-bias sensitivity analysis

No record is removed from the main experiments. The table below shows what would happen if a stricter inclusion policy were applied. Fisher's exact test compares class counts inside and outside each subset; SMD is the absolute standardised mean difference.

| Subset | n | Cesarean | Prevalence | Excluded | Excl. cesarean | Fisher p | SMD duration | SMD FHR missing | CV AUROC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 469 | 39 | 8.32% | 0 | 0 | n/a | n/a | n/a | 0.745 |
| good_or_acceptable | 301 | 28 | 9.30% | 168 | 11 | 0.384 | 0.429 | 1.146 | 0.718 |
| good_only | 78 | 10 | 12.82% | 391 | 29 | 0.119 | 0.265 | 1.362 | 0.622 |

Interpretation: a non-significant Fisher test does not prove absence of bias when the excluded group is small. The SMDs and the raw excluded counts are therefore reported alongside the p-value, and the main analyses keep all eligible records.

## 5. Window duration experiments

For every recording the most recent N minutes of valid signal are extracted, with FHR and UC kept on the same sample grid. Records shorter than the window are excluded and listed with the reason; they are never padded silently. The target label is unchanged.
Only intrapartum signal samples enter the window. No outcome-derived variable (pH, base deficit, Apgar, NICU stay) and no post-delivery information is used, and the delivery-mode label is never part of the input.

| Window | Signal used | Records | Inclusion note |
|:---|:---|:---|:---|
| 30 min | last 30 min | 552 | production baseline; no exclusions |
| 45 min | last 45 min | 548 | 4 records excluded (3 cesareans, 6.5% of all cesareans) |
| 60 min | last 60 min | 511 | 41 records excluded (10 cesareans, 21.7% of all cesareans) |

| Configuration | Model | Selection | n | Cesarean | AUROC (mean ± SD) | AUROC 95% CI | AUPRC | Accuracy | Bal. acc. | Sensitivity | Specificity | F1 | MCC |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w30_fs1:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.729 ± 0.138 | [0.610, 0.801] | 0.394 | 0.902 | 0.673 | 0.396 | 0.949 | 0.407 | 0.361 |
| w30_fs1:logistic:none | logistic | none | 469 | 39 | 0.753 ± 0.100 | [0.648, 0.830] | 0.385 | 0.778 | 0.685 | 0.571 | 0.798 | 0.297 | 0.239 |
| w45_fs1:gradient_boosting:none | gradient_boosting | none | 466 | 37 | 0.701 ± 0.117 | [0.605, 0.780] | 0.270 | 0.904 | 0.593 | 0.224 | 0.963 | 0.277 | 0.240 |
| w45_fs1:logistic:none | logistic | none | 466 | 37 | 0.739 ± 0.107 | [0.635, 0.820] | 0.288 | 0.801 | 0.695 | 0.569 | 0.821 | 0.311 | 0.259 |
| w60_fs1:gradient_boosting:none | gradient_boosting | none | 438 | 31 | 0.570 ± 0.094 | [0.431, 0.660] | 0.150 | 0.902 | 0.522 | 0.079 | 0.966 | 0.084 | 0.050 |
| w60_fs1:logistic:none | logistic | none | 438 | 31 | 0.625 ± 0.132 | [0.518, 0.737] | 0.202 | 0.753 | 0.588 | 0.396 | 0.779 | 0.176 | 0.104 |

The 30-minute window supports every record and therefore has the cleanest class balance. Longer windows add information but change the population being evaluated. Comparing their AUROC without this caveat would overstate the benefit of a longer window.

## 6. Sampling frequency experiments

All rates are obtained from the native 4 Hz signal. Before decimation, a 4th-order zero-phase Butterworth low-pass filter is applied to the samples inside the requested window only: cutoff `0.4 × fs_out` for FHR and `0.2 × fs_out` for UC. Filtering after cropping prevents information after the window end from entering the input. Long gaps remain missing in the mask; interpolation is used only for the numerical filtering step and is never treated as observed data.

| Rate | Decimation | Anti-alias cutoff | 30-min samples |
|:---|:---|:---|---:|
| 4 Hz | native | none | 7200 |
| 2 Hz | ÷2 | 0.8 Hz FHR / 0.4 Hz UC | 3600 |
| 1 Hz | ÷4 | 0.4 Hz FHR / 0.2 Hz UC | 1800 |
| 0.5 Hz | ÷8 | 0.2 Hz FHR / 0.1 Hz UC | 900 |

| Configuration | Model | Selection | n | Cesarean | AUROC (mean ± SD) | AUROC 95% CI | AUPRC | Accuracy | Bal. acc. | Sensitivity | Specificity | F1 | MCC |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w30_fs4:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.755 ± 0.104 | [0.655, 0.826] | 0.417 | 0.902 | 0.648 | 0.343 | 0.953 | 0.388 | 0.351 |
| w30_fs4:logistic:none | logistic | none | 469 | 39 | 0.748 ± 0.107 | [0.637, 0.825] | 0.408 | 0.774 | 0.682 | 0.571 | 0.793 | 0.294 | 0.235 |
| w30_fs2:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.740 ± 0.112 | [0.634, 0.809] | 0.400 | 0.902 | 0.661 | 0.371 | 0.951 | 0.382 | 0.338 |
| w30_fs2:logistic:none | logistic | none | 469 | 39 | 0.754 ± 0.095 | [0.652, 0.832] | 0.397 | 0.789 | 0.700 | 0.593 | 0.807 | 0.321 | 0.267 |
| w30_fs1:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.722 ± 0.139 | [0.602, 0.797] | 0.393 | 0.902 | 0.673 | 0.396 | 0.949 | 0.407 | 0.361 |
| w30_fs1:logistic:none | logistic | none | 469 | 39 | 0.753 ± 0.100 | [0.648, 0.830] | 0.385 | 0.778 | 0.685 | 0.571 | 0.798 | 0.297 | 0.239 |
| w30_fs0p5:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.715 ± 0.160 | [0.605, 0.789] | 0.337 | 0.889 | 0.594 | 0.239 | 0.949 | 0.263 | 0.210 |
| w30_fs0p5:logistic:none | logistic | none | 469 | 39 | 0.753 ± 0.105 | [0.648, 0.827] | 0.407 | 0.781 | 0.699 | 0.600 | 0.798 | 0.315 | 0.260 |

Higher sampling frequency is not assumed to be better. At 4 Hz the convolutional models see twice as many samples per minute with the same receptive field in seconds only if the architecture is adjusted; the screening therefore treats rate as an empirical question, not a monotone improvement.

## 7. Feature extraction and feature selection

The candidate set is the production feature set: 15 clinical features (4 numeric, 7 binary, presentation one-hot with a missing indicator) and 25 hand-crafted CTG descriptors. Selection is always fitted inside each training fold using a scikit-learn `Pipeline`; the validation fold is transformed with the selector fitted on the other four folds. The test split is never used.

| Method | Rule | Family |
|:---|:---|:---|
| No selection | all 40 features | reference |
| Correlation filter | remove one feature from each pair with |r| ≥ 0.90 | redundancy |
| Mutual information | top 15 by MI + protected features | non-linear relevance |
| RFE | logistic regression, 15 features + protected | wrapper |
| L1 | L1 logistic regression (C = 0.10) + protected | embedded |
| Random forest | top 15 by impurity importance + protected | non-linear embedded |

Protected clinical features (`age`, `gest_weeks`, `parity`, `meconium`, `presentation_2`, `presentation_3`) are forced into every subset. They are clinically important even when their univariate association is weak, and removing them could change the clinical meaning of the model rather than only its statistical fit.

### 7.1 Redundancy in the candidate set (w30_fs1)

| Feature A | Feature B | |r| |
|:---|:---|---:|
| fhr_mean | fhr_median | 0.957 |
| fhr_std | fhr_range | 0.953 |
| uc_std | uc_p90 | 0.942 |
| pct_fhr_below_110 | pct_fhr_below_100 | 0.936 |
| acceleration_count | acceleration_fraction | 0.917 |
| deceleration_count | deceleration_fraction | 0.908 |
| uc_mean | uc_p90 | 0.906 |


Features with the weakest univariate association with the outcome are listed for inspection only. A low univariate AUC is not by itself a reason to remove a clinically meaningful variable.

| Feature | Univariate AUC | SD | Zero fraction |
|:---|---:|---:|---:|
| pyrexia | 0.498 | 0.065 | 0.996 |
| presentation_missing | 0.498 | 0.065 | 0.996 |
| diabetes | 0.502 | 0.259 | 0.928 |
| preeclampsia | 0.495 | 0.182 | 0.966 |
| presentation_2 | 0.495 | 0.182 | 0.966 |
| uc_std | 0.508 | 9.843 | 0.034 |
| uc_mean | 0.489 | 13.571 | 0.004 |
| age | 0.513 | 4.470 | 0.000 |
| gravidity | 0.482 | 1.046 | 0.000 |
| hypertension | 0.481 | 0.279 | 0.915 |


### 7.2 Performance by selection method

| Configuration | Model | Selection | n | Cesarean | AUROC (mean ± SD) | AUROC 95% CI | AUPRC | Accuracy | Bal. acc. | Sensitivity | Specificity | F1 | MCC |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w30_fs1:gradient_boosting:correlation | gradient_boosting | correlation | 469 | 39 | 0.743 ± 0.120 | [0.641, 0.813] | 0.409 | 0.898 | 0.644 | 0.339 | 0.949 | 0.364 | 0.317 |
| w30_fs1:gradient_boosting:l1 | gradient_boosting | l1 | 469 | 39 | 0.739 ± 0.125 | [0.618, 0.813] | 0.407 | 0.883 | 0.708 | 0.496 | 0.919 | 0.401 | 0.351 |
| w30_fs1:gradient_boosting:mutual_info | gradient_boosting | mutual_info | 469 | 39 | 0.689 ± 0.126 | [0.568, 0.765] | 0.305 | 0.872 | 0.583 | 0.236 | 0.930 | 0.240 | 0.172 |
| w30_fs1:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.722 ± 0.139 | [0.602, 0.797] | 0.393 | 0.902 | 0.673 | 0.396 | 0.949 | 0.407 | 0.361 |
| w30_fs1:gradient_boosting:rf_importance | gradient_boosting | rf_importance | 469 | 39 | 0.665 ± 0.146 | [0.550, 0.753] | 0.324 | 0.876 | 0.634 | 0.343 | 0.926 | 0.312 | 0.248 |
| w30_fs1:gradient_boosting:rfe | gradient_boosting | rfe | 469 | 39 | 0.708 ± 0.109 | [0.588, 0.792] | 0.309 | 0.885 | 0.639 | 0.343 | 0.935 | 0.311 | 0.257 |
| w30_fs1:logistic:correlation | logistic | correlation | 469 | 39 | 0.742 ± 0.103 | [0.630, 0.816] | 0.380 | 0.768 | 0.679 | 0.571 | 0.786 | 0.288 | 0.229 |
| w30_fs1:logistic:l1 | logistic | l1 | 469 | 39 | 0.732 ± 0.121 | [0.612, 0.804] | 0.338 | 0.770 | 0.704 | 0.625 | 0.784 | 0.308 | 0.259 |
| w30_fs1:logistic:mutual_info | logistic | mutual_info | 469 | 39 | 0.701 ± 0.093 | [0.586, 0.782] | 0.335 | 0.732 | 0.648 | 0.546 | 0.749 | 0.260 | 0.189 |
| w30_fs1:logistic:none | logistic | none | 469 | 39 | 0.753 ± 0.100 | [0.648, 0.830] | 0.385 | 0.778 | 0.685 | 0.571 | 0.798 | 0.297 | 0.239 |
| w30_fs1:logistic:rf_importance | logistic | rf_importance | 469 | 39 | 0.738 ± 0.118 | [0.608, 0.810] | 0.381 | 0.761 | 0.711 | 0.650 | 0.772 | 0.316 | 0.268 |
| w30_fs1:logistic:rfe | logistic | rfe | 469 | 39 | 0.760 ± 0.102 | [0.648, 0.827] | 0.360 | 0.770 | 0.704 | 0.625 | 0.784 | 0.308 | 0.259 |
| w45_fs1:gradient_boosting:correlation | gradient_boosting | correlation | 466 | 37 | 0.666 ± 0.131 | [0.550, 0.752] | 0.242 | 0.887 | 0.571 | 0.195 | 0.946 | 0.230 | 0.178 |
| w45_fs1:gradient_boosting:l1 | gradient_boosting | l1 | 466 | 37 | 0.691 ± 0.112 | [0.583, 0.772] | 0.260 | 0.874 | 0.600 | 0.274 | 0.925 | 0.254 | 0.187 |
| w45_fs1:gradient_boosting:mutual_info | gradient_boosting | mutual_info | 466 | 37 | 0.639 ± 0.058 | [0.534, 0.735] | 0.220 | 0.869 | 0.575 | 0.224 | 0.925 | 0.221 | 0.153 |
| w45_fs1:gradient_boosting:none | gradient_boosting | none | 466 | 37 | 0.701 ± 0.117 | [0.605, 0.780] | 0.270 | 0.904 | 0.593 | 0.224 | 0.963 | 0.277 | 0.240 |
| w45_fs1:gradient_boosting:rf_importance | gradient_boosting | rf_importance | 466 | 37 | 0.700 ± 0.113 | [0.604, 0.782] | 0.223 | 0.882 | 0.526 | 0.104 | 0.949 | 0.123 | 0.068 |
| w45_fs1:gradient_boosting:rfe | gradient_boosting | rfe | 466 | 37 | 0.719 ± 0.071 | [0.630, 0.794] | 0.245 | 0.880 | 0.565 | 0.190 | 0.939 | 0.214 | 0.181 |
| w45_fs1:logistic:correlation | logistic | correlation | 466 | 37 | 0.718 ± 0.109 | [0.606, 0.797] | 0.266 | 0.762 | 0.667 | 0.552 | 0.781 | 0.266 | 0.207 |
| w45_fs1:logistic:l1 | logistic | l1 | 466 | 37 | 0.710 ± 0.112 | [0.570, 0.783] | 0.263 | 0.736 | 0.693 | 0.639 | 0.746 | 0.272 | 0.227 |
| w45_fs1:logistic:mutual_info | logistic | mutual_info | 466 | 37 | 0.715 ± 0.119 | [0.582, 0.788] | 0.261 | 0.749 | 0.669 | 0.574 | 0.765 | 0.261 | 0.205 |
| w45_fs1:logistic:none | logistic | none | 466 | 37 | 0.739 ± 0.107 | [0.635, 0.820] | 0.288 | 0.801 | 0.695 | 0.569 | 0.821 | 0.311 | 0.259 |
| w45_fs1:logistic:rf_importance | logistic | rf_importance | 466 | 37 | 0.764 ± 0.146 | [0.649, 0.833] | 0.352 | 0.760 | 0.715 | 0.661 | 0.769 | 0.306 | 0.265 |
| w45_fs1:logistic:rfe | logistic | rfe | 466 | 37 | 0.719 ± 0.104 | [0.604, 0.803] | 0.278 | 0.773 | 0.695 | 0.602 | 0.788 | 0.299 | 0.248 |
| w60_fs1:gradient_boosting:correlation | gradient_boosting | correlation | 438 | 31 | 0.599 ± 0.100 | [0.470, 0.679] | 0.127 | 0.886 | 0.514 | 0.082 | 0.946 | 0.095 | 0.038 |
| w60_fs1:gradient_boosting:l1 | gradient_boosting | l1 | 438 | 31 | 0.614 ± 0.136 | [0.472, 0.709] | 0.170 | 0.872 | 0.505 | 0.079 | 0.931 | 0.094 | 0.029 |
| w60_fs1:gradient_boosting:mutual_info | gradient_boosting | mutual_info | 438 | 31 | 0.508 ± 0.083 | [0.387, 0.590] | 0.097 | 0.873 | 0.494 | 0.054 | 0.934 | 0.065 | 0.000 |
| w60_fs1:gradient_boosting:none | gradient_boosting | none | 438 | 31 | 0.570 ± 0.094 | [0.431, 0.660] | 0.150 | 0.902 | 0.522 | 0.079 | 0.966 | 0.084 | 0.050 |
| w60_fs1:gradient_boosting:rf_importance | gradient_boosting | rf_importance | 438 | 31 | 0.589 ± 0.075 | [0.458, 0.692] | 0.159 | 0.883 | 0.499 | 0.054 | 0.945 | 0.075 | 0.028 |
| w60_fs1:gradient_boosting:rfe | gradient_boosting | rfe | 438 | 31 | 0.536 ± 0.136 | [0.399, 0.642] | 0.139 | 0.886 | 0.501 | 0.054 | 0.949 | 0.069 | 0.022 |
| w60_fs1:logistic:correlation | logistic | correlation | 438 | 31 | 0.618 ± 0.138 | [0.508, 0.721] | 0.176 | 0.733 | 0.552 | 0.343 | 0.762 | 0.149 | 0.062 |
| w60_fs1:logistic:l1 | logistic | l1 | 438 | 31 | 0.658 ± 0.117 | [0.511, 0.746] | 0.215 | 0.717 | 0.638 | 0.543 | 0.732 | 0.200 | 0.146 |
| w60_fs1:logistic:mutual_info | logistic | mutual_info | 438 | 31 | 0.598 ± 0.118 | [0.449, 0.694] | 0.179 | 0.715 | 0.613 | 0.493 | 0.732 | 0.187 | 0.124 |
| w60_fs1:logistic:none | logistic | none | 438 | 31 | 0.625 ± 0.132 | [0.518, 0.737] | 0.202 | 0.753 | 0.588 | 0.396 | 0.779 | 0.176 | 0.104 |
| w60_fs1:logistic:rf_importance | logistic | rf_importance | 438 | 31 | 0.661 ± 0.181 | [0.521, 0.759] | 0.212 | 0.729 | 0.651 | 0.561 | 0.742 | 0.211 | 0.164 |
| w60_fs1:logistic:rfe | logistic | rfe | 438 | 31 | 0.576 ± 0.143 | [0.458, 0.690] | 0.168 | 0.730 | 0.551 | 0.343 | 0.759 | 0.151 | 0.063 |

The highest single cross-validated value was 0.764 (w45_fs1:logistic:rf_importance), but the no-selection reference at the production configuration reached 0.753. The difference is well inside the fold-to-fold standard deviation, so feature selection is not supported as a performance improvement here. Its main value is interpretability and a smaller feature set, not a higher AUC.

### 7.3 Selection stability

| Configuration:model | Method | Folds | Mean selected | Union | Always selected | Mean Jaccard |
|:---|:---|---:|---:|---:|---:|---:|
| w30_fs1:logistic | none | 5 | 40.0 | 40 | 40 | 1.000 |
| w30_fs1:logistic | correlation | 5 | 34.0 | 34 | 34 | 1.000 |
| w30_fs1:logistic | mutual_info | 5 | 18.0 | 31 | 9 | 0.549 |
| w30_fs1:logistic | rfe | 5 | 19.2 | 28 | 12 | 0.669 |
| w30_fs1:logistic | l1 | 5 | 17.4 | 24 | 12 | 0.700 |
| w30_fs1:logistic | rf_importance | 5 | 20.8 | 25 | 14 | 0.779 |
| w30_fs1:gradient_boosting | none | 5 | 40.0 | 40 | 40 | 1.000 |
| w30_fs1:gradient_boosting | correlation | 5 | 34.0 | 34 | 34 | 1.000 |
| w30_fs1:gradient_boosting | mutual_info | 5 | 18.0 | 31 | 9 | 0.549 |
| w30_fs1:gradient_boosting | rfe | 5 | 19.2 | 28 | 12 | 0.669 |
| w30_fs1:gradient_boosting | l1 | 5 | 17.4 | 24 | 12 | 0.700 |
| w30_fs1:gradient_boosting | rf_importance | 5 | 20.8 | 25 | 14 | 0.779 |
| w45_fs1:logistic | none | 5 | 40.0 | 40 | 40 | 1.000 |
| w45_fs1:logistic | correlation | 5 | 34.0 | 34 | 34 | 1.000 |
| w45_fs1:logistic | mutual_info | 5 | 17.8 | 27 | 11 | 0.631 |
| w45_fs1:logistic | rfe | 5 | 20.2 | 31 | 11 | 0.620 |
| w45_fs1:logistic | l1 | 5 | 16.6 | 25 | 11 | 0.671 |
| w45_fs1:logistic | rf_importance | 5 | 20.8 | 26 | 14 | 0.758 |
| w45_fs1:gradient_boosting | none | 5 | 40.0 | 40 | 40 | 1.000 |
| w45_fs1:gradient_boosting | correlation | 5 | 34.0 | 34 | 34 | 1.000 |
| w45_fs1:gradient_boosting | mutual_info | 5 | 17.8 | 27 | 11 | 0.631 |
| w45_fs1:gradient_boosting | rfe | 5 | 20.2 | 31 | 11 | 0.620 |
| w45_fs1:gradient_boosting | l1 | 5 | 16.6 | 25 | 11 | 0.671 |
| w45_fs1:gradient_boosting | rf_importance | 5 | 20.8 | 26 | 14 | 0.758 |
| w60_fs1:logistic | none | 5 | 40.0 | 40 | 40 | 1.000 |
| w60_fs1:logistic | correlation | 5 | 34.0 | 34 | 34 | 1.000 |
| w60_fs1:logistic | mutual_info | 5 | 19.4 | 32 | 8 | 0.541 |
| w60_fs1:logistic | rfe | 5 | 19.0 | 34 | 9 | 0.501 |
| w60_fs1:logistic | l1 | 5 | 18.0 | 25 | 12 | 0.699 |
| w60_fs1:logistic | rf_importance | 5 | 20.8 | 26 | 11 | 0.686 |
| w60_fs1:gradient_boosting | none | 5 | 40.0 | 40 | 40 | 1.000 |
| w60_fs1:gradient_boosting | correlation | 5 | 34.0 | 34 | 34 | 1.000 |
| w60_fs1:gradient_boosting | mutual_info | 5 | 19.4 | 32 | 8 | 0.541 |
| w60_fs1:gradient_boosting | rfe | 5 | 19.0 | 34 | 9 | 0.501 |
| w60_fs1:gradient_boosting | l1 | 5 | 18.0 | 25 | 12 | 0.699 |
| w60_fs1:gradient_boosting | rf_importance | 5 | 20.8 | 26 | 11 | 0.686 |

A high Jaccard value means the method chooses nearly the same subset in every fold. Low stability is itself a finding: it means the selected subset is sensitive to the particular training records and should not be interpreted as a unique optimal set.

## 8. Systematic comparison and validation

### 8.1 Staged design

A full 3 × 4 × 6 factorial would require hundreds of deep-learning runs on a CPU-only machine. Phase A therefore uses a staged design:

1. All three window durations at the production 1 Hz with both classical baselines.
2. All four sampling rates at the 30-minute window with both classical baselines.
3. All six feature-selection methods at the production configuration and at the development-CV-selected window.
4. A focused deep-learning screen of the existing multimodal architecture on six representative configurations, using the same 35-epoch budget and the same folds.

### 8.2 Validation protocol

Every comparison uses the same fixed, record-level 5-fold stratified split of the 469 development records (39 cesareans). Because folds are generated once and merged by `record_id`, a configuration that excludes some records still uses the same partition for the records it retains. Metrics are reported as fold mean ± SD and as pooled out-of-fold values with 1000-sample stratified bootstrap 95% confidence intervals for AUROC, AUPRC and MCC.

| Fold | Records | Cesarean | Prevalence |
|---:|---:|---:|---:|
| 1 | 94 | 8 | 8.5% |
| 2 | 94 | 8 | 8.5% |
| 3 | 94 | 8 | 8.5% |
| 4 | 94 | 8 | 8.5% |
| 5 | 93 | 7 | 7.5% |

The minority class is present in every fold, but the absolute number of cesareans per fold is small (about 8). Fold-level metrics therefore have wide uncertainty; the pooled out-of-fold estimate is more stable and is the primary comparison.

### 8.3 Ranking of all configurations

| Configuration | Model | Selection | n | Cesarean | AUROC (mean ± SD) | AUROC 95% CI | AUPRC | Accuracy | Bal. acc. | Sensitivity | Specificity | F1 | MCC |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w30_fs0p5:multimodal | multimodal | none | 469 | 39 | 0.790 ± 0.136 | [0.685, 0.861] | 0.430 | 0.893 | 0.655 | 0.368 | 0.942 | 0.351 | 0.311 |
| w30_fs1:multimodal | multimodal | none | 469 | 39 | 0.758 ± 0.131 | [0.668, 0.845] | 0.414 | 0.906 | 0.634 | 0.307 | 0.960 | 0.364 | 0.349 |
| w30_fs4:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.755 ± 0.104 | [0.655, 0.826] | 0.417 | 0.902 | 0.648 | 0.343 | 0.953 | 0.388 | 0.351 |
| w30_fs2:multimodal | multimodal | none | 469 | 39 | 0.754 ± 0.103 | [0.660, 0.841] | 0.424 | 0.910 | 0.602 | 0.232 | 0.972 | 0.278 | 0.242 |
| w30_fs2:logistic:none | logistic | none | 469 | 39 | 0.754 ± 0.095 | [0.652, 0.832] | 0.397 | 0.789 | 0.700 | 0.593 | 0.807 | 0.321 | 0.267 |
| w30_fs0p5:logistic:none | logistic | none | 469 | 39 | 0.753 ± 0.105 | [0.648, 0.827] | 0.407 | 0.781 | 0.699 | 0.600 | 0.798 | 0.315 | 0.260 |
| w30_fs1:logistic:none | logistic | none | 469 | 39 | 0.753 ± 0.100 | [0.648, 0.830] | 0.385 | 0.778 | 0.685 | 0.571 | 0.798 | 0.297 | 0.239 |
| w30_fs4:logistic:none | logistic | none | 469 | 39 | 0.748 ± 0.107 | [0.637, 0.825] | 0.408 | 0.774 | 0.682 | 0.571 | 0.793 | 0.294 | 0.235 |
| w30_fs2:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.740 ± 0.112 | [0.634, 0.809] | 0.400 | 0.902 | 0.661 | 0.371 | 0.951 | 0.382 | 0.338 |
| w45_fs1:logistic:none | logistic | none | 466 | 37 | 0.739 ± 0.107 | [0.635, 0.820] | 0.288 | 0.801 | 0.695 | 0.569 | 0.821 | 0.311 | 0.259 |
| w45_fs1:multimodal | multimodal | none | 466 | 37 | 0.738 ± 0.149 | [0.641, 0.825] | 0.375 | 0.904 | 0.631 | 0.307 | 0.956 | 0.333 | 0.290 |
| w60_fs1:multimodal | multimodal | none | 438 | 31 | 0.730 ± 0.170 | [0.649, 0.830] | 0.372 | 0.918 | 0.577 | 0.179 | 0.975 | 0.241 | 0.238 |
| w30_fs1:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.729 ± 0.138 | [0.610, 0.801] | 0.394 | 0.902 | 0.673 | 0.396 | 0.949 | 0.407 | 0.361 |
| w30_fs4:multimodal | multimodal | none | 469 | 39 | 0.721 ± 0.112 | [0.635, 0.799] | 0.344 | 0.891 | 0.614 | 0.282 | 0.947 | 0.303 | 0.250 |
| w30_fs0p5:gradient_boosting:none | gradient_boosting | none | 469 | 39 | 0.715 ± 0.160 | [0.605, 0.789] | 0.337 | 0.889 | 0.594 | 0.239 | 0.949 | 0.263 | 0.210 |
| w45_fs1:gradient_boosting:none | gradient_boosting | none | 466 | 37 | 0.701 ± 0.117 | [0.605, 0.780] | 0.270 | 0.904 | 0.593 | 0.224 | 0.963 | 0.277 | 0.240 |
| w60_fs1:logistic:none | logistic | none | 438 | 31 | 0.625 ± 0.132 | [0.518, 0.737] | 0.202 | 0.753 | 0.588 | 0.396 | 0.779 | 0.176 | 0.104 |
| w60_fs1:gradient_boosting:none | gradient_boosting | none | 438 | 31 | 0.570 ± 0.094 | [0.431, 0.660] | 0.150 | 0.902 | 0.522 | 0.079 | 0.966 | 0.084 | 0.050 |

### 8.4 Deep-learning screening

| Configuration | Model | Selection | n | Cesarean | AUROC (mean ± SD) | AUROC 95% CI | AUPRC | Accuracy | Bal. acc. | Sensitivity | Specificity | F1 | MCC |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w30_fs0p5:multimodal | multimodal | none | 469 | 39 | 0.790 ± 0.136 | [0.685, 0.861] | 0.430 | 0.893 | 0.655 | 0.368 | 0.942 | 0.351 | 0.311 |
| w30_fs1:multimodal | multimodal | none | 469 | 39 | 0.758 ± 0.131 | [0.668, 0.845] | 0.414 | 0.906 | 0.634 | 0.307 | 0.960 | 0.364 | 0.349 |
| w30_fs2:multimodal | multimodal | none | 469 | 39 | 0.754 ± 0.103 | [0.660, 0.841] | 0.424 | 0.910 | 0.602 | 0.232 | 0.972 | 0.278 | 0.242 |
| w45_fs1:multimodal | multimodal | none | 466 | 37 | 0.738 ± 0.149 | [0.641, 0.825] | 0.375 | 0.904 | 0.631 | 0.307 | 0.956 | 0.333 | 0.290 |
| w60_fs1:multimodal | multimodal | none | 438 | 31 | 0.730 ± 0.170 | [0.649, 0.830] | 0.372 | 0.918 | 0.577 | 0.179 | 0.975 | 0.241 | 0.238 |
| w30_fs4:multimodal | multimodal | none | 469 | 39 | 0.721 ± 0.112 | [0.635, 0.799] | 0.344 | 0.891 | 0.614 | 0.282 | 0.947 | 0.303 | 0.250 |

These runs are a screening budget, not final models: 35 epochs with early stopping on the validation fold. They are directly comparable to each other because the folds, seed, architecture and budget are identical.

Two patterns stand out. First, extending the window to 60 minutes hurts the deep model much less than the classical baselines: 0.730 versus 0.625 for the best baseline on the same reduced subset. The deep encoder pools across the whole input, so it degrades more gracefully when the 60-minute requirement removes the shortest recordings, whereas the hand-crafted descriptors are computed over a differently composed population. This does not make 60 minutes preferable, because the subset still excludes 10 cesareans; it means the baseline-versus-deep gap at 60 minutes should not be read as a fair architecture comparison.

Second, the highest deep fold-mean AUROC came from the lowest rate (0.790 at 0.5 Hz versus 0.758 at 1 Hz), but the two bootstrap intervals overlap substantially and this is a single-seed screen. At 0.5 Hz much of the beat-to-beat detail is removed, so the apparent gain is more plausibly fold noise or a smoothing effect than a genuine advantage. It is reported as a hypothesis for a future multi-seed check, not as a reason to change the production sampling rate.

### 8.5 Generated figures

| Figure |
|:---|
| [Quality overview](experiments/phase_a/data_quality/figures/fig_quality_overview.png) | [Window and sampling AUROC (logistic)](experiments/phase_a/comparisons/figures/window_and_sampling_auroc.png) | [Window and sampling AUROC (gradient boosting)](experiments/phase_a/comparisons/figures/window_and_sampling_auroc_gb.png) | [Best baseline ROC and PR](experiments/phase_a/comparisons/figures/roc_pr_best_baselines.png) | [Feature-selection comparison](experiments/phase_a/comparisons/figures/feature_selection_auroc.png) | [Confusion matrix (logistic)](experiments/phase_a/comparisons/figures/confusion_logistic.png) | [Confusion matrix (gradient boosting)](experiments/phase_a/comparisons/figures/confusion_gradient_boosting.png) | [Deep multimodal screening](experiments/phase_a/comparisons/figures/deep_screening_auroc.png) |

Feature-selection stability figures for every method are written under `experiments/phase_a/feature_selection/<configuration>/figures/`.

## 9. Decision

The development-CV-selected baseline configuration is **w30_fs1:logistic:none** (logistic, selection `none`), with mean out-of-fold AUROC 0.753 and MCC 0.239.

The best mean AUROC was 0.755 with a standard error of 0.047. Using a one-standard-error rule (threshold 0.708), 9 configurations were statistically indistinguishable. Among them the current production configuration (30 min, 1 Hz, no feature selection) was preferred, because a difference smaller than fold-to-fold noise does not justify extra storage, compute, or a change in the deployed preprocessing path.

The held-out test split was not evaluated. The rule was fixed before the test set could influence any decision, and the selected configuration should be treated as a defensible default rather than a uniquely superior one.

## 10. Recommendations

1. **Keep the 30-minute production window as the default.** It is the only window that supports all 552 records and it avoids the cesarean-selective loss of longer windows. Offer 45 and 60 minutes as explicit sensitivity analyses, not as silent defaults.
2. **Keep 1 Hz for the production model.** Anti-aliased 4/2/1/0.5 Hz inputs do not justify a change on development CV, and 1 Hz has the lowest storage and compute cost. If 4 Hz is ever used, keep the anti-aliasing filter and re-tune the architecture receptive field rather than only changing the input length. The single deep run at 0.5 Hz is the main loose end: it is cheap to confirm with a multi-seed repeat, but it should not be adopted on the strength of one overlapping screening result.
3. **Treat recording duration as a stratification variable.** Cesarean recordings are shorter and have different missingness. Future reports should always show class balance alongside performance for any duration-dependent preprocessing choice.
4. **Use fold-specific feature selection only.** Correlation filtering or MI computed on the full dataset leaks outcome information. The pipeline in `src/phase_a/selection.py` is fitted inside every training fold and forces the protected clinical features.
5. **Prefer no selection or a stable, low-aggression filter.** If the selected subsets are unstable across folds, a simpler model with all features is easier to defend clinically. Do not remove a feature solely because its univariate AUC is near 0.5.
6. **Report confidence intervals and confusion matrices, not only AUC.** With 39 cesareans in development data, a point estimate of AUC is not enough to justify a clinical claim.

## 11. Limitations

- The dataset is small and highly imbalanced: 46 cesareans in total and 39 in the development set. Fold-level estimates remain noisy despite stratification.
- The cesarean subset comes from a separately released 46-case collection. Its recording length, missingness and acquisition characteristics differ from the vaginal majority, so outcome is partly confounded with recording practice.
- Zero runs are treated as missing based on strong empirical evidence, but the raw format does not include an explicit validity channel. A small number of genuine zero-flow samples could be misclassified.
- For anti-aliased downsampling, missing samples are linearly interpolated only so that the digital filter can run; the output mask still marks the whole gap as missing. Valid output samples immediately adjacent to a very long gap can therefore be mildly influenced by the interpolation, although no gap sample is ever presented as observed.
- Noise and artefact detection uses jump and flatline proxies rather than a validated CTG quality index. This is descriptive, not a clinical quality score.
- Hand-crafted CTG descriptors are engineering approximations, not a validated FIGO score. Feature-selection results describe this feature set, not CTG physiology in general.
- Deep-learning screening uses a fixed short budget. A longer training schedule could change the ranking, although all configurations were treated identically.
- All Phase A conclusions are development-set conclusions. The held-out test set remains untouched for a single final confirmation after preprocessing decisions are frozen.

## 12. Reproduction

```bash
# 1. Verify the processed metadata exists (fixed split; raw data is read-only)
python -m src.data.prepare_dataset

# 2. Full Phase A pipeline (quality, windows, sampling, features, comparison, DL, report)
python -m src.phase_a.run --stage all

# Without the CPU-expensive deep-learning screen:
python -m src.phase_a.run --stage all --no-dl

# Reproduce one stage with fresh outputs:
python -m src.phase_a.run --stage quality --force
python -m src.phase_a.run --stage window --force
python -m src.phase_a.run --stage sampling --force
python -m src.phase_a.run --stage features --force
python -m src.phase_a.run --stage report

# Fast smoke run (2 epochs, 30 min / 1 Hz)
python -m src.phase_a.run --stage all --quick --force
```

Every run writes `experiments/phase_a/config.json` (seed, thresholds, rates, windows, environment) and `experiments/phase_a/folds.csv` (record IDs and fold assignments). Per-configuration caches contain the raw window tensors, so downstream stages do not need to re-read the WFDB files.

## 13. Artefact index

| Artefact | Contents |
|:---|:---|
| `experiments/phase_a/config.json` | seed, thresholds, windows, rates, environment |
| `experiments/phase_a/folds.csv` | fixed development-set fold assignments |
| `data_quality/record_quality.csv` | per-record quality metrics, category and reasons |
| `data_quality/quality_summary.json` | aggregate quality and missingness findings |
| `data_quality/quality_sensitivity.csv` | inclusion-policy selection-bias analysis |
| `data_quality/figures/` | duration, missingness and category plots |
| `window_experiments/` | per-window configs, exclusions, CV results and OOF predictions |
| `sampling_experiments/` | per-rate configs, CV results and OOF predictions |
| `feature_selection/` | selector comparison, per-fold selections, stability tables |
| `deep_screening/` | multimodal screening results, histories and OOF predictions |
| `comparisons/` | combined ranking, selected configuration, ROC/PR and confusion figures |
| `logs/` | execution logs for every stage |


---
_This report was generated automatically from the Phase A artefacts. The test split was not used at any point._
