# OB-SIGHT — AI-Based Mobile Clinical Decision Support System for Delivery Outcome Prediction

**Research prototype — not a medical device and not for clinical use.**

OB-SIGHT investigates whether intrapartum cardiotocography (CTG) and maternal clinical data can be used to estimate the **eventual mode of delivery** (cesarean versus vaginal). It is an end-to-end academic project: dataset inspection, leakage-aware preprocessing, PyTorch models, classical baselines, held-out evaluation, calibration, explainability, a FastAPI backend, a mobile-first React interface, automated tests, and reproducible research experiments.

The system is built on the public **CTU-UHB Intrapartum Cardiotocography Database v1.0.0** (PhysioNet, DOI `10.13026/C22013`). It does **not** diagnose fetal distress and does **not** recommend a delivery mode. Version 1.0.0 of this database contains no expert CTG annotations, so fetal-status classification is not a valid target for this data.

> **Research status.** The production prototype and **Phase A** are implemented. **Phase B (Stages B0 and B1)** is implemented and its experiment artefacts are committed to this repository, but its research audit is still pending. Its results are development-set cross-validation estimates only, not held-out test or clinical validation results. B2, B3, Phase C and Phase D are planned. See [Section J](#j-future-research-roadmap).

---

## A. Project overview

### A.1 What OB-SIGHT is

OB-SIGHT is a research-oriented clinical decision support prototype for studying delivery-outcome prediction during labour. It combines two data sources that are available before the mode of delivery is known:

- the most recent **30 minutes of CTG** (fetal heart rate, FHR, and uterine contractions, UC), and
- **maternal clinical variables** such as age, parity, gestational age and selected risk factors.

A model produces a probability of cesarean delivery. The probability is a statistical score from a small, selected, single-centre retrospective dataset. It is not a diagnosis, an individual risk estimate, or a substitute for professional obstetric judgement.

### A.2 Objective and inputs

| Item | Description |
|---|---|
| Task | Binary classification: cesarean delivery (positive) versus vaginal delivery |
| Target field | `Deliv. type` — 1 = vaginal, 2 = cesarean |
| Signal input | 30-minute CTG window at 1 Hz, four channels: FHR, UC, FHR-valid mask, UC-valid mask (`4 × 1800`) |
| Clinical input | 15 encoded features from 12 raw maternal fields |
| Prediction time | During labour, using the most recent 30 minutes of recording |
| Output | Calibrated cesarean probability, binary decision at a stated threshold, and exploratory attributions |

The target is delivery mode, **not** fetal status. The database release has an empty `ANNOTATIONS` file and no `.atr` files, so there are no expert CTG labels (for example FIGO categories) that could support a fetal-status classifier.

### A.3 Research phase status

| Component | Status |
|---|---|
| Production pipeline: preprocessing, four deep models, four classical baselines, calibration, evaluation, API, frontend | Implemented and documented |
| Phase A: data-quality, window, sampling-frequency and feature-selection experiments | **Completed**; committed |
| Phase B, Stage B0: baseline reproduction on the development set | **Implemented and committed**; research audit pending |
| Phase B, Stage B1: imbalance-aware learning | **Implemented and committed**; research audit pending |
| Phase B audit, B2 architecture optimisation, B3 multimodal fusion | Planned |
| Phase C independent external validation | Planned; external dataset access and delivery-mode label availability are still being investigated |
| Phase D application integration and clinical expert feedback | Planned |

### A.4 How to read the metrics

| Metric | Meaning in this project |
|---|---|
| ROC-AUC | Probability that a randomly chosen cesarean record is ranked above a randomly chosen vaginal record. 0.5 is chance. |
| PR-AUC | Area under the precision-recall curve. Because cesarean prevalence is only 8.33%, the no-skill PR-AUC is about 0.083, not 0.5. |
| Sensitivity | Cesarean recall: the proportion of cesarean deliveries detected. |
| Specificity | Vaginal recall: the proportion of vaginal deliveries correctly classified as vaginal. |
| Balanced accuracy | Mean of sensitivity and specificity; less misleading than accuracy under imbalance. |
| MCC | Matthews correlation coefficient; a single balanced summary in [−1, 1]. |
| Brier score | Mean squared error of predicted probabilities; lower is better. |
| ECE | Expected calibration error; lower is better. |

Accuracy is reported for completeness but is **not** an acceptable primary metric here: a classifier that always predicts vaginal delivery reaches 91.7% accuracy while detecting no cesareans.

### A.5 Project at a glance

| Item | Value |
|---|---|
| Cohort | 552 records: 506 vaginal, 46 cesarean (8.33% prevalence) |
| Production split | Fixed stratified 70/15/15: 386 train / 83 validation / 83 test, seed 42 |
| Experiment development set | 469 records (39 cesareans); the 83-record test split is never loaded |
| Deep models | Multimodal CNN + clinical MLP; CTG-only 1D CNN; CTG-only CNN-BiLSTM; clinical-only MLP |
| Classical baselines | Clinical logistic regression; CTG-descriptor logistic regression; combined logistic regression; combined gradient boosting |
| Evaluation | Held-out test, stratified 5-fold CV, bootstrap CIs, paired AUC comparison, calibration |
| Research experiments | Phase A preprocessing study (completed); Phase B imbalance study (local, unpublished) |
| Backend | FastAPI + Pydantic, CSV/JSON CTG uploads |
| Frontend | React 19 + Vite, mobile-first |
| Tests | 121 pytest tests (verified) |

### A.6 Intended use and safety

**Intended use**

- Teaching and demonstration of an end-to-end clinical machine-learning workflow.
- Research on multimodal intrapartum CTG and maternal-data modelling.
- Reproducible benchmarking against simple baselines on a public research dataset.

**Prohibited use**

- Any real clinical decision, triage, counselling, or mode-of-delivery decision.
- Use as a medical device or as input to one without regulatory approval and prospective validation.
- Use on patient data without a lawful basis, ethics approval, and data-protection safeguards.
- Treating demonstration records as real patient cases.

The repository contains no confidential patient information. All records are from the public PhysioNet release, which is distributed for research use.

---

## B. Problem statement and motivation

### B.1 CTG interpretation

Cardiotocography is widely used for intrapartum fetal monitoring. It records fetal heart rate together with uterine activity, and clinicians interpret features such as baseline rate, variability, accelerations and decelerations. Manual interpretation is challenging because:

- the traces are long, noisy and frequently interrupted by signal loss;
- interpretation depends on pattern recognition and clinical context;
- inter-observer and intra-observer agreement is limited, even among experienced clinicians;
- the same trace can be described differently depending on the observer's training and the clinical setting.

These issues motivate research into reproducible, quantitative representations of CTG data. They do **not** imply that an automated model should replace clinical judgement.

### B.2 Why delivery-outcome prediction

Cesarean delivery is a major obstetric intervention with substantial implications for the mother, the baby and the health system. An early, data-driven estimate of the eventual mode of delivery is investigated here as a **research question**: can routinely recorded intrapartum signals and maternal variables carry information about delivery mode, and how stable is that information under different preprocessing and modelling choices?

Delivery mode is also a useful research target because it is recorded in the database and can be verified, unlike fetal-status labels that are absent from this release. The project does not assume that prediction is clinically useful; establishing that would require external and prospective validation.

### B.3 Class imbalance

Only 8.33% of records are cesarean deliveries. This creates three problems:

1. **Accuracy is misleading.** A majority-class classifier is 91.7% accurate and clinically useless.
2. **Sensitivity is unstable.** The development set contains 39 cesareans, so a single extra detection changes pooled sensitivity by about 2.6 percentage points; within a fold of 7–8 cesareans, one case changes fold sensitivity by 12.5–14.3 points.
3. **Rebalancing trades errors.** Weighting or resampling can raise cesarean sensitivity while increasing false positives and worsening calibration. The trade-off must be measured, not assumed.

The project therefore emphasises **PR-AUC**, **cesarean sensitivity at a fixed threshold**, **specificity** and **calibration**, and reports accuracy only as a secondary descriptive value.

### B.4 Research question

> On the public CTU-UHB database, which combination of CTG preprocessing and imbalance-handling strategy gives the most defensible cesarean-detection performance, and how much of the observed difference is distinguishable from fold-to-fold and training variance?

Phase A addresses the preprocessing part; Phase B addresses the imbalance part.

---

## C. Dataset description

### C.1 Source

The project uses the **CTU-UHB Intrapartum Cardiotocography Database v1.0.0** (published on PhysioNet under the title **CTU-CHB Intrapartum Cardiotocography Database**), distributed by PhysioNet and originating from the Czech Technical University in Prague and the University Hospital in Brno.

- PhysioNet: <https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/>
- DOI: `10.13026/C22013`

The original dataset directory is treated as **read-only**. Generated arrays, models and reports are written to `data/processed/`, `models/` and `reports/`.

### C.2 Cohort and target

| Property | Value |
|---|---|
| Records | 552 |
| Vaginal deliveries (`Deliv. type = 1`) | 506, including spontaneous and operative vaginal deliveries |
| Cesarean deliveries (`Deliv. type = 2`) | 46 |
| Cesarean prevalence | 8.33% |
| Missing target labels | 0 |
| Record ID ranges | Vaginal 1001–1506; cesarean 2001–2046 |

The target was verified from every WFDB header, not inferred from file names. All 46 cesarean records belong to the separate 2001–2046 series. The dataset release has no expert CTG annotations, so the project predicts delivery mode rather than fetal status.

### C.3 Signals and recording characteristics

| Property | Value |
|---|---|
| Channels | Two interleaved 16-bit channels: FHR and UC |
| Native sampling frequency | 4 Hz |
| Gain | 100 for both channels |
| FHR unit | beats per minute |
| UC unit | uncalibrated tocodynamometer units (`nd`), not mmHg |
| Raw duration | 60.0–90.1 minutes before trimming |
| Storage | Zero-padded to a whole number of minutes |
| Missing values | Samples `<= 0` are treated as missing, not as physiology |
| Usable 30-minute windows | 552 |

The production pipeline trims leading and trailing zero padding, selects the **last 30 minutes**, screens FHR to 30–240 bpm, interpolates short gaps (≤5 s for FHR, ≤30 s for UC), and reduces the signal to **1 Hz** with a masked block mean. This produces a `4 × 1800` tensor containing FHR, UC and their validity masks.

### C.4 Clinical variables and CTG descriptors

| Group | Contents |
|---|---|
| Raw maternal fields (12) | age, gravidity, parity, gestational age, diabetes, hypertension, preeclampsia, liquor praecox, pyrexia, meconium, presentation, induced labour |
| Encoded clinical features (15) | 4 numeric + 7 binary + 3 presentation one-hot + 1 presentation-missing indicator |
| CTG descriptors (25) | baseline level, variability, accelerations, decelerations, FHR range/percentiles, missing fractions, UC summary statistics |
| Total tabular features | 40 |

Only variables that are plausibly available **during labour and before the mode of delivery is known** are used. Post-delivery and decision variables are excluded to prevent leakage: cord pH, base deficit, pCO2, BE, Apgar scores, NICU stay, seizures, HIE, intubation, neonatal diagnosis, birth weight, sex, `I.stage`, `II.stage`, `NoProgress`, `CK/KP`, `Pos. II.st.`, `Sig2Birth`, `Rec. type` and identifiers.

### C.5 Splits and leakage control

| Split | Records | Cesareans |
|---|---:|---:|
| Training | 386 | 32 |
| Validation | 83 | 7 |
| Held-out test | 83 | 7 |
| **Total** | **552** | **46** |
| Phase A/B development set (train + validation) | 469 | 39 |

The production split is fixed and stratified at record level with seed 42. Preprocessing statistics, calibration and thresholds are fitted without the test set.

Phase A and Phase B use only the **469 development records**. The Phase B loader explicitly raises an error if a `test` record appears in its bundle, and the 83-record held-out split was not loaded during Phase B development or experimentation.

### C.6 Data-quality summary

Phase A audited every record. Mean missingness in the 30-minute window is **18.4% for FHR** and **21.0% for UC**. Missingness is substantially higher for vaginal than for cesarean recordings, and recording duration is associated with outcome: cesarean recordings are systematically shorter. Both facts are treated as potential confounders rather than ignored.

Full details: [`reports/dataset_inspection_report.md`](reports/dataset_inspection_report.md) and [`experiments/phase_a/reports/PHASE_A_DATA_HANDLING_REPORT.md`](experiments/phase_a/reports/PHASE_A_DATA_HANDLING_REPORT.md).

### C.7 Citation

If you use the dataset, cite:

> Chudáček V, Spilka J, Burša M, et al. Open access intrapartum CTG database. *BMC Pregnancy and Childbirth*. 2014;14:16. doi:10.1186/1471-2393-14-16.

---

## D. System architecture and models

### D.1 End-to-end pipeline

```text
Raw WFDB record (.hea + .dat, FHR + UC at 4 Hz)
        │
        ├── trim zero padding → select last 30 min → screen FHR 30–240 bpm
        │   → interpolate short gaps → masked block mean to 1 Hz
        │   → training-set standardisation → 4 × 1800 tensor
        │
        └── header clinical fields → 15 encoded maternal features
        │
        ▼
Model family (deep or classical)
        │
        ▼
Validation-fitted Platt calibration → validation-selected threshold
        │
        ▼
FastAPI service → React mobile interface / JSON API
```

### D.2 Implemented model families

| # | Model | Inputs | Role |
|---|---|---|---|
| 1 | CTG-only 1D CNN (`signal`) | CTG tensor | Deep ablation; CTG-only comparison |
| 2 | CNN-BiLSTM (`bilstm`) | CTG tensor | Temporal deep ablation |
| 3 | Multimodal CNN + Clinical MLP (`multimodal`) | CTG tensor + clinical vector | Primary deep model in the shipped artefacts |
| 4 | Clinical-only MLP (`clinical`) | Clinical vector | Clinical ablation |
| 5 | Logistic regression | Clinical, CTG descriptors, or combined | Classical baseline |
| 6 | Gradient boosting | Clinical + CTG descriptors | Classical baseline |

The **multimodal model is the primary model in the shipped artefacts and model card**. The CNN-BiLSTM and 1D CNN are comparison models, and the web UI exposes the multimodal and CNN-BiLSTM variants for assessment. The 1D CNN and clinical-only model are available through the API. None of these models is clinically validated, and “primary” here means “the reference artefact in this repository”, not “approved for care”.

### D.3 Model descriptions

**1. CTG-only 1D CNN.** Four 1D convolution blocks (24–48–64–96 channels) with batch normalisation, ReLU and pooling, followed by concatenated average/max pooling and a 96-unit embedding. It has 58,009 parameters. It tests how much information is available from the CTG tensor alone.

**2. CNN-BiLSTM.** Three convolution blocks (16–32–48 channels) with two pooling stages produce a 225-step sequence. A one-layer bidirectional LSTM (32 hidden units per direction) models temporal context, followed by dropout and a small fully connected head. It has 31,009 parameters. It consumes the same sequential `4 × 1800` tensor, not tabular descriptors.

**3. Multimodal CNN + Clinical MLP.** The signal encoder above is combined with a 48 → 32 clinical MLP; the two embeddings are fused through a 64-unit layer. It has 64,121 parameters and is the primary reference deep model.

**4. Clinical-only MLP.** A 48 → 32 clinical encoder with a small head. It has 3,041 parameters and provides a clinical-data-only reference point.

**5. Logistic regression.** Three variants are fitted: clinical features only, the 25 CTG descriptors only, and clinical + CTG descriptors. The implementation uses L2 regularisation (`C = 0.5`) and `class_weight='balanced'`.

**6. Gradient boosting.** A histogram gradient-boosting classifier on the combined 40-feature tabular matrix, with balanced class weights. It provides a non-linear classical reference.

### D.4 Production training and calibration protocol

Deep models use class-weighted binary cross-entropy, AdamW (learning rate `1e-3`, weight decay `1e-4`), dropout, batch normalisation, mild Gaussian noise on the FHR/UC channels only, gradient clipping, and early stopping on validation loss. The current CLI defaults allow up to 120 epochs with an early-stopping patience of 20; the shipped checkpoints were trained with the recorded configuration of **80 epochs, patience 15, batch size 32 and seed 42**.

After training:

- **Platt scaling** is fitted on the validation split.
- The **decision threshold** is selected by Youden's J on the validation split.
- The threshold is applied unchanged to the held-out test set.
- Preprocessing statistics, calibration and thresholds are fitted without the test set.

The test split is never used for model selection, early stopping, calibration or threshold choice.

### D.5 Production reference results (held-out test)

These are **production pipeline results**, not Phase A or Phase B experimental results. They come from the fixed 83-record held-out test set (7 cesareans / 76 vaginal). Thresholds are validation-selected.

| Model | Threshold | ROC-AUC (95% CI) | PR-AUC | Sensitivity | Specificity | F1 |
|---|---:|---|---:|---:|---:|---:|
| Multimodal deep | 0.14 | 0.733 (0.551–0.910) | 0.402 | 0.429 | 0.842 | 0.273 |
| CTG-only 1D CNN | 0.08 | 0.797 (0.660–0.914) | 0.305 | 0.571 | 0.684 | 0.229 |
| CTG-only CNN-BiLSTM | 0.08 | 0.720 (0.453–0.938) | 0.450 | 0.571 | 0.750 | 0.267 |
| Clinical-only deep | 0.11 | 0.498 (0.244–0.727) | 0.105 | 0.143 | 0.829 | 0.095 |
| CTG logistic regression | 0.33 | 0.701 (0.451–0.904) | 0.204 | 0.714 | 0.618 | 0.244 |
| Combined gradient boosting | 0.37 | 0.675 (0.425–0.882) | 0.194 | 0.286 | 0.895 | 0.235 |
| Combined logistic regression | 0.64 | 0.611 (0.397–0.810) | 0.184 | 0.286 | 0.855 | 0.200 |
| Clinical logistic regression | 0.61 | 0.468 (0.238–0.716) | 0.103 | 0.286 | 0.697 | 0.125 |

**Read these numbers carefully:**

- Only **7 cesareans** are in the test set. Confidence intervals are wide and overlap.
- The CNN-BiLSTM had the highest PR-AUC (0.450), but its ROC-AUC was lower than the 1D CNN, and the paired AUC difference was −0.077 (95% CI −0.331 to +0.167). The experiment does not establish a statistically meaningful improvement.
- A majority-class classifier reaches 91.6% accuracy at threshold 0.5 while detecting zero cesareans.
- The clinical-only deep model is at chance on this test set.
- At the validation-selected threshold, the multimodal model detected 3/7 cesareans with 12/76 false positives.
- Low validation-selected thresholds trade specificity for sensitivity; the resulting false-positive rates are not clinically acceptable.
- No external or prospective validation exists.

All key numbers in the web interface are served from `reports/metrics.json`; none are hand-typed into the frontend.

### D.6 Production five-fold cross-validation

Stratified 5-fold cross-validation on all 552 records gives pooled out-of-fold ROC-AUC: multimodal 0.767, CTG-only 1D CNN 0.763, CNN-BiLSTM 0.810, clinical-only 0.561, CTG logistic regression 0.744, combined logistic regression 0.726, combined gradient boosting 0.702, clinical logistic regression 0.577. The CNN-BiLSTM also had the highest pooled PR-AUC (0.399).

The test-set and CV rankings disagree: CV ranks the CNN-BiLSTM first, while the test set ranks the 1D CNN first. On 46 cesareans, this reversal is evidence of instability, not of a reliable winner. See [`reports/cv_summary.md`](reports/cv_summary.md).

**Overfitting.** The CNN-BiLSTM training loss fell from 1.253 at epoch 1 to 0.054 at epoch 34, while validation loss reached its minimum of 0.334 at epoch 19 and then rose again. Validation ROC-AUC peaked around epoch 14. The epoch-19 checkpoint was retained and early stopping ended training at epoch 34. With 386 training records and only 32 training cesareans, this is clear overfitting; regularisation reduced but did not eliminate it. No synthetic data were used, and the test split was never used for early stopping.

---

## E. Phase A — Data handling and preprocessing optimisation

> **Status: COMPLETED.** Phase A is an additive research layer under `src/phase_a/`. It does not change the production preprocessing, saved models, API or frontend, and it never loads the held-out test split. The completed report is [`experiments/phase_a/reports/PHASE_A_DATA_HANDLING_REPORT.md`](experiments/phase_a/reports/PHASE_A_DATA_HANDLING_REPORT.md) (also copied to [`PHASE_A_DATA_HANDLING_REPORT.md`](PHASE_A_DATA_HANDLING_REPORT.md)).

### E.1 Scope and validation protocol

Phase A asked four questions: how good is the data, how long should the CTG window be, at what sampling frequency should it be represented, and should features be selected. All experiments use the **469 development records** (39 cesareans), the same fixed record-level 5-fold split, seed 42, and pooled out-of-fold metrics with 1000-sample stratified bootstrap confidence intervals. Fold sizes are 94/94/94/94/93 with 8/8/8/8/7 cesareans.

The 83-record held-out test split was not used at any point. A full 3 × 4 × 6 factorial over window, rate and selection method would have required hundreds of CPU deep-learning runs, so Phase A used a staged design: all window durations at 1 Hz, all sampling rates at 30 minutes, all selection methods at the production configuration and the development-CV-selected window, and a focused six-configuration deep screen of the multimodal model.

### E.2 Data-quality analysis

Each record receives a per-record quality row and a descriptive category.

| Category | Records |
|---|---:|
| Good | 89 |
| Acceptable | 267 |
| Poor | 196 |

The most frequent reasons for not being “good” were UC missing fraction (132 records), FHR missing fraction (111), longest UC gap (71) and longest FHR gap (24). These labels are descriptive; no record is deleted from the main experiments, and every exclusion is recorded explicitly.

Trimmed recording duration has mean 70.2 minutes, median 69.9, minimum 41.2 and maximum 90.1. The shortest recording still supports a 30-minute window, so the production window never requires padding.

### E.3 Missing-value analysis

Target labels are complete. Clinical missingness is very low: gravidity 0.72%, presentation 0.54%, and 0% for the remaining clinical fields.

CTG missingness in the 30-minute window is substantial:

| Metric | Mean | Median | p90 | Max |
|---|---:|---:|---:|---:|
| FHR missing fraction | **18.39%** | 14.68% | 41.09% | 92.36% |
| UC missing fraction | **20.95%** | 11.88% | 52.77% | 100.00% |
| Longest FHR gap | 1.79 min | 0.70 min | 5.18 min | 22.93 min |
| Longest UC gap | 3.85 min | 1.32 min | 11.42 min | 30.00 min |

Missingness differs by outcome: mean FHR missingness is 19.2% for vaginal and 9.3% for cesarean records; mean UC missingness is 21.5% and 14.8% respectively. No record has an entirely missing FHR channel in the 30-minute window, and 2 records have an entirely missing UC channel. Records with more than half of the window missing (25 for FHR, 66 for UC) are retained and flagged rather than deleted.

### E.4 Zero-value investigation

Exact zeros are treated as **missing values**, not as physiological measurements. Three pieces of evidence support this:

1. The smallest positive UC value in the entire database is **0.5**, and no positive UC sample is below 0.1. A genuine quiescence reading of 0.0 would produce a continuous distribution near zero, which is not observed.
2. FHR of 0 bpm is not physiological during labour.
3. Zero runs are long and contiguous, with a maximum UC run of 75 minutes, and partly coincide across channels — characteristic of signal loss.

The raw zero fractions and the count of positive values below 0.1 are retained in the audit so the assumption can be re-checked against the source data.

### E.5 CTG observation-window experiments

Windows are taken from the most recent valid signal. Records shorter than the requested window are excluded and listed; they are never silently padded.

| Window | Database records supporting window | Development records in CV | Database cesareans | Excluded records (cesareans) | Logistic ROC-AUC (mean ± SD) | Gradient boosting ROC-AUC (mean ± SD) |
|---|---:|---:|---:|---:|---:|---:|
| 30 min | 552 | 469 | 46 | 0 | **0.753 ± 0.100** | **0.729 ± 0.138** |
| 45 min | 548 | 466 | 43 | 4 (3) | 0.739 ± 0.107 | 0.701 ± 0.117 |
| 60 min | 511 | 438 | 36 | 41 (10) | 0.625 ± 0.132 | 0.570 ± 0.094 |

AUC columns are development-set cross-validation estimates on the records retained by each window (469, 466 and 438 records respectively).

The 45-minute window removes 3 cesareans (6.5% of all cesareans); the 60-minute window removes 10 (21.7%). Because cesarean recordings are systematically shorter, longer windows change the population being evaluated. The apparent performance drop is therefore partly a **selection-bias effect**, not only a signal-length effect. The 60-minute requirement also removes 41 of 552 records. Longer windows are not adopted as a default.

### E.6 Sampling-frequency experiments

All rates are derived from the native 4 Hz signal with a 4th-order zero-phase Butterworth anti-aliasing filter applied **inside the requested window only**. Long gaps remain missing in the mask; interpolation is used only so the digital filter can run.

| Rate | 30-minute samples | Logistic ROC-AUC | Gradient boosting ROC-AUC |
|---|---:|---:|---:|
| 4 Hz | 7200 | 0.748 | 0.755 |
| 2 Hz | 3600 | 0.754 | 0.740 |
| 1 Hz | 1800 | 0.753 | 0.729 |
| 0.5 Hz | 900 | 0.753 | 0.715 |

Higher sampling frequency did **not** demonstrate a consistent advantage. The logistic results are essentially flat from 0.5 to 2 Hz; the gradient-boosting results favour 4 Hz, while the deep screen favoured 0.5 Hz (0.790 versus 0.758 at 1 Hz), but those deep intervals overlap substantially and the 0.5 Hz result is a single-seed screening observation. At 0.5 Hz much of the beat-to-beat detail is removed, so the apparent gain is more plausibly fold noise or smoothing than a genuine advantage. It remains a hypothesis for a future multi-seed check, not a reason to change the production rate.

### E.7 Feature-selection experiments

The candidate set contains 40 features: 15 clinical and 25 hand-crafted CTG descriptors. Six methods were compared: no selection, correlation filtering (`|r| ≥ 0.90`), mutual information (top 15 plus protected features), recursive feature elimination (15 plus protected), L1 logistic regression, and random-forest importance (top 15 plus protected). Six clinically important features are always protected: age, gestational age, parity, meconium and two presentation indicators.

Selection is fitted **inside every training fold** using a scikit-learn pipeline; the validation fold is transformed with the selector fitted on the other four folds. The test split is never used.

| Observation | Result |
|---|---|
| Best single cross-validated value | 0.764 (45 min, logistic regression, random-forest importance) |
| Production configuration, no selection | 0.753 |
| Difference | Within fold-to-fold standard deviation, about 0.10 |
| Stability | Correlation filtering selected 34 features with Jaccard 1.000; mutual information selected about 18 features with Jaccard 0.549 |

Feature selection did **not** produce consistent improvements. Mutual-information and random-forest selection were clearly worse for gradient boosting at 30 minutes; recursive elimination reached 0.760 for logistic regression but 0.708 for gradient boosting. The highest value was a single configuration among many comparisons, and it did not survive a one-standard-error interpretation. Feature selection's defensible value here is interpretability and a smaller feature set, not higher AUC.

### E.8 Cross-validation and reproducibility

- Fixed record-level 5-fold stratified split of the 469 development records.
- Pooled out-of-fold metrics with 1000-sample stratified bootstrap 95% confidence intervals for ROC-AUC, PR-AUC and MCC.
- Fold mean ± SD reported alongside pooled values.
- Seed 42, deterministic classical estimators, fixed fold assignments.
- Every run writes `experiments/phase_a/config.json` (seed, thresholds, windows, rates, environment) and `experiments/phase_a/folds.csv` (record IDs and fold assignments).
- Performance is reported with confidence intervals and confusion matrices; a point AUC alone is not treated as sufficient.

### E.9 Selected preprocessing configuration

The development-CV-selected baseline configuration is:

> **30-minute CTG window + 1 Hz sampling + no feature selection** (`w30_fs1:logistic:none`; mean out-of-fold ROC-AUC 0.753, MCC 0.239).

Across the baseline window/sampling comparison, the best mean AUC was 0.755 with a standard error of 0.047. A one-standard-error rule (threshold 0.708) left **nine configurations statistically indistinguishable**. The current production configuration was retained because a difference smaller than fold-to-fold noise does not justify extra storage, compute, or a change to the deployed preprocessing path.

This configuration was retained by the Phase A **selection procedure**, not because every experiment proved it universally optimal. The held-out test split was not evaluated during selection, and the configuration should be treated as a defensible default rather than a uniquely superior choice.

### E.10 Key Phase A findings

- All dataset assumptions were verified: 552 records, two channels, 4 Hz native rate, 552 usable 30-minute windows, 46 cesareans.
- Exact zeros are missing values, not physiology.
- Recording length is confounded with outcome; longer windows preferentially remove cesarean cases.
- The 30-minute window supports every record and has the cleanest class balance.
- Higher sampling frequency did not show a consistent advantage.
- Feature selection changed the feature count much more than it changed out-of-fold ROC-AUC.
- The production configuration (30 min / 1 Hz / no selection) is retained as a defensible default, not as a proven optimum.

---

## F. Phase B — Model optimisation and imbalance-aware learning

> **IMPLEMENTED — RESEARCH AUDIT PENDING — DEVELOPMENT CROSS-VALIDATION ONLY.**
>
> Phase B code (`src/phase_b/`), tests (`tests/test_phase_b.py`) and artefacts (`experiments/phase_b/`) are included in this repository. The results below are **development-set, out-of-fold cross-validation estimates**, not held-out test performance, not external validation, and not clinically validated findings. The planned research audit (multi-seed checks, calibration review, independent review of the selection rule) has not yet been completed.

Phase B is an additive research layer under `src/phase_b/`. It reuses the existing architectures, the Phase A 30-minute / 1 Hz preprocessing, the Phase A fold assignments and the existing loss and metric conventions. The complete methodology, per-fold tables, confidence intervals, calibration results and limitations are in [`experiments/phase_b/reports/PHASE_B_IMBALANCE_REPORT.md`](experiments/phase_b/reports/PHASE_B_IMBALANCE_REPORT.md).

### F.1 Scope, data and leakage controls

| Item | Value |
|---|---|
| Records | 469 development records |
| Cesareans | 39 (8.3%) |
| Vaginal | 430 |
| Outer validation | 5 stratified patient-level folds, fixed Phase A assignments, seed 42 |
| Inner validation | 3 stratified folds inside each outer training partition |
| Primary threshold | 0.5, fixed before evaluation |
| Held-out test | 83 records; **not loaded and not evaluated** |
| Deep strategies | 5 (plain BCE, weighted BCE, focal γ=1, focal γ=2, weighted random sampling) |
| Classical strategies | 4 (none, class weights, random oversampling, tabular SMOTE) |

Leakage controls:

- The loader raises an exception if any `test` record appears in the Phase B bundle.
- Signal standardisation and clinical imputation/scaling are fitted inside each training partition.
- Class weights, sampling weights, oversampling and SMOTE are computed from **outer-training labels only**.
- The weighted sampler is created over the training dataset only; validation data are never resampled.
- SMOTE is applied only to the standardised 2-D tabular matrix. Raw `(4, 1800)` CTG tensors are **never** interpolated or oversampled.
- The outer validation fold is scored exactly once per fold and is never used for early stopping or threshold selection.

Deep models use a common reproduction protocol: weighted BCE, fixed learning rate `1e-3`, a **25-epoch budget with no outer-validation early stopping** (shorter than the production deep-training budgets: 80 epochs for the shipped checkpoints, with a CLI default of up to 120), and the production preprocessing. Classical models use the production `class_weight='balanced'` settings in B0.

### F.2 Stage B0 — baseline reproduction

B0 reruns the six existing model families on the fixed outer folds with seed 42 and threshold 0.5, using the production settings only (weighted BCE for deep models, balanced class weights for classical models). No additional imbalance strategy is applied. All values are pooled out-of-fold development-set estimates.

| Model | Family | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|
| CTG-only 1D CNN | deep | 0.751 | 0.335 | 0.256 | 0.965 |
| CNN-BiLSTM | deep | **0.801** | **0.391** | 0.487 | 0.937 |
| Multimodal CNN + Clinical MLP | deep | 0.761 | 0.355 | 0.359 | 0.940 |
| Clinical-only MLP | deep | 0.574 | 0.154 | 0.436 | 0.644 |
| Logistic regression | classical | 0.745 | 0.338 | 0.564 | 0.798 |
| Gradient boosting | classical | 0.708 | 0.296 | 0.385 | 0.949 |

The **CNN-BiLSTM was the strongest baseline deep-learning architecture** by ROC-AUC and PR-AUC. It did not have the highest sensitivity: logistic regression detected 56.4% of cesareans at 0.5 but with much lower specificity (0.798). The clinical-only MLP was close to chance on ROC-AUC, consistent with the CTG signal carrying most of the discriminative information in this dataset.

B0 is a reproduction protocol, not the production 120-epoch schedule. Small differences from the production CV values in Section D are therefore expected. The `none` augmentation control in Appendix A of the Phase B report reproduced B0 multimodal exactly at a fixed thread count.

### F.3 Stage B1 — strategies investigated

**Deep strategies (applied one at a time):**

| Strategy | Loss | Sampling |
|---|---|---|
| Plain BCE | `BCEWithLogitsLoss` | Shuffle |
| Weighted BCE | `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` | Shuffle |
| Focal loss γ=1 | `FocalLoss(gamma=1, alpha=n_neg/n)` | Shuffle |
| Focal loss γ=2 | `FocalLoss(gamma=2, alpha=n_neg/n)` | Shuffle |
| Weighted random sampling | `BCEWithLogitsLoss` | `WeightedRandomSampler` with inverse class frequency |

Focal-loss α is computed from the training-fold class distribution. The strategies are deliberately **not combined**: this stage isolates one intervention at a time.

**Classical strategies:**

| Strategy | Implementation | Constraint |
|---|---|---|
| No rebalancing | Plain estimator fit | Reference |
| Class weights | `class_weight='balanced'` | Production classical baseline |
| Random oversampling | Minority rows sampled with replacement to parity | Training partition only |
| SMOTE | Custom tabular SMOTE, k=5, interpolation to parity | Training partition only; never raw CTG tensors |

**Exploratory CTG augmentation** is kept outside the B1 strategy comparison and applied only to the multimodal model: no augmentation, mild Gaussian noise, and a mixed mode combining Gaussian noise, ±5% amplitude scaling and ±30-second time shifting. Masks are never modified.

### F.4 Deep-learning strategy results

Mean over the four deep architectures (pooled out-of-fold metrics, threshold 0.5):

| Strategy | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Plain BCE | 0.720 | 0.261 | 0.019 | 0.996 |
| Weighted BCE | 0.716 | 0.258 | 0.494 | 0.813 |
| Focal Loss γ=1 | 0.715 | 0.268 | 0.538 | 0.791 |
| Focal Loss γ=2 | 0.711 | 0.272 | 0.583 | 0.760 |
| **Weighted random sampling** | **0.728** | **0.290** | 0.449 | 0.872 |

Key observations:

- **Weighted random sampling** gave the best mean PR-AUC and ROC-AUC, while retaining high specificity (0.872). It also had the best calibration among the rebalancing methods (mean Brier 0.119 and ECE 0.139, versus 0.138 / 0.178 for weighted BCE and 0.171 / 0.298 for focal γ=2).
- **Plain BCE collapsed at threshold 0.5**: mean sensitivity 0.019, with almost all cesareans below the cut-off. Its low Brier score (0.072) reflects conservative probabilities near the low prevalence, not useful detection. Its threshold-free ROC-AUC/PR-AUC are also below the sampling strategy, so the problem is not only the cut-off.
- **Focal losses raised sensitivity** (0.538 at γ=1, 0.583 at γ=2) but lowered PR-AUC and substantially worsened calibration. Focal γ=2 had the highest sensitivity and the worst Brier score.
- The best PR-AUC strategy is only 0.032 above weighted BCE. With 39 cesareans, differences of this size are exploratory.

### F.5 Architecture-specific responses

The pooled ranking hides substantial heterogeneity. Best strategy by PR-AUC for each architecture:

| Model | Best PR-AUC strategy | PR-AUC | ROC-AUC | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|
| CTG-only 1D CNN | Weighted random sampling | 0.315 | 0.754 | 0.385 | 0.958 |
| CNN-BiLSTM | Weighted random sampling | **0.404** | 0.818 | 0.615 | 0.895 |
| Multimodal CNN + Clinical MLP | Focal loss γ=1 | 0.342 | 0.740 | 0.513 | 0.891 |
| Clinical-only MLP | Weighted random sampling | 0.146 | 0.592 | 0.513 | 0.677 |

**CNN-BiLSTM with weighted random sampling achieved PR-AUC 0.404**, the best single configuration in Phase B. Its ROC-AUC was 0.818, sensitivity 0.615 and specificity 0.895. The same architecture under weighted BCE reached 0.323 / 0.825 / 0.667 / 0.888. The sampling gain in PR-AUC is therefore about 0.081, but the paired bootstrap interval for that specific comparison is wide and overlaps zero.

For the CNN-BiLSTM specifically:

| Strategy | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Plain BCE | 0.813 | 0.314 | 0.026 | 0.991 |
| Weighted BCE | 0.825 | 0.323 | 0.667 | 0.888 |
| Focal loss γ=1 | 0.824 | 0.361 | 0.667 | 0.844 |
| Focal loss γ=2 | 0.821 | 0.374 | 0.667 | 0.830 |
| Weighted random sampling | 0.818 | **0.404** | 0.615 | 0.895 |

The multimodal model did **not** improve under any Phase B strategy relative to its B0 result (weighted BCE 0.752 / 0.340 versus B0 0.761 / 0.355; sampling 0.749 / 0.294). This is a negative finding: adding clinical data did not make imbalance handling more effective in this cohort.

### F.6 Classical results

| Model | Strategy | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|
| Logistic regression | No rebalancing | **0.760** | **0.342** | 0.179 | 0.991 |
| Logistic regression | Class weights | 0.727 | 0.286 | 0.564 | 0.807 |
| Logistic regression | Random oversampling | 0.730 | 0.284 | 0.564 | 0.812 |
| Logistic regression | SMOTE (tabular) | 0.736 | 0.300 | 0.564 | 0.809 |
| Gradient boosting | No rebalancing | 0.672 | 0.269 | 0.128 | 0.988 |
| Gradient boosting | Class weights | 0.704 | **0.337** | 0.359 | 0.951 |
| Gradient boosting | Random oversampling | 0.703 | 0.260 | 0.282 | 0.958 |
| Gradient boosting | SMOTE (tabular) | **0.736** | 0.292 | 0.333 | 0.935 |

Logistic regression ranked best **without rebalancing** on ROC-AUC and PR-AUC, but detected only 17.9% of cesareans at 0.5. Class weighting raised sensitivity to 56.4% while lowering ROC-AUC to 0.727 and worsening the Brier score from 0.065 to 0.158. For gradient boosting, class weighting gave the best PR-AUC (0.337); SMOTE gave the best ROC-AUC (0.736) but a slightly lower PR-AUC. Tabular SMOTE and random oversampling did not produce a consistent advantage over class weighting.

### F.7 Exploratory CTG augmentation

Applied only to the multimodal architecture and kept outside the B1 strategy comparison:

| Mode | ROC-AUC | PR-AUC | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|
| None | 0.761 | 0.355 | 0.359 | 0.940 | 0.091 |
| Gaussian noise | 0.761 | 0.336 | 0.333 | 0.940 | 0.094 |
| Mixed Gaussian / scale / time shift | 0.756 | 0.358 | 0.487 | 0.905 | 0.103 |

The mixed mode raised sensitivity but lowered specificity and calibration. No augmentation mode produced a clear improvement in ROC-AUC. The `none` mode reproduced B0 multimodal exactly, which is a useful determinism check rather than evidence of augmentation benefit.

### F.8 Statistical uncertainty and interpretation

- Pooled confidence intervals use a stratified bootstrap with 1000 resamples and seed 42.
- Strategy differences use paired stratified bootstrap against weighted BCE (deep) or class-weighted (classical) references.
- **Only one paired PR-AUC difference had a 95% CI excluding zero**: CTG-only 1D CNN with weighted sampling versus weighted BCE, +0.092 (95% CI 0.016–0.160). Most other differences overlap zero.
- Sensitivity differences can appear significant at the fixed 0.5 threshold without a corresponding improvement in ranking (ROC-AUC/PR-AUC).
- Fold-level sensitivity for the recommended strategy ranges from **0.125 to 0.625** across the five outer folds (7–8 cesareans each).
- A difference below roughly **0.02 PR-AUC** is treated as a tie, and p-values are unadjusted for multiple comparisons.

**No strategy has yet been established as statistically superior across all models.** The pooled recommendation is a default, not a claim that one loss dominates for every architecture.

### F.9 Phase B artefacts

All outputs are written to `experiments/phase_b/`:

| Path | Contents |
|---|---|
| `experiments/phase_b/config.json` | Phase B configuration and environment |
| `experiments/phase_b/folds.csv` | Outer fold assignments (Phase A folds) |
| `experiments/phase_b/dataset_manifest.json` | Dataset counts, hashes, fold composition |
| `experiments/phase_b/b0/<model>/result.json` | B0 metrics, predictions and folds |
| `experiments/phase_b/b1/deep/<model>/<strategy>/` | Checkpoints, predictions, histories, metrics |
| `experiments/phase_b/b1/classical/<model>/<strategy>/` | Predictions, selections, metrics |
| `experiments/phase_b/b1/augmentation/<mode>/` | Exploratory multimodal augmentation runs |
| `experiments/phase_b/comparisons/` | Comparison tables, paired differences, recommendation |
| `experiments/phase_b/comparisons/figures/` | ROC, PR, calibration, confusion and fold figures |
| `experiments/phase_b/logs/` | Stage logs |

Each run records fold assignments, configuration, losses, class weights, sampler configuration, seed, checkpoints, predictions, metrics, training histories and logs.

---

## G. Research findings and limitations

### G.1 Consolidated findings

**Phase A (preprocessing):**

- The 30-minute window supports all 552 records and avoids the cesarean-selective exclusion of 45- and 60-minute windows.
- 45- and 60-minute windows did not improve logistic regression (0.739 and 0.625 versus 0.753) and removed 3 and 10 cesareans respectively.
- Sampling rates from 0.5 to 4 Hz produced no consistent advantage; 1 Hz remains the storage- and compute-efficient default.
- Feature selection did not improve out-of-fold performance consistently; the best value (0.764) was within noise of no selection (0.753).
- The production configuration is a defensible default selected by a one-standard-error rule, not a proven optimum.

**Phase B (imbalance):**

- The CNN-BiLSTM was the strongest baseline deep architecture (ROC-AUC 0.801, PR-AUC 0.391).
- Weighted random sampling gave the best mean PR-AUC (0.290) and ROC-AUC (0.728) across four deep architectures, with the best calibration among rebalancing methods.
- CNN-BiLSTM with weighted sampling reached PR-AUC 0.404, the best single Phase B result.
- Focal loss raised sensitivity (up to 0.583 at γ=2) but worsened PR-AUC and calibration.
- Plain BCE at threshold 0.5 detected almost no cesareans, despite a low Brier score.
- Different architectures responded differently: sampling was best for the CTG-only and CNN-BiLSTM models, focal γ=1 for multimodal, and sampling for clinical-only.
- Tabular SMOTE and oversampling did not produce a consistent classical advantage.
- Exploratory CTG augmentation did not produce a clear gain.
- No strategy is statistically established as superior across all models.

### G.2 Negative and inconclusive findings

These are reported explicitly rather than omitted:

- Longer CTG windows did not help and introduced selection bias.
- Higher sampling frequency did not help consistently.
- Feature selection did not help consistently.
- The multimodal model did not improve under Phase B imbalance strategies.
- The clinical-only model was near chance for ranking.
- CTG augmentation did not improve ROC-AUC.
- Tabular SMOTE did not consistently beat class weighting.
- Most paired strategy differences overlap zero; only one PR-AUC comparison excluded zero.
- The test-set and cross-validation rankings of deep models disagree.

### G.3 Limitations

| Limitation | Consequence |
|---|---|
| **Small cesarean sample** | 46 cesareans in total, 39 in development, 7 in test. Confidence intervals are wide and single cases move estimates materially. |
| **Class imbalance** | 8.33% prevalence makes accuracy misleading and sensitivity unstable. |
| **Single primary seed** | Deep experiments use seed 42; no multi-seed variance study has been completed. Training variance may be comparable to strategy effects. |
| **Fold-level variability** | Recommended-strategy sensitivity ranges 0.125–0.625 across outer folds; per-fold rankings should not be treated as stable. |
| **Threshold selection and calibration** | Primary metrics use threshold 0.5. Inner-selected thresholds are reported secondarily, but no outer-fold calibration study has been completed. A transferred threshold can over-predict the minority class. |
| **Multimodal optimisation** | The multimodal model did not improve under Phase B strategies and requires further work on fusion, missingness handling and regularisation. |
| **No external validation** | All results are from one public single-centre retrospective database. No independent cross-dataset validation has been performed. |
| **No prospective clinical validation** | No prospective or clinical workflow evaluation exists, and no regulatory approval has been sought. |
| **Recording-practice confounding** | Cesarean recordings are shorter and come from a separately released 2001–2046 series; outcome is partly confounded with acquisition practice. |
| **No expert CTG labels** | The release has no annotations, so fetal-status performance cannot be measured and the CTG descriptors are engineering approximations rather than a validated FIGO score. |
| **Zero-as-missing assumption** | The format has no explicit validity channel; a small number of genuine zero-flow samples could be misclassified as missing. |
| **CPU-only training** | Epoch budgets are limited by runtime. Deep results are bitwise reproducible only at a fixed thread count (`PHASE_B_THREADS=3`); a different thread count changed pooled PR-AUC by about 0.016 in a control run. |
| **Custom tabular SMOTE** | `imbalanced-learn` is not installed; the local SMOTE implementation is tested but may differ slightly from the reference library. |
| **Multiple comparisons** | Paired p-values are unadjusted; some “significant” findings are expected by chance. |
| **Phase B audit pending** | Phase B is committed but has not completed independent research review, multi-seed checks or external validation; its results are development cross-validation estimates only. |

---

## H. Testing and reproducibility

### H.1 Test suite

Run from the project root:

```bash
python -m pytest -q
```

**Verified result at the time of writing: `121 passed` (2 warnings).** This covers the production pipeline, Phase A and Phase B.

The suite includes:

- WFDB header parsing, target verification and raw signal layout;
- windowing, padding, gap interpolation, masks and train-only scaling;
- metric calculations and threshold selection;
- model forward shapes, checkpoint loading, parameter counts, the CNN-BiLSTM architecture, calibrators, model-card consistency and paired AUC comparison;
- CTG CSV/JSON validation and rejection paths;
- API contracts, thresholds, error codes, demo records, sample downloads and static hosting;
- Phase A quality audit, zero-padding, anti-aliased resampling, mask preservation, fractional-rate CTG features, fold-specific selectors and the expanded metric set;
- Phase B class-weight balancing, focal-loss gamma/alpha behaviour, weighted sampling, SMOTE shape and leakage guards, inner/outer fold isolation, threshold independence of ranking metrics, bootstrap reproducibility and controlled CTG augmentation.

The tests use temporary directories and do not modify the original dataset.

### H.2 Seeds and configurations

| Component | Seeding / configuration |
|---|---|
| Global project seed | `RANDOM_SEED = 42` in `src/config.py` |
| Production split | Stratified 70/15/15, seed 42, fixed record-level assignment |
| Phase A | Seed 42; `experiments/phase_a/config.json`; `experiments/phase_a/folds.csv` |
| Phase B | Seed 42; `experiments/phase_b/config.json`; `experiments/phase_b/folds.csv`; outer Phase A folds preserved |
| Deep determinism | Fixed CPU thread count, `PHASE_B_THREADS=3` (default); bitwise reproducibility only at the same thread count |
| Classical estimators | Deterministic configurations with `random_state=42` |

Deep results are **not** claimed to be bitwise reproducible across different hardware or thread counts. The Phase B artefacts were generated with `PHASE_B_THREADS=3`.

### H.3 Reproducing the research experiments

```bash
# Phase A: quality, window, sampling, feature-selection, comparison, deep screen, report
python -m src.phase_a.run --stage all

# Phase A without the CPU-expensive deep screen
python -m src.phase_a.run --stage all --no-dl

# Phase B: development bundle, B0, classical, deep, augmentation, comparison, report
# Windows PowerShell: use $env:PHASE_B_THREADS=3 instead of export
export PHASE_B_THREADS=3
python -m src.phase_b.run --stage prepare
python -m src.phase_b.run --stage b0 --force
python -m src.phase_b.run --stage classical --force
python -m src.phase_b.run --stage deep --force
python -m src.phase_b.run --stage augment --force
python -m src.phase_b.run --stage compare --force
python -m src.phase_b.run --stage report --force

python -m pytest -q
```

Runtime notes:

- Deep training is CPU-only. A full Phase B deep run is substantially slower than inference.
- The `compare` stage takes roughly 20 minutes because the paired bootstrap evaluates the full metric set on each resample.
- `--quick` provides a small smoke run. `--force` recomputes cached stages.
- `PHASE_B_THREADS` can cap CPU threads per process; `PHASE_B_THREADS=3` reproduces the reported local artefacts.

Experiment configurations, fold assignments, class weights, sampler settings, checkpoints, predictions, metrics, histories and logs are stored under `experiments/phase_a/` and `experiments/phase_b/`.

---

## I. Installation and execution

### I.1 Python environment

Python 3.12 is recommended. The pinned dependencies are in [`requirements.txt`](requirements.txt).

**Windows (PowerShell / cmd):**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project runs on CPU. Training is feasible on CPU but takes longer than inference.

### I.2 Repository layout

```text
project/
├── backend/
│   ├── main.py                 # FastAPI routes and static frontend hosting
│   ├── schemas.py              # Pydantic request/response models
│   └── services/
│       ├── ctg_parser.py       # CSV/JSON CTG validation and parsing
│       └── model_service.py    # inference, calibration, attributions, demos
├── src/
│   ├── config.py               # paths, features, target mapping, constants
│   ├── data/
│   │   ├── load_dataset.py     # strict WFDB header/signal parser
│   │   └── prepare_dataset.py  # processed arrays, features, splits
│   ├── preprocessing/
│   │   ├── preprocess.py       # windowing, masking, imputation, scaling
│   │   └── ctg_features.py     # 25 hand-crafted CTG descriptors
│   ├── models/
│   │   ├── deep_learning_model.py
│   │   ├── baseline.py
│   │   └── calibration.py      # Platt scaling (pickle-safe)
│   ├── training/train.py       # final training + 5-fold CV
│   ├── phase_a/                # additive Phase A experiments (completed)
│   │   ├── run.py              # staged CLI
│   │   ├── quality.py          # per-record CTG quality audit
│   │   ├── signals.py          # anti-aliased window extraction and masks
│   │   ├── features.py         # clinical + CTG feature assembly
│   │   ├── selection.py        # fold-specific feature selectors and stability
│   │   ├── experiment.py       # fixed folds, CV runners, metrics
│   │   ├── config.py
│   │   ├── plots.py
│   │   └── report.py
│   ├── phase_b/                # additive Phase B experiments (local, unpublished)
│   │   ├── run.py              # staged CLI
│   │   ├── data.py             # development-only loader and nested folds
│   │   ├── imbalance.py        # class weights, focal loss, samplers, SMOTE, augmentation
│   │   ├── metrics.py          # metrics, bootstrap CIs, paired differences
│   │   ├── training.py         # nested deep CV with inner selection
│   │   ├── classical.py        # nested classical CV
│   │   ├── config.py
│   │   ├── plots.py
│   │   └── report.py
│   └── evaluation/
│       ├── evaluate.py         # held-out metrics, CIs, figures, report
│       ├── metrics.py
│       ├── cv_report.py
│       └── make_model_card.py
├── frontend/                   # React/Vite mobile-first app
├── tests/                      # pytest suite (121 tests)
├── data/processed/             # generated; safe to rebuild
├── models/                     # checkpoints, calibrators, model card
├── reports/                    # metrics, figures, reports
├── experiments/phase_a/        # Phase A outputs, comparison tables, figures, logs
├── experiments/phase_b/        # Phase B outputs and imbalance report
├── samples/                    # sample and synthetic CTG uploads
├── docs/hospital_integration.md
├── requirements.txt
├── .env.example
└── ctu-chb-intrapartum-cardiotocography-database-1.0.0/   # original data (read-only)
```

The original dataset directory is treated as read-only. Generated files go to `data/processed/`, `models/`, `reports/` and `experiments/`.

### I.3 Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### I.4 Start the API and app

From the project root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:

- app: <http://127.0.0.1:8000>
- interactive API docs: <http://127.0.0.1:8000/docs>
- health check: <http://127.0.0.1:8000/health>

FastAPI serves `frontend/dist` when it exists. If you have not built the frontend, `/` returns a small JSON status message and the API still works.

The Dashboard lists the **Multimodal CNN + clinical MLP** and the **CTG-only CNN-BiLSTM** as two separate model cards with their own held-out metrics and “Use this model” links. The Assessment page exposes them as separate top-level choices (`/assess?model=multimodal` or `/assess?model=bilstm`), alongside the clinical-only model. The CTG-only 1D CNN remains available through the API (`model=signal`). Held-out demo cases are scored with whichever CTG model is selected.

### I.5 Frontend development mode

Run the backend on port 8000, then:

```bash
cd frontend
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/health`, `/predict`, `/predict-ctg`, `/demo`, `/demo-cases`, `/metrics`, `/model-info` and `/api` to the backend. Set `VITE_API_BASE` in `frontend/.env.local` (template: `frontend/.env.example`) only if you need a different API origin.

### I.6 Reproducing the production pipeline

The pipeline is deterministic with seed 42. Run the commands **from the project root**, after installing the requirements and placing the dataset in the expected directory.

```bash
# 1. Parse headers, build processed arrays and fixed splits
python -m src.data.prepare_dataset

# 2. Fit and save classical baselines
python -m src.models.baseline

# 3. Train the four deep models on the fixed split
python -m src.training.train --variant multimodal
python -m src.training.train --variant signal
python -m src.training.train --variant bilstm
python -m src.training.train --variant clinical

# 4. Evaluate on the held-out test set, calibrate, and write figures/report
python -m src.evaluation.evaluate

# 5. Stratified 5-fold cross-validation for each deep variant
python -m src.training.train --variant multimodal --cv
python -m src.training.train --variant signal --cv
python -m src.training.train --variant bilstm --cv
python -m src.training.train --variant clinical --cv

# 6. Aggregate CV results (also cross-validates the baselines)
python -m src.evaluation.cv_report

# 7. Rebuild the machine-readable model card
python -m src.evaluation.make_model_card
```

Notes:

- Step 1 writes `data/processed/*` and `models/preprocessing.joblib`; it never writes to the raw dataset.
- Step 4 writes `reports/metrics.json`, `reports/model_evaluation_report.md`, `reports/test_predictions.csv` and figures.
- Step 7 must run after step 4 (and after CV if you want CV fields in the card).
- Training uses validation loss for early stopping; the test split is never used for model selection.
- A full CPU run may take several minutes per deep model. The reported checkpoints are already included.

### I.7 Phase A commands

```bash
# Full Phase A pipeline
python -m src.phase_a.run --stage all

# Same without the CPU-expensive deep-learning screening
python -m src.phase_a.run --stage all --no-dl

# Individual stages (all deterministic with seed 42)
python -m src.phase_a.run --stage quality --force
python -m src.phase_a.run --stage window
python -m src.phase_a.run --stage sampling
python -m src.phase_a.run --stage features
python -m src.phase_a.run --stage compare
python -m src.phase_a.run --stage report

# Fast smoke run
python -m src.phase_a.run --stage all --quick --force
```

Phase A writes the per-record quality report, exclusion lists, fixed fold assignments, per-fold selected features, out-of-fold predictions, confidence intervals, comparison tables and figures. The generated report is written to both the repository root and `experiments/phase_a/reports/PHASE_A_DATA_HANDLING_REPORT.md`.

### I.8 Phase B commands

```bash
# Fix Torch to 3 CPU threads so deep runs are reproducible at that thread count
export PHASE_B_THREADS=3        # PowerShell: $env:PHASE_B_THREADS=3

# Build the development-only bundle, fixed folds and dataset manifest
python -m src.phase_b.run --stage prepare

# Stage B0: reproduce the six baselines on the fixed folds
python -m src.phase_b.run --stage b0 --force

# Stage B1: nested classical and deep imbalance experiments
python -m src.phase_b.run --stage classical --force
python -m src.phase_b.run --stage deep --force

# Exploratory controlled CTG augmentation (multimodal only)
python -m src.phase_b.run --stage augment --force

# Assemble comparisons, figures, paired bootstrap differences and the report
python -m src.phase_b.run --stage compare --force
python -m src.phase_b.run --stage report --force

# Fast smoke versions
python -m src.phase_b.run --stage classical --quick --force
python -m src.phase_b.run --stage deep --models clinical --quick --force
```

### I.9 Command-line options

```bash
python -m src.training.train --help
python -m src.phase_a.run --help
python -m src.phase_b.run --help
```

Production training options include `--variant {multimodal,signal,bilstm,clinical}`, `--cv`, `--epochs`, `--batch-size`, `--lr`, `--patience` and `--seed`. Phase B also accepts `--models`, `--strategies`, `--classical-models`, `--classical-strategies`, `--augment-modes`, `--epochs-outer`, `--force`, `--quick`, `--no-deep` and `--no-classical`.

### I.10 API reference

Base URL: `http://127.0.0.1:8000`. Interactive schema: `/docs`.

#### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

#### `GET /model-info`

Returns the model card: dataset, architecture, evaluation, limitations, intended and prohibited use, disclaimer.

#### `GET /metrics`

Returns the canonical held-out production metrics and interpretation text.

#### `POST /predict` — clinical only

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 31, "gravidity": 2, "parity": 1, "gest_weeks": 39.5,
    "diabetes": false, "hypertension": false, "preeclampsia": false,
    "liquor_praecox": false, "pyrexia": false, "meconium": false,
    "induced": false, "presentation": 1
  }'
```

#### `POST /predict-ctg` — multimodal, CNN-BiLSTM or 1D CNN

Multipart form with a `file` (CSV with `fhr`, `uc`, optional `time`; or JSON `{"fs": 4, "fhr": [...], "uc": [...]}`), the same clinical fields, and an optional `model` field: `multimodal` (default), `bilstm` (CTG-only CNN-BiLSTM), or `signal` (CTG-only 1D CNN).

```bash
curl -X POST http://127.0.0.1:8000/predict-ctg \
  -F "file=@samples/sample_ctg.csv" \
  -F "age=31" -F "gravidity=2" -F "parity=1" -F "gest_weeks=39.5" \
  -F "diabetes=false" -F "hypertension=false" -F "preeclampsia=false" \
  -F "liquor_praecox=false" -F "pyrexia=false" -F "meconium=false" \
  -F "induced=false" -F "presentation=1" -F "fs=4" -F "model=bilstm"
```

`bilstm` and `signal` ignore the clinical vector; the fields are still validated so the form contract stays uniform.

Upload constraints: CSV/JSON only, maximum 5 MB, at least 10 minutes of recording, no NaN/infinite values, and a non-zero UC channel. Recordings shorter than 30 minutes are left-padded and flagged. The model window is always the most recent 30 minutes.

#### Demo endpoints

```bash
curl http://127.0.0.1:8000/demo-cases
curl -X POST "http://127.0.0.1:8000/demo/2005?model=bilstm"
```

Demo records come from the held-out public test split: 14 cases, 7 vaginal and 7 cesarean. The assessment page shows 6 at a time with a control to reveal the rest. True labels are shown only for teaching and are never model inputs.

#### Response fields

Every prediction includes `prediction`, `predicted_class`, `cesarean_probability`, `vaginal_probability`, `decision_threshold`, `model_variant`, `calibrated`, clinical and signal attributions, warnings and a disclaimer. The CTG endpoint additionally returns `ctg_summary`.

For a deployment-oriented discussion (FHIR/HL7, security, governance), see [`docs/hospital_integration.md`](docs/hospital_integration.md). It is a design discussion, not an implemented integration.

### I.11 Troubleshooting

**`ModuleNotFoundError: backend`** — run commands from the project root, not from `backend/` or `frontend/`.

**`Model artifacts could not be loaded`** — ensure `models/*.pt`, `models/*.joblib`, `models/model_card.json`, `reports/metrics.json` and `data/processed/*.npz` exist. Re-run the pipeline if they do not.

**Port 8000 already in use** — stop the old server or run `uvicorn backend.main:app --port 8001` and set `VITE_API_BASE=http://127.0.0.1:8001` when developing the frontend.

**Frontend shows JSON instead of the app** — build it with `npm run build` in `frontend/`, then reload `/`.

**Torch installation is slow or large** — install the CPU-only wheel for your platform from <https://pytorch.org/get-started/locally/>, then install the remaining requirements.

**Different Python version** — Python 3.12 is recommended. The pinned versions may not have wheels for older interpreters.

---

## J. Future research roadmap

| Phase | Status | Planned scope |
|---|---|---|
| **Phase A** — data handling and preprocessing | **Completed** | Quality audit, window/sampling/feature-selection experiments, fixed 30 min / 1 Hz / no-selection configuration |
| **Phase B, B0–B1** — baseline reproduction and imbalance learning | **Implemented; audit pending** | B0 reproduction, five deep strategies, four classical strategies, exploratory augmentation, nested CV, paired comparisons |
| **Phase B final audit** | Planned | Multi-seed variance study, calibration and threshold review, multiplicity handling, independent code/results review, commit and publication decision |
| **B2 — architecture optimisation** | Planned | Receptive-field and sequence-model tuning, regularisation and capacity controls, principled hyperparameter search within CPU limits |
| **B3 — multimodal fusion optimisation** | Planned | Fusion strategy, missingness-aware clinical encoding, modality dropout, calibration of the fused model |
| **Phase C — independent cross-dataset validation** | Planned | External dataset identification, harmonisation of CTG and maternal variables, external evaluation without refitting preprocessing on the target cohort |
| **Phase D — application integration and clinical expert feedback** | Planned | Clinical review of presentation and thresholds, usability study, governance and deployment requirements |

External dataset access and the availability of delivery-mode labels in candidate external cohorts are **still being investigated**; no external validation has been completed. Timelines are not promised, and no planned phase should be described as finished until its artefacts and review are complete.

---

## Additional documentation

| Document | Contents |
|---|---|
| [`reports/dataset_inspection_report.md`](reports/dataset_inspection_report.md) | File inventory, target verification, leakage audit, missingness, signal quality, splits |
| [`reports/technical_report.md`](reports/technical_report.md) | Methods, architectures, evaluation protocol, full production results, limitations, references |
| [`reports/model_evaluation_report.md`](reports/model_evaluation_report.md) | Generated held-out production metric tables |
| [`reports/cv_summary.md`](reports/cv_summary.md) | Generated 5-fold cross-validation summary |
| [`models/MODEL_CARD.md`](models/MODEL_CARD.md) | Concise model card for reviewers |
| [`models/model_card.json`](models/model_card.json) | Machine-readable card served by `/model-info` |
| [`experiments/phase_a/reports/PHASE_A_DATA_HANDLING_REPORT.md`](experiments/phase_a/reports/PHASE_A_DATA_HANDLING_REPORT.md) | Phase A methodology, results, decision, limitations and artefact index |
| [`experiments/phase_b/reports/PHASE_B_IMBALANCE_REPORT.md`](experiments/phase_b/reports/PHASE_B_IMBALANCE_REPORT.md) | Phase B methodology, per-fold tables, confidence intervals, calibration, limitations and recommendation |
| [`docs/teacher_presentation.md`](docs/teacher_presentation.md) | 8-slide outline, 5–6 minute spoken script, one-page handout and viva Q&A |
| [`docs/hospital_integration.md`](docs/hospital_integration.md) | Hypothetical integration, security and governance considerations |

The Phase B report is included in this repository. It documents development-set cross-validation experiments whose research audit is still pending; it is not an external-validation or clinical-validation report.

---

## Licence and data use

The original CTU-UHB database is distributed by PhysioNet under its own terms. Consult the PhysioNet page and the database documentation before redistributing data. The code in this repository is an academic project and is provided for research and teaching; no clinical warranty is given.

If you use the dataset, cite:

> Chudáček V, Spilka J, Burša M, et al. Open access intrapartum CTG database. *BMC Pregnancy and Childbirth*. 2014;14:16. doi:10.1186/1471-2393-14-16.

---

## Final reminder

**This is a research prototype.** A high score in this repository would not mean the system is safe or effective. The dataset is small, selected, single-centre, and lacks external validation. The only responsible use of this code is teaching, methodological study, and further research under appropriate ethical and regulatory oversight.
