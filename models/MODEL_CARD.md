# Model Card — OB-SIGHT Delivery-Mode Research Prototype

**Version:** 1.1.0  
**Date:** 6 November 2026  
**Card status:** research prototype, not clinically validated  
**Machine-readable version:** [`model_card.json`](model_card.json) (served by `GET /model-info`)

---

## 1. Model details

| Item | Description |
|---|---|
| Project | AI-Based Mobile Clinical Decision Support System for Delivery Outcome Prediction |
| Task | Binary classification of delivery mode |
| Positive class | Cesarean delivery (`Deliv. type = 2`) |
| Negative class | Vaginal delivery (`Deliv. type = 1`) |
| Prediction time | During labour, using the most recent 30 minutes of CTG |
| Primary model | Multimodal 1D CNN + clinical MLP (64,121 parameters) |
| Comparison models | CTG-only 1D CNN (58,009 parameters); CTG-only CNN-BiLSTM (31,009 parameters); clinical-only MLP (3,041 parameters) |
| Signal input | 4 × 1800 tensor: FHR, UC, FHR-valid mask, UC-valid mask at 1 Hz |
| Clinical input | 15 features: age, gravidity, parity, gestational age, 7 binary risk factors, presentation one-hot (3), presentation-missing indicator |
| Output | Calibrated probability of cesarean delivery and a binary decision at a stated threshold |
| Training data | 386 records from the public CTU-CHB database (32 cesarean) |
| Calibration | Platt scaling fitted on the validation split (83 records, 7 cesarean) |
| Threshold | Youden's J on the validation split; applied unchanged to the test split |
| Framework | PyTorch 2.8.0 (CPU), scikit-learn 1.6.0 |

The signal encoder uses four 1D convolution blocks (24–48–64–96 channels) with batch normalisation and pooling, concatenated average/max pooling, and a 96-unit embedding. The clinical encoder is a 48 → 32 MLP. The multimodal head fuses both embeddings through a 64-unit layer. The CNN-BiLSTM comparison model uses three convolution blocks (16–32–48 channels), two pooling stages, a 225-step sequence, a one-layer bidirectional LSTM with 32 hidden units per direction, dropout, and a 32-unit fully connected head. Training uses class-weighted binary cross-entropy, AdamW, dropout, mild Gaussian input noise, gradient clipping, and early stopping on validation loss.

---

## 2. Intended use

**Intended**

- Teaching and demonstration of an end-to-end clinical machine-learning workflow.
- Research on multimodal intrapartum CTG and maternal-data modelling.
- Reproducible benchmarking against classical baselines on a public research dataset.

**Prohibited**

- Any real clinical decision, triage, counselling, or mode-of-delivery decision.
- Use as a medical device, or as input to a device, without regulatory approval and prospective clinical validation.
- Use on patient data without a lawful basis, ethics approval, and data-protection safeguards.
- Treating demonstration records as real patient cases.

**Primary intended users:** students, researchers, and developers studying clinical prediction methods. **Not** clinicians making decisions, patients, or hospital staff.

---

## 3. Training data

| Item | Value |
|---|---|
| Dataset | CTU-CHB Intrapartum Cardiotocography Database, v1.0.0 |
| Source | PhysioNet / Czech Technical University in Prague, University Hospital in Brno |
| DOI | 10.13026/0d3m-5v02 |
| Records | 552 (506 vaginal, 46 cesarean) |
| Prevalence | 8.33% cesarean |
| Splits | Train 386 (32 CS), validation 83 (7 CS), test 83 (7 CS) |
| Split method | Stratified, record-level, fixed seed 42 |
| Unit | One intrapartum recording per delivery |

The 552 records are a selected subset of a larger single-centre database (documented as 9164 recordings). The 46 cesarean records were released as a separate subset and differ in recording duration and signal quality. No external data were used.

**Excluded variables:** all post-delivery outcomes (cord blood gases, Apgar, NICU, neonatal morbidity, birth weight, sex) and labour-progress/decision/recording-selection fields (`I.stage`, `II.stage`, `NoProgress`, `CK/KP`, `Pos. II.st.`, `Sig2Birth`, `Rec. type`). This prevents direct leakage of the target or the delivery decision.

---

## 4. Evaluation

### 4.1 Held-out test set

83 records: 7 cesarean, 76 vaginal, prevalence 8.4%. The test set was never used for training, early stopping, calibration, or threshold selection.

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier |
|---|---|---|---|
| Multimodal CNN + clinical | 0.733 (0.551–0.910) | 0.402 (0.129–0.757) | 0.068 |
| CTG-only 1D CNN | 0.797 (0.660–0.914) | 0.305 (0.156–0.672) | 0.069 |
| CTG-only CNN-BiLSTM | 0.720 (0.453–0.938) | 0.450 (0.161–0.808) | 0.066 |
| Clinical-only MLP | 0.498 (0.244–0.727) | 0.105 (0.071–0.272) | 0.078 |
| CTG logistic regression | 0.701 (0.451–0.904) | 0.204 (0.115–0.451) | 0.180 |
| Combined gradient boosting | 0.675 (0.425–0.882) | 0.194 (0.105–0.467) | 0.100 |

### 4.2 Operating points

| Model | Threshold | Sensitivity | Specificity | F1 | Accuracy | Confusion (TN, FP / FN, TP) |
|---|---:|---:|---:|---:|---:|---|
| Multimodal | 0.14 | 0.429 | 0.842 | 0.273 | 0.807 | 64, 12 / 4, 3 |
| CTG-only 1D CNN | 0.08 | 0.571 | 0.684 | 0.229 | 0.675 | 52, 24 / 3, 4 |
| CTG-only CNN-BiLSTM | 0.08 | 0.571 | 0.750 | 0.267 | 0.735 | 57, 19 / 3, 4 |
| Clinical-only | 0.11 | 0.143 | 0.829 | 0.095 | 0.771 | 63, 13 / 6, 1 |

At the conventional 0.5 threshold the multimodal model has sensitivity 0.286, specificity 0.987, and F1 0.400; the CTG-only model has sensitivity 0.000, specificity 1.000, and F1 0.000. A majority-class classifier also has 91.6% accuracy and zero sensitivity at 0.5, which is why accuracy is not a meaningful headline metric here.

### 4.3 Cross-validation

Stratified 5-fold cross-validation on all 552 records (pooled out-of-fold predictions):

| Model | Pooled ROC-AUC | Fold mean AUC ± SD | Pooled PR-AUC |
|---|---:|---:|---:|
| Multimodal | 0.767 | 0.774 ± 0.053 | 0.368 |
| CTG-only 1D CNN | 0.763 | 0.770 ± 0.039 | 0.327 |
| CTG-only CNN-BiLSTM | 0.810 | 0.800 ± 0.096 | 0.399 |
| Clinical-only | 0.561 | 0.552 ± 0.091 | 0.100 |
| CTG logistic regression | 0.744 | 0.744 ± 0.083 | 0.251 |
| Combined logistic regression | 0.726 | 0.725 ± 0.065 | 0.305 |
| Combined gradient boosting | 0.702 | 0.704 ± 0.071 | 0.217 |
| Clinical logistic regression | 0.577 | 0.586 ± 0.053 | 0.112 |

### 4.4 Interpretation

- CTG-based models show a modest, unstable association with delivery mode; the clinical-only model is at chance on the test set.
- The CNN-BiLSTM had the highest test PR-AUC (0.450) but a lower ROC-AUC than the 1D CNN (0.720 versus 0.797). The paired AUC difference was −0.077 (95% CI −0.331 to +0.167), so no statistically meaningful improvement is established.
- Cross-validation ranked the CNN-BiLSTM highest (pooled AUC 0.810, PR-AUC 0.399; fold mean 0.800 ± 0.096 versus 0.763 for the 1D CNN), so the test-set ranking should be treated as unstable rather than decisive.
- Deep learning does not clearly outperform a CTG-descriptor logistic regression.
- Confidence intervals are wide and overlap; no ranking between CTG models is statistically established.
- Low validation-selected thresholds produce false-positive rates that are not clinically acceptable.
- Calibration and thresholds are fitted to this cohort and cannot be assumed to transfer.
- The CNN-BiLSTM overfitted quickly: training loss fell to 0.054 by epoch 34 while validation loss bottomed at 0.334 at epoch 19 and then rose. The epoch-19 checkpoint was retained; early stopping ended at epoch 34.

---

## 5. Quantitative and demographic analysis

No subgroup performance analysis (age, parity, presentation, ethnicity, site) is reported. With only 46 cesarean records in total, subgroup estimates would be too imprecise to support claims. The dataset does not provide ethnicity or a stable patient identifier, and it comes from one hospital.

No fairness or bias mitigation was applied. The model may encode acquisition differences between the cesarean and vaginal subsets, including recording duration and missingness. This is a substantive unresolved confound.

---

## 6. Ethical and safety considerations

- **Not a diagnosis.** The output is a statistical association with the eventual mode of delivery, not a judgement about fetal status or the necessity of cesarean.
- **No clinical utility established.** Sensitivity and specificity on 7 test events do not demonstrate benefit, safety, or net clinical value.
- **Risk of automation bias.** A confident-looking probability could anchor clinicians even when wrong. The interface therefore displays warnings, the threshold, and a prominent prototype disclaimer.
- **Confidentiality.** Only public, de-identified research data are used in the demonstration. Any real patient use would require a lawful basis, ethics approval, access controls, audit trails, and data-protection review.
- **Explainability is not justification.** Gradient attributions show which inputs influenced the model; they do not establish causation or clinical validity.
- **Threshold instability.** The API uses low validation-selected thresholds. A score near the threshold is explicitly flagged as unstable.
- **No post-market surveillance.** There is no monitoring, drift detection, incident reporting, or regulatory quality system.

---

## 7. Caveats and recommendations

1. Do not use this model, its scores, or its explanations for patient care.
2. Do not interpret a high probability as a recommendation for cesarean delivery.
3. Do not compare this model's accuracy with clinical judgement; no such study was performed.
4. Treat all reported metrics as exploratory and imprecise.
5. Before any future clinical study: obtain external/multi-centre data, pre-register the analysis, validate calibration and subgroup performance, evaluate decision curves and net benefit, and obtain ethics and regulatory approval.
6. Re-train and re-calibrate locally if the model is ever used for research on another cohort; do not transfer these thresholds.
7. Retain the original dataset's terms of use and citation requirements.

---

## 8. Citation

If you use this code or the underlying data, cite:

> Chudáček V, Spilka J, Burša M, et al. Open access intrapartum CTG database. *BMC Pregnancy and Childbirth*. 2014;14:16. doi:10.1186/1471-2393-14-16.

Database: CTU-CHB Intrapartum Cardiotocography Database, version 1.0.0. PhysioNet. doi:10.13026/0d3m-5v02.

---

## 9. Contact and version control

This is an academic project. The model version is `1.0.0`; the card was generated from `reports/metrics.json`, `models/preprocessing_meta.json`, and training summaries by `src/evaluation/make_model_card.py`. Any change to data, preprocessing, training, calibration, or thresholds requires a new version and a new evaluation; do not edit the metrics by hand.
