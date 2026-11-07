# Teacher Presentation Pack — OB-SIGHT Delivery-Mode Prototype

**Project:** AI-Based Mobile Clinical Decision Support System for Delivery Outcome Prediction Using Cardiotocography and Maternal Clinical Data
**Prepared:** 6 November 2026
**Use:** 8-slide talk (about 5–6 minutes), spoken script, and a one-page handout

> **Safety statement (repeat this verbally and keep it on screen).**
> This is a research prototype. It is not a medical device, has not been externally or prospectively validated, and must not be used for clinical decisions, triage, counselling, or mode-of-delivery planning. The output is a statistical score from a small public dataset — not a diagnosis and not a substitute for obstetric judgement.

---

## 1. The 30-second version

The project asks a narrow, testable question: **does the last 30 minutes of intrapartum CTG, combined with routine maternal clinical data, contain information about the eventual mode of delivery — cesarean versus vaginal?**

I built the full pipeline: a strict parser for the public CTU-CHB database, leakage-aware preprocessing, four compact PyTorch models plus four classical baselines, held-out evaluation with confidence intervals and five-fold cross-validation, calibration, explainability, a FastAPI backend, and a mobile-first React interface.

The honest answer is **a weak, imprecise signal, not a usable clinical tool**. CTG-based models reach a test ROC-AUC of about 0.72–0.80 and cross-validated ROC-AUC of about 0.76–0.81, but the test set contains only 7 cesareans, the confidence intervals are wide and overlapping, a simple CTG logistic regression is competitive with the deep models, and the clinical-only model is at chance.

---

## 2. Slide outline (8 slides)

### Slide 1 — Title and safety statement
- **Title:** OB-SIGHT — Predicting Delivery Mode from CTG and Maternal Data
- **Subtitle:** A research prototype, not a clinical decision tool
- **On screen:** student name, course, date, and the one-line disclaimer
- **Visual:** app dashboard screenshot or the project pipeline diagram
- **Speaker goal:** state the task and the safety boundary in the first 30 seconds

### Slide 2 — The question, and what it is not
- **Question:** cesarean (class 1) vs vaginal (class 0) delivery, using the most recent 30 minutes of CTG during labour
- **Not the question:** fetal distress, fetal status, or a recommendation for or against cesarean
- **Why:** version 1.0.0 of the database has **no expert CTG annotations** — the `ANNOTATIONS` file is empty and there are no `.atr` files
- **Visual:** a two-box diagram: “what the model outputs” vs “what it must never be read as”

### Slide 3 — Data
- **Database:** CTU-CHB Intrapartum Cardiotocography Database v1.0.0 (PhysioNet, DOI `10.13026/0d3m-5v02`)
- **Cohort:** 552 records — **506 vaginal, 46 cesarean**, prevalence **8.33%**
- **Setting:** single Czech university hospital (Brno); the 552 records are a selected subset of a larger 9,164-recording database
- **Target:** `Deliv. type`, verified from every header — 1 = vaginal, 2 = cesarean; all 46 cesareans are in the 2001–2046 series
- **Inputs:** 4 × 1800 CTG tensor (FHR, UC, FHR mask, UC mask at 1 Hz) + 15 maternal features
- **Visual:** class distribution figure and one example CTG tracing

### Slide 4 — Preprocessing and leakage control
- **Signal path:** trim zero-padding → last 30 minutes → screen FHR 30–240 bpm → interpolate short gaps → 1 Hz masked block mean → training-set standardisation
- **Clinical path:** median imputation, standardisation, presentation one-hot plus missing indicator — all fitted on training data only
- **Excluded to prevent leakage:** post-delivery outcomes (cord gases, Apgar, NICU, birth weight, sex) and labour-progress/decision fields (`I.stage`, `II.stage`, `NoProgress`, `CK/KP`, `Pos. II.st.`, `Sig2Birth`, `Rec. type`)
- **Splits:** 386 train / 83 validation / 83 test, stratified, seed 42
- **Visual:** preprocessing pipeline with the read-only raw data box on the left

### Slide 5 — Models
- **Primary:** multimodal 1D CNN + clinical MLP — four convolution blocks (24–48–64–96 channels), 96-unit signal embedding, 48 → 32 clinical MLP, 64-unit fusion head, **64,121 parameters**
- **Ablations:** CTG-only 1D CNN (58,009 parameters), CTG-only CNN-BiLSTM (31,009 parameters) and clinical-only MLP (3,041 parameters)
- **Baselines:** clinical logistic regression, CTG-descriptor logistic regression, combined logistic regression, combined gradient boosting
- **Training:** class-weighted binary cross-entropy, AdamW, dropout, Gaussian input noise, gradient clipping, early stopping on validation loss
- **Visual:** architecture diagram with parameter counts

### Slide 6 — Evaluation protocol
- **Primary test:** fixed 83-record held-out set (7 cesarean, 76 vaginal), never used for early stopping, calibration, or threshold selection
- **Calibration:** Platt scaling fitted on validation only
- **Threshold:** Youden's J on validation only, applied unchanged to test; the 0.5 operating point is also reported
- **Uncertainty:** stratified bootstrap 95% confidence intervals (1,000 resamples)
- **Paired comparison:** paired stratified bootstrap (2,000 resamples) of the CNN-BiLSTM minus 1D CNN AUC difference
- **Secondary:** stratified 5-fold cross-validation over all 552 records
- **Visual:** split diagram and a one-line explanation of why the test set stayed untouched

### Slide 7 — Results
- **Test ROC-AUC:** multimodal **0.733** (95% CI 0.551–0.910); CTG-only 1D CNN **0.797** (0.660–0.914); CTG-only CNN-BiLSTM **0.720** (0.453–0.938); clinical-only **0.498** (0.244–0.727)
- **Test PR-AUC:** 0.402 / 0.305 / 0.450 / 0.105 against an 8.4% prevalence baseline
- **Five-fold pooled ROC-AUC:** 0.767 / 0.763 / 0.810 / 0.561
- **CV versus test:** cross-validation ranks the CNN-BiLSTM highest, but the held-out test ranks the 1D CNN higher; the ranking is not stable with this cohort
- **Operating point:** at validation thresholds the multimodal model detects **3 of 7 cesareans with 12 of 76 false positives**; the 1D CNN detects 4 of 7 but flags 24 of 76; the CNN-BiLSTM detects 4 of 7 and flags 19 of 76
- **Accuracy trap:** a majority-class classifier reaches **91.6% accuracy with zero sensitivity**
- **Honest comparison:** the CTG logistic regression reaches test ROC-AUC 0.701 and cross-validated 0.744 — the deep models do not clearly beat it. The CNN-BiLSTM minus 1D CNN AUC difference is −0.077 (95% CI −0.331 to +0.167), so no meaningful improvement is established
- **Visual:** ROC/PR curves and confusion matrices

### Slide 8 — Limitations, safety, and next steps
- **Limitations:** 7 test events; selected single-centre cohort; cesarean recordings released as a separate subset with different duration and missingness; no expert CTG labels; no external or prospective validation; clinical-only model at chance; the CNN-BiLSTM overfits after about 19 epochs
- **Safety:** no clinical use, no automation of mode-of-delivery decisions, attributions are model behaviour not causation
- **Next steps:** external/multi-centre validation, expert CTG annotations, explicit modelling of recording quality, pre-registered analysis, decision-curve and subgroup evaluation, clinician comparison study
- **Take-home:** a complete, reproducible, leakage-aware pipeline — and a result that is reported honestly rather than inflated

---

## 3. Spoken script (about 5–6 minutes)

> Adjust names and course details. Timings assume a steady pace of about 150 words per minute. If you have only about four minutes, keep Slides 1, 3, 5, 7, and 8 and drop the detail on Slides 4 and 6.

**[Slide 1 — 0:00–0:30]**
Good morning. My project is OB-SIGHT, an AI-based mobile clinical decision support prototype for predicting delivery outcome from cardiotocography and maternal clinical data. I want to say one thing before anything else: this is a research prototype, not a medical device, and it must not be used for clinical decisions. The rest of the talk explains what it does, how it was built, and why the results should be read cautiously.

**[Slide 2 — 0:30–1:00]**
The task is binary classification of the eventual mode of delivery: cesarean versus vaginal, using the most recent 30 minutes of CTG during labour. It is deliberately **not** a fetal-distress detector. Version 1.0.0 of the database has no expert CTG annotations — the annotation file is empty and there are no annotation records — so training a supervised fetal-status model would not be scientifically valid. The model learns an association with delivery mode, and the interface says so explicitly.

**[Slide 3 — 1:00–1:45]**
The data come from the public CTU-CHB Intrapartum Cardiotocography Database, version 1.0.0, from PhysioNet. There are 552 recordings: 506 vaginal and 46 cesarean, so the prevalence is 8.33%. This is a selected subset of a much larger single-centre database from Brno, and the 46 cesarean recordings were released as a separate subset. I verified the target from every header: `Deliv. type` 1 means vaginal and 2 means cesarean, and all 46 cesareans are in the 2001 to 2046 record series. The inputs are the last 30 minutes of FHR and uterine contractions, plus 15 maternal features such as age, parity, gestational age, hypertension, meconium, and presentation.

**[Slide 4 — 1:45–2:30]**
Preprocessing was designed around leakage control. The raw signals are 4 hertz with zero-coded missing samples, so I trim the padding, take the last 30 minutes, screen FHR to 30–240 beats per minute, interpolate only short gaps, and reduce to 1 hertz using a masked block mean. That gives a four-channel tensor: FHR, uterine contractions, and a validity mask for each. The masks let the model distinguish a real zero from missing data. All scaling statistics are fitted on the training split only. I also excluded post-delivery outcomes and labour-progress fields that would leak the target. The split is fixed at 386 training, 83 validation, and 83 test records, stratified with seed 42.

**[Slide 5 — 2:30–3:05]**
The primary model is a multimodal network: a small one-dimensional CNN for the CTG, a clinical MLP, and a late-fusion head. It has about 64,000 parameters. I deliberately kept it small because there are only 46 cesarean examples in the whole dataset; a larger network would memorise them. I also trained three ablations — CTG-only 1D CNN, CTG-only CNN-BiLSTM, and clinical-only MLP — plus four classical baselines: clinical logistic regression, CTG-descriptor logistic regression, combined logistic regression, and gradient boosting. The CNN-BiLSTM is the temporal test: it sees the same four-channel sequence, but a bidirectional LSTM can use context from both earlier and later in the window. Training uses class-weighted cross-entropy, AdamW, dropout, mild input noise, gradient clipping, and early stopping on validation loss.

**[Slide 6 — 3:05–3:40]**
Evaluation is where a lot of projects accidentally overstate their results, so the protocol was fixed in advance. The test set was never used for early stopping, calibration, or threshold selection. Platt calibration and the Youden-J threshold were fitted on the validation split only and then applied unchanged to test. I report both the conventional 0.5 threshold and the validation-selected one. Uncertainty is estimated with stratified bootstrap confidence intervals, and I also ran five-fold cross-validation over all 552 records as a secondary check.

**[Slide 7 — 3:40–4:55]**
The headline is a weak signal, not a strong one. On the held-out test set, the multimodal model has ROC-AUC 0.733, with a 95% interval from 0.551 to 0.910. The CTG-only 1D CNN is higher at 0.797, the CNN-BiLSTM is 0.720, and the clinical-only model is 0.498 — chance. PR-AUC is 0.402, 0.305, 0.450, and 0.105, against a prevalence baseline of 0.084. Cross-validation tells a more favourable story for the recurrent model: pooled ROC-AUC is 0.767 for multimodal, 0.763 for the 1D CNN, 0.810 for the CNN-BiLSTM, and 0.561 for clinical-only. But the CNN-BiLSTM folds vary substantially, and the held-out test ranking reverses, so I do not claim a stable advantage.

The operating point matters. At the validation-selected threshold, the multimodal model detects 3 of 7 cesareans but incorrectly flags 12 of 76 vaginal deliveries. The 1D CNN detects 4 of 7 but flags 24 of 76; the CNN-BiLSTM detects 4 of 7 and flags 19 of 76. And here is the key caution: a classifier that always says “vaginal” gets 91.6% accuracy while detecting no cesareans at all. That is why I lead with ROC-AUC, PR-AUC, sensitivity, and specificity rather than accuracy. The CTG logistic regression also reaches 0.701 on test and 0.744 in cross-validation, so I cannot claim the deep models are clearly better. The CNN-BiLSTM did have the highest PR-AUC, but its AUC difference versus the 1D CNN was −0.077 with a confidence interval from −0.331 to +0.167 — it includes zero, so there is no statistically meaningful improvement.

**[Slide 8 — 4:55–5:45]**
So what does this mean? The pipeline is complete and reproducible, and the evaluation is honest, but the clinical claim is not there. There are only 7 cesareans in the test set, the cohort is small, selected, and single-centre, and the cesarean recordings differ in duration and signal quality — which the missingness masks could partly pick up as a shortcut. There are no expert CTG labels and no external validation. The clinical-only model is at chance, which suggests the available maternal fields are not enough. The CNN-BiLSTM also overfits quickly: its validation loss bottomed at epoch 19 and then rose while training loss kept falling.

The responsible conclusion is that this is a teaching and research prototype. Before any clinical claim, I would need external or multi-centre data, expert annotations, explicit modelling of recording quality, a pre-registered analysis, and a prospective comparison with clinician judgement. Thank you — I’m happy to take questions.

---

## 4. One-page handout

> Print at A4, single side. Keep the disclaimer at the top.

### OB-SIGHT — Delivery-Mode Prediction from CTG and Maternal Data
**Research prototype — not a medical device and not for clinical use.**

**Question.** Does the last 30 minutes of intrapartum CTG plus routine maternal data contain information about the eventual mode of delivery — cesarean versus vaginal?

**What it is not.** Not a fetal-distress detector and not a treatment recommendation. Database v1.0.0 has no expert CTG annotations, so fetal-status classification is not a valid target.

| Data | Value |
|---|---|
| Database | CTU-CHB Intrapartum CTG Database v1.0.0 (PhysioNet, DOI 10.13026/0d3m-5v02) |
| Cohort | 552 records: 506 vaginal, 46 cesarean (8.33% prevalence) |
| Setting | Single Czech university hospital (Brno); selected subset of 9,164 recordings |
| Signal input | 4 × 1800 tensor: FHR, UC, FHR mask, UC mask at 1 Hz |
| Clinical input | 15 features: age, gravidity, parity, gestational age, 7 risk factors, presentation (3), missing indicator |
| Splits | 386 train / 83 validation / 83 test, stratified, seed 42 |

**Method.** Trim padding → last 30 minutes → FHR 30–240 bpm screening → short-gap interpolation → 1 Hz masked mean → training-set standardisation. Post-delivery outcomes and labour-progress/decision fields excluded to prevent leakage. Primary model: multimodal 1D CNN + clinical MLP, 64,121 parameters. Ablations: CTG-only 1D CNN (58,009), CTG-only CNN-BiLSTM (31,009) and clinical-only MLP (3,041). Baselines: clinical LR, CTG-descriptor LR, combined LR, combined gradient boosting.

**Evaluation.** Fixed held-out test never used for early stopping, calibration, or threshold selection. Platt calibration and Youden-J threshold fitted on validation only. Bootstrap 95% confidence intervals; stratified 5-fold cross-validation as a secondary check.

| Held-out test (83 records: 7 cesarean, 76 vaginal) | ROC-AUC (95% CI) | PR-AUC | Sensitivity / specificity at validation threshold |
|---|---:|---:|---:|
| Multimodal CNN + clinical | 0.733 (0.551–0.910) | 0.402 | 0.429 / 0.842 (threshold 0.14) |
| CTG-only 1D CNN | 0.797 (0.660–0.914) | 0.305 | 0.571 / 0.684 (threshold 0.08) |
| CTG-only CNN-BiLSTM | 0.720 (0.453–0.938) | 0.450 | 0.571 / 0.750 (threshold 0.08) |
| Clinical-only MLP | 0.498 (0.244–0.727) | 0.105 | 0.143 / 0.829 (threshold 0.11) |
| CTG logistic regression | 0.701 (0.451–0.904) | 0.204 | 0.714 / 0.618 (threshold 0.33) |

**Five-fold pooled ROC-AUC:** multimodal 0.767; CTG-only 1D CNN 0.763; CNN-BiLSTM 0.810; clinical-only 0.561; CTG LR 0.744; combined LR 0.726; combined GB 0.702; clinical LR 0.577.

**Three things to remember**
1. **The signal is weak and imprecise.** Only 7 test cesareans; all confidence intervals are wide and overlap.
2. **Accuracy is misleading.** A majority-class classifier reaches 91.6% accuracy with zero sensitivity.
3. **Deep learning does not clearly win.** A simple CTG logistic regression is competitive; the CNN-BiLSTM’s AUC difference versus the 1D CNN includes zero; and the clinical-only model is at chance.

**Main limitations.** Tiny positive class; selected single-centre cohort; cesarean recordings released as a separate subset with different duration and missingness, so recording quality may act as a shortcut; no expert CTG labels; no external or prospective validation; thresholds and calibration unlikely to transfer.

**Bottom line.** A complete, reproducible, leakage-aware research pipeline with an honest negative-leaning result — not a clinical tool.

---

## 5. Q&A and viva preparation

### “Why didn’t you predict fetal distress?”
Because the data do not support it. Version 1.0.0 has an empty `ANNOTATIONS` file and no `.atr` files, so there are no expert labels for normal, suspicious, or pathological CTG. Training on delivery mode and calling it fetal status would be a label substitution error. The project predicts delivery mode and states that limitation in the API, UI, model card, and reports.

### “Why is accuracy misleading here?”
Because 91.6% of the test set is vaginal delivery. A classifier that always predicts vaginal gets 91.6% accuracy while detecting **zero** cesareans. That is why the project reports ROC-AUC, PR-AUC, sensitivity, specificity, and confusion matrices at explicit thresholds. Accuracy is included only to show the trap, not as a headline result.

### “Is it clinically usable?”
No. There are 7 cesareans in the test set, the cohort is small, selected, and single-centre, there is no external or prospective validation, and the operating point produces false-positive rates that would not be acceptable in practice. The clinical-only model is at chance. It is a teaching and research prototype, and its output is not a diagnosis or a risk estimate for an individual patient.

### “What would you do next?”
First, obtain external or multi-centre data and validate calibration and subgroup performance. Second, add expert CTG annotations and test whether predicted fetal status explains any of the delivery-mode signal. Third, model recording quality explicitly — duration and missingness — and check whether performance survives quality matching. Fourth, pre-register the analysis and report decision curves, net benefit, and subgroup fairness. Only then consider a prospective comparison with clinician judgement.

### “Why use deep learning if logistic regression is similar?”
The deep models were the hypothesis-driven approach because they can learn temporal patterns directly from the raw CTG without hand-crafted descriptors. The CNN-BiLSTM tested whether longer-range bidirectional context helps beyond the 1D CNN; its AUC was 0.720 versus 0.797, and the paired bootstrap interval included zero, so it did not. The baselines are there to test whether complexity is justified. On this dataset it is not clearly justified: the CTG logistic regression reaches test ROC-AUC 0.701 and cross-validated 0.744, and the confidence intervals overlap with the CNNs. Reporting that honestly is part of the result.

### “What was your own contribution?”
The end-to-end implementation and evaluation: strict WFDB parsing, target verification, leakage audit, signal preprocessing with masks, the four PyTorch models (including the CNN-BiLSTM), the baseline suite, calibration and threshold policy, bootstrap confidence intervals, five-fold cross-validation, explainability, the FastAPI backend, the React interface, and the automated test suite. The dataset is public; the pipeline, evaluation, and documentation are the project work.

### “What is the biggest threat to validity?”
Acquisition confounding. The 46 cesarean recordings were released as a separate subset: their raw signals contain more zero-coded FHR overall (32.9% vs 17.5%) and their usable length after trimming is shorter (58.4 vs 71.3 minutes). Within the last 30-minute window, FHR missingness is actually lower for cesareans (9.3% vs 19.2%). Duration is not an input, but the missingness masks are, so the model could partly learn recording quality rather than physiology. This has not been eliminated.

---

## 6. Numbers and sources

All figures in this pack come from generated artefacts, not hand-edited text:

| Claim | Source |
|---|---|
| Cohort, target, splits, prevalence | [`../reports/dataset_inspection_report.md`](../reports/dataset_inspection_report.md) |
| Held-out metrics, confidence intervals, operating points | [`../reports/metrics.json`](../reports/metrics.json) and [`../reports/model_evaluation_report.md`](../reports/model_evaluation_report.md) |
| Architecture, parameter counts, thresholds, limitations | [`../models/model_card.json`](../models/model_card.json) and [`../models/MODEL_CARD.md`](../models/MODEL_CARD.md) |
| Five-fold cross-validation | [`../reports/cv_summary.md`](../reports/cv_summary.md) |
| Full methods and discussion | [`../reports/technical_report.md`](../reports/technical_report.md) |
| Deployment and governance caveats | [`hospital_integration.md`](hospital_integration.md) |

**Key figures used above**

- Dataset: 552 records, 506 vaginal, 46 cesarean, 8.33% prevalence; train 386 (32 CS), validation 83 (7 CS), test 83 (7 CS).
- Multimodal model: 64,121 parameters; CTG-only 1D CNN 58,009; CNN-BiLSTM 31,009; clinical-only 3,041.
- Test ROC-AUC: multimodal 0.733 (0.551–0.910), CTG-only 1D CNN 0.797 (0.660–0.914), CNN-BiLSTM 0.720 (0.453–0.938), clinical-only 0.498 (0.244–0.727).
- Test PR-AUC: 0.402, 0.305, 0.450, 0.105; prevalence baseline 0.084.
- Validation thresholds: multimodal 0.14, CTG-only 1D CNN 0.08, CNN-BiLSTM 0.08, clinical-only 0.11.
- At those thresholds: multimodal 3/7 cesareans detected, 12/76 false positives; 1D CNN 4/7 detected, 24/76 false positives; CNN-BiLSTM 4/7 detected, 19/76 false positives.
- At 0.5: multimodal sensitivity 0.286, specificity 0.987; CNN-BiLSTM sensitivity 0.286, specificity 0.987; 1D CNN and clinical-only sensitivity 0.000, specificity 1.000.
- Paired AUC difference, CNN-BiLSTM minus 1D CNN: −0.077 (95% CI −0.331 to +0.167), not statistically meaningful.
- Five-fold pooled ROC-AUC: 0.767 / 0.763 / 0.810 / 0.561.
- Majority-class classifier at 0.5: 91.6% accuracy, zero sensitivity.

---

## 7. Delivery checklist

- [ ] Say the research-prototype disclaimer in the first 30 seconds and leave it on Slide 1.
- [ ] Use “eventual mode of delivery”, never “fetal distress” or “predicts cesarean necessity”.
- [ ] Lead with ROC-AUC and PR-AUC; mention accuracy only to explain why it is misleading.
- [ ] State the test-set size (7 cesareans) whenever quoting a metric.
- [ ] Say explicitly that deep learning did not clearly outperform the CTG logistic regression.
- [ ] End with limitations and next steps, not with a performance claim.
- [ ] Keep the handout’s disclaimer visible if it is printed or projected.

---

**Final reminder.** This pack describes a research prototype built on a small, selected, single-centre public dataset. It is suitable for teaching, methodological demonstration, and further research. It is not suitable for patient care, triage, counselling, or any mode-of-delivery decision.
