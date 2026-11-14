# OB-SIGHT — AI-Based Mobile Clinical Decision Support System for Delivery Outcome Prediction

**Research prototype — not a medical device and not for clinical use.**

OB-SIGHT predicts the **eventual mode of delivery** (cesarean versus vaginal) from intrapartum cardiotocography (CTG) and maternal clinical data. It is an end-to-end academic project: dataset inspection, leakage-aware preprocessing, PyTorch models, classical baselines, held-out evaluation, calibration, explainability, a FastAPI backend, a mobile-first React interface, automated tests, and reports.

The system is built on the public **CTU-CHB Intrapartum Cardiotocography Database v1.0.0** (PhysioNet, DOI `10.13026/0d3m-5v02`). It does **not** diagnose fetal distress and does **not** recommend a delivery mode. Version 1.0.0 of this database contains no expert CTG annotations, so fetal-status classification is not a valid target for this data.

---

## 1. At a glance

| Item | Value |
|---|---|
| Task | Binary classification: cesarean (positive) vs vaginal delivery |
| Target field | `Deliv. type` — 1 = vaginal, 2 = cesarean |
| Cohort | 552 records (506 vaginal, 46 cesarean; 8.33% prevalence) |
| Inputs | 30 min of CTG at 1 Hz (4 channels) + 15 maternal clinical features |
| Deep models | Multimodal CNN + clinical MLP; CTG-only 1D CNN; CTG-only CNN-BiLSTM; clinical-only MLP |
| Baselines | Clinical logistic regression, CTG-descriptor logistic regression, combined logistic regression, combined gradient boosting |
| Split | Fixed stratified 70/15/15 (386 / 83 / 83), seed 42 |
| Evaluation | Held-out test + stratified 5-fold CV + bootstrap CIs + paired AUC comparison |
| Backend | FastAPI + Pydantic, CSV/JSON CTG uploads |
| Frontend | React 19 + Vite, mobile-first |
| Tests | 105 pytest tests |

**Headline result (held-out test, 7 cesareans / 76 vaginal):** the CTG-only 1D CNN achieved ROC-AUC 0.797 (95% CI 0.660–0.914); the multimodal model achieved 0.733 (0.551–0.910); the new CTG-only CNN-BiLSTM achieved 0.720 (0.453–0.938) but the highest PR-AUC (0.450). The paired AUC difference between the CNN-BiLSTM and the 1D CNN was −0.077 (95% CI −0.331 to +0.167), so the experiment does not establish a statistically meaningful improvement. At validation-selected thresholds, the multimodal model detected 3/7 cesareans with 12/76 false positives. A majority-class classifier reaches 91.6% accuracy while detecting no cesareans. These results are exploratory, imprecise, and not clinically usable. See `reports/model_evaluation_report.md` and `reports/technical_report.md`.

---

## 2. Safety and intended use

**Intended use**

- Teaching and demonstration of an end-to-end clinical machine-learning workflow.
- Research on multimodal intrapartum CTG and maternal-data modelling.
- Reproducible benchmarking against simple baselines on a public research dataset.

**Prohibited use**

- Any real clinical decision, triage, counselling, or mode-of-delivery decision.
- Use as a medical device or as input to one without regulatory approval and prospective validation.
- Use on patient data without a lawful basis, ethics approval, and data-protection safeguards.
- Treating demonstration records as real patient cases.

The output is a statistical score from a small, selected, single-centre dataset. It is **not** a diagnosis, a risk estimate for an individual patient, or a substitute for professional obstetric judgement.

---

## 3. Repository layout

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
│   │   └── ctg_features.py     # hand-crafted CTG descriptors
│   ├── models/
│   │   ├── deep_learning_model.py
│   │   ├── baseline.py
│   │   └── calibration.py      # Platt scaling (pickle-safe)
│   ├── training/train.py       # final training + 5-fold CV
│   ├── phase_a/                # additive Phase A data-handling experiments
│   │   ├── run.py              # staged CLI (quality/window/sampling/features/deep/report)
│   │   ├── quality.py          # per-record CTG quality audit
│   │   ├── signals.py          # anti-aliased window extraction and masks
│   │   ├── features.py         # clinical + CTG feature assembly
│   │   ├── selection.py        # fold-specific feature selectors and stability
│   │   ├── experiment.py       # fixed folds, CV runners, metrics
│   │   ├── plots.py
│   │   └── report.py
│   └── evaluation/
│       ├── evaluate.py         # held-out metrics, CIs, figures, report
│       ├── metrics.py
│       ├── cv_report.py
│       └── make_model_card.py
├── frontend/                   # React/Vite mobile-first app
├── tests/                      # pytest suite
├── data/processed/             # generated; safe to rebuild
├── models/                     # checkpoints, calibrators, model card
├── reports/                    # metrics, figures, reports
├── experiments/phase_a/        # Phase A outputs, comparison tables, figures, logs
├── samples/                    # sample and synthetic CTG uploads
├── docs/hospital_integration.md
├── requirements.txt
├── .env.example
└── ctu-chb-intrapartum-cardiotocography-database-1.0.0/   # original data (read-only)
```

The original dataset directory is treated as read-only. Generated files go to `data/processed/`, `models/`, and `reports/`.

---

## 4. Quick start (use the included artefacts)

### 4.1 Python environment

Python 3.12 is recommended.

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

### 4.2 Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4.3 Start the API and app

From the project root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:

- app: <http://127.0.0.1:8000>
- interactive API docs: <http://127.0.0.1:8000/docs>
- health check: <http://127.0.0.1:8000/health>

FastAPI serves `frontend/dist` when it exists. If you have not built the frontend, `/` returns a small JSON status message and the API still works.

The Dashboard lists the **Multimodal CNN + clinical MLP** and the **CTG-only CNN-BiLSTM** as two separate model cards with their own held-out metrics and “Use this model” links. The Assessment page exposes them as separate top-level choices (`/assess?model=multimodal` or `/assess?model=bilstm`), alongside the clinical-only model. Held-out demo cases are scored with whichever CTG model is selected.

### 4.4 Frontend development mode

Run the backend on port 8000, then:

```bash
cd frontend
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/health`, `/predict`, `/predict-ctg`, `/demo`, `/demo-cases`, `/metrics`, `/model-info`, and `/api` to the backend. Set `VITE_API_BASE` in `frontend/.env.local` (template: `frontend/.env.example`) only if you need a different API origin.

---

## 5. Reproducing the full pipeline

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
- Step 4 writes `reports/metrics.json`, `reports/model_evaluation_report.md`, `reports/test_predictions.csv`, and figures.
- Step 7 must run after step 4 (and after CV if you want CV fields in the card).
- Training uses validation loss for early stopping; the test split is never used for model selection.
- A full CPU run may take several minutes per deep model. The reported checkpoints are already included.

### 5.1 Command-line options

```bash
python -m src.training.train --help
```

Options include `--variant {multimodal,signal,bilstm,clinical}`, `--cv`, `--epochs`, `--batch-size`, `--lr`, `--patience`, and `--seed`.

### 5.2 Phase A — data-handling and preprocessing experiments

Phase A is an additive research layer. It does not change the production preprocessing, the saved models, the API or the frontend. It reuses the raw recordings, the fixed split and the existing model architectures, and writes everything to `experiments/phase_a/`.

```bash
# Full Phase A pipeline (quality audit, window/sampling/feature experiments,
# development-set CV comparison, deep-learning screening, report)
python -m src.phase_a.run --stage all

# Same without the CPU-expensive deep-learning screening
python -m src.phase_a.run --stage all --no-dl

# Individual stages (all are deterministic with seed 42)
python -m src.phase_a.run --stage quality --force
python -m src.phase_a.run --stage window
python -m src.phase_a.run --stage sampling
python -m src.phase_a.run --stage features
python -m src.phase_a.run --stage compare
python -m src.phase_a.run --stage report

# Fast smoke run
python -m src.phase_a.run --stage all --quick --force
```

All Phase A cross-validation uses only the 469 development records (`split` in `train`/`validation`). The 83 held-out test records are never loaded. Outputs include the per-record quality report, explicit exclusion lists, fixed fold assignments, per-fold selected features and stability, OOF predictions with bootstrap confidence intervals, comparison tables, and figures. The generated report is written to both the repository root and `experiments/phase_a/reports/PHASE_A_DATA_HANDLING_REPORT.md`.

---

## 6. Data and preprocessing in brief

- **Target:** `Deliv. type` 1 = vaginal (506), 2 = cesarean (46). All cesareans are the 2001–2046 series. This was verified from every header and the PhysioNet documentation.
- **No fetal-status labels:** `ANNOTATIONS` is empty and there are no `.atr` files. The project therefore predicts delivery mode, not CTG status.
- **Clinical inputs (15):** age, gravidity, parity, gestational age, diabetes, hypertension, preeclampsia, liquor praecox, pyrexia, meconium, induced labour, presentation one-hot (3), presentation-missing indicator.
- **Excluded (leakage):** cord pH/BDecf/pCO2/BE, Apgar, NICU, seizures, HIE, intubation, neonatal diagnosis, birth weight, sex, `I.stage`, `II.stage`, `NoProgress`, `CK/KP`, `Pos. II.st.`, `Sig2Birth`, `Rec. type`, identifiers.
- **Signal:** trim zero-padding → last 30 min → screen FHR 30–240 bpm → interpolate gaps ≤5 s (FHR) / ≤30 s (UC) → 1 Hz masked block mean → training-set standardisation. Output `4 × 1800`.
- **Splits:** 386 train (32 CS), 83 validation (7 CS), 83 test (7 CS), stratified at record level. Preprocessing, calibration, and thresholds are fitted without the test set.

Full audit: [`reports/dataset_inspection_report.md`](reports/dataset_inspection_report.md).

---

## 7. Models

| Model | Role | Inputs |
|---|---|---|
| Multimodal CNN + MLP | Primary deep model | CTG tensor + clinical vector |
| CTG-only 1D CNN | Ablation | CTG tensor |
| CTG-only CNN-BiLSTM | Temporal ablation | CTG tensor |
| Clinical-only MLP | Ablation | clinical vector |
| Clinical LR | Classical baseline | clinical fields |
| CTG LR | Classical baseline | 25 hand-crafted CTG descriptors only |
| Combined LR | Classical baseline | clinical + CTG descriptors |
| Combined GB | Classical baseline | clinical + CTG descriptors |

Deep models use class-weighted binary cross-entropy, AdamW, dropout, batch normalisation, mild Gaussian input noise, gradient clipping, and early stopping on validation loss. Platt calibration is fitted on validation data. Decision thresholds are selected by Youden's J on validation data and applied unchanged to the test set.

The CTG-only CNN-BiLSTM consumes the same sequential `4 × 1800` tensor, not tabular descriptors. Three convolution blocks (16–32–48 channels) with ReLU and two pooling stages produce a 225-step sequence; a one-layer bidirectional LSTM (32 hidden units per direction) then models temporal context, followed by dropout and a small fully connected head. It has 31,009 parameters and was trained with the same 80-epoch / patience-15 configuration as the other deep models. The live prediction UI uses the multimodal and clinical-only models; the CTG-only models are comparison/ablation models, and all four checkpoints are covered by the automated tests.

---

## 8. Results summary

### Held-out test at validation-selected thresholds

| Model | Threshold | ROC-AUC (95% CI) | PR-AUC | Sensitivity | Specificity | F1 |
|---|---:|---|---:|---:|---:|---:|
| Multimodal DL | 0.14 | 0.733 (0.551–0.910) | 0.402 | 0.429 | 0.842 | 0.273 |
| CTG-only 1D CNN | 0.08 | 0.797 (0.660–0.914) | 0.305 | 0.571 | 0.684 | 0.229 |
| CTG-only CNN-BiLSTM | 0.08 | 0.720 (0.453–0.938) | 0.450 | 0.571 | 0.750 | 0.267 |
| Clinical-only DL | 0.11 | 0.498 (0.244–0.727) | 0.105 | 0.143 | 0.829 | 0.095 |
| CTG LR | 0.33 | 0.701 (0.451–0.904) | 0.204 | 0.714 | 0.618 | 0.244 |
| Combined GB | 0.37 | 0.675 (0.425–0.882) | 0.194 | 0.286 | 0.895 | 0.235 |
| Combined LR | 0.64 | 0.611 (0.397–0.810) | 0.184 | 0.286 | 0.855 | 0.200 |
| Clinical LR | 0.61 | 0.468 (0.238–0.716) | 0.103 | 0.286 | 0.697 | 0.125 |

Five-fold pooled ROC-AUC: multimodal 0.767, CTG-only 1D CNN 0.763, CNN-BiLSTM 0.810, clinical-only 0.561, CTG LR 0.744, combined LR 0.726, combined GB 0.702, clinical LR 0.577.

**Read these numbers carefully:**

- Only 7 cesareans are in the test set; confidence intervals are wide and overlap.
- The CNN-BiLSTM had the highest PR-AUC (0.450), but its ROC-AUC was lower than the 1D CNN and the paired bootstrap interval for the AUC difference (−0.331 to +0.167) includes zero. The experiment does not establish a statistically meaningful improvement.
- Cross-validation tells a different story: the CNN-BiLSTM had the highest pooled ROC-AUC (0.810 versus 0.763 for the 1D CNN), with substantial fold-to-fold variation (mean 0.800 ± 0.096). The test-set reversal shows that the ranking is not stable on this small cohort.
- A majority-class classifier reaches 91.6% accuracy at threshold 0.5 while detecting zero cesareans.
- The clinical-only model is at chance on the test set.
- Low validation-selected thresholds trade specificity for sensitivity; the resulting false-positive rates are not clinically acceptable.
- No external or prospective validation exists. The model is not ready for any clinical use.

**Overfitting.** The CNN-BiLSTM training loss fell from 1.253 at epoch 1 to 0.054 at epoch 34, while validation loss reached its minimum of 0.334 at epoch 19 and then rose again. Validation ROC-AUC peaked around epoch 14. The epoch-19 checkpoint was retained and early stopping ended training at epoch 34. With 386 training records and only 32 training cesareans, this is clear overfitting; regularisation reduced but did not eliminate it. No synthetic data were used, and the test split was never used for early stopping.

All key numbers are generated from `reports/metrics.json`; nothing is hand-typed into the UI.

---

## 9. API

Base URL: `http://127.0.0.1:8000`. Interactive schema: `/docs`.

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

### `GET /model-info`

Returns the model card: dataset, architecture, evaluation, limitations, intended and prohibited use, disclaimer.

### `GET /metrics`

Returns the canonical held-out metrics and interpretation text.

### `POST /predict` — clinical only

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

### `POST /predict-ctg` — multimodal, CNN-BiLSTM or 1D CNN

Multipart form with a `file` (CSV with `fhr`, `uc`, optional `time`; or JSON `{"fs": 4, "fhr": [...], "uc": [...]}`), the same clinical fields, and an optional `model` field: `multimodal` (default), `bilstm` (CTG-only CNN-BiLSTM), or `signal` (CTG-only 1D CNN). The website exposes `multimodal` and `bilstm` in the Assessment view.

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

### Demo endpoints

```bash
curl http://127.0.0.1:8000/demo-cases
curl -X POST "http://127.0.0.1:8000/demo/2005?model=bilstm"
```

Demo records come from the held-out public test split: 14 cases, 7 vaginal and 7 cesarean. The assessment page shows 6 at a time with a control to reveal the rest. True labels are shown only for teaching and are never model inputs.

### Response fields

Every prediction includes `prediction`, `predicted_class`, `cesarean_probability`, `vaginal_probability`, `decision_threshold`, `model_variant`, `calibrated`, clinical and signal attributions, warnings, and a disclaimer. The CTG endpoint additionally returns `ctg_summary`.

For a deployment-oriented discussion (FHIR/HL7, security, governance), see [`docs/hospital_integration.md`](docs/hospital_integration.md). It is a design discussion, not an implemented integration.

---

## 10. Tests

```bash
python -m pytest -q
```

The suite covers:

- WFDB header parsing, target verification, and raw signal layout;
- windowing, padding, gap interpolation, masks, and train-only scaling;
- metric calculations and threshold selection;
- model forward shapes, checkpoint loading, parameter counts, the CNN-BiLSTM architecture, calibrators, model-card consistency, and the paired AUC comparison;
- CTG CSV/JSON validation and rejection paths;
- API contracts, thresholds, error codes, demo records, sample downloads, and static hosting;
- Phase A quality audit, zero-padding, anti-aliased resampling, mask preservation, fractional-rate CTG features, fold-specific selectors, and the expanded metric set.

Expected result with the included artefacts: **105 passed**. The tests do not modify the original dataset.

---

## 11. Documentation

| Document | Contents |
|---|---|
| [`reports/dataset_inspection_report.md`](reports/dataset_inspection_report.md) | file inventory, target verification, leakage audit, missingness, signal quality, splits |
| [`reports/technical_report.md`](reports/technical_report.md) | methods, architectures, evaluation protocol, full results, limitations, references |
| [`reports/model_evaluation_report.md`](reports/model_evaluation_report.md) | generated held-out metric tables |
| [`reports/cv_summary.md`](reports/cv_summary.md) | generated 5-fold cross-validation summary |
| [`models/MODEL_CARD.md`](models/MODEL_CARD.md) | concise model card for reviewers |
| [`models/model_card.json`](models/model_card.json) | machine-readable card served by `/model-info` |
| [`docs/teacher_presentation.md`](docs/teacher_presentation.md) | 8-slide outline, 5–6 minute spoken script, one-page handout, and viva Q&A |
| [`docs/hospital_integration.md`](docs/hospital_integration.md) | hypothetical integration, security, and governance considerations |

---

## 12. Troubleshooting

**`ModuleNotFoundError: backend`** — run commands from the project root, not from `backend/` or `frontend/`.

**`Model artifacts could not be loaded`** — ensure `models/*.pt`, `models/*.joblib`, `models/model_card.json`, `reports/metrics.json`, and `data/processed/*.npz` exist. Re-run the pipeline if they do not.

**Port 8000 already in use** — stop the old server or run `uvicorn backend.main:app --port 8001` and set `VITE_API_BASE=http://127.0.0.1:8001` when developing the frontend.

**Frontend shows JSON instead of the app** — build it with `npm run build` in `frontend/`, then reload `/`.

**Torch installation is slow or large** — install the CPU-only wheel for your platform from <https://pytorch.org/get-started/locally/>, then install the remaining requirements.

**Different Python version** — Python 3.12 is recommended. The pinned versions may not have wheels for older interpreters.

---

## 13. Licence and data use

The original CTU-CHB database is distributed by PhysioNet under its own terms. Consult the PhysioNet page and the database documentation before redistributing data. The code in this repository is an academic project and is provided for research and teaching; no clinical warranty is given.

If you use the dataset, cite:

> Chudáček V, Spilka J, Burša M, et al. Open access intrapartum CTG database. *BMC Pregnancy and Childbirth*. 2014;14:16. doi:10.1186/1471-2393-14-16.

---

## 14. Final reminder

**This is a research prototype.** A high score in this repository would not mean the system is safe or effective. The dataset is small, selected, single-centre, and lacks external validation. The only responsible use of this code is teaching, methodological study, and further research under appropriate ethical and regulatory oversight.
