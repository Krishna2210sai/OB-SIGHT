# Dataset Inspection Report

**Project:** AI-Based Mobile Clinical Decision Support System for Delivery Outcome Prediction  
**Dataset:** CTU-CHB Intrapartum Cardiotocography Database, version 1.0.0  
**Local directory:** `ctu-chb-intrapartum-cardiotocography-database-1.0.0/`  
**Inspection date:** 21 October 2026  
**Status:** research dataset; the original files were opened read-only and were not modified, renamed, or deleted.

---

## 1. Summary

The downloaded database contains **552 complete WFDB records**. Each record is a pair of files:

- `<record_id>.hea` — WFDB header, signal specification, and clinical metadata in comment lines;
- `<record_id>.dat` — two interleaved 16-bit channels sampled at **4 Hz**.

The verified prediction target is **mode of delivery**, derived from `Deliv. type`:

| `Deliv. type` | Meaning | Records |
|---:|---|---:|
| 1 | Vaginal delivery (spontaneous or operative vaginal) | 506 |
| 2 | Cesarean delivery | 46 |
| **Total** | | **552** |

There are no missing or unexpected target values. The cesarean records are the separate 2001–2046 series; the vaginal records are 1001–1506. The positive class (cesarean) is **8.33%** of the dataset.

This is **not** a fetal-status dataset in this distribution. The `ANNOTATIONS` file is empty and no `.atr` annotation files are present, so there are no expert CTG labels (for example FIGO categories) that could be used as a classification target. The model therefore predicts the eventual mode of delivery, not “abnormal CTG” or fetal hypoxia.

---

## 2. Files examined

| Item | Count / value |
|---|---:|
| `.hea` header files | 552 |
| `.dat` signal files | 552 |
| `RECORDS` | list of the 552 record IDs |
| `ANNOTATIONS` | empty |
| `SHA256SUMS` (`.txt`) | checksums for the distributed files |
| `wfdbcal` | WFDB calibration definitions |
| Total size | approximately 39.9 MB |
| Record ID range, vaginal | 1001–1506 (contiguous, 506 records) |
| Record ID range, cesarean | 2001–2046 (contiguous, 46 records) |
| Signal duration before trimming | 60.0–90.1 minutes |
| Sampling rate | 4 Hz |
| Signals per record | 2 (FHR, UC) |

Example header (`1001.hea`):

```text
1001 2 4 19200
1001.dat 16 100(0)/bpm 12 0 15050 20101 0 FHR
1001.dat 16 100/nd 12 0 700 378 0 UC

#----- Additional parameters for record 1001
#pH           7.14
...
#Deliv. type  1
```

Interpretation:

- `19200` samples at 4 Hz = 80 minutes.
- Format `16` means 16-bit little-endian samples; channels are interleaved.
- Gain is `100` for both channels; ADC baseline is 0.
- FHR is reported in bpm. UC is reported in `nd` (uncalibrated tocodynamometer units), not mmHg.
- Zero samples denote missing or invalid values; they are **not** physiological zeros.
- The raw files are zero-padded to a whole number of minutes. The pipeline trims leading/trailing samples where both channels are zero before selecting the last 30 minutes.

---

## 3. Metadata fields

The parser extracts 35 clinical/administrative fields from the header comments:

| Group | Fields |
|---|---|
| Neonatal / outcome | `pH`, `BDecf`, `pCO2`, `BE`, `Apgar1`, `Apgar5`, `NICU days`, `Seizures`, `HIE`, `Intubation`, `Main diag.`, `Other diag.` |
| Fetus / neonate | `Gest. weeks`, `Weight(g)`, `Sex` |
| Maternal | `Age`, `Gravidity`, `Parity`, `Diabetes`, `Hypertension`, `Preeclampsia`, `Liq. praecox`, `Pyrexia`, `Meconium` |
| Delivery | `Presentation`, `Induced`, `I.stage`, `NoProgress`, `CK/KP`, `II.stage`, `Deliv. type` |
| Recording | `dbID`, `Rec. type`, `Pos. II.st.`, `Sig2Birth` |

All fields are parsed strictly: a missing value becomes `NaN`; values are never fabricated or forward-filled from another record.

---

## 4. Target verification

The target was verified from the raw headers rather than inferred from file names:

1. Every one of the 552 headers contains exactly one `Deliv. type` line.
2. The only observed values are `1` and `2`.
3. All 46 records with value `2` belong to the 2001–2046 series.
4. The PhysioNet database documentation defines `Deliv. type 1` as vaginal delivery and `Deliv. type 2` as cesarean section.
5. No record has a missing or contradictory target.

Mapping used by the project (`src/config.py`):

```python
LABEL_MAP = {1: 0, 2: 1}          # 0 = vaginal, 1 = cesarean
CLASS_NAMES = ["Vaginal delivery", "Cesarean delivery"]
```

The positive class is cesarean delivery because it is the clinically consequential minority class and the class for which sensitivity matters in the research question.

---

## 5. Feature availability and leakage review

### 5.1 Inputs used by the models

**Maternal clinical inputs (12 raw fields, expanded to 15 model features):**

- `age`, `gravidity`, `parity`, `gest_weeks`;
- `diabetes`, `hypertension`, `preeclampsia`, `liquor_praecox`, `pyrexia`, `meconium`, `induced`;
- `presentation` (1 = vertex, 2 = breech, 3 = other/abnormal), one-hot encoded, plus a missing-presentation indicator.

These are plausibly available during labour and before the delivery mode is known.

**CTG inputs:** the most recent 30 minutes of FHR and UC, stored as four channels at 1 Hz: FHR, UC, FHR-valid mask, UC-valid mask.

### 5.2 Excluded fields and reasons

| Field(s) | Reason for exclusion |
|---|---|
| `pH`, `BDecf`, `pCO2`, `BE` | Cord-blood values measured after delivery; unavailable at prediction time. |
| `Apgar1`, `Apgar5`, `NICU days`, `Seizures`, `HIE`, `Intubation`, `Main diag.`, `Other diag.` | Neonatal outcomes known only after birth. |
| `Weight(g)`, `Sex` | Neonatal descriptors available only after delivery. |
| `II.stage`, `NoProgress`, `CK/KP`, `I.stage`, `Pos. II.st.`, `Sig2Birth`, `Rec. type` | Labour-progress, decision, or recording-selection variables. They can encode the decision to perform a cesarean or the timing of the recording and would create target leakage. |
| `dbID` | Administrative identifier, not a clinical predictor. |
| `Deliv. type` | Target. |
| `record_id` / `record_num` | Identifiers. The 2000-series also encodes the target, so it is never a feature. |

`I.stage` is also numerically implausible in places (negative values and extreme outliers), which is a further reason not to use it without careful validation.

### 5.3 Missing metadata

| Field | Missing records | Share |
|---|---:|---:|
| `BDecf`, `pCO2`, `BE` | 11 each | 2.0% |
| `Weight(g)` | 1 | 0.2% |
| `Gravidity` | 4 | 0.7% |
| `Presentation` | 3 | 0.5% |
| `Age`, `Parity`, `Gest. weeks`, all binary risk factors, `Deliv. type` | 0 | 0% |

All three missing presentations occur in vaginal records. For modelling, numeric values are median-imputed and presentation is mode-imputed with an explicit `presentation_missing` indicator. Imputation statistics are fitted on the training split only.

---

## 6. Signal quality

After trimming zero-padding and selecting the last 30 minutes:

| Group | n | Trimmed duration, min (mean) | FHR zero fraction, full raw recording (mean) | FHR missing fraction, last 30 min (mean) | UC missing fraction, last 30 min (mean) |
|---|---:|---:|---:|---:|---:|
| Vaginal | 506 | 71.3 | 17.5% | 19.2% | 15.3% |
| Cesarean | 46 | 58.4 | 32.9% | 9.3% | 13.9% |
| All | 552 | 70.2 | — | 15.4% | 16.0% |

Additional observations:

- Every trimmed recording is at least **41.2 minutes** long, so all 552 records provide a complete 30-minute model window. No record required left-padding.
- 23 vaginal records have more than 50% missing FHR in the last 30 minutes; no cesarean record does.
- 2 vaginal records have an entirely zero UC channel in the window; no cesarean record does.
- Cesarean recordings come from a separately released 46-case subset. Their raw recordings contain substantially more zero-coded FHR samples overall, but their last-30-minute missingness is lower than in vaginal records. Recording length and quality therefore differ by class and are potential shortcut signals. Duration is not an input, but the two missingness-mask channels are, so this confound cannot be ruled out.
- The signal pipeline treats FHR outside 30–240 bpm as invalid, linearly interpolates FHR gaps of at most 5 s and UC gaps of at most 30 s, keeps longer gaps missing (value 0, mask 0), and block-averages the four native samples per second into one 1 Hz sample.

---

## 7. Class distribution and splits

A fixed, record-level, stratified split was created once with `random_state=42` and stored in `data/processed/splits.csv`. The test set was not used for early stopping, calibration, or threshold selection.

| Split | Vaginal | Cesarean | Total | Cesarean share |
|---|---:|---:|---:|---:|
| Train | 354 | 32 | 386 | 8.3% |
| Validation | 76 | 7 | 83 | 8.4% |
| Test | 76 | 7 | 83 | 8.4% |
| **Total** | **506** | **46** | **552** | **8.3%** |

Because each record is one delivery, splitting by record prevents the same recording from appearing in more than one split. The dataset does not expose a stable patient identifier, so repeated pregnancies of the same person cannot be grouped; the split is record-level, not patient-level.

---

## 8. Implications for modelling

1. **Severe class imbalance.** With 46 events in total and 7 in the test set, accuracy is misleading. A majority-class classifier reaches 91.6% accuracy while detecting zero cesareans.
2. **Wide uncertainty.** Bootstrap confidence intervals for ROC-AUC span roughly 0.2–0.4 AUC units. Cross-validation is reported as a secondary, more stable estimate, not as external validation.
3. **Recording-quality confounding.** The cesarean subset differs in duration and missingness. Any learned signal may partly reflect how and when the recording was collected rather than fetal physiology.
4. **No expert CTG labels.** The model cannot be validated against FIGO or other expert interpretations in this distribution.
5. **No external validation.** All results come from a single-centre, selected cohort. They do not establish transferability to another hospital or population.
6. **Research use only.** The dataset and model are unsuitable for clinical deployment without prospective validation, regulatory review, and local governance.

---

## 9. Reproduction

The inspection and processed artefacts can be regenerated with:

```bash
python -m src.data.prepare_dataset
```

This writes only to `data/processed/` and `models/`; the original dataset directory is read-only. See `README.md` for the full pipeline and `reports/technical_report.md` for methods and results.
