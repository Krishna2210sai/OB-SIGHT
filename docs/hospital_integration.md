# Hospital Integration and Governance Notes

**Status: hypothetical design discussion. No hospital integration is implemented in this repository.**

OB-SIGHT is a research prototype. This document describes what a responsible integration *would* require so that reviewers can see the gap between the prototype and a clinical system. It is not an implementation guide for deployment, and nothing here constitutes regulatory, legal, or clinical advice.

---

## 1. Scope and non-goals

### In scope of this discussion

- How CTG and maternal data could be supplied to the model in a hospital workflow.
- Authentication, authorisation, audit, and data-protection expectations.
- Validation, monitoring, change control, and incident response.
- Failure modes and safe fallback behaviour.

### Explicit non-goals

- This repository does **not** contain an EHR connector, HL7/FHIR server, message queue, identity provider, audit database, or deployment manifests.
- The API is **not** hardened for internet exposure. It has no authentication, rate limiting, TLS termination, or multi-tenancy.
- No clinical evaluation, regulatory submission, CE marking, FDA clearance, or ethics approval has been obtained.
- No real patient data are used.

A hospital must treat any future deployment as a new regulated project, not as an extension of this prototype.

---

## 2. Intended clinical workflow (if it were ever validated)

The model predicts the **eventual mode of delivery**, not fetal status and not the immediate clinical decision. A safe hypothetical workflow would be:

1. A clinician opens the assessment for a labouring patient and confirms identity and encounter context.
2. The system retrieves the most recent 30 minutes of intrapartum FHR and UC, plus maternal age, gravidity, parity, gestational age, risk factors, presentation, and induction status.
3. The model returns a calibrated score, a binary classification at a disclosed threshold, warnings, and non-causal attributions.
4. The result is displayed **as advisory research information only**, alongside the raw CTG and the full clinical picture.
5. The clinician documents their own interpretation and decision. The model never orders, schedules, or recommends a cesarean.
6. The encounter is logged with model version, input provenance, output, threshold, and user action.

The model must not run silently in the background, trigger alerts, or populate orders. It must not be used for triage or resource allocation.

---

## 3. Data sources and mapping

### 3.1 CTG waveform

CTG waveforms are often stored in vendor-specific formats or as raw sampling streams. A hospital would need a validated, versioned mapping to the model's expected two-channel input:

| Model input | Expected representation |
|---|---|
| FHR | beats per minute, 4 Hz recommended, zero/blank for missing |
| UC | uncalibrated tocodynamometer units, 4 Hz, zero/blank for missing |
| Time | sample index at a declared sampling frequency; timezone and clock source recorded |
| Window | most recent 30 minutes before the prediction timestamp |

The API accepts CSV (`fhr`, `uc`, optional `time`) or JSON (`{"fs": 4, "fhr": [...], "uc": [...]}`) only. A production connector would need to extract, validate, and de-identify data before calling the API, and to preserve units and sampling frequency explicitly. Unit confusion (for example UC in mmHg versus arbitrary units) is a serious failure mode; the prototype cannot detect it.

### 3.2 Maternal data

The 15 model features map to ordinary clinical concepts. A future integration would define a local terminology mapping (for example to SNOMED CT or local codes) and a provenance record for every field. The mapping must distinguish:

- confirmed diagnosis versus suspected condition;
- pre-existing versus gestational diabetes;
- chronic versus pregnancy-induced hypertension;
- spontaneous versus induced labour;
- presentation confirmed by ultrasound versus clinical examination.

The prototype accepts only the simplified values defined in `backend/schemas.py`. A connector must not silently coerce ambiguous source data into these categories.

### 3.3 Missing data

Missing values are allowed for some clinical fields; the model imputes them with training statistics and flags missing presentation. A hospital integration must make missingness visible to the user rather than hiding it. If the CTG window is shorter than 30 minutes, the API left-pads and warns; a clinical workflow should probably refuse the assessment instead of relying on a padded window.

---

## 4. Reference architecture (conceptual)

```text
EHR / CTG archive
      │  (HL7 v2, FHIR R4, or vendor API; locally validated mapping)
      ▼
Integration gateway
  - authentication, authorisation
  - encounter and patient context
  - unit / sampling-rate validation
  - de-identification or pseudonymisation
  - audit event creation
      │  mTLS, internal network
      ▼
Model service (containerised FastAPI)
  - pinned model + preprocessing + calibrator versions
  - input schema validation
  - inference and explanation
  - response includes model version, threshold, warnings
      │
      ├──► Clinical UI (advisory display, no auto-orders)
      ├──► Audit store (immutable, access-controlled)
      └──► Monitoring (latency, errors, drift proxies, usage)
```

Design requirements for any real build:

- **No direct internet exposure.** Place the service behind a reverse proxy with TLS, network segmentation, and a web application firewall.
- **Workload identity.** Use OIDC/OAuth2 or mTLS between services; no shared static API keys in source control.
- **Least privilege.** Separate roles for clinical users, data engineers, model maintainers, and auditors.
- **Immutable artefact versions.** Model weights, preprocessing, calibrator, threshold, and code commit must be recorded together. A prediction must be reproducible from the audit record.
- **No PHI in logs.** Log request IDs, model version, latency, and error codes, not raw signals or identifiers.
- **Clock discipline.** The “most recent 30 minutes” depends on the prediction timestamp; clock skew can silently select the wrong window.
- **Fail closed for clinical use.** If artefacts are missing or inputs are invalid, return an explicit unavailable status. Never substitute a default model or silently impute a whole signal.

---

## 5. Security and privacy

A production system would require, at minimum:

- a data-protection impact assessment and lawful basis for processing;
- ethics approval and, where applicable, a clinical trial or evaluation agreement;
- encryption in transit and at rest, key management, and rotation;
- strong authentication, session timeouts, and role-based access;
- audit logs for access, inference, and configuration changes, retained per local policy;
- data minimisation: only the features required for the assessment;
- retention and deletion rules, including backups;
- vulnerability management, dependency pinning, and secure build pipelines;
- penetration testing and incident-response exercises;
- business-associate or data-processing agreements with vendors;
- compliance with applicable law (for example GDPR in the EU, HIPAA in the US, and local medical-device and AI regulations).

The demonstration records in this repository are public research data, but the same technical controls would be required for any real patient data.

---

## 6. Validation and evidence expectations

The current evaluation is a small, single-centre, retrospective held-out test with 7 cesarean events. It is **not** sufficient for clinical use. A credible pathway would include:

1. **External validation** in at least one independent hospital, with a pre-specified protocol and sample size.
2. **Temporal validation** to test drift over time.
3. **Prospective silent evaluation** in which the model runs alongside care without influencing decisions, to measure performance and workflow fit.
4. **Calibration analysis** (calibration slope/intercept, reliability curves, decision-curve/net-benefit analysis) in the target population.
5. **Subgroup analyses** for age, parity, presentation, BMI, comorbidities, and site, with explicit power calculations.
6. **Comparison with clinician judgement** and with simpler rules or existing tools.
7. **Human-factors testing** of the interface, warnings, and hand-off.
8. **Health-economic and equity assessment**, including false-positive harms and downstream testing.
9. **Regulatory classification** and a quality-management system (for example ISO 13485, IEC 62304, ISO 14971) if the tool meets the definition of a medical device.
10. **Post-market surveillance** with predefined drift and safety triggers.

With 46 events in the full dataset, even a well-run external study would need substantially more cesarean cases to estimate sensitivity with useful precision. The current model should be treated as a hypothesis-generating prototype.

---

## 7. Monitoring and maintenance

If a validated system existed, monitoring would include:

| Dimension | Example signal |
|---|---|
| Availability | request success rate, latency p50/p95, error codes |
| Input quality | missing FHR/UC fraction, sampling-rate mismatches, window padding rate |
| Distribution shift | feature summaries versus training reference; signal-quality drift |
| Performance | periodic audit against adjudicated outcomes, sensitivity/specificity, calibration |
| Usage | assessments per shift, override rate, user feedback |
| Safety | near-misses, incorrect-unit events, downtime, incident reports |

Thresholds for alerting and automatic suspension must be defined **before** deployment. A drop in data quality or a shift in case mix should trigger review, not automatic retraining. Retraining requires a new version, new validation, and change-control approval.

---

## 8. Failure modes and safe fallback

| Failure | Expected behaviour |
|---|---|
| Model artefacts missing or checksum mismatch | Return HTTP 503; do not predict. |
| Unsupported file type or malformed CSV/JSON | Return HTTP 400 with a clear message; do not guess. |
| Recording shorter than 10 minutes | Reject. |
| Recording 10–30 minutes | Prototype pads and warns; clinical workflow should reject or require review. |
| NaN, infinite, all-zero UC, implausible FHR | Reject with a specific error. |
| Score near the decision threshold | Display an instability warning; do not present a confident classification. |
| Model/API unavailable | Clinical workflow continues without the tool; no degradation of standard care. |
| Version mismatch between UI, API, and artefacts | Block the assessment and alert support. |

The system must never fail “open” into an automatic clinical action. The safe default is always standard care without the model.

---

## 9. Change control and documentation

Any change to the following requires a new model version, a new model card, and re-evaluation:

- training data or labels;
- preprocessing, imputation, or signal window;
- architecture or training procedure;
- calibration method;
- decision threshold or operating point;
- feature definitions or source mappings;
- dependency versions that affect numerical output.

The audit record for each prediction should include the model version, artefact hashes, threshold, input provenance, output, and user action. Model cards and evaluation reports must be retained with the release.

---

## 10. Decommissioning

A future system would need a decommissioning plan covering:

- notification of users and stakeholders;
- preservation of audit records for the legally required period;
- secure deletion of patient data and derived artefacts;
- revocation of credentials and network access;
- archiving of model versions and documentation;
- a communication plan for any patients or clinicians affected by historical outputs.

---

## 11. Integration readiness checklist

Use this as a gap analysis, not as approval:

- [ ] Intended use and regulatory classification agreed.
- [ ] External validation completed and published.
- [ ] Calibration and subgroup performance acceptable in the target population.
- [ ] Clinical workflow and human-factors study completed.
- [ ] Data-protection impact assessment and ethics approval obtained.
- [ ] Security architecture reviewed and penetration-tested.
- [ ] Authentication, authorisation, and audit implemented.
- [ ] Input mapping and unit validation verified against source systems.
- [ ] Monitoring, alerting, drift detection, and rollback tested.
- [ ] Incident-response and downtime procedures exercised.
- [ ] Model card, instructions for use, and training materials approved.
- [ ] Post-market surveillance and change-control plan in place.

Until every item is addressed, the only appropriate status is **research prototype, not for clinical use**.

---

## 12. Summary

The prototype demonstrates a complete technical pipeline, but the distance to safe clinical integration is large. The main gaps are external validation, evidence of clinical utility, acquisition confounding, regulatory approval, security hardening, and workflow evaluation. Integration should not begin with an API connection; it should begin with a validated clinical question, a prospective protocol, and a governance decision about whether the tool should exist at all.
