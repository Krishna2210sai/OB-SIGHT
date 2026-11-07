import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { getDemoCases, getMetrics, predictClinical, predictDemo, predictWithCTG, sampleUrl } from "../api.js";
import ClinicalForm, { EMPTY_FORM, SAMPLE_FORM, toPayload, validateClinical } from "../components/ClinicalForm.jsx";
import { Badge, ErrorNote, Spinner, useDocumentTitle } from "../components/UI.jsx";
import { saveHistory } from "../history.js";
import { IconAlert, IconChevron, IconDownload } from "../components/Icons.jsx";

const MAX_BYTES = 5 * 1024 * 1024;
const DEMO_PREVIEW_COUNT = 6;

const MODE_NOTES = {
  multimodal:
    "Multimodal CNN + clinical MLP: 30-minute CTG window, signal-quality masks and maternal variables.",
  bilstm:
    "CTG-only CNN-BiLSTM: 30-minute CTG tensor; maternal fields are collected but ignored by this model.",
  clinical:
    "Clinical-only MLP: uses only the maternal variables. In held-out testing this model did not discriminate better than chance (ROC-AUC ≈ 0.50)."
};

function validateFile(file) {
  if (!file) return "Choose a CTG file to upload";
  const lower = file.name.toLowerCase();
  if (!lower.endsWith(".csv") && !lower.endsWith(".json")) {
    return "Only .csv and .json CTG files are supported";
  }
  if (file.size === 0) return "The selected file is empty";
  if (file.size > MAX_BYTES) return "The file is larger than the 5 MB limit";
  return "";
}

function interleave(cases) {
  const byLabel = { vaginal: [], cesarean: [] };
  for (const item of cases) {
    const key = item.label === "Cesarean delivery" ? "cesarean" : "vaginal";
    byLabel[key].push(item);
  }
  const out = [];
  const max = Math.max(byLabel.vaginal.length, byLabel.cesarean.length);
  for (let i = 0; i < max; i += 1) {
    if (byLabel.vaginal[i]) out.push(byLabel.vaginal[i]);
    if (byLabel.cesarean[i]) out.push(byLabel.cesarean[i]);
  }
  return out;
}

function riskCount(clinical) {
  return ["diabetes", "hypertension", "preeclampsia", "liquor_praecox", "pyrexia", "meconium", "induced"].filter(
    (key) => clinical[key]
  ).length;
}

const ASSESS_MODES = ["multimodal", "bilstm", "clinical"];

export default function Assess() {
  useDocumentTitle("New assessment");
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedMode = searchParams.get("model");
  const [mode, setMode] = useState(
    ASSESS_MODES.includes(requestedMode) ? requestedMode : "multimodal"
  );
  const [metrics, setMetrics] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [errors, setErrors] = useState({});
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState("");
  const [fs, setFs] = useState("4");
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [demoCases, setDemoCases] = useState([]);
  const [showAllDemos, setShowAllDemos] = useState(false);
  const [demoError, setDemoError] = useState("");
  const [demoBusy, setDemoBusy] = useState("");
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    getDemoCases()
      .then((cases) => setDemoCases(interleave(cases)))
      .catch((err) => setDemoError(err.message));
    getMetrics()
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, []);

  const activeSummary = (() => {
    const key = mode === "bilstm" ? "bilstm_dl" : mode === "multimodal" ? "multimodal_dl" : "clinical_dl";
    const entry = metrics?.models?.[key];
    if (!entry) return null;
    const operating = entry.at_tuned_threshold || entry.calibrated || entry.raw;
    return {
      auc: operating.auroc,
      ap: operating.auprc,
      sensitivity: operating.recall_sensitivity,
      specificity: operating.specificity,
      threshold: entry.tuned_threshold,
    };
  })();

  const storeAndGo = (payload, modeUsed, extra = {}) => {
    const entry = {
      id: window.crypto?.randomUUID ? window.crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
      timestamp: new Date().toISOString(),
      mode: modeUsed,
      ...extra,
      result: payload
    };
    try {
      window.sessionStorage.setItem("obsight.lastResult", JSON.stringify(entry));
    } catch {
      // Session storage is optional; navigation still carries React state below.
    }
    saveHistory(entry);
    navigate("/result", { state: { entry } });
  };

  const chooseFile = (candidate) => {
    const message = validateFile(candidate);
    setFileError(message);
    setFile(candidate);
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    const candidate = event.dataTransfer?.files?.[0];
    if (candidate) chooseFile(candidate);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setSubmitError("");
    const nextErrors = validateClinical(form);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length) {
      const first = document.querySelector(".field-invalid input, .field-invalid select");
      first?.focus();
      return;
    }
    if (mode !== "clinical") {
      const message = validateFile(file);
      setFileError(message);
      if (message) return;
    }

    setBusy(true);
    try {
      const clinical = toPayload(form);
      if (mode === "clinical") {
        const result = await predictClinical(clinical);
        storeAndGo(result, "clinical");
      } else {
        const data = new FormData();
        for (const [key, value] of Object.entries(clinical)) {
          data.append(key, String(value));
        }
        data.append("fs", String(Number(fs) || 4));
        data.append("model", mode);
        data.append("file", file);
        const result = await predictWithCTG(data);
        storeAndGo(result, mode);
      }
    } catch (err) {
      setSubmitError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const visibleDemoCases = showAllDemos
    ? demoCases
    : demoCases.slice(0, DEMO_PREVIEW_COUNT);

  const runDemo = async (recordId) => {
    setDemoError("");
    setDemoBusy(recordId);
    try {
      const result = await predictDemo(recordId, mode === "clinical" ? "multimodal" : mode);
      storeAndGo(result, "demo", { demoId: recordId, model: mode });
    } catch (err) {
      setDemoError(err.message);
    } finally {
      setDemoBusy("");
    }
  };

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <span className="eyebrow">Assessment</span>
          <h1>Run a research prediction</h1>
          <p>
            Enter the maternal variables available during labour and, optionally, attach
            the most recent CTG recording. Nothing is written to a database by this
            prototype.
          </p>
        </div>
        <Badge tone="amber">Research use only</Badge>
      </header>

      <div className="assess-layout">
        <form className="card assess-form" onSubmit={handleSubmit} noValidate>
          <div className="segmented" role="tablist" aria-label="Prediction model">
            <button
              type="button"
              role="tab"
              aria-selected={mode === "multimodal"}
              className={mode === "multimodal" ? "segment active" : "segment"}
              onClick={() => setMode("multimodal")}
            >
              Multimodal + CTG
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "bilstm"}
              className={mode === "bilstm" ? "segment active" : "segment"}
              onClick={() => setMode("bilstm")}
            >
              CNN-BiLSTM
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "clinical"}
              className={mode === "clinical" ? "segment active" : "segment"}
              onClick={() => setMode("clinical")}
            >
              Clinical only
            </button>
          </div>

          <p className="mode-note">{MODE_NOTES[mode]}</p>

          {activeSummary ? (
            <div className="model-metrics" aria-label="Selected model held-out performance">
              <span>
                <strong>{activeSummary.auc.toFixed(3)}</strong> ROC-AUC
              </span>
              <span>
                <strong>{activeSummary.ap.toFixed(3)}</strong> PR-AUC
              </span>
              <span>
                <strong>{activeSummary.sensitivity.toFixed(2)}</strong> sensitivity
              </span>
              <span>
                <strong>{activeSummary.specificity.toFixed(2)}</strong> specificity
              </span>
              <span>
                <strong>{activeSummary.threshold.toFixed(2)}</strong> threshold
              </span>
            </div>
          ) : null}

          <ClinicalForm form={form} onChange={setForm} errors={errors} disabled={busy} />

          {mode !== "clinical" ? (
            <fieldset className="form-section" disabled={busy}>
              <legend>Cardiotocography recording</legend>
              <div
                className={`dropzone${dragActive ? " dragging" : ""}${fileError ? " invalid" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={onDrop}
              >
                <input
                  id="ctg-file"
                  type="file"
                  accept=".csv,.json,text/csv,application/json"
                  onChange={(event) => chooseFile(event.target.files?.[0] || null)}
                  className="sr-only"
                />
                <label htmlFor="ctg-file" className="dropzone-label">
                  <strong>{file ? file.name : "Choose a CTG file"}</strong>
                  <span>
                    {file
                      ? `${(file.size / 1024).toFixed(1)} KB · tap to replace`
                      : "CSV with fhr,uc columns or JSON with fhr,uc,fs · up to 5 MB"}
                  </span>
                </label>
              </div>
              {fileError ? <span className="field-error">{fileError}</span> : null}
              <div className="file-options">
                <label htmlFor="fs">
                  Sampling frequency (Hz)
                  <input
                    id="fs"
                    type="number"
                    min="1"
                    max="100"
                    step="0.1"
                    value={fs}
                    onChange={(event) => setFs(event.target.value)}
                    disabled={busy}
                  />
                </label>
                <div className="sample-links">
                  <span>Try a file:</span>
                  <a href={sampleUrl("sample_ctg.csv")} download>
                    <IconDownload size={15} /> real record CSV
                  </a>
                  <a href={sampleUrl("sample_ctg.json")} download>
                    <IconDownload size={15} /> JSON
                  </a>
                  <a href={sampleUrl("synthetic_ctg.csv")} download>
                    <IconDownload size={15} /> synthetic CSV
                  </a>
                </div>
              </div>
              <p className="section-note">
                The sampling-frequency field is ignored when the CSV contains a time
                column or an <code># fs=</code> comment. Raw WFDB pairs are not accepted.
              </p>
            </fieldset>
          ) : null}

          {submitError ? <ErrorNote message={submitError} /> : null}

          <div className="form-actions">
            <button type="submit" className="btn btn-primary btn-large" disabled={busy}>
              {busy ? <Spinner label="Scoring" /> : "Run research prediction"}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => {
                setForm(SAMPLE_FORM);
                setErrors({});
                setFile(null);
                setFileError("");
              }}
              disabled={busy}
            >
              Fill with example values
            </button>
          </div>
          <p className="privacy-note">
            <IconAlert size={16} /> Do not enter names, record numbers or any other
            identifiable patient information. Uploaded files are processed in memory
            and are not stored by the API.
          </p>
        </form>

        <aside className="assess-aside">
          {mode !== "clinical" ? (
          <section className="card">
            <h2>Held-out demonstration cases</h2>
            <p className="section-note">
              {demoCases.length
                ? `${demoCases.length} public, de-identified records from the fixed test split. `
                : "Public, de-identified records from the fixed test split. "}
              Each case is scored with the currently selected model
              ({mode === "bilstm" ? "CNN-BiLSTM" : "multimodal"}); the true outcome is
              revealed only on the result page, for teaching.
            </p>
            {demoError ? <ErrorNote message={demoError} /> : null}
            {!demoCases.length && !demoError ? <Spinner label="Loading cases" /> : null}
            <div className="demo-list">
              {visibleDemoCases.map((item) => (
                <article className="demo-card" key={item.record_id}>
                  <div className="demo-card-head">
                    <span className="mono">Record {item.record_id}</span>
                    <span className="demo-meta">
                      {item.clinical.age} y · {item.clinical.gest_weeks} wk
                    </span>
                  </div>
                  <p>
                    Para {item.clinical.parity} ·{" "}
                    {item.clinical.presentation === 1 ? "vertex" : item.clinical.presentation === 2 ? "breech" : "abnormal presentation"} ·{" "}
                    {riskCount(item.clinical)} risk factor{riskCount(item.clinical) === 1 ? "" : "s"}
                  </p>
                  <button
                    type="button"
                    className="btn btn-secondary btn-small"
                    onClick={() => runDemo(item.record_id)}
                    disabled={Boolean(demoBusy)}
                  >
                    {demoBusy === item.record_id ? <Spinner label="Scoring" /> : <>Run case <IconChevron size={16} /></>}
                  </button>
                </article>
              ))}
            </div>
            {demoCases.length > DEMO_PREVIEW_COUNT ? (
              <button
                type="button"
                className="btn btn-ghost btn-small demo-more"
                onClick={() => setShowAllDemos((value) => !value)}
              >
                {showAllDemos
                  ? "Show fewer cases"
                  : `Show ${demoCases.length - DEMO_PREVIEW_COUNT} more cases`}
              </button>
            ) : null}
          </section>
          ) : null}

          <section className="callout callout-info">
            <h2>Before you interpret a score</h2>
            <ul>
              <li>The score is a statistical estimate from 552 research recordings.</li>
              <li>The test split contains only 7 cesarean deliveries.</li>
              <li>Cesarean recordings came from a separate subset, so recording quality may act as a shortcut.</li>
              <li>Attributions describe the model, not physiology or causality.</li>
            </ul>
          </section>
        </aside>
      </div>
    </div>
  );
}
