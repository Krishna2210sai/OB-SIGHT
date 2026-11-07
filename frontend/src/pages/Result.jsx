import { useMemo, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ClinicalContributions, SignalContributions } from "../components/Contributions.jsx";
import ProbabilityMeter from "../components/ProbabilityMeter.jsx";
import { Badge, useDocumentTitle } from "../components/UI.jsx";
import { IconAlert, IconCheck, IconDownload, IconRefresh } from "../components/Icons.jsx";

const MODEL_LABELS = {
  multimodal: "Multimodal CNN + clinical MLP",
  clinical: "Clinical-only MLP",
  signal: "CTG-only 1D CNN",
  bilstm: "CTG-only CNN-BiLSTM"
};

function readEntry(location) {
  if (location.state?.entry) return location.state.entry;
  try {
    const raw = window.sessionStorage.getItem("obsight.lastResult");
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function downloadJson(entry) {
  const blob = new Blob([JSON.stringify(entry, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `obsight-result-${entry.timestamp?.replace(/[:.]/g, "-") || "export"}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function summaryText(entry) {
  const r = entry.result;
  return [
    "OB-SIGHT research prototype — NOT FOR CLINICAL USE",
    `Model: ${MODEL_LABELS[r.model_variant] || r.model_variant}`,
    `Model score: ${(r.cesarean_probability * 100).toFixed(1)}% (calibrated cesarean)`,
    `Decision threshold: ${r.decision_threshold}`,
    `Model classification: ${r.prediction}`,
    entry.demoId ? `Demonstration record: ${entry.demoId} (true outcome: ${r.true_label})` : null,
    "This is a statistical research output, not a diagnosis or treatment recommendation."
  ]
    .filter(Boolean)
    .join("\n");
}

export default function Result() {
  useDocumentTitle("Result");
  const location = useLocation();
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);
  const entry = useMemo(() => readEntry(location), [location]);

  if (!entry?.result) {
    return (
      <div className="page narrow">
        <div className="empty-state card">
          <h1>No result to display</h1>
          <p>Run an assessment first, or open one of the demonstration cases.</p>
          <Link className="btn btn-primary" to="/assess">
            Go to assessment
          </Link>
        </div>
      </div>
    );
  }

  const r = entry.result;
  const cesarean = r.predicted_class === 1;
  const truthMatch = entry.demoId && r.true_label ? r.true_label === r.prediction : null;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(summaryText(entry));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="page result-page">
      <div className="result-topbar">
        <div>
          <span className="eyebrow">{entry.demoId ? `Demonstration record ${entry.demoId}` : "Assessment result"}</span>
          <h1>Research prediction</h1>
        </div>
        <div className="result-top-actions">
          <Link className="btn btn-ghost btn-small" to="/assess">
            <IconRefresh size={16} /> New assessment
          </Link>
          <button type="button" className="btn btn-ghost btn-small" onClick={() => downloadJson(entry)}>
            <IconDownload size={16} /> JSON
          </button>
        </div>
      </div>

      {entry.demoId ? (
        <div className={`truth-banner ${truthMatch ? "match" : "miss"}`} role="note">
          <div>
            <strong>Demonstration only.</strong> The true outcome was{" "}
            <strong>{r.true_label}</strong>. The model said <strong>{r.prediction}</strong>.
          </div>
          <Badge tone={truthMatch ? "teal" : "amber"}>{truthMatch ? "Match" : "Mismatch"}</Badge>
        </div>
      ) : null}

      <section className={`result-summary card ${cesarean ? "tone-cesarean" : "tone-vaginal"}`}>
        <div className="result-verdict">
          <span className="verdict-label">Model classification</span>
          <div className="verdict-row">
            <h2>{r.prediction}</h2>
            <Badge tone={cesarean ? "amber" : "teal"}>{cesarean ? "Above threshold" : "Below threshold"}</Badge>
          </div>
          <p className="verdict-note">
            {cesarean
              ? "The calibrated score is at or above the validation-selected threshold."
              : "The calibrated score is below the validation-selected threshold."}{" "}
            This is a statistical classification, not a clinical recommendation.
          </p>
        </div>
        <ProbabilityMeter probability={r.cesarean_probability} threshold={r.decision_threshold} />
      </section>

      <section className="meta-grid" aria-label="Prediction metadata">
        <div>
          <span className="meta-label">Model</span>
          <span className="meta-value">{MODEL_LABELS[r.model_variant] || r.model_variant}</span>
        </div>
        <div>
          <span className="meta-label">Input mode</span>
          <span className="meta-value">
            {entry.mode === "ctg" ? "Clinical + CTG" : entry.mode === "demo" ? "Held-out demo record" : "Clinical only"}
          </span>
        </div>
        <div>
          <span className="meta-label">Threshold</span>
          <span className="meta-value">{r.decision_threshold} · validation-selected</span>
        </div>
        <div>
          <span className="meta-label">Probability calibration</span>
          <span className="meta-value">{r.calibrated ? "Platt scaling (validation)" : "Uncalibrated"}</span>
        </div>
      </section>

      {r.warnings?.length ? (
        <section className="alert alert-warn" aria-label="Warnings">
          <IconAlert size={20} />
          <ul>
            {r.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <div className="result-columns">
        <section className="card">
          <div className="card-head">
            <h2>Clinical feature contributions</h2>
            <span className="pill">{r.clinical_contributions?.length || 0} shown</span>
          </div>
          {r.clinical_contributions?.length ? (
            <ClinicalContributions items={r.clinical_contributions} />
          ) : (
            <p className="section-note">No attributions were returned for this model.</p>
          )}
        </section>

        {r.signal_contributions?.length ? (
          <section className="card">
            <div className="card-head">
              <h2>CTG window saliency</h2>
              <span className="pill">oldest → newest</span>
            </div>
            <SignalContributions items={r.signal_contributions} />
          </section>
        ) : (
          <section className="card">
            <div className="card-head">
              <h2>CTG window saliency</h2>
            </div>
            <p className="section-note">
              No CTG was used in this assessment, so no signal attribution is available.
            </p>
          </section>
        )}
      </div>

      {r.ctg_summary ? (
        <section className="card">
          <div className="card-head">
            <h2>CTG processing summary</h2>
            <span className="pill">most recent 30 min</span>
          </div>
          <dl className="summary-grid">
            <div>
              <dt>Available duration</dt>
              <dd>{Number(r.ctg_summary.duration_minutes).toFixed(1)} min</dd>
            </div>
            <div>
              <dt>Input sampling frequency</dt>
              <dd>{r.ctg_summary.sampling_frequency_hz} Hz</dd>
            </div>
            <div>
              <dt>FHR missing</dt>
              <dd>{(r.ctg_summary.fhr_missing_fraction * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>UC missing</dt>
              <dd>{(r.ctg_summary.uc_missing_fraction * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>Mean FHR</dt>
              <dd>{r.ctg_summary.mean_fhr_bpm ? `${r.ctg_summary.mean_fhr_bpm.toFixed(0)} bpm` : "—"}</dd>
            </div>
            <div>
              <dt>Window padded</dt>
              <dd>{r.ctg_summary.window_padded ? "Yes" : "No"}</dd>
            </div>
          </dl>
          <p className="section-note">
            Missing fractions refer to the 30-minute model window after artefact screening.
            High missingness materially changes the model input.
          </p>
        </section>
      ) : null}

      <section className="card">
        <div className="card-head">
          <h2>About this output</h2>
          <span className="pill pill-amber">research prototype</span>
        </div>
        <p>{r.disclaimer}</p>
        <ul className="plain-list">
          <li>
            The model was trained on 552 public intrapartum recordings; the held-out test
            set contains seven cesarean and 76 vaginal deliveries.
          </li>
          <li>
            The decision threshold was chosen on the validation split. It trades sensitivity
            against specificity and is unstable with so few events.
          </li>
          <li>
            Cesarean recordings in this database come from a separate subset and differ in
            signal duration and quality. The model may exploit recording artefacts.
          </li>
          <li>
            Feature attributions are gradient-based summaries of model behaviour. They do not
            establish cause, physiological mechanism or clinical relevance.
          </li>
        </ul>
        <div className="form-actions">
          <button type="button" className="btn btn-secondary" onClick={copy}>
            <IconCheck size={17} /> {copied ? "Copied" : "Copy summary"}
          </button>
          <button type="button" className="btn btn-primary" onClick={() => navigate("/assess")}>
            Run another assessment
          </button>
        </div>
      </section>

      <details className="card details-card">
        <summary>Technical details (JSON)</summary>
        <pre>{JSON.stringify(r, null, 2)}</pre>
      </details>
    </div>
  );
}
