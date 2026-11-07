import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getMetrics, getModelInfo } from "../api.js";
import MetricsTable from "../components/MetricsTable.jsx";
import { ErrorNote, LoadingBlock, useDocumentTitle } from "../components/UI.jsx";
import { useEffect } from "react";
import { IconAssess, IconChevron, IconInfo } from "../components/Icons.jsx";

export default function Dashboard() {
  useDocumentTitle("Dashboard");
  const [metrics, setMetrics] = useState(null);
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    setError("");
    Promise.all([getMetrics(), getModelInfo()])
      .then(([m, i]) => {
        setMetrics(m);
        setInfo(i);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const dataset = info?.dataset;
  const stats = useMemo(
    () => [
      { label: "Public recordings", value: dataset?.n_records ?? "—", hint: "CTU-CHB v1.0.0" },
      { label: "Cesarean deliveries", value: dataset?.n_cesarean ?? "—", hint: `${dataset ? ((dataset.n_cesarean / dataset.n_records) * 100).toFixed(1) : "—"}% prevalence` },
      { label: "Held-out test cases", value: metrics?.test_set?.n ?? "—", hint: `${metrics?.test_set?.n_positive ?? "—"} cesarean` },
      { label: "Input window", value: dataset ? `${dataset.window_minutes} min` : "—", hint: dataset ? `${dataset.sampling_frequency_hz} Hz · 4 channels` : "" }
    ],
    [dataset, metrics]
  );

  return (
    <div className="page">
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Clinical decision-support research</span>
          <h1>
            Predicting delivery outcome from intrapartum CTG and maternal data
          </h1>
          <p>
            A multimodal deep-learning prototype that estimates whether a labour will
            end vaginally or by cesarean section, using the most recent 30 minutes of
            cardiotocography and routinely recorded maternal variables.
          </p>
          <div className="hero-actions">
            <Link className="btn btn-primary" to="/assess">
              <IconAssess size={19} />
              New assessment
            </Link>
            <Link className="btn btn-secondary" to="/model">
              <IconInfo size={18} />
              Read the model card
            </Link>
          </div>
          <p className="hero-footnote">
            For teaching and research only. The system is not validated for patient care.
          </p>
        </div>
        <div className="hero-panel" aria-label="Model performance summary">
          <div className="hero-panel-head">
            <span>Flagship model</span>
            <span className="pill pill-amber">research</span>
          </div>
          <dl className="hero-metrics">
            <div>
              <dt>ROC-AUC</dt>
              <dd>{metrics ? (metrics.models.multimodal_dl.at_tuned_threshold?.auroc ?? metrics.models.multimodal_dl.raw.auroc).toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>PR-AUC</dt>
              <dd>{metrics ? (metrics.models.multimodal_dl.at_tuned_threshold?.auprc ?? metrics.models.multimodal_dl.raw.auprc).toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>Sensitivity</dt>
              <dd>{metrics ? (metrics.models.multimodal_dl.at_tuned_threshold?.recall_sensitivity ?? metrics.models.multimodal_dl.raw.recall_sensitivity).toFixed(2) : "—"}</dd>
            </div>
            <div>
              <dt>Specificity</dt>
              <dd>{metrics ? (metrics.models.multimodal_dl.at_tuned_threshold?.specificity ?? metrics.models.multimodal_dl.raw.specificity).toFixed(2) : "—"}</dd>
            </div>
          </dl>
          <p className="hero-panel-note">
            Held-out test split · 7 cesarean vs 76 vaginal records. Not a measure of
            clinical safety or effectiveness.
          </p>
        </div>
      </section>

      {error ? <ErrorNote message={error} onRetry={load} /> : null}
      {loading && !metrics ? <LoadingBlock label="Loading evaluation summary" /> : null}

      <section className="stat-grid" aria-label="Dataset and evaluation facts">
        {stats.map((stat) => (
          <div className="stat-card" key={stat.label}>
            <span className="stat-value">{stat.value}</span>
            <span className="stat-label">{stat.label}</span>
            <span className="stat-hint">{stat.hint}</span>
          </div>
        ))}
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <h2>How it works</h2>
            <p>Three steps, one deliberately small model pipeline.</p>
          </div>
        </div>
        <ol className="steps">
          <li>
            <span className="step-num">1</span>
            <h3>Enter maternal data</h3>
            <p>Age, gravidity, parity, gestational age, presentation and seven binary risk factors recorded during labour.</p>
          </li>
          <li>
            <span className="step-num">2</span>
            <h3>Attach a CTG recording</h3>
            <p>Upload a CSV or JSON file at 4 Hz. The most recent 30 minutes are cleaned, masked and downsampled to 1 Hz.</p>
          </li>
          <li>
            <span className="step-num">3</span>
            <h3>Read the research score</h3>
            <p>A calibrated probability, a fixed threshold, and gradient-based attributions that describe model behaviour.</p>
          </li>
        </ol>
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <h2>Model comparison</h2>
            <p>Deep learning against simple clinical and CTG baselines on the same fixed test split.</p>
          </div>
          <Link className="link-arrow" to="/model">
            Full model card <IconChevron size={16} />
          </Link>
        </div>
        {metrics ? <MetricsTable models={metrics.models} /> : null}
      </section>

      <section className="callout callout-warn">
        <h2>What this system does not do</h2>
        <ul>
          <li>It does not interpret CTG traces like an obstetric expert.</li>
          <li>It does not recommend, time or justify a cesarean section.</li>
          <li>It does not estimate risk for an individual patient; the test cohort has only seven cesarean cases.</li>
          <li>It has not been externally validated, and it is not a medical device.</li>
        </ul>
      </section>
    </div>
  );
}
