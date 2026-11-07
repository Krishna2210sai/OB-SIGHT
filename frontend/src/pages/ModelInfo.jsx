import { useEffect, useState } from "react";
import { getMetrics, getModelInfo } from "../api.js";
import MetricsTable from "../components/MetricsTable.jsx";
import { Badge, ErrorNote, LoadingBlock, fixed, useDocumentTitle } from "../components/UI.jsx";
import { IconAlert } from "../components/Icons.jsx";

const VARIANT_ORDER = ["multimodal", "signal", "bilstm", "clinical"];

export default function ModelInfo() {
  useDocumentTitle("Model card");
  const [card, setCard] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    setError("");
    Promise.all([getModelInfo(), getMetrics()])
      .then(([info, metricPayload]) => {
        setCard(info);
        setMetrics(metricPayload);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  if (loading && !card) {
    return (
      <div className="page">
        <LoadingBlock label="Loading model card" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <ErrorNote message={error} onRetry={load} />
      </div>
    );
  }

  if (!card) return null;
  const dataset = card.dataset;
  const evaluation = card.evaluation;
  const cv = evaluation.cross_validation;

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <span className="eyebrow">Model card · v{card.version}</span>
          <h1>{card.project}</h1>
          <p>{card.task}</p>
        </div>
        <Badge tone="amber">Not a medical device</Badge>
      </header>

      <section className="card callout-inline">
        <IconAlert size={22} />
        <p>{card.disclaimer}</p>
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <h2>At a glance</h2>
            <p>What the model predicts, when, and from which data.</p>
          </div>
        </div>
        <dl className="info-grid">
          <div>
            <dt>Target</dt>
            <dd>{card.target}</dd>
          </div>
          <div>
            <dt>Prediction time</dt>
            <dd>{card.prediction_time}</dd>
          </div>
          <div>
            <dt>Dataset</dt>
            <dd>{dataset.name}</dd>
            <p className="info-sub">{dataset.source}</p>
          </div>
          <div>
            <dt>Records</dt>
            <dd>
              {dataset.n_records} total · {dataset.n_vaginal} vaginal · {dataset.n_cesarean} cesarean
            </dd>
            <p className="info-sub">
              Train {dataset.splits.train["1"]}/{dataset.splits.train["0"]} · validation{" "}
              {dataset.splits.val["1"]}/{dataset.splits.val["0"]} · test {dataset.splits.test["1"]}/
              {dataset.splits.test["0"]} (cesarean/vaginal)
            </p>
          </div>
          <div>
            <dt>Inputs</dt>
            <dd>
              {dataset.window_minutes}-minute CTG window at {dataset.sampling_frequency_hz} Hz
              (FHR, UC, two quality masks) plus {dataset.clinical_features.length} clinical features
            </dd>
          </div>
          <div>
            <dt>Identifier</dt>
            <dd className="mono">{dataset.doi}</dd>
          </div>
        </dl>
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <h2>Evaluation on the held-out test set</h2>
            <p>
              {evaluation.test_set.n} records · {evaluation.test_set.n_positive} cesarean ·{" "}
              prevalence {(evaluation.test_set.prevalence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        {metrics ? <MetricsTable models={metrics.models} /> : null}
        <div className="callout callout-info">
          <h3>Operating-point policy</h3>
          <p>{evaluation.threshold_policy}</p>
          <p>
            Best ROC-AUC on this split: <strong>{evaluation.best_test_auroc.model}</strong> (
            {fixed(evaluation.best_test_auroc.auroc)}). Ranking is unstable with seven events.
          </p>
        </div>
        {evaluation.paired_comparison?.bilstm_vs_signal ? (
          <div className="callout">
            <h3>CNN-BiLSTM versus 1D CNN</h3>
            <p>
              Paired AUC difference:{" "}
              <strong>
                {evaluation.paired_comparison.bilstm_vs_signal.difference >= 0 ? "+" : ""}
                {fixed(evaluation.paired_comparison.bilstm_vs_signal.difference)}
              </strong>{" "}
              (95% bootstrap CI{" "}
              {evaluation.paired_comparison.bilstm_vs_signal.ci_low >= 0 ? "+" : ""}
              {fixed(evaluation.paired_comparison.bilstm_vs_signal.ci_low)} to{" "}
              {evaluation.paired_comparison.bilstm_vs_signal.ci_high >= 0 ? "+" : ""}
              {fixed(evaluation.paired_comparison.bilstm_vs_signal.ci_high)}).
            </p>
            <p>{evaluation.paired_comparison.bilstm_vs_signal.interpretation}</p>
          </div>
        ) : null}
      </section>

      {cv ? (
        <section className="section">
          <div className="section-head">
            <div>
              <h2>Five-fold cross-validation</h2>
              <p>Pooled out-of-fold estimates across all 552 records. Early stopping used each validation fold.</p>
            </div>
          </div>
          <div className="table-scroll">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th scope="col">Variant</th>
                  <th scope="col">Pooled ROC-AUC</th>
                  <th scope="col">Pooled PR-AUC</th>
                  <th scope="col">Mean ROC-AUC ± SD</th>
                  <th scope="col">Mean PR-AUC ± SD</th>
                </tr>
              </thead>
              <tbody>
                {VARIANT_ORDER.filter((variant) => cv[variant]).map((variant) => {
                  const row = cv[variant];
                  return (
                    <tr key={variant}>
                      <th scope="row">{card.architecture[variant]?.name || variant}</th>
                      <td>{fixed(row.pooled_auroc)}</td>
                      <td>{fixed(row.pooled_auprc)}</td>
                      <td>
                        {fixed(row.mean_auroc)} ± {fixed(row.std_auroc)}
                      </td>
                      <td>
                        {fixed(row.mean_auprc)} ± {fixed(row.std_auprc)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="section">
        <div className="section-head">
          <div>
            <h2>Architecture</h2>
            <p>Four deliberately small PyTorch models, trained from scratch.</p>
          </div>
        </div>
        <div className="architecture-grid">
          {VARIANT_ORDER.map((variant) => {
            const model = card.architecture[variant];
            if (!model) return null;
            return (
              <article className="card architecture-card" key={variant}>
                <div className="card-head">
                  <h3>{model.name}</h3>
                  {model.parameters ? <span className="pill">{model.parameters.toLocaleString()} params</span> : null}
                </div>
                <p className="arch-input">{model.inputs}</p>
                <p>{model.description}</p>
                {model.best_epoch ? (
                  <p className="info-sub">
                    Best epoch {model.best_epoch} · selected on validation loss
                  </p>
                ) : null}
              </article>
            );
          })}
        </div>
        <div className="card training-card">
          <h3>Training and calibration</h3>
          <dl className="training-list">
            <div>
              <dt>Loss</dt>
              <dd>{card.architecture.training.loss}</dd>
            </div>
            <div>
              <dt>Optimiser</dt>
              <dd>{card.architecture.training.optimiser}</dd>
            </div>
            <div>
              <dt>Regularisation</dt>
              <dd>{card.architecture.training.regularisation}</dd>
            </div>
            <div>
              <dt>Model selection</dt>
              <dd>{card.architecture.training.model_selection}</dd>
            </div>
            <div>
              <dt>Calibration</dt>
              <dd>{card.architecture.training.calibration}</dd>
            </div>
            <div>
              <dt>Threshold</dt>
              <dd>{card.architecture.training.threshold_selection}</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <h2>Limitations</h2>
            <p>Read these before interpreting any output.</p>
          </div>
        </div>
        <ul className="limitation-list">
          {card.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="section two-col">
        <div className="card">
          <h2>Intended use</h2>
          <ul className="plain-list">
            {card.intended_use.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="card forbidden-card">
          <h2>Prohibited use</h2>
          <ul className="plain-list">
            {card.prohibited_use.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <details className="card details-card">
        <summary>Machine-readable model card (JSON)</summary>
        <pre>{JSON.stringify(card, null, 2)}</pre>
      </details>
    </div>
  );
}
