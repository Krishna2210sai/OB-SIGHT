import { fixed } from "./UI.jsx";

function summarize(entry) {
  const operating = entry.at_tuned_threshold || entry.calibrated || entry.raw;
  return {
    name: entry.display_name,
    type: entry.type,
    auc: operating.auroc,
    ap: operating.auprc,
    sens: operating.recall_sensitivity,
    spec: operating.specificity,
    threshold: entry.tuned_threshold,
    ciLow: entry.ci?.auroc?.[0],
    ciHigh: entry.ci?.auroc?.[1]
  };
}

export default function MetricsTable({ models, highlight = "multimodal_dl" }) {
  if (!models) return null;
  const rows = Object.entries(models).map(([key, entry]) => ({ key, ...summarize(entry) }));
  const deep = rows.filter((row) => row.type === "deep_learning");
  const baselines = rows.filter((row) => row.type !== "deep_learning");

  return (
    <div className="metrics-table-wrap">
      <table className="metrics-table">
        <caption className="sr-only">
          Held-out test performance at the validation-selected operating point
        </caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">ROC-AUC (95% CI)</th>
            <th scope="col">PR-AUC</th>
            <th scope="col">Sens.</th>
            <th scope="col">Spec.</th>
            <th scope="col">Thr.</th>
          </tr>
        </thead>
        <tbody>
          <tr className="table-group">
            <th colSpan={6} scope="colgroup">Deep learning</th>
          </tr>
          {deep.map((row) => (
            <Row key={row.key} row={row} highlight={row.key === highlight} />
          ))}
          <tr className="table-group">
            <th colSpan={6} scope="colgroup">Reference baselines</th>
          </tr>
          {baselines.map((row) => (
            <Row key={row.key} row={row} highlight={false} />
          ))}
        </tbody>
      </table>
      <p className="table-note">
        Threshold-free ranking is shown as ROC-AUC and PR-AUC. Sensitivity and
        specificity use the Youden-J threshold selected on the validation split only.
        With 7 cesarean cases in the test set, confidence intervals are wide.
      </p>
    </div>
  );
}

function Row({ row, highlight }) {
  return (
    <tr className={highlight ? "row-highlight" : undefined}>
      <th scope="row">
        <span className="model-name">{row.name}</span>
        {highlight ? <span className="pill pill-teal">flagship</span> : null}
      </th>
      <td>
        <span className="metric-strong">{fixed(row.auc)}</span>
        {row.ciLow !== undefined ? (
          <span className="metric-ci">
            {fixed(row.ciLow, 2)}–{fixed(row.ciHigh, 2)}
          </span>
        ) : null}
      </td>
      <td>{fixed(row.ap)}</td>
      <td>{fixed(row.sens, 2)}</td>
      <td>{fixed(row.spec, 2)}</td>
      <td>{fixed(row.threshold, 2)}</td>
    </tr>
  );
}
