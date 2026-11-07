import { fixed } from "./UI.jsx";

const BINARY_FEATURES = new Set([
  "diabetes",
  "hypertension",
  "preeclampsia",
  "liquor_praecox",
  "pyrexia",
  "meconium",
  "induced"
]);

function formatValue(item) {
  if (item.input_value === null || item.input_value === undefined || Number.isNaN(Number(item.input_value))) {
    return "—";
  }
  const value = Number(item.input_value);
  if (item.feature === "presentation") {
    return value === 1 ? "Vertex" : value === 2 ? "Breech" : "Other";
  }
  if (BINARY_FEATURES.has(item.feature)) {
    return value === 1 ? "Yes" : "No";
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

export function ClinicalContributions({ items }) {
  if (!items?.length) return null;
  const max = Math.max(...items.map((item) => Math.abs(item.contribution)), 1e-9);
  return (
    <div className="contributions">
      <div className="contribution-head" aria-hidden="true">
        <span>toward vaginal</span>
        <span>toward cesarean</span>
      </div>
      {items.map((item) => {
        const width = (Math.abs(item.contribution) / max) * 50;
        const positive = item.contribution >= 0;
        return (
          <div className="contribution-row" key={item.feature}>
            <div className="contribution-label">
              <span>{item.display_name}</span>
              <span className="contribution-input">{formatValue(item)}</span>
            </div>
            <div className="contribution-track">
              <span className="contribution-axis" aria-hidden="true" />
              <span
                className={`contribution-bar ${positive ? "positive" : "negative"}`}
                style={positive ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
              />
            </div>
            <span className={`contribution-value ${positive ? "positive" : "negative"}`}>
              {positive ? "+" : ""}
              {fixed(item.contribution, 3)}
            </span>
          </div>
        );
      })}
      <p className="chart-note">
        Gradient × input, in log-odds units. These values describe model behaviour and
        are not causal or physiological explanations.
      </p>
    </div>
  );
}

export function SignalContributions({ items }) {
  if (!items?.length) return null;
  const max = Math.max(...items.map((item) => item.importance), 1e-9);
  return (
    <div className="signal-importance">
      {items.map((item) => (
        <div className="signal-row" key={item.window}>
          <span className="signal-label">{item.window}</span>
          <div className="signal-track">
            <span className="signal-bar" style={{ width: `${(item.importance / max) * 100}%` }} />
          </div>
          <span className="signal-value">{fixed(item.importance, 2)}</span>
        </div>
      ))}
      <p className="chart-note">
        Absolute gradient saliency averaged over six five-minute windows, normalised
        within this recording. Not a clinical interpretation of the trace.
      </p>
    </div>
  );
}
