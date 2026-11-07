import { fixed } from "./UI.jsx";

export default function ProbabilityMeter({ probability, threshold }) {
  const percent = Math.max(0, Math.min(1, Number(probability))) * 100;
  const thresholdPercent = Math.max(0, Math.min(1, Number(threshold))) * 100;
  return (
    <div className="probability-meter">
      <div className="probability-readout">
        <span className="probability-value">{percent.toFixed(1)}%</span>
        <span className="probability-caption">calibrated cesarean score</span>
      </div>
      <div
        className="meter-track"
        role="img"
        aria-label={`Model score ${percent.toFixed(1)} percent; decision threshold ${(thresholdPercent).toFixed(0)} percent`}
      >
        <div className="meter-fill" style={{ width: `${percent}%` }} />
        <div className="meter-threshold" style={{ left: `${thresholdPercent}%` }}>
          <span>threshold {fixed(threshold, 2)}</span>
        </div>
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
      <p className="meter-note">
        The score is a model output calibrated on 83 validation records. It is not a
        patient-specific risk of cesarean delivery.
      </p>
    </div>
  );
}
