import { useEffect } from "react";
import { IconAlert } from "./Icons.jsx";

export function Spinner({ label = "Loading" }) {
  return (
    <span className="spinner-wrap" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{label}…</span>
    </span>
  );
}

export function ErrorNote({ message, onRetry }) {
  return (
    <div className="alert alert-error" role="alert">
      <IconAlert size={20} />
      <div>
        <p>{message}</p>
        {onRetry ? (
          <button type="button" className="btn btn-small btn-ghost" onClick={onRetry}>
            Try again
          </button>
        ) : null}
      </div>
    </div>
  );
}

export function LoadingBlock({ label = "Loading" }) {
  return (
    <div className="card loading-block">
      <Spinner label={label} />
    </div>
  );
}

export function useDocumentTitle(title) {
  useEffect(() => {
    document.title = title ? `${title} · OB-SIGHT` : "OB-SIGHT · Delivery Outcome Research Prototype";
  }, [title]);
}

export function Badge({ tone = "neutral", children }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function pct(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function fixed(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}
