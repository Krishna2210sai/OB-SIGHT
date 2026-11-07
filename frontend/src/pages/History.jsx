import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Badge, useDocumentTitle } from "../components/UI.jsx";
import { loadHistory, clearHistory, deleteHistoryEntry } from "../history.js";
import { IconTrash } from "../components/Icons.jsx";

const MODE_LABELS = {
  multimodal: "Multimodal (Clinical + CTG)",
  bilstm: "CNN-BiLSTM (CTG only)",
  ctg: "Clinical + CTG",
  clinical: "Clinical only",
  demo: "Demo record"
};

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return date.toLocaleString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

export default function History() {
  useDocumentTitle("History");
  const navigate = useNavigate();
  const [entries, setEntries] = useState(loadHistory);

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <span className="eyebrow">Local history</span>
          <h1>Recent assessments</h1>
          <p>
            Stored only in this browser's local storage. Up to 25 entries; no data is
            sent anywhere for storage.
          </p>
        </div>
        {entries.length ? (
          <button
            type="button"
            className="btn btn-ghost btn-small"
            onClick={() => setEntries(clearHistory())}
          >
            <IconTrash size={16} /> Clear all
          </button>
        ) : null}
      </header>

      {!entries.length ? (
        <div className="empty-state card">
          <h2>Nothing here yet</h2>
          <p>Results you generate in this browser will appear here for quick review.</p>
          <Link className="btn btn-primary" to="/assess">
            Run an assessment
          </Link>
        </div>
      ) : (
        <ul className="history-list">
          {entries.map((entry) => {
            const result = entry.result;
            const cesarean = result?.predicted_class === 1;
            const match =
              entry.demoId && result?.true_label ? result.true_label === result.prediction : null;
            return (
              <li key={entry.id}>
                <div className="history-item">
                  <button
                    type="button"
                    className="history-open"
                    onClick={() => navigate("/result", { state: { entry } })}
                  >
                    <div className="history-main">
                      <div className="history-head">
                        <strong>{result?.prediction || "Unknown result"}</strong>
                        <Badge tone={cesarean ? "amber" : "teal"}>
                          {result ? `${(result.cesarean_probability * 100).toFixed(1)}%` : "—"}
                        </Badge>
                        {match !== null ? (
                          <Badge tone={match ? "teal" : "amber"}>{match ? "match" : "mismatch"}</Badge>
                        ) : null}
                      </div>
                      <span className="history-meta">
                        {MODE_LABELS[entry.mode] || entry.mode}
                        {entry.demoId ? ` · record ${entry.demoId}` : ""} ·{" "}
                        {formatDate(entry.timestamp)}
                      </span>
                    </div>
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`Delete result from ${formatDate(entry.timestamp)}`}
                    onClick={() => setEntries(deleteHistoryEntry(entry.id))}
                  >
                    <IconTrash size={17} />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <p className="privacy-note">
        Clearing browser data, using private mode, or switching devices removes this list.
        Exported JSON files contain the values you entered; handle them accordingly.
      </p>
    </div>
  );
}
