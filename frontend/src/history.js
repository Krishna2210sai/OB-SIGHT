const KEY = "obsight.history.v1";
const LIMIT = 25;

export function loadHistory() {
  try {
    const raw = window.localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveHistory(entry) {
  const existing = loadHistory();
  const next = [entry, ...existing].slice(0, LIMIT);
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Storage can be unavailable in private mode; history is optional.
  }
  return next;
}

export function deleteHistoryEntry(id) {
  const next = loadHistory().filter((item) => item.id !== id);
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // ignore
  }
  return next;
}

export function clearHistory() {
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
  return [];
}
