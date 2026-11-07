const API_BASE = import.meta.env.VITE_API_BASE ?? "";

function formatDetail(payload, status) {
  if (!payload) return `Request failed (HTTP ${status})`;
  const detail = payload.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const where = Array.isArray(item.loc) ? item.loc.filter((x) => x !== "body").join(".") : "";
        return where ? `${where}: ${item.msg}` : item.msg;
      })
      .join("; ");
  }
  return `Request failed (HTTP ${status})`;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error(
      "Could not reach the API. Start it with: uvicorn backend.main:app --port 8000"
    );
  }
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text.slice(0, 300) };
    }
  }
  if (!response.ok) {
    throw new Error(formatDetail(payload, response.status));
  }
  return payload;
}

export const getHealth = () => request("/health");
export const getModelInfo = () => request("/model-info");
export const getMetrics = () => request("/metrics");
export const getDemoCases = () => request("/demo-cases");

export function predictClinical(payload) {
  return request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function predictWithCTG(formData) {
  return request("/predict-ctg", { method: "POST", body: formData });
}

export function predictDemo(recordId, model = "multimodal") {
  const query = model ? `?model=${encodeURIComponent(model)}` : "";
  return request(`/demo/${encodeURIComponent(recordId)}${query}`, { method: "POST" });
}

export const sampleUrl = (name) => `${API_BASE}/api/sample/${name}`;
