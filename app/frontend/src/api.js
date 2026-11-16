// Thin API client. In dev, Vite proxies /api -> FastAPI :8000.
const base = "";

async function j(path, opts) {
  const r = await fetch(base + path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return r.json();
}

export const api = {
  health: () => j("/api/health"),
  config: () => j("/api/config"),
  status: () => j("/api/status"),
  results: () => j("/api/results"),
  run: (params) =>
    j("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
};
