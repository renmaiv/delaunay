import type { AnalysisResponse, AnalysisResult } from "./types";

// Backend base URL. Empty in local dev (the Vite proxy forwards /api to
// localhost:8000); set VITE_API_BASE at build time (e.g. the Render URL) for a
// split frontend/backend deploy. Trailing slash trimmed so `${API_BASE}/api/x`
// is always well-formed.
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

const POLL_INTERVAL_MS = 1500;
const OVERALL_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Extract a user-facing error message from a non-2xx response body. */
async function errorFromResponse(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string; error?: string };
    if (body && typeof body.detail === "string") return body.detail;
    if (body && typeof body.error === "string") return body.error;
  } catch {
    // fall through to status text
  }
  return res.statusText || `HTTP ${res.status}`;
}

const NO_BACKEND_MESSAGE =
  "Couldn't reach the analysis backend — the server returned a page instead of " +
  "data. The API isn't deployed or the site isn't pointed at it " +
  "(set VITE_API_BASE to your backend URL). The example below still works offline.";

/** Parse a response as JSON, or throw a clear "no backend" error when the
 * server returned HTML (e.g. the SPA itself, meaning /api reached no backend). */
async function jsonOrThrow(res: Response): Promise<AnalysisResponse> {
  const contentType = res.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    throw new Error(NO_BACKEND_MESSAGE);
  }
  try {
    return (await res.json()) as AnalysisResponse;
  } catch {
    throw new Error(NO_BACKEND_MESSAGE);
  }
}

/**
 * Upload a file for analysis and resolve with the completed AnalysisResult.
 *
 * Protocol (docs/sop/contracts/api_contract.md § "Client polling protocol"):
 *   1. POST /api/analyze with FormData field `file`.
 *   2. If the response is already `completed`, return its `result`.
 *   3. Otherwise poll GET /api/analysis/{id} every 1500 ms, invoking
 *      `onStatus` per poll, until `completed` (return) or `failed` (throw),
 *      with a 5 minute overall timeout.
 */
export async function analyzeFile(
  file: File,
  onStatus?: (r: AnalysisResponse) => void,
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: form });
  } catch {
    // network error / CORS / DNS — no backend reachable at all
    throw new Error(NO_BACKEND_MESSAGE);
  }
  if (!res.ok) {
    throw new Error(await errorFromResponse(res));
  }
  const initial = await jsonOrThrow(res);
  onStatus?.(initial);

  if (initial.status === "completed") {
    if (!initial.result) throw new Error("Analysis completed without a result.");
    return initial.result;
  }
  if (initial.status === "failed") {
    throw new Error(initial.error ?? "Analysis failed.");
  }

  const deadline = Date.now() + OVERALL_TIMEOUT_MS;
  const id = initial.analysis_id;

  while (Date.now() < deadline) {
    await delay(POLL_INTERVAL_MS);

    const pollRes = await fetch(`${API_BASE}/api/analysis/${encodeURIComponent(id)}`);
    if (!pollRes.ok) {
      throw new Error(await errorFromResponse(pollRes));
    }
    const status = await jsonOrThrow(pollRes);
    onStatus?.(status);

    if (status.status === "completed") {
      if (!status.result) {
        throw new Error("Analysis completed without a result.");
      }
      return status.result;
    }
    if (status.status === "failed") {
      throw new Error(status.error ?? "Analysis failed.");
    }
  }

  throw new Error("Analysis timed out after 5 minutes.");
}

/** GET /api/taxonomy — returns the raw shared/taxonomy.json contents. */
export async function fetchTaxonomy(): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/taxonomy`);
  if (!res.ok) {
    throw new Error(await errorFromResponse(res));
  }
  return res.json();
}
