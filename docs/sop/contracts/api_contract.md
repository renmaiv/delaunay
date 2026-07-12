# CONTRACT — HTTP API (implemented in WS-D2, consumed by WS-A3/WS-E*)

All response bodies use the types in `contracts/schemas.py` / `contracts/types.ts`.

| Method & path | Request | Responses |
|---|---|---|
| `POST /api/analyze` | multipart form, field `file` (`.json`/`.jsonl`/`.txt`), limits: `analysis.max_upload_bytes` (default 2 MB), `analysis.max_turns` (default 500) | `200` → `AnalysisResponse` with `status:"completed"` and inline `result` (when parsed turn count ≤ `analysis.sync_turn_threshold`, default 30) · `202` → `AnalysisResponse` with `status:"pending"` (job submitted) · `400` unparseable file or too many turns (body `{"detail": "<user-facing message>"}`) · `413` file too large · `422` missing `file` field |
| `GET /api/analysis/{id}` | — | `200` → `AnalysisResponse` (any status) · `404` unknown id |
| `GET /api/taxonomy` | — | `200` → contents of `shared/taxonomy.json` |
| `GET /api/health` | — | `200` → `{"status":"ok","judge":{"provider":"anthropic"\|"mock","model":"..."},"encoders":{"<scorer name>":true\|false}}` |
| `GET /` + any non-`/api` path | — | built SPA from `frontend/dist` (`StaticFiles(html=True)`); if `dist/` missing → `200` JSON `{"detail":"frontend not built; run npm run build in frontend/"}` |

## Client polling protocol (implemented in `frontend/src/api.ts`, WS-A3)

`analyzeFile(file, onStatus?)`:
1. `POST /api/analyze` with `FormData` field `file`.
2. If response `status === "completed"` → return `result`.
3. Otherwise poll `GET /api/analysis/{analysis_id}` every **1500 ms**, calling
   `onStatus(response)` on each poll, until `completed` (return `result`) or
   `failed` (throw `Error(error)`), with a **5 minute** overall timeout (throw).
4. Any non-2xx HTTP response → throw `Error` with the response `detail` when
   present, else the status text.

## Server-side behavior notes

- Async jobs run on a module-level `ThreadPoolExecutor(max_workers=2)`;
  job state lives in the in-memory `AnalysisStore` (WS-D1) with 1-hour TTL.
- Progress fractions reported by the orchestrator: encoders 0.00→0.35,
  judge per-turn windows 0.35→0.85, conversation-level judge 0.85→1.00.
- `/api/*` routes are registered BEFORE the StaticFiles mount.
