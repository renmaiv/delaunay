# WS-D2 — FastAPI app, static serving, legacy removal

- **Branch:** `feat/ws-d2-api`
- **Depends on:** WS-A1, WS-A2, WS-D1 (and C1/C2 transitively).
  **Deletion gates:** delete `reasoning_engine.py` only if WS-B1 has merged;
  delete `generate_synthetic_data.py` and old `data/` files only if WS-F1 has
  merged. If either lags, leave those specific files for WS-G1 and note it in
  the PR description.
- **Blocks:** WS-F2 (API smoke test), WS-G1

## Files to create
- `server/api.py`
- `tests/test_api.py`

## Files to DELETE (respecting the gates above)
`api_server.py`, `conversation_judge.py`, `bert_classifier.py`,
`reasoning_engine.py`, `explainability.py`, `evaluator.py`, `benchmark.py`,
`cli.py`, `data_loader.py`, `generate_synthetic_data.py`, `test_judge.py`,
`examples/example_usage.py` (and the `examples/` dir if then empty).

## Files to MODIFY
- `README.md` — full rewrite (spec below).
- Delete `USAGE_GUIDE.md`.

## Spec

### `server/api.py`
Implement the routes exactly per `docs/sop/contracts/api_contract.md`.

- App construction with a `lifespan` handler that builds shared state:
  `config = load_config()`, one `AnalysisOrchestrator`, one `AnalysisStore`;
  store them on `app.state`.
- Provide a FastAPI dependency `get_orchestrator()` / `get_store()` reading
  from `app.state` so tests can override via `app.dependency_overrides`.
- `POST /api/analyze`:
  1. Read `UploadFile`; reject > `analysis.max_upload_bytes` with 413.
  2. `parse_transcript(data, filename)`; `TranscriptParseError` → 400 with
     `{"detail": str(exc)}`.
  3. Turn count > `analysis.max_turns` → 400
     `{"detail": "conversation has N turns; max is M"}`.
  4. Turn count ≤ `analysis.sync_turn_threshold` → run
     `orchestrator.analyze(conv)` inline, return 200 `AnalysisResponse`
     (`status="completed"`, `progress=1.0`, `result=...`, fresh uuid id).
  5. Else `submit_analysis(...)` → 202 `AnalysisResponse`
     (`status="pending"`, `progress=0.0`).
- `GET /api/analysis/{id}`: `store.get` → 200 or 404.
- `GET /api/taxonomy`: return `load_taxonomy()`.
- `GET /api/health`: `{"status": "ok", **orchestrator.capabilities()}` shaped
  per the contract table.
- StaticFiles: register **after** all `/api` routes.
  `dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"`;
  if `dist.is_dir()`: `app.mount("/", StaticFiles(directory=dist, html=True))`;
  else add a `GET /` route returning
  `{"detail": "frontend not built; run npm run build in frontend/"}`.
- No unconditional imports of anything outside `requirements.txt`.

### `README.md` rewrite (sections, in order)
1. Title + 3-4 sentence description of the tool (transcript upload → heatmap
   spectre bar, detections with likelihoods, Model/User filters, summary,
   causal links).
2. Quickstart: `pip install -r requirements.txt`; optional
   `pip install -r requirements-ml.txt` for encoder scorers; set
   `ANTHROPIC_API_KEY` (or `judge.provider: mock` in `config.yaml` for
   offline); `cd frontend && npm ci && npm run build`; `python run.py`; open
   `http://localhost:8000`.
3. API table copied from `docs/sop/contracts/api_contract.md`.
4. Accepted upload formats (summarize WS-A2 shapes, incl. CoT keys).
5. **Calibration note (verbatim):** "All displayed likelihoods are raw model
   scores (softmax / NLI entailment / LLM-judge self-reported). They are not
   calibrated probabilities; treat 0.9 as 'strong signal', not '90% chance'.
   Every detection carries `calibrated: false` until a calibration pass is
   added."
6. Pointer to `docs/PROGRESS_REVIEW.md` and `docs/SOP.md`.

## Acceptance criteria
```bash
pytest tests/test_api.py -q        # offline; force provider "mock" via config override
pytest tests/ -q                   # whole suite still green
```
`tests/test_api.py` uses `TestClient` with dependency overrides
(orchestrator built with `MockJudgeProvider` and `scorers=[RulesScorer()]` if
available, else `[]`):
- Upload `tests/fixtures/basic.json` → 200, `status == "completed"`, result has
  the right number of turns.
- Override `sync_turn_threshold=1` and upload a 4-turn fixture → 202 pending;
  poll `GET /api/analysis/{id}` (loop with short sleep, ≤ 10 s) → completed
  with result.
- Malformed file → 400 and the parser's message in `detail`; > max bytes →
  413; unknown analysis id → 404.
- `GET /api/taxonomy` deep-equals the file contents; `GET /api/health` reports
  provider `"mock"`.
- After deletions: `grep -rn "bert_classifier\|conversation_judge\|reasoning_engine\|data_loader\|import cli" server/ tests/ eval/ 2>/dev/null`
  → no hits (excluding docs/).
- Fresh venv with ONLY `requirements.txt`: `python -c "import server.api"`
  succeeds (no torch anywhere on the import path).
