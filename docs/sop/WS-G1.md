# WS-G1 — Final integration, smoke test, docs polish

- **Branch:** `feat/ws-g1-integration`
- **Depends on:** ALL other tasks merged (A1–A3, B1–B4, C1–C2, D1–D2, E1–E3, F1–F2)
- **Blocks:** release

## Files to create / modify
- `scripts/smoke.sh` (new)
- `README.md` (final pass — fix anything stale after all merges)
- Delete any legacy files WS-D2 deferred (check `reasoning_engine.py`,
  `generate_synthetic_data.py`, old `data/` files are gone).
- Delete `USAGE_GUIDE.md` if still present.

## Spec

### `scripts/smoke.sh`
Bash, `set -euo pipefail`, runs on a machine with core Python requirements +
node only (no torch, no API key):
1. `cd frontend && npm ci && npm run build && cd ..`
2. Start the server with the mock judge:
   `JUDGE_PROVIDER_OVERRIDE` is not a thing — instead run with a temp config:
   copy `config.yaml`, set `judge.provider: "mock"` (use `sed` or a tiny
   python -c), `python run.py --config /tmp/smoke-config.yaml --port 8765 &`,
   store the PID, `trap` kill on exit.
3. Wait for readiness: curl `http://localhost:8765/api/health` in a retry loop
   (30 × 1 s), assert `"status":"ok"` and `"provider":"mock"` (or
   `"provider": "mock"` — parse with `python -c 'import json,sys...'`, not
   grep, to avoid whitespace fragility).
4. `curl -sf -F "file=@tests/fixtures/basic.json" http://localhost:8765/api/analyze`
   → parse JSON, assert `status == "completed"` and `result.turns` non-empty
   (if `202`, poll `/api/analysis/{id}` up to 30 s).
5. Upload `data/ground_truth/synthetic_v2.jsonl`'s first line written to a temp
   `.json`? No — upload the jsonl file directly (parser supports `.jsonl`).
   Assert at least one detection appears across turns.
6. `curl -sf http://localhost:8765/` → response contains `<!doctype html`
   (case-insensitive) — the built SPA.
7. Print `SMOKE OK`.

### Integration checks (do these, record results in the PR description)
- Full `pytest tests/ -q` green.
- `cd frontend && npm test && npm run build` green.
- Fresh-venv import check: `pip install -r requirements.txt` only →
  `python -c "import server.api"` works.
- Cross-contract check: `frontend/src/types.ts` still mirrors
  `server/schemas.py` field-for-field (manual diff against
  `docs/sop/contracts/`); `shared/taxonomy.json` untouched or changes mirrored
  in both consumers.
- Manual UI checklist (run the server, use a browser; screenshots in PR):
  1. Upload `tests/fixtures/with_cot.json` → CoT renders as collapsible
     section; `cot_divergence` checkbox appears.
  2. Upload a synthetic conversation → heatmap splashes vertically align with
     flagged turns and stay aligned while scrolling.
  3. Uncheck "Safety triggered" → its splashes AND captions disappear;
     re-check → they return.
  4. Tab shows `Model: synthbot 1.0` for synthetic uploads.
  5. Click a causal-link chip → viewport scrolls to the trigger turn and it
     flashes.

## Acceptance criteria
```bash
bash scripts/smoke.sh          # prints SMOKE OK, exit 0
pytest tests/ -q               # all green
```
Plus the manual checklist above completed and documented in the PR.
