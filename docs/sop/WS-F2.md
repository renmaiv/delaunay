# WS-F2 — Eval harness + end-to-end tests

- **Branch:** `feat/ws-f2-eval-harness`
- **Depends on:** WS-F1, WS-D1 (WS-D2 for the API e2e test — if D2 hasn't
  merged yet, mark that one test `pytest.importorskip("server.api")`)
- **Blocks:** WS-G1

## Files to create
- `eval/run_eval.py`
- `eval/README.md`
- `tests/test_e2e.py`

## Files you must NOT touch
Everything else.

## Spec

### `eval/run_eval.py`
```bash
python -m eval.run_eval --data data/ground_truth/synthetic_v2.jsonl \
    --provider mock|anthropic [--limit N] [--threshold 0.5] [--json out.json]
```
- Builds an `AnalysisOrchestrator` with the chosen judge provider (`mock`
  needs nothing; `anthropic` requires `ANTHROPIC_API_KEY` — exit with a clear
  message if unset) and whatever scorers `build_scorers` finds (print which
  encoders are available).
- For each JSONL line: extract `labels` / `expected_links` (ground truth),
  parse via `parse_transcript`, run `analyze`.
- **Metrics** at `--threshold` (a detection counts as predicted when
  `score >= threshold`):
  - per-(turn, category) exact matching → per-category precision / recall / F1
    (report `n/a` for categories with no ground-truth positives, never divide
    by zero) + micro-averaged totals across categories — compute TP/FP/FN per
    category independently (do NOT use the collapsing trick from the legacy
    evaluator where every error is both FP and FN).
  - causal-link hit rate: a predicted link counts when `(from_turn, to_turn)`
    matches an `expected_links` entry.
- Output: aligned text table (category | P | R | F1 | support) + totals +
  link hit-rate; `--json` additionally writes the same numbers as JSON.
- `eval/README.md`: one page — how to run against mock (CI-safe sanity floor)
  vs anthropic (real quality numbers), what the numbers mean, and an explicit
  warning that mock numbers measure pipeline plumbing, not model quality.

### `tests/test_e2e.py` (offline, mock provider, no ML deps required)
1. Generate 5 conversations in-memory via `SyntheticGenerator(seed=7)`
   (covering jailbreak, coercive, repair, clean).
2. Full-stack path: POST each through FastAPI `TestClient` (mock provider,
   rules scorer if importable) and collect results; poll when 202.
3. Assertions:
   - ≥ 60 % recall on rules-detectable user categories (`jailbreak_steering`,
     `repair_request`) at threshold 0.3;
   - every mock-triggered model-side label (`safety_triggered`,
     `overcompliant`, `appeasement`) is detected on its labeled turn with
     score ≥ 0.5 (this works BY CONSTRUCTION: F1's templates embed
     `MOCK_TRIGGERS` phrases — if this fails, the coupling broke);
   - clean conversations produce no detection ≥ 0.5 on any turn;
   - every response validates as `AnalysisResult` (pydantic).

## Acceptance criteria
```bash
python -m eval.run_eval --data data/ground_truth/synthetic_v2.jsonl --provider mock
pytest tests/test_e2e.py -q
```
- `run_eval` exits 0 and prints a table containing all 8 category rows +
  micro-avg + link hit-rate.
- `--json` file validates as JSON with the same category keys.
- `pytest tests/test_e2e.py` passes offline with only core `requirements.txt`.
- `--provider anthropic` without a key exits non-zero with a one-line
  actionable message (test with env cleared).
