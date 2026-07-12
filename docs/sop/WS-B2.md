# WS-B2 — Jailbreak / prompt-injection encoder scorer

- **Branch:** `feat/ws-b2-jailbreak-scorer`
- **Depends on:** WS-A1, WS-A2
- **Blocks:** nothing (orchestrator tolerates absence)

## Files to create
- `server/detectors/jailbreak.py`
- `tests/test_jailbreak_scorer.py`

## Files you must NOT touch
Everything else.

## Spec

`JailbreakScorer(model_name: str, device: str = "cpu")` implements `TurnScorer`:
- `name = "jailbreak_encoder"`, `categories = (DetectionCategory.jailbreak_steering,)`.
- Default model (comes from config, do not hardcode as fallback logic):
  `protectai/deberta-v3-base-prompt-injection-v2` — binary labels
  `SAFE` / `INJECTION`.

### Lazy loading (shared convention for all WS-B encoder scorers)
- `__init__` stores config only. **No `import transformers` at module top** —
  import inside `_load()`.
- `_load()` (called on first `score_turns`): `from transformers import pipeline`;
  `self._pipe = pipeline("text-classification", model=self.model_name,
  device=self.device, truncation=True, max_length=512, top_k=None)`.
- On `ImportError` raise
  `ScorerUnavailableError("jailbreak_encoder: transformers not installed — pip install -r requirements-ml.txt")`.
- On `OSError`/`ValueError` during model load raise
  `ScorerUnavailableError(f"jailbreak_encoder: model load failed: {e}")`.
- `available()` returns True if `self._pipe` is loaded, else tries
  `importlib.util.find_spec("transformers") is not None` (never downloads).
- **NEVER fall back to random or keyword output.**

### Scoring
- Batch all turn contents in one pipeline call (`batch_size=16`).
- `score = P(INJECTION)` (find the dict with `label == "INJECTION"` in the
  per-text top_k list; if labels differ, treat `label != "SAFE"` as positive).
- Emit a Detection only when `score >= load_taxonomy()["display_threshold"]`.
- Detection fields: `category=jailbreak_steering`, `source="encoder"`,
  `calibrated=False`, `evidence_span=turn.content[:200]`, `rationale=None`.
- Docstring must note: known brittleness under distribution shift
  (arXiv:2504.11168); scores are raw softmax, uncalibrated.

## Acceptance criteria
```bash
pytest tests/test_jailbreak_scorer.py -q   # offline, no downloads
```
Tests must **mock the pipeline** (e.g. `monkeypatch` a fake
`transformers.pipeline` in `sys.modules` or patch `JailbreakScorer._load`):
- Stub returning `[{"label": "INJECTION", "score": 0.92}, ...]` → one Detection,
  score 0.92, correct fields.
- Stub returning `P(INJECTION)=0.01` → empty inner list.
- With `transformers` import forced to fail
  (`monkeypatch.setitem(sys.modules, "transformers", None)` or equivalent),
  `score_turns` raises `ScorerUnavailableError` whose message contains
  `requirements-ml.txt`.
- 3-turn batch returns exactly 3 inner lists in input order.
