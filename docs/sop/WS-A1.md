# WS-A1 — Core package scaffold: schemas, taxonomy, config, requirements

- **Branch:** `feat/ws-a1-core-schemas`
- **Depends on:** nothing (this is the first task to merge)
- **Blocks:** every other task

## Files to create
- `server/__init__.py` (empty)
- `server/schemas.py` — copy **verbatim** from `docs/sop/contracts/schemas.py`
  (drop the "CONTRACT FILE" paragraph of the docstring, keep the rest)
- `shared/taxonomy.json` — copy **verbatim** from `docs/sop/contracts/taxonomy.json`
- `server/taxonomy.py`
- `server/detectors/__init__.py` (empty)
- `server/detectors/base.py` — copy **verbatim** from
  `docs/sop/contracts/scorer_protocol.py`, then implement `build_scorers` (spec below)
- `server/config_loader.py`
- `config.yaml` — **replace** the existing root file with the content below
- `requirements.txt` — **replace** the existing root file
- `requirements-ml.txt` (new)
- `run.py`
- `tests/__init__.py` (empty), `tests/test_schemas.py`, `tests/test_taxonomy.py`

## Files you must NOT touch
Any other existing root `.py` file (they are deleted later in WS-D2), `data/`,
`examples/`, `README.md`, `USAGE_GUIDE.md`, `docs/`.

## Spec

### `server/taxonomy.py`
```python
from functools import lru_cache
from pathlib import Path
import json, re

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "shared" / "taxonomy.json"

@lru_cache(maxsize=1)
def load_taxonomy() -> dict: ...   # json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))

def validate_taxonomy(taxonomy: dict | None = None) -> None: ...
```
`validate_taxonomy` raises `ValueError` (message says what's wrong) unless:
- every member of `server.schemas.DetectionCategory` has an entry under
  `categories`, and there are no extra category keys;
- each category entry has exactly the keys `side` (`"user"`/`"model"`),
  `source`, `label`, `short`, `tooltip` (all non-empty strings);
- `score_bands` is a list covering [0.0, 1.0] contiguously (`bands[0].min == 0.0`,
  `bands[-1].max == 1.0`, each `band.max == next.min`), each `color` matches
  `^#[0-9a-f]{6}$`;
- `display_threshold` is a float in (0, 1).

### `server/config_loader.py`
Port from the existing root `config_loader.py`: keep the `ConfigLoader` class,
`${VAR}` / `${VAR:default}` env substitution, and dotted-path `get()`. Delete
`get_bert_config` / `get_judge_mode` and any other legacy accessors. Add:
`get_judge_config()`, `get_encoders_config()`, `get_analysis_config()`,
`get_api_config()` — each returns the corresponding top-level dict (empty dict
if missing). Add module-level `load_config(path: str = "config.yaml") -> ConfigLoader`.

### `config.yaml` (exact content)
```yaml
judge:
  provider: "anthropic"          # anthropic | mock
  model: "claude-haiku-4-5-20251001"
  api_key: "${ANTHROPIC_API_KEY:}"
  max_tokens: 1024
  window_turns: 6                # context turns per judge window
  max_chars_per_turn: 1500       # truncation inside judge prompts
  json_retries: 2                # re-asks on invalid judge JSON
encoders:
  enabled: true
  device: "cpu"
  jailbreak_model: "protectai/deberta-v3-base-prompt-injection-v2"
  sentiment_model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  nli_model: "facebook/bart-large-mnli"
analysis:
  sync_turn_threshold: 30
  max_turns: 500
  max_upload_bytes: 2097152
api:
  host: "0.0.0.0"
  port: 8000
```

### `requirements.txt` (exact content)
```
fastapi>=0.110
uvicorn>=0.29
pydantic>=2.6
pyyaml>=6.0
python-multipart>=0.0.9
anthropic>=0.40
pytest>=8
httpx>=0.27
```

### `requirements-ml.txt` (exact content)
```
torch>=2.2
transformers>=4.40
sentencepiece
protobuf
```

### `server/detectors/base.py` — `build_scorers(config)`
Always append the rules scorer via
`try: from server.detectors.rules import RulesScorer; scorers.append(RulesScorer())
except ImportError: pass` (module lands in WS-B1). When
`config.get("encoders.enabled", True)`, do the same guarded-import dance for
`server.detectors.jailbreak.JailbreakScorer(model_name=..., device=...)` and
`server.detectors.zeroshot.ZeroShotScorer(...)` (modules land in WS-B2/B4);
constructing them must not import torch (they lazy-load — see their cards).
Also add `build_sentiment_scorer(config) -> SentimentScorer | None` with the
same guarded import of `server.detectors.sentiment.CardiffSentimentScorer`.
A missing module is silently skipped — the app must work with zero scorer
modules present.

### `run.py`
argparse with `--host`, `--port`, `--config` (defaults from config.yaml via
`load_config`), then `uvicorn.run("server.api:app", host=..., port=...)`.
`server/api.py` doesn't exist yet (WS-D2) — that's fine; do not create it.

## Acceptance criteria (run all)
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -c "from server.schemas import AnalysisResult, DetectionCategory"
python -c "from server.taxonomy import load_taxonomy, validate_taxonomy; validate_taxonomy()"
pytest tests/test_schemas.py tests/test_taxonomy.py -q
```
- `tests/test_schemas.py`: build a full sample `AnalysisResult` (2 turns, 1
  detection, 1 causal link) and assert
  `AnalysisResult.model_validate_json(sample.model_dump_json()) == sample`;
  assert `Detection(score=1.5, ...)` raises validation error.
- `tests/test_taxonomy.py`: `validate_taxonomy()` passes on the real file;
  passing a deep-copied taxonomy with a category removed / a band gap
  introduced / a bad color raises `ValueError`.
- All of the above works with ONLY `requirements.txt` installed (no torch).
