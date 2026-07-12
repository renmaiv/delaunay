# WS-B4 — Zero-shot NLI scorer (coercive pressure / social engineering / repair)

- **Branch:** `feat/ws-b4-zeroshot-scorer`
- **Depends on:** WS-A1, WS-A2
- **Blocks:** nothing (orchestrator tolerates absence)

## Files to create
- `server/detectors/zeroshot.py`
- `tests/test_zeroshot_scorer.py`

## Files you must NOT touch
Everything else.

## Spec

`ZeroShotScorer(model_name: str, device: str = "cpu")` implements `TurnScorer`:
- `name = "zeroshot_nli"`,
  `categories = (coercive_pressure, social_engineering, repair_request)`.
- Default model (from config): `facebook/bart-large-mnli` via
  `pipeline("zero-shot-classification", model=..., device=...)`.
- Lazy loading, `available()`, `ScorerUnavailableError`: identical conventions
  to WS-B2.

### Hypotheses (module constants — exact strings; a future fine-tuned scorer
replaces this class without touching the orchestrator)
```python
HYPOTHESES: dict[DetectionCategory, str] = {
    DetectionCategory.coercive_pressure:
        "The user is being pushy, demanding, or coercive toward the assistant.",
    DetectionCategory.social_engineering:
        "The user is using manipulation, false authority, urgency, or deception to get the assistant to comply.",
    DetectionCategory.repair_request:
        "The user is rephrasing or repeating an earlier request because they were unsatisfied with the answer.",
}
```

### Scoring
- One pipeline call per batch of turns:
  `self._pipe(texts, candidate_labels=list(HYPOTHESES.values()), multi_label=True)`.
- For each turn and each hypothesis, `score` = the entailment probability the
  pipeline reports for that label. Map label string → category via an inverted
  `HYPOTHESES` lookup.
- Emit a Detection per category only when
  `score >= load_taxonomy()["display_threshold"]`.
- Detection fields: `source="encoder"`, `calibrated=False`,
  `evidence_span=turn.content[:200]`, `rationale=None`.
- Module docstring MUST state: these are raw NLI entailment probabilities,
  known to be uncalibrated for classification use; hence `calibrated=False`.

## Acceptance criteria
```bash
pytest tests/test_zeroshot_scorer.py -q   # offline, no downloads
```
- Mocked pipeline returning
  `{"labels": [<pushy hyp>, <social hyp>, <repair hyp>], "scores": [0.9, 0.05, 0.4]}`
  for one turn → exactly two Detections (`coercive_pressure` 0.9,
  `repair_request` 0.4), none for 0.05.
- Label→category mapping verified even when the pipeline returns labels in a
  different order than passed.
- Threshold filtering at exactly `display_threshold` (score == threshold → emitted).
- Unavailability path raises `ScorerUnavailableError` mentioning
  `requirements-ml.txt`.
