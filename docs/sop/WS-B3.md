# WS-B3 — Sentiment scorer

- **Branch:** `feat/ws-b3-sentiment-scorer`
- **Depends on:** WS-A1, WS-A2
- **Blocks:** nothing (orchestrator tolerates absence)

## Files to create
- `server/detectors/sentiment.py`
- `tests/test_sentiment_scorer.py`

## Files you must NOT touch
Everything else.

## Spec

`CardiffSentimentScorer(model_name: str, device: str = "cpu")` implements the
`SentimentScorer` protocol from `server/detectors/base.py` (NOT `TurnScorer` —
sentiment is a scalar, not a Detection):
- `name = "sentiment_encoder"`.
- Default model (from config): `cardiffnlp/twitter-roberta-base-sentiment-latest`,
  labels `negative` / `neutral` / `positive`.
- Lazy loading, `available()`, and `ScorerUnavailableError` behavior: identical
  conventions to WS-B2 (import transformers inside `_load()`, actionable error
  messages mentioning `requirements-ml.txt`, never fabricate output).
- Pipeline: `pipeline("text-classification", model=..., device=...,
  truncation=True, max_length=512, top_k=None)` → per text, a list of all three
  label probabilities.
- Scalar per turn: `P(positive) - P(negative)`, clamped to [-1.0, 1.0].

Also export a module-level pure function:
```python
def overall_sentiment(per_turn: list[float]) -> float:
    """Recency-weighted mean: the last 3 entries get weight 2.0, all earlier
    entries weight 1.0; result clamped to [-1, 1]. Empty list -> 0.0."""
```
(The docstring formula is normative; the test asserts against it exactly.)

## Acceptance criteria
```bash
pytest tests/test_sentiment_scorer.py -q   # offline, no downloads
```
- Mocked pipeline returning probs (neg 0.8 / neu 0.1 / pos 0.1) → scalar
  `-0.7` (±1e-6).
- Mocked (neg 0.05 / neu 0.15 / pos 0.8) → `0.75` (±1e-6).
- `overall_sentiment([]) == 0.0`;
  `overall_sentiment([1.0, 1.0, -1.0])` == `(1*1 + 2*1 + 2*(-1) + ... )` — with
  3 entries all are "last 3", so weights `[2,2,2]` → mean `1/3` (±1e-6);
  `overall_sentiment([0.0, 0.0, 0.0, 1.0, 1.0, -1.0])` → weights
  `[1,1,1,2,2,2]` → `(0+0+0+2+2-2)/9 = 2/9` (±1e-6).
- Unavailability path raises `ScorerUnavailableError` mentioning
  `requirements-ml.txt` (same technique as WS-B2).
- Batch ordering preserved.
