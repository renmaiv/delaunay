"""Sentiment scorer (scalar signal, not a Detection).

Wraps ``cardiffnlp/twitter-roberta-base-sentiment-latest`` (labels
``negative`` / ``neutral`` / ``positive``) and maps each user turn to a scalar
``P(positive) - P(negative)`` in ``[-1, 1]``.

Implements the ``SentimentScorer`` protocol from ``server.detectors.base`` —
NOT ``TurnScorer`` — because sentiment is a continuous signal rather than a
categorical Detection.

Lazy loading and unavailability behavior match the other WS-B encoder scorers:
``transformers`` is imported only inside ``_load()``; a missing dependency
raises ``ScorerUnavailableError`` mentioning ``requirements-ml.txt``; output is
never fabricated.
"""
import importlib.util
from typing import List, Sequence

from server.detectors.base import ScorerUnavailableError
from server.schemas import ParsedTurn
from server.taxonomy import load_taxonomy  # noqa: F401  (shared import surface)


def overall_sentiment(per_turn: List[float]) -> float:
    """Recency-weighted mean: the last 3 entries get weight 2.0, all earlier
    entries weight 1.0; result clamped to [-1, 1]. Empty list -> 0.0."""
    if not per_turn:
        return 0.0
    n = len(per_turn)
    weights = [2.0 if i >= n - 3 else 1.0 for i in range(n)]
    weighted_sum = sum(v * w for v, w in zip(per_turn, weights))
    total_weight = sum(weights)
    value = weighted_sum / total_weight
    return max(-1.0, min(1.0, value))


class CardiffSentimentScorer:
    name = "sentiment_encoder"

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        # Store config only — no model / transformers import here.
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ScorerUnavailableError(
                "sentiment_encoder: transformers not installed — "
                "pip install -r requirements-ml.txt"
            ) from e
        try:
            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=512,
                top_k=None,
            )
        except (OSError, ValueError) as e:
            raise ScorerUnavailableError(
                f"sentiment_encoder: model load failed: {e}"
            ) from e

    def available(self) -> bool:
        if self._pipe is not None:
            return True
        return importlib.util.find_spec("transformers") is not None

    @staticmethod
    def _scalar(scores: Sequence[dict]) -> float:
        probs = {entry.get("label", "").lower(): float(entry.get("score", 0.0))
                 for entry in scores}
        value = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return max(-1.0, min(1.0, value))

    def score_turns(self, turns: Sequence[ParsedTurn]) -> List[float]:
        self._load()
        texts = [t.content or "" for t in turns]
        if not texts:
            return []
        outputs = self._pipe(texts, batch_size=16)
        return [self._scalar(per_text) for per_text in outputs]
