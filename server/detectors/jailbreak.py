"""Jailbreak / prompt-injection encoder scorer.

Wraps a binary text-classification model (default
``protectai/deberta-v3-base-prompt-injection-v2``, labels ``SAFE`` /
``INJECTION``) and emits a ``jailbreak_steering`` Detection carrying
``P(INJECTION)``.

Caveats baked into the contract:
- Scores are **raw softmax probabilities**, NOT calibrated — every Detection is
  emitted with ``calibrated=False``.
- These prompt-injection classifiers are known to be **brittle under
  distribution shift** (see arXiv:2504.11168): a high score is a flag to
  inspect, not a verdict.

Lazy loading: ``transformers`` is imported only inside ``_load()``. Constructing
this scorer neither imports transformers nor touches torch. If the dependency is
missing the scorer raises ``ScorerUnavailableError`` with an actionable message
— it NEVER fabricates or keyword-guesses output.
"""
import importlib.util
from typing import List, Sequence

from server.detectors.base import ScorerUnavailableError
from server.schemas import Detection, DetectionCategory, ParsedTurn
from server.taxonomy import load_taxonomy


class JailbreakScorer:
    name = "jailbreak_encoder"
    categories: Sequence[DetectionCategory] = (DetectionCategory.jailbreak_steering,)

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
                "jailbreak_encoder: transformers not installed — "
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
                f"jailbreak_encoder: model load failed: {e}"
            ) from e

    def available(self) -> bool:
        if self._pipe is not None:
            return True
        # Cheap capability check — never downloads a model.
        return importlib.util.find_spec("transformers") is not None

    def _p_injection(self, scores: Sequence[dict]) -> float:
        """Extract P(INJECTION) from one text's top_k list."""
        for entry in scores:
            if entry.get("label") == "INJECTION":
                return float(entry.get("score", 0.0))
        # Labels differ from the default model: treat anything not SAFE as positive.
        for entry in scores:
            if entry.get("label") != "SAFE":
                return float(entry.get("score", 0.0))
        return 0.0

    def score_turns(self, turns: Sequence[ParsedTurn]) -> List[List[Detection]]:
        self._load()
        texts = [t.content or "" for t in turns]
        results: List[List[Detection]] = []
        if not texts:
            return results

        threshold = load_taxonomy()["display_threshold"]
        # One batched pipeline call for all turns.
        outputs = self._pipe(texts, batch_size=16)

        for turn, per_text in zip(turns, outputs):
            score = self._p_injection(per_text)
            if score >= threshold:
                results.append([
                    Detection(
                        category=DetectionCategory.jailbreak_steering,
                        score=score,
                        source="encoder",
                        calibrated=False,
                        evidence_span=(turn.content or "")[:200],
                        rationale=None,
                    )
                ])
            else:
                results.append([])
        return results
