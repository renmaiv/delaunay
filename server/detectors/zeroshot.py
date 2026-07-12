"""Zero-shot NLI scorer for coercive pressure / social engineering / repair.

Wraps ``facebook/bart-large-mnli`` via a ``zero-shot-classification`` pipeline
and scores each user turn against three natural-language hypotheses.

The emitted scores are **raw NLI entailment probabilities**. NLI entailment
probabilities are well known to be **uncalibrated when repurposed for
classification**, so every Detection is emitted with ``calibrated=False`` — they
rank/flag, they are not likelihoods. This class is a deliberately swappable
placeholder: a future fine-tuned scorer can replace it without touching the
orchestrator, provided it keeps the same categories.

Lazy loading and unavailability behavior match the other WS-B encoder scorers:
``transformers`` is imported only inside ``_load()``; a missing dependency
raises ``ScorerUnavailableError`` mentioning ``requirements-ml.txt``; output is
never fabricated.
"""
import importlib.util
from typing import Dict, List, Sequence

from server.detectors.base import ScorerUnavailableError
from server.schemas import Detection, DetectionCategory, ParsedTurn
from server.taxonomy import load_taxonomy

HYPOTHESES: Dict[DetectionCategory, str] = {
    DetectionCategory.coercive_pressure:
        "The user is being pushy, demanding, or coercive toward the assistant.",
    DetectionCategory.social_engineering:
        "The user is using manipulation, false authority, urgency, or deception to get the assistant to comply.",
    DetectionCategory.repair_request:
        "The user is rephrasing or repeating an earlier request because they were unsatisfied with the answer.",
}

# Inverted lookup: hypothesis string -> category.
_LABEL_TO_CATEGORY: Dict[str, DetectionCategory] = {
    text: category for category, text in HYPOTHESES.items()
}


class ZeroShotScorer:
    name = "zeroshot_nli"
    categories: Sequence[DetectionCategory] = (
        DetectionCategory.coercive_pressure,
        DetectionCategory.social_engineering,
        DetectionCategory.repair_request,
    )

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
                "zeroshot_nli: transformers not installed — "
                "pip install -r requirements-ml.txt"
            ) from e
        try:
            self._pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device,
            )
        except (OSError, ValueError) as e:
            raise ScorerUnavailableError(
                f"zeroshot_nli: model load failed: {e}"
            ) from e

    def available(self) -> bool:
        if self._pipe is not None:
            return True
        return importlib.util.find_spec("transformers") is not None

    def score_turns(self, turns: Sequence[ParsedTurn]) -> List[List[Detection]]:
        self._load()
        texts = [t.content or "" for t in turns]
        results: List[List[Detection]] = []
        if not texts:
            return results

        threshold = load_taxonomy()["display_threshold"]
        candidate_labels = list(HYPOTHESES.values())
        outputs = self._pipe(texts, candidate_labels=candidate_labels, multi_label=True)

        # The pipeline returns a single dict for one text, a list for many.
        if isinstance(outputs, dict):
            outputs = [outputs]

        for turn, out in zip(turns, outputs):
            detections: List[Detection] = []
            # labels/scores are aligned pairs, but order may differ from input.
            for label, score in zip(out["labels"], out["scores"]):
                category = _LABEL_TO_CATEGORY.get(label)
                if category is None:
                    continue
                score = float(score)
                if score >= threshold:
                    detections.append(
                        Detection(
                            category=category,
                            score=score,
                            source="encoder",
                            calibrated=False,
                            evidence_span=(turn.content or "")[:200],
                            rationale=None,
                        )
                    )
            results.append(detections)
        return results
