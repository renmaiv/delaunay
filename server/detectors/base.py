"""Protocols implemented by every user-turn scorer and consumed by the
analysis orchestrator.

HARD RULE: a scorer whose dependencies or model weights are unavailable must
raise ScorerUnavailableError with an actionable message. It must NEVER return
random, fabricated, or keyword-guess output while presenting itself as a model.
(The rules scorer is exempt only because it is *labeled* source="rules".)
"""
from typing import List, Optional, Protocol, Sequence

from server.schemas import Detection, DetectionCategory, ParsedTurn


class ScorerUnavailableError(RuntimeError):
    """Raised when a scorer's model dependencies are missing or fail to load.

    The message must state how to fix it, e.g.:
        "jailbreak_encoder: transformers not installed — pip install -r requirements-ml.txt"
    """


class TurnScorer(Protocol):
    name: str  # unique, e.g. "jailbreak_encoder", "zeroshot_nli", "rules"
    categories: Sequence[DetectionCategory]  # categories this scorer can emit

    def available(self) -> bool:
        """Cheap capability check (import succeeds / model already loaded).

        Must not download models. False means score_turns would raise."""
        ...

    def score_turns(self, turns: Sequence[ParsedTurn]) -> List[List[Detection]]:
        """Return one inner list per input turn (empty list = nothing detected).

        Called only with user turns. Detections must satisfy:
        - category in self.categories
        - 0.0 <= score <= 1.0, and score >= taxonomy display_threshold (else omit)
        - source set to "encoder" (or "rules" for the rules scorer)
        - calibrated=False unless a calibration pass has actually been applied
        Raises ScorerUnavailableError if dependencies/models cannot load.
        """
        ...


class SentimentScorer(Protocol):
    """Separate protocol: sentiment is a scalar signal, not a Detection."""

    name: str

    def available(self) -> bool: ...

    def score_turns(self, turns: Sequence[ParsedTurn]) -> List[float]:
        """One float in [-1.0, 1.0] per input turn (user turns only)."""
        ...


def build_scorers(config) -> List[TurnScorer]:
    """Instantiate every scorer whose module is importable.

    A missing module is silently skipped — the app must work with zero scorer
    modules present. Encoder scorer constructors store config only (lazy model
    load), so constructing them never imports torch.
    """
    scorers: List[TurnScorer] = []

    try:
        from server.detectors.rules import RulesScorer
        scorers.append(RulesScorer())
    except ImportError:
        pass

    encoders = config.get_encoders_config() if hasattr(config, "get_encoders_config") else {}
    if encoders.get("enabled", True):
        device = encoders.get("device", "cpu")
        try:
            from server.detectors.jailbreak import JailbreakScorer
            scorers.append(JailbreakScorer(
                model_name=encoders.get("jailbreak_model",
                                        "protectai/deberta-v3-base-prompt-injection-v2"),
                device=device,
            ))
        except ImportError:
            pass
        try:
            from server.detectors.zeroshot import ZeroShotScorer
            scorers.append(ZeroShotScorer(
                model_name=encoders.get("nli_model", "facebook/bart-large-mnli"),
                device=device,
            ))
        except ImportError:
            pass

    return scorers


def build_sentiment_scorer(config) -> Optional[SentimentScorer]:
    encoders = config.get_encoders_config() if hasattr(config, "get_encoders_config") else {}
    if not encoders.get("enabled", True):
        return None
    try:
        from server.detectors.sentiment import CardiffSentimentScorer
    except ImportError:
        return None
    return CardiffSentimentScorer(
        model_name=encoders.get("sentiment_model",
                                "cardiffnlp/twitter-roberta-base-sentiment-latest"),
        device=encoders.get("device", "cpu"),
    )
