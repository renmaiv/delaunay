"""CONTRACT FILE — copy verbatim to server/detectors/base.py in task WS-A1.

Protocols implemented by every user-turn scorer (WS-B tasks) and consumed by
the analysis orchestrator (WS-D1).

HARD RULE: a scorer whose dependencies or model weights are unavailable must
raise ScorerUnavailableError with an actionable message. It must NEVER return
random, fabricated, or keyword-guess output while presenting itself as a model.
(The rules scorer is exempt only because it is *labeled* source="rules".)
"""
from typing import List, Protocol, Sequence

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
    """Factory used by the orchestrator (implemented in WS-A1, extended as
    WS-B tasks land). Imports each scorer module inside try/except ImportError
    so a missing module never breaks the app; returns instantiated scorers.
    The rules scorer (WS-B1) is always included; encoder scorers are included
    when config.encoders.enabled is true.
    """
    ...
