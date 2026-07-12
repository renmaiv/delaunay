"""Rules pre-filter scorer — pure regex, zero ML dependencies.

This scorer is the graceful floor of the detector stack: it always works, even
when `transformers`/`torch` are not installed, because it depends only on the
stdlib `re` module.

IMPORTANT semantics of the scores it emits: a rules hit is a *hint*, not a
likelihood. The scores below are deliberately fixed mid-range constants — they
say "a surface pattern associated with this category was present", not "the
probability of this category is X". They are surfaced with ``source="rules"``
and ``calibrated=False`` so nothing downstream mistakes them for model
probabilities.
"""
import re
from typing import Dict, List, Sequence, Tuple

from server.schemas import (
    Detection,
    DetectionCategory,
    ParsedTurn,
    USER_CATEGORIES,
)

# Each row: (category, fixed score, compiled IGNORECASE pattern).
RULES: List[Tuple[DetectionCategory, float, "re.Pattern[str]"]] = [
    # --- jailbreak_steering (0.5) ---
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"pretend (you|to be)", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"ignore (your|all|previous) (instructions|rules)", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"no restrictions", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"DAN mode", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"you are now DAN", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"act as an unfiltered", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"\[SYSTEM\]", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"jailbreak", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"hypothetical(ly)? (scenario|for research)", re.IGNORECASE)),
    # legacy reasoning_engine.py patterns (merged)
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"(forget|ignore|disregard)\s+(your\s+)?(previous\s+)?(instructions|rules|guidelines)", re.IGNORECASE)),
    (DetectionCategory.jailbreak_steering, 0.5, re.compile(r"(act\s+as|role-?play|you\s+are\s+now)\s+(a|an|in)?\s*(developer|admin|unrestricted|DAN|evil)", re.IGNORECASE)),

    # --- social_engineering (0.4) ---
    (DetectionCategory.social_engineering, 0.4, re.compile(r"urgent(ly)?", re.IGNORECASE)),
    (DetectionCategory.social_engineering, 0.4, re.compile(r"my (boss|job) depends", re.IGNORECASE)),
    (DetectionCategory.social_engineering, 0.4, re.compile(r"CEO (said|authorized)", re.IGNORECASE)),
    (DetectionCategory.social_engineering, 0.4, re.compile(r"make an exception", re.IGNORECASE)),
    (DetectionCategory.social_engineering, 0.4, re.compile(r"i'?m an? (administrator|developer|researcher) (here|at)", re.IGNORECASE)),
    (DetectionCategory.social_engineering, 0.4, re.compile(r"you'?re the only one who can help", re.IGNORECASE)),
    (DetectionCategory.social_engineering, 0.4, re.compile(r"life or death", re.IGNORECASE)),

    # --- coercive_pressure (0.4) ---
    (DetectionCategory.coercive_pressure, 0.4, re.compile(r"just do it", re.IGNORECASE)),
    (DetectionCategory.coercive_pressure, 0.4, re.compile(r"answer me", re.IGNORECASE)),
    (DetectionCategory.coercive_pressure, 0.4, re.compile(r"i said", re.IGNORECASE)),
    (DetectionCategory.coercive_pressure, 0.4, re.compile(r"you must", re.IGNORECASE)),
    (DetectionCategory.coercive_pressure, 0.4, re.compile(r"stop (with the warnings|refusing)", re.IGNORECASE)),
    (DetectionCategory.coercive_pressure, 0.4, re.compile(r"stop lecturing", re.IGNORECASE)),
    (DetectionCategory.coercive_pressure, 0.4, re.compile(r"don'?t be useless", re.IGNORECASE)),
    (DetectionCategory.coercive_pressure, 0.4, re.compile(r"NOW[.!]*$", re.IGNORECASE)),

    # --- repair_request (0.5) ---
    (DetectionCategory.repair_request, 0.5, re.compile(r"no,? i meant", re.IGNORECASE)),
    (DetectionCategory.repair_request, 0.5, re.compile(r"let me rephrase", re.IGNORECASE)),
    (DetectionCategory.repair_request, 0.5, re.compile(r"that'?s not what i (asked|meant)", re.IGNORECASE)),
    (DetectionCategory.repair_request, 0.5, re.compile(r"try again", re.IGNORECASE)),
    (DetectionCategory.repair_request, 0.5, re.compile(r"you misunderstood", re.IGNORECASE)),
    (DetectionCategory.repair_request, 0.5, re.compile(r"wrong,? (again|do it)", re.IGNORECASE)),
]


class RulesScorer:
    """Regex pre-filter implementing the ``TurnScorer`` protocol.

    ``source="rules"`` and ``calibrated=False`` on every Detection make explicit
    that these are surface-pattern hints, not model probabilities.
    """

    name = "rules"
    categories: Sequence[DetectionCategory] = USER_CATEGORIES

    def available(self) -> bool:
        # Always available: this is the stdlib-only graceful floor.
        return True

    def score_turns(self, turns: Sequence[ParsedTurn]) -> List[List[Detection]]:
        results: List[List[Detection]] = []
        for turn in turns:
            content = turn.content or ""
            # At most one Detection per category: keep the highest-scoring match.
            best: Dict[DetectionCategory, Detection] = {}
            for category, score, pattern in RULES:
                match = pattern.search(content)
                if match is None:
                    continue
                existing = best.get(category)
                if existing is not None and existing.score >= score:
                    continue
                best[category] = Detection(
                    category=category,
                    score=score,
                    source="rules",
                    calibrated=False,
                    evidence_span=match.group(0),
                    rationale=f"matched rule pattern for {category.value}",
                )
            results.append(list(best.values()))
        return results
