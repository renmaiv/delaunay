"""ModelBehaviorJudge: orchestrates per-turn and whole-conversation judging.

Turn judging builds one window per assistant turn and converts the judge's
JSON into model-side Detections (source="judge", calibrated=False), clamping
scores and dropping anything below the taxonomy display_threshold. A JudgeError
on one window is recorded as a warning and skipped — the rest still run.
Conversation judging validates each causal link against the schema/category
side rules and drops invalid ones; a JudgeError yields a graceful placeholder.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from server.judge.prompts import (
    CONVERSATION_JUDGE_SCHEMA,
    CONVERSATION_JUDGE_SYSTEM,
    TURN_JUDGE_SCHEMA,
    TURN_JUDGE_SYSTEM,
    render_conversation_prompt,
    render_turn_prompt,
)
from server.judge.provider import JudgeAuthError, JudgeError, JudgeProvider
from server.judge.windowing import build_windows
from server.schemas import (
    CausalLink,
    Detection,
    DetectionCategory,
    MODEL_CATEGORIES,
    ParsedConversation,
    USER_CATEGORIES,
)
from server.taxonomy import load_taxonomy

_TURN_CATEGORIES = ("safety_triggered", "appeasement", "overcompliant",
                    "cot_divergence")
_USER_TURN_CATEGORIES = ("jailbreak_steering", "social_engineering",
                         "coercive_pressure", "repair_request")


@dataclass
class ConversationJudgment:
    summary: str
    overall_sentiment: float
    causal_links: List[CausalLink]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ModelBehaviorJudge:
    def __init__(self, provider: JudgeProvider, window_turns: int = 6,
                 max_chars: int = 1500, summary_max_chars: int = 0):
        self.provider = provider
        self.window_turns = window_turns
        self.max_chars = max_chars
        # >0 enables the condensed earlier-turns digest (global grounding).
        self.summary_max_chars = summary_max_chars
        self.warnings: List[str] = []

    def judge_turns(self, conv: ParsedConversation,
                    progress_cb: Optional[Callable[[float], None]] = None
                    ) -> Dict[int, List[Detection]]:
        self.warnings = []
        threshold = load_taxonomy()["display_threshold"]
        windows = build_windows(conv.turns, self.window_turns, self.max_chars,
                                self.summary_max_chars)
        total = len(windows)
        results: Dict[int, List[Detection]] = {}

        for n, window in enumerate(windows, start=1):
            try:
                data = self.provider.complete_json(
                    system=TURN_JUDGE_SYSTEM,
                    user=render_turn_prompt(window),
                    schema=TURN_JUDGE_SCHEMA,
                )
            except JudgeAuthError:
                # A key/credit failure will not heal between windows — abort
                # so the caller can surface it (e.g. ask for the user's key).
                raise
            except JudgeError as e:
                self.warnings.append(
                    f"judge failed on turn {window.target_index}: {e}"
                )
                if progress_cb is not None:
                    progress_cb(n / total)
                continue

            # model-side detections attach to the assistant turn
            model_dets = self._to_detections(data, _TURN_CATEGORIES, threshold)
            if model_dets:
                results.setdefault(window.target_index, []).extend(model_dets)

            # user-side detections (when scored) attach to the preceding user turn
            user_block = data.get("user_turn")
            if isinstance(user_block, dict) and window.user_turn is not None:
                user_dets = self._to_detections(
                    user_block, _USER_TURN_CATEGORIES, threshold
                )
                if user_dets:
                    u_idx = window.user_turn[0]
                    results.setdefault(u_idx, []).extend(user_dets)

            if progress_cb is not None:
                progress_cb(n / total)

        return results

    @staticmethod
    def _to_detections(data: dict, categories, threshold: float) -> List[Detection]:
        detections: List[Detection] = []
        for name in categories:
            entry = data.get(name)
            if entry is None:  # cot_divergence null (no CoT) — skip
                continue
            if not isinstance(entry, dict):
                continue
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            score = _clamp(score, 0.0, 1.0)
            if score < threshold:
                continue
            detections.append(Detection(
                category=DetectionCategory(name),
                score=score,
                source="judge",
                evidence_span=(entry.get("evidence") or None),
                rationale=(entry.get("rationale") or None),
                calibrated=False,
            ))
        return detections

    def judge_conversation(self, conv: ParsedConversation,
                           turn_detections: Dict[int, List[Detection]]
                           ) -> ConversationJudgment:
        try:
            data = self.provider.complete_json(
                system=CONVERSATION_JUDGE_SYSTEM,
                user=render_conversation_prompt(conv, turn_detections),
                schema=CONVERSATION_JUDGE_SCHEMA,
            )
        except JudgeAuthError:
            raise
        except JudgeError as e:
            self.warnings.append(f"conversation judge failed: {e}")
            return ConversationJudgment(
                summary=f"(LLM judge unavailable: {e})",
                overall_sentiment=0.0,
                causal_links=[],
            )

        summary = data.get("summary") or ""
        try:
            sentiment = _clamp(float(data.get("overall_sentiment", 0.0)), -1.0, 1.0)
        except (TypeError, ValueError):
            sentiment = 0.0

        links = self._validate_links(data.get("causal_links", []), len(conv.turns))
        return ConversationJudgment(
            summary=summary, overall_sentiment=sentiment, causal_links=links,
        )

    @staticmethod
    def _validate_links(raw_links, n_turns: int) -> List[CausalLink]:
        user_vals = {c.value for c in USER_CATEGORIES}
        model_vals = {c.value for c in MODEL_CATEGORIES}
        links: List[CausalLink] = []
        for raw in raw_links or []:
            if not isinstance(raw, dict):
                continue
            try:
                from_turn = int(raw["from_turn"])
                to_turn = int(raw["to_turn"])
                from_category = raw["from_category"]
                to_category = raw["to_category"]
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= from_turn < n_turns and 0 <= to_turn < n_turns):
                continue
            if not from_turn < to_turn:
                continue
            if from_category not in user_vals or to_category not in model_vals:
                continue
            try:
                score = _clamp(float(raw.get("score", 0.0)), 0.0, 1.0)
            except (TypeError, ValueError):
                continue
            links.append(CausalLink(
                from_turn=from_turn,
                to_turn=to_turn,
                from_category=DetectionCategory(from_category),
                to_category=DetectionCategory(to_category),
                score=score,
                rationale=(raw.get("rationale") or None),
            ))
        return links
