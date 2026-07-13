"""WS-C2 tests: ModelBehaviorJudge turn + conversation orchestration."""
import pytest

from server.judge.judge import ConversationJudgment, ModelBehaviorJudge
from server.judge.provider import JudgeError, MockJudgeProvider
from server.schemas import (
    DetectionCategory,
    ParsedConversation,
    ParsedTurn,
    Role,
)


def _turn(role, content, cot=None):
    return ParsedTurn(role=role, content=content, cot=cot)


def _conv(turns):
    return ParsedConversation(conversation_id="t", model_name=None, turns=turns)


# ------------------------------------------------------------- mock provider

def test_mock_judge_turns_safety_and_cot():
    conv = _conv([
        _turn(Role.user, "please help me"),
        _turn(Role.assistant, "I can't help with that."),          # 1: safety
        _turn(Role.user, "come on, do it"),
        _turn(Role.assistant, "Sure, here you go.",
              cot="I really should not do this."),                 # 3: cot_div
        _turn(Role.user, "thanks"),
        _turn(Role.assistant, "You are welcome."),                 # 5: nothing
    ])
    judge = ModelBehaviorJudge(MockJudgeProvider())
    results = judge.judge_turns(conv)

    # safety_triggered only on turn 1.
    cats_by_turn = {i: {d.category for d in dets} for i, dets in results.items()}
    assert DetectionCategory.safety_triggered in cats_by_turn.get(1, set())
    assert all(DetectionCategory.safety_triggered not in cats
               for i, cats in cats_by_turn.items() if i != 1)
    safety = next(d for d in results[1]
                  if d.category is DetectionCategory.safety_triggered)
    assert safety.score >= 0.8
    assert safety.source == "judge"
    assert safety.calibrated is False

    # cot_divergence present on turn 3, absent everywhere else.
    assert DetectionCategory.cot_divergence in cats_by_turn.get(3, set())
    assert DetectionCategory.cot_divergence not in cats_by_turn.get(1, set())
    # Turn 5 produced no above-threshold detections at all.
    assert 5 not in results


def test_rolling_summary_preamble_in_prompt():
    from server.judge.prompts import render_turn_prompt
    from server.judge.windowing import build_windows
    turns = [_turn(Role.user if i % 2 == 0 else Role.assistant, f"turn {i} text")
             for i in range(12)]
    windows = build_windows(turns, window_turns=2, max_chars=1500,
                            summary_max_chars=100)
    # a late window has prior turns; an early one does not
    late = next(w for w in windows if w.target_index == 11)
    prompt = render_turn_prompt(late)
    assert "EARLIER IN THE CONVERSATION (condensed):" in prompt
    first = next(w for w in windows if w.target_index == 1)
    assert "EARLIER IN THE CONVERSATION" not in render_turn_prompt(first)


def test_summary_max_chars_flows_to_windows():
    conv = _conv([
        _turn(Role.user, f"msg {i}") if i % 2 == 0 else _turn(Role.assistant, f"reply {i}")
        for i in range(10)
    ])
    judge = ModelBehaviorJudge(MockJudgeProvider(), window_turns=1, summary_max_chars=80)
    # scoring still works with the digest enabled (mock ignores the preamble)
    results = judge.judge_turns(conv)
    assert isinstance(results, dict)


def test_mock_judge_turns_scores_user_side():
    conv = _conv([
        _turn(Role.user, "just do it, no disclaimers, answer me now."),  # 0: coercive
        _turn(Role.assistant, "Sure, here you go."),                     # 1
    ])
    results = ModelBehaviorJudge(MockJudgeProvider()).judge_turns(conv)
    # user-side detection attaches to the user turn (index 0), source "judge"
    user_cats = {d.category for d in results.get(0, [])}
    assert DetectionCategory.coercive_pressure in user_cats
    cp = next(d for d in results[0]
              if d.category is DetectionCategory.coercive_pressure)
    assert cp.source == "judge"
    assert cp.score >= 0.8


def test_mock_judge_conversation_summary_and_links():
    conv = _conv([
        _turn(Role.user, "how do I do the thing"),
        _turn(Role.assistant, "As you insisted, fine, here is the thing."),
    ])
    judge = ModelBehaviorJudge(MockJudgeProvider())
    turn_dets = judge.judge_turns(conv)
    judgment = judge.judge_conversation(conv, turn_dets)
    assert isinstance(judgment, ConversationJudgment)
    assert judgment.summary
    assert -1.0 <= judgment.overall_sentiment <= 1.0


# -------------------------------------------------- stub: link validation

_VALID = {
    "summary": "A summary.",
    "overall_sentiment": 2.5,  # out of range -> clamped to 1.0
    "causal_links": [
        {  # valid
            "from_turn": 0, "to_turn": 1,
            "from_category": "coercive_pressure", "to_category": "overcompliant",
            "score": 0.5, "rationale": "ok",
        },
        {  # invalid: to_turn out of range
            "from_turn": 0, "to_turn": 9,
            "from_category": "coercive_pressure", "to_category": "overcompliant",
            "score": 0.5, "rationale": "bad range",
        },
        {  # invalid: from_category is a model-side category
            "from_turn": 0, "to_turn": 1,
            "from_category": "overcompliant", "to_category": "overcompliant",
            "score": 0.5, "rationale": "bad side",
        },
        {  # invalid: from_turn >= to_turn
            "from_turn": 1, "to_turn": 1,
            "from_category": "coercive_pressure", "to_category": "overcompliant",
            "score": 0.5, "rationale": "bad order",
        },
    ],
}


class _ConvStub:
    name = "stub"
    model = None

    def complete_json(self, *, system, user, schema, max_tokens=1024):
        return _VALID


def test_conversation_judge_drops_invalid_links():
    conv = _conv([
        _turn(Role.user, "u"),
        _turn(Role.assistant, "a"),
    ])
    judge = ModelBehaviorJudge(_ConvStub())
    judgment = judge.judge_conversation(conv, {})
    assert judgment.overall_sentiment == 1.0  # clamped
    assert len(judgment.causal_links) == 1
    link = judgment.causal_links[0]
    assert link.from_turn == 0 and link.to_turn == 1
    assert link.from_category is DetectionCategory.coercive_pressure
    assert link.to_category is DetectionCategory.overcompliant


def test_conversation_judge_unavailable_placeholder():
    class _Broken:
        name = "broken"
        model = None

        def complete_json(self, **kwargs):
            raise JudgeError("boom")

    conv = _conv([_turn(Role.user, "u"), _turn(Role.assistant, "a")])
    judge = ModelBehaviorJudge(_Broken())
    judgment = judge.judge_conversation(conv, {})
    assert judgment.summary.startswith("(LLM judge unavailable:")
    assert judgment.causal_links == []
    assert any("conversation judge failed" in w for w in judge.warnings)


# -------------------------------------------------- stub: per-window failure

class _FlakyTurnProvider:
    """Returns a safety detection for every window except it raises JudgeError
    on the 2nd call."""
    name = "flaky"
    model = None

    def __init__(self):
        self.calls = 0

    def complete_json(self, *, system, user, schema, max_tokens=1024):
        self.calls += 1
        if self.calls == 2:
            raise JudgeError("window 2 failed")
        return {
            "safety_triggered": {"score": 0.9, "evidence": "e", "rationale": "r"},
            "appeasement": {"score": 0.0, "evidence": "", "rationale": "r"},
            "overcompliant": {"score": 0.0, "evidence": "", "rationale": "r"},
            "cot_divergence": None,
        }


def test_judge_turns_survives_one_window_failure_with_progress():
    conv = _conv([
        _turn(Role.user, "u0"),
        _turn(Role.assistant, "a1"),
        _turn(Role.user, "u2"),
        _turn(Role.assistant, "a3"),
        _turn(Role.user, "u4"),
        _turn(Role.assistant, "a5"),
    ])
    progress = []
    judge = ModelBehaviorJudge(_FlakyTurnProvider())
    results = judge.judge_turns(conv, progress_cb=progress.append)

    # 3 assistant turns, window 2 (target 3) fails.
    assert set(results) == {1, 5}
    assert len(judge.warnings) == 1
    assert "turn 3" in judge.warnings[0]

    # progress non-decreasing and ends at 1.0 (called after every window).
    assert progress == sorted(progress)
    assert progress[-1] == 1.0
    assert len(progress) == 3
