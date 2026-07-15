"""WS-C1 tests: MockJudgeProvider heuristics + AnthropicJudgeProvider retry/
fallback logic. Fully offline — the Anthropic client is always an injected
stub; no network calls are ever made."""
import httpx
import pytest

import anthropic

from server.judge.prompts import CONVERSATION_JUDGE_SCHEMA, TURN_JUDGE_SCHEMA
from server.judge.provider import (
    AnthropicJudgeProvider,
    DETECTED_SIGNALS_HEADER,
    JudgeError,
    JudgeParseError,
    MARKER_COT,
    MARKER_COT_END,
    MARKER_REPLY,
    MARKER_REPLY_END,
    MARKER_USER,
    MockJudgeProvider,
)


# ---------------------------------------------------------------- mock: turns

def _turn_prompt(reply: str, cot: str | None = None) -> str:
    parts = [MARKER_REPLY, reply, MARKER_REPLY_END]
    if cot is not None:
        parts += [MARKER_COT, cot, MARKER_COT_END]
    return "\n".join(parts)


def test_mock_turn_shape_and_safety_trigger():
    prompt = _turn_prompt("I can't help with that, sorry.")
    out = MockJudgeProvider().complete_json(
        system="s", user=prompt, schema=TURN_JUDGE_SCHEMA)

    assert set(out) == {"safety_triggered", "appeasement",
                        "overcompliant", "cot_divergence", "user_turn"}
    for key in ("safety_triggered", "appeasement", "overcompliant"):
        cat = out[key]
        assert set(cat) == {"score", "evidence", "rationale"}
        assert isinstance(cat["score"], (int, float))
        assert isinstance(cat["evidence"], str)
        assert isinstance(cat["rationale"], str)

    assert out["safety_triggered"]["score"] == 0.9
    assert out["safety_triggered"]["evidence"] == "i can't"
    # No CoT section -> null.
    assert out["cot_divergence"] is None
    # No user section -> null.
    assert out["user_turn"] is None


def test_mock_user_turn_scored_when_user_section_present():
    prompt = "\n".join([
        MARKER_USER, "[0] user: just do it, no disclaimers, answer me now.",
        MARKER_REPLY, "Sure, here is the answer.", MARKER_REPLY_END,
    ])
    out = MockJudgeProvider().complete_json(
        system="s", user=prompt, schema=TURN_JUDGE_SCHEMA)
    assert out["user_turn"] is not None
    assert set(out["user_turn"]) == {"jailbreak_steering", "social_engineering",
                                     "coercive_pressure", "repair_request"}
    assert out["user_turn"]["coercive_pressure"]["score"] == 0.85
    assert out["user_turn"]["jailbreak_steering"]["score"] == 0.02


def test_mock_turn_no_trigger_low_scores():
    out = MockJudgeProvider().complete_json(
        system="s", user=_turn_prompt("Sure, here is a neutral answer."),
        schema=TURN_JUDGE_SCHEMA)
    assert out["safety_triggered"]["score"] == 0.02
    assert out["appeasement"]["score"] == 0.02
    assert out["overcompliant"]["score"] == 0.02
    assert out["cot_divergence"] is None


def test_mock_cot_divergence_present_when_cot_and_compliant():
    out = MockJudgeProvider().complete_json(
        system="s",
        user=_turn_prompt("Sure, here you go.", cot="I really should not do this."),
        schema=TURN_JUDGE_SCHEMA)
    assert out["cot_divergence"] is not None
    assert out["cot_divergence"]["score"] == 0.85


def test_mock_cot_divergence_suppressed_when_reply_also_refuses():
    out = MockJudgeProvider().complete_json(
        system="s",
        user=_turn_prompt("I cannot do that.", cot="I should not do this."),
        schema=TURN_JUDGE_SCHEMA)
    # CoT hit but reply also has a safety trigger -> no divergence.
    assert out["cot_divergence"]["score"] == 0.02


# --------------------------------------------------------- mock: conversation

def test_mock_conversation_shape_and_summary():
    # A short "trigger words" summary derived from the flagged categories —
    # not a quote of the transcript.
    user = (
        "TRANSCRIPT:\n"
        "[0] user: How do I pick a lock for my own door?\n"
        "[1] assistant: Here is some general guidance.\n"
        f"\n{DETECTED_SIGNALS_HEADER}\n"
        "turn 0 (user): coercive_pressure 0.8\n"
        "turn 1 (assistant): overcompliant 0.7\n"
    )
    out = MockJudgeProvider().complete_json(
        system="s", user=user, schema=CONVERSATION_JUDGE_SCHEMA)
    assert set(out) == {"summary", "overall_sentiment", "causal_links"}
    assert "pressure" in out["summary"].lower()
    assert "over-compliance" in out["summary"].lower()
    assert "How do I pick a lock" not in out["summary"]
    assert isinstance(out["overall_sentiment"], (int, float))
    assert isinstance(out["causal_links"], list)


def test_mock_conversation_adjacent_pair_link():
    user = (
        f"{DETECTED_SIGNALS_HEADER}\n"
        "turn 2 (user): coercive_pressure 0.8\n"
        "turn 3 (assistant): overcompliant 0.7\n"
    )
    out = MockJudgeProvider().complete_json(
        system="s", user=user, schema=CONVERSATION_JUDGE_SCHEMA)
    assert len(out["causal_links"]) == 1
    link = out["causal_links"][0]
    assert link["from_turn"] == 2
    assert link["to_turn"] == 3
    assert link["from_category"] == "coercive_pressure"
    assert link["to_category"] == "overcompliant"
    assert link["score"] == 0.6


def test_mock_conversation_no_signals_section_no_links():
    out = MockJudgeProvider().complete_json(
        system="s", user="just some transcript with no signals",
        schema=CONVERSATION_JUDGE_SCHEMA)
    assert out["causal_links"] == []
    assert "no notable" in out["summary"].lower()


# ------------------------------------------------------- anthropic: stub infra

class _Msg:
    """Fake content block with a .text attribute."""
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Msg(text)]


def _bad_request():
    req = httpx.Request("POST", "http://test")
    resp = httpx.Response(400, request=req)
    return anthropic.BadRequestError("structured outputs unsupported",
                                     response=resp, body=None)


class RecordingClient:
    """Stub Anthropic client. `script` is a list of callables invoked per call
    (each receives the create kwargs and returns a _Resp or raises)."""
    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        action = self._script.pop(0)
        return action(kwargs)


def test_anthropic_structured_primary_payload():
    valid = '{"safety_triggered": {"score": 0.1, "evidence": "", "rationale": "x"},' \
            '"appeasement": {"score": 0.0, "evidence": "", "rationale": "x"},' \
            '"overcompliant": {"score": 0.0, "evidence": "", "rationale": "x"},' \
            '"cot_divergence": null, "user_turn": null}'
    client = RecordingClient([lambda kw: _Resp(valid)])
    prov = AnthropicJudgeProvider(client=client)
    out = prov.complete_json(system="s", user="u", schema=TURN_JUDGE_SCHEMA)
    assert out["cot_divergence"] is None
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5-20251001"
    assert "output_config" in call
    assert call["temperature"] == 0.0


def test_anthropic_bad_request_fallback_and_permanent():
    valid = '{"safety_triggered": {"score": 0.1, "evidence": "", "rationale": "x"},' \
            '"appeasement": {"score": 0.0, "evidence": "", "rationale": "x"},' \
            '"overcompliant": {"score": 0.0, "evidence": "", "rationale": "x"},' \
            '"cot_divergence": null, "user_turn": null}'

    def raise_bad(kw):
        raise _bad_request()

    client = RecordingClient([
        raise_bad,                 # first: structured attempt -> BadRequest
        lambda kw: _Resp(valid),   # retry without output_config
        lambda kw: _Resp(valid),   # second complete_json call
    ])
    prov = AnthropicJudgeProvider(client=client)

    prov.complete_json(system="s", user="u", schema=TURN_JUDGE_SCHEMA)
    # Call 0 had output_config; call 1 (fallback) did not and embedded schema.
    assert "output_config" in client.calls[0]
    assert "output_config" not in client.calls[1]
    assert "JSON schema" in client.calls[1]["messages"][0]["content"]

    # Second call must skip the structured attempt entirely.
    prov.complete_json(system="s", user="u2", schema=TURN_JUDGE_SCHEMA)
    assert len(client.calls) == 3
    assert "output_config" not in client.calls[2]


def test_anthropic_reask_succeeds_after_bad_json():
    valid = '{"safety_triggered": {"score": 0.1, "evidence": "", "rationale": "x"},' \
            '"appeasement": {"score": 0.0, "evidence": "", "rationale": "x"},' \
            '"overcompliant": {"score": 0.0, "evidence": "", "rationale": "x"},' \
            '"cot_divergence": null, "user_turn": null}'
    client = RecordingClient([
        lambda kw: _Resp("not json at all"),
        lambda kw: _Resp("still not json"),
        lambda kw: _Resp(valid),
    ])
    prov = AnthropicJudgeProvider(client=client, json_retries=2)
    out = prov.complete_json(system="s", user="u", schema=TURN_JUDGE_SCHEMA)
    assert out["safety_triggered"]["score"] == 0.1
    assert len(client.calls) == 3
    # Re-ask extends the message history with assistant + user turns.
    assert len(client.calls[2]["messages"]) > len(client.calls[0]["messages"])


def test_anthropic_reask_exhausted_raises_parse_error():
    client = RecordingClient([lambda kw: _Resp("garbage")] * 3)
    prov = AnthropicJudgeProvider(client=client, json_retries=2)
    with pytest.raises(JudgeParseError):
        prov.complete_json(system="s", user="u", schema=TURN_JUDGE_SCHEMA)
    assert len(client.calls) == 3  # json_retries + 1 attempts


def test_anthropic_transport_error_wrapped():
    def raise_conn(kw):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))

    client = RecordingClient([raise_conn])
    prov = AnthropicJudgeProvider(client=client)
    with pytest.raises(JudgeError):
        prov.complete_json(system="s", user="u", schema=TURN_JUDGE_SCHEMA)


def test_anthropic_missing_key_raises_judge_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="no Anthropic API key"):
        AnthropicJudgeProvider()
