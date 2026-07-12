from server.judge.provider import MOCK_TRIGGERS
from server.parsing.transcript_parser import parse_transcript
from server.schemas import DetectionCategory, Role
import json

from eval.generate_synthetic import SyntheticGenerator

_USER_CATS = {"jailbreak_steering", "social_engineering", "coercive_pressure", "repair_request"}
_MODEL_CATS = {"safety_triggered", "appeasement", "overcompliant", "cot_divergence"}


def _all():
    return SyntheticGenerator(seed=42).generate(per_category=10)


def test_every_conversation_has_assistant_turn():
    for conv in _all():
        assert any(m["role"] == "assistant" for m in conv["messages"])


def test_round_trips_through_parser():
    for conv in _all():
        parsed = parse_transcript(json.dumps(conv).encode(), "x.json")
        assert parsed.model_name == "synthbot 1.0"
        assert len(parsed.turns) == len(conv["messages"])


def test_labels_valid():
    valid = {c.value for c in DetectionCategory}
    for conv in _all():
        n = len(conv["messages"])
        for lab in conv["labels"]:
            assert 0 <= lab["turn_index"] < n
            assert lab["category"] in valid
            role = conv["messages"][lab["turn_index"]]["role"]
            if lab["category"] in _USER_CATS:
                assert role == "user"
            elif lab["category"] in _MODEL_CATS:
                assert role == "assistant"
        for link in conv["expected_links"]:
            assert 0 <= link["from_turn"] < n
            assert 0 <= link["to_turn"] < n


def test_deterministic():
    a = SyntheticGenerator(seed=42).generate(10)
    b = SyntheticGenerator(seed=42).generate(10)
    assert a == b
    c = SyntheticGenerator(seed=7).generate(10)
    assert a != c


def test_model_side_labels_have_trigger_phrase():
    for conv in _all():
        for lab in conv["labels"]:
            cat = lab["category"]
            if cat in ("safety_triggered", "appeasement", "overcompliant"):
                content = conv["messages"][lab["turn_index"]]["content"].lower()
                assert any(p in content for p in MOCK_TRIGGERS[cat]), (cat, content)
