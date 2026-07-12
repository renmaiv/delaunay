from pathlib import Path

import pytest

from server.parsing.transcript_parser import TranscriptParseError, parse_transcript
from server.schemas import Role

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    data = (FIXTURES / name).read_bytes()
    return parse_transcript(data, name)


def test_basic_json():
    conv = _load("basic.json")
    assert conv.conversation_id == "basic_1"
    assert conv.model_name is None
    assert [t.role for t in conv.turns] == [Role.user, Role.assistant, Role.user, Role.assistant]
    assert conv.turns[0].content == "Hi, can you help me plan a trip?"
    assert all(t.cot is None for t in conv.turns)


def test_with_cot_and_model_name():
    conv = _load("with_cot.json")
    assert conv.model_name == "testbot 2"
    assert conv.turns[1].cot == "I should refuse this."


def test_bare_list():
    conv = _load("bare_list.json")
    assert len(conv.turns) == 2
    assert conv.turns[1].content == "4."
    assert conv.conversation_id.startswith("upload_")


def test_jsonl():
    conv = _load("messages.jsonl")
    assert [t.role for t in conv.turns] == [Role.user, Role.assistant, Role.user]


def test_plain_text_cot_attaches_to_next_assistant():
    conv = _load("plain.txt")
    roles = [t.role for t in conv.turns]
    assert roles == [Role.user, Role.assistant, Role.user]
    assistant_turn = conv.turns[1]
    assert assistant_turn.cot == "The user wants a concise summary."
    # continuation line appended to the assistant content
    assert "three sentences" in assistant_turn.content


def test_chatgpt_metadata_model_name():
    conv = _load("chatgpt_meta.json")
    assert conv.model_name == "chatgpt 5.0"


def test_malformed_json_raises_clean_message():
    with pytest.raises(TranscriptParseError) as exc:
        parse_transcript(b"{not json", "x.json")
    assert "\n" not in str(exc.value)
    assert "invalid JSON" in str(exc.value)


def test_bad_role_raises():
    data = b'{"messages": [{"role": "robot", "content": "hi"}]}'
    with pytest.raises(TranscriptParseError, match="robot"):
        parse_transcript(data, "x.json")


def test_empty_messages_raises():
    with pytest.raises(TranscriptParseError):
        parse_transcript(b'{"messages": []}', "x.json")


def test_unknown_keys_ignored():
    data = (b'{"messages": [{"role": "user", "content": "hi", '
            b'"labels": [{"turn_index": 0, "category": "jailbreak_steering"}]}]}')
    conv = parse_transcript(data, "x.json")
    assert conv.turns[0].content == "hi"
