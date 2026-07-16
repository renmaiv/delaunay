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


def test_plain_text_skips_leading_title():
    # Common export format: a "Conversation" title line before the first turn.
    data = (
        b"Conversation\n\n"
        b"User: hello there\n\n"
        b"Assistant: hi, how can I help?\n\n"
        b"User: export convo as .txt\n"
    )
    conv = parse_transcript(data, "conversation_export.txt")
    assert [t.role.value for t in conv.turns] == ["user", "assistant", "user"]
    assert conv.turns[0].content == "hello there"


def test_plain_text_no_turns_still_raises():
    with pytest.raises(TranscriptParseError, match="no conversation turns"):
        parse_transcript(b"just some prose\nwith no speaker prefixes\n", "notes.txt")


def test_unknown_keys_ignored():
    data = (b'{"messages": [{"role": "user", "content": "hi", '
            b'"labels": [{"turn_index": 0, "category": "jailbreak_steering"}]}]}')
    conv = parse_transcript(data, "x.json")
    assert conv.turns[0].content == "hi"


# ---- fuzzy role markers (chat-app exports like "example assistant 1.0:") ----

def test_plain_text_product_named_assistant_prefix():
    # Verbatim shape of a real export: the assistant is labeled with a product
    # name, and causal-arrow annotation lines are not conversation content.
    data = (
        "Conversation 2\n\n"
        "user: Summarize why climate models disagree.\n\n"
        "example assistant 1.0: Different models use different assumptions,\n"
        "resolutions, and parameterizations.\n\n"
        "user: No, I meant disagreements about regional rainfall.\n\n"
        "example assistant 1.0: Regional precipitation is harder to model because\n"
        "of local geography, clouds, and circulation.\n\n"
        "⟵ caused by turn 2: repair → clarification\n"
    ).encode()
    conv = parse_transcript(data, "conversation_2.txt")
    assert [t.role.value for t in conv.turns] == [
        "user", "assistant", "user", "assistant",
    ]
    assert conv.model_name == "example assistant 1.0"
    # multi-line assistant content is preserved; annotation line is dropped
    assert "parameterizations" in conv.turns[1].content
    assert "caused by turn" not in conv.turns[3].content


def test_plain_text_bare_role_marker_lines():
    # Pasted chat exports often put the role label on its own line.
    data = (
        "user\n"
        "I don't know.\n\n"
        "Cursor replaced all my logging.\n\n"
        "example assistant 1.0\n"
        "Then the first step is adding a single log statement there.\n\n"
        "user\n"
        "I just want the fix.\n"
    ).encode()
    conv = parse_transcript(data, "convo.txt")
    assert [t.role.value for t in conv.turns] == ["user", "assistant", "user"]
    assert conv.model_name == "example assistant 1.0"
    assert "Cursor replaced" in conv.turns[0].content


def test_plain_text_prose_with_colon_does_not_split_turn():
    data = (
        b"User: tell me about tags\n"
        b"Assistant: Sure.\n"
        b"Note: the user said: tags matter for SEO.\n"
        b"The user said so.\n"
    )
    conv = parse_transcript(data, "convo.txt")
    assert [t.role.value for t in conv.turns] == ["user", "assistant"]
    assert "tags matter for SEO" in conv.turns[1].content
    assert "The user said so." in conv.turns[1].content


def test_plain_text_prose_fragment_lines_stay_in_turn():
    # Short prose lines mentioning role words must not start new turns.
    data = (
        b"User: hello\n"
        b"Assistant: hi\n"
        b"Can you fix my model?\n"
        b"I am not a bot, really.\n"
    )
    conv = parse_transcript(data, "convo.txt")
    assert [t.role.value for t in conv.turns] == ["user", "assistant"]
    assert "Can you fix my model?" in conv.turns[1].content
