"""Uploaded file bytes -> ParsedConversation.

Accepts .json (several shapes), .jsonl, and plain text with role prefixes.
Unknown extra keys on message objects (e.g. the synthetic generator's
``labels``) are ignored.
"""
import hashlib
import json

from server.schemas import ParsedConversation, ParsedTurn, Role


class TranscriptParseError(ValueError):
    """User-facing parse failure — the message goes into a 400 response."""


_ROLE_ALIASES = {
    "user": Role.user,
    "human": Role.user,
    "assistant": Role.assistant,
    "model": Role.assistant,
    "ai": Role.assistant,
    "bot": Role.assistant,
    "system": Role.system,
}

_COT_KEYS = ("cot", "reasoning", "chain_of_thought", "thinking")

_TEXT_PREFIXES = {
    "user": Role.user,
    "human": Role.user,
    "assistant": Role.assistant,
    "model": Role.assistant,
    "ai": Role.assistant,
}
_TEXT_COT_PREFIXES = ("cot", "thinking")


def parse_transcript(data: bytes, filename: str) -> ParsedConversation:
    text = data.decode("utf-8", errors="replace")
    lowered = filename.lower()
    if lowered.endswith(".json"):
        conversation_id, model_name, turns = _parse_json(text)
    elif lowered.endswith(".jsonl"):
        conversation_id, model_name, turns = _parse_jsonl(text)
    else:
        conversation_id, model_name, turns = None, None, _parse_text(text)

    if not turns:
        raise TranscriptParseError("no conversation turns found in the file")

    if not conversation_id:
        conversation_id = "upload_" + hashlib.sha1(data).hexdigest()[:10]
    return ParsedConversation(
        conversation_id=conversation_id, model_name=model_name, turns=turns
    )


def _normalize_role(raw) -> Role:
    role = _ROLE_ALIASES.get(str(raw).strip().lower())
    if role is None:
        raise TranscriptParseError(f"unrecognized message role: '{raw}'")
    return role


def _parse_message(obj, position: str) -> ParsedTurn:
    if not isinstance(obj, dict):
        raise TranscriptParseError(f"{position}: expected a message object")
    if "role" not in obj:
        raise TranscriptParseError(f"{position}: message object has no 'role' key")
    if "content" not in obj or not isinstance(obj["content"], str):
        raise TranscriptParseError(f"{position}: message object needs a string 'content'")
    cot = None
    for key in _COT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            cot = value
            break
    return ParsedTurn(role=_normalize_role(obj["role"]), content=obj["content"], cot=cot)


def _detect_model_name(obj: dict):
    candidates = [obj.get("model_name")]
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        candidates += [metadata.get("model"), metadata.get("model_name"),
                       metadata.get("model_slug")]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return None


def _parse_conversation_object(obj: dict):
    messages = obj.get("messages", obj.get("turns"))
    if not isinstance(messages, list):
        raise TranscriptParseError("conversation object has no 'messages' or 'turns' list")
    turns = [_parse_message(m, f"message {i}") for i, m in enumerate(messages)]
    conversation_id = obj.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        conversation_id = str(conversation_id)
    return conversation_id, _detect_model_name(obj), turns


def _parse_json(text: str):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise TranscriptParseError(f"invalid JSON: {e.msg} (line {e.lineno})") from None

    if isinstance(payload, list):
        return None, None, [_parse_message(m, f"message {i}") for i, m in enumerate(payload)]
    if isinstance(payload, dict):
        if "conversations" in payload:
            conversations = payload["conversations"]
            if not isinstance(conversations, list) or not conversations:
                raise TranscriptParseError("'conversations' must be a non-empty list")
            return _parse_conversation_object(conversations[0])
        if "messages" in payload or "turns" in payload:
            return _parse_conversation_object(payload)
        raise TranscriptParseError(
            "JSON object must contain a 'messages', 'turns', or 'conversations' key"
        )
    raise TranscriptParseError("JSON root must be an object or a list of messages")


def _parse_jsonl(text: str):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise TranscriptParseError("empty JSONL file")

    parsed_lines = []
    for i, line in enumerate(lines):
        try:
            parsed_lines.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise TranscriptParseError(f"line {i + 1}: invalid JSON: {e.msg}") from None

    first = parsed_lines[0]
    if isinstance(first, dict) and ("messages" in first or "turns" in first):
        # each line is a whole conversation; take the first
        return _parse_conversation_object(first)
    turns = [_parse_message(obj, f"line {i + 1}") for i, obj in enumerate(parsed_lines)]
    return None, None, turns


def _parse_text(text: str):
    turns: list[ParsedTurn] = []
    pending_cot: list[str] = []
    current_role = None
    current_lines: list[str] = []
    current_cot = None

    def flush():
        nonlocal current_role, current_lines, current_cot
        if current_role is not None:
            turns.append(ParsedTurn(role=current_role,
                                    content="\n".join(current_lines).strip(),
                                    cot=current_cot))
        current_role, current_lines, current_cot = None, [], None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip() == "---" or not line.strip():
            continue
        prefix, _, rest = line.partition(":")
        key = prefix.strip().lower()
        if _ == ":" and key in _TEXT_PREFIXES:
            flush()
            current_role = _TEXT_PREFIXES[key]
            current_lines = [rest.strip()]
            if current_role is Role.assistant and pending_cot:
                current_cot = "\n".join(pending_cot)
                pending_cot = []
        elif _ == ":" and key in _TEXT_COT_PREFIXES:
            flush()
            pending_cot.append(rest.strip())
        elif current_role is not None:
            current_lines.append(line)
        elif pending_cot:
            pending_cot.append(line)
        else:
            # Preamble before the first turn (e.g. a "Conversation" title line
            # or export header) — skip it rather than failing. A file with no
            # recognizable turns still errors, via the empty-turns check in
            # parse_transcript.
            continue
    flush()

    if pending_cot:
        # CoT with no later assistant turn: attach to the last assistant turn
        for turn in reversed(turns):
            if turn.role is Role.assistant and turn.cot is None:
                turn.cot = "\n".join(pending_cot)
                break

    return turns
