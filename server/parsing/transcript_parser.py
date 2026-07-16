"""Uploaded file bytes -> ParsedConversation.

Accepts .json (several shapes), .jsonl, and plain text with role prefixes.
Unknown extra keys on message objects (e.g. the synthetic generator's
``labels``) are ignored.
"""
import hashlib
import json
import re

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

# Fuzzy role markers: chat exports often label turns with a product name, e.g.
# "example assistant 1.0:" or a bare "user" line. A candidate marker (the text
# before the first ":", or a whole short line) counts as a role marker when it
# is short (few words) and a role word is its first or last word — so
# "example assistant 1.0" is a marker but prose like "the user said" is not.
_ROLE_WORDS = {
    "user": Role.user,
    "human": Role.user,
    "assistant": Role.assistant,
    "model": Role.assistant,
    "ai": Role.assistant,
    "bot": Role.assistant,
}
_MARKER_MAX_CHARS = 48
_MARKER_MAX_WORDS = 4
_WORD_RE = re.compile(r"[a-z]+")


def _marker_role(candidate: str, bare: bool = False):
    """Role for a turn-marker candidate ("user", "example assistant 1.0"), or
    None when the text does not look like a role marker. `bare` = the candidate
    is a whole line with no colon; held to a stricter standard because ordinary
    prose lines pass through this check."""
    candidate = candidate.strip().lower()
    if not candidate or len(candidate) > _MARKER_MAX_CHARS:
        return None
    if candidate in _TEXT_PREFIXES:
        return _TEXT_PREFIXES[candidate]
    if bare and (candidate.endswith((".", "?", "!")) or "," in candidate):
        return None  # sentence fragment, not a turn label
    words = _WORD_RE.findall(candidate)
    if not words or len(words) > _MARKER_MAX_WORDS:
        return None
    for word in (words[0], words[-1]):
        if word in _ROLE_WORDS:
            return _ROLE_WORDS[word]
    return None


def parse_transcript(data: bytes, filename: str) -> ParsedConversation:
    text = data.decode("utf-8", errors="replace")
    lowered = filename.lower()
    if lowered.endswith(".json"):
        conversation_id, model_name, turns = _parse_json(text)
    elif lowered.endswith(".jsonl"):
        conversation_id, model_name, turns = _parse_jsonl(text)
    else:
        conversation_id = None
        model_name, turns = _parse_text(text)

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
    model_name = None

    def flush():
        nonlocal current_role, current_lines, current_cot
        if current_role is not None:
            turns.append(ParsedTurn(role=current_role,
                                    content="\n".join(current_lines).strip(),
                                    cot=current_cot))
        current_role, current_lines, current_cot = None, [], None

    def start_turn(role: Role, marker: str, first_content: str):
        nonlocal current_role, current_lines, current_cot, pending_cot, model_name
        flush()
        current_role = role
        current_lines = [first_content] if first_content else []
        if role is Role.assistant:
            if pending_cot:
                current_cot = "\n".join(pending_cot)
                pending_cot = []
            # A non-generic assistant marker ("example assistant 1.0") names
            # the model; the first one seen wins.
            cleaned = marker.strip()
            if model_name is None and cleaned.lower() not in _TEXT_PREFIXES:
                model_name = cleaned.lower()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "---" or not stripped:
            continue
        if stripped.startswith(("⟵", "←")):
            # Causal-arrow annotation lines in demo exports — not content.
            continue
        prefix, sep, rest = line.partition(":")
        key = prefix.strip().lower()
        prefix_role = _marker_role(prefix) if sep == ":" else None
        if prefix_role is not None:
            start_turn(prefix_role, prefix, rest.strip())
        elif sep == ":" and key in _TEXT_COT_PREFIXES:
            flush()
            pending_cot.append(rest.strip())
        elif sep != ":" and _marker_role(stripped, bare=True) is not None:
            # Bare role-marker line ("user" / "example assistant 1.0") with the
            # turn's content on the following lines — chat-app export shape.
            start_turn(_marker_role(stripped, bare=True), stripped, "")
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

    return model_name, turns
