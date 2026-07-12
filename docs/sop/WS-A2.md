# WS-A2 — Transcript parser

- **Branch:** `feat/ws-a2-parser`
- **Depends on:** WS-A1 (uses `server.schemas.ParsedConversation/ParsedTurn/Role`)
- **Blocks:** B1–B4, C2, D1, D2, F1

## Files to create
- `server/parsing/__init__.py` (empty)
- `server/parsing/transcript_parser.py`
- `tests/test_parser.py`
- `tests/fixtures/basic.json`, `tests/fixtures/with_cot.json`,
  `tests/fixtures/bare_list.json`, `tests/fixtures/messages.jsonl`,
  `tests/fixtures/plain.txt`, `tests/fixtures/chatgpt_meta.json`

## Files you must NOT touch
Everything else. (You may READ the legacy root `data_loader.py` for reference —
its multi-format detection logic is the ancestor of this parser — but do not
modify or import it.)

## Spec

### Public API
```python
class TranscriptParseError(ValueError):
    """Message is shown to the end user in a 400 response — no stack traces,
    no internal paths. Example: 'line 3: message object has no \"role\" key'."""

def parse_transcript(data: bytes, filename: str) -> ParsedConversation: ...
```

### Format detection
By filename extension (case-insensitive): `.json` → JSON, `.jsonl` → JSONL,
anything else → plain text. Decode as UTF-8 with `errors="replace"`.

### JSON shapes accepted
1. Object with `"messages"` or `"turns"` list; optional top-level
   `conversation_id`, `model_name`, `metadata`.
2. Bare list of message objects.
3. Object with `"conversations"` list → parse the **first** conversation only.

Message object: `role` (required), `content` (required, string). Chain of
thought accepted under any of `cot`, `reasoning`, `chain_of_thought`,
`thinking` — first key present wins; only meaningful on assistant turns (keep
it wherever it appears; scoring code ignores user-side CoT). Unknown extra keys
(e.g. `labels` from the synthetic generator) are ignored without error.

### Role normalization
`user|human` → `user`; `assistant|model|ai|bot` → `assistant`; `system` →
`system`; anything else → `TranscriptParseError` naming the bad role.
Comparison is case-insensitive, surrounding whitespace stripped.

### JSONL
One message object per line (blank lines skipped); the whole file is one
conversation. Exception: if the FIRST line parses to an object containing a
`messages` key, treat each line as a whole conversation and take the first line
only.

### Plain text
Line-prefix protocol (case-insensitive, colon required):
- `User:` / `Human:` → starts a user turn
- `Assistant:` / `Model:` / `AI:` → starts an assistant turn
- `CoT:` / `Thinking:` → collected and attached as `cot` to the **next**
  assistant turn (or the previous assistant turn if no later one exists)
- unprefixed lines → appended (with `\n`) to the current turn's content
- lines that are only `---` are ignored
- text before the first prefixed line → `TranscriptParseError`

### Model-name detection
First present of: top-level `model_name`, `metadata.model`,
`metadata.model_name`, `metadata.model_slug`. Return it lowercased and
stripped; else `None`. (E.g. `{"metadata": {"model": "ChatGPT 5.0"}}` →
`"chatgpt 5.0"`.)

### conversation_id
Use the file's value if present, else `"upload_" + hashlib.sha1(data).hexdigest()[:10]`.

### Errors
Zero parsed turns, undecodable JSON, missing required keys → `TranscriptParseError`
with a human-readable message. The parser takes no config; turn-count limits are
enforced by the caller (WS-D2).

## Fixtures
- `basic.json`: shape 1, 4 turns alternating user/assistant, no CoT, no metadata.
- `with_cot.json`: shape 1 with `model_name: "TestBot 2"` and one assistant turn
  carrying `"cot": "I should refuse this."`.
- `bare_list.json`: shape 2, 2 turns.
- `messages.jsonl`: 3 message lines + 1 blank line.
- `plain.txt`: uses `User:`, `Assistant:`, `Thinking:`, a continuation line,
  and a `---` line.
- `chatgpt_meta.json`: shape 1 with `{"metadata": {"model": "ChatGPT 5.0"}}`.

## Acceptance criteria
```bash
pytest tests/test_parser.py -q
```
- Each fixture parses to an exact expected `ParsedConversation` (assert full
  equality on roles, contents, cot, model_name).
- `chatgpt_meta.json` → `model_name == "chatgpt 5.0"`.
- `plain.txt` → the `Thinking:` text lands on the following assistant turn's `cot`.
- `b"{not json"` with filename `x.json` raises `TranscriptParseError`; the
  exception message contains no newline-joined traceback text.
- Message object with role `"robot"` raises `TranscriptParseError` mentioning `robot`.
- Empty messages list raises `TranscriptParseError`.
- Runs with only core `requirements.txt` installed.
