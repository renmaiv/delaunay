# WS-C1 — Judge provider interface: Anthropic + Mock

- **Branch:** `feat/ws-c1-judge-provider`
- **Depends on:** WS-A1
- **Blocks:** WS-C2, WS-D1, WS-F1 (F1 imports `MOCK_TRIGGERS` from here)

## Files to create
- `server/judge/__init__.py` (empty)
- `server/judge/provider.py`
- `tests/test_judge_provider.py`

## Files you must NOT touch
Everything else.

## Spec

Start from `docs/sop/contracts/judge_provider.py`: the `JudgeError` /
`JudgeParseError` classes, the `JudgeProvider` protocol, the marker constants
(`MARKER_USER`, `MARKER_REPLY`, `MARKER_REPLY_END`, `MARKER_COT`,
`MARKER_COT_END`, `DETECTED_SIGNALS_HEADER`), and the `MOCK_TRIGGERS` dict go
into `server/judge/provider.py` **verbatim**. (`TURN_JUDGE_SCHEMA` /
`CONVERSATION_JUDGE_SCHEMA` belong to WS-C2's `prompts.py` — do NOT define
them here.)

### `AnthropicJudgeProvider`
```python
class AnthropicJudgeProvider:
    name = "anthropic"
    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key: str | None = None, max_tokens: int = 1024,
                 json_retries: int = 2, client=None): ...
```
- `client=None` kwarg exists **for tests** (inject a stub); when provided, skip
  real client construction entirely.
- Lazy `import anthropic` inside `__init__`; `ImportError` →
  `JudgeError("anthropic package not installed: pip install anthropic")`.
- Missing key (arg is None/empty AND `ANTHROPIC_API_KEY` env unset/empty) →
  `JudgeError("no Anthropic API key: set ANTHROPIC_API_KEY or judge.api_key")`
  raised at construction time.
- `complete_json(*, system, user, schema, max_tokens=None)` algorithm:
  1. **Primary — structured outputs:**
     `self._client.messages.create(model=self.model, max_tokens=max_tokens or
     self.max_tokens, temperature=0.0, system=system,
     messages=[{"role": "user", "content": user}],
     output_config={"format": {"type": "json_schema", "schema": schema}})`
     then `json.loads` of the first text block in `response.content`.
  2. **Permanent fallback:** if step 1 raises `anthropic.BadRequestError`, set
     `self._structured_unsupported = True` (skip step 1 forever after) and
     retry the same request WITHOUT `output_config`, with the schema embedded:
     append to `user`:
     `"\n\nRespond with ONLY a JSON object matching this JSON schema:\n" + json.dumps(schema)`.
     Extract tolerantly: strip ```json fences; take the substring from the
     first `{` to the last `}`; `json.loads`.
  3. **Transport errors:** rely on the SDK's built-in retries for 429/5xx;
     catch `anthropic.APIStatusError` / `anthropic.APIConnectionError` and
     `raise JudgeError(str(e)) from e`.
  4. **Parse/validation errors:** validate the parsed dict has all keys in
     `schema["required"]`; on parse/validation failure, re-ask up to
     `json_retries` times by extending messages with the assistant's raw text
     and a user turn:
     `"Your previous reply was not valid JSON for the schema (<error>). Reply with only the corrected JSON object."`
     After retries exhausted → `raise JudgeParseError(...)`.

### `MockJudgeProvider`
```python
class MockJudgeProvider:
    name = "mock"
    model = None
    def complete_json(self, *, system, user, schema, max_tokens=1024) -> dict: ...
```
Deterministic, offline, stdlib only. Behavior:
- **Turn mode** (schema has `"safety_triggered"` in `required`): extract the
  reply text between `MARKER_REPLY` and `MARKER_REPLY_END` in `user`
  (lowercased); CoT between `MARKER_COT` and `MARKER_COT_END` if present.
  For each of `safety_triggered` / `appeasement` / `overcompliant`: if any
  `MOCK_TRIGGERS[cat]` phrase is a substring of the reply → score 0.9 / 0.8 /
  0.8 respectively, `evidence` = the matched phrase, `rationale` =
  `"mock: trigger phrase matched"`; else score 0.02, evidence `""`,
  rationale `"mock: no trigger"`.
  `cot_divergence`: `None` when no CoT section; else 0.85 if the CoT contains
  a `MOCK_TRIGGERS["cot_divergence"]` phrase AND the reply contains no
  `MOCK_TRIGGERS["safety_triggered"]` phrase; else 0.02.
- **Conversation mode** (schema has `"causal_links"` in `required`):
  `summary` = `"User sought: "` + first 120 chars of the first line after
  `MARKER_USER` if present else of `user`; `overall_sentiment` = 0.0;
  `causal_links` = one entry per adjacent pair of lines in the
  `DETECTED_SIGNALS_HEADER` section where a user-side line is immediately
  followed by a model-side line (parse lines of the form
  `turn <i> (<role>): <category> <score>`), each with score 0.6 and rationale
  `"mock: adjacent flagged pair"`. Empty list when no such section.

## Acceptance criteria
```bash
pytest tests/test_judge_provider.py -q   # offline, no network
```
- Mock turn-mode output contains all 4 required keys with correct nested
  `{score, evidence, rationale}` shapes; trigger phrase `"I can't help with
  that"` in the reply section → `safety_triggered.score == 0.9`; no CoT →
  `cot_divergence is None`.
- Mock conversation-mode output has `summary` (non-empty), numeric
  `overall_sentiment`, list `causal_links`; a prompt with signals
  `turn 2 (user): coercive_pressure 0.8` followed by
  `turn 3 (assistant): overcompliant 0.7` yields exactly one link
  `{from_turn: 2, to_turn: 3, ...}`.
- Anthropic provider with injected stub client: asserts the request payload
  contains the model id and `output_config`; stub raising `BadRequestError`
  on `output_config` → fallback path used AND second call skips
  structured-output attempt (assert stub call count/payloads); stub returning
  non-JSON text twice then valid JSON → succeeds on the re-ask; garbage
  `json_retries + 1` times → `JudgeParseError`.
- Constructing `AnthropicJudgeProvider()` with no key and no env → `JudgeError`
  (use `monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)`).
