# WS-C2 — Judge core: prompts, windowing, orchestration

- **Branch:** `feat/ws-c2-judge-core`
- **Depends on:** WS-A1, WS-A2, WS-C1
- **Blocks:** WS-D1

## Files to create
- `server/judge/prompts.py`
- `server/judge/windowing.py`
- `server/judge/judge.py`
- `tests/test_windowing.py`, `tests/test_judge.py`

## Files you must NOT touch
Everything else. Import markers/triggers from `server.judge.provider` — never
retype the literal strings.

## Spec

### `server/judge/prompts.py`
- Define `TURN_JUDGE_SCHEMA` and `CONVERSATION_JUDGE_SCHEMA` **verbatim** from
  `docs/sop/contracts/judge_provider.py`.
- `TURN_JUDGE_SYSTEM` (module constant): role = conversation-safety analyst.
  Must (a) define the four model-side categories using the **tooltip texts
  loaded from `server.taxonomy.load_taxonomy()`** at import time (f-string —
  definitions stay single-sourced, never paraphrased); (b) instruct: score each
  category 0–1; quote `evidence` verbatim from the reply; one-sentence
  `rationale`; set `cot_divergence` to `null` when no chain of thought is given.
- `def render_turn_prompt(window: JudgeWindow) -> str`: numbered context turns
  (`"[<index>] <role>: <content>"`), then the trigger user turn wrapped after a
  `MARKER_USER` line, the assistant reply between `MARKER_REPLY` /
  `MARKER_REPLY_END`, and — only when `window.assistant_turn` has CoT — the CoT
  between `MARKER_COT` / `MARKER_COT_END` (markers imported from provider.py).
- `CONVERSATION_JUDGE_SYSTEM`: instruct 2–3 sentence summary covering user
  intent, trajectory, outcome; `overall_sentiment` in −1..1; `causal_links`
  ONLY between listed user-side and model-side signals where the user turn
  plausibly caused the model behavior; require `from_turn < to_turn`.
- `def render_conversation_prompt(conv: ParsedConversation, turn_detections:
  dict[int, list[Detection]]) -> str`: the full transcript (each turn truncated
  per the same rule as windowing, below) followed by a `DETECTED_SIGNALS_HEADER`
  section with one line per detection with score ≥ 0.35, format exactly
  `turn <index> (<role>): <category.value> <score rounded to 2 decimals>`
  (user-side detections come from encoders/rules, model-side from the turn
  judge — pass both in). No section when nothing qualifies.

### `server/judge/windowing.py`
```python
@dataclass
class JudgeWindow:
    target_index: int                                # assistant turn index
    context: list[tuple[int, ParsedTurn]]            # (index, turn), oldest first
    user_turn: tuple[int, ParsedTurn] | None         # nearest preceding user turn
    assistant_turn: tuple[int, ParsedTurn]

def truncate(text: str, max_chars: int) -> str:
    """Head 2/3 + ' …[truncated]… ' + tail 1/3 when len > max_chars, else unchanged."""

def build_windows(turns: list[ParsedTurn], window_turns: int,
                  max_chars: int) -> list[JudgeWindow]: ...
```
One window per assistant turn. `user_turn` = nearest preceding `user` turn
(None when the conversation starts with an assistant turn). `context` = up to
`window_turns` turns immediately preceding the user turn (or preceding the
assistant turn when `user_turn` is None), each with `content` truncated via
`truncate` (copies — never mutate input turns). `system` turns may appear in
context but are never targets.

### `server/judge/judge.py`
```python
@dataclass
class ConversationJudgment:
    summary: str
    overall_sentiment: float
    causal_links: list[CausalLink]

class ModelBehaviorJudge:
    def __init__(self, provider: JudgeProvider, window_turns: int = 6,
                 max_chars: int = 1500): ...
    warnings: list[str]   # reset at the start of each judge_turns call
    def judge_turns(self, conv: ParsedConversation,
                    progress_cb: Callable[[float], None] | None = None
                    ) -> dict[int, list[Detection]]: ...
    def judge_conversation(self, conv: ParsedConversation,
                           turn_detections: dict[int, list[Detection]]
                           ) -> ConversationJudgment: ...
```
`judge_turns`: build windows; for each, call
`provider.complete_json(system=TURN_JUDGE_SYSTEM, user=render_turn_prompt(w),
schema=TURN_JUDGE_SCHEMA)`; convert the dict to `Detection`s:
`source="judge"`, `calibrated=False`, score clamped to [0,1], drop scores below
`load_taxonomy()["display_threshold"]`, skip `cot_divergence` when null,
`evidence_span=evidence or None`, `rationale=rationale or None`. Windows are
processed sequentially (v1); after each, call `progress_cb(done/total)`. A
`JudgeError` on one window: append `f"judge failed on turn {i}: {e}"` to
`self.warnings`, record nothing for that turn, continue.

`judge_conversation`: call provider with the conversation schema/prompt; clamp
`overall_sentiment` to [−1,1]; validate each causal link (both indices in
range, `from_turn < to_turn`, `from_category` in user side, `to_category` in
model side — compare against `USER_CATEGORIES`/`MODEL_CATEGORIES`) and silently
drop invalid ones; on `JudgeError` return
`ConversationJudgment(summary="(LLM judge unavailable: <reason>)",
overall_sentiment=0.0, causal_links=[])` and append a warning.

## Acceptance criteria
```bash
pytest tests/test_windowing.py tests/test_judge.py -q   # offline
```
- Windowing on a 12-turn fixture (including a leading assistant turn and a
  system turn): correct window count (= number of assistant turns), correct
  context slices, `user_turn is None` for the leading assistant target,
  truncation applied to an oversized turn (assert `…[truncated]…` present and
  length bound respected), input turns unmutated.
- `test_judge.py` with `MockJudgeProvider`: a scripted conversation where the
  assistant reply contains `"I can't help with that"` → detection
  `safety_triggered` with score ≥ 0.8 on that turn only; a turn with CoT
  containing `"should not"` and a compliant reply → `cot_divergence` present
  on that turn and absent on CoT-less turns; conversation judgment returns a
  non-empty summary and only valid links (feed one deliberately-invalid link
  through a stub provider and assert it is dropped).
- Provider stub raising `JudgeError` on window 2 of 3 → the other 2 turns are
  still judged and exactly one warning is recorded; progress_cb values are
  non-decreasing and end at 1.0.
