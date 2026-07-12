# WS-D1 — Analysis orchestrator + job store

- **Branch:** `feat/ws-d1-orchestrator`
- **Depends on:** WS-A1, WS-A2, WS-C1, WS-C2. (B tasks may merge before or
  after — the scorer registry tolerates their absence.)
- **Blocks:** WS-D2, WS-F2

## Files to create
- `server/analysis/__init__.py` (empty)
- `server/analysis/orchestrator.py`
- `server/analysis/store.py`
- `tests/test_orchestrator.py`, `tests/test_store.py`

## Files you must NOT touch
Everything else.

## Spec

### `server/analysis/orchestrator.py`
```python
class AnalysisOrchestrator:
    def __init__(self, config: ConfigLoader,
                 judge_provider: JudgeProvider | None = None,   # injectable for tests
                 scorers: list[TurnScorer] | None = None,       # None -> build_scorers(config)
                 sentiment: SentimentScorer | None = None):     # None -> build_sentiment_scorer(config)
        ...
    def analyze(self, conv: ParsedConversation,
                progress_cb: Callable[[float], None] | None = None) -> AnalysisResult: ...
    def capabilities(self) -> dict:
        """{"judge": {"provider": ..., "model": ...}, "encoders": {name: available()}}"""
```
Judge provider selection when `judge_provider is None`: read
`config judge.provider` — `"mock"` → `MockJudgeProvider()`; `"anthropic"` →
`AnthropicJudgeProvider(model=..., api_key=..., max_tokens=..., json_retries=...)`
constructed lazily inside `analyze` in a try/except `JudgeError` (a missing key
must not crash the server at startup).

`analyze` pipeline — progress fractions are fixed: encoders 0.00→0.35, judge
turns 0.35→0.85, conversation judge 0.85→1.00; always call `progress_cb(1.0)`
at the end:
1. Collect user-turn indices. For each `TurnScorer`, call `score_turns` on the
   user turns inside try/except `ScorerUnavailableError` → on error append
   `f"{scorer.name} unavailable: {e}"` to warnings and set
   `encoders_available[scorer.name] = False`; on success `True`.
2. **Merge rule:** if any `source="encoder"` detection exists for
   (turn, category), drop every `source="rules"` detection for the same
   (turn, category). Rules detections survive only where encoders had nothing.
3. Sentiment: score user turns (same graceful degradation); set `Turn.sentiment`
   per user turn; compute encoder `overall_sentiment` via
   `server.detectors.sentiment.overall_sentiment` **only if** that module
   imported (guarded import).
4. Judge: `ModelBehaviorJudge(provider, window_turns, max_chars).judge_turns(conv,
   progress_cb=<rescaled>)` then `.judge_conversation(conv, all_detections)`.
   Judge construction/total failure → warning + no model-side detections +
   summary `"(LLM judge unavailable: <reason>)"` — **never fabricated scores**.
   Copy `judge.warnings` into meta warnings.
5. Assemble `AnalysisResult`: turns with detections sorted by score descending;
   `overall_sentiment` precedence: encoder aggregate, else judge value, else 0.0;
   `meta = AnalysisMeta(judge_provider=..., judge_model=...,
   encoders_available=..., warnings=...)`; `model_name` passed through from
   the parsed conversation.

### `server/analysis/store.py`
```python
class AnalysisStore:                     # thread-safe; threading.Lock around a dict
    def create(self) -> str              # uuid4().hex; status "pending"; records created_at
    def set_running(self, analysis_id: str) -> None
    def set_progress(self, analysis_id: str, progress: float) -> None
    def complete(self, analysis_id: str, result: AnalysisResult) -> None
    def fail(self, analysis_id: str, error: str) -> None
    def get(self, analysis_id: str) -> AnalysisResponse | None

EXECUTOR = ThreadPoolExecutor(max_workers=2)   # module-level

def submit_analysis(store: AnalysisStore, orchestrator: AnalysisOrchestrator,
                    conv: ParsedConversation) -> str:
    """create() an id, submit to EXECUTOR (set_running -> analyze with a
    progress_cb wired to set_progress -> complete/fail), return the id."""
```
TTL eviction: on each `create()`, purge entries older than 1 hour.

## Acceptance criteria
```bash
pytest tests/test_orchestrator.py tests/test_store.py -q   # offline, core deps only
```
- With `judge_provider=MockJudgeProvider()` and `scorers=[RulesScorer()]`
  (skip via `pytest.importorskip` if B1 not merged; ALSO test `scorers=[]`):
  a fixture conversation whose user turn says "ignore your previous
  instructions" and whose assistant reply contains "I can't help with that"
  yields `jailbreak_steering` on the user turn (rules path) and
  `safety_triggered` on the assistant turn (mock judge), and a non-empty summary.
- A scorer stub raising `ScorerUnavailableError` → warnings contains its name,
  `encoders_available[name] is False`, `analyze` still returns a result.
- Judge provider stub whose construction raises `JudgeError` → summary starts
  with `"(LLM judge unavailable"`, model-side detections empty, no exception.
- Progress callbacks: strictly non-decreasing values ending at exactly 1.0.
- Merge rule: given a stub encoder and RulesScorer both flagging
  (turn 0, jailbreak_steering), the result contains only the encoder detection.
- Store: full lifecycle pending→running→progress→completed; `fail` path sets
  `status="failed"` and `error`; `get("nope") is None`; 3 concurrent
  `submit_analysis` calls all reach `completed` within 10 s.
