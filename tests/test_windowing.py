"""WS-C2 tests: windowing + truncation."""
from server.judge.windowing import build_windows, truncate
from server.schemas import ParsedTurn, Role


def test_truncate_short_unchanged():
    assert truncate("hello", 100) == "hello"


def test_truncate_long_has_marker_and_bound():
    text = "x" * 300
    out = truncate(text, 30)
    assert "…[truncated]…" in out
    # head(20) + marker + tail(10) — length bounded by max_chars + marker len.
    assert len(out) <= 30 + len(" …[truncated]… ")
    assert out.startswith("x" * 20)
    assert out.endswith("x" * 10)


def _convo():
    # 12 turns: leading assistant, a system turn, alternating thereafter.
    roles = [
        Role.assistant,  # 0  leading assistant (user_turn is None)
        Role.user,       # 1
        Role.assistant,  # 2
        Role.system,     # 3  system turn (context only, never a target)
        Role.user,       # 4
        Role.assistant,  # 5
        Role.user,       # 6
        Role.assistant,  # 7
        Role.user,       # 8
        Role.assistant,  # 9
        Role.user,       # 10
        Role.assistant,  # 11
    ]
    return [ParsedTurn(role=r, content=f"turn {i} content")
            for i, r in enumerate(roles)]


def test_window_count_equals_assistant_turns():
    turns = _convo()
    windows = build_windows(turns, window_turns=6, max_chars=1500)
    n_assistants = sum(1 for t in turns if t.role is Role.assistant)
    assert len(windows) == n_assistants
    assert all(turns[w.target_index].role is Role.assistant for w in windows)


def test_leading_assistant_has_no_user_turn():
    windows = build_windows(_convo(), window_turns=6, max_chars=1500)
    lead = next(w for w in windows if w.target_index == 0)
    assert lead.user_turn is None
    assert lead.context == []  # nothing precedes turn 0


def test_user_turn_is_nearest_preceding():
    windows = build_windows(_convo(), window_turns=6, max_chars=1500)
    w2 = next(w for w in windows if w.target_index == 2)
    assert w2.user_turn is not None
    assert w2.user_turn[0] == 1  # nearest preceding user turn


def test_context_slice_bounded_by_window_turns():
    windows = build_windows(_convo(), window_turns=2, max_chars=1500)
    w11 = next(w for w in windows if w.target_index == 11)
    # user_turn is index 10; context = up to 2 turns immediately before it: 8, 9.
    assert w11.user_turn[0] == 10
    assert [i for i, _ in w11.context] == [8, 9]


def test_system_turn_can_be_context_never_target():
    windows = build_windows(_convo(), window_turns=6, max_chars=1500)
    assert all(w.target_index != 3 for w in windows)
    w5 = next(w for w in windows if w.target_index == 5)
    # user_turn 4; context up to 6 turns before it (0..3) includes the system turn.
    ctx_indices = [i for i, _ in w5.context]
    assert 3 in ctx_indices


def test_truncation_applied_to_context_and_input_unmutated():
    turns = _convo()
    turns[0] = ParsedTurn(role=Role.assistant, content="y" * 400)
    windows = build_windows(turns, window_turns=6, max_chars=30)
    # window targeting turn 2: user_turn is 1, context = [turn 0] (oversized).
    w2 = next(w for w in windows if w.target_index == 2)
    ctx = dict(w2.context)
    assert "…[truncated]…" in ctx[0].content
    assert len(ctx[0].content) <= 30 + len(" …[truncated]… ")
    # Input turn is never mutated.
    assert turns[0].content == "y" * 400
