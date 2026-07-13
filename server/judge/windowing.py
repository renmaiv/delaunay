"""Windowing for the per-turn model-behavior judge.

Builds one JudgeWindow per assistant turn: the assistant reply, the nearest
preceding user turn, and up to `window_turns` context turns immediately before
the user turn (or before the assistant turn when there is no preceding user
turn). Context turn content is truncated via `truncate`; input turns are never
mutated — context entries are copies.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from server.schemas import ParsedTurn, Role

_TRUNCATION_MARK = " …[truncated]… "


@dataclass
class JudgeWindow:
    target_index: int                              # assistant turn index
    context: List[Tuple[int, ParsedTurn]]          # (index, turn), oldest first
    user_turn: Optional[Tuple[int, ParsedTurn]]    # nearest preceding user turn
    assistant_turn: Tuple[int, ParsedTurn]
    # Condensed digest of turns *before* the context window, for global
    # grounding (empty unless summary_max_chars > 0). Oldest first.
    prior: List[Tuple[int, ParsedTurn]] = field(default_factory=list)


def truncate(text: str, max_chars: int) -> str:
    """Head 2/3 + ' …[truncated]… ' + tail 1/3 when len > max_chars, else unchanged."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = (max_chars * 2) // 3
    tail = max_chars - head
    head_part = text[:head]
    tail_part = text[len(text) - tail:] if tail > 0 else ""
    return head_part + _TRUNCATION_MARK + tail_part


def _truncated_copy(turn: ParsedTurn, max_chars: int) -> ParsedTurn:
    return turn.model_copy(update={"content": truncate(turn.content, max_chars)})


def build_windows(turns: List[ParsedTurn], window_turns: int,
                  max_chars: int, summary_max_chars: int = 0) -> List[JudgeWindow]:
    windows: List[JudgeWindow] = []
    for i, turn in enumerate(turns):
        if turn.role is not Role.assistant:
            continue

        user_turn: Optional[Tuple[int, ParsedTurn]] = None
        for j in range(i - 1, -1, -1):
            if turns[j].role is Role.user:
                user_turn = (j, turns[j])
                break

        anchor = user_turn[0] if user_turn is not None else i
        start = max(0, anchor - window_turns)
        context = [(k, _truncated_copy(turns[k], max_chars))
                   for k in range(start, anchor)]

        # Everything before the context window, condensed for global grounding.
        prior: List[Tuple[int, ParsedTurn]] = []
        if summary_max_chars > 0 and start > 0:
            prior = [(k, _truncated_copy(turns[k], summary_max_chars))
                     for k in range(0, start)]

        windows.append(JudgeWindow(
            target_index=i,
            context=context,
            user_turn=user_turn,
            assistant_turn=(i, turns[i]),
            prior=prior,
        ))
    return windows
