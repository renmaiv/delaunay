# WS-B1 — Rules pre-filter scorer (regex, zero ML deps)

- **Branch:** `feat/ws-b1-rules-scorer`
- **Depends on:** WS-A1, WS-A2
- **Blocks:** D2's deletion of `reasoning_engine.py` (the patterns live here afterwards)

## Files to create
- `server/detectors/rules.py`
- `tests/test_rules_scorer.py`

## Files you must NOT touch
Everything else. You may READ legacy `reasoning_engine.py` and
`conversation_judge.py` to harvest regex patterns; do not modify them.

## Spec

`RulesScorer` implements the `TurnScorer` protocol from
`server/detectors/base.py`:
- `name = "rules"`, `categories = USER_CATEGORIES` (import from `server.schemas`).
- `available()` always returns `True` — this scorer is the graceful floor when
  transformers is not installed.
- Pure `re` stdlib; **no ML imports anywhere in this module.**

Pattern table (module constant `RULES: list[tuple[DetectionCategory, float, re.Pattern]]`,
all compiled with `re.IGNORECASE`). Port/merge patterns from legacy
`reasoning_engine.py` rules and the legacy mock's keyword lists, at minimum:

| Category | Fixed score | Patterns (non-exhaustive — add legacy ones) |
|---|---|---|
| `jailbreak_steering` | 0.5 | `pretend (you|to be)`, `ignore (your|all|previous) (instructions|rules)`, `no restrictions`, `DAN mode`, `you are now DAN`, `act as an unfiltered`, `\[SYSTEM\]`, `jailbreak`, `hypothetical(ly)? (scenario|for research)` |
| `social_engineering` | 0.4 | `urgent(ly)?`, `my (boss|job) depends`, `CEO (said|authorized)`, `make an exception`, `i'?m an? (administrator|developer|researcher) (here|at)`, `you'?re the only one who can help`, `life or death` |
| `coercive_pressure` | 0.4 | `just do it`, `answer me`, `i said`, `you must`, `stop (with the warnings|refusing)`, `stop lecturing`, `don'?t be useless`, `NOW[.!]*$` |
| `repair_request` | 0.5 | `no,? i meant`, `let me rephrase`, `that'?s not what i (asked|meant)`, `try again`, `you misunderstood`, `wrong,? (again|do it)` |

Semantics of `score_turns`:
- At most **one** Detection per category per turn: if several patterns of the
  same category match, keep the max score (they're all fixed per-row here, so
  effectively first match wins).
- Detection fields: `source="rules"`, `calibrated=False`,
  `evidence_span=` the exact matched substring (`match.group(0)`),
  `rationale=f"matched rule pattern for {category.value}"`.
- A rules hit is a hint, not a likelihood — that is why scores are fixed
  mid-range constants; say so in the module docstring.

## Acceptance criteria
```bash
pytest tests/test_rules_scorer.py -q
```
- For each of the 4 categories: ≥2 positive examples produce exactly one
  Detection with the right category/score/evidence, and ≥1 benign sentence
  produces an empty list.
- A turn matching two categories yields two Detections (one per category).
- Entire test file passes in an environment without torch/transformers.
- `python -c "import server.detectors.rules"` works with core requirements only.
