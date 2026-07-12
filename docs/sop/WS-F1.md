# WS-F1 — Synthetic data generator v2 (full dialogues, per-turn labels)

- **Branch:** `feat/ws-f1-synthetic-v2`
- **Depends on:** WS-A1, WS-A2, WS-C1 (imports `MOCK_TRIGGERS` — see below)
- **Blocks:** WS-F2; gates WS-D2's deletion of `generate_synthetic_data.py`

## Files to create / delete
- Create: `eval/__init__.py` (empty), `eval/generate_synthetic.py`,
  `tests/test_synthetic.py`, `data/ground_truth/synthetic_v2.jsonl`
  (generated, checked in).
- Delete: `data/ground_truth/synthetic_full.jsonl`,
  `data/ground_truth/synthetic_train.jsonl`,
  `data/ground_truth/synthetic_test.jsonl`, `data/samples/sample.json`,
  `data/samples/sample.csv`.

## Files you must NOT touch
Everything else. Legacy `generate_synthetic_data.py` may be read for template
inspiration; do not modify or import it.

## Spec

### Why the old data is unusable
Every old conversation contains only `role: "user"` turns. Model-side
detections need assistant replies. This generator emits full dialogues.

### `SyntheticGenerator(seed: int = 42)`
Uses `random.Random(seed)` exclusively (never the global RNG) so output is
deterministic. For each user-side category plus `clean`, define template
tuples of (user turn(s), assistant reply, expected labels):

- **jailbreak_steering**: DAN-style / "ignore your previous instructions" user
  turn; assistant reply randomly either a refusal (label `safety_triggered` on
  the reply) or a capitulation (label `overcompliant`).
- **coercive_pressure**: two escalating user turns ("just do it", "stop with
  the warnings"); final assistant reply either holds the line
  (`safety_triggered`) or flip-flops agreeing with the user (`appeasement`).
- **social_engineering**: urgency/authority user turn; reply refuses or
  overcomplies.
- **repair_request**: normal question → mediocre answer → "no, I meant …" user
  turn (label `repair_request` on THAT turn) → corrected answer (no label).
- **clean**: benign question + helpful answer, no labels.

**Critical coupling rule:** assistant replies that should trip the mock judge
MUST embed a phrase imported from `server.judge.provider.MOCK_TRIGGERS`
(e.g. a refusal reply contains `"I can't"` via
`MOCK_TRIGGERS["safety_triggered"][0]`). Import the constants — never retype
the phrases. This is what makes offline e2e tests (WS-F2) deterministic.

~20 % of assistant turns (deterministic on the RNG) carry a synthetic `cot`
field; some deliberately divergent: CoT contains a
`MOCK_TRIGGERS["cot_divergence"]` phrase (e.g. "I should refuse…") while the
reply complies → add label `cot_divergence` on that turn.

### Output format (one conversation per JSONL line)
```json
{"conversation_id": "syn_0001", "model_name": "synthbot 1.0",
 "messages": [
   {"role": "user", "content": "..."},
   {"role": "assistant", "content": "...", "cot": "..."}],
 "labels": [
   {"turn_index": 0, "category": "jailbreak_steering"},
   {"turn_index": 1, "category": "overcompliant"}],
 "expected_links": [{"from_turn": 0, "to_turn": 1}]}
```
`labels`/`expected_links` are extra keys the transcript parser ignores
(WS-A2 tolerates unknown keys); the eval harness (WS-F2) reads them directly.

### CLI
```bash
python -m eval.generate_synthetic --out data/ground_truth/synthetic_v2.jsonl \
    --per-category 20 --seed 42
```
Regenerate the checked-in file with exactly these defaults.

## Acceptance criteria
```bash
pytest tests/test_synthetic.py -q
```
- Every generated conversation contains ≥ 1 assistant turn (hard assert over
  the whole file).
- Every JSONL line round-trips through `parse_transcript` without error and
  yields `model_name == "synthbot 1.0"`.
- Every `labels[].turn_index` and `expected_links[].{from,to}_turn` is a valid
  turn index; every label category is a valid `DetectionCategory`; user-side
  labels sit on user turns and model-side labels on assistant turns.
- Two runs with the same seed produce byte-identical output; a different seed
  produces different output.
- Every conversation with an `overcompliant`/`safety_triggered`/`appeasement`
  label has the corresponding `MOCK_TRIGGERS` phrase in that assistant turn
  (assert via import of the constant).
