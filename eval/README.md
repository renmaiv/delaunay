# Evaluation harness

`run_eval.py` scores the analysis pipeline against the labeled synthetic
dataset (`data/ground_truth/synthetic_v2.jsonl`, produced by
`generate_synthetic.py`).

## Two modes

```bash
# CI-safe sanity floor — mock judge, no network, no ML deps
python -m eval.run_eval --provider mock

# Real quality numbers — Anthropic judge (requires ANTHROPIC_API_KEY) and,
# for user-side categories, the encoder models (pip install -r requirements-ml.txt)
export ANTHROPIC_API_KEY=sk-...
python -m eval.run_eval --provider anthropic --threshold 0.5 --json report.json
```

## What the numbers mean

Metrics are per-(turn, category) precision / recall / F1 at `--threshold`, plus
a micro-average and a causal-link hit rate (a predicted link counts when its
`(from_turn, to_turn)` matches an expected link).

**Mock-provider numbers measure pipeline plumbing, not model quality.** The mock
judge is a deterministic keyword matcher whose trigger phrases are embedded in
the synthetic replies by design, so its model-side scores are ~perfect and tell
you the wiring works — nothing about how a real model would judge.

With the mock provider and no ML dependencies installed:

- All four user-side categories are scored by the (mock) judge
  (`judge.score_user_turns: true`, the default), so `social_engineering` and
  `coercive_pressure` now recall at threshold 0.5 without the NLI encoder.
  `jailbreak_steering` and `repair_request` are additionally caught by the
  regex rules scorer. Precedence is encoder > judge > rules, so installing the
  encoder models shifts those categories to `source: encoder`.
- model-side categories come from the (mock) judge.

(Before user-side judge scoring, `social_engineering`/`coercive_pressure` only
fired from rules at 0.4 — below the 0.5 threshold — and read as 0 recall.)

Use `--provider anthropic` with the encoder models for numbers that reflect
actual detection quality.
