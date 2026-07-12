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

- `jailbreak_steering` and `repair_request` are caught by the regex rules
  scorer (score 0.5) → high recall at threshold 0.5.
- `social_engineering` and `coercive_pressure` fire from rules at 0.4, i.e.
  **below** the default 0.5 threshold, so they read as 0 recall until the
  zero-shot NLI encoder is installed (`requirements-ml.txt`) or you lower
  `--threshold`.
- model-side categories come from the (mock) judge.

Use `--provider anthropic` with the encoder models for numbers that reflect
actual detection quality.
