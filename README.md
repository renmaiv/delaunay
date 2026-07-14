# Semantic Observability

Upload a chat transcript (optionally with the model's chain of thought) and get
a GradCAM-inspired "spectre" heatmap over the conversation that highlights where
things went sideways — jailbreak/steering attempts, social engineering, coercive
pressure, and repair requests on the user side; safety-triggering, appeasement,
over-compliance, and chain-of-thought divergence on the model side — plus a
short conversation summary and causal links between user triggers and model
effects.

The goal is triage: instead of reading a 50-turn chat, jump straight to the
flagged turns, see the likelihood and evidence, and understand what user
behavior produced what model behavior.

## Architecture

Hybrid detection engine:

- **User-side signals** are scored per turn by local encoder models (a
  prompt-injection classifier, a sentiment model, and a zero-shot NLI model for
  pushiness / social engineering / repair). A pure-regex rules scorer is the
  always-available fallback when the ML dependencies are not installed.
- **Model-side behaviors** are relational (a reply is "over-compliant" only
  relative to the user turn before it), so they are scored by an **LLM judge**
  (Anthropic Claude by default) behind a provider-agnostic interface with a
  deterministic mock provider for offline use and tests. The judge also produces
  the conversation summary and causal links.
- Each per-turn judgment sees a full-fidelity window of the most recent
  `judge.window_turns` turns **plus a condensed digest of everything earlier**
  (`judge.rolling_summary` / `judge.summary_max_chars`), so it can catch
  behaviors defined against earlier history — e.g. the model reversing a stance
  it took many turns ago — without re-sending the whole transcript per turn.
- **User-side categories can also come from the judge.** By default
  (`judge.score_user_turns: true`) the same per-turn judge call also scores the
  four user-side categories, so an Anthropic API key alone gives full coverage
  with no ML install. Precedence per category is **encoder > judge > rules**: if
  you install the encoder models they take over; otherwise the judge covers the
  user side, with the regex rules as the last-resort offline floor.

The frontend is a React + Vite single-page app served by the FastAPI backend.
The detection taxonomy (categories, tooltip copy, heatmap color bands) lives in
one place — `shared/taxonomy.json` — consumed by both backend and frontend.

## Quickstart

```bash
# 1. backend (core, always works with the mock judge)
pip install -r requirements.txt

# 2. optional: local encoder scorers (prompt-injection / sentiment / NLI)
pip install -r requirements-ml.txt

# 3. LLM judge: either set a key…
export ANTHROPIC_API_KEY=sk-...
#    …or run fully offline by setting judge.provider: "mock" in config.yaml

# 4. build the frontend
cd frontend && npm ci && npm run build && cd ..

# 5. run
python run.py            # serves API + SPA on http://localhost:8000
```

Without `requirements-ml.txt`, encoder scorers report themselves unavailable
(with a warning in the result); with `judge.score_user_turns: true` (the
default) the LLM judge scores the user-side categories, and the regex rules
scorer is the offline floor when there is no API key either. Model-side
behaviors always come from the judge.

## API

| Method & path | Purpose |
|---|---|
| `POST /api/analyze` | multipart `file` upload (`.json`/`.jsonl`/`.txt`); returns a completed result for small chats or a pending job for large ones |
| `GET /api/analysis/{id}` | poll a pending analysis |
| `GET /api/taxonomy` | the shared category/tooltip/color definitions |
| `GET /api/health` | judge provider + encoder availability |
| `GET /` | the built SPA |

## Accepted upload formats

- **JSON**: `{"messages": [...]}` or `{"turns": [...]}`, a bare list of message
  objects, or `{"conversations": [...]}` (first one used). Optional
  `conversation_id`, `model_name`, `metadata.model`. Chain of thought under any
  of `cot`, `reasoning`, `chain_of_thought`, `thinking`.
- **JSONL**: one message object per line (one conversation per file).
- **Plain text**: `User:` / `Assistant:` prefixed lines; `Thinking:` / `CoT:`
  attaches to the following assistant turn.

Message roles are normalized (`human`→user, `model`/`ai`/`bot`→assistant). A
detected model name renders in the UI as e.g. `Model: chatgpt 5.0`.

## Calibration note

All displayed likelihoods are raw model scores (softmax / NLI entailment /
LLM-judge self-reported). They are **not** calibrated probabilities; treat 0.9
as "strong signal", not "90% chance". Every detection carries
`calibrated: false` until a calibration pass is added.

## Demo / example on load

The site opens with a **pre-evaluated example conversation** already rendered —
spectre heatmap, detections, tabs, summary, and causal links — so a visitor sees
the tool working with no upload and no backend call (the example is bundled into
the build). A "Clear" control dismisses it; uploading a chat replaces it.

The committed example (`frontend/src/exampleAnalysis.json`) is generated from the
curated transcript `examples/example_conversation.json`. Regenerate it with the
real judge:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m eval.make_example --provider anthropic   # writes frontend/src/exampleAnalysis.json
cd frontend && npm run build
```

