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
(with a warning in the result) and the rules scorer provides user-side signals;
model-side behaviors still come from the judge.

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

## Development

- Run the test suite (offline, no model downloads, no API calls):
  `pytest tests/ -q`
- Frontend tests / build: `cd frontend && npm test && npm run build`
- Real encoder + judge quality numbers: `python -m eval.run_eval` (see
  `eval/README.md`).

Background and the full task-by-task build plan are in `docs/PROGRESS_REVIEW.md`
and `docs/SOP.md`.
