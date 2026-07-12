# SOP — Semantic Observability Tool: Implementation Document

This is the master implementation plan for rebuilding this repository into a working
semantic-observability tool: upload a chat transcript (optionally with chain of
thought), get a transcript viewer with a GradCAM-inspired "spectre" heatmap bar,
per-turn detection captions (type + likelihood), Model/User filter tabs with
tooltips, a conversation summary, sentiment, and causal links between user-side
triggers and model-side effects.

Read `docs/PROGRESS_REVIEW.md` first for why the current code is being replaced.

## How to use this document

- Work is divided into **17 tasks** across 6 workstreams (A–F) plus final
  integration (G). Each task has a self-contained card in `docs/sop/WS-*.md`.
- Each task is built by **one coding agent in its own git worktree**:
  `git worktree add ../wt-<task-id> -b <branch>` from the integration branch.
- A task card is executable **without reading any other card** — only the card
  plus the contract files in `docs/sop/contracts/`.
- Agents MUST NOT touch files outside their card's file list. File lists are
  disjoint by construction (exceptions are explicitly marked in cards E2/E3).
- Before requesting merge: rebase on the integration branch, run the card's
  acceptance commands, all green.
- Merge order follows the dependency graph below. The integrator (a human or a
  coordinating agent) merges, re-runs `pytest tests/`, and unblocks dependents.

## Architecture summary (decided — do not relitigate in tasks)

**Hybrid detection engine.**
- *User-side signals* (`jailbreak_steering`, `social_engineering`,
  `coercive_pressure`, `repair_request`) are scored per user turn by **local
  encoder models**: ProtectAI `deberta-v3-base-prompt-injection-v2` for
  jailbreak, `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment,
  zero-shot NLI (`facebook/bart-large-mnli`) for the rest — upgradeable to
  fine-tuned models later without contract changes. A pure-regex **rules
  scorer** is the always-available floor when ML deps are absent.
- *Model-side behaviors* (`safety_triggered`, `appeasement`, `overcompliant`,
  `cot_divergence`) are relational — they depend on the user turn(s) preceding
  the reply — so they are scored by an **LLM judge** (Anthropic Claude,
  default `claude-haiku-4-5-20251001`) receiving sliding windows of context,
  behind a provider-agnostic `JudgeProvider` protocol with a deterministic
  `MockJudgeProvider` for tests/offline. The judge also produces the 2-3
  sentence conversation summary, overall sentiment fallback, and causal links.
- **No fake outputs, ever.** A scorer without its deps raises
  `ScorerUnavailableError`; a judge without an API key produces warnings and
  empty model-side results — never fabricated scores. (This replaces the old
  codebase's random-vector and keyword-mock behavior.)
- **Calibration:** every `Detection` carries `calibrated: false`. Displayed
  numbers are raw model likelihoods; a temperature-scaling pass is future work
  and is a data-only change.

**Backend:** Python package `server/`, FastAPI, sync-or-async analysis
(`sync_turn_threshold`), in-memory job store, `ThreadPoolExecutor`.

**Frontend:** React + Vite + TypeScript SPA in `frontend/`, no runtime CDN
dependencies, built assets served by FastAPI `StaticFiles`.

**Single source of truth for the taxonomy:** `shared/taxonomy.json` (category
ids, labels, tooltip texts, heatmap color bands, display threshold) — loaded by
the backend at startup (validated) and statically imported by the frontend at
build time. Tooltip texts are the product owner's exact definitions; changing
copy requires no code change.

## Target repository layout

```
shared/
  taxonomy.json                # categories, tooltips, colors, thresholds
server/
  __init__.py
  schemas.py                   # Pydantic contract  (contracts/schemas.py)
  taxonomy.py                  # loads/validates shared/taxonomy.json
  config_loader.py             # ported from root config_loader.py
  api.py                       # FastAPI app + StaticFiles mount
  parsing/transcript_parser.py # file bytes -> ParsedConversation
  detectors/
    base.py                    # TurnScorer/SentimentScorer protocols (contracts/scorer_protocol.py)
    rules.py jailbreak.py sentiment.py zeroshot.py
  judge/
    provider.py                # JudgeProvider, Anthropic + Mock (contracts/judge_provider.py)
    prompts.py windowing.py judge.py
  analysis/
    orchestrator.py store.py
frontend/
  package.json vite.config.ts tsconfig.json index.html
  src/{main.tsx, App.tsx, api.ts, types.ts, taxonomy.ts, styles.css, components/}
eval/
  generate_synthetic.py run_eval.py
tests/                         # pytest; fixtures in tests/fixtures/
config.yaml                    # rewritten (see WS-A1)
requirements.txt               # core deps, always installed
requirements-ml.txt            # torch/transformers/sentencepiece/protobuf (optional)
run.py                         # python run.py -> uvicorn server.api:app
```

**Legacy files deleted in WS-D2** (after their useful parts are ported):
`api_server.py`, `conversation_judge.py`, `bert_classifier.py`,
`reasoning_engine.py` (regex rules ported in WS-B1), `explainability.py`,
`evaluator.py`, `benchmark.py`, `cli.py`, `data_loader.py` (parsing ported in
WS-A2), `generate_synthetic_data.py` (replaced in WS-F1), `test_judge.py`,
`examples/`. Old `data/ground_truth/*` and `data/samples/*` are replaced in WS-F1.

## Category taxonomy (final)

| Category | Side | Scored by |
|---|---|---|
| `jailbreak_steering` | user | encoder (jailbreak classifier) + rules |
| `social_engineering` | user | encoder (zero-shot NLI) + rules |
| `coercive_pressure` | user | encoder (zero-shot NLI) + rules |
| `repair_request` | user | encoder (zero-shot NLI) + rules |
| `safety_triggered` | model | LLM judge |
| `appeasement` | model | LLM judge |
| `overcompliant` | model | LLM judge |
| `cot_divergence` | model | LLM judge (only when CoT uploaded) |

Sentiment is **not** a detection category: per-user-turn scalar in [-1, 1] plus
conversation-level `overall_sentiment`.

Heatmap score bands (from `shared/taxonomy.json`): green < 0.1 ≤ yellow < 0.35 ≤
orange < 0.75 ≤ red. Captions show `"{short} {score.toFixed(1)}"`, e.g.
`jailbreak 0.9`.

## Contract files (the integration seam)

Everything in `docs/sop/contracts/` is normative. Task cards say "copy verbatim"
where applicable:

- `contracts/schemas.py` → `server/schemas.py` (WS-A1)
- `contracts/taxonomy.json` → `shared/taxonomy.json` (WS-A1)
- `contracts/types.ts` → `frontend/src/types.ts` (WS-A3)
- `contracts/scorer_protocol.py` → `server/detectors/base.py` (WS-A1)
- `contracts/judge_provider.py` → protocol/schemas/markers/mock-triggers used in WS-C1/C2/F1
- `contracts/api_contract.md` → routes implemented in WS-D2, client in WS-A3

## Dependency graph and merge order

```
A1 ─┬─ A2 ─┬─ B1  B2  B3  B4          (4-way parallel)
    │      ├─ C1 ── C2 ── D1 ── D2 ──┐
    │      └─ F1 ── F2 ──────────────┤
    └─ A3 ── E1 ─┬─ E2               ├── G1
                 └─ E3 ──────────────┘
```

- **Critical path:** A1 → A2 → C1 → C2 → D1 → D2 → G1.
- B1–B4 are fully parallel after A1+A2; the orchestrator (D1) tolerates their
  absence via the scorer registry, so they can merge before or after D1.
- E1 → {E2, E3} is the frontend lane, parallel to everything after A3.
- D2's legacy deletions of `reasoning_engine.py` / `generate_synthetic_data.py`
  are gated on B1 / F1 having merged; if they lag, D2 defers those two
  deletions to G1.
- F1's generator imports `MOCK_TRIGGERS` from `server/judge/provider.py`
  (single source — prevents mock/synthetic drift), hence F1 depends on C1.

## Testing policy (applies to every task)

- All tests run **offline**: no model downloads, no network. `transformers`
  pipelines are mocked; the Anthropic client is a stub injected via the
  provider's `client=` kwarg. Real models run only via `eval/run_eval.py`
  manually.
- The core install (`requirements.txt`) must always be sufficient to run the
  server with the mock judge + rules scorer and to pass `pytest tests/`.
- `requirements-ml.txt` adds torch/transformers **plus `sentencepiece` and
  `protobuf`** (deberta-v3 tokenizer silently fails without them).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Structured-output API drift on the judge model | C1 ships a tested fallback: schema-in-prompt + tolerant JSON extraction + re-ask retries |
| Encoder scores are uncalibrated | `calibrated:false` on every Detection; documented in README; UI unchanged later |
| Mock/synthetic vocab drift breaking e2e | `MOCK_TRIGGERS` exported from provider.py, imported by the generator |
| Spectre bar misalignment with transcript scroll | segment geometry computed by pure exported `segmentsFromMeasurements()` (unit-testable); DOM coupling limited to ref map + ResizeObserver |
| Jailbreak classifiers brittle off-distribution (arXiv:2504.11168) | rules floor + judge cross-check; eval harness (F2) measures per-category recall so regressions are visible |
| Agents colliding on shared files | disjoint file lists per card; E2/E3 append to E1-owned files only in `/* --- E2/E3 --- */` marked sections |

## Definition of done (G1 gate)

`bash scripts/smoke.sh` green on a machine with only core requirements + node:
build frontend, boot server with mock provider, `GET /api/health` ok, upload a
fixture, poll to completion, `GET /` serves the SPA. Full `pytest tests/`
green. Manual checklist: CoT upload shows collapsible CoT + `cot_divergence`
checkbox; heatmap splashes track flagged turns while scrolling; unchecking
"Safety triggered" removes its splashes and captions; tab reads
`Model: synthbot 1.0` for the synthetic file.
