# AGENTS.md

Guidance for Codex when working in this repository.

## Project

A semantic-observability tool for chat transcripts: upload a conversation
(optionally with chain-of-thought), get a GradCAM-inspired "spectre" heatmap
over the transcript highlighting jailbreak attempts, social engineering,
coercive pressure, and repair requests on the user side, and safety-triggered,
appeasement, overcompliant, and CoT-divergence behavior on the model side,
plus a conversation summary and causal links between triggers and effects.

The repository is being rebuilt from an early scaffold-only state. Before
making changes, read:

- `docs/PROGRESS_REVIEW.md` — honest assessment of what currently works vs.
  what is placeholder/fake, and why.
- `docs/SOP.md` — the implementation plan: architecture, repo layout,
  taxonomy, dependency graph between tasks, testing policy.
- `docs/sop/WS-*.md` — granular task cards (one per workstream task) with
  exact files, interface contracts, and acceptance criteria.
- `docs/sop/contracts/` — normative shared contracts (Pydantic schemas, the
  TypeScript mirror, `taxonomy.json`, scorer/judge protocols, API table).
  Any change to a contract must be mirrored across every consumer listed in
  that file's header.

## Commit attribution

Commits in this repository must be attributed to the user **`renmaiv`**
(GitHub: [renmaiv](https://github.com/renmaiv)), not to Codex. Before
committing, set local git identity for this repo if it is not already
correct:

```bash
git config user.name "renmaiv"
git config user.email "<renmaiv's GitHub noreply or account email>"
```

Do not add `Co-Authored-By: Codex` trailers to commits in this repository.

## Working conventions

- **Always verify with Playwright before pushing commits.** For any change that
  touches the frontend or the API surface, build the frontend, run the server,
  and drive the real app in a browser (upload a transcript, check the rendered
  transcript/detections) before pushing. Unit tests alone are not sufficient —
  a passing suite has already shipped a UI where every turn rendered as the
  user's. Chromium is pre-installed for Playwright in the remote environment.

- No fabricated model output, ever: a detector without its dependencies must
  raise a clear, actionable error (see `ScorerUnavailableError` /
  `JudgeError` in the contracts) — never fall back to random or
  keyword-guessed values presented as model output.
- Every `Detection` carries `calibrated: false` until an explicit calibration
  pass is added; do not imply raw scores are probabilities in UI copy or docs.
- Tests must run offline: mock `transformers` pipelines and stub the
  Anthropic client (inject via the provider's `client=` kwarg). Real models
  run only via `eval/run_eval.py`, invoked manually.
- Tooltip/label copy for detection categories lives only in
  `shared/taxonomy.json` — never hardcode it in frontend or backend code.
- When implementing a `docs/sop/WS-*.md` task card, stay within that card's
  listed files. Shared files (e.g. `App.tsx`, `styles.css`) are only touched
  in the marked extension points called out in later cards.
