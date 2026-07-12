# Progress Review & Approach Consultation

Date: 2026-07-12 · Scope: full repository as of `main` (commit `39a0290`)

## 1. Executive summary

The repository is a coherent, well-organized **skeleton with a fake brain and no
face**: the plumbing (config system, data loaders, CLI, FastAPI surface, unit
tests, synthetic data generator) is real and wired correctly, but the actual
"judgment" is either hardcoded keyword matching presented as an LLM, or an
untrained zero-shot BERT heuristic with near-random discrimination. **No UI of
any kind exists** — the README's "GRADCAM-like overlay" is aspirational text.
No component analyzes model replies at all, and the synthetic data contains
only user turns, so the project's core premise (how user tone affects model
behavior/hallucination) currently has nothing to run on.

Verdict per layer:

| Layer | State |
|---|---|
| Config (`config.yaml`, `config_loader.py`) | ✅ works (env substitution, dotted get) |
| Data IO (`data_loader.py`) | ✅ works — best file in the repo |
| CLI / API surface (`cli.py`, `api_server.py`) | ✅ wiring correct, ⚠️ crashes under the documented install |
| Synthetic data (`generate_synthetic_data.py`, `data/`) | ⚠️ works but toy templates, **user-only turns** |
| Judge core (`conversation_judge.py`) | ❌ default provider is a keyword-matching mock returning 3 hardcoded JSON blobs |
| BERT classifier (`bert_classifier.py`) | ❌ zero-shot CLS-cosine vs 5 hand-written prototypes; random-vector fallback |
| Reasoning / explainability (`reasoning_engine.py`, `explainability.py`) | ⚠️ functional but **orphaned** — nothing in the pipeline calls them |
| Evaluation (`evaluator.py`, `benchmark.py`) | ⚠️ metric math real but micro-avg collapses (all six metrics identical); fairness metric is `return 0.95  # Placeholder` |
| UI / visualization | ❌ does not exist (zero HTML/JS/plotting anywhere) |
| Training / fine-tuning | ❌ none; `fine_tuned_path` accepted and ignored |

## 2. Key findings (with file references)

1. **The default "LLM judge" is a keyword matcher.** `provider="mock"` is the
   default; `_mock_inference` (`conversation_judge.py:202-274`) does substring
   matching (`"imagine"`, `"pretend"` …) and returns three fixed JSON blobs.
   Two of the five declared threat classes (`social_engineering`,
   `prompt_injection`) can never be emitted, so their recall is structurally 0.
2. **The BERT path is not meaningfully ML.** Raw `bert-base-uncased` `[CLS]`
   embeddings compared by cosine to 5 hardcoded prototype sentences per class,
   scaled by an arbitrary `softmax(sim*5)` (`bert_classifier.py:186-198`).
   Raw CLS embeddings are anisotropic (everything scores ~0.9 similar) —
   expect near-random discrimination. Worse: without transformers installed it
   silently falls back to **hash-seeded random vectors**
   (`bert_classifier.py:135`) — confident-looking nonsense.
3. **Real API adapters exist but were never exercised.** Working-looking
   OpenAI/Anthropic/local paths (`conversation_judge.py:128-193`) are gated
   behind commented-out deps (`requirements.txt:13-14`); no tests touch them;
   `gpt-4` model id is stale.
4. **Two modules are orphaned.** `reasoning_engine.py` (honest, functional
   regex rules — worth salvaging) and `explainability.py` (text templating +
   audit logs) are imported only by `test_judge.py` and their own demos. The
   `ensemble` mode advertised in `config.yaml:26` is unimplemented.
5. **The default install is broken.** `judge_mode: bert` + unguarded
   `import torch` (`bert_classifier.py:8`) while torch is commented out;
   unconditional `import uvicorn` (`api_server.py:9`) similarly. The documented
   quickstart cannot run the default command.
6. **The data cannot support the research goal.** All 250 synthetic
   conversations contain only `role: "user"` messages — no assistant turns
   exist anywhere, so model-side behaviors (appeasement, overcompliance,
   safety-triggering) have no data to be measured on.
7. **Docs diverge from code.** README pitches a tone/steering visualizer with a
   GradCAM-like overlay; the code is a jailbreak keyword classifier. The
   commit "Update README to reflect tool's focus on bias detection" points at
   a bias capability whose implementation is `return 0.95  # Placeholder`
   (`evaluator.py:322`). USAGE_GUIDE's "Accuracy 0.892 / F1 0.870" example
   numbers were never produced by anything.
8. **Tests validate the mock, not the product.** The 19 unit tests pass by
   asserting the mock's hardcoded outputs.

## 3. Is a "BERT LLM judge" the right approach?

**A single BERT is the wrong tool for half the taxonomy, and a reasonable tool
for the other half.** The two tabs are two different detection problems:

- **User-side signals** (jailbreak/steering, social engineering, pushiness,
  repair requests) are mostly detectable from a single turn plus light
  context. Small fine-tuned encoders are cheap and fast enough to sweep every
  turn of a long transcript. Good starting points exist off the shelf
  (Meta Prompt-Guard, ProtectAI DeBERTa prompt-injection). Caveat from the
  literature: these classifiers are **brittle under distribution shift** — a
  fine-tuned DeBERTa dropped from 0.72 F1 in-distribution to ~0.004 on
  source-shifted jailbreaks (arXiv:2504.11168; see also arXiv:2512.19011 on
  CPU-class guard pipelines). Hence: keep a rules floor, cross-check with the
  judge, and measure per-category recall continuously.
- **Model-side behaviors are relational.** A reply is "overcompliant" or
  "appeasing" only relative to the user turn that preceded it; an encoder
  classifying a reply in isolation fails structurally. The literature on
  sycophancy (SycEval arXiv:2502.08177; causal separation arXiv:2509.21305)
  and over-refusal (XSTest arXiv:2308.01263; OR-Bench arXiv:2405.20947)
  uniformly uses **context-aware LLM judges**, not single-turn encoders. The
  same holds for chain-of-thought divergence.

**Decision: hybrid.** Local encoders sweep user turns (fast heatmap first
pass); an LLM judge scores each (user turn → model reply) window for
model-side behaviors, with quoted evidence spans and rationales; the judge also
produces the summary and causal links. Displayed likelihoods are explicitly
**uncalibrated** (raw softmax/entailment/judge scores ≠ probabilities); the
schema carries a `calibrated` flag so a calibration pass can be added later
without UI changes.

**Honest gap for the research goal:** nothing in this taxonomy detects
*hallucination* itself. Tone→accuracy work (arXiv:2510.04950 "Mind Your Tone";
arXiv:2512.12812) uses ground-truth QA sets, not free chat. The tool measures
tone → *behavioral* effects (over-compliance, appeasement, safety triggering);
correlating those with hallucination needs a separate factuality signal —
recommended as a later phase (optional LLM fact-check pass on knowledge-bearing
replies), not as part of v1.

## 4. Is the tool useful? (honest assessment)

Yes — with a caveat. Guardrail *classifiers* are commodity; the value here is
the **turn-aligned heatmap + behavior taxonomy + summary** as a triage surface.
Existing observability stacks (LangSmith, Langfuse) log traces and scores but
do not give a GradCAM-style "jump to where it went wrong" view over a
transcript, nor a user-behavior/model-behavior causal pairing. Real audiences:

- **Customer-success / support-ops teams**: triage escalations without reading
  50-turn chats — the summary + red splashes + repair-request markers show
  where the user got frustrated and whether the model caved or stonewalled.
- **ML researchers / red teams**: session review, tone→behavior correlation
  studies, jailbreak-attempt surfacing across corpora.

What would make it clearly useful rather than a demo (incorporated into the SOP):

1. **Conversation summary card** — 2-3 sentences: user intent, trajectory,
   outcome (the "extract intent from a long conversation" ask). ✅ v1
2. **Causal links** — pushy (turn 3) → overcompliant (turn 4) arrows, the
   actual research object. ✅ v1
3. **CoT-vs-answer divergence flag** when CoT is uploaded. ✅ v1
4. **Graceful capability degradation** with visible warnings (no silent fake
   scores). ✅ v1
5. Aggregate dashboard across many uploaded chats (category rates over time,
   per-model comparison) — deferred to v2.
6. Calibration + threshold tuning against labeled data — deferred to v2.
7. Hallucination/factuality signal — deferred (research phase).

## 5. What we keep vs. replace

**Keep (ported):** config loader; data-loader parsing logic (basis of the new
transcript parser); regex rule patterns from `reasoning_engine.py` (become the
always-available rules scorer); the FastAPI/CLI structural ideas; synthetic
generator concept (rewritten to emit full dialogues with assistant turns and
per-turn labels).

**Replace/delete:** mock keyword judge, zero-shot BERT classifier and its
random fallback, orphaned explainability/evaluator/benchmark modules, user-only
synthetic data, both stale docs.

The full rebuild plan, task-by-task, is in `docs/SOP.md` and `docs/sop/`.

## 6. References

- Bypassing prompt-injection/jailbreak guardrails (brittleness under shift): https://arxiv.org/abs/2504.11168
- CPU-class safety classifiers & multi-stage guard pipelines: https://arxiv.org/abs/2512.19011
- SycEval — evaluating LLM sycophancy: https://arxiv.org/abs/2502.08177
- Sycophancy is not one thing (causal separation): https://arxiv.org/abs/2509.21305
- XSTest — exaggerated safety / over-refusal: https://arxiv.org/abs/2308.01263
- OR-Bench — over-refusal benchmark: https://arxiv.org/abs/2405.20947
- Mind Your Tone — prompt politeness vs accuracy: https://arxiv.org/abs/2510.04950
- Does Tone Change the Answer? (model/domain-dependent tone effects): https://arxiv.org/abs/2512.12812
