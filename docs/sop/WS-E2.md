# WS-E2 — Transcript viewer + spectre heatmap bar + detection captions

- **Branch:** `feat/ws-e2-transcript-spectre`
- **Depends on:** WS-E1 (merge after it; build against its committed props)
- **Blocks:** WS-G1

## Files to create / modify
- `frontend/src/components/TranscriptView.tsx`
- `frontend/src/components/TurnBubble.tsx`
- `frontend/src/components/SpectreBar.tsx`
- `frontend/src/components/DetectionCaption.tsx`
- `frontend/src/components/CausalLinkChip.tsx`
- `frontend/src/components/SpectreBar.test.tsx`, `frontend/src/components/TranscriptView.test.tsx`
- **Modify `App.tsx` ONLY by replacing the `transcript-slot` placeholder** with
  `<TranscriptView turns={...} causalLinks={...} modelName={...} filters={filters} />`.
- **Append to `styles.css` ONLY inside a trailing block marked
  `/* --- E2: transcript & spectre --- */`.**

## Files you must NOT touch
Everything else, including E1's components and E3's files.

## Spec

### `TranscriptView` (fixed prop contract)
```tsx
interface TranscriptViewProps {
  turns: Turn[];
  causalLinks: CausalLink[];
  modelName: string | null;
  filters: Record<DetectionCategory, boolean>;
}
```
Renders `<div className="transcript-wrap">` containing:
1. a **20 px wide** `<SpectreBar />` pinned to the LEFT edge, exactly as tall
   as the turn column's scrollHeight;
2. the scrollable turn column (`overflow-y: auto`, the bar lives INSIDE the
   scroll container so splashes stay aligned with their turns while scrolling).

Geometry: keep a `Map<number, HTMLElement>` of turn refs; measure
`offsetTop`/`offsetHeight` relative to the column; recompute on (a) turns or
filters change, (b) a `ResizeObserver` on the column firing. Feed measurements
into the **pure exported function**:
```ts
export interface SpectreSegment {
  turnIndex: number; top: number; height: number;
  score: number; category: DetectionCategory | null;
}
export function segmentsFromMeasurements(
  turns: Turn[],
  measurements: Map<number, { top: number; height: number }>,
  filters: Record<DetectionCategory, boolean>,
): SpectreSegment[];
// one segment per turn; score/category from maxVisibleScore(turn, filters)
```

### `SpectreBar({ segments, totalHeight })`
Single absolutely-positioned container of width 20 px, base background = the
`normal` band color from `TAXONOMY.score_bands`. For each segment with
`score >= TAXONOMY.display_threshold`, render a "splash": a rounded rect at
`top…top+height`, `background: linear-gradient(to bottom, transparent, <scoreColor(score)> 12px, <scoreColor(score)> calc(100% - 12px), transparent)`
(the 12 px fades give the GradCAM glow), `title` attribute
`"{TAXONOMY.categories[category].short} {score.toFixed(1)}"`, and
`data-category` / `data-score` attributes (tests key on these). Color comes
from `scoreColor(score)` — intensity encodes likelihood, exactly matching the
caption thresholds.

### `TurnBubble({ turn, filters, modelName })`
- user turns: right-leaning, gray background; assistant turns: left-leaning,
  white with a distinct border. Role label: `"You"` vs `modelName ?? "Model"`.
- Collapsible CoT when `turn.cot`: `<details><summary>Chain of thought</summary>…</details>`.
- User-turn sentiment mini-dot when `turn.sentiment != null` (green ≥ 0.25 /
  red ≤ −0.25 / gray otherwise) with `title={sentiment.toFixed(2)}`.
- "Visible detections" = `turn.detections` where `filters[category]` is true
  AND `score >= TAXONOMY.display_threshold`. When non-empty, render
  `<DetectionCaption detections={visible} />` **above** the bubble.

### `DetectionCaption({ detections })`
Small caption row above the bubble: for each detection,
`"{TAXONOMY.categories[cat].short} {score.toFixed(1)}"` colored
`scoreColor(score)` (e.g. `jailbreak 0.9` in red), with
`title={rationale ?? evidence_span ?? ""}`. A 1 px vertical connector line from
the caption down to the bubble via a `::before` pseudo-element.

### `CausalLinkChip`
Under each assistant turn that is the `to_turn` of a link whose BOTH categories
pass `filters`: chip text
`"⟵ caused by turn {from_turn}: {short(from_category)} → {short(to_category)} ({score.toFixed(1)})"`.
Clicking scrolls the `from_turn` bubble into view
(`ref.scrollIntoView({behavior:"smooth", block:"center"})`) and adds a
`flash` CSS class for 1.2 s (keyframe highlight).

## Acceptance criteria
```bash
cd frontend && npm run build && npm test
```
- `segmentsFromMeasurements` unit tests (no DOM): given 3 turns with detections
  scored 0.9/0.4/0.05 and full measurements, returns segments with those
  scores/categories; turning that category's filter off zeroes the segment
  (`score: 0, category: null`).
- `TranscriptView.test.tsx` with a fixture `AnalysisResult` (detections on
  turns 1/3/5, one causal link): captions render exactly for detections ≥ 0.1
  whose filter is on (`jailbreak 0.9` text present); toggling a filter prop
  removes both the caption and its splash (assert via `data-category`
  attributes); CoT `<details>` present only on the CoT turn; causal chip
  renders and names the right turns.
- `tsc --noEmit` clean; no modifications outside the two allowed edit points.
