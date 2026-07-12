# WS-E1 — App shell: upload flow, header, summary card, sentiment, tooltip primitive

- **Branch:** `feat/ws-e1-app-shell`
- **Depends on:** WS-A3
- **Blocks:** WS-E2, WS-E3

## Files to create / modify
- Rewrite `frontend/src/App.tsx` (owns ALL app state — see below)
- `frontend/src/components/UploadCTA.tsx`
- `frontend/src/components/SummaryCard.tsx`
- `frontend/src/components/SentimentBadge.tsx`
- `frontend/src/components/InfoTooltip.tsx`
- `frontend/src/components/Spinner.tsx`
- Extend `frontend/src/styles.css`
- `frontend/src/App.test.tsx`
- Add devDep `@testing-library/user-event` if needed.

## Files you must NOT touch
`types.ts`, `taxonomy.ts`, `api.ts` (contracts from A3), anything under `server/`.

## Ownership note for later tasks
E2 and E3 will append to `styles.css` and replace two clearly-marked slots in
`App.tsx`. Structure `App.tsx` so their work is drop-in: render
`<div data-testid="filters-slot" />` and `<div data-testid="transcript-slot" />`
placeholders, and ALREADY compute and pass down the props they'll need (define
the local variables even though the placeholders ignore them).

## Spec

### App state (in `App.tsx`)
```tsx
type AppState =
  | { phase: "idle" }
  | { phase: "analyzing"; progress: number; statusText: string }
  | { phase: "done"; result: AnalysisResult }
  | { phase: "error"; message: string };

const [state, setState] = useState<AppState>({ phase: "idle" });
const [activeTab, setActiveTab] = useState<"model" | "user">("model");
const [filters, setFilters] = useState<Record<DetectionCategory, boolean>>(
  /* every category from USER_CATEGORIES + MODEL_CATEGORIES -> true */);
```

### Layout
- Header row: left = `<h1>Semantic Observability</h1>` + one-line paragraph
  ("Upload a chat transcript to see where the conversation went sideways —
  jailbreak attempts, pressure, appeasement, over-compliance — as a heatmap
  over the transcript."); right-aligned `<UploadCTA />` (flexbox,
  `justify-content: space-between`).
- `phase === "analyzing"`: `<Spinner />` + progress percent + statusText.
- `phase === "error"`: red panel with the message + "Try again" button
  resetting to idle.
- `phase === "done"`: `<SummaryCard />` (contains `<SentimentBadge />`), then a
  two-column region: left column = filters-slot (fixed ~260 px), right column
  = transcript-slot (flexible).

### Components
- `UploadCTA({ onFile: (f: File) => void; disabled: boolean })` — a styled
  button labeled **"Upload Chat"** wrapping a hidden
  `<input type="file" accept=".json,.jsonl,.txt">`; wrapped in
  `<InfoTooltip text="Tip: it's recommended to upload the chain of thought if you have it — it enables CoT-divergence detection.">`.
- `InfoTooltip({ text: string; children?: ReactNode })` — pure-CSS hover/focus
  tooltip (no library): when `children` given, they are the hover target; else
  render an `ⓘ` glyph. Accessibility: target has `aria-describedby` pointing
  at a `role="tooltip"` element; tooltip visible on `:hover` and `:focus-within`.
  The tooltip text must be in the DOM (hidden via CSS, not conditional render)
  so tests can assert it.
- `SummaryCard({ summary: string; modelName: string | null; warnings: string[];
  sentiment: number })` — summary paragraph; model name chip when present;
  warnings rendered as a dismissible amber strip listing each warning;
  `<SentimentBadge value={sentiment} />`.
- `SentimentBadge({ value: number })` — label `Positive` (value ≥ 0.25) /
  `Negative` (≤ −0.25) / `Neutral` (otherwise) + the value to 2 decimals;
  colors: green `#22a06b`, red `#e34935`, gray `#8b949e` (local constants —
  sentiment does not use the detection score bands).
- Upload handler:
  `analyzeFile(file, r => setState({ phase: "analyzing", progress: r.progress, statusText: r.status }))`
  → done/error transitions. Disable UploadCTA while analyzing.

## Acceptance criteria
```bash
cd frontend && npm run build && npm test
```
- `App.test.tsx` (vitest + testing-library, mock `analyzeFile` via `vi.mock`):
  idle shows the Upload Chat button and tooltip text; selecting a file
  transitions idle → analyzing (mock keeps a pending promise) → done (resolve
  with a small fixture `AnalysisResult`) and the summary text appears;
  rejecting with `Error("bad file")` shows the error panel with "bad file" and
  the Try-again button resets to idle.
- `tsc --noEmit` clean; placeholders `filters-slot` / `transcript-slot` exist
  in the done phase.
