# WS-E3 — Tabs, filter checkboxes, tooltip panel

- **Branch:** `feat/ws-e3-tabs-filters`
- **Depends on:** WS-E1 (parallel with WS-E2)
- **Blocks:** WS-G1

## Files to create / modify
- `frontend/src/components/Tabs.tsx`
- `frontend/src/components/FilterPanel.tsx`
- `frontend/src/components/FilterPanel.test.tsx`
- **Modify `App.tsx` ONLY by replacing the `filters-slot` placeholder** with
  `<Tabs …/>` + `<FilterPanel …/>` wired to the existing `activeTab` /
  `filters` state (state itself stays in App — do not move it).
- **Append to `styles.css` ONLY inside a trailing block marked
  `/* --- E3: tabs & filters --- */`.**

## Files you must NOT touch
Everything else, including E1's components and E2's files.

## Spec

### `Tabs`
```tsx
interface TabsProps {
  active: "model" | "user";
  onChange: (t: "model" | "user") => void;
  modelName: string | null;
}
```
Two tabs with `role="tablist"` / `role="tab"` / `aria-selected` semantics.
Model tab label: `modelName ? `Model: ${modelName}` : "Model"` — e.g.
`Model: chatgpt 5.0` when the uploaded file's metadata carried a model name.
User tab label: `"User"`.

### `FilterPanel`
```tsx
interface FilterPanelProps {
  side: "model" | "user";
  filters: Record<DetectionCategory, boolean>;
  onChange: (c: DetectionCategory, checked: boolean) => void;
  hasCot?: boolean;   // default false
}
```
- Iterates `side === "user" ? USER_CATEGORIES : MODEL_CATEGORIES` (from
  `types.ts`).
- Hide the `cot_divergence` row when `hasCot` is false (no CoT in the upload →
  the checkbox would be dead weight).
- Each row:
  ```tsx
  <label>
    <input type="checkbox" checked={filters[c]}
           onChange={e => onChange(c, e.target.checked)} />
    {TAXONOMY.categories[c].label}
    <InfoTooltip text={TAXONOMY.categories[c].tooltip} />
  </label>
  ```
  Labels and tooltip texts come ONLY from `TAXONOMY` (never hardcode copy —
  the product owner's definitions live in `shared/taxonomy.json`).
- All checkboxes default checked (App already initializes filter state; this
  component is fully controlled).
- **Filter semantics (document in a comment):** filter state is GLOBAL — both
  sides' selections always apply to the transcript and spectre bar; the tab
  only chooses which side's panel is displayed. Unchecking "Safety triggered"
  removes its captions and splashes even while the User tab is open.
- In `App.tsx`, compute `hasCot = result.turns.some(t => !!t.cot)`.

## Acceptance criteria
```bash
cd frontend && npm run build && npm test
```
- `FilterPanel.test.tsx`: model side renders 3 checkboxes without `hasCot`,
  4 with `hasCot`; user side renders 4; label texts equal
  `TAXONOMY.categories[c].label`; the tooltip text for Appeasement (from
  taxonomy.json) is present in the DOM; clicking a checkbox fires `onChange`
  with the right category and value; tab renders `Model: chatgpt 5.0` given
  `modelName="chatgpt 5.0"` and plain `Model` given null.
- `tsc --noEmit` clean; no modifications outside the two allowed edit points.
