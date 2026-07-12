# WS-A3 — Frontend scaffold: Vite + React + TS, types, taxonomy, API client

- **Branch:** `feat/ws-a3-frontend-scaffold`
- **Depends on:** WS-A1 (needs `shared/taxonomy.json` merged)
- **Blocks:** E1 (and transitively E2, E3)

## Files to create
- `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`,
  `frontend/index.html`
- `frontend/src/main.tsx`, `frontend/src/App.tsx` (placeholder shell),
  `frontend/src/styles.css` (reset + CSS variables only)
- `frontend/src/types.ts` — copy **verbatim** from `docs/sop/contracts/types.ts`
- `frontend/src/taxonomy.ts`, `frontend/src/taxonomy.test.ts`
- `frontend/src/api.ts`
- Append to root `.gitignore`: `frontend/node_modules/`, `frontend/dist/`

## Files you must NOT touch
Everything under `server/`, `tests/`, root configs other than `.gitignore`.

## Spec

### Tooling
- Dependencies: `react`, `react-dom`. DevDependencies: `typescript`, `vite`,
  `@vitejs/plugin-react`, `vitest`, `@testing-library/react`, `jsdom`,
  `@types/react`, `@types/react-dom`.
- **No UI framework, no CSS framework, no runtime CDN dependencies.** All
  styling is hand-rolled CSS bundled by Vite.
- `package.json` scripts: `dev` (vite), `build` (`tsc --noEmit && vite build`),
  `test` (`vitest run`).
- `vite.config.ts`: `@vitejs/plugin-react`;
  `server: { proxy: { "/api": "http://localhost:8000" }, fs: { allow: [".."] } }`;
  `test: { environment: "jsdom" }`.
- `tsconfig.json`: `strict: true`, `resolveJsonModule: true`, `jsx: "react-jsx"`.

### `frontend/src/taxonomy.ts`
```ts
import taxonomyJson from "../../shared/taxonomy.json";
import type { DetectionCategory, Turn } from "./types";

export interface CategoryInfo {
  side: "user" | "model"; source: string;
  label: string; short: string; tooltip: string;
}
export interface ScoreBand { min: number; max: number; label: string; color: string; }
export const TAXONOMY = taxonomyJson as unknown as {
  version: number;
  display_threshold: number;
  score_bands: ScoreBand[];
  categories: Record<DetectionCategory, CategoryInfo>;
};

/** Color for a score: first band where min <= score < max; the last band is
 *  max-inclusive so scoreColor(1.0) returns the "high" color. */
export function scoreColor(score: number): string;

/** Highest-scoring detection on the turn among categories whose filter is on
 *  and whose score >= display_threshold. Returns {score: 0, category: null}
 *  when none qualify. */
export function maxVisibleScore(
  t: Turn,
  filters: Record<DetectionCategory, boolean>,
): { score: number; category: DetectionCategory | null };
```

### `frontend/src/api.ts`
Implement exactly per `docs/sop/contracts/api_contract.md` § "Client polling
protocol":
```ts
import type { AnalysisResponse, AnalysisResult } from "./types";
export async function analyzeFile(
  file: File,
  onStatus?: (r: AnalysisResponse) => void,
): Promise<AnalysisResult>;
export async function fetchTaxonomy(): Promise<unknown>; // GET /api/taxonomy
```
Poll interval 1500 ms, overall timeout 5 min, throw `Error` with the server's
`detail`/`error` message when available.

### `frontend/src/App.tsx` (placeholder — rewritten in WS-E1)
Header with title "Semantic Observability", a one-line description, and a
disabled "Upload Chat" button. Nothing else. Keep it trivial; E1 replaces it.

## Acceptance criteria
```bash
cd frontend && npm ci && npm run build && npm test
```
- `npm run build` succeeds and produces `frontend/dist/index.html`.
- `taxonomy.test.ts` (vitest): `scoreColor(0.0) === "#22a06b"`,
  `scoreColor(0.2) === "#f5cd47"`, `scoreColor(0.5) === "#f5820b"`,
  `scoreColor(0.9) === "#e34935"`, `scoreColor(1.0) === "#e34935"`;
  `maxVisibleScore` returns the max eligible detection and respects both the
  filter map and `display_threshold` (test all three branches).
- `frontend/src/types.ts` is byte-identical to `docs/sop/contracts/types.ts`
  minus the header comment (verify by eye; do not add fields).
