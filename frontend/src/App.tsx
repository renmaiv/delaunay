import { useState } from "react";
import type { AnalysisResult, DetectionCategory } from "./types";
import { USER_CATEGORIES, MODEL_CATEGORIES } from "./types";
import { analyzeFile } from "./api";
import UploadCTA from "./components/UploadCTA";
import SummaryCard from "./components/SummaryCard";
import Spinner from "./components/Spinner";
import TranscriptView from "./components/TranscriptView";
import Tabs from "./components/Tabs";
import FilterPanel from "./components/FilterPanel";

type AppState =
  | { phase: "idle" }
  | { phase: "analyzing"; progress: number; statusText: string }
  | { phase: "done"; result: AnalysisResult }
  | { phase: "error"; message: string };

function initialFilters(): Record<DetectionCategory, boolean> {
  const f = {} as Record<DetectionCategory, boolean>;
  for (const c of [...USER_CATEGORIES, ...MODEL_CATEGORIES]) f[c] = true;
  return f;
}

const HEADER_BLURB =
  "Upload a chat transcript to see where the conversation went sideways — " +
  "jailbreak attempts, pressure, appeasement, over-compliance — as a heatmap " +
  "over the transcript.";

export default function App() {
  const [state, setState] = useState<AppState>({ phase: "idle" });
  const [activeTab, setActiveTab] = useState<"model" | "user">("model");
  const [filters, setFilters] =
    useState<Record<DetectionCategory, boolean>>(initialFilters);

  async function handleFile(file: File) {
    setState({ phase: "analyzing", progress: 0, statusText: "Uploading…" });
    try {
      const result = await analyzeFile(file, (r) =>
        setState({
          phase: "analyzing",
          progress: r.progress,
          statusText: r.status,
        }),
      );
      setState({ phase: "done", result });
    } catch (err) {
      setState({
        phase: "error",
        message: err instanceof Error ? err.message : "Analysis failed.",
      });
    }
  }

  function setFilter(c: DetectionCategory, checked: boolean) {
    setFilters((prev) => ({ ...prev, [c]: checked }));
  }

  const analyzing = state.phase === "analyzing";

  // Props for the done phase (computed regardless so child wiring is stable).
  const result = state.phase === "done" ? state.result : null;
  const modelName = result?.model_name ?? null;
  const hasCot = result ? result.turns.some((t) => !!t.cot) : false;

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__title">
          <h1>Semantic Observability</h1>
          <p className="app__blurb">{HEADER_BLURB}</p>
        </div>
        <UploadCTA onFile={handleFile} disabled={analyzing} />
      </header>

      {state.phase === "analyzing" && (
        <div className="app__status">
          <Spinner />
          <span>{Math.round(state.progress * 100)}%</span>
          <span className="app__status-text">{state.statusText}</span>
        </div>
      )}

      {state.phase === "error" && (
        <div className="app__error" role="alert">
          <p>{state.message}</p>
          <button
            type="button"
            className="btn"
            onClick={() => setState({ phase: "idle" })}
          >
            Try again
          </button>
        </div>
      )}

      {result && (
        <>
          <SummaryCard
            summary={result.summary}
            modelName={modelName}
            warnings={result.meta.warnings}
            sentiment={result.overall_sentiment}
          />
          <div className="app__analysis">
            <div className="app__filters" data-testid="filters-slot">
              <Tabs
                active={activeTab}
                onChange={setActiveTab}
                modelName={modelName}
              />
              <FilterPanel
                side={activeTab}
                filters={filters}
                onChange={setFilter}
                hasCot={hasCot}
              />
            </div>
            <div className="app__transcript" data-testid="transcript-slot">
              <TranscriptView
                turns={result.turns}
                causalLinks={result.causal_links}
                modelName={modelName}
                filters={filters}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
