import { useRef, useState } from "react";
import { Analytics } from "@vercel/analytics/react";
import type { AnalysisResult, DetectionCategory } from "./types";
import { USER_CATEGORIES, MODEL_CATEGORIES } from "./types";
import { analyzeFile } from "./api";
import { EXAMPLE_ANALYSIS } from "./example";
import UploadCTA from "./components/UploadCTA";
import SummaryCard from "./components/SummaryCard";
import ExampleBadge from "./components/ExampleBadge";
import Spinner from "./components/Spinner";
import TranscriptView from "./components/TranscriptView";
import Tabs from "./components/Tabs";
import FilterPanel from "./components/FilterPanel";
import ConversationsButton from "./components/ConversationsButton";
import ConversationsPage from "./components/ConversationsPage";
import KeyOverlay from "./components/KeyOverlay";

type AppState =
  | { phase: "idle" }
  | { phase: "analyzing"; progress: number; statusText: string }
  | { phase: "done"; result: AnalysisResult; isExample: boolean }
  | { phase: "error"; message: string };

function initialFilters(): Record<DetectionCategory, boolean> {
  const f = {} as Record<DetectionCategory, boolean>;
  for (const c of [...USER_CATEGORIES, ...MODEL_CATEGORIES]) f[c] = true;
  return f;
}

const HEADER_BLURB =
  "Upload a chat transcript to see where the conversation went sideways — " +
  "jailbreak attempts, pressure, appeasement, over-compliance — as a heatmap " +
  "over the transcript. We don't see or store your chats: analysis runs " +
  "transiently and nothing is kept.";

export default function App() {
  // Open with the pre-evaluated example so the landing page shows the tool in
  // action with no backend call.
  const [state, setState] = useState<AppState>({
    phase: "done",
    result: EXAMPLE_ANALYSIS,
    isExample: true,
  });
  const [activeTab, setActiveTab] = useState<"model" | "user">("model");
  const [view, setView] = useState<"analysis" | "conversations">("analysis");
  const [filters, setFilters] =
    useState<Record<DetectionCategory, boolean>>(initialFilters);
  // BYOK: the user's own Anthropic key, held in memory for this session only
  // (never persisted), plus the last uploaded file so we can re-run with it.
  const [userKey, setUserKey] = useState<string | null>(null);
  const lastFileRef = useRef<File | null>(null);

  async function handleFile(file: File, keyOverride?: string) {
    lastFileRef.current = file;
    setState({ phase: "analyzing", progress: 0, statusText: "Uploading…" });
    try {
      const result = await analyzeFile(
        file,
        (r) =>
          setState({
            phase: "analyzing",
            progress: r.progress,
            statusText: r.status,
          }),
        keyOverride ?? userKey ?? undefined,
      );
      setState({ phase: "done", result, isExample: false });
    } catch (err) {
      setState({
        phase: "error",
        message: err instanceof Error ? err.message : "Analysis failed.",
      });
    }
  }

  function handleUserKey(key: string) {
    setUserKey(key);
    if (lastFileRef.current) {
      void handleFile(lastFileRef.current, key);
    }
  }

  function showExample() {
    setState({ phase: "done", result: EXAMPLE_ANALYSIS, isExample: true });
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
          <h1>Delaunay</h1>
          <p className="app__blurb">{HEADER_BLURB}</p>
        </div>
        <UploadCTA onFile={handleFile} disabled={analyzing} />
      </header>

      {state.phase === "done" && state.isExample && (
        <ExampleBadge onClear={() => setState({ phase: "idle" })} />
      )}

      {state.phase === "idle" && (
        <div className="app__idle">
          <p>Upload a chat transcript to analyze it, or</p>
          <button type="button" className="btn" onClick={showExample}>
            view the example again
          </button>
        </div>
      )}

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

      {result && view === "conversations" && (
        <ConversationsPage
          onBack={() => setView("analysis")}
          onSelect={(analysis) => {
            setState({ phase: "done", result: analysis, isExample: true });
            setView("analysis");
          }}
        />
      )}

      {result && view === "analysis" && (
        <div className="app__result">
          <SummaryCard
            modelName={modelName}
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
              <ConversationsButton onClick={() => setView("conversations")} />
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
          {state.phase === "done" &&
            !state.isExample &&
            result.meta.judge_error === "auth" && (
              <KeyOverlay onSubmit={handleUserKey} />
            )}
        </div>
      )}
      <Analytics />
    </div>
  );
}
