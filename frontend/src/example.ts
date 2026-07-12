// The pre-evaluated example analysis shown on first load. Regenerate with the
// real judge via `python -m eval.make_example --provider anthropic`.
import exampleJson from "./exampleAnalysis.json";
import type { AnalysisResult } from "./types";

export const EXAMPLE_ANALYSIS = exampleJson as unknown as AnalysisResult;
