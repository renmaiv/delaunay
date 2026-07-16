// CONTRACT FILE — copy verbatim to frontend/src/types.ts in task WS-A3.
// Must mirror contracts/schemas.py exactly. Any change to one requires
// changing the other in the same PR.

export type Role = "user" | "assistant" | "system";

export type DetectionCategory =
  | "jailbreak_steering"
  | "social_engineering"
  | "coercive_pressure"
  | "repair_request"
  | "safety_triggered"
  | "appeasement"
  | "overcompliant"
  | "cot_divergence";

export const USER_CATEGORIES: DetectionCategory[] = [
  "jailbreak_steering",
  "social_engineering",
  "coercive_pressure",
  "repair_request",
];

export const MODEL_CATEGORIES: DetectionCategory[] = [
  "safety_triggered",
  "appeasement",
  "overcompliant",
  "cot_divergence",
];

export interface Detection {
  category: DetectionCategory;
  score: number;
  source: "encoder" | "judge" | "rules";
  evidence_span?: string | null;
  rationale?: string | null;
  calibrated: boolean;
}

export interface Turn {
  index: number;
  role: Role;
  content: string;
  cot?: string | null;
  sentiment?: number | null;
  detections: Detection[];
}

export interface CausalLink {
  from_turn: number;
  to_turn: number;
  from_category: DetectionCategory;
  to_category: DetectionCategory;
  score: number;
  rationale?: string | null;
}

export interface AnalysisMeta {
  judge_provider: string;
  judge_model?: string | null;
  encoders_available: Record<string, boolean>;
  warnings: string[];
  /** "auth" when the judge could not run because the server's Anthropic key
   *  is missing, invalid, or out of credit — the client may offer BYOK. */
  judge_error?: "auth" | null;
}

export interface AnalysisResult {
  conversation_id: string;
  model_name?: string | null;
  summary: string;
  overall_sentiment: number;
  turns: Turn[];
  causal_links: CausalLink[];
  meta: AnalysisMeta;
}

export type AnalysisStatus = "pending" | "running" | "completed" | "failed";

export interface AnalysisResponse {
  analysis_id: string;
  status: AnalysisStatus;
  progress: number;
  result?: AnalysisResult | null;
  error?: string | null;
}
