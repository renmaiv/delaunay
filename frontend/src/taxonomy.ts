import taxonomyJson from "../../shared/taxonomy.json";
import type { DetectionCategory, Turn } from "./types";

export interface CategoryInfo {
  side: "user" | "model";
  source: string;
  label: string;
  short: string;
  tooltip: string;
}
export interface ScoreBand {
  min: number;
  max: number;
  label: string;
  color: string;
}
export const TAXONOMY = taxonomyJson as unknown as {
  version: number;
  display_threshold: number;
  score_bands: ScoreBand[];
  categories: Record<DetectionCategory, CategoryInfo>;
};

/** Color for a score: first band where min <= score < max; the last band is
 *  max-inclusive so scoreColor(1.0) returns the "high" color. */
export function scoreColor(score: number): string {
  const bands = TAXONOMY.score_bands;
  for (let i = 0; i < bands.length; i++) {
    const band = bands[i];
    const isLast = i === bands.length - 1;
    if (score >= band.min && (score < band.max || (isLast && score <= band.max))) {
      return band.color;
    }
  }
  // Fallback: clamp to the nearest edge band.
  return score < bands[0].min ? bands[0].color : bands[bands.length - 1].color;
}

/** Highest-scoring detection on the turn among categories whose filter is on
 *  and whose score >= display_threshold. Returns {score: 0, category: null}
 *  when none qualify. */
export function maxVisibleScore(
  t: Turn,
  filters: Record<DetectionCategory, boolean>,
): { score: number; category: DetectionCategory | null } {
  let best: { score: number; category: DetectionCategory | null } = {
    score: 0,
    category: null,
  };
  for (const d of t.detections) {
    if (!filters[d.category]) continue;
    if (d.score < TAXONOMY.display_threshold) continue;
    if (d.score > best.score) {
      best = { score: d.score, category: d.category };
    }
  }
  return best;
}
