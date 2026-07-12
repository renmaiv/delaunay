import type { DetectionCategory, Turn } from "../types";
import { TAXONOMY, scoreColor, maxVisibleScore } from "../taxonomy";

export interface SpectreSegment {
  turnIndex: number;
  top: number;
  height: number;
  score: number;
  category: DetectionCategory | null;
}

/**
 * Pure geometry: one segment per turn, positioned from the measured
 * top/height, with score/category taken from maxVisibleScore(turn, filters).
 * Turns without a measurement are skipped. No DOM access — unit-testable.
 */
export function segmentsFromMeasurements(
  turns: Turn[],
  measurements: Map<number, { top: number; height: number }>,
  filters: Record<DetectionCategory, boolean>,
): SpectreSegment[] {
  const segments: SpectreSegment[] = [];
  for (const turn of turns) {
    const m = measurements.get(turn.index);
    if (!m) continue;
    const { score, category } = maxVisibleScore(turn, filters);
    segments.push({
      turnIndex: turn.index,
      top: m.top,
      height: m.height,
      score,
      category,
    });
  }
  return segments;
}

const NORMAL_COLOR =
  TAXONOMY.score_bands.find((b) => b.label === "normal")?.color ?? "#22a06b";

interface SpectreBarProps {
  segments: SpectreSegment[];
  totalHeight: number;
}

export default function SpectreBar({ segments, totalHeight }: SpectreBarProps) {
  return (
    <div
      className="spectre-bar"
      style={{ height: totalHeight, background: NORMAL_COLOR }}
      aria-hidden="true"
    >
      {segments
        .filter((s) => s.category !== null && s.score >= TAXONOMY.display_threshold)
        .map((s) => {
          const color = scoreColor(s.score);
          const category = s.category as DetectionCategory;
          const short = TAXONOMY.categories[category].short;
          return (
            <div
              key={s.turnIndex}
              className="spectre-bar__splash"
              data-category={category}
              data-score={s.score}
              title={`${short} ${s.score.toFixed(1)}`}
              style={{
                top: s.top,
                height: s.height,
                background: `linear-gradient(to bottom, transparent, ${color} 12px, ${color} calc(100% - 12px), transparent)`,
              }}
            />
          );
        })}
    </div>
  );
}
