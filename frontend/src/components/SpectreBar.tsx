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

export interface SpectreMarker {
  turnIndex: number;
  top: number;
  category: DetectionCategory;
  score: number;
  stackIndex: number;
}

/**
 * Pure geometry: one marker per VISIBLE detection on a turn (not just the
 * max, unlike segmentsFromMeasurements) — each detection gets its own note
 * on the bar. stackIndex orders multiple detections on the same turn so they
 * can be rendered with a vertical offset instead of overlapping. No DOM
 * access — unit-testable.
 */
export function markersFromMeasurements(
  turns: Turn[],
  measurements: Map<number, { top: number; height: number }>,
  filters: Record<DetectionCategory, boolean>,
): SpectreMarker[] {
  const markers: SpectreMarker[] = [];
  for (const turn of turns) {
    const m = measurements.get(turn.index);
    if (!m) continue;
    const visible = turn.detections.filter(
      (d) => filters[d.category] && d.score >= TAXONOMY.display_threshold,
    );
    visible.forEach((d, i) => {
      markers.push({
        turnIndex: turn.index,
        top: m.top,
        category: d.category,
        score: d.score,
        stackIndex: i,
      });
    });
  }
  return markers;
}

// The "normal" (no-detection) stretch of the bar is a green gradient rather
// than a flat fill; the severity splashes paint over it where detections land.
const NORMAL_GRADIENT =
  "linear-gradient(2.42deg, #00d800 -4.26%, #007200 113.21%)";

// Vertical spacing between stacked markers on the same turn.
const STACK_OFFSET_PX = 15;

interface SpectreBarProps {
  segments: SpectreSegment[];
  markers: SpectreMarker[];
  totalHeight: number;
}

/**
 * The 20px severity bar (the app's one color exception) plus, for each
 * visible detection, a plain-white note marker: a 16px outline rhombus
 * sitting on the bar with a short horizontal line running from its right
 * corner out into the gutter, toward where the caption sits above it.
 */
export default function SpectreBar({ segments, markers, totalHeight }: SpectreBarProps) {
  return (
    <div className="spectre-bar-layer" style={{ height: totalHeight }}>
      <div
        className="spectre-bar"
        style={{ background: NORMAL_GRADIENT }}
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
      <div className="spectre-bar__markers" aria-hidden="true">
        {markers.map((mk) => (
          <div
            key={`${mk.turnIndex}-${mk.category}`}
            className="spectre-bar__marker"
            data-category={mk.category}
            data-score={mk.score}
            title={`${TAXONOMY.categories[mk.category].short} ${mk.score.toFixed(1)}`}
            style={{ top: mk.top - mk.stackIndex * STACK_OFFSET_PX }}
          >
            <span className="spectre-bar__rhombus" />
            <span className="spectre-bar__connector-line" />
          </div>
        ))}
      </div>
    </div>
  );
}
