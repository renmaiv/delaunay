import type { Detection } from "../types";
import { TAXONOMY, scoreColor } from "../taxonomy";

interface DetectionCaptionProps {
  detections: Detection[];
}

/**
 * Small caption row rendered ABOVE a flagged bubble. Each detection reads
 * "{short} {score.toFixed(1)}" colored by scoreColor(score). A 1px vertical
 * connector line to the bubble is drawn via the ::before pseudo-element (CSS).
 */
export default function DetectionCaption({ detections }: DetectionCaptionProps) {
  return (
    <div className="detection-caption">
      {detections.map((d, i) => (
        <span
          key={`${d.category}-${i}`}
          className="detection-caption__item"
          data-category={d.category}
          data-score={d.score}
          style={{ color: scoreColor(d.score) }}
          title={d.rationale ?? d.evidence_span ?? ""}
        >
          {TAXONOMY.categories[d.category].short} {d.score.toFixed(1)}
        </span>
      ))}
    </div>
  );
}
