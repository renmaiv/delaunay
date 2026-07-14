import type { Detection } from "../types";
import { TAXONOMY } from "../taxonomy";

interface DetectionCaptionProps {
  detections: Detection[];
}

/**
 * White caption list rendered ABOVE a flagged bubble, one line per
 * detection: "{short} {score.toFixed(1)}". Each detection's connecting line
 * + outline rhombus render on the spectre bar itself (SpectreBar.tsx),
 * stacked in the same order as these lines, so the caption visually sits
 * above its marker without the two components needing pixel-level coupling.
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
          title={d.rationale ?? d.evidence_span ?? ""}
        >
          {TAXONOMY.categories[d.category].short} {d.score.toFixed(1)}
        </span>
      ))}
    </div>
  );
}
