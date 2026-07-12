import type { DetectionCategory } from "../types";
import { USER_CATEGORIES, MODEL_CATEGORIES } from "../types";
import { TAXONOMY } from "../taxonomy";
import InfoTooltip from "./InfoTooltip";

interface FilterPanelProps {
  side: "model" | "user";
  filters: Record<DetectionCategory, boolean>;
  onChange: (c: DetectionCategory, checked: boolean) => void;
  hasCot?: boolean;
}

/**
 * Filter semantics: the filter state is GLOBAL. Both the user and model
 * selections always apply to the transcript captions and the spectre bar; the
 * active tab only chooses which side's panel is displayed here. Unchecking
 * "Safety triggered" removes its captions and splashes even while the User tab
 * is open. Labels and tooltip copy come ONLY from TAXONOMY (shared/taxonomy.json).
 */
export default function FilterPanel({
  side,
  filters,
  onChange,
  hasCot = false,
}: FilterPanelProps) {
  const categories = side === "user" ? USER_CATEGORIES : MODEL_CATEGORIES;
  const visible = categories.filter(
    (c) => c !== "cot_divergence" || hasCot,
  );

  return (
    <div className="filter-panel">
      {visible.map((c) => (
        <label key={c} className="filter-panel__row">
          <input
            type="checkbox"
            checked={filters[c]}
            onChange={(e) => onChange(c, e.target.checked)}
          />
          <span className="filter-panel__label">{TAXONOMY.categories[c].label}</span>
          <InfoTooltip text={TAXONOMY.categories[c].tooltip} />
        </label>
      ))}
    </div>
  );
}
