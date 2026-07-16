import type { AnalysisResult, DetectionCategory, Turn } from "../types";
import { USER_CATEGORIES, MODEL_CATEGORIES } from "../types";
import { scoreColor, maxVisibleScore } from "../taxonomy";
import { CONVERSATION_EXAMPLES } from "../conversations";

interface ConversationsPageProps {
  onBack: () => void;
  onSelect: (analysis: AnalysisResult) => void;
}

// Spine colors come from the real per-turn detections with every category
// visible, mapping each turn to its severity-band color.
const ALL_ON = (() => {
  const f = {} as Record<DetectionCategory, boolean>;
  for (const c of [...USER_CATEGORIES, ...MODEL_CATEGORIES]) f[c] = true;
  return f;
})();

/** Stepped vertical gradient over the conversation's turns: one band per turn,
 *  colored by that turn's highest-scoring detection (green when clean). */
function makeSpine(turns: Turn[]): string {
  const bands = turns.length || 1;
  const stops: string[] = [];
  turns.forEach((turn, i) => {
    const color = scoreColor(maxVisibleScore(turn, ALL_ON).score);
    const start = (i / bands) * 100;
    const end = ((i + 1) / bands) * 100;
    stops.push(`${color} ${start.toFixed(2)}%`, `${color} ${end.toFixed(2)}%`);
  });
  return `linear-gradient(180deg, ${stops.join(", ")})`;
}

/** Grid of "compressed" conversation previews: each cell is a 100×100 spectre
 *  spine plus the conversation title and its real turn count. The analyses are
 *  pre-evaluated by the real judge (scripts/make_conversation_examples.py). */
export default function ConversationsPage({
  onBack,
  onSelect,
}: ConversationsPageProps) {
  return (
    <div className="conversations-page">
      <div className="conversations-page__toolbar">
        <button type="button" className="conversations-btn" onClick={onBack}>
          <svg
            className="conversations-btn__arrow"
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M10 3L5 8l5 5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span>Back</span>
        </button>
      </div>
      <div className="conversations-grid">
        {CONVERSATION_EXAMPLES.map((c) => (
          <button
            key={c.id}
            type="button"
            className="conversation-cell"
            title={c.title}
            aria-label={`Open ${c.title}`}
            onClick={() => onSelect(c.analysis)}
          >
            <div
              className="conversation-cell__spine"
              style={{ background: makeSpine(c.analysis.turns) }}
              aria-hidden="true"
            />
            <div className="conversation-cell__title">{c.title}</div>
            <div className="conversation-cell__turns">
              {c.analysis.turns.length} turns
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
