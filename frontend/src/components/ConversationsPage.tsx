interface ConversationsPageProps {
  onBack: () => void;
}

// Fixed "spectre spine" gradient used for every compressed conversation cell.
const CELL_GRADIENT =
  "linear-gradient(180deg, #008300 0%, #02cf02 18.27%, #f2ea00 34.12%, " +
  "#f2ea00 34.13%, #008d00 48.08%, #02cf02 52.88%, #ff3700 60.57%, " +
  "#ff3700 60.58%, #008900 80.29%, #01ce01 87.5%)";

interface MockConversation {
  id: number;
  turns: number;
}

// 32 mock conversations for the grid preview. Deterministic so the layout is
// stable across renders (this is placeholder data, not real analysis output).
const MOCK: MockConversation[] = Array.from({ length: 32 }, (_, i) => ({
  id: 200 + i * 7 + (i % 3) * 2,
  turns: 6 + ((i * 5) % 40),
}));

/** Grid of "compressed" conversation previews: each cell is a 100×100 spectre
 *  spine plus the chat id and turn count. */
export default function ConversationsPage({ onBack }: ConversationsPageProps) {
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
        {MOCK.map((c) => (
          <div key={c.id} className="conversation-cell">
            <div
              className="conversation-cell__spine"
              style={{ background: CELL_GRADIENT }}
              aria-hidden="true"
            />
            <div className="conversation-cell__id">#{c.id}</div>
            <div className="conversation-cell__turns">{c.turns} turns</div>
          </div>
        ))}
      </div>
    </div>
  );
}
