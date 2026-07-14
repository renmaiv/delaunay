interface ConversationsPageProps {
  onBack: () => void;
}

// Palette for the compressed "spectre spine": mostly greens with occasional
// yellow/orange/red hotspots, matching the severity bands on the analysis bar.
const GREENS = [
  "#008300",
  "#02cf02",
  "#008d00",
  "#008900",
  "#01ce01",
  "#007200",
  "#00a000",
];
const HOT = ["#f2ea00", "#ff8000", "#ff3700"];

// Deterministic PRNG so each cell's gradient is stable across renders but the
// 32 cells differ from one another.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Build a blocky vertical gradient standing in for a conversation's per-turn
// severity: each band is one turn, greens dominate, hotter colors are rarer.
function makeSpine(seed: number): string {
  const rand = mulberry32(seed);
  const bands = 7 + Math.floor(rand() * 5); // 7–11 bands
  const stops: string[] = [];
  for (let i = 0; i < bands; i++) {
    const r = rand();
    let color: string;
    if (r < 0.68) color = GREENS[Math.floor(rand() * GREENS.length)];
    else if (r < 0.82) color = HOT[0];
    else if (r < 0.93) color = HOT[1];
    else color = HOT[2];
    const start = (i / bands) * 100;
    const end = ((i + 1) / bands) * 100;
    stops.push(`${color} ${start.toFixed(2)}%`, `${color} ${end.toFixed(2)}%`);
  }
  return `linear-gradient(180deg, ${stops.join(", ")})`;
}

interface MockConversation {
  id: number;
  turns: number;
  spine: string;
}

// 32 mock conversations for the grid preview. Deterministic so the layout is
// stable across renders (this is placeholder data, not real analysis output).
const MOCK: MockConversation[] = Array.from({ length: 32 }, (_, i) => {
  const id = 200 + i * 7 + (i % 3) * 2;
  return {
    id,
    turns: 6 + ((i * 5) % 40),
    spine: makeSpine(id + i * 101),
  };
});

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
              style={{ background: c.spine }}
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
