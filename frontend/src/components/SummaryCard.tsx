import { useState } from "react";
import SentimentBadge from "./SentimentBadge";

interface SummaryCardProps {
  summary: string;
  modelName: string | null;
  warnings: string[];
  sentiment: number;
}

export default function SummaryCard({
  summary,
  modelName,
  warnings,
  sentiment,
}: SummaryCardProps) {
  const [warningsDismissed, setWarningsDismissed] = useState(false);

  return (
    <section className="summary-card">
      <div className="summary-card__header">
        <div className="summary-card__meta">
          {modelName && <span className="chip chip--model">{modelName}</span>}
          <SentimentBadge value={sentiment} />
        </div>
      </div>

      {warnings.length > 0 && !warningsDismissed && (
        <div className="summary-card__warnings" role="alert">
          <ul>
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
          <button
            type="button"
            className="summary-card__warnings-dismiss"
            aria-label="Dismiss warnings"
            onClick={() => setWarningsDismissed(true)}
          >
            ×
          </button>
        </div>
      )}

      <p className="summary-card__text">{summary}</p>
    </section>
  );
}
