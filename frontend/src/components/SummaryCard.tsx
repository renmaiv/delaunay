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
  sentiment,
}: SummaryCardProps) {
  // Warning messages are intentionally not surfaced in the UI.
  return (
    <section className="summary-card">
      <div className="summary-card__header">
        <div className="summary-card__meta">
          {modelName && <span className="chip chip--model">{modelName}</span>}
          <SentimentBadge value={sentiment} />
        </div>
      </div>

      <p className="summary-card__text">{summary}</p>
    </section>
  );
}
