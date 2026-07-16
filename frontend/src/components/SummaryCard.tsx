import SentimentBadge from "./SentimentBadge";

interface SummaryCardProps {
  modelName: string | null;
  sentiment: number;
}

// The conversation-summary text is intentionally not rendered: we don't
// surface (or store) the contents of the user's chat beyond the transcript
// they are already looking at. Warning messages are also not surfaced.
export default function SummaryCard({ modelName, sentiment }: SummaryCardProps) {
  return (
    <section className="summary-card">
      <div className="summary-card__header">
        <div className="summary-card__meta">
          {modelName && <span className="chip chip--model">{modelName}</span>}
          <SentimentBadge value={sentiment} />
        </div>
      </div>
    </section>
  );
}
