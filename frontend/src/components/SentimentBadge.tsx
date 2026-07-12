interface SentimentBadgeProps {
  value: number;
}

// Sentiment uses its own palette — NOT the detection score bands.
const POSITIVE = "#22a06b";
const NEGATIVE = "#e34935";
const NEUTRAL = "#8b949e";

export default function SentimentBadge({ value }: SentimentBadgeProps) {
  let label: string;
  let color: string;
  if (value >= 0.25) {
    label = "Positive";
    color = POSITIVE;
  } else if (value <= -0.25) {
    label = "Negative";
    color = NEGATIVE;
  } else {
    label = "Neutral";
    color = NEUTRAL;
  }
  return (
    <span className="sentiment-badge" style={{ borderColor: color, color }}>
      <span className="sentiment-badge__dot" style={{ background: color }} />
      {label} {value.toFixed(2)}
    </span>
  );
}
