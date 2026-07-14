interface SentimentBadgeProps {
  value: number;
}

// Monochrome: no color is used off the spectre bar, so positive/negative/
// neutral are distinguished by glyph + weight, not hue.
export default function SentimentBadge({ value }: SentimentBadgeProps) {
  let label: string;
  let glyph: string;
  if (value >= 0.25) {
    label = "Positive";
    glyph = "▲";
  } else if (value <= -0.25) {
    label = "Negative";
    glyph = "▼";
  } else {
    label = "Neutral";
    glyph = "●";
  }
  return (
    <span className="sentiment-badge">
      <span className="sentiment-badge__dot">{glyph}</span>
      {label} {value.toFixed(2)}
    </span>
  );
}
