import type { DetectionCategory, Turn } from "../types";
import { TAXONOMY } from "../taxonomy";
import DetectionCaption from "./DetectionCaption";

interface TurnBubbleProps {
  turn: Turn;
  filters: Record<DetectionCategory, boolean>;
  modelName: string | null;
}

function sentimentColor(value: number): string {
  if (value >= 0.25) return "#22a06b";
  if (value <= -0.25) return "#e34935";
  return "#8b949e";
}

export default function TurnBubble({ turn, filters, modelName }: TurnBubbleProps) {
  const isUser = turn.role === "user";
  const roleLabel = isUser ? "You" : modelName ?? "Model";

  const visible = turn.detections.filter(
    (d) => filters[d.category] && d.score >= TAXONOMY.display_threshold,
  );

  return (
    <div className={`turn turn--${isUser ? "user" : "assistant"}`}>
      {visible.length > 0 && <DetectionCaption detections={visible} />}
      <div className="turn__bubble">
        <div className="turn__role">
          <span className="turn__role-name">{roleLabel}</span>
          {isUser && turn.sentiment != null && (
            <span
              className="turn__sentiment-dot"
              title={turn.sentiment.toFixed(2)}
              style={{ background: sentimentColor(turn.sentiment) }}
            />
          )}
        </div>
        <div className="turn__content">{turn.content}</div>
        {turn.cot && (
          <details className="turn__cot">
            <summary>Chain of thought</summary>
            <div className="turn__cot-body">{turn.cot}</div>
          </details>
        )}
      </div>
    </div>
  );
}
