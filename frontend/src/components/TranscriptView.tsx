import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CausalLink, DetectionCategory, Turn } from "../types";
import TurnBubble from "./TurnBubble";
import SpectreBar, {
  segmentsFromMeasurements,
  type SpectreSegment,
} from "./SpectreBar";
import CausalLinkChip from "./CausalLinkChip";

interface TranscriptViewProps {
  turns: Turn[];
  causalLinks: CausalLink[];
  modelName: string | null;
  filters: Record<DetectionCategory, boolean>;
}

const FLASH_MS = 1200;

export default function TranscriptView({
  turns,
  causalLinks,
  modelName,
  filters,
}: TranscriptViewProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const turnRefs = useRef<Map<number, HTMLElement>>(new Map());
  const [segments, setSegments] = useState<SpectreSegment[]>([]);
  const [totalHeight, setTotalHeight] = useState(0);

  const measure = useCallback(() => {
    const measurements = new Map<number, { top: number; height: number }>();
    let maxBottom = 0;
    for (const [index, el] of turnRefs.current) {
      const top = el.offsetTop;
      const height = el.offsetHeight;
      measurements.set(index, { top, height });
      if (top + height > maxBottom) maxBottom = top + height;
    }
    setSegments(segmentsFromMeasurements(turns, measurements, filters));
    const wrap = wrapRef.current;
    setTotalHeight(Math.max(maxBottom, wrap ? wrap.scrollHeight : 0));
  }, [turns, filters]);

  // Recompute on turns/filters change.
  useLayoutEffect(() => {
    measure();
  }, [measure]);

  // Recompute when the column resizes.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [measure]);

  const navigateTo = useCallback((fromTurn: number) => {
    const el = turnRefs.current.get(fromTurn);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("flash");
    window.setTimeout(() => el.classList.remove("flash"), FLASH_MS);
  }, []);

  // Causal links keyed by to_turn, keeping only links whose BOTH categories
  // pass the current filters.
  const linksByToTurn = new Map<number, CausalLink[]>();
  for (const link of causalLinks) {
    if (!filters[link.from_category] || !filters[link.to_category]) continue;
    const arr = linksByToTurn.get(link.to_turn) ?? [];
    arr.push(link);
    linksByToTurn.set(link.to_turn, arr);
  }

  return (
    <div className="transcript-wrap" ref={wrapRef}>
      <SpectreBar segments={segments} totalHeight={totalHeight} />
      <div className="transcript-column">
        {turns.map((turn) => {
          const chips =
            turn.role === "assistant"
              ? linksByToTurn.get(turn.index) ?? []
              : [];
          return (
            <div
              key={turn.index}
              className="transcript-turn"
              ref={(el) => {
                if (el) turnRefs.current.set(turn.index, el);
                else turnRefs.current.delete(turn.index);
              }}
            >
              <TurnBubble turn={turn} filters={filters} modelName={modelName} />
              {chips.map((link, i) => (
                <CausalLinkChip key={i} link={link} onNavigate={navigateTo} />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
