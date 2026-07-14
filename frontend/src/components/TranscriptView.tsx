import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CausalLink, DetectionCategory, Turn } from "../types";
import { TAXONOMY } from "../taxonomy";
import TurnBubble from "./TurnBubble";
import SpectreBar, {
  markersFromMeasurements,
  segmentsFromMeasurements,
  type SpectreMarker,
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
  const [markers, setMarkers] = useState<SpectreMarker[]>([]);
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
    setMarkers(markersFromMeasurements(turns, measurements, filters));
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

  // Turns carrying at least one visible detection ("captions") — the targets
  // the on-scroll navigation bar steps through.
  const flaggedTurns = turns
    .filter((t) =>
      t.detections.some(
        (d) => filters[d.category] && d.score >= TAXONOMY.display_threshold,
      ),
    )
    .map((t) => t.index);

  const [navVisible, setNavVisible] = useState(false);
  const [navPos, setNavPos] = useState(0);

  const handleScroll = () => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const st = wrap.scrollTop;
    setNavVisible(st > 24);
    let pos = 0;
    for (let i = 0; i < flaggedTurns.length; i++) {
      const el = turnRefs.current.get(flaggedTurns[i]);
      if (el && el.offsetTop - 8 <= st) pos = i;
    }
    setNavPos(pos);
  };

  const goToCaption = (pos: number) => {
    if (flaggedTurns.length === 0) return;
    const clamped = Math.max(0, Math.min(flaggedTurns.length - 1, pos));
    const el = turnRefs.current.get(flaggedTurns[clamped]);
    const wrap = wrapRef.current;
    if (el && wrap) wrap.scrollTo({ top: el.offsetTop - 8, behavior: "smooth" });
    setNavPos(clamped);
  };

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
    <div className="transcript-wrap" ref={wrapRef} onScroll={handleScroll}>
      <SpectreBar segments={segments} markers={markers} totalHeight={totalHeight} />
      {flaggedTurns.length > 0 && (
        <div
          className={`transcript-nav${navVisible ? " transcript-nav--visible" : ""}`}
        >
          <button
            type="button"
            className="transcript-nav__btn"
            aria-label="Previous detection"
            disabled={navPos <= 0}
            onClick={() => goToCaption(navPos - 1)}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M4 10l4-4 4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <span className="transcript-nav__counter">
            {navPos + 1} / {flaggedTurns.length}
          </span>
          <button
            type="button"
            className="transcript-nav__btn"
            aria-label="Next detection"
            disabled={navPos >= flaggedTurns.length - 1}
            onClick={() => goToCaption(navPos + 1)}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      )}
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
