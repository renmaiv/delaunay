import { useId, type ReactNode } from "react";

interface InfoTooltipProps {
  text: string;
  children?: ReactNode;
}

/**
 * Pure-CSS hover/focus tooltip (no library). When `children` are given they act
 * as the hover target; otherwise an ⓘ glyph is rendered. The tooltip text is
 * always present in the DOM (hidden via CSS, not conditional render) so it can
 * be asserted in tests. The target carries `aria-describedby` referencing the
 * `role="tooltip"` element; the tooltip is revealed on `:hover`/`:focus-within`.
 */
export default function InfoTooltip({ text, children }: InfoTooltipProps) {
  const id = useId();
  return (
    <span className="info-tooltip">
      <span
        className="info-tooltip__target"
        tabIndex={0}
        aria-describedby={id}
        role={children ? undefined : "img"}
        aria-label={children ? undefined : "More information"}
      >
        {children ?? <span className="info-tooltip__glyph" aria-hidden="true">ⓘ</span>}
      </span>
      <span id={id} role="tooltip" className="info-tooltip__bubble">
        {text}
      </span>
    </span>
  );
}
