import type { CausalLink } from "../types";
import { TAXONOMY } from "../taxonomy";

interface CausalLinkChipProps {
  link: CausalLink;
  onNavigate: (fromTurn: number) => void;
}

/**
 * Chip under an assistant turn that is the `to_turn` of a causal link.
 * Clicking navigates to the `from_turn` bubble (scroll + flash), handled by
 * the parent via `onNavigate`.
 */
export default function CausalLinkChip({ link, onNavigate }: CausalLinkChipProps) {
  const fromShort = TAXONOMY.categories[link.from_category].short;
  const toShort = TAXONOMY.categories[link.to_category].short;
  return (
    <button
      type="button"
      className="causal-chip"
      title={link.rationale ?? ""}
      onClick={() => onNavigate(link.from_turn)}
    >
      ⟵ caused by turn {link.from_turn}: {fromShort} → {toShort} (
      {link.score.toFixed(1)})
    </button>
  );
}
