interface ExampleBadgeProps {
  onClear: () => void;
}

/** Banner shown while the pre-loaded example analysis is on screen. */
export default function ExampleBadge({ onClear }: ExampleBadgeProps) {
  return (
    <div className="example-badge" role="note">
      <span className="example-badge__dot" aria-hidden="true" />
      <span className="example-badge__text">
        <strong>Example analysis.</strong> This is a sample conversation, already
        evaluated. Upload your own chat to analyze it.
      </span>
      <button type="button" className="example-badge__clear" onClick={onClear}>
        Clear
      </button>
    </div>
  );
}
