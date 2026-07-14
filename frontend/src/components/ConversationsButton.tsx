interface ConversationsButtonProps {
  onClick: () => void;
}

/** Button sitting under the user/model filter panel that opens the
 *  Conversations grid. The leading "←" is an inline SVG arrow. */
export default function ConversationsButton({ onClick }: ConversationsButtonProps) {
  return (
    <button
      type="button"
      className="conversations-btn"
      onClick={onClick}
    >
      <svg
        className="conversations-btn__arrow"
        width="16"
        height="16"
        viewBox="0 0 16 16"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="M10 3L5 8l5 5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span>Conversations</span>
    </button>
  );
}
