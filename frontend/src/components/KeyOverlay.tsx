import { useState } from "react";

interface KeyOverlayProps {
  onSubmit: (key: string) => void;
}

/** Blurred overlay shown over the analysis when the server's Anthropic key is
 *  missing, invalid, or out of credit (meta.judge_error === "auth"). The key
 *  the user pastes lives in component/App state only for this session — it is
 *  sent once per analysis as a request header and never stored or logged. */
export default function KeyOverlay({ onSubmit }: KeyOverlayProps) {
  const [key, setKey] = useState("");

  return (
    <div
      className="key-overlay"
      role="dialog"
      aria-label="Bring your Anthropic key"
    >
      <div className="key-overlay__panel">
        <h2 className="key-overlay__title">Bring your Anthropic key</h2>
        <p className="key-overlay__text">
          Our analysis credit is exhausted right now. Paste your own Anthropic
          API key to run this analysis on your account. The key is used only
          for this request — we never store or log it.
        </p>
        <form
          className="key-overlay__form"
          onSubmit={(e) => {
            e.preventDefault();
            const trimmed = key.trim();
            if (trimmed) onSubmit(trimmed);
          }}
        >
          <input
            type="password"
            className="key-overlay__input"
            placeholder="sk-ant-…"
            autoComplete="off"
            spellCheck={false}
            value={key}
            onChange={(e) => setKey(e.target.value)}
            aria-label="Anthropic API key"
          />
          <button type="submit" className="btn btn--primary" disabled={!key.trim()}>
            Analyze with my key
          </button>
        </form>
      </div>
    </div>
  );
}
