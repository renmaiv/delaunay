import { useRef } from "react";
import InfoTooltip from "./InfoTooltip";

interface UploadCTAProps {
  onFile: (f: File) => void;
  disabled: boolean;
}

const TOOLTIP_TEXT =
  "Tip: it's recommended to upload the chain of thought if you have it — it enables CoT-divergence detection.";

export default function UploadCTA({ onFile, disabled }: UploadCTAProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <InfoTooltip text={TOOLTIP_TEXT}>
      <span className="upload-cta">
        <button
          type="button"
          className="btn btn--primary"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          Upload Chat
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".json,.jsonl,.txt"
          className="upload-cta__input"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            // reset so selecting the same file again re-triggers change
            e.target.value = "";
          }}
        />
      </span>
    </InfoTooltip>
  );
}
