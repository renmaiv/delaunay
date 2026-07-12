import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { AnalysisResult } from "./types";

vi.mock("./api", () => ({
  analyzeFile: vi.fn(),
}));

import App from "./App";
import { analyzeFile } from "./api";

beforeAll(() => {
  if (typeof ResizeObserver === "undefined") {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
  // scrollIntoView is not implemented in jsdom
  window.HTMLElement.prototype.scrollIntoView = () => {};
});

const fixture: AnalysisResult = {
  conversation_id: "c1",
  model_name: "chatgpt 5.0",
  summary: "A tense exchange about restricted content.",
  overall_sentiment: -0.3,
  turns: [{ index: 0, role: "user", content: "hi", detections: [] }],
  causal_links: [],
  meta: { judge_provider: "mock", encoders_available: {}, warnings: [] },
};

function selectFile() {
  const input = document.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement;
  const file = new File(["{}"], "chat.json", { type: "application/json" });
  fireEvent.change(input, { target: { files: [file] } });
}

describe("App", () => {
  beforeEach(() => {
    vi.mocked(analyzeFile).mockReset();
  });

  it("opens with the pre-evaluated example and the Upload CTA", () => {
    render(<App />);
    expect(screen.getByRole("button", { name: "Upload Chat" })).toBeTruthy();
    expect(
      screen.getByText(/recommended to upload the chain of thought/i),
    ).toBeTruthy();
    // example badge + example content are shown on load
    expect(screen.getByText(/Example analysis\./i)).toBeTruthy();
    expect(screen.getByTestId("transcript-slot")).toBeTruthy();
  });

  it("Clear dismisses the example to an idle state, and it can be restored", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    // idle: example gone, restore affordance shown
    expect(screen.queryByText(/Example analysis\./i)).toBeNull();
    const restore = screen.getByRole("button", { name: /view the example again/i });
    fireEvent.click(restore);
    expect(screen.getByText(/Example analysis\./i)).toBeTruthy();
  });

  it("transitions example → analyzing → done and shows the summary", async () => {
    let resolveFn!: (r: AnalysisResult) => void;
    vi.mocked(analyzeFile).mockImplementation(
      () => new Promise<AnalysisResult>((res) => (resolveFn = res)),
    );
    render(<App />);
    selectFile();

    // analyzing: spinner (role=status) is shown while the promise is pending
    await screen.findByRole("status");

    resolveFn(fixture);
    await screen.findByText(fixture.summary);
    // done phase renders the slot placeholders
    expect(screen.getByTestId("filters-slot")).toBeTruthy();
    expect(screen.getByTestId("transcript-slot")).toBeTruthy();
  });

  it("shows the error panel and resets to idle on Try again", async () => {
    vi.mocked(analyzeFile).mockRejectedValue(new Error("bad file"));
    render(<App />);
    selectFile();

    await screen.findByText("bad file");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(screen.getByRole("button", { name: "Upload Chat" })).toBeTruthy();
  });
});
