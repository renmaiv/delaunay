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

  it("transitions example → analyzing → done and shows the result", async () => {
    let resolveFn!: (r: AnalysisResult) => void;
    vi.mocked(analyzeFile).mockImplementation(
      () => new Promise<AnalysisResult>((res) => (resolveFn = res)),
    );
    render(<App />);
    selectFile();

    // analyzing: spinner (role=status) is shown while the promise is pending
    await screen.findByRole("status");

    resolveFn(fixture);
    // done phase: model chip renders; the summary text is intentionally NOT
    // shown (we don't surface the contents of the user's chat).
    await screen.findByText(fixture.model_name!);
    expect(screen.queryByText(fixture.summary)).toBeNull();
    expect(screen.getByTestId("filters-slot")).toBeTruthy();
    expect(screen.getByTestId("transcript-slot")).toBeTruthy();
  });

  it("shows the BYOK overlay when the judge reports an auth error", async () => {
    const authResult: AnalysisResult = {
      ...fixture,
      meta: { ...fixture.meta, judge_error: "auth" },
    };
    vi.mocked(analyzeFile).mockResolvedValue(authResult);
    render(<App />);
    selectFile();

    await screen.findByRole("dialog", { name: /bring your anthropic key/i });

    // submitting a key re-runs the analysis with it
    vi.mocked(analyzeFile).mockResolvedValue(fixture);
    fireEvent.change(screen.getByLabelText("Anthropic API key"), {
      target: { value: "sk-ant-test" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: /analyze with my key/i }),
    );
    await screen.findByText(fixture.model_name!);
    const calls = vi.mocked(analyzeFile).mock.calls;
    expect(calls[calls.length - 1][2]).toBe("sk-ant-test");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("shows the error panel and resets to idle on Try again", async () => {
    vi.mocked(analyzeFile).mockRejectedValue(new Error("bad file"));
    render(<App />);
    selectFile();

    await screen.findByText("bad file");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(screen.getByRole("button", { name: "Upload Chat" })).toBeTruthy();
  });

  it("opens the selected pre-evaluated conversation from the grid", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /conversations/i }));

    fireEvent.click(
      screen.getByRole("button", { name: "Open Taxes across two states" }),
    );

    expect(screen.getByTestId("transcript-slot")).toBeTruthy();
    expect(screen.getByText(/worked remotely in two states/i)).toBeTruthy();
  });
});
