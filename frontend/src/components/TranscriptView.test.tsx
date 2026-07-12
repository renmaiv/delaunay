import { describe, it, expect, beforeAll } from "vitest";
import { render, screen } from "@testing-library/react";
import TranscriptView from "./TranscriptView";
import type {
  AnalysisResult,
  DetectionCategory,
  Detection,
  Turn,
} from "../types";

beforeAll(() => {
  // jsdom lacks ResizeObserver; provide a no-op so the effect can run.
  if (typeof ResizeObserver === "undefined") {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
});

const allOn = (): Record<DetectionCategory, boolean> => ({
  jailbreak_steering: true,
  social_engineering: true,
  coercive_pressure: true,
  repair_request: true,
  safety_triggered: true,
  appeasement: true,
  overcompliant: true,
  cot_divergence: true,
});

function det(category: DetectionCategory, score: number): Detection {
  return { category, score, source: "encoder", calibrated: false };
}

function turn(index: number, role: Turn["role"], extra: Partial<Turn> = {}): Turn {
  return { index, role, content: `turn ${index}`, detections: [], ...extra };
}

const fixture: AnalysisResult = {
  conversation_id: "c1",
  model_name: "chatgpt 5.0",
  summary: "s",
  overall_sentiment: 0,
  turns: [
    turn(0, "user"),
    turn(1, "user", { detections: [det("jailbreak_steering", 0.9)] }),
    turn(2, "assistant"),
    turn(3, "assistant", {
      detections: [det("appeasement", 0.5)],
      cot: "internally I disagreed",
    }),
    turn(4, "user"),
    turn(5, "assistant", { detections: [det("overcompliant", 0.4)] }),
  ],
  causal_links: [
    {
      from_turn: 1,
      to_turn: 3,
      from_category: "jailbreak_steering",
      to_category: "appeasement",
      score: 0.8,
    },
  ],
  meta: { judge_provider: "mock", encoders_available: {}, warnings: [] },
};

function renderView(filters: Record<DetectionCategory, boolean>) {
  return render(
    <TranscriptView
      turns={fixture.turns}
      causalLinks={fixture.causal_links}
      modelName={fixture.model_name ?? null}
      filters={filters}
    />,
  );
}

describe("TranscriptView", () => {
  it("renders captions for detections >= 0.1 whose filter is on", () => {
    renderView(allOn());
    expect(screen.getByText("jailbreak 0.9")).toBeTruthy();
    expect(screen.getByText("appeasement 0.5")).toBeTruthy();
    expect(screen.getByText("overcompliant 0.4")).toBeTruthy();
  });

  it("removes the caption and the splash when its filter is toggled off", () => {
    const { rerender, container } = renderView(allOn());
    expect(
      container.querySelectorAll('[data-category="jailbreak_steering"]').length,
    ).toBeGreaterThan(0);

    const off = allOn();
    off.jailbreak_steering = false;
    rerender(
      <TranscriptView
        turns={fixture.turns}
        causalLinks={fixture.causal_links}
        modelName={fixture.model_name ?? null}
        filters={off}
      />,
    );
    expect(screen.queryByText("jailbreak 0.9")).toBeNull();
    expect(
      container.querySelectorAll('[data-category="jailbreak_steering"]').length,
    ).toBe(0);
  });

  it("renders a CoT <details> only on the CoT turn", () => {
    const { container } = renderView(allOn());
    const details = container.querySelectorAll("details");
    expect(details).toHaveLength(1);
    expect(screen.getByText("Chain of thought")).toBeTruthy();
  });

  it("renders a causal chip naming the source and target categories", () => {
    renderView(allOn());
    const chip = screen.getByText(/caused by turn 1/);
    expect(chip.textContent).toContain("jailbreak");
    expect(chip.textContent).toContain("appeasement");
    expect(chip.textContent).toContain("0.8");
  });
});
