import { describe, it, expect } from "vitest";
import { scoreColor, maxVisibleScore, TAXONOMY } from "./taxonomy";
import type { DetectionCategory, Turn, Detection } from "./types";

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

function turn(detections: Detection[]): Turn {
  return { index: 0, role: "user", content: "x", detections };
}

describe("scoreColor", () => {
  it("maps scores to the exact band colors", () => {
    expect(scoreColor(0.0)).toBe("#00a000");
    expect(scoreColor(0.2)).toBe("#f2ea00");
    expect(scoreColor(0.5)).toBe("#ff8000");
    expect(scoreColor(0.9)).toBe("#ff3700");
  });

  it("treats the last band as max-inclusive", () => {
    expect(scoreColor(1.0)).toBe("#ff3700");
  });

  it("uses band boundaries as [min, max) except the last band", () => {
    expect(scoreColor(0.1)).toBe("#f2ea00");
    expect(scoreColor(0.35)).toBe("#ff8000");
    expect(scoreColor(0.75)).toBe("#ff3700");
  });
});

describe("maxVisibleScore", () => {
  it("returns the max eligible detection", () => {
    const t = turn([
      det("jailbreak_steering", 0.4),
      det("social_engineering", 0.9),
      det("coercive_pressure", 0.2),
    ]);
    expect(maxVisibleScore(t, allOn())).toEqual({
      score: 0.9,
      category: "social_engineering",
    });
  });

  it("respects the filter map", () => {
    const t = turn([
      det("jailbreak_steering", 0.4),
      det("social_engineering", 0.9),
    ]);
    const filters = allOn();
    filters.social_engineering = false;
    expect(maxVisibleScore(t, filters)).toEqual({
      score: 0.4,
      category: "jailbreak_steering",
    });
  });

  it("respects display_threshold and returns null when nothing qualifies", () => {
    // display_threshold is 0.1; a 0.05 detection is below it.
    expect(TAXONOMY.display_threshold).toBe(0.1);
    const t = turn([det("jailbreak_steering", 0.05)]);
    expect(maxVisibleScore(t, allOn())).toEqual({ score: 0, category: null });
  });

  it("returns null when the only high detection is filtered off", () => {
    const t = turn([det("appeasement", 0.8)]);
    const filters = allOn();
    filters.appeasement = false;
    expect(maxVisibleScore(t, filters)).toEqual({ score: 0, category: null });
  });
});
