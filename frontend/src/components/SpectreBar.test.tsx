import { describe, it, expect } from "vitest";
import { markersFromMeasurements, segmentsFromMeasurements } from "./SpectreBar";
import type { DetectionCategory, Detection, Turn } from "../types";

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

function turn(index: number, dets: Detection[]): Turn {
  return { index, role: "user", content: `t${index}`, detections: dets };
}

describe("segmentsFromMeasurements", () => {
  const turns = [
    turn(0, [det("jailbreak_steering", 0.9)]),
    turn(1, [det("social_engineering", 0.4)]),
    turn(2, [det("coercive_pressure", 0.05)]),
  ];
  const measurements = new Map([
    [0, { top: 0, height: 100 }],
    [1, { top: 100, height: 100 }],
    [2, { top: 200, height: 100 }],
  ]);

  it("produces one segment per measured turn with score/category from maxVisibleScore", () => {
    const segs = segmentsFromMeasurements(turns, measurements, allOn());
    expect(segs).toHaveLength(3);
    expect(segs[0]).toMatchObject({
      turnIndex: 0,
      top: 0,
      height: 100,
      score: 0.9,
      category: "jailbreak_steering",
    });
    expect(segs[1]).toMatchObject({ score: 0.4, category: "social_engineering" });
    // 0.05 is below display_threshold → not visible
    expect(segs[2]).toMatchObject({ score: 0, category: null });
  });

  it("zeroes a segment when its category's filter is turned off", () => {
    const filters = allOn();
    filters.jailbreak_steering = false;
    const segs = segmentsFromMeasurements(turns, measurements, filters);
    expect(segs[0]).toMatchObject({ turnIndex: 0, score: 0, category: null });
    // other turns unaffected
    expect(segs[1]).toMatchObject({ score: 0.4, category: "social_engineering" });
  });

  it("skips turns without a measurement", () => {
    const partial = new Map([[0, { top: 0, height: 50 }]]);
    const segs = segmentsFromMeasurements(turns, partial, allOn());
    expect(segs).toHaveLength(1);
    expect(segs[0].turnIndex).toBe(0);
  });
});

describe("markersFromMeasurements", () => {
  const measurements = new Map([
    [0, { top: 0, height: 100 }],
    [1, { top: 100, height: 100 }],
    [2, { top: 200, height: 100 }],
  ]);

  it("produces one marker PER VISIBLE detection (not just the max)", () => {
    const turns = [
      turn(0, [det("jailbreak_steering", 0.9), det("social_engineering", 0.4)]),
      turn(1, [det("coercive_pressure", 0.05)]), // below threshold -> no marker
      turn(2, []),
    ];
    const markers = markersFromMeasurements(turns, measurements, allOn());
    expect(markers).toHaveLength(2);
    expect(markers[0]).toMatchObject({
      turnIndex: 0,
      top: 0,
      category: "jailbreak_steering",
      score: 0.9,
      stackIndex: 0,
    });
    expect(markers[1]).toMatchObject({
      turnIndex: 0,
      top: 0,
      category: "social_engineering",
      score: 0.4,
      stackIndex: 1,
    });
  });

  it("respects filters independently of the max-score segment", () => {
    const turns = [turn(0, [det("jailbreak_steering", 0.9), det("social_engineering", 0.4)])];
    const filters = allOn();
    filters.jailbreak_steering = false;
    const markers = markersFromMeasurements(turns, measurements, filters);
    expect(markers).toHaveLength(1);
    expect(markers[0].category).toBe("social_engineering");
  });

  it("returns an empty list when a turn has no visible detections", () => {
    const turns = [turn(0, [])];
    expect(markersFromMeasurements(turns, measurements, allOn())).toEqual([]);
  });

  it("skips turns without a measurement", () => {
    const turns = [turn(5, [det("repair_request", 0.7)])];
    expect(markersFromMeasurements(turns, measurements, allOn())).toEqual([]);
  });
});
