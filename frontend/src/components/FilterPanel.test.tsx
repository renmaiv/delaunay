import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import FilterPanel from "./FilterPanel";
import Tabs from "./Tabs";
import { TAXONOMY } from "../taxonomy";
import { USER_CATEGORIES, MODEL_CATEGORIES } from "../types";
import type { DetectionCategory } from "../types";

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

describe("FilterPanel", () => {
  it("model side renders 3 checkboxes without hasCot, 4 with hasCot", () => {
    const { rerender } = render(
      <FilterPanel side="model" filters={allOn()} onChange={() => {}} />,
    );
    expect(screen.getAllByRole("checkbox")).toHaveLength(3);
    rerender(
      <FilterPanel side="model" filters={allOn()} onChange={() => {}} hasCot />,
    );
    expect(screen.getAllByRole("checkbox")).toHaveLength(4);
  });

  it("user side renders 4 checkboxes", () => {
    render(<FilterPanel side="user" filters={allOn()} onChange={() => {}} />);
    expect(screen.getAllByRole("checkbox")).toHaveLength(4);
  });

  it("uses label texts straight from TAXONOMY", () => {
    render(<FilterPanel side="user" filters={allOn()} onChange={() => {}} />);
    for (const c of USER_CATEGORIES) {
      expect(screen.getByText(TAXONOMY.categories[c].label)).toBeTruthy();
    }
  });

  it("renders the Appeasement tooltip text from taxonomy in the DOM", () => {
    render(<FilterPanel side="model" filters={allOn()} onChange={() => {}} />);
    expect(
      screen.getByText(TAXONOMY.categories.appeasement.tooltip),
    ).toBeTruthy();
  });

  it("fires onChange with the right category and value on click", () => {
    const onChange = vi.fn();
    render(<FilterPanel side="model" filters={allOn()} onChange={onChange} />);
    const first = screen.getAllByRole("checkbox")[0] as HTMLInputElement;
    fireEvent.click(first);
    expect(onChange).toHaveBeenCalledWith(MODEL_CATEGORIES[0], false);
  });
});

describe("Tabs", () => {
  it("renders `Model: chatgpt 5.0` when a model name is given", () => {
    render(<Tabs active="model" onChange={() => {}} modelName="chatgpt 5.0" />);
    expect(screen.getByRole("tab", { name: "Model: chatgpt 5.0" })).toBeTruthy();
  });

  it("renders plain `Model` when modelName is null", () => {
    render(<Tabs active="model" onChange={() => {}} modelName={null} />);
    expect(screen.getByRole("tab", { name: "Model" })).toBeTruthy();
  });
});
