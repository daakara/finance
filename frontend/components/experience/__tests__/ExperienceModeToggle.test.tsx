import React from "react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ExperienceModeToggle from "../ExperienceModeToggle";
import { useExperienceStore } from "../../../state/experience-store";

describe("ExperienceModeToggle Accessibility & Behavior", () => {
  beforeEach(() => {
    localStorage.clear();
    useExperienceStore.setState({ mode: "STANDARD" });
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("provides accessible tablist name", () => {
    render(<ExperienceModeToggle />);
    const tablist = screen.getByRole("tablist", { name: /experience mode/i });
    expect(tablist).toBeDefined();
    expect(tablist.getAttribute("data-testid")).toBe("experience-mode-toggle");
  });

  it("marks active mode as selected with aria-selected='true'", () => {
    useExperienceStore.setState({ mode: "STANDARD" });
    render(<ExperienceModeToggle />);

    const standardTab = screen.getByTestId("mode-STANDARD");
    expect(standardTab.getAttribute("role")).toBe("tab");
    expect(standardTab.getAttribute("aria-selected")).toBe("true");
    expect(standardTab.getAttribute("tabIndex")).toBe("0");

    const guidedTab = screen.getByTestId("mode-GUIDED");
    expect(guidedTab.getAttribute("aria-selected")).toBe("false");
    expect(guidedTab.getAttribute("tabIndex")).toBe("-1");

    const quantTab = screen.getByTestId("mode-QUANT");
    expect(quantTab.getAttribute("aria-selected")).toBe("false");
    expect(quantTab.getAttribute("tabIndex")).toBe("-1");
  });

  it("switches selected tab when clicked", () => {
    render(<ExperienceModeToggle />);

    const guidedTab = screen.getByTestId("mode-GUIDED");
    fireEvent.click(guidedTab);

    expect(useExperienceStore.getState().mode).toBe("GUIDED");
    expect(guidedTab.getAttribute("aria-selected")).toBe("true");

    const quantTab = screen.getByTestId("mode-QUANT");
    fireEvent.click(quantTab);

    expect(useExperienceStore.getState().mode).toBe("QUANT");
    expect(quantTab.getAttribute("aria-selected")).toBe("true");
  });

  it("supports keyboard arrow navigation across tabs", () => {
    render(<ExperienceModeToggle />);

    const standardTab = screen.getByTestId("mode-STANDARD");
    standardTab.focus();

    // ArrowRight: Standard (index 1) -> Quant (index 2)
    fireEvent.keyDown(standardTab, { key: "ArrowRight" });
    expect(useExperienceStore.getState().mode).toBe("QUANT");

    const quantTab = screen.getByTestId("mode-QUANT");
    // ArrowRight from last item loops to Guided (index 0)
    fireEvent.keyDown(quantTab, { key: "ArrowRight" });
    expect(useExperienceStore.getState().mode).toBe("GUIDED");

    const guidedTab = screen.getByTestId("mode-GUIDED");
    // ArrowLeft from first item loops to Quant (index 2)
    fireEvent.keyDown(guidedTab, { key: "ArrowLeft" });
    expect(useExperienceStore.getState().mode).toBe("QUANT");
  });

  it("supports Home and End keyboard keys", () => {
    render(<ExperienceModeToggle />);

    const standardTab = screen.getByTestId("mode-STANDARD");
    standardTab.focus();

    // Home: moves to first (GUIDED)
    fireEvent.keyDown(standardTab, { key: "Home" });
    expect(useExperienceStore.getState().mode).toBe("GUIDED");

    // End: moves to last (QUANT)
    fireEvent.keyDown(standardTab, { key: "End" });
    expect(useExperienceStore.getState().mode).toBe("QUANT");
  });
});
