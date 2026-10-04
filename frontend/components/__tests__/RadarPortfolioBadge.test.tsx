import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RadarPortfolioBadge } from "../radar/RadarPortfolioBadge";

describe("RadarPortfolioBadge (Accessible Portfolio Ownership Indicator)", () => {
  it("renders explicit text 'HELD' and screen-reader aria-label when HELD", () => {
    render(<RadarPortfolioBadge ownershipState="HELD" shares={15} />);

    // Non-color-only requirement: explicit textual string "HELD"
    const badge = screen.getByRole("status");
    expect(badge).toBeDefined();
    expect(badge.textContent).toContain("HELD");
    expect(badge.textContent).toContain("(15 sh)");

    // WCAG AA screen reader label
    expect(badge.getAttribute("aria-label")).toBe(
      "Position status: Held in portfolio, 15 shares"
    );
  });

  it("renders null for UNKNOWN state to preserve clean presentation", () => {
    const { container } = render(<RadarPortfolioBadge ownershipState="UNKNOWN" />);
    expect(container.firstChild).toBeNull();
  });

  it("renders null for default NOT_HELD state", () => {
    const { container } = render(
      <RadarPortfolioBadge ownershipState="NOT_HELD" showWhenNotHeld={false} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders explicit 'NEW' badge when showWhenNotHeld is true", () => {
    render(<RadarPortfolioBadge ownershipState="NOT_HELD" showWhenNotHeld={true} />);

    const badge = screen.getByRole("status");
    expect(badge).toBeDefined();
    expect(badge.textContent).toContain("NEW");
    expect(badge.getAttribute("aria-label")).toBe(
      "Position status: Not currently held in portfolio"
    );
  });
});
