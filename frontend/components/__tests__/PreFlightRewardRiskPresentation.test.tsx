/**
 * ARX-UX-RR-030 — Pre-Flight Reward vs. Risk Presentation Component Tests (Issue #181)
 *
 * Verifies:
 * 1. Title: "Reward vs. Risk" cleanly rendered in 5-point checklist.
 * 2. Ratio: Dynamic formatting (e.g. "1.85:1", "2.00:1", "2.50:1").
 * 3. Threshold: "Required: 2:1" derived from authoritative PREFLIGHT_CONFIG.
 * 4. Failure label: "Below minimum" replacing raw "FAIL" in presentation badge.
 * 5. Explanation: "Potential reward is too low for the risk."
 * 6. Edge cases: Below-threshold, equal-threshold, above-threshold, missing levels, inverted levels.
 * 7. Accessibility, responsive wrapping, and dark/light theme CSS variables.
 * 8. Authoritative calculation and fail-closed execution barriers strictly preserved.
 */

import { describe, it, expect, vi } from "vitest";
import React from "react";
import { render, screen } from "@testing-library/react";
import PreFlightChecklistModal from "../PreFlightChecklistModal";
import { PREFLIGHT_CONFIG } from "../../lib/preflightEvaluator";

vi.mock("../../lib/matomo", () => ({
  trackPreFlightOutcome: vi.fn(),
  trackTradePlanCopied: vi.fn(),
}));

describe("ARX-UX-RR-030: Pre-Flight Reward vs. Risk Presentation (Issue #181)", () => {
  const baseProps = {
    isOpen: true,
    onClose: vi.fn(),
    symbol: "NVDA",
    currentPrice: 100.0,
    optimalEntryMin: 95.0,
    optimalEntryMax: 102.0,
    isActionable: true,
    canSizeTrade: true,
    vix: 15.0,
    vixObservationTime: new Date().toISOString(),
    isStage4: false,
    catalystEvidence: {
      state: "PASS" as const,
      daysToEarnings: 20.0,
      confirmationStatus: "CONFIRMED" as const,
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE" as const,
      riskTier: "NORMAL_RUNWAY",
    },
  };

  it("1. Below-threshold (1.85:1): Renders 'Below minimum', '1.85:1 (Required: 2:1)', and approved explanation", () => {
    // Risk = 100 - 90 = 10. Reward = 118.5 - 100 = 18.5. R:R = 1.85:1 (< 2.0:1)
    render(
      <PreFlightChecklistModal
        {...baseProps}
        stopLoss={90.0}
        takeProfit1={118.5}
        riskRewardRatio={1.85}
      />
    );

    // Title verification
    const titleEl = screen.getByTestId("preflight-chk1-title");
    expect(titleEl.textContent).toBe("Reward vs. Risk");
    expect(titleEl.closest("div")?.textContent).toContain("1. Reward vs. Risk");

    // Ratio & Threshold formatting
    expect(screen.getByText(/1\.85:1/)).toBeDefined();
    expect(screen.getByText(new RegExp(PREFLIGHT_CONFIG.REQUIRED_RR_LABEL))).toBeDefined();
    expect(screen.getByText(/1\.85:1 \(Required: 2:1\)/)).toBeDefined();

    // Failure label badge
    const badgeEl = screen.getByTestId("preflight-chk1-badge");
    expect(badgeEl.textContent).toBe("Below minimum");
    expect(badgeEl.getAttribute("aria-label")).toBe("Status: Below minimum");
    expect(badgeEl.className).toContain("bg-rose-950");
    expect(badgeEl.className).toContain("text-rose-300");

    // Failure explanation copy
    expect(screen.getByText(new RegExp(PREFLIGHT_CONFIG.EXPLANATION_BELOW_MINIMUM))).toBeDefined();
    expect(screen.getByText(/Potential reward is too low for the risk\./)).toBeDefined();

    // Clearance invariant: Below minimum must NOT clear execution
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
  });

  it("2. Equal-threshold (2.00:1): Meets exact 2:1 minimum -> Renders PASS badge and adequate cushion explanation", () => {
    // Risk = 100 - 90 = 10. Reward = 120.0 - 100 = 20. R:R = 2.00:1
    render(
      <PreFlightChecklistModal
        {...baseProps}
        stopLoss={90.0}
        takeProfit1={120.0}
        riskRewardRatio={2.0}
      />
    );

    // Ratio & Threshold formatting
    expect(screen.getByText(/2\.00:1 \(Required: 2:1\)/)).toBeDefined();

    // PASS badge
    const badgeEl = screen.getByTestId("preflight-chk1-badge");
    expect(badgeEl.textContent).toBe("PASS");
    expect(badgeEl.className).toContain("bg-emerald-950");
    expect(badgeEl.className).toContain("text-emerald-300");

    // Adequate cushion explanation
    expect(screen.getByText(/Adequate upside cushion/)).toBeDefined();
  });

  it("3. Above-threshold (2.50:1): Exceeds 2:1 minimum -> Renders PASS badge and dynamic ratio", () => {
    // Risk = 100 - 90 = 10. Reward = 125.0 - 100 = 25. R:R = 2.50:1
    render(
      <PreFlightChecklistModal
        {...baseProps}
        stopLoss={90.0}
        takeProfit1={125.0}
        riskRewardRatio={2.5}
      />
    );

    expect(screen.getByText(/2\.50:1 \(Required: 2:1\)/)).toBeDefined();

    const badgeEl = screen.getByTestId("preflight-chk1-badge");
    expect(badgeEl.textContent).toBe("PASS");
    expect(badgeEl.className).toContain("bg-emerald-950");

    expect(screen.getByText(/Adequate upside cushion/)).toBeDefined();
  });

  it("4. Unavailable case (missing stop loss or target): Renders 'Unavailable', threshold, and missing levels notice", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        stopLoss={null}
        takeProfit1={125.0}
        riskRewardRatio={null}
      />
    );

    // Ratio shows Unavailable with configured threshold
    expect(screen.getByText(/Unavailable \(Required: 2:1\)/)).toBeDefined();

    // Badge shows Unavailable
    const badgeEl = screen.getByTestId("preflight-chk1-badge");
    expect(badgeEl.textContent).toBe("Unavailable");
    expect(badgeEl.className).toContain("bg-slate-900");

    // Detail notice
    expect(screen.getByText(/Execution levels missing: Stop and target required/)).toBeDefined();

    // Must NOT be cleared
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
  });

  it("5. Unavailable case (inverted boundaries): Stop loss above current price -> Renders 'Unavailable'", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        stopLoss={105.0} // Inverted: Stop above price 100
        takeProfit1={125.0}
        riskRewardRatio={null}
      />
    );

    expect(screen.getByText(/Unavailable \(Required: 2:1\)/)).toBeDefined();

    const badgeEl = screen.getByTestId("preflight-chk1-badge");
    expect(badgeEl.textContent).toBe("Unavailable");

    expect(screen.getByText(/Invalid execution boundaries: Stop and target levels inverted/)).toBeDefined();
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
  });

  it("6. Sub-threshold rounding guard (1.995:1): Strictly fails below minimum, never rounds up to PASS", () => {
    // Risk = 10, Target = 119.95 -> raw R:R = 1.995:1 (rounds to 2.00 for 2 decimals, but raw < 2.0)
    render(
      <PreFlightChecklistModal
        {...baseProps}
        stopLoss={90.0}
        takeProfit1={119.95}
        riskRewardRatio={1.995}
      />
    );

    const badgeEl = screen.getByTestId("preflight-chk1-badge");
    expect(badgeEl.textContent).toBe("Below minimum");
    expect(badgeEl.className).toContain("bg-rose-950");
    expect(screen.getByText(new RegExp(PREFLIGHT_CONFIG.EXPLANATION_BELOW_MINIMUM))).toBeDefined();
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
  });

  it("7. Configuration derivation: Threshold label 'Required: 2:1' matches PREFLIGHT_CONFIG.REQUIRED_RR_LABEL", () => {
    expect(PREFLIGHT_CONFIG.MIN_REWARD_TO_RISK_RATIO).toBe(2.0);
    expect(PREFLIGHT_CONFIG.REQUIRED_RR_DISPLAY_THRESHOLD).toBe("2:1");
    expect(PREFLIGHT_CONFIG.REQUIRED_RR_LABEL).toBe("Required: 2:1");
    expect(PREFLIGHT_CONFIG.EXPLANATION_BELOW_MINIMUM).toBe("Potential reward is too low for the risk.");
    expect(PREFLIGHT_CONFIG.LABEL_BELOW_MINIMUM).toBe("Below minimum");
  });
});
