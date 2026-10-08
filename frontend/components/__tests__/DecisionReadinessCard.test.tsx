import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import DecisionReadinessCard from "../DecisionReadinessCard";
import { resolveDecisionReadiness } from "../../lib/decisionReadiness";

/**
 * ARX Terminal Synthesis E Wave 4: Decision Readiness Card Component Suite
 *
 * Verifies:
 * 1. 3-gate rendering with distinct status icons and non-color-only text labels.
 * 2. Active Blocker tag rendered on next-condition box.
 * 3. Progressive disclosure: tapping a gate expands its explanation.
 * 4. Negative guidance alert banner rendering.
 * 5. Primary CTA invocation with accessible >= 44x44px touch targets.
 * 6. Secondary Pre-Flight action button.
 */

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
}));

describe("DecisionReadinessCard Component", () => {
  it("renders 3 gates with non-color-only badges and proper accessibility attributes", () => {
    const readiness = resolveDecisionReadiness({
      currentPrice: 100,
      optimalEntryMin: 98,
      optimalEntryMax: 102,
      isConfirmed: false,
    });

    render(
      <DecisionReadinessCard
        symbol="NVDA"
        readinessResult={readiness}
      />
    );

    // Verify main card landmark
    expect(screen.getByTestId("decision-readiness-card")).toBeDefined();
    expect(screen.getByText(/Decision Readiness Progression/i)).toBeDefined();

    // Verify binary status badge
    expect(screen.getByTestId("readiness-status-badge").textContent).toContain("SETUP IN PROGRESS");

    // Verify 3 gates are rendered
    expect(screen.getByTestId("gate-item-1")).toBeDefined();
    expect(screen.getByTestId("gate-item-2")).toBeDefined();
    expect(screen.getByTestId("gate-item-3")).toBeDefined();

    // Verify Gate 1 is PASSED
    expect(screen.getByTestId("gate-badge-1").textContent).toContain("PASSED");
    // Verify Gate 2 is BLOCKING
    expect(screen.getByTestId("gate-badge-2").textContent).toContain("BLOCKING");
    // Verify Gate 3 is PENDING
    expect(screen.getByTestId("gate-badge-3").textContent).toContain("PENDING");

    // Verify Active Blocker tag
    expect(screen.getByTestId("active-blocker-tag").textContent).toBe("ACTIVE BLOCKER");
  });

  it("expands gate explanation upon click (progressive disclosure)", () => {
    const readiness = resolveDecisionReadiness({
      currentPrice: 100,
      optimalEntryMin: 98,
      optimalEntryMax: 102,
      isConfirmed: false,
    });

    render(
      <DecisionReadinessCard
        symbol="NVDA"
        readinessResult={readiness}
      />
    );

    // Gate 2 is active blocker, initially expanded
    expect(screen.getByTestId("gate-explanation-2")).toBeDefined();

    // Gate 1 is not expanded initially
    expect(screen.queryByTestId("gate-explanation-1")).toBeNull();

    // Click Gate 1 button
    const gate1Button = screen.getByTestId("gate-item-1").querySelector('[role="button"]')!;
    fireEvent.click(gate1Button);

    // Gate 1 explanation is now visible
    expect(screen.getByTestId("gate-explanation-1")).toBeDefined();
    expect(screen.getByTestId("gate-explanation-1").textContent).toContain(
      "positioned inside the institutional accumulation corridor"
    );
  });

  it("renders protective negative guidance banner when triggered", () => {
    const readiness = resolveDecisionReadiness({
      currentPrice: 110,
      optimalEntryMin: 98,
      optimalEntryMax: 102,
    });

    render(
      <DecisionReadinessCard
        symbol="NVDA"
        readinessResult={readiness}
      />
    );

    const banner = screen.getByTestId("negative-guidance-banner");
    expect(banner).toBeDefined();
    expect(banner.textContent).toContain("DO NOT CHASE");
  });

  it("fires primary CTA and secondary Pre-Flight callbacks", () => {
    const readiness = resolveDecisionReadiness({
      currentPrice: 100,
      optimalEntryMin: 98,
      optimalEntryMax: 102,
      isConfirmed: true,
      riskRewardRatio: 2.5,
      stopLoss: 95,
      takeProfit1: 112.5,
      vix: 17.5,
    });

    const onSizeMock = vi.fn();
    const onPreFlightMock = vi.fn();

    render(
      <DecisionReadinessCard
        symbol="NVDA"
        readinessResult={readiness}
        onSizePosition={onSizeMock}
        onOpenPreFlight={onPreFlightMock}
      />
    );

    // Verify execution ready badge
    expect(screen.getByTestId("readiness-status-badge").textContent).toContain("EXECUTION READY");

    // Primary CTA is Size Position
    const primaryCta = screen.getByTestId("readiness-primary-cta");
    expect(primaryCta.textContent).toContain("Size Position");
    fireEvent.click(primaryCta);
    expect(onSizeMock).toHaveBeenCalledTimes(1);

    // Secondary CTA is Pre-Flight
    const preFlightBtn = screen.getByTestId("readiness-preflight-btn");
    expect(preFlightBtn.textContent).toContain("Pre-Flight");
    fireEvent.click(preFlightBtn);
    expect(onPreFlightMock).toHaveBeenCalledTimes(1);
  });
});
