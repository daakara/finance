import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import StandardTerminalView from "../terminal/StandardTerminalView";
import PreFlightReadinessBar from "../PreFlightReadinessBar";
import SynthesisVerdictCard from "../synthesis/SynthesisVerdictCard";
import { generateQuantitativeInsight } from "../../lib/insightGenerator";

describe("Execution Authority & Preflight Clearance Precedence (Gate 2 Adversarial)", () => {
  const createActionableInsight = () => {
    const insight = generateQuantitativeInsight(
      "NVDA",
      "NVIDIA Corporation",
      292.00,
      2.45,
      94,
      2,
      "SWING",
      "NOT_OWNED",
      "USER_DECLARED",
      [],
      "live"
    );
    insight.terminalState = {
      ...insight.terminalState,
      posture: "ACQUIRE",
      decisionState: "ACTIONABLE_SETUP",
      isActionable: true,
      canSizeTrade: true,
      uiStateLabel: "Actionable Setup Confirmed",
    };
    insight.standard.keyLevels = {
      ...insight.standard.keyLevels,
      currentPrice: 292.00,
      entryMin: 290.79,
      entryMax: 296.61,
      breakoutPivot: 292.00,
      stopLoss: 268.00,
      target1: 340.00,
      profitRiskRatio: 2.0,
    };
    return insight;
  };

  it("1. Adversarial Case: Attractive Actionable Setup with Unavailable Catalyst Coverage blocks execution", () => {
    const insight = createActionableInsight();
    const onOpenSizer = vi.fn();

    // Render PreFlight with catalyst UNAVAILABLE
    const preflightSlot = (
      <PreFlightReadinessBar
        symbol="NVDA"
        currentPrice={292.00}
        stopLoss={268.00}
        takeProfit1={340.00}
        riskRewardRatio={2.0}
        setupPattern="Minervini VCP"
        isActionable={true}
        vix={16.5}
        isStage4={false}
        optimalEntryMin={290.79}
        optimalEntryMax={296.61}
        catalystState="UNAVAILABLE"
        catalystEvidence={null}
      />
    );

    render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={onOpenSizer}
        onOpenWhy={vi.fn()}
        preflightSlot={preflightSlot}
      />
    );

    // PreFlight must show NOT CLEARED (3/5) and CATALYST COVERAGE UNAVAILABLE blocker
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED (3/5)");
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toContain("CATALYST COVERAGE UNAVAILABLE");

    // Sizing cell must NOT claim Budget Utilization is Authorized
    expect(screen.queryByText(/Budget Utilization Authorized/i)).toBeNull();
    expect(screen.getByText(/Position Sizing Basis/i)).toBeDefined();
    expect(screen.getByText(/Informational Sizing/i)).toBeDefined();

    // CTA must NOT claim execution authorization
    expect(screen.queryByText(/Size & Execute/i)).toBeNull();
    const sizeBtn = screen.getByText(/Size.*Position/i);
    expect(sizeBtn).toBeDefined();

    // Invoking CTA only opens informational sizing model, does not submit trade
    fireEvent.click(sizeBtn);
    expect(onOpenSizer).toHaveBeenCalledTimes(1);
  });

  it("2. Adversarial Case: Macro VIX >= 26.0 blocks execution clearance even for actionable setup", () => {
    const preflightSlot = (
      <PreFlightReadinessBar
        symbol="NVDA"
        currentPrice={292.00}
        stopLoss={268.00}
        takeProfit1={340.00}
        riskRewardRatio={2.0}
        isActionable={true}
        vix={28.5} // Extreme volatility regime
        isStage4={false}
        optimalEntryMin={290.79}
        optimalEntryMax={296.61}
        catalystEvidence={{
          state: "PASS" as const,
          daysToEarnings: 18.0,
          eventDate: "2026-11-15",
          confirmationStatus: "CONFIRMED" as const,
          coverageStatus: "CONFIRMED_UPCOMING",
          freshnessStatus: "LIVE" as const,
          riskTier: "NORMAL_RUNWAY",
        }}
      />
    );

    render(preflightSlot);

    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED");
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toContain("MACRO VOLATILITY REGIME (VIX >= 26.0)");
  });

  it("3. Invariant: SynthesisVerdictCard clearly separates setup actionability from preflight clearance", () => {
    render(
      <SynthesisVerdictCard
        symbol="NVDA"
        companyName="NVIDIA Corporation"
        exchange="NASDAQ"
        currentPrice={292.00}
        changePct={2.45}
        verdictTitle="Actionable Setup Confirmed"
        verdictSummary="High momentum accumulation base with confirmed pivot reference."
        setupScore={94}
        isActionable={true}
        buyZone="$290.79 - $296.61"
        maxStop={268.00}
        profitTarget={340.00}
        profitRiskRatio={2.0}
        preflightCleared={false}
        preflightChecksPassed={3}
        preflightTotalChecks={5}
        activeBlockerText="CATALYST COVERAGE UNAVAILABLE"
      />
    );

    // Setup actionability
    expect(screen.getByTestId("actionability-badge").textContent).toBe("ACTIONABLE");

    // Preflight clearance state
    expect(screen.getByText(/NOT CLEARED \(3\/5\)/i)).toBeDefined();
    expect(screen.getByText(/Blocker: CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();

    // Price ladder must preserve whole currency values with whitespace-nowrap
    const buyZoneEl = screen.getByText("$290.79 - $296.61");
    expect(buyZoneEl).toBeDefined();
    expect(buyZoneEl.className).toContain("whitespace-nowrap");
  });
});
