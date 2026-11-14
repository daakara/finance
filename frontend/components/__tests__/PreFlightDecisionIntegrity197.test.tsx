import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import PreFlightChecklistModal from "../PreFlightChecklistModal";
import PreFlightReadinessBar from "../PreFlightReadinessBar";

vi.mock("../../lib/matomo", () => ({
  trackPreFlightOutcome: vi.fn(),
  trackTradePlanCopied: vi.fn(),
}));

describe("Pre-Flight Decision Integrity & Epistemic Separation (Issue #197)", () => {
  const baseLiveProps = {
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    setupPattern: "Minervini VCP",
    isActionable: true,
    canSizeTrade: true,
    vix: 15.0,
    vixObservationTime: new Date().toISOString(),
    isStage4: false,
    optimalEntryMin: 215.0,
    optimalEntryMax: 222.0,
    catalystEvidence: {
      state: "PASS" as const,
      daysToEarnings: 15.0,
      eventDate: "2026-11-20",
      confirmationStatus: "CONFIRMED" as const,
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE" as const,
      riskTier: "NORMAL_RUNWAY",
    },
    dataSource: "live",
    liveSource: "ALPACA_IEX",
    liveFreshnessStatus: "LIVE",
  };

  it("1. Synthetic fixture modal displays SIMULATED CLEARANCE — NOT EXECUTABLE", () => {
    render(
      <PreFlightChecklistModal
        isOpen={true}
        onClose={vi.fn()}
        {...baseLiveProps}
        dataSource="synthetic"
        isSynthetic={true}
      />
    );

    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/SIMULATED CLEARANCE — NOT EXECUTABLE/i)).toBeDefined();
    expect(screen.getByText("5/5")).toBeDefined();
  });

  it("2. Historical EOD modal displays HISTORICAL REFERENCE — NOT EXECUTABLE", () => {
    render(
      <PreFlightChecklistModal
        isOpen={true}
        onClose={vi.fn()}
        {...baseLiveProps}
        dataSource="historical"
        isHistorical={true}
      />
    );

    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/HISTORICAL REFERENCE — NOT EXECUTABLE/i)).toBeDefined();
    expect(screen.getByText("5/5")).toBeDefined();
  });

  it("3. Stale price observation blocks execution clearance in modal", () => {
    render(
      <PreFlightChecklistModal
        isOpen={true}
        onClose={vi.fn()}
        {...baseLiveProps}
        liveFreshnessStatus="STALE"
      />
    );

    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — STALE PRICE OBSERVATION/i)).toBeDefined();
  });

  it("4. Historical tag renders in PreFlightReadinessBar when historical", () => {
    render(
      <PreFlightReadinessBar
        {...baseLiveProps}
        dataSource="historical"
        isHistorical={true}
      />
    );

    expect(screen.getByTestId("inline-historical-tag").textContent).toContain("HISTORICAL");
  });
});
