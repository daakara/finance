import { describe, it, expect, vi } from "vitest";
import React from "react";
import { render, screen } from "@testing-library/react";
import PreFlightChecklistModal from "../PreFlightChecklistModal";

// Mock analytics to avoid network calls
vi.mock("../../lib/telemetry/analytics", () => ({
  trackPreFlightOutcome: vi.fn(),
  trackTradePlanCopied: vi.fn(),
}));

describe("PreFlightChecklistModal — Catalyst & Corporate Events Fail-Closed State Model", () => {
  const baseProps = {
    isOpen: true,
    onClose: vi.fn(),
    symbol: "AXP",
    currentPrice: 270.0,
    stopLoss: 250.0,
    takeProfit1: 310.0,
    riskRewardRatio: 2.0,
    setupPattern: "Minervini VCP",
    isActionable: true,
    vix: 15.0,
  };

  it("1. Missing catalyst evidence fails closed to UNAVAILABLE (never PASS)", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        hasImminentEarnings={null}
        catalystEvidence={null}
      />
    );

    // Badges must include UNAVAILABLE, never fake PASS
    const unavailableBadges = screen.getAllByText("UNAVAILABLE");
    expect(unavailableBadges.length).toBeGreaterThan(0);

    // Copy must state event runway uncertified
    expect(
      screen.getByText(/Earnings calendar coverage unavailable: Event runway cannot be certified/i)
    ).toBeDefined();
  });

  it("2. Active imminent earnings within 24h renders HAZARD and fails Check 4", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystState="WARN"
        catalystEvidence={{
          state: "WARN",
          daysToEarnings: 0.25,
          eventDate: "2026-10-23",
          callTimeEt: "08:30",
          confirmationStatus: "CONFIRMED",
        }}
      />
    );

    const hazardBadge = screen.getByText("HAZARD");
    expect(hazardBadge).toBeDefined();
    expect(screen.getByText(/Scheduled earnings announcement/i)).toBeDefined();
  });

  it("3. Verified clean runway (>7 days) renders PASS", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystState="PASS"
        catalystEvidence={{
          state: "PASS",
          daysToEarnings: 14.5,
          eventDate: "2026-11-06",
          confirmationStatus: "CONFIRMED",
          coverageStatus: "CONFIRMED_UPCOMING",
          freshnessStatus: "LIVE",
          riskTier: "NORMAL_RUNWAY",
        }}
      />
    );

    const passBadges = screen.getAllByText("PASS");
    expect(passBadges.length).toBeGreaterThan(0);
    expect(
      screen.getByText(/Verified event runway: No earnings announcement scheduled within the 7-day trade window/i)
    ).toBeDefined();
  });

  it("4. Elapsed prior earnings alone (negative DTE) evaluates to UNAVAILABLE (Invariant 3 & 4: cannot certify forward runway)", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystEvidence={{
          daysToEarnings: -2.0,
          eventDate: "2026-10-21",
          confirmationStatus: "CONFIRMED",
        }}
      />
    );

    const unavailableBadges = screen.getAllByText("UNAVAILABLE");
    expect(unavailableBadges.length).toBeGreaterThan(0);
    expect(
      screen.getByText(/Prior earnings reported \(2026-10-21\), but forward calendar is unverified\. Cannot certify forward runway\./i)
    ).toBeDefined();
  });

  it("5. Technical error explicitly renders ERROR badge", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystState="ERROR"
      />
    );

    expect(screen.getByText("ERROR")).toBeDefined();
    expect(screen.getByText(/Technical error retrieving corporate earnings schedule/i)).toBeDefined();
  });
});
