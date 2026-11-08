/** @vitest-environment jsdom */
import { describe, it, expect, vi, afterEach } from "vitest";
import React from "react";
import { render, screen, cleanup } from "@testing-library/react";
import PreFlightChecklistModal from "../PreFlightChecklistModal";

// Mock analytics to avoid network calls
vi.mock("../../lib/telemetry/analytics", () => ({
  trackPreFlightOutcome: vi.fn(),
  trackTradePlanCopied: vi.fn(),
}));

afterEach(() => {
  cleanup();
});

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
        catalystState={undefined}
        catalystEvidence={null}
      />
    );

    const unavailableBadges = screen.getAllByText("UNAVAILABLE");
    expect(unavailableBadges.length).toBeGreaterThan(0);
    expect(
      screen.getByText(/Earnings calendar coverage unavailable: Event runway cannot be certified/i)
    ).toBeDefined();
  });

  it("2. Active imminent earnings within 24h renders HAZARD and fails Check 4", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystState="WARN"
        hasImminentEarnings={true}
        catalystEvidence={{
          daysToEarnings: 0.8,
          eventDate: "2026-10-24",
          confirmationStatus: "CONFIRMED",
          coverageStatus: "CONFIRMED_UPCOMING",
          freshnessStatus: "LIVE",
          riskTier: "HAZARD_IMMINENT_EARNINGS",
        }}
      />
    );

    expect(screen.getByText("HAZARD")).toBeDefined();
    expect(
      screen.getByText(/Scheduled earnings announcement (?:in <= 24h|within [^)]+) \(2026-10-24\)/i)
    ).toBeDefined();
  });

  it("3. Verified clean runway (>7 days) renders PASS", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystEvidence={{
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
      screen.getByText(/Verified event runway: Confirmed earnings announcement scheduled beyond the 7-day trade window \(14.5d DTE\)/i)
    ).toBeDefined();
  });

  it("6. Negative fixture: Missing coverage or freshness fails closed to UNAVAILABLE (never PASS)", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystEvidence={{
          daysToEarnings: 14.5,
          eventDate: "2026-11-06",
          confirmationStatus: "CONFIRMED",
          // coverageStatus and freshnessStatus deliberately omitted
        }}
      />
    );

    const unavailableBadges = screen.getAllByText("UNAVAILABLE");
    expect(unavailableBadges.length).toBeGreaterThan(0);
    expect(
      screen.getByText(/Earnings calendar coverage unavailable: Event runway cannot be certified/i)
    ).toBeDefined();
  });

  it("7. Unverified source: Clean runway with UNVERIFIED confirmation never claims verified provenance", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystEvidence={{
          daysToEarnings: 14.5,
          eventDate: "2026-11-06",
          confirmationStatus: "UNVERIFIED",
          coverageStatus: "CONFIRMED_UPCOMING",
          freshnessStatus: "LIVE",
          riskTier: "NORMAL_RUNWAY",
        }}
      />
    );

    expect(
      screen.getByText(/Unconfirmed event runway: Estimated announcement scheduled beyond the 7-day trade window/i)
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
          coverageStatus: "ELAPSED_RECENT",
          freshnessStatus: "LIVE",
          riskTier: "UNVERIFIED_RISK",
        }}
      />
    );

    const unavailableBadges = screen.getAllByText("UNAVAILABLE");
    expect(unavailableBadges.length).toBeGreaterThan(0);
    expect(
      screen.getByText(/(?:Prior earnings reported|Historical earnings report elapsed) \(2026-10-21\)/i)
    ).toBeDefined();
  });

  it("5. Technical error explicitly renders ERROR badge", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        catalystState="ERROR"
        catalystEvidence={{
          state: "ERROR",
        }}
      />
    );

    expect(screen.getByText("ERROR")).toBeDefined();
  });

  it("8. DHLGY confirmed calendar: Forward date (2027-03-05) renders PASS with verified runway copy and Check 2 label", () => {
    render(
      <PreFlightChecklistModal
        {...baseProps}
        symbol="DHLGY"
        currentPrice={31.96}
        stopLoss={29.50}
        takeProfit1={37.00}
        riskRewardRatio={2.05}
        catalystState="PASS"
        catalystEvidence={{
          state: "PASS",
          daysToEarnings: 117.1,
          eventDate: "2027-03-05",
          callTimeEt: "04:00",
          resultsReleaseTimeEt: "01:00",
          sourceName: "DHL Group Investor Relations / Financial Calendar",
          confirmationStatus: "CONFIRMED",
          coverageStatus: "CONFIRMED_UPCOMING",
          freshnessStatus: "LIVE",
          riskTier: "NORMAL_RUNWAY",
          isEstimate: false,
        }}
      />
    );

    const passBadges = screen.getAllByText("PASS");
    expect(passBadges.length).toBeGreaterThan(0);
    expect(
      screen.getByText(/Verified event runway: Confirmed earnings announcement scheduled beyond the 7-day trade window \(117.1d DTE\)/i)
    ).toBeDefined();
    expect(
      screen.getByText(/2\. Trend & Moving Averages \(Stage Discipline & Buying Range\)/i)
    ).toBeDefined();
  });
});
