import { describe, it, expect, vi } from "vitest";
import React from "react";
import { render, screen } from "@testing-library/react";
import PreFlightChecklistModal from "../PreFlightChecklistModal";

// Mock analytics to avoid external dependencies
vi.mock("../lib/matomo", () => ({
  trackPreFlightOutcome: vi.fn(),
  trackTradePlanCopied: vi.fn(),
}));

describe("Track A Pre-Flight 16 Mandatory Safety Scenarios (S01–S16)", () => {
  const basePristineProps = {
    isOpen: true,
    onClose: vi.fn(),
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    setupPattern: "Minervini VCP",
    isActionable: true,
    vix: 14.5,
    vixObservationTime: new Date().toISOString(),
    isStage4: false,
    optimalEntryMin: 215.0,
    optimalEntryMax: 222.0,
    catalystEvidence: {
      state: "PASS" as const,
      daysToEarnings: 18.0,
      eventDate: "2026-11-15",
      confirmationStatus: "CONFIRMED" as const,
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE" as const,
      riskTier: "NORMAL_RUNWAY",
    },
  };

  it("S01: Pristine Actionable, all mandatory verified PASS -> Eligible (isCleared = true)", () => {
    render(<PreFlightChecklistModal {...basePristineProps} />);
    expect(screen.getByText(/🟢 CLEARED TO EXECUTE/i)).toBeDefined();
    expect(screen.getByText(/100% Checklist/i)).toBeDefined();
    expect(screen.getByText("5/5")).toBeDefined();
  });

  it("S02: Four of five applicable checks PASS -> Blocked (Fixes legacy convictionPct >= 80% fail-open)", () => {
    // 4 checks pass, Check 4 (Catalyst) is UNAVAILABLE
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystEvidence={null}
        catalystState={undefined}
        hasImminentEarnings={null}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();
    expect(screen.getByText(/80% Checklist/i)).toBeDefined();
    expect(screen.getByText("4/5")).toBeDefined();
  });

  it("S03: Earnings within acute hazard window (<= 1 day) -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystState="WARN"
        catalystEvidence={{
          state: "WARN",
          daysToEarnings: 0.5,
          eventDate: "2026-10-25",
          confirmationStatus: "CONFIRMED",
          coverageStatus: "CONFIRMED_UPCOMING",
          freshnessStatus: "LIVE",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — IMMINENT EARNINGS HAZARD \(<= 7 DAYS\)/i)).toBeDefined();
  });

  it("S04: Earnings in protected caution window (4-7 days) -> Blocked under verified policy", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystEvidence={{
          state: "WARN",
          daysToEarnings: 5.0,
          eventDate: "2026-10-29",
          confirmationStatus: "CONFIRMED",
          coverageStatus: "CONFIRMED_UPCOMING",
          freshnessStatus: "LIVE",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — IMMINENT EARNINGS HAZARD \(<= 7 DAYS\)/i)).toBeDefined();
  });

  it("S05: Missing forward-event coverage -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystEvidence={{
          coverageStatus: "ABSENT",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();
  });

  it("S06: Stale forward-event coverage -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystEvidence={{
          state: "PASS",
          daysToEarnings: 15.0,
          coverageStatus: "CONFIRMED_UPCOMING",
          freshnessStatus: "STALE",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — STALE CATALYST COVERAGE/i)).toBeDefined();
  });

  it("S07: Provider timeout / Error -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystState="ERROR"
        catalystEvidence={{
          state: "ERROR",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST EVALUATION ERROR/i)).toBeDefined();
  });

  it("S08: Elapsed-only earnings record (negative DTE) -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystEvidence={{
          daysToEarnings: -3.0,
          eventDate: "2026-10-21",
          confirmationStatus: "CONFIRMED",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();
  });

  it("S09: Unknown coverage status -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystEvidence={{
          daysToEarnings: 20.0,
          coverageStatus: "UNRECOGNIZED_PROVIDER_STATUS",
          freshnessStatus: "LIVE",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();
  });

  it("S10: Contradictory event-risk fields -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        catalystState="PASS"
        hasImminentEarnings={true}
        catalystEvidence={{
          state: "WARN",
          daysToEarnings: 2.0,
          freshnessStatus: "LIVE",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — IMMINENT EARNINGS HAZARD \(<= 7 DAYS\)/i)).toBeDefined();
  });

  it("S11: Missing VIX -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        vix={undefined}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — VIX DATA UNAVAILABLE/i)).toBeDefined();
  });

  it("S12: Elevated VIX (>= 26.0) -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        vix={28.5}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — MACRO VOLATILITY REGIME \(VIX >= 26\.0\)/i)).toBeDefined();
  });

  it("S13: Incomplete capital-risk evidence -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        symbol="UNKNOWN_OTC"
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — CAPITAL RISK EVIDENCE INCOMPLETE/i)).toBeDefined();
  });

  it("S14: Valid but non-actionable setup -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        isActionable={false}
        decisionState="WAIT_FOR_TRIGGER"
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — CANONICAL SETUP NON-ACTIONABLE/i)).toBeDefined();
  });

  it("S15: Reward/risk below mandatory threshold (< 2.0) -> Blocked", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        riskRewardRatio={1.5}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — REWARD\/RISK ASYMMETRY \(< 2\.0:1\)/i)).toBeDefined();
  });

  it("S16: Unsupported benchmark ETF Pre-Flight -> No invented clearance", () => {
    render(
      <PreFlightChecklistModal
        {...basePristineProps}
        symbol="SPY"
        catalystEvidence={{
          coverageStatus: "ABSENT",
        }}
      />
    );
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED/i)).toBeDefined();
  });

  describe("Adversarial Positive Coverage & Hazard Precedence Matrix (G01–G16)", () => {
    it("G01: Verified clear coverage, fresh, nonhazardous -> PASS (Cleared)", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            state: "PASS",
            daysToEarnings: 14.0,
            eventDate: "2026-11-20",
            confirmationStatus: "CONFIRMED",
            coverageStatus: "CONFIRMED_UPCOMING",
            freshnessStatus: "LIVE",
            riskTier: "NORMAL_RUNWAY",
          }}
        />
      );
      expect(screen.getByText(/🟢 CLEARED TO EXECUTE/i)).toBeDefined();
      expect(screen.getByText("5/5")).toBeDefined();
    });

    it("G02: Future event, coverage status missing -> UNAVAILABLE (Blocked)", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            daysToEarnings: 14.0,
            freshnessStatus: "LIVE",
            coverageStatus: undefined as any,
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();
      expect(screen.getByText("4/5")).toBeDefined();
    });

    it("G03: Future event, freshness missing -> UNAVAILABLE (Blocked)", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            daysToEarnings: 14.0,
            coverageStatus: "CONFIRMED_UPCOMING",
            freshnessStatus: undefined as any,
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();
    });

    it("G04: Derived PASS but evidence missing -> Not PASS (Blocked)", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystState="PASS"
          catalystEvidence={undefined}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();
    });

    it("G05: Derived PASS but stale source -> Not PASS (Blocked)", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystState="PASS"
          catalystEvidence={{
            daysToEarnings: 14.0,
            coverageStatus: "CONFIRMED_UPCOMING",
            freshnessStatus: "STALE",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — STALE CATALYST COVERAGE/i)).toBeDefined();
    });

    it("G06: Source WARN with future earnings date -> WARN / Blocked (Source Hazard Precedence)", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            daysToEarnings: 15.0,
            coverageStatus: "CONFIRMED_UPCOMING",
            freshnessStatus: "LIVE",
            preflightState: "WARN",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — IMMINENT EARNINGS HAZARD \(<= 7 DAYS\)/i)).toBeDefined();
    });

    it("G07: Source ERROR with future earnings date -> ERROR / Blocked", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            daysToEarnings: 15.0,
            coverageStatus: "CONFIRMED_UPCOMING",
            freshnessStatus: "LIVE",
            preflightState: "ERROR",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST EVALUATION ERROR/i)).toBeDefined();
    });

    it("G08: Unknown coverage enum -> UNAVAILABLE (Blocked)", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            daysToEarnings: 14.0,
            coverageStatus: "UNKNOWN_ENUM_CODE" as any,
            freshnessStatus: "LIVE",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST COVERAGE UNAVAILABLE/i)).toBeDefined();
    });

    it("G09: Elapsed-only record -> UNAVAILABLE (Blocked)", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            daysToEarnings: -5.0,
            coverageStatus: "ELAPSED_RECENT",
            freshnessStatus: "RECENT",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED/i)).toBeDefined();
    });

    it("G10: Imminent earnings -> WARN / Blocked", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          hasImminentEarnings={true}
          catalystEvidence={{
            daysToEarnings: 0.5,
            coverageStatus: "CONFIRMED_UPCOMING",
            freshnessStatus: "LIVE",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — IMMINENT EARNINGS HAZARD \(<= 7 DAYS\)/i)).toBeDefined();
    });

    it("G11: Confirmed event within seven days -> WARN / Blocked", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            daysToEarnings: 4.0,
            coverageStatus: "CONFIRMED_UPCOMING",
            freshnessStatus: "LIVE",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — IMMINENT EARNINGS HAZARD \(<= 7 DAYS\)/i)).toBeDefined();
    });

    it("G12: Provider timeout -> ERROR / Blocked", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            state: "ERROR",
            sourceName: "Provider Timeout",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — CATALYST EVALUATION ERROR/i)).toBeDefined();
    });

    it("G13: Contradictory corporate-event fields -> Blocked", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          catalystEvidence={{
            daysToEarnings: -3.0,
            coverageStatus: "CONFIRMED_UPCOMING",
            freshnessStatus: "STALE",
          }}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED/i)).toBeDefined();
    });

    it("G14: Missing VIX, other checks pass -> Not cleared", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          vix={undefined as any}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — VIX DATA UNAVAILABLE/i)).toBeDefined();
    });

    it("G15: Four of five mandatory checks pass -> Not cleared", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          stopLoss={null as any}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED/i)).toBeDefined();
    });

    it("G16: Five checks pass but canonical setup non-actionable -> Not cleared", () => {
      render(
        <PreFlightChecklistModal
          {...basePristineProps}
          isActionable={false}
        />
      );
      expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
      expect(screen.getByText(/⚠️ NOT CLEARED — CANONICAL SETUP NON-ACTIONABLE/i)).toBeDefined();
    });
  });
});
