import { describe, it, expect } from "vitest";
import { evaluatePreFlightChecks } from "../preflightEvaluator";

describe("Gate 1: CHK-1 Reward/Risk Threshold Full Numeric Precision", () => {
  // Baseline price = 100, stop = 90 (risk = 10).
  // Target = 100 + 10 * ratio:
  // For ratio 1.994: target = 119.94, reward = 19.94, ratio = 1.994
  // For ratio 1.995: target = 119.95, reward = 19.95, ratio = 1.995 (rounds to 2.00)
  // For ratio 1.999: target = 119.99, reward = 19.99, ratio = 1.999 (rounds to 2.00)
  // For ratio 2.000: target = 120.00, reward = 20.00, ratio = 2.000 (rounds to 2.00)
  // For ratio 2.001: target = 120.01, reward = 20.01, ratio = 2.001 (rounds to 2.00)

  const baseInputs = {
    symbol: "NVDA",
    currentPrice: 100.0,
    stopLoss: 90.0,
    optimalEntryMin: 95.0,
    optimalEntryMax: 102.0,
    isActionable: true,
    vix: 15.0,
    vixObservationTime: new Date().toISOString(),
    catalystEvidence: {
      daysToEarnings: 15.0,
      confirmationStatus: "CONFIRMED" as const,
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE" as const,
      riskTier: "NORMAL_RUNWAY",
    },
  };

  it("1. Raw ratio 1.994: Fails closed (summary: '1.99:1.0', status: FAIL)", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: 119.94,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("FAIL");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("1.99:1.0");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("REWARD/RISK ASYMMETRY (< 2.0:1)");
  });

  it("2. Raw ratio 1.995: Rounds to 2.00 for display but strictly FAILS threshold check (Invariant: never pass on display rounding)", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: 119.95,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("FAIL");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("2.00:1.0"); // Rounded for display
    expect(chk1.detail).toContain("below 2.0:1 institutional floor");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("REWARD/RISK ASYMMETRY (< 2.0:1)");
  });

  it("3. Raw ratio 1.999: Rounds to 2.00 for display but strictly FAILS threshold check", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: 119.99,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("FAIL");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("2.00:1.0");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("REWARD/RISK ASYMMETRY (< 2.0:1)");
  });

  it("4. Raw ratio exactly 2.000: PASS (summary: '2.00:1.0', status: PASS)", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: 120.0,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("PASS");
    expect(chk1.passed).toBe(true);
    expect(chk1.summary).toBe("2.00:1.0");
    expect(res.isCleared).toBe(true);
  });

  it("5. Raw ratio 2.001: PASS (summary: '2.00:1.0', status: PASS)", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: 120.01,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("PASS");
    expect(chk1.passed).toBe(true);
    expect(chk1.summary).toBe("2.00:1.0");
    expect(res.isCleared).toBe(true);
  });

  it("6. Missing stop loss: Returns MISSING LEVELS", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      stopLoss: null,
      takeProfit1: 120.0,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("MISSING LEVELS");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("N/A");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("UNVERIFIED EXECUTION LEVELS");
  });

  it("7. Inverted stop loss (stop >= price): Returns FAIL with Invalid Levels", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      stopLoss: 105.0, // Above currentPrice 100
      takeProfit1: 120.0,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("FAIL");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("Invalid Levels");
    expect(chk1.detail).toContain("Inverted risk boundary: Stop loss ($105.00) is above current price ($100.00)");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("INVERTED EXECUTION BOUNDARIES");
  });

  it("8. Missing profit target: Returns MISSING LEVELS", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: null,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("MISSING LEVELS");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("N/A");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("UNVERIFIED EXECUTION LEVELS");
  });

  it("9. Inverted profit target (target <= price): Returns FAIL with Invalid Levels", () => {
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: 95.0, // Below currentPrice 100
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("FAIL");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("Invalid Levels");
    expect(chk1.detail).toContain("Inverted reward boundary: Profit target ($95.00) is below current price ($100.00)");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("INVERTED EXECUTION BOUNDARIES");
  });

  it("10. Stale model ratio higher than computed ratio: Fails closed to lower computed ratio", () => {
    // Model claims 2.5:1, but spot at $100 with stop $90 and target $115 has spot RR = 1.5:1
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: 115.0,
      riskRewardRatio: 2.5, // Optimistic model claim
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("FAIL");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("1.50:1.0");
    expect(res.isCleared).toBe(false);
  });

  it("11. Stale model ratio lower than computed ratio: Fails closed to lower model ratio", () => {
    // Spot at $100 with stop $90 and target $130 has spot RR = 3.0:1, but model ratio is 1.8:1
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      takeProfit1: 130.0,
      riskRewardRatio: 1.8, // Conservative / restrictive model claim
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("FAIL");
    expect(chk1.passed).toBe(false);
    expect(chk1.summary).toBe("1.80:1.0");
    expect(res.isCleared).toBe(false);
  });

  it("12. Different price snapshot bases: spot price outside corridor", () => {
    // Spot moved up to 105 (extended above corridor 95-102), making risk = 105 - 90 = 15, reward = 125 - 105 = 20, RR = 1.33:1
    const res = evaluatePreFlightChecks({
      ...baseInputs,
      currentPrice: 105.0,
      takeProfit1: 125.0,
      riskRewardRatio: 2.5,
    });
    const chk1 = res.checks[0];
    expect(chk1.status).toBe("FAIL");
    expect(chk1.summary).toBe("1.33:1.0");
    expect(res.isCleared).toBe(false);
  });
});

describe("Gate 2: CHK-4 Catalyst Confirmation Semantics & Evidence Hierarchy", () => {
  const basePristine = {
    symbol: "NVDA",
    currentPrice: 100.0,
    stopLoss: 90.0,
    takeProfit1: 125.0, // RR = 2.5:1
    optimalEntryMin: 95.0,
    optimalEntryMax: 102.0,
    isActionable: true,
    vix: 15.0,
    vixObservationTime: new Date().toISOString(),
  };

  it("C01: Confirmed upcoming earnings, more than seven days away -> PASS (Verified event runway)", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: 18.0,
        eventDate: "2026-11-15",
        confirmationStatus: "CONFIRMED",
        coverageStatus: "CONFIRMED_UPCOMING",
        freshnessStatus: "LIVE",
        riskTier: "NORMAL_RUNWAY",
      },
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("PASS");
    expect(chk4.passed).toBe(true);
    expect(chk4.summary).toBe("Runway > 7d (18.0d)");
    expect(chk4.detail).toContain("Clean runway verified");
    expect(res.isCleared).toBe(true);
  });

  it("C02: Estimated upcoming earnings, more than seven days away -> PASS with Unconfirmed disclosure", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: 14.0,
        eventDate: "2026-11-10",
        confirmationStatus: "ESTIMATED",
        coverageStatus: "ESTIMATED_UPCOMING",
        freshnessStatus: "LIVE",
        riskTier: "NORMAL_RUNWAY",
      },
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("PASS");
    expect(chk4.passed).toBe(true);
    expect(chk4.summary).toBe("Runway > 7d (Unconfirmed)");
    expect(chk4.detail).toContain("unverified by primary source");
    expect(chk4.detail).not.toContain("Clean runway verified");
    expect(res.isCleared).toBe(true);
  });

  it("C03: Coverage present (COVERED / UNVERIFIED aggregator), more than seven days away -> PASS with Unconfirmed disclosure", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: 21.0,
        eventDate: "2026-11-17",
        confirmationStatus: "UNVERIFIED",
        coverageStatus: "COVERED",
        freshnessStatus: "LIVE",
        riskTier: "NORMAL_RUNWAY",
      },
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("PASS");
    expect(chk4.passed).toBe(true);
    expect(chk4.summary).toBe("Runway > 7d (Unconfirmed)");
    expect(chk4.detail).toContain("unverified by primary source");
    expect(res.isCleared).toBe(true);
  });

  it("C04: Confirmed earnings within seven days -> WARN / HAZARD (Blocked)", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: 3.5,
        eventDate: "2026-10-30",
        confirmationStatus: "CONFIRMED",
        coverageStatus: "CONFIRMED_UPCOMING",
        freshnessStatus: "LIVE",
        riskTier: "CAUTION_BUFFER",
      },
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("HAZARD");
    expect(chk4.passed).toBe(false);
    expect(chk4.summary).toBe("Imminent Earnings Event");
    expect(chk4.detail).toContain("Scheduled earnings announcement within 7-day risk window");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("IMMINENT EARNINGS HAZARD (<= 7 DAYS)");
  });

  it("C05: Estimated earnings within seven days -> WARN / HAZARD (Blocked)", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: 2.0,
        eventDate: "2026-10-28",
        confirmationStatus: "ESTIMATED",
        coverageStatus: "ESTIMATED_UPCOMING",
        freshnessStatus: "LIVE",
        riskTier: "HIGH_GAP_RISK",
      },
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("HAZARD");
    expect(chk4.passed).toBe(false);
    expect(chk4.summary).toBe("Imminent Earnings Event");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("IMMINENT EARNINGS HAZARD (<= 7 DAYS)");
  });

  it("C06: Stale earnings coverage (freshnessStatus !== LIVE) -> UNAVAILABLE (Blocked)", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: 15.0,
        eventDate: "2026-11-10",
        confirmationStatus: "CONFIRMED",
        coverageStatus: "CONFIRMED_UPCOMING",
        freshnessStatus: "STALE",
        riskTier: "NORMAL_RUNWAY",
      },
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("UNAVAILABLE");
    expect(chk4.passed).toBe(false);
    expect(chk4.summary).toBe("Coverage Unavailable");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("STALE CATALYST COVERAGE");
  });

  it("C07: Missing event date / null evidence -> UNAVAILABLE (Blocked)", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: null,
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("UNAVAILABLE");
    expect(chk4.passed).toBe(false);
    expect(chk4.summary).toBe("Coverage Unavailable");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("CATALYST COVERAGE UNAVAILABLE");
  });

  it("C08: Conflicting provider dates / preflightState ERROR -> ERROR (Blocked)", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        state: "ERROR",
        preflightState: "ERROR",
        sourceName: "Conflicting Feed Sources",
      },
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("ERROR");
    expect(chk4.passed).toBe(false);
    expect(chk4.summary).toBe("Coverage Unavailable");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("CATALYST EVALUATION ERROR");
  });

  it("C09: Elapsed earnings event (negative DTE) without forward coverage -> UNAVAILABLE (Blocked)", () => {
    const res = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: -2.5,
        eventDate: "2026-10-24",
        confirmationStatus: "CONFIRMED",
        coverageStatus: "ELAPSED_RECENT",
      },
    });
    const chk4 = res.checks[3];
    expect(chk4.status).toBe("UNAVAILABLE");
    expect(chk4.passed).toBe(false);
    expect(chk4.detail).toContain("Historical Event Elapsed");
    expect(res.isCleared).toBe(false);
    expect(res.activeBlocker).toBe("CATALYST COVERAGE UNAVAILABLE");
  });

  it("C10: Primary-source confirmation versus secondary-source estimate correctly reflects in detail copy", () => {
    const confirmedRes = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: 20.0,
        eventDate: "2026-11-15",
        confirmationStatus: "CONFIRMED",
        coverageStatus: "CONFIRMED_UPCOMING",
        freshnessStatus: "LIVE",
      },
    });
    expect(confirmedRes.checks[3].summary).toBe("Runway > 7d (20.0d)");
    expect(confirmedRes.checks[3].detail).toContain("Clean runway verified");

    const estimateRes = evaluatePreFlightChecks({
      ...basePristine,
      catalystEvidence: {
        daysToEarnings: 20.0,
        eventDate: "2026-11-15",
        confirmationStatus: "ESTIMATED",
        coverageStatus: "ESTIMATED_UPCOMING",
        freshnessStatus: "LIVE",
      },
    });
    expect(estimateRes.checks[3].summary).toBe("Runway > 7d (Unconfirmed)");
    expect(estimateRes.checks[3].detail).toContain("unverified by primary source");
  });
});
