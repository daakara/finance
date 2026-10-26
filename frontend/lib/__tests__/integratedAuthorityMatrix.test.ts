import { describe, it, expect } from "vitest";
import { evaluatePreFlightChecks } from "../preflightEvaluator";
import { deriveAssessmentState } from "../assessmentEngine";
import { generateQuantitativeInsight } from "../insightGenerator";

describe("Gate 3: Integrated Authority & Preflight Cross-Component Regression Matrix", () => {
  const defaultPristineInputs = {
    symbol: "NVDA",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0, // Risk = 10, Reward = 25, RR = 2.5:1
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    decisionState: "ACTIONABLE_SETUP",
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

  it("Scenario 1: Analytical actionability true, preflight blocked (e.g. catalyst unavailable)", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
      catalystEvidence: null, // Check 4 unavailable
    });

    // Verification
    expect(pfRes.isActionableGranted).toBe(true);
    expect(pfRes.isCleared).toBe(false);
    expect(pfRes.activeBlocker).toBe("CATALYST COVERAGE UNAVAILABLE");
    expect(pfRes.checks[3].status).toBe("UNAVAILABLE");
    expect(pfRes.checks[3].passed).toBe(false);
    expect(pfRes.clearanceStatusLabel).toBe("NOT CLEARED (4/5)");
  });

  it("Scenario 2: Analytical actionability true, preflight cleared (5/5 checks pass)", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
    });

    expect(pfRes.isActionableGranted).toBe(true);
    expect(pfRes.mandatoryChecksVerified).toBe(true);
    expect(pfRes.isCleared).toBe(true);
    expect(pfRes.activeBlocker).toBe("NONE");
    expect(pfRes.clearanceStatusLabel).toBe("CLEARED");
  });

  it("Scenario 3: Analytical actionability false, all five checks passing", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
      isActionable: false,
      decisionState: "WAIT_FOR_TRIGGER",
    });

    expect(pfRes.isActionableGranted).toBe(false);
    expect(pfRes.mandatoryChecksVerified).toBe(true); // All 5 individual checks physically pass
    expect(pfRes.isCleared).toBe(false); // But overall clearance is BLOCKED because setup is not actionable
    expect(pfRes.activeBlocker).toBe("CANONICAL SETUP NON-ACTIONABLE");
    expect(pfRes.clearanceStatusLabel).toBe("NOT CLEARED · 5/5 PASS (AWAITING CONFIRMATION)");
  });

  it("Scenario 4: CHK-1 just below threshold (raw 1.995:1)", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
      takeProfit1: 239.951, // Risk = 10, Reward = 19.951, RR = 1.9951 (rounds to 2.00 for display)
    });

    expect(pfRes.checks[0].status).toBe("FAIL");
    expect(pfRes.checks[0].passed).toBe(false);
    expect(pfRes.checks[0].summary).toBe("2.00:1.0"); // Rounded display
    expect(pfRes.isCleared).toBe(false);
    expect(pfRes.activeBlocker).toBe("REWARD/RISK ASYMMETRY (< 2.0:1)");
    expect(pfRes.whatNeedsToChange).toContain("reward/risk ratio (>= 2.0:1)");
  });

  it("Scenario 5: CHK-4 estimated but unconfirmed (>7d runway)", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
      catalystEvidence: {
        daysToEarnings: 16.0,
        confirmationStatus: "ESTIMATED",
        coverageStatus: "ESTIMATED_UPCOMING",
        freshnessStatus: "LIVE",
        riskTier: "NORMAL_RUNWAY",
      },
    });

    expect(pfRes.checks[3].status).toBe("PASS");
    expect(pfRes.checks[3].passed).toBe(true);
    expect(pfRes.checks[3].summary).toBe("Runway > 7d (Unconfirmed)");
    expect(pfRes.checks[3].detail).toContain("unverified by primary source");
    expect(pfRes.isCleared).toBe(true);
  });

  it("Scenario 6: Stale market data (stale tape in assessment engine blocks sizing)", () => {
    const assessRes = deriveAssessmentState({
      symbol: "NVDA",
      currentPrice: 220.0,
      freshnessStatus: "STALE_HISTORICAL",
      domains: [
        { name: "valuation", status: "STABLE", score: 80, availability: "AVAILABLE" },
        { name: "growth", status: "STABLE", score: 85, availability: "AVAILABLE" },
        { name: "health", status: "STABLE", score: 90, availability: "AVAILABLE" },
        { name: "trend", status: "STABLE", score: 80, availability: "AVAILABLE" },
      ],
      decisionTrace: {
        verdict: "ACTIONABLE_BUY_ZONE",
        posture: "ACQUIRE",
        isActionable: true,
      },
    });

    expect(assessRes.canSizeTrade).toBe(false);
    expect(assessRes.isActionable).toBe(false);
    expect(assessRes.posture).toBe("RESEARCH");
    expect(assessRes.decisionState).toBe("STALE_DATA");
  });

  it("Scenario 7: Missing macro evidence (VIX undefined fails closed to UNAVAILABLE)", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
      vix: undefined,
    });

    expect(pfRes.checks[4].status).toBe("UNAVAILABLE");
    expect(pfRes.checks[4].passed).toBe(false);
    expect(pfRes.isCleared).toBe(false);
    expect(pfRes.activeBlocker).toBe("VIX DATA UNAVAILABLE");
  });

  it("Scenario 8: Entry outside corridor (price > entryMax * 1.02)", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
      currentPrice: 232.0, // entryMax is 225, 232 is > 225 * 1.02 (229.5)
      takeProfit1: 290.0, // Risk = 22, Reward = 58, RR = 2.64:1 (CHK-1 passes)
    });

    expect(pfRes.checks[1].status).toBe("CHASING");
    expect(pfRes.checks[1].passed).toBe(false);
    expect(pfRes.checks[1].summary).toBe("Out of Corridor");
    expect(pfRes.isCleared).toBe(false);
    expect(pfRes.activeBlocker).toBe("TECHNICAL STRUCTURE NOT CONFIRMED");
  });

  it("Scenario 9: Stage 4 correction structure", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
      isStage4: true,
    });

    expect(pfRes.checks[1].status).toBe("STAGE 4 WAIT");
    expect(pfRes.checks[1].passed).toBe(false);
    expect(pfRes.checks[1].summary).toBe("Stage 4 Correction");
    expect(pfRes.isCleared).toBe(false);
    expect(pfRes.activeBlocker).toBe("STAGE 4 DISTRIBUTION STRUCTURE");
  });

  it("Scenario 10: Invalid stop or target (inverted levels)", () => {
    const pfRes = evaluatePreFlightChecks({
      ...defaultPristineInputs,
      stopLoss: 225.0, // Stop > currentPrice 220
    });

    expect(pfRes.checks[0].status).toBe("FAIL");
    expect(pfRes.checks[0].passed).toBe(false);
    expect(pfRes.checks[0].summary).toBe("Invalid Levels");
    expect(pfRes.isCleared).toBe(false);
    expect(pfRes.activeBlocker).toBe("INVERTED EXECUTION BOUNDARIES");
  });
});
