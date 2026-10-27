import { describe, it, expect } from "vitest";
import { evaluatePreFlightChecks, PreFlightChecklistInputs } from "../preflightEvaluator";

const BASE_CLEARED_INPUTS: PreFlightChecklistInputs = {
  symbol: "NVDA",
  currentPrice: 292.0,
  stopLoss: 268.0,
  takeProfit1: 340.0,
  riskRewardRatio: 2.0,
  optimalEntryMin: 290.79,
  optimalEntryMax: 296.61,
  breakoutPivot: 292.0,
  isActionable: true,
  isStage4: false,
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 18.0,
    confirmationStatus: "CONFIRMED",
    coverageStatus: "CONFIRMED_UPCOMING",
    riskTier: "NORMAL_RUNWAY",
    freshnessStatus: "LIVE",
  },
  vix: 16.5,
  macroEvidence: {
    vix: 16.5,
    freshnessStatus: "LIVE",
    observationTime: new Date().toISOString(),
  },
};

describe("PreFlight Provenance & Epistemic Authority Invariants (Gate 2 & Gate 3)", () => {
  it("Invariant 1: Five passing checks DO NOT prove a live observation when provenance is missing", () => {
    // Inputs have 5/5 passing checks and isActionable=true, but NO dataSource specified
    const res = evaluatePreFlightChecks(BASE_CLEARED_INPUTS);

    expect(res.isCleared).toBe(true);
    expect(res.passedCount).toBe(5);
    expect(res.provenanceCategory).toBe("UNVERIFIED");
    expect(res.evidenceMeta).not.toContain("LIVE MARKET OBSERVATION");
    expect(res.evidenceMeta).toBe("Evidence: UNVERIFIED PROVENANCE · Feed Verification Required");
    expect(res.whatNeedsToChange).toContain("Notice: Unverified market feed — requires verified real-time data before execution.");
  });

  it("Invariant 2: Historical EOD evidence must NEVER be labeled live even with 5/5 passing checks", () => {
    const historicalInputs: PreFlightChecklistInputs = {
      ...BASE_CLEARED_INPUTS,
      dataSource: "historical",
      isHistorical: true,
      quoteStatus: "HISTORICAL_EOD",
    };

    const res = evaluatePreFlightChecks(historicalInputs);

    expect(res.isCleared).toBe(true);
    expect(res.provenanceCategory).toBe("HISTORICAL_EOD");
    expect(res.evidenceMeta).not.toContain("LIVE");
    expect(res.evidenceMeta).toBe("Evidence: HISTORICAL EOD · Reconciled Close Observation");
    expect(res.whatNeedsToChange).toContain("Notice: Historical EOD data — requires live market confirmation before execution.");
  });

  it("Invariant 2b: Adversarial false-live evidenceStatus string on historical EOD is sanitized fail-closed", () => {
    const adversarialHistorical: PreFlightChecklistInputs = {
      ...BASE_CLEARED_INPUTS,
      dataSource: "historical",
      evidenceStatus: "Evidence: LIVE MARKET OBSERVATION · 5/5 Mandatory Checks", // False spoof attempt
    };

    const res = evaluatePreFlightChecks(adversarialHistorical);

    expect(res.provenanceCategory).toBe("HISTORICAL_EOD");
    expect(res.evidenceMeta).not.toContain("LIVE");
    expect(res.evidenceMeta).toBe("Evidence: HISTORICAL EOD · Reconciled Close Observation");
  });

  it("Invariant 3: Synthetic fixture evidence must NEVER be labeled live", () => {
    const syntheticInputs: PreFlightChecklistInputs = {
      ...BASE_CLEARED_INPUTS,
      dataSource: "synthetic",
      isSynthetic: true,
    };

    const res = evaluatePreFlightChecks(syntheticInputs);

    expect(res.isCleared).toBe(true);
    expect(res.provenanceCategory).toBe("SYNTHETIC_FIXTURE");
    expect(res.evidenceMeta).not.toContain("LIVE MARKET OBSERVATION");
    expect(res.evidenceMeta).toBe("Evidence: SYNTHETIC FIXTURE · Simulated Pre-Flight Verification");
    expect(res.isSimulatedClearance).toBe(true);
    expect(res.whatNeedsToChange).toContain("Notice: Synthetic fixture — not authorized for live order execution.");
  });

  it("Invariant 3b: Adversarial false-live evidenceStatus string on synthetic fixture is sanitized fail-closed", () => {
    const adversarialSynthetic: PreFlightChecklistInputs = {
      ...BASE_CLEARED_INPUTS,
      isSynthetic: true,
      evidenceStatus: "Evidence: LIVE MARKET OBSERVATION · 5/5 Mandatory Checks", // False spoof attempt
    };

    const res = evaluatePreFlightChecks(adversarialSynthetic);

    expect(res.provenanceCategory).toBe("SYNTHETIC_FIXTURE");
    expect(res.evidenceMeta).not.toContain("LIVE");
    expect(res.evidenceMeta).toBe("Evidence: SYNTHETIC FIXTURE · Simulated Pre-Flight Verification");
  });

  it("Invariant 4: Missing provenance remains strictly unverified", () => {
    const fallbackInputs: PreFlightChecklistInputs = {
      ...BASE_CLEARED_INPUTS,
      dataSource: "fallback",
    };

    const res = evaluatePreFlightChecks(fallbackInputs);

    expect(res.provenanceCategory).toBe("UNVERIFIED");
    expect(res.evidenceMeta).toBe("Evidence: UNVERIFIED PROVENANCE · Feed Verification Required");
    expect(res.evidenceMeta).not.toContain("LIVE");
  });

  it("Invariant 5: Simulated preflight clearance strictly disclaims live execution authorization", () => {
    const syntheticCleared: PreFlightChecklistInputs = {
      ...BASE_CLEARED_INPUTS,
      isSynthetic: true,
    };

    const res = evaluatePreFlightChecks(syntheticCleared);

    expect(res.isCleared).toBe(true);
    expect(res.isSimulatedClearance).toBe(true);
    expect(res.whatNeedsToChange).toBe(
      "All pre-flight clearance conditions satisfied in simulation. Notice: Synthetic fixture — not authorized for live order execution."
    );
  });

  it("Invariant 6: Genuine live observation requires live source + live freshness + valid price", () => {
    const genuineLive: PreFlightChecklistInputs = {
      ...BASE_CLEARED_INPUTS,
      dataSource: "live",
      liveSource: "ALPACA_IEX",
      liveFreshnessStatus: "LIVE",
    };

    const res = evaluatePreFlightChecks(genuineLive);

    expect(res.isCleared).toBe(true);
    expect(res.isSimulatedClearance).toBe(false);
    expect(res.provenanceCategory).toBe("LIVE_OBSERVATION");
    expect(res.evidenceMeta).toBe("Evidence: LIVE MARKET OBSERVATION · 5/5 Mandatory Checks");
    expect(res.whatNeedsToChange).toBe("All pre-flight clearance conditions satisfied. Setup cleared for controlled execution.");
  });

  it("Invariant 7: Financial clearance calculations and thresholds are strictly identical across provenance modes", () => {
    const liveRes = evaluatePreFlightChecks({ ...BASE_CLEARED_INPUTS, dataSource: "live", liveSource: "ALPACA_IEX", liveFreshnessStatus: "LIVE" });
    const histRes = evaluatePreFlightChecks({ ...BASE_CLEARED_INPUTS, dataSource: "historical", isHistorical: true });
    const synthRes = evaluatePreFlightChecks({ ...BASE_CLEARED_INPUTS, dataSource: "synthetic", isSynthetic: true });
    const unverRes = evaluatePreFlightChecks(BASE_CLEARED_INPUTS);

    // Clearance and check pass states must be invariant
    expect(liveRes.isCleared).toBe(histRes.isCleared);
    expect(histRes.isCleared).toBe(synthRes.isCleared);
    expect(synthRes.isCleared).toBe(unverRes.isCleared);
    expect(liveRes.passedCount).toBe(histRes.passedCount);
    expect(liveRes.safeRR).toBe(synthRes.safeRR);
    expect(liveRes.safePrice).toBe(histRes.safePrice);
    expect(liveRes.safeStop).toBe(synthRes.safeStop);
    expect(liveRes.safeTarget).toBe(unverRes.safeTarget);
  });
});
