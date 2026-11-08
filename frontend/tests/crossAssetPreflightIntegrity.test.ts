/**
 * ARX Universal Pre-Flight Quantitative Evaluator — Cross-Asset Integrity Test Suite
 *
 * Directives:
 *   - ARX — Cross-Asset Pre-Flight Integrity Remediation (P1 Decision Integrity)
 *   - Universal coverage across Equities (NAUT, AAPL), ADRs (DHLGY), and ETFs (SPY).
 */

import { describe, it, expect } from "vitest";
import { evaluatePreFlightChecks, PreFlightChecklistInputs } from "../lib/preflightEvaluator";

describe("Cross-Asset Pre-Flight Decision Integrity Suite", () => {
  it("NAUT: Sub-60 quality score (33) triggers CAPITAL RISK and elapsed earnings fails closed to UNAVAILABLE", () => {
    const res = evaluatePreFlightChecks({
      symbol: "NAUT",
      currentPrice: 2.22,
      stopLoss: 1.73,
      takeProfit1: 3.16,
      riskRewardRatio: 1.86,
      optimalEntryMin: 1.78,
      optimalEntryMax: 2.48,
      breakoutPivot: null,
      shortFloat: 8.96,
      qualityScore: 33,
      assetClass: "EQUITY",
      securityType: "COMMON_STOCK",
      verdict: "Speculative Profile",
      vix: 25.6,
      vixObservationTime: new Date().toISOString(),
      catalystEvidence: {
        state: "UNAVAILABLE",
        daysToEarnings: -11.65,
        eventDate: "2026-10-27",
        confirmationStatus: "UNVERIFIED",
        coverageStatus: "ELAPSED_RECENT",
        freshnessStatus: "STALE",
        preflightState: "UNAVAILABLE",
      },
      isActionable: false,
    });

    // Check 1: R:R 1.86:1 < 2.0:1 threshold -> FAIL
    expect(res.checks[0].status).toBe("FAIL");
    expect(res.checks[0].passed).toBe(false);

    // Check 3: Quality Score 33 < 60 threshold -> CAPITAL RISK (Elevated Capital Distress)
    expect(res.checks[2].status).toBe("CAPITAL RISK");
    expect(res.checks[2].passed).toBe(false);
    expect(res.checks[2].summary).toBe("Elevated Capital Distress (Q-Score 33)");
    expect(res.checks[2].detail).toContain("33/100 < 60");

    // Check 4: Elapsed event without forward calendar -> UNAVAILABLE
    expect(res.checks[3].status).toBe("UNAVAILABLE");
    expect(res.checks[3].passed).toBe(false);

    // Check 5: VIX 25.6 -> PASS with Elevated Caution
    expect(res.checks[4].status).toBe("PASS");
    expect(res.checks[4].passed).toBe(true);
    expect(res.checks[4].summary).toBe("VIX 25.6 (Elevated Caution)");
    expect(res.checks[4].detail).toContain("Elevated market volatility (VIX 25.6)");

    // Overall: Clearance strictly blocked
    expect(res.isCleared).toBe(false);
  });

  it("DHLGY: Confirmed forward runway and solvent fundamental score pass Check 3 and Check 4", () => {
    const res = evaluatePreFlightChecks({
      symbol: "DHLGY",
      currentPrice: 38.5,
      stopLoss: 35.0,
      takeProfit1: 46.0,
      riskRewardRatio: 2.14,
      optimalEntryMin: 37.0,
      optimalEntryMax: 39.0,
      breakoutPivot: 39.2,
      shortFloat: 1.1,
      qualityScore: 84,
      assetClass: "EQUITY",
      securityType: "ADR",
      verdict: "Global Express & Logistics Monopoly",
      vix: 18.5,
      vixObservationTime: new Date().toISOString(),
      catalystEvidence: {
        state: "PASS",
        daysToEarnings: 117.0,
        eventDate: "2027-03-05",
        confirmationStatus: "CONFIRMED",
        coverageStatus: "CONFIRMED_UPCOMING",
        freshnessStatus: "LIVE",
        riskTier: "NORMAL_RUNWAY",
        preflightState: "PASS",
      },
      isActionable: true,
    });

    // Check 1: R:R 2.14 >= 2.0 -> PASS
    expect(res.checks[0].status).toBe("PASS");

    // Check 3: Short float 1.1%, Q-Score 84 >= 60 -> PASS
    expect(res.checks[2].status).toBe("PASS");
    expect(res.checks[2].summary).toBe("Float 1.1% | Q-Score 84");

    // Check 4: 117d confirmed runway -> PASS
    expect(res.checks[3].status).toBe("PASS");
    expect(res.checks[3].summary).toContain("Runway > 7d");

    // Check 5: VIX 18.5 < 20.0 -> Normal
    expect(res.checks[4].status).toBe("PASS");
    expect(res.checks[4].summary).toBe("VIX 18.5 (Normal)");

    // Fully cleared
    expect(res.isCleared).toBe(true);
    expect(res.passedCount).toBe(5);
  });

  it("SPY: Exchange-Traded Fund (ETF) receives NOT_APPLICABLE pass on corporate earnings and capital structure", () => {
    const res = evaluatePreFlightChecks({
      symbol: "SPY",
      currentPrice: 580.0,
      stopLoss: 565.0,
      takeProfit1: 615.0,
      riskRewardRatio: 2.33,
      optimalEntryMin: 575.0,
      optimalEntryMax: 585.0,
      breakoutPivot: 586.0,
      assetClass: "ETF",
      securityType: "ETF",
      vix: 16.2,
      vixObservationTime: new Date().toISOString(),
      catalystEvidence: {
        state: "PASS",
        daysToEarnings: null,
        confirmationStatus: "NOT_APPLICABLE",
        coverageStatus: "NOT_APPLICABLE",
        freshnessStatus: "NOT_APPLICABLE",
        riskTier: "NO_EARNINGS_HAZARD",
        preflightState: "PASS",
      },
      isActionable: true,
    });

    // Check 3: ETF Basket Structure -> PASS
    expect(res.checks[2].status).toBe("PASS");
    expect(res.checks[2].summary).toBe("Fund Structure | Solvency Exempt");
    expect(res.checks[2].detail).toContain("Exchange-Traded Fund (ETF): Diversified pooled capital structure.");

    // Check 4: ETF Earnings N/A -> PASS
    expect(res.checks[3].status).toBe("PASS");
    expect(res.checks[3].summary).toBe("ETF: Earnings N/A");
    expect(res.checks[3].detail).toContain("Exchange-Traded Fund (ETF): Does not report single-stock corporate earnings.");

    // Check 5: VIX 16.2 -> Normal PASS
    expect(res.checks[4].status).toBe("PASS");
    expect(res.checks[4].summary).toBe("VIX 16.2 (Normal)");

    expect(res.isCleared).toBe(true);
    expect(res.passedCount).toBe(5);
  });

  it("VIX Volatility Tiers: Strictly differentiates Normal (<20), Elevated Caution (20-26), and High Volatility (>=26)", () => {
    const baseInput: PreFlightChecklistInputs = {
      symbol: "AAPL",
      currentPrice: 225.0,
      stopLoss: 215.0,
      takeProfit1: 250.0,
      riskRewardRatio: 2.5,
      optimalEntryMin: 220.0,
      optimalEntryMax: 230.0,
      breakoutPivot: 232.0,
      qualityScore: 90,
      shortFloat: 0.8,
      vixObservationTime: new Date().toISOString(),
      catalystEvidence: {
        state: "PASS",
        daysToEarnings: 45.0,
        confirmationStatus: "CONFIRMED",
        coverageStatus: "CONFIRMED_UPCOMING",
        freshnessStatus: "LIVE",
        riskTier: "NORMAL_RUNWAY",
        preflightState: "PASS",
      },
      isActionable: true,
    };

    // 1. VIX 14.5 -> Normal (<20.0)
    const normalRes = evaluatePreFlightChecks({ ...baseInput, vix: 14.5 });
    expect(normalRes.checks[4].status).toBe("PASS");
    expect(normalRes.checks[4].summary).toBe("VIX 14.5 (Normal)");
    expect(normalRes.checks[4].detail).toContain("volatility is calm (VIX 14.5 < 20.0)");

    // 2. VIX 22.4 -> Elevated Caution (20.0 to 26.0)
    const elevatedRes = evaluatePreFlightChecks({ ...baseInput, vix: 22.4 });
    expect(elevatedRes.checks[4].status).toBe("PASS");
    expect(elevatedRes.checks[4].summary).toBe("VIX 22.4 (Elevated Caution)");
    expect(elevatedRes.checks[4].detail).toContain("Elevated market volatility (VIX 22.4)");

    // 3. VIX 26.5 -> High Volatility Blocker (>= 26.0)
    const highRes = evaluatePreFlightChecks({ ...baseInput, vix: 26.5 });
    expect(highRes.checks[4].status).toBe("HIGH VIX");
    expect(highRes.checks[4].summary).toBe("VIX 26.5 (High Volatility)");
    expect(highRes.checks[4].passed).toBe(false);
    expect(highRes.isCleared).toBe(false);

    // 4. Missing VIX -> UNAVAILABLE
    const unavailRes = evaluatePreFlightChecks({ ...baseInput, vix: null });
    expect(unavailRes.checks[4].status).toBe("UNAVAILABLE");
    expect(unavailRes.checks[4].summary).toBe("VIX Unavailable");
    expect(unavailRes.checks[4].passed).toBe(false);
    expect(unavailRes.isCleared).toBe(false);
  });

  it("Epistemic Honesty: Missing qualityScore and uncataloged symbol fails closed to UNAVAILABLE", () => {
    const unverifiedSymbolRes = evaluatePreFlightChecks({
      symbol: "UNKNOWN_TICKER_XYZ",
      currentPrice: 50.0,
      stopLoss: 45.0,
      takeProfit1: 65.0,
      riskRewardRatio: 3.0,
      optimalEntryMin: 48.0,
      optimalEntryMax: 52.0,
      breakoutPivot: 53.0,
      shortFloat: null,
      qualityScore: null,
      assetClass: "EQUITY",
      vix: 15.0,
      vixObservationTime: new Date().toISOString(),
      catalystEvidence: {
        state: "PASS",
        daysToEarnings: 30.0,
        confirmationStatus: "CONFIRMED",
        coverageStatus: "CONFIRMED_UPCOMING",
        freshnessStatus: "LIVE",
        riskTier: "NORMAL_RUNWAY",
        preflightState: "PASS",
      },
      isActionable: true,
    });

    expect(unverifiedSymbolRes.checks[2].status).toBe("UNAVAILABLE");
    expect(unverifiedSymbolRes.checks[2].passed).toBe(false);
    expect(unverifiedSymbolRes.checks[2].summary).toBe("Evidence Incomplete");
    expect(unverifiedSymbolRes.isCleared).toBe(false);
  });
});
