/**
 * ARX Universal Pre-Flight Quantitative Evaluator — Elevated VIX Caution Test Suite
 *
 * Directives:
 *   - ARX — Targeted Closure and Radar P2 Phase 1 (Workstream A)
 *   - Preserve VIX thresholds: < 20.0 normal/calm, 20.0–25.99 elevated caution, >= 26.0 blocker
 *   - When VIX is 20.0–25.99 and trading is otherwise cleared, display CLEARED — ELEVATED VOLATILITY CAUTION
 *     with amber styling.
 *   - Carry the elevated-volatility warning into both exported trade-plan formats.
 *   - Correct Quant Mode title so it never describes elevated volatility as unequivocally safe.
 */

import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { evaluatePreFlightChecks, PreFlightChecklistInputs } from "../lib/preflightEvaluator";

const BASE_CLEARED_INPUTS: PreFlightChecklistInputs = {
  symbol: "AAPL",
  currentPrice: 220.0,
  stopLoss: 210.0,
  takeProfit1: 245.0,
  riskRewardRatio: 2.5,
  setupPattern: "Minervini VCP",
  isActionable: true,
  canSizeTrade: true,
  optimalEntryMin: 215.0,
  optimalEntryMax: 222.0,
  breakoutPivot: 220.0,
  shortFloat: 1.5,
  qualityScore: 88,
  assetClass: "EQUITY",
  securityType: "COMMON_STOCK",
  isStage4: false,
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 45.0,
    eventDate: "2026-12-20",
    confirmationStatus: "CONFIRMED",
    coverageStatus: "CONFIRMED_UPCOMING",
    freshnessStatus: "LIVE",
    riskTier: "NORMAL_RUNWAY",
    preflightState: "PASS",
  },
  dataSource: "live",
  liveSource: "ALPACA_IEX",
  liveFreshnessStatus: "LIVE",
};

describe("Pre-Flight Clearance Elevated VIX Caution Suite (Workstream A)", () => {
  it("VIX = 22.5 (20.0 <= VIX < 26.0) produces CLEARED — ELEVATED VOLATILITY CAUTION with amber styling", () => {
    const res = evaluatePreFlightChecks({
      ...BASE_CLEARED_INPUTS,
      vix: 22.5,
      vixObservationTime: new Date().toISOString(),
      macroEvidence: {
        vix: 22.5,
        observationTime: new Date().toISOString(),
        marketSession: "REGULAR",
        freshnessStatus: "LIVE",
      },
    });

    expect(res.isCleared).toBe(true);
    expect(res.passedCount).toBe(5);
    expect(res.isVixElevated).toBe(true);
    expect(res.clearanceStatusLabel).toBe("CLEARED — ELEVATED VOLATILITY CAUTION");
    expect(res.clearanceStatusBadgeClass).toContain("bg-amber-950/80 text-amber-300 border-amber-700/80");

    // Check 5 specific verification
    expect(res.checks[4].status).toBe("PASS");
    expect(res.checks[4].passed).toBe(true);
    expect(res.checks[4].summary).toBe("VIX 22.5 (Elevated Caution)");
  });

  it("VIX = 16.4 (< 20.0) produces standard CLEARED with emerald styling", () => {
    const res = evaluatePreFlightChecks({
      ...BASE_CLEARED_INPUTS,
      vix: 16.4,
      vixObservationTime: new Date().toISOString(),
      macroEvidence: {
        vix: 16.4,
        observationTime: new Date().toISOString(),
        marketSession: "REGULAR",
        freshnessStatus: "LIVE",
      },
    });

    expect(res.isCleared).toBe(true);
    expect(res.passedCount).toBe(5);
    expect(res.isVixElevated).toBe(false);
    expect(res.clearanceStatusLabel).toBe("CLEARED");
    expect(res.clearanceStatusBadgeClass).toContain("bg-emerald-950/80 text-emerald-300 border-emerald-700/80");

    // Check 5 specific verification
    expect(res.checks[4].status).toBe("PASS");
    expect(res.checks[4].passed).toBe(true);
    expect(res.checks[4].summary).toBe("VIX 16.4 (Normal)");
  });

  it("VIX = 26.5 (>= 26.0) blocks clearance with HIGH VIX status and amber blocker styling", () => {
    const res = evaluatePreFlightChecks({
      ...BASE_CLEARED_INPUTS,
      vix: 26.5,
      vixObservationTime: new Date().toISOString(),
      macroEvidence: {
        vix: 26.5,
        observationTime: new Date().toISOString(),
        marketSession: "REGULAR",
        freshnessStatus: "LIVE",
      },
    });

    expect(res.isCleared).toBe(false);
    expect(res.checks[4].status).toBe("HIGH VIX");
    expect(res.checks[4].passed).toBe(false);
    expect(res.activeBlocker).toContain("MACRO VOLATILITY REGIME (VIX >= 26.0)");
    expect(res.clearanceStatusLabel).toBe("NOT CLEARED (4/5)");
  });

  it("PreFlightChecklistModal source integrity verifies title, barometer, and export warnings", () => {
    const modalPath = path.join(__dirname, "../components/PreFlightChecklistModal.tsx");
    const modalSrc = fs.readFileSync(modalPath, "utf8");

    // 1. Quant Mode title must differentiate elevated volatility from safe volatility
    expect(modalSrc).toContain(
      'isVixElevated\n                          ? "5. Macro Regime Guard (VIX Elevated, Caution Warranted)"'
    );
    expect(modalSrc).toContain(
      'isMacroPassed\n                          ? "5. Macro Regime Guard (VIX Volatility Guardrails Safe)"'
    );

    // 2. Barometer must render CLEARED — ELEVATED VOLATILITY CAUTION with amber styling
    expect(modalSrc).toContain(
      'isCleared\n                    ? (isVixElevated\n                        ? "⚠️ CLEARED — ELEVATED VOLATILITY CAUTION"\n                        : "🟢 CLEARED TO EXECUTE")'
    );
    expect(modalSrc).toContain(
      'isCleared\n              ? (isVixElevated\n                  ? "bg-amber-950/40 border-amber-700/80 text-amber-300"\n                  : "bg-emerald-950/40 border-emerald-700/80 text-emerald-300")'
    );

    // 3. Trade plan export must carry caution notice in both Plain English and Quant Mode formats
    expect(modalSrc).toContain(
      'isVixElevated ? `> ⚠️ **CAUTION: ELEVATED MARKET VOLATILITY (VIX ${currentVix?.toFixed(1)})**\\n> Volatility regime elevated (20.0–26.0) — exercise caution on position sizing.\\n\\n` : ""'
    );
    expect(modalSrc).toContain(
      '- **Pre-Flight Score**: ${passedCount}/5 Checks Passed (${convictionPct}% Checklist) · ${isCleared ? (isVixElevated ? "⚠️ CLEARED — ELEVATED VOLATILITY CAUTION" : "🟢 CLEARED")'
    );
    expect(modalSrc).toContain(
      '- **Pre-Flight Clearance**: ${passedCount}/5 Checks Passed (${convictionPct}% Checklist) — ${isCleared ? (isVixElevated ? "⚠️ CLEARED — ELEVATED VOLATILITY CAUTION" : "🟢 CLEARED FOR EXECUTION")'
    );
    expect(modalSrc).toContain(
      '- **VIX Regime**: ${currentVix !== null ? currentVix.toFixed(1) : "UNAVAILABLE"}${isVixElevated ? " (ELEVATED CAUTION)" : ""}'
    );
  });
});
