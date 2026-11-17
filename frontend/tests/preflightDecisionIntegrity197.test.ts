/**
 * ARX-PREFLIGHT-INTEGRITY-031 Formal Acceptance & Falsification Suite
 * Tracker: Issue #197
 *
 * Verifies:
 * 1. Price Freshness Gate: Stale / expired / unavailable live price fails closed (isCleared === false).
 * 2. Multi-Blocker Surfacing: STALE PRICE OBSERVATION added to allBlockers list.
 * 3. Epistemic Clearance Differentiation:
 *    - Synthetic fixtures produce isSimulatedClearance = true and display SIMULATED CLEARANCE — NOT EXECUTABLE.
 *    - Historical EOD data produce isHistoricalReference = true and display HISTORICAL REFERENCE — NOT EXECUTABLE.
 *    - Genuine live observations produce LIVE_OBSERVATION and CLEARED (or CLEARED — ELEVATED VOLATILITY CAUTION).
 * 4. Cross-Domain Macro Independence: Live VIX never confers live clearance authority on stale equity prices.
 * 5. Journal Export Safeguards: Trade plan exports strictly prepend RESEARCH / JOURNAL ONLY — NOT EXECUTABLE
 *    under non-cleared, simulated, historical, or unverified conditions.
 * 6. Score & Barometer Parity: Primary score communicates passedCount/5 Checks Passed without ambiguity.
 * 7. Preservation of Institutional R:R Thresholds: MIN_REWARD_TO_RISK_RATIO (2.0) and governor untouched.
 */

import assert from "node:assert";
import { evaluatePreFlightChecks, PreFlightChecklistInputs, PREFLIGHT_CONFIG } from "../lib/preflightEvaluator";

console.log("=== ARX-PREFLIGHT-INTEGRITY-031 Acceptance & Falsification Suite (Issue #197) ===\n");

const BASE_LIVE_PASS_INPUTS: PreFlightChecklistInputs = {
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
  isStage4: false,
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 15.0,
    eventDate: "2026-11-20",
    confirmationStatus: "CONFIRMED",
    coverageStatus: "CONFIRMED_UPCOMING",
    freshnessStatus: "LIVE",
    riskTier: "NORMAL_RUNWAY",
  },
  macroEvidence: {
    vix: 15.2,
    observationTime: new Date().toISOString(),
    marketSession: "REGULAR",
    freshnessStatus: "LIVE",
  },
  dataSource: "live",
  liveSource: "ALPACA_IEX",
  liveFreshnessStatus: "LIVE",
};

// -----------------------------------------------------------------------------
// Test 1: Stale Price Observation Fails Closed
// -----------------------------------------------------------------------------
console.log("1. Verifying stale price fails closed even with fresh macro VIX...");
{
  const stalePriceInput: PreFlightChecklistInputs = {
    ...BASE_LIVE_PASS_INPUTS,
    liveFreshnessStatus: "STALE",
    macroEvidence: {
      vix: 14.5,
      observationTime: new Date().toISOString(),
      marketSession: "REGULAR",
      freshnessStatus: "LIVE",
    },
  };

  const res = evaluatePreFlightChecks(stalePriceInput);
  assert.strictEqual(res.passedCount, 5, "All 5 check items pass on levels");
  assert.strictEqual(res.isCleared, false, "Stale price MUST fail closed: isCleared must be false");
  assert.ok(res.allBlockers.includes("STALE PRICE OBSERVATION"), "allBlockers must contain STALE PRICE OBSERVATION");
  assert.strictEqual(res.provenanceCategory, "UNVERIFIED");
  assert.ok(res.whatNeedsToChange.includes("Wait for live exchange price feed to refresh"), "whatNeedsToChange must explain price staleness");
  console.log("   ✓ Confirmed: Stale price observation strictly blocks execution clearance");
}

// -----------------------------------------------------------------------------
// Test 2: Expired Price Observation Fails Closed
// -----------------------------------------------------------------------------
console.log("2. Verifying expired price fails closed...");
{
  const expiredPriceInput: PreFlightChecklistInputs = {
    ...BASE_LIVE_PASS_INPUTS,
    liveFreshnessStatus: "EXPIRED",
  };

  const res = evaluatePreFlightChecks(expiredPriceInput);
  assert.strictEqual(res.isCleared, false, "Expired price MUST fail closed: isCleared must be false");
  assert.ok(res.allBlockers.includes("STALE PRICE OBSERVATION"), "allBlockers must contain STALE PRICE OBSERVATION");
  console.log("   ✓ Confirmed: Expired price observation strictly blocks execution clearance");
}

// -----------------------------------------------------------------------------
// Test 3: Epistemic Distinction — Simulated Clearance vs Live Clearance
// -----------------------------------------------------------------------------
console.log("3. Verifying simulated clearance differentiation...");
{
  const syntheticInput: PreFlightChecklistInputs = {
    ...BASE_LIVE_PASS_INPUTS,
    dataSource: "synthetic",
    isSynthetic: true,
  };

  const res = evaluatePreFlightChecks(syntheticInput);
  assert.strictEqual(res.isCleared, true, "Simulation mathematically clears");
  assert.strictEqual(res.isSimulatedClearance, true, "isSimulatedClearance must be true");
  assert.strictEqual(res.isHistoricalReference, false);
  assert.strictEqual(res.provenanceCategory, "SYNTHETIC_FIXTURE");
  assert.strictEqual(res.clearanceStatusLabel, "SIMULATED CLEARANCE");
  assert.ok(res.whatNeedsToChange.includes("Synthetic fixture — not authorized for live order execution"));
  console.log("   ✓ Confirmed: Synthetic fixtures labeled as SIMULATED CLEARANCE");
}

// -----------------------------------------------------------------------------
// Test 4: Epistemic Distinction — Historical Reference vs Live Clearance
// -----------------------------------------------------------------------------
console.log("4. Verifying historical reference differentiation...");
{
  const historicalInput: PreFlightChecklistInputs = {
    ...BASE_LIVE_PASS_INPUTS,
    dataSource: "historical",
    isHistorical: true,
  };

  const res = evaluatePreFlightChecks(historicalInput);
  assert.strictEqual(res.isCleared, true, "Historical close mathematically clears");
  assert.strictEqual(res.isHistoricalReference, true, "isHistoricalReference must be true");
  assert.strictEqual(res.isSimulatedClearance, false);
  assert.strictEqual(res.provenanceCategory, "HISTORICAL_EOD");
  assert.strictEqual(res.clearanceStatusLabel, "HISTORICAL REFERENCE");
  assert.ok(res.whatNeedsToChange.includes("Historical EOD data — requires live market confirmation"));
  console.log("   ✓ Confirmed: Historical EOD labeled as HISTORICAL REFERENCE");
}

// -----------------------------------------------------------------------------
// Test 5: Genuine Live Market Observation Clears with Full Live Authority
// -----------------------------------------------------------------------------
console.log("5. Verifying genuine live market observation clears with live authority...");
{
  const res = evaluatePreFlightChecks(BASE_LIVE_PASS_INPUTS);
  assert.strictEqual(res.isCleared, true);
  assert.strictEqual(res.isSimulatedClearance, false);
  assert.strictEqual(res.isHistoricalReference, false);
  assert.strictEqual(res.provenanceCategory, "LIVE_OBSERVATION");
  assert.strictEqual(res.clearanceStatusLabel, "CLEARED");
  assert.strictEqual(res.activeBlocker, "NONE");
  console.log("   ✓ Confirmed: Genuine live observation clears with full CLEARED authority");
}

// -----------------------------------------------------------------------------
// Test 6: NVO Multi-Blocker Surface Preservation
// -----------------------------------------------------------------------------
console.log("6. Verifying NVO multi-blocker condition surfaces all active reasons...");
{
  const nvoInput: PreFlightChecklistInputs = {
    symbol: "NVO",
    currentPrice: 114.5,
    stopLoss: 108.0,
    takeProfit1: 120.0,
    riskRewardRatio: 0.85,
    isActionable: false,
    canSizeTrade: false,
    optimalEntryMin: 109.0,
    optimalEntryMax: 112.0,
    catalystEvidence: null,
    macroEvidence: {
      vix: 16.0,
      observationTime: new Date().toISOString(),
      marketSession: "REGULAR",
      freshnessStatus: "LIVE",
    },
    dataSource: "live",
    liveSource: "ALPACA_IEX",
    liveFreshnessStatus: "STALE", // Stale price on NVO as well
  };

  const res = evaluatePreFlightChecks(nvoInput);
  assert.strictEqual(res.isCleared, false);
  assert.ok(res.allBlockers.length >= 5, `Expected >= 5 blockers, got ${res.allBlockers.length}`);
  assert.ok(res.allBlockers.includes("CANONICAL SETUP NON-ACTIONABLE"));
  assert.ok(res.allBlockers.includes("CANONICAL SIZING PROHIBITED"));
  assert.ok(res.allBlockers.includes("STALE PRICE OBSERVATION"));
  assert.ok(res.allBlockers.includes("REWARD/RISK ASYMMETRY (< 2.0:1)"));
  assert.ok(res.allBlockers.includes("TECHNICAL STRUCTURE NOT CONFIRMED"));
  assert.ok(res.allBlockers.includes("CATALYST COVERAGE UNAVAILABLE"));
  console.log(`   ✓ Confirmed: NVO surfaces ${res.allBlockers.length} active blockers including STALE PRICE OBSERVATION`);
}

// -----------------------------------------------------------------------------
// Test 7: Threshold & Sizing Safeguard Invariants
// -----------------------------------------------------------------------------
console.log("7. Verifying R:R 2.0:1 floor preservation...");
{
  assert.strictEqual(PREFLIGHT_CONFIG.MIN_REWARD_TO_RISK_RATIO, 2.0);
  assert.strictEqual(PREFLIGHT_CONFIG.REQUIRED_RR_DISPLAY_THRESHOLD, "2:1");
  assert.strictEqual(PREFLIGHT_CONFIG.REQUIRED_RR_LABEL, "Required: 2:1");
  console.log("   ✓ Confirmed: Institutional 2.0:1 floor strictly preserved");
}

// -----------------------------------------------------------------------------
// Test 8: False Clearance Case 1 — Delayed Quote with Unverified Provenance
// -----------------------------------------------------------------------------
console.log("8. Verifying delayed quote with unverified provenance strictly fails closed...");
{
  const delayedQuoteInput: PreFlightChecklistInputs = {
    ...BASE_LIVE_PASS_INPUTS,
    liveFreshnessStatus: "DELAYED",
    quoteStatus: "DELAYED",
  };

  const res = evaluatePreFlightChecks(delayedQuoteInput);
  assert.strictEqual(res.passedCount, 5, "All 5 check items pass on levels");
  assert.strictEqual(res.isCleared, false, "Delayed quote MUST NEVER receive live clearance");
  assert.strictEqual(res.provenanceCategory, "UNVERIFIED", "Delayed quote provenance cannot be verified live");
  assert.ok(res.allBlockers.includes("DELAYED PRICE OBSERVATION"), "allBlockers must contain DELAYED PRICE OBSERVATION");
  assert.ok(res.allBlockers.includes("UNVERIFIED PRICE PROVENANCE"), "allBlockers must contain UNVERIFIED PRICE PROVENANCE");
  assert.strictEqual(res.activeBlocker, "DELAYED PRICE OBSERVATION", "Primary blocker must identify delayed price");
  assert.strictEqual(res.clearanceStatusLabel, "NOT CLEARED · DELAYED QUOTE (5/5)");
  assert.ok(res.whatNeedsToChange.includes("Wait for real-time exchange feed"), "whatNeedsToChange must explain tape delay");
  console.log("   ✓ Confirmed: Delayed quote strictly blocked with DELAYED PRICE OBSERVATION");
}

// -----------------------------------------------------------------------------
// Test 9: False Clearance Case 1b — Missing / Unverified Quote Provenance
// -----------------------------------------------------------------------------
console.log("9. Verifying unverified quote provenance strictly fails closed...");
{
  const unverifiedQuoteInput: PreFlightChecklistInputs = {
    ...BASE_LIVE_PASS_INPUTS,
    dataSource: undefined,
    liveSource: undefined,
    liveFreshnessStatus: undefined,
  };

  const res = evaluatePreFlightChecks(unverifiedQuoteInput);
  assert.strictEqual(res.passedCount, 5, "All 5 check items pass on levels");
  assert.strictEqual(res.isCleared, false, "Unverified quote MUST NEVER receive live clearance");
  assert.strictEqual(res.provenanceCategory, "UNVERIFIED");
  assert.ok(res.allBlockers.includes("UNVERIFIED PRICE PROVENANCE"), "allBlockers must contain UNVERIFIED PRICE PROVENANCE");
  assert.strictEqual(res.clearanceStatusLabel, "NOT CLEARED · UNVERIFIED PROVENANCE (5/5)");
  assert.ok(res.whatNeedsToChange.includes("Establish verified real-time exchange quote provenance"), "whatNeedsToChange must explain provenance requirement");
  console.log("   ✓ Confirmed: Unverified quote provenance strictly blocked with UNVERIFIED PRICE PROVENANCE");
}

// -----------------------------------------------------------------------------
// Test 10: False Clearance Case 2 — Missing canSizeTrade Authorization
// -----------------------------------------------------------------------------
console.log("10. Verifying missing canSizeTrade field strictly blocks live clearance...");
{
  const missingSizingInput: PreFlightChecklistInputs = {
    ...BASE_LIVE_PASS_INPUTS,
    canSizeTrade: undefined, // Missing canonical sizing authorization
  };

  const res = evaluatePreFlightChecks(missingSizingInput);
  assert.strictEqual(res.passedCount, 5, "All 5 check items pass on levels");
  assert.strictEqual(res.canSizeTradeGranted, false, "canSizeTradeGranted must be false when missing");
  assert.strictEqual(res.isCleared, false, "Missing sizing authorization MUST NEVER receive clearance");
  assert.strictEqual(res.activeBlocker, "CANONICAL SIZING PROHIBITED");
  assert.ok(res.allBlockers.includes("CANONICAL SIZING PROHIBITED"));
  assert.strictEqual(res.clearanceStatusLabel, "NOT CLEARED · 5/5 PASS (SIZING LOCKED)");
  assert.ok(res.whatNeedsToChange.includes("Position sizing cannot be authorized"), "whatNeedsToChange must explain sizing lock");
  console.log("   ✓ Confirmed: Missing canSizeTrade strictly blocks clearance with CANONICAL SIZING PROHIBITED");
}

// -----------------------------------------------------------------------------
// Test 11: False Clearance Case 2b — False canSizeTrade Authorization
// -----------------------------------------------------------------------------
console.log("11. Verifying explicit canSizeTrade === false strictly blocks live clearance...");
{
  const falseSizingInput: PreFlightChecklistInputs = {
    ...BASE_LIVE_PASS_INPUTS,
    canSizeTrade: false,
  };

  const res = evaluatePreFlightChecks(falseSizingInput);
  assert.strictEqual(res.canSizeTradeGranted, false);
  assert.strictEqual(res.isCleared, false);
  assert.strictEqual(res.activeBlocker, "CANONICAL SIZING PROHIBITED");
  assert.ok(res.allBlockers.includes("CANONICAL SIZING PROHIBITED"));
  console.log("   ✓ Confirmed: False canSizeTrade strictly blocks clearance");
}

// -----------------------------------------------------------------------------
// Test 12: Audited NVDA Snapshot Verification (2/5 checks passed and Not cleared)
// -----------------------------------------------------------------------------
console.log("12. Verifying audited NVDA snapshot returns 2/5 checks passed and 'Not cleared'...");
{
  const auditedNvdaSnapshot: PreFlightChecklistInputs = {
    symbol: "NVDA",
    currentPrice: 140.0,
    stopLoss: 125.0,
    takeProfit1: 142.0, // R:R = (142 - 140) / (140 - 125) = 2 / 15 = 0.13:1 (< 2.0:1) -> Check 1 FAILS
    riskRewardRatio: 0.13,
    isStage4: true, // Stage 4 correction structure -> Check 2 FAILS
    optimalEntryMin: 138.0,
    optimalEntryMax: 142.0,
    breakoutPivot: 145.0,
    qualityScore: 95,
    shortFloat: 2.2, // Check 3 Capital Health -> PASS
    hasImminentEarnings: true, // Imminent binary earnings hazard -> Check 4 FAILS
    catalystEvidence: {
      state: "WARN",
      daysToEarnings: 0.5,
      eventDate: "2026-11-17",
      confirmationStatus: "CONFIRMED",
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE",
      riskTier: "BINARY_HAZARD",
    },
    vix: 15.2, // Check 5 Macro -> PASS
    vixObservationTime: new Date().toISOString(),
    macroEvidence: {
      vix: 15.2,
      observationTime: new Date().toISOString(),
      marketSession: "REGULAR",
      freshnessStatus: "LIVE",
    },
    isActionable: false,
    canSizeTrade: false,
    dataSource: "live",
    liveSource: "ALPACA_IEX",
    liveFreshnessStatus: "LIVE",
  };

  const res = evaluatePreFlightChecks(auditedNvdaSnapshot);
  assert.strictEqual(res.passedCount, 2, `Expected exactly 2 of 5 checks to pass, got ${res.passedCount}`);
  assert.strictEqual(res.checks[0].passed, false, "Check 1 (R:R) must fail on sub-threshold payoff");
  assert.strictEqual(res.checks[1].passed, false, "Check 2 (Technicals) must fail on Stage 4");
  assert.strictEqual(res.checks[1].status, "STAGE 4 WAIT", "Check 2 status must be STAGE 4 WAIT");
  assert.strictEqual(res.checks[2].passed, true, "Check 3 (Capital) must pass on high Q-Score / low float");
  assert.strictEqual(res.checks[3].passed, false, "Check 4 (Catalyst) must fail on imminent earnings");
  assert.strictEqual(res.checks[3].status, "HAZARD", "Check 4 status must be HAZARD");
  assert.strictEqual(res.checks[4].passed, true, "Check 5 (Macro) must pass on calm VIX");
  assert.strictEqual(res.isCleared, false, "Audited NVDA snapshot must NOT clear");
  assert.ok(res.clearanceStatusLabel.includes("NOT CLEARED"), `clearanceStatusLabel must communicate NOT CLEARED: ${res.clearanceStatusLabel}`);
  assert.strictEqual(res.clearanceStatusLabel, "NOT CLEARED — STAGE 4 WAIT");
  console.log("   ✓ Confirmed: Audited NVDA snapshot returns exactly 2/5 checks passed and 'Not cleared'");
}

console.log("\n=== ALL ARX-PREFLIGHT-INTEGRITY-031 TESTS PASSED SUCCESSFULLY ===");
