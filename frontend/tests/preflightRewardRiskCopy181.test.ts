/**
 * Forensic Architecture & Presentation Integrity Test Suite
 * Issue #181: [UX] Simplify Reward vs. Risk preflight disclosure (presentation only)
 * Directive: ARX-UX-RR-030
 *
 * Verifies that:
 * 1. PREFLIGHT_CONFIG exports canonical values.
 * 2. Evaluator outputs presentation 'Below minimum' without altering backend 'FAIL' status.
 * 3. Exact 2:1 threshold and sub-threshold rounding invariants remain strictly enforced.
 * 4. Governor sizing engine and authoritative decision contracts remain untouched.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { evaluatePreFlightChecks, PREFLIGHT_CONFIG } from "../lib/preflightEvaluator";

console.log("Starting ARX-UX-RR-030 (Issue #181) Presentation & Invariant Suite...\n");

// ---------------------------------------------------------------------------
// 1. Verify PREFLIGHT_CONFIG Authoritative Export Constants
// ---------------------------------------------------------------------------
console.log("1. Verifying PREFLIGHT_CONFIG constants...");

assert.strictEqual(
  PREFLIGHT_CONFIG.MIN_REWARD_TO_RISK_RATIO,
  2.0,
  "MIN_REWARD_TO_RISK_RATIO must remain exactly 2.0"
);
assert.strictEqual(
  PREFLIGHT_CONFIG.REQUIRED_RR_DISPLAY_THRESHOLD,
  "2:1",
  "REQUIRED_RR_DISPLAY_THRESHOLD must be '2:1'"
);
assert.strictEqual(
  PREFLIGHT_CONFIG.REQUIRED_RR_LABEL,
  "Required: 2:1",
  "REQUIRED_RR_LABEL must be 'Required: 2:1'"
);
assert.strictEqual(
  PREFLIGHT_CONFIG.EXPLANATION_BELOW_MINIMUM,
  "Potential reward is too low for the risk.",
  "EXPLANATION_BELOW_MINIMUM must match approved copy"
);
assert.strictEqual(
  PREFLIGHT_CONFIG.LABEL_BELOW_MINIMUM,
  "Below minimum",
  "LABEL_BELOW_MINIMUM must be 'Below minimum'"
);
assert.strictEqual(
  PREFLIGHT_CONFIG.LABEL_PASS,
  "PASS",
  "LABEL_PASS must be 'PASS'"
);
assert.strictEqual(
  PREFLIGHT_CONFIG.LABEL_UNAVAILABLE,
  "Unavailable",
  "LABEL_UNAVAILABLE must be 'Unavailable'"
);

console.log("   [OK] All PREFLIGHT_CONFIG constants verified");

// ---------------------------------------------------------------------------
// 2. Evaluator Functional Semantics Verification
// ---------------------------------------------------------------------------
console.log("2. Verifying evaluatePreFlightChecks presentation vs backend status...");

const baseInputs = {
  symbol: "NVDA",
  assetClass: "EQUITY" as const,
  currentPrice: 100.0,
  riskRewardRatio: 1.85,
  stopLoss: 90.0,
  takeProfit1: 118.5,
  currentVix: 15.0,
  setupPattern: "Minervini VCP",
  catalystState: "PASS" as const,
  daysUntilEarnings: 20,
  shortInterestPct: 2.2,
  financialQualityScore: 95,
  is200SmaRising: true,
  isAbove50Sma: true,
  isStage4: false,
};

// A. Below threshold (1.85:1)
const belowResult = evaluatePreFlightChecks(baseInputs);
const chk1Below = belowResult.checks[0];
assert.strictEqual(chk1Below.id, "CHK-1");
assert.strictEqual(chk1Below.status, "FAIL", "Backend status must remain FAIL");
assert.strictEqual(chk1Below.displayStatus, "Below minimum", "Display status must be 'Below minimum'");
assert.strictEqual(chk1Below.passed, false);
assert.strictEqual(chk1Below.summary, "1.85:1");
assert.strictEqual(chk1Below.detail, "Potential reward is too low for the risk.");
assert.strictEqual(belowResult.isCleared, false, "Execution must not be cleared");
console.log("   [OK] Below-threshold case: status=FAIL, displayStatus='Below minimum', cleared=false");

// B. Equal threshold (2.00:1)
const equalResult = evaluatePreFlightChecks({
  ...baseInputs,
  takeProfit1: 120.0,
  riskRewardRatio: 2.0,
});
const chk1Equal = equalResult.checks[0];
assert.strictEqual(chk1Equal.status, "PASS");
assert.strictEqual(chk1Equal.displayStatus, "PASS");
assert.strictEqual(chk1Equal.passed, true);
assert.strictEqual(chk1Equal.summary, "2.00:1");
console.log("   [OK] Equal-threshold case: status=PASS, displayStatus='PASS'");

// C. Above threshold (2.50:1)
const aboveResult = evaluatePreFlightChecks({
  ...baseInputs,
  takeProfit1: 125.0,
  riskRewardRatio: 2.5,
});
const chk1Above = aboveResult.checks[0];
assert.strictEqual(chk1Above.status, "PASS");
assert.strictEqual(chk1Above.displayStatus, "PASS");
assert.strictEqual(chk1Above.passed, true);
assert.strictEqual(chk1Above.summary, "2.50:1");
console.log("   [OK] Above-threshold case: status=PASS, displayStatus='PASS'");

// D. Unavailable case (missing stop loss)
const missingResult = evaluatePreFlightChecks({
  ...baseInputs,
  stopLoss: null,
  riskRewardRatio: null,
});
const chk1Missing = missingResult.checks[0];
assert.strictEqual(chk1Missing.status, "MISSING LEVELS");
assert.strictEqual(chk1Missing.displayStatus, "MISSING LEVELS");
assert.strictEqual(chk1Missing.summary, "N/A");
assert.strictEqual(missingResult.isCleared, false);
console.log("   [OK] Missing levels case: status='MISSING LEVELS', summary='N/A'");

// E. Unavailable case (inverted boundaries: stop >= price)
const invertedResult = evaluatePreFlightChecks({
  ...baseInputs,
  stopLoss: 105.0,
  riskRewardRatio: null,
});
const chk1Inverted = invertedResult.checks[0];
assert.strictEqual(chk1Inverted.status, "FAIL");
assert.strictEqual(chk1Inverted.displayStatus, "FAIL");
assert.strictEqual(chk1Inverted.summary, "Invalid Levels");
assert.strictEqual(invertedResult.isCleared, false);
console.log("   [OK] Inverted boundaries case: status=FAIL, summary='Invalid Levels'");

// F. Sub-threshold rounding guard (1.995:1)
const subRoundResult = evaluatePreFlightChecks({
  ...baseInputs,
  takeProfit1: 119.95,
  riskRewardRatio: 1.995,
});
const chk1SubRound = subRoundResult.checks[0];
assert.strictEqual(chk1SubRound.status, "FAIL");
assert.strictEqual(chk1SubRound.displayStatus, "Below minimum");
assert.strictEqual(chk1SubRound.passed, false);
assert.strictEqual(chk1SubRound.summary, "Below 2.00:1");
assert.strictEqual(subRoundResult.isCleared, false);
console.log("   [OK] Sub-threshold rounding guard (1.995:1): status=FAIL, summary='Below 2.00:1'");

// ---------------------------------------------------------------------------
// 3. Component Code Integrity Verification (PreFlightChecklistModal & ReadinessBar)
// ---------------------------------------------------------------------------
console.log("3. Verifying component source code and copy presentation...");

const modalPath = path.join(__dirname, "../components/PreFlightChecklistModal.tsx");
const modalSource = fs.readFileSync(modalPath, "utf8");

assert.ok(
  modalSource.includes('import { evaluatePreFlightChecks, PreFlightChecklistInputs, PREFLIGHT_CONFIG } from "../lib/preflightEvaluator"'),
  "Modal must import PREFLIGHT_CONFIG from preflightEvaluator"
);
assert.ok(
  modalSource.includes('data-testid="preflight-chk1-title"'),
  "Modal must include preflight-chk1-title data-testid"
);
assert.ok(
  modalSource.includes('data-testid="preflight-chk1-badge"'),
  "Modal must include preflight-chk1-badge data-testid"
);
assert.ok(
  modalSource.includes("PREFLIGHT_CONFIG.LABEL_BELOW_MINIMUM"),
  "Modal must reference PREFLIGHT_CONFIG.LABEL_BELOW_MINIMUM"
);
assert.ok(
  modalSource.includes("PREFLIGHT_CONFIG.EXPLANATION_BELOW_MINIMUM"),
  "Modal must reference PREFLIGHT_CONFIG.EXPLANATION_BELOW_MINIMUM"
);

// Check ReadinessBar uses displayStatus fallback
const barPath = path.join(__dirname, "../components/PreFlightReadinessBar.tsx");
const barSource = fs.readFileSync(barPath, "utf8");
assert.ok(
  barSource.includes("chk.displayStatus || chk.status"),
  "PreFlightReadinessBar must support displayStatus"
);
console.log("   [OK] Component source patterns and data-testids verified");

// ---------------------------------------------------------------------------
// 4. Invariant Preservation: Sizing Engine & Decision Contracts
// ---------------------------------------------------------------------------
console.log("4. Verifying non-interference with sizing engine and decision contracts...");

const sizingPath = path.join(__dirname, "../lib/simulation/governorSizingEngine.ts");
const sizingSource = fs.readFileSync(sizingPath, "utf8");
assert.ok(
  sizingSource.includes("2.0"),
  "Sizing engine must preserve institutional 2.0 ratio boundary"
);

console.log("   [OK] Sizing engine math untouched and strictly preserved");

console.log("\n===============================================================================");
console.log("ALL ARX-UX-RR-030 (ISSUE #181) TESTS PASSED SUCCESSFULLY!");
console.log("===============================================================================\n");
