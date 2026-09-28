import assert from "node:assert";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import { isDecisionActionable, isStatusActionable, ACTIONABLE_EXECUTION_STATUSES } from "../types/decisionContract";

console.log("Starting Wave 1 Fail-Closed Decision Integrity Test Suite...");

// ============================================================================
// 1. F17: Canonical Execution Levels / Removal of Synthetic Multipliers
// ============================================================================
console.log("\n--- Testing F17: Canonical Execution Levels ---");

// Test 1.1: 100 valid candles but missing optimalExecution must yield NO synthetic levels
{
  const mockCandles = Array.from({ length: 100 }, (_, i) => ({
    time: `2026-01-${String(i + 1).padStart(2, '0')}`,
    open: 100,
    high: 105,
    low: 95,
    close: 100,
    volume: 1000000,
  }));

  const insight = generateQuantitativeInsight(
    "TEST",
    "Test Asset",
    100,
    0,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    mockCandles,
    "live",
    undefined,
    undefined,
    undefined // optimalExecution intentionally undefined
  );

  // WatchZone must NOT be fabricated from safePrice * 0.975 / 1.052
  assert.strictEqual(
    insight.standard.keyLevels.watchZone,
    "Unavailable",
    "watchZone must be 'Unavailable' when canonical optimalExecution levels are missing"
  );
  assert.strictEqual(
    insight.human.watchLevels.watchZone,
    "Unavailable",
    "Human tier watchZone must be 'Unavailable' when canonical levels are missing"
  );

  // Targets must be undefined
  assert.strictEqual(
    insight.standard.keyLevels.target1,
    undefined,
    "Target 1 must be undefined when optimalExecution is missing"
  );
  assert.strictEqual(
    insight.standard.keyLevels.target2,
    undefined,
    "Target 2 must be undefined when optimalExecution is missing"
  );
  assert.strictEqual(
    insight.standard.keyLevels.profitRiskRatio,
    undefined,
    "Profit/Risk ratio must be undefined when optimalExecution is missing"
  );

  // Fallback confluence pillars must NOT contain fabricated scores (88, 80, 60)
  const breakdown = insight.standard.confluenceBreakdown;
  for (const pillar of breakdown) {
    assert.strictEqual(
      pillar.score,
      0,
      `Unassessed pillar '${pillar.dimension}' must have score 0, not fabricated score`
    );
  }

  console.log("✓ Test 1.1 Passed: Missing optimalExecution yields Unavailable watchZone and 0-score pillars without synthetic multipliers");
}

// Test 1.2: Canonical optimalExecution levels present -> faithfully consumed
{
  const mockCandles = Array.from({ length: 60 }, (_, i) => ({
    time: `2026-01-${String(i + 1).padStart(2, '0')}`,
    open: 100,
    high: 105,
    low: 95,
    close: 100,
    volume: 1000000,
  }));

  const insight = generateQuantitativeInsight(
    "TEST_CANONICAL",
    "Test Asset",
    100,
    0,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    mockCandles,
    "live",
    undefined,
    undefined,
    {
      optimal_entry_min: 98.50,
      optimal_entry_max: 101.50,
      stop_loss: 94.00,
      take_profit_1: 115.00,
      take_profit_2: 125.00,
      risk_reward_ratio: 2.5,
      execution_status: "IN_BUY_ZONE",
    } as any
  );

  assert.strictEqual(
    insight.standard.keyLevels.watchZone,
    "$98.50 – $101.50",
    "watchZone must faithfully reflect canonical optimal_entry_min and max"
  );
  assert.strictEqual(insight.standard.keyLevels.stopLoss, 94.00);
  assert.strictEqual(insight.standard.keyLevels.target1, 115.00);
  assert.strictEqual(insight.standard.keyLevels.target2, 125.00);
  assert.strictEqual(insight.standard.keyLevels.profitRiskRatio, 2.5);

  console.log("✓ Test 1.2 Passed: Canonical execution levels are faithfully preserved");
}

// ============================================================================
// 2. F11: Execution Status Fail-Closed & Taxonomy Alignment
// ============================================================================
console.log("\n--- Testing F11: Execution Status Fail-Closed ---");

// Test 2.1: Novel / unmapped status must yield non-actionable UNKNOWN
{
  const novelStatus = "SUSPENDED_CIRCUIT_BREAKER";
  assert.strictEqual(
    isStatusActionable(novelStatus),
    false,
    "Novel status must not be actionable"
  );
  assert.strictEqual(
    isDecisionActionable("ACTIONABLE_SETUP", novelStatus),
    false,
    "Novel status must fail closed in isDecisionActionable"
  );

  console.log("✓ Test 2.1 Passed: Novel status strictly non-actionable");
}

// Test 2.2: WAITING_PULLBACK is canonically non-actionable
{
  assert.strictEqual(isStatusActionable("WAITING_PULLBACK"), false);
  assert.strictEqual(isStatusActionable("PULLBACK_SUPPORT"), false);
  assert.strictEqual(isStatusActionable(null), false);
  assert.strictEqual(isStatusActionable(undefined), false);
  assert.strictEqual(isStatusActionable("UNKNOWN"), false);

  console.log("✓ Test 2.2 Passed: Non-actionable statuses verified fail-closed");
}

// ============================================================================
// 3. F08: Fail-Closed Actionability Checks
// ============================================================================
console.log("\n--- Testing F08: Fail-Closed Actionability ---");

{
  // Undefined or missing decisionState must fail closed
  assert.strictEqual(isDecisionActionable(undefined, "IN_BUY_ZONE"), false);
  assert.strictEqual(isDecisionActionable(null, "IN_BUY_ZONE"), false);
  assert.strictEqual(isDecisionActionable("EVIDENCE_INCOMPLETE", "IN_BUY_ZONE"), false);
  assert.strictEqual(isDecisionActionable("VALID_SETUP", "IN_BUY_ZONE"), false);

  // Both ACTIONABLE_SETUP and actionable status required
  assert.strictEqual(isDecisionActionable("ACTIONABLE_SETUP", "IN_BUY_ZONE"), true);
  assert.strictEqual(isDecisionActionable("ACTIONABLE_SETUP", "READY_TO_BUY"), true);
  assert.strictEqual(isDecisionActionable("ACTIONABLE_SETUP", "WAITING_PULLBACK"), false);

  console.log("✓ Test 3.1 Passed: Actionability requires unanimous gate clearance");
}

// ============================================================================
// 4. Source-Level Invariant Checks for F08, F09, F10, F11, F14, F17
// ============================================================================
import fs from "node:fs";
import path from "node:path";

console.log("\n--- Testing Source Invariants (F08, F09, F10, F11, F14, F17) ---");

// F17 & F11: Verify removal of synthetic multipliers and PULLBACK_SUPPORT
{
  const insightSource = fs.readFileSync(path.join(__dirname, "../lib/insightGenerator.ts"), "utf8");
  assert.ok(!insightSource.includes("* 0.975"), "insightGenerator.ts must not contain * 0.975 synthetic multiplier");
  assert.ok(!insightSource.includes("* 1.052"), "insightGenerator.ts must not contain * 1.052 synthetic multiplier");
  assert.ok(!insightSource.includes("(-7.0%)"), "insightGenerator.ts must not hardcode (-7.0%) string");
  console.log("✓ F17 Invariant: Synthetic multipliers completely removed from insightGenerator.ts");

  const radarSource = fs.readFileSync(path.join(__dirname, "../app/radar/page.tsx"), "utf8");
  assert.ok(!radarSource.includes("PULLBACK_SUPPORT"), "radar/page.tsx must not contain PULLBACK_SUPPORT");
  assert.ok(!radarSource.includes("Consolidation Base"), "radar/page.tsx must not default to Consolidation Base");
  assert.ok(radarSource.includes("Base Under Evaluation"), "radar/page.tsx must use Base Under Evaluation");
  assert.ok(radarSource.includes("WAITING_PULLBACK"), "radar/page.tsx must map to canonical WAITING_PULLBACK");
  console.log("✓ F11 & F10 Invariant: Canonical WAITING_PULLBACK and Base Under Evaluation verified in radar/page.tsx");
}

// F08 & F09: Verify fail-closed defaults in modals
{
  const sizerSource = fs.readFileSync(path.join(__dirname, "../components/PositionSizerModal.tsx"), "utf8");
  assert.ok(sizerSource.includes("canSizeTrade = false"), "PositionSizerModal must default canSizeTrade to false");
  assert.ok(sizerSource.includes("isActionable = false"), "PositionSizerModal must default isActionable to false");
  console.log("✓ F08 Invariant: PositionSizerModal defaults canSizeTrade and isActionable to false");

  const preflightSource = fs.readFileSync(path.join(__dirname, "../components/PreFlightChecklistModal.tsx"), "utf8");
  assert.ok(!preflightSource.includes("vix = 15.4"), "PreFlightChecklistModal must not default vix to 15.4");
  assert.ok(preflightSource.includes("isActionable = false"), "PreFlightChecklistModal must default isActionable to false");
  assert.ok(preflightSource.includes("const isActionableGranted = isActionable === true;"), "PreFlightChecklistModal must require strict isActionable === true");
  console.log("✓ F09 & F08 Invariant: PreFlightChecklistModal purged of fabricated VIX default and strictly fail-closed");
}

// F14: Verify userRole propagation across callers
{
  const pageSource = fs.readFileSync(path.join(__dirname, "../app/page.tsx"), "utf8");
  assert.ok(pageSource.includes("fetchAssetAnalytics(selectedSymbol, \"1y\", \"1d\", userRole)"), "page.tsx retry must pass userRole");

  const radarSource = fs.readFileSync(path.join(__dirname, "../app/radar/page.tsx"), "utf8");
  assert.ok(radarSource.includes("fetchAssetAnalytics(clean, \"1y\", \"1d\", \"SWING_TRADER\")"), "radar/page.tsx on-demand must pass SWING_TRADER role");

  const apiSource = fs.readFileSync(path.join(__dirname, "../lib/api.ts"), "utf8");
  assert.ok(apiSource.includes("prefetchAssetAnalytics(symbol: string, period: string = \"1y\", interval: string = \"1d\", userRole: string = \"LONG_TERM\")"), "api.ts prefetchAssetAnalytics must accept and forward userRole");

  const compareSource = fs.readFileSync(path.join(__dirname, "../components/ComparePairMatrix.tsx"), "utf8");
  assert.ok(compareSource.includes("fetchAssetAnalytics(symA, \"1mo\", \"1d\", userRole)"), "ComparePairMatrix must pass userRole to symA");
  assert.ok(compareSource.includes("fetchAssetAnalytics(symB, \"1mo\", \"1d\", userRole)"), "ComparePairMatrix must pass userRole to symB");
  console.log("✓ F14 Invariant: userRole horizon context propagated faithfully to all fetchAssetAnalytics callers");
}

console.log("\n========================================================");
console.log("ALL WAVE 1 DECISION INTEGRITY UNIT & INVARIANT TESTS PASSED!");
console.log("========================================================");
