import assert from "node:assert";
import {
  calculateGovernedPositionSize,
  TraderContext,
  TradeSetupSpec,
} from "../lib/simulation/governorSizingEngine";

console.log("Starting Governor Sizing Engine Permanent Regression Suite...");

const baseContext: TraderContext = {
  accountEquity: 500,
  standardRiskBudgetPct: 0.02, // 2% risk = 
  consecutiveLossStreak: 0,
  dailyDrawdownPct: 0,
  tradingHour: 10, // Optimal morning window
  liquidRunwayMonths: 12,
  isAvailable: true,
};

// Test 1: Undersized account ( budget vs  stop risk) -> 0 shares
{
  const setup: TradeSetupSpec = {
    ticker: "TEST",
    setupName: "VCP Breakout",
    entryPivot: 100,
    stopLoss: 75,
    target1: 150,
    target2: 175,
    confluenceScore: 85,
    isActionable: true,
  };

  const res = calculateGovernedPositionSize(setup, baseContext);
  assert.strictEqual(res.recommendedDollarRisk, 10, "Risk budget should be ");
  assert.strictEqual(res.stopDistanceDollar, 25.0, "Stop distance should be ");
  assert.strictEqual(res.rawShares !== undefined ? res.rawShares : res.unclampedShares, 0, "Raw shares must be 0");
  assert.strictEqual(res.recommendedShares, 0, "Recommended shares must be 0 (never force 1 share)");
  assert.ok(
    res.cleanRoomRationale.includes("Position sizing suppressed (0 shares)"),
    "Clean room rationale must explicitly state suppressed 0 shares"
  );
  assert.ok(
    res.cleanRoomRationale.includes("Risk per share exceeds approved risk allowance"),
    "Rationale must note risk per share exceeds allowance"
  );
  console.log("✓ Test 1 Passed: Small account sizing safely suppressed to 0 shares");
}

// Test 2: Boundary test - riskBudget exactly equals riskPerShare ( budget vs  stop risk) -> 1 share
{
  const exactContext: TraderContext = {
    ...baseContext,
    accountEquity: 1250, // 2% of  = 
  };

  const setup: TradeSetupSpec = {
    ticker: "TEST",
    setupName: "VCP Breakout",
    entryPivot: 100,
    stopLoss: 75,
    target1: 150,
    confluenceScore: 85,
    isActionable: true,
  };

  const res = calculateGovernedPositionSize(setup, exactContext);
  assert.strictEqual(res.recommendedDollarRisk, 25, "Risk budget should be ");
  assert.strictEqual(res.stopDistanceDollar, 25.0, "Stop distance should be ");
  assert.strictEqual(res.recommendedShares, 1, "Exactly 1 share when budget equals stop risk");
  console.log("✓ Test 2 Passed: Exact boundary budget yields 1 share");
}

// Test 3: Boundary test - riskBudget just below riskPerShare (.98 budget vs  stop risk) -> 0 shares
{
  const belowContext: TraderContext = {
    ...baseContext,
    accountEquity: 1249, // 2% of 1249 = 24.98 -> Math.round is 25, so let's use exact Math.round
    standardRiskBudgetPct: 0.0199, //  * 0.0199 = 24.85 -> Math.round = 25. Let's make standardDollarRisk = 24
  };
  // Explicitly set account equity to 1200 * 0.02 = 24
  belowContext.accountEquity = 1200;
  belowContext.standardRiskBudgetPct = 0.02; //  budget

  const setup: TradeSetupSpec = {
    ticker: "TEST",
    setupName: "VCP Breakout",
    entryPivot: 100,
    stopLoss: 75,
    target1: 150,
    confluenceScore: 85,
    isActionable: true,
  };

  const res = calculateGovernedPositionSize(setup, belowContext);
  assert.strictEqual(res.recommendedDollarRisk, 24, "Risk budget should be ");
  assert.strictEqual(res.stopDistanceDollar, 25.0, "Stop distance should be ");
  assert.strictEqual(res.recommendedShares, 0, "0 shares when budget () < stop risk ()");
  console.log("✓ Test 3 Passed: Just below boundary budget yields 0 shares");
}

// Test 4: Missing execution levels -> 0 shares and degraded state
{
  const missingSetup: TradeSetupSpec = {
    ticker: "TEST_MISSING",
    setupName: "Unverified Levels",
    entryPivot: 0,
    stopLoss: 0,
    confluenceScore: 70,
    isActionable: false,
    reasonSuppressed: "Missing historical data for pivot points",
  };

  const res = calculateGovernedPositionSize(missingSetup, baseContext);
  assert.strictEqual(res.recommendedShares, 0, "Missing levels must yield 0 shares");
  assert.strictEqual(res.recommendedDollarRisk, 0, "Missing levels must yield  risk");
  assert.strictEqual(res.unclampedShares, 0, "Unclamped shares must be 0");
  assert.strictEqual(res.rMultipleTarget1, 0, "R-multiple target must be 0");
  assert.ok(res.cleanRoomRationale.includes("Missing historical data"), "Rationale must reflect missing data");
  console.log("✓ Test 4 Passed: Missing execution levels yield 0 shares and clean suppression");
}

// Test 5: Inverted execution levels (stop > entry) -> non-actionable, 0 shares
{
  const invertedSetup: TradeSetupSpec = {
    ticker: "TEST_INVERTED",
    setupName: "Invalid Inverted",
    entryPivot: 50,
    stopLoss: 100,
    confluenceScore: 70,
    isActionable: true,
  };

  const res = calculateGovernedPositionSize(invertedSetup, baseContext);
  assert.strictEqual(res.recommendedShares, 0, "Inverted stop must yield 0 shares");
  assert.strictEqual(res.recommendedDollarRisk, 0, "Inverted stop must yield  risk");
  console.log("✓ Test 5 Passed: Inverted stop/entry safely rejects sizing");
}

console.log("ALL 5 GOVERNOR SIZING ENGINE REGRESSION TESTS PASSED!");
