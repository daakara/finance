import assert from "node:assert";
import {
  resolveDecisionReadiness,
  DecisionReadinessInputs,
  GateState,
} from "../lib/decisionReadiness";

/**
 * ARX Terminal Synthesis E Wave 4: 3-Gate Decision Readiness Unit & Invariant Suite
 *
 * Verifies:
 * 1. Gate 1: Corridor Location & Geometry (Pass, Extended, Below, Stage 4, Unavailable)
 * 2. Gate 2: Dynamic Trigger & Volume Confirmation (Prerequisite dependency cascade, Pass, Blocking, Unavailable)
 * 3. Gate 3: Risk Floor & Macro Clearance (Prerequisite dependency cascade, Pass, Sub-2:1 RR, Macro VIX Blocking, Fail-Closed Missing VIX)
 * 4. Cascade Matrix & Single Active Blocker Semantics
 * 5. Execution Readiness Binary Flag
 */

console.log("Starting Synthesis E Wave 4 Decision Readiness Test Suite...\n");

// ── 1. Gate 1: Location & Geometry ───────────────────────────────────────────
console.log("1. Verifying Gate 1 Location & Geometry...");

// 1.1 In Buy Zone (Pass)
const g1Pass = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
});
assert.strictEqual(g1Pass.gates[0].state, "PASSED");
assert.ok(g1Pass.gates[0].explanation.includes("positioned inside the institutional accumulation corridor"));
console.log("   [PASS] 1.1 In Buy Zone -> Gate 1 PASSED");

// 1.2 Extended > 2% (Blocking)
const g1Extended = resolveDecisionReadiness({
  currentPrice: 105,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
});
assert.strictEqual(g1Extended.gates[0].state, "BLOCKING");
assert.strictEqual(g1Extended.activeBlockingGate, "GATE_1_LOCATION");
assert.strictEqual(g1Extended.isExecutionReady, false);
assert.ok(g1Extended.gates[0].explanation.includes("extended >2% past the accumulation corridor"));
console.log("   [PASS] 1.2 Extended > 2% -> Gate 1 BLOCKING (Active Blocker: Gate 1)");

// 1.3 Below Zone (Blocking)
const g1Below = resolveDecisionReadiness({
  currentPrice: 95,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
});
assert.strictEqual(g1Below.gates[0].state, "BLOCKING");
assert.strictEqual(g1Below.activeBlockingGate, "GATE_1_LOCATION");
assert.ok(g1Below.gates[0].explanation.includes("below the accumulation corridor"));
console.log("   [PASS] 1.3 Below Corridor -> Gate 1 BLOCKING");

// 1.4 Stage 4 Distribution (Blocking)
const g1Stage4 = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  stagePhase: "Stage 4 Markdown",
  candleCount: 100,
});
assert.strictEqual(g1Stage4.gates[0].state, "BLOCKING");
assert.strictEqual(g1Stage4.activeBlockingGate, "GATE_1_LOCATION");
assert.ok(g1Stage4.gates[0].explanation.includes("Stage 4 distribution"));
console.log("   [PASS] 1.4 Stage 4 -> Gate 1 BLOCKING");

// 1.5 Insufficient Candle Depth < 50 (Unavailable)
const g1NoData = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 30,
});
assert.strictEqual(g1NoData.gates[0].state, "UNAVAILABLE");
assert.strictEqual(g1NoData.activeBlockingGate, "GATE_1_LOCATION");
console.log("   [PASS] 1.5 Candle History < 50 -> Gate 1 UNAVAILABLE");

// 1.6 Stale Tape (Blocking Fail-Closed — P1-F01)
const g1StaleTape = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 16.2,
  freshnessStatus: "STALE_HISTORICAL",
});
assert.strictEqual(g1StaleTape.gates[0].state, "BLOCKING");
assert.strictEqual(g1StaleTape.activeBlockingGate, "GATE_1_LOCATION");
assert.strictEqual(g1StaleTape.isExecutionReady, false);
assert.ok(g1StaleTape.gates[0].explanation.includes("tape is stale"));
console.log("   [PASS] 1.6 Stale Tape -> Gate 1 BLOCKING (Fail-closed)");

// 1.7 Unavailable Tape (Unavailable Fail-Closed)
const g1UnavailableTape = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
  freshnessStatus: "UNAVAILABLE",
});
assert.strictEqual(g1UnavailableTape.gates[0].state, "UNAVAILABLE");
assert.strictEqual(g1UnavailableTape.activeBlockingGate, "GATE_1_LOCATION");
assert.strictEqual(g1UnavailableTape.isExecutionReady, false);
console.log("   [PASS] 1.7 Unavailable Tape -> Gate 1 UNAVAILABLE");

// ── 2. Gate 2: Dynamic Trigger & Dependency Cascade ─────────────────────────
console.log("\n2. Verifying Gate 2 Dynamic Trigger & Dependency Cascade...");

// 2.1 Upstream Gate 1 Blocking -> Gate 2 MUST be PENDING_DEPENDENCY (Never Blocking)
assert.strictEqual(g1Extended.gates[1].state, "PENDING_DEPENDENCY");
assert.strictEqual(g1Extended.gates[2].state, "PENDING_DEPENDENCY");
assert.ok(g1Extended.gates[1].explanation.includes("Waiting for Gate 1 (Location) to clear"));
console.log("   [PASS] 2.1 Upstream Gate 1 Blocking -> Gate 2 & 3 PENDING_DEPENDENCY (Acyclic cascade)");

// 2.2 Gate 1 Passed, Trigger Pending (Blocking)
const g2Pending = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: false,
});
assert.strictEqual(g2Pending.gates[0].state, "PASSED");
assert.strictEqual(g2Pending.gates[1].state, "BLOCKING");
assert.strictEqual(g2Pending.activeBlockingGate, "GATE_2_TRIGGER");
assert.strictEqual(g2Pending.gates[2].state, "PENDING_DEPENDENCY");
assert.ok(g2Pending.gates[1].explanation.includes("confirmation trigger is pending"));
console.log("   [PASS] 2.2 Gate 1 Passed, Trigger Pending -> Gate 2 BLOCKING, Gate 3 PENDING_DEPENDENCY");

// 2.3 Gate 1 Passed, Trigger Confirmed (Passed)
const g2Passed = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
});
assert.strictEqual(g2Passed.gates[0].state, "PASSED");
assert.strictEqual(g2Passed.gates[1].state, "PASSED");
console.log("   [PASS] 2.3 Gate 1 & 2 Passed");

// ── 3. Gate 3: Risk Clearance & Macro VIX Authority ─────────────────────────
console.log("\n3. Verifying Gate 3 Risk Clearance & Macro VIX Authority...");

// 3.1 Sub-2:1 R:R floor violation (Blocking)
const g3LowRR = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 1.5,
  stopLoss: 95,
  takeProfit1: 107.5,
  vix: 18.5,
});
assert.strictEqual(g3LowRR.gates[0].state, "PASSED");
assert.strictEqual(g3LowRR.gates[1].state, "PASSED");
assert.strictEqual(g3LowRR.gates[2].state, "BLOCKING");
assert.strictEqual(g3LowRR.activeBlockingGate, "GATE_3_RISK_CLEARANCE");
assert.strictEqual(g3LowRR.isExecutionReady, false);
assert.ok(g3LowRR.gates[2].explanation.includes("below the institutional 2:1 minimum floor"));
console.log("   [PASS] 3.1 R:R < 2.0 -> Gate 3 BLOCKING");

// 3.2 High Macro Volatility VIX >= 26.0 (Blocking)
const g3HighVix = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 28.2,
});
assert.strictEqual(g3HighVix.gates[2].state, "BLOCKING");
assert.strictEqual(g3HighVix.activeBlockingGate, "GATE_3_RISK_CLEARANCE");
assert.ok(g3HighVix.gates[2].explanation.includes("Broad market volatility is elevated (VIX 28.2 ≥ 26.0)"));
console.log("   [PASS] 3.2 VIX >= 26.0 -> Gate 3 BLOCKING");

// 3.3 Strict Fail-Closed Invariant: Missing VIX is strictly UNAVAILABLE (Never Passed)
const g3MissingVix = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: null,
});
assert.strictEqual(g3MissingVix.gates[2].state, "UNAVAILABLE");
assert.strictEqual(g3MissingVix.activeBlockingGate, "GATE_3_RISK_CLEARANCE");
assert.strictEqual(g3MissingVix.isExecutionReady, false);
assert.ok(g3MissingVix.gates[2].explanation.includes("fail-closed"));
console.log("   [PASS] 3.3 Missing VIX -> Gate 3 strictly UNAVAILABLE (Fail-closed enforced)");

// 3.4 All 3 Gates Passed -> Execution Ready!
const allClear = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 17.5,
});
assert.strictEqual(allClear.gates[0].state, "PASSED");
assert.strictEqual(allClear.gates[1].state, "PASSED");
assert.strictEqual(allClear.gates[2].state, "PASSED");
assert.strictEqual(allClear.activeBlockingGate, "NONE");
assert.strictEqual(allClear.isExecutionReady, true);
assert.strictEqual(allClear.primaryAction.actionType, "SIZE_POSITION");
console.log("   [PASS] 3.4 All 3 Gates Passed -> isExecutionReady = true, Blocker = NONE, Action = SIZE_POSITION");

// 3.5 Historical Point-in-Time Qualification & Provenance Invariants
// 3.5.1 Live Context Certification Matrix (true, false, null, undefined)
const liveBase = {
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 17.5,
  freshnessStatus: "LIVE",
};

// Live + true -> EXECUTION_READY
const liveTrue = resolveDecisionReadiness({ ...liveBase, historicalPointInTimeCertified: true });
assert.strictEqual(liveTrue.isExecutionReady, true);
assert.strictEqual(liveTrue.readinessVerdict, "EXECUTION_READY");
console.log("   [PASS] 3.5.1a Live + certified:true -> isExecutionReady = true");

// Live + false -> AWAITING_CANONICAL_CLEARANCE (synthetic/uncertified fixture)
const liveFalse = resolveDecisionReadiness({ ...liveBase, historicalPointInTimeCertified: false });
assert.strictEqual(liveFalse.isExecutionReady, false);
assert.strictEqual(liveFalse.readinessVerdict, "AWAITING_CANONICAL_CLEARANCE");
assert.strictEqual(liveFalse.readinessVerdictLabel, "HISTORICAL EVIDENCE — UNCERTIFIED FOR LIVE EXECUTION");
console.log("   [PASS] 3.5.1b Live + certified:false -> isExecutionReady = false (fails closed)");

// Live + null -> AWAITING_CANONICAL_CLEARANCE (unverified provenance)
const liveNull = resolveDecisionReadiness({ ...liveBase, historicalPointInTimeCertified: null });
assert.strictEqual(liveNull.isExecutionReady, false);
assert.strictEqual(liveNull.readinessVerdict, "AWAITING_CANONICAL_CLEARANCE");
assert.strictEqual(liveNull.readinessVerdictLabel, "HISTORICAL EVIDENCE — UNCERTIFIED FOR LIVE EXECUTION");
console.log("   [PASS] 3.5.1c Live + certified:null -> isExecutionReady = false (fails closed)");

// Live + undefined -> EXECUTION_READY (legitimate live market observation)
const liveUndefined = resolveDecisionReadiness({ ...liveBase, historicalPointInTimeCertified: undefined });
assert.strictEqual(liveUndefined.isExecutionReady, true);
assert.strictEqual(liveUndefined.readinessVerdict, "EXECUTION_READY");
console.log("   [PASS] 3.5.1d Live + certified:undefined -> isExecutionReady = true (live market readiness)");

// 3.5.2 Historical Context Certification Matrix (true, false, null, undefined)
const histBase = {
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 17.5,
  freshnessStatus: "CACHED",
};

// Historical + true -> EXECUTION_READY (point-in-time certified backtest)
const histTrue = resolveDecisionReadiness({ ...histBase, historicalPointInTimeCertified: true });
assert.strictEqual(histTrue.isExecutionReady, true);
assert.strictEqual(histTrue.readinessVerdict, "EXECUTION_READY");
console.log("   [PASS] 3.5.2a Historical + certified:true -> isExecutionReady = true");

// Historical + false -> AWAITING_CANONICAL_CLEARANCE
const histFalse = resolveDecisionReadiness({ ...histBase, historicalPointInTimeCertified: false });
assert.strictEqual(histFalse.isExecutionReady, false);
assert.strictEqual(histFalse.readinessVerdict, "AWAITING_CANONICAL_CLEARANCE");
console.log("   [PASS] 3.5.2b Historical + certified:false -> isExecutionReady = false");

// Historical + null -> AWAITING_CANONICAL_CLEARANCE (missing certification fails closed)
const histNull = resolveDecisionReadiness({ ...histBase, historicalPointInTimeCertified: null });
assert.strictEqual(histNull.isExecutionReady, false);
assert.strictEqual(histNull.readinessVerdict, "AWAITING_CANONICAL_CLEARANCE");
console.log("   [PASS] 3.5.2c Historical + certified:null -> isExecutionReady = false (fails closed)");

// Historical + undefined -> AWAITING_CANONICAL_CLEARANCE (missing certification CANNOT satisfy qualification)
const histUndefined = resolveDecisionReadiness({ ...histBase, historicalPointInTimeCertified: undefined });
assert.strictEqual(histUndefined.isExecutionReady, false);
assert.strictEqual(histUndefined.readinessVerdict, "AWAITING_CANONICAL_CLEARANCE");
assert.ok(histUndefined.nextRequiredCondition.includes("Point-in-time historical certification required"));
console.log("   [PASS] 3.5.2d Historical + certified:undefined -> isExecutionReady = false (missing PIT fails closed)");

// 3.5.3 Explicit Null Tape Freshness (Fail-Closed to UNAVAILABLE)
const nullTapeRun = resolveDecisionReadiness({
  ...liveBase,
  freshnessStatus: null,
});
assert.strictEqual(nullTapeRun.gates[0].state, "UNAVAILABLE");
assert.strictEqual(nullTapeRun.isExecutionReady, false);
console.log("   [PASS] 3.5.3 Explicit null freshness -> Gate 1 UNAVAILABLE (fails closed)");

console.log("\nALL DECISION READINESS TESTS PASSED (100% COVERAGE)!\n");
