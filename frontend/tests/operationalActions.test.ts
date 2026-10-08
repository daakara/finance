import assert from "node:assert";
import { resolveDecisionReadiness } from "../lib/decisionReadiness";

/**
 * ARX Terminal Synthesis E Wave 4: Operational Actions & Capability Grounding Suite
 *
 * Verifies:
 * 1. USER_VISIBLE_OPERATIONAL_ACTION => REAL_BACKING_CAPABILITY (AC-W4-006).
 * 2. Primary CTA resolution strictly maps to verified real handlers:
 *    - ACTIONABLE_SETUP -> SIZE_POSITION (PositionSizerModal)
 *    - Extended Price -> SET_PULLBACK_ALERT (AlertTriggerModal)
 *    - In Buy Zone, Awaiting Trigger -> SET_BUY_ZONE_ALERT (AlertTriggerModal)
 *    - Stage 4 Correction -> EXPLORE_RADAR (Next.js navigation)
 *    - Sub-2:1 R:R -> EXPLORE_RADAR
 * 3. Fake / ungrounded CTAs strictly prohibited:
 *    - No generic 'Set Breakout Alert' without Stage 4 50-SMA pivot backing.
 *    - No 'Await Market Refresh' rendered as clickable button.
 *    - Zero fake toasts.
 */

console.log("Starting Synthesis E Wave 4 Operational Actions Test Suite...\n");

// ── 1. Actionable Setup -> SIZE_POSITION ─────────────────────────────────────
console.log("1. Verifying Actionable Setup Primary CTA...");
const actionableSetup = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 17.0,
});
assert.strictEqual(actionableSetup.primaryAction.actionType, "SIZE_POSITION");
assert.strictEqual(actionableSetup.primaryAction.label, "Size Position");
console.log("   [PASS] 1. Actionable setup maps directly to SIZE_POSITION (PositionSizerModal)");

// ── 2. Extended Price -> SET_PULLBACK_ALERT ──────────────────────────────────
console.log("\n2. Verifying Extended Price Primary CTA...");
const extendedSetup = resolveDecisionReadiness({
  currentPrice: 110,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: false,
  candleCount: 100,
});
assert.strictEqual(extendedSetup.primaryAction.actionType, "SET_PULLBACK_ALERT");
assert.strictEqual(extendedSetup.primaryAction.label, "Set Pullback Alert");
console.log("   [PASS] 2. Extended price maps to SET_PULLBACK_ALERT (AlertTriggerModal)");

// ── 3. In Buy Zone, Awaiting Trigger -> SET_BUY_ZONE_ALERT ───────────────────
console.log("\n3. Verifying In Buy Zone Awaiting Trigger Primary CTA...");
const inZonePending = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: false,
  candleCount: 100,
});
assert.strictEqual(inZonePending.primaryAction.actionType, "SET_BUY_ZONE_ALERT");
assert.strictEqual(inZonePending.primaryAction.label, "Set Buy Zone Alert");
console.log("   [PASS] 3. In Buy Zone awaiting trigger maps to SET_BUY_ZONE_ALERT (AlertTriggerModal)");

// ── 4. Stage 4 Correction -> EXPLORE_RADAR ───────────────────────────────────
console.log("\n4. Verifying Stage 4 Correction Primary CTA...");
const stage4Setup = resolveDecisionReadiness({
  currentPrice: 85,
  optimalEntryMin: 80,
  optimalEntryMax: 88,
  stagePhase: "Stage 4 Distribution",
  candleCount: 100,
});
assert.strictEqual(stage4Setup.primaryAction.actionType, "EXPLORE_RADAR");
assert.strictEqual(stage4Setup.primaryAction.label, "Explore Radar Setups");
console.log("   [PASS] 4. Stage 4 correction maps to EXPLORE_RADAR (Next.js client route)");

// ── 5. Sub-2:1 R:R Floor Violation -> EXPLORE_RADAR ──────────────────────────
console.log("\n5. Verifying Sub-2:1 R:R Floor Violation Primary CTA...");
const lowRRSetup = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 1.4,
  stopLoss: 95,
  takeProfit1: 107,
  vix: 18.0,
});
assert.strictEqual(lowRRSetup.primaryAction.actionType, "EXPLORE_RADAR");
assert.strictEqual(lowRRSetup.primaryAction.label, "Explore Radar Setups");
console.log("   [PASS] 5. Sub-2:1 R:R maps to EXPLORE_RADAR to preserve capital asymmetry");

// ── 6. Prohibition on Fake CTAs ──────────────────────────────────────────────
console.log("\n6. Verifying Prohibition on Fake / Ungrounded CTAs...");
const prohibitedActionTypes = [
  "SET_BREAKOUT_ALERT", // No generic breakout trigger in AlertManager
  "AWAIT_MARKET_REFRESH", // Status banner only, never an operational action
  "MOCK_TOAST",
  "FAKE_SYNC",
];

for (const prohibited of prohibitedActionTypes) {
  assert.notStrictEqual(actionableSetup.primaryAction.actionType, prohibited);
  assert.notStrictEqual(extendedSetup.primaryAction.actionType, prohibited);
  assert.notStrictEqual(inZonePending.primaryAction.actionType, prohibited);
  assert.notStrictEqual(stage4Setup.primaryAction.actionType, prohibited);
  assert.notStrictEqual(lowRRSetup.primaryAction.actionType, prohibited);
}
console.log("   [PASS] 6. Prohibited fake actions are strictly absent across all states");

console.log("\nALL OPERATIONAL ACTIONS TESTS PASSED!\n");
