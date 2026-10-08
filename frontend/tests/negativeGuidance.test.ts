import assert from "node:assert";
import { resolveDecisionReadiness } from "../lib/decisionReadiness";

/**
 * ARX Terminal Synthesis E Wave 4: Epistemic Negative Guidance Test Suite
 *
 * Verifies:
 * 1. CLAIM_SET ⊆ EVIDENCE_SET: Every negative guidance assertion derives strictly
 *    from authentic runtime evidence.
 * 2. All 6 canonical negative guidance rules:
 *    - Extended Price -> "DO NOT CHASE"
 *    - In Buy Zone, Awaiting Trigger -> "DO NOT PRE-EMPT"
 *    - Stage 4 Markdown -> "CAPITAL DEFENSE"
 *    - Sub-2:1 R:R Floor -> "INADEQUATE ASYMMETRY"
 *    - High Macro Volatility (VIX >= 26.0) -> "MACRO CAUTION"
 *    - Missing Macro VIX -> "MACRO DATA DEGRADED"
 * 3. Zero Hallucination: Clean setups display zero negative guidance.
 */

console.log("Starting Synthesis E Wave 4 Epistemic Negative Guidance Test Suite...\n");

// ── 1. Price Extended -> DO NOT CHASE ────────────────────────────────────────
console.log("1. Verifying 'DO NOT CHASE' Negative Guidance...");
const extended = resolveDecisionReadiness({
  currentPrice: 110,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
});
assert.ok(extended.negativeGuidance, "Expected negative guidance for extended price");
assert.ok(extended.negativeGuidance!.startsWith("DO NOT CHASE:"));
assert.ok(extended.negativeGuidance!.includes("extended past the accumulation corridor"));
console.log("   [PASS] 1. Price extended >2% -> 'DO NOT CHASE' banner verified");

// ── 2. In Buy Zone, Awaiting Trigger -> DO NOT PRE-EMPT ──────────────────────
console.log("\n2. Verifying 'DO NOT PRE-EMPT' Negative Guidance...");
const awaitingTrigger = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: false,
  candleCount: 100,
});
assert.ok(awaitingTrigger.negativeGuidance, "Expected negative guidance for unconfirmed trigger");
assert.ok(awaitingTrigger.negativeGuidance!.startsWith("DO NOT PRE-EMPT:"));
assert.ok(awaitingTrigger.negativeGuidance!.includes("confirmation trigger is pending"));
console.log("   [PASS] 2. In buy zone awaiting trigger -> 'DO NOT PRE-EMPT' banner verified");

// ── 3. Stage 4 Distribution -> CAPITAL DEFENSE ───────────────────────────────
console.log("\n3. Verifying 'CAPITAL DEFENSE' Negative Guidance...");
const stage4 = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  stagePhase: "Stage 4 Distribution",
  candleCount: 100,
});
assert.ok(stage4.negativeGuidance, "Expected negative guidance for Stage 4");
assert.ok(stage4.negativeGuidance!.startsWith("CAPITAL DEFENSE:"));
assert.ok(stage4.negativeGuidance!.includes("Stage 4 correction below 50-day SMA"));
console.log("   [PASS] 3. Stage 4 markdown -> 'CAPITAL DEFENSE' banner verified");

// ── 4. Sub-2:1 R:R -> INADEQUATE ASYMMETRY ──────────────────────────────────
console.log("\n4. Verifying 'INADEQUATE ASYMMETRY' Negative Guidance...");
const lowRR = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 1.5,
  stopLoss: 95,
  takeProfit1: 107.5,
  vix: 18.0,
});
assert.ok(lowRR.negativeGuidance, "Expected negative guidance for R:R < 2.0");
assert.ok(lowRR.negativeGuidance!.startsWith("INADEQUATE ASYMMETRY:"));
assert.ok(lowRR.negativeGuidance!.includes("below the 2:1 institutional floor"));
console.log("   [PASS] 4. Prospective R:R < 2.0 -> 'INADEQUATE ASYMMETRY' banner verified");

// ── 5. High Macro Volatility -> MACRO CAUTION ────────────────────────────────
console.log("\n5. Verifying 'MACRO CAUTION' Negative Guidance...");
const highVix = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 28.5,
});
assert.ok(highVix.negativeGuidance, "Expected negative guidance for VIX >= 26.0");
assert.ok(highVix.negativeGuidance!.startsWith("MACRO CAUTION:"));
assert.ok(highVix.negativeGuidance!.includes("VIX 28.5 ≥ 26.0"));
console.log("   [PASS] 5. VIX >= 26.0 -> 'MACRO CAUTION' banner verified");

// ── 6. Missing Macro VIX -> MACRO DATA DEGRADED ─────────────────────────────
console.log("\n6. Verifying 'MACRO DATA DEGRADED' Negative Guidance...");
const missingVix = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: null,
});
assert.ok(missingVix.negativeGuidance, "Expected negative guidance for missing VIX");
assert.ok(missingVix.negativeGuidance!.startsWith("MACRO DATA DEGRADED:"));
assert.ok(missingVix.negativeGuidance!.includes("Live market volatility unavailable"));
console.log("   [PASS] 6. Missing VIX -> 'MACRO DATA DEGRADED' banner verified");

// ── 7. Clean Actionable Setup -> Zero Hallucinated Negative Guidance ─────────
console.log("\n7. Verifying Zero Hallucination on Clean Setups...");
const cleanSetup = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 17.5,
});
assert.strictEqual(cleanSetup.negativeGuidance, null, "Clean actionable setup must NOT display negative guidance!");
console.log("   [PASS] 7. Clean actionable setup has negativeGuidance === null (Zero hallucination)");

console.log("\nALL EPISTEMIC NEGATIVE GUIDANCE TESTS PASSED!\n");
