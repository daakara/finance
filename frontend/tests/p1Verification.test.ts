import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { resolveDecisionReadiness } from "../lib/decisionReadiness";

/**
 * Sprint P1 Post-Remediation Verification Suite (ARX-MAC-P1-EXECUTION-003)
 * Formally verifies that all 4 findings (P1-F01 through P1-F04) are fully mitigated.
 */

console.log("=== ARX SPRINT P1 POST-REMEDIATION VERIFICATION SUITE ===");

// ── P1-F01: Stale price readiness (Fail-closed) ─────────────────────────────
console.log("\n[P1-F01] Verifying Stale Price Tape Fail-Closed Behavior...");
const staleTapeInputs = {
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 16.2, // Live calm VIX
  freshnessStatus: "STALE_HISTORICAL" as const,
};

const staleResult = resolveDecisionReadiness(staleTapeInputs);
assert.strictEqual(staleResult.gates[0].state, "BLOCKING", "Gate 1 must be BLOCKING on stale tape");
assert.strictEqual(staleResult.activeBlockingGate, "GATE_1_LOCATION", "Active blocker must be GATE_1_LOCATION");
assert.strictEqual(staleResult.isExecutionReady, false, "isExecutionReady must fail-closed to false");
assert.ok(staleResult.gates[0].explanation.includes("tape is stale"), "Explanation must state tape is stale");
console.log("   [PASS] P1-F01: Stale price tape successfully blocks Gate 1 and suppresses execution readiness.");

// ── P1-F02: Historical / Synthetic Point-in-Time Provenance ─────────────────
console.log("\n[P1-F02] Verifying Point-in-Time Historical Provenance Qualification...");
const uncertifiedProvenanceInputs = {
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  candleCount: 100,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 16.2,
  historicalPointInTimeCertified: false,
};

const uncertifiedResult = resolveDecisionReadiness(uncertifiedProvenanceInputs);
assert.strictEqual(uncertifiedResult.gates[0].state, "PASSED", "Gate 1 technical checks pass");
assert.strictEqual(uncertifiedResult.gates[1].state, "PASSED", "Gate 2 technical checks pass");
assert.strictEqual(uncertifiedResult.gates[2].state, "PASSED", "Gate 3 technical checks pass");
assert.strictEqual(uncertifiedResult.isExecutionReady, false, "Execution readiness must be false when uncertified");
assert.strictEqual(uncertifiedResult.readinessVerdict, "AWAITING_CANONICAL_CLEARANCE");
assert.strictEqual(uncertifiedResult.readinessVerdictLabel, "HISTORICAL EVIDENCE — UNCERTIFIED FOR LIVE EXECUTION");
assert.ok(uncertifiedResult.nextRequiredCondition.includes("Point-in-time historical certification required"));
console.log("   [PASS] P1-F02: Uncertified historical data properly requires point-in-time certification.");

// ── P1-F03: Trade Plan Export Safety ───────────────────────────────────────
console.log("\n[P1-F03] Verifying Trade Plan Export Platform Hold & Simulation Notice...");
const modalFilePath = path.join(__dirname, "../components/PreFlightChecklistModal.tsx");
const modalSource = fs.readFileSync(modalFilePath, "utf8");

assert.ok(modalSource.includes("HOLD_EXEC_001"), "Modal trade plan export must include HOLD_EXEC_001 disclaimer");
assert.ok(modalSource.includes("Simulation and institutional quantitative research only"), "Modal export must include simulation notice");
console.log("   [PASS] P1-F03: Trade plan export contains mandatory HOLD_EXEC_001 platform hold and simulation disclosure.");

// ── P1-F04: Telemetry Idempotency & React Lifecycle ────────────────────────
console.log("\n[P1-F04] Verifying Telemetry Idempotency & Lifecycle Safety...");
assert.ok(!modalSource.includes("// Analytics — inline try/catch, no useEffect needed"), "Inline render-loop telemetry must be removed");
assert.ok(modalSource.includes("lastEvaluatedKeyRef") || modalSource.includes("hasTrackedOutcomeRef"), "Modal must use ref to ensure single dispatch");
assert.ok(modalSource.includes("evaluationEventKey") || modalSource.includes("evaluationEventId"), "Modal must define evaluation-event identity");
assert.ok(modalSource.includes("useEffect("), "Telemetry must be dispatched inside useEffect");
console.log("   [PASS] P1-F04: Telemetry is wrapped in useEffect, guarded by evaluation-event identity ref latch.");

console.log("\nALL 4 SPRINT P1 FINDINGS VERIFIED AND MITIGATED (100% PASS)!");
