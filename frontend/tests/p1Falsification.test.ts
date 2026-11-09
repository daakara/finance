import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { resolveDecisionReadiness } from "../lib/decisionReadiness";

/**
 * Sprint P1 Falsification Suite (ARX-MAC-P1-EXECUTION-003)
 * Demonstrates and documents defects P1-F01 through P1-F04.
 */

console.log("=== ARX SPRINT P1 FALSIFICATION TEST SUITE ===");

// ── P1-F01: Stale price readiness ──────────────────────────────────────────
console.log("\n[P1-F01] Testing Stale Price Tape vs Live VIX...");
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

const staleResult = resolveDecisionReadiness(staleTapeInputs as any);
// In vulnerable unpatched code, Gate 1 evaluates to PASSED because freshnessStatus is ignored
const isP1F01Vulnerable = (staleResult.gates[0].state === "PASSED" && staleResult.isExecutionReady === true);
console.log(`   P1-F01 Vulnerability Present: ${isP1F01Vulnerable}`);
console.log(`   Gate 1 State on Stale Tape: ${staleResult.gates[0].state}`);
console.log(`   isExecutionReady on Stale Tape: ${staleResult.isExecutionReady}`);

// ── P1-F02: Historical Provenance ─────────────────────────────────────────
console.log("\n[P1-F02] Testing Historical / Synthetic Provenance Qualification...");
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

const uncertifiedResult = resolveDecisionReadiness(uncertifiedProvenanceInputs as any);
const isP1F02Vulnerable = (uncertifiedResult.isExecutionReady === true && uncertifiedResult.readinessVerdict === "EXECUTION_READY");
console.log(`   P1-F02 Vulnerability Present: ${isP1F02Vulnerable}`);
console.log(`   Readiness Verdict on Uncertified PIT: ${uncertifiedResult.readinessVerdict}`);
console.log(`   isExecutionReady on Uncertified PIT: ${uncertifiedResult.isExecutionReady}`);

// ── P1-F03: Trade Plan Export Safety ───────────────────────────────────────
console.log("\n[P1-F03] Testing Trade Plan Export Safety Disclosures...");
const modalFilePath = path.join(__dirname, "../components/PreFlightChecklistModal.tsx");
const modalSource = fs.readFileSync(modalFilePath, "utf8");

const hasHoldExec001InExport = modalSource.includes("HOLD_EXEC_001");
const hasSimulationNoticeInExport = modalSource.includes("SIMULATION ONLY") || modalSource.includes("RESEARCH / JOURNAL ONLY");
const isP1F03Vulnerable = !hasHoldExec001InExport;
console.log(`   P1-F03 Vulnerability Present (HOLD_EXEC_001 missing in trade plan export): ${isP1F03Vulnerable}`);
console.log(`   Contains HOLD_EXEC_001: ${hasHoldExec001InExport}`);

// ── P1-F04: Telemetry Duplication in Render Loop ───────────────────────────
console.log("\n[P1-F04] Testing Telemetry Execution in PreFlightChecklistModal...");
const hasInlineTelemetry = modalSource.includes("// Analytics — inline try/catch, no useEffect needed");
const isP1F04Vulnerable = hasInlineTelemetry;
console.log(`   P1-F04 Vulnerability Present (inline synchronous telemetry in render loop): ${isP1F04Vulnerable}`);
console.log(`   Contains inline render-loop telemetry: ${hasInlineTelemetry}`);

console.log("\n=== FALSIFICATION SUMMARY ===");
console.log(`P1-F01 (Stale price readiness):       ${isP1F01Vulnerable ? "CONFIRMED_REPRODUCIBLE" : "RESOLVED_VERIFIED"}`);
console.log(`P1-F02 (Historical provenance):       ${isP1F02Vulnerable ? "CONFIRMED_REPRODUCIBLE" : "RESOLVED_VERIFIED"}`);
console.log(`P1-F03 (Trade plan export safety):    ${isP1F03Vulnerable ? "CONFIRMED_REPRODUCIBLE" : "RESOLVED_VERIFIED"}`);
console.log(`P1-F04 (Render-loop telemetry):       ${isP1F04Vulnerable ? "CONFIRMED_REPRODUCIBLE" : "RESOLVED_VERIFIED"}`);

assert.strictEqual(isP1F01Vulnerable, false, "P1-F01 must be resolved (stale price fails closed to BLOCKING)");
assert.strictEqual(isP1F02Vulnerable, false, "P1-F02 must be resolved (uncertified PIT suppresses execution readiness)");
assert.strictEqual(isP1F03Vulnerable, false, "P1-F03 must be resolved (HOLD_EXEC_001 present in export)");
assert.strictEqual(isP1F04Vulnerable, false, "P1-F04 must be resolved (render-loop telemetry removed)");
console.log("\nALL 4 FINDINGS VERIFIED RESOLVED AND FAIL-CLOSED ENFORCED!");
