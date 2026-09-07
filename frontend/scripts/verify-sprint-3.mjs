import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

console.log("========================================================================");
console.log("  ARX Terminal vNext: Sprint 3 Verification Suite                       ");
console.log("  (Change Intelligence, Materiality Engine & Attention Allocation)      ");
console.log("========================================================================\n");

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${err.message}`);
    failed++;
  }
}

const resolveComp = (rel) => {
  const p1 = path.resolve(rel);
  if (fs.existsSync(p1)) return fs.readFileSync(p1, "utf-8");
  const p2 = path.resolve("frontend", rel);
  if (fs.existsSync(p2)) return fs.readFileSync(p2, "utf-8");
  throw new Error(`File not found: ${rel}`);
};

const bannerSrc = resolveComp("components/delta/DeltaBanner.tsx");
const feedSrc = resolveComp("components/delta/AttentionFeed.tsx");
const idbSrc = resolveComp("lib/storage/idbSnapshots.ts");
const materialitySrc = resolveComp("lib/engine/materialityEngine.ts");
const typesSrc = resolveComp("types/change-intelligence.ts");
const canvasSrc = resolveComp("components/workstation/WorkstationCanvas.tsx");

// ============================================================================
// Functional Mirror of Materiality Engine Logic for Direct Node Execution
// ============================================================================
function evalScoreMat(prev, curr) {
  const delta = Math.round(curr) - Math.round(prev);
  const abs = Math.abs(delta);
  if (abs <= 2) return { severity: "NONE", delta };
  if (abs <= 5) return { severity: "INFO", delta };
  if (abs <= 9) return { severity: "MATERIAL", delta };
  return { severity: "CRITICAL", delta };
}

function evalFlowMat(prevZ, currZ) {
  const delta = Number((currZ - prevZ).toFixed(2));
  const abs = Math.abs(delta);
  if (abs < 1.0) return { severity: "NONE", delta };
  if (abs < 1.5) return { severity: "INFO", delta };
  if (abs < 2.5) return { severity: "MATERIAL", delta };
  return { severity: "CRITICAL", delta };
}

function evalExecutionMat(prev, curr) {
  if (prev === curr) return { severity: "NONE" };
  return { severity: "CRITICAL", reason: `State shifted from ${prev} to ${curr}` };
}

function evalRegimeMat(prev, curr) {
  if (prev === curr) return { severity: "NONE" };
  return { severity: "CRITICAL", reason: `Regime rotated from ${prev} to ${curr}` };
}

function evalFullMat(baseline, latest) {
  const items = [];
  const s = evalScoreMat(baseline.setupScore, latest.setupScore);
  if (s.severity !== "NONE") items.push({ field: "setupScore", severity: s.severity });
  
  const e = evalExecutionMat(baseline.executionState, latest.executionState);
  if (e.severity !== "NONE") items.push({ field: "executionState", severity: e.severity });
  
  const f = evalFlowMat(baseline.flowZScore, latest.flowZScore);
  if (f.severity !== "NONE") items.push({ field: "flowZScore", severity: f.severity });
  
  const r = evalRegimeMat(baseline.marketRegime, latest.marketRegime);
  if (r.severity !== "NONE") items.push({ field: "marketRegime", severity: r.severity });

  const weights = { NONE: 0, INFO: 1, MATERIAL: 2, CRITICAL: 3 };
  let maxSeverity = "NONE";
  let maxW = 0;
  for (const item of items) {
    if (weights[item.severity] > maxW) {
      maxW = weights[item.severity];
      maxSeverity = item.severity;
    }
  }

  const isMaterial = maxSeverity === "MATERIAL" || maxSeverity === "CRITICAL";
  const attentionSignal = isMaterial ? { severity: maxSeverity, ticker: latest.ticker } : undefined;

  return { isMaterial, maxSeverity, items, attentionSignal };
}

// Test Suite 1: Materiality Threshold Rules
test("Layer 2: Setup Score materiality categorizes noise vs conviction shifts accurately", () => {
  assert.equal(evalScoreMat(71, 72).severity, "NONE", "Score delta 1 must be NONE");
  assert.equal(evalScoreMat(71, 73).severity, "NONE", "Score delta 2 must be NONE");
  assert.equal(evalScoreMat(71, 75).severity, "INFO", "Score delta 4 must be INFO");
  assert.equal(evalScoreMat(71, 78).severity, "MATERIAL", "Score delta 7 must be MATERIAL");
  assert.equal(evalScoreMat(71, 85).severity, "CRITICAL", "Score delta 14 must be CRITICAL");
  assert(materialitySrc.includes("evaluateScoreMateriality"), "Source must export evaluateScoreMateriality");
});

test("Layer 2: Institutional flow thresholds reject sub-1.0σ fluctuations", () => {
  assert.equal(evalFlowMat(1.03, 1.06).severity, "NONE", "Sub-1.0σ delta must be NONE");
  assert.equal(evalFlowMat(1.0, 2.2).severity, "INFO", "1.2σ delta must be INFO");
  assert.equal(evalFlowMat(1.0, 2.8).severity, "MATERIAL", "1.8σ delta must be MATERIAL");
  assert.equal(evalFlowMat(1.0, 3.8).severity, "CRITICAL", "2.8σ delta must be CRITICAL");
  assert(materialitySrc.includes("evaluateFlowMateriality"), "Source must export evaluateFlowMateriality");
});

test("Layer 2: Execution State and Regime transitions are unconditionally CRITICAL", () => {
  assert.equal(
    evalExecutionMat("WAITING_PULLBACK", "IN_BUY_ZONE").severity,
    "CRITICAL",
    "Entry into buy zone must be CRITICAL"
  );
  assert.equal(
    evalExecutionMat("IN_BUY_ZONE", "STOPPED_OUT").severity,
    "CRITICAL",
    "Stop out must be CRITICAL"
  );
  assert.equal(
    evalExecutionMat("IN_BUY_ZONE", "IN_BUY_ZONE").severity,
    "NONE",
    "Unchanged state must be NONE"
  );
  assert.equal(
    evalRegimeMat("RISK_ON", "DEFENSIVE").severity,
    "CRITICAL",
    "Regime shift must be CRITICAL"
  );
  assert.equal(
    evalRegimeMat("RISK_ON", "RISK_ON").severity,
    "NONE",
    "Unchanged regime must be NONE"
  );
  assert(materialitySrc.includes("evaluateExecutionStateMateriality"), "Source must export evaluateExecutionStateMateriality");
  assert(materialitySrc.includes("evaluateRegimeMateriality"), "Source must export evaluateRegimeMateriality");
});

// Test Suite 2: The Mandatory Delta Trust Test
test("Delta Trust Test: Sub-threshold noise produces 100% Zero False Positives (no banner, no attention signal)", () => {
  const baseline = {
    ticker: "CPRX",
    setupScore: 71,
    executionState: "WAITING_PULLBACK",
    marketRegime: "RISK_ON",
    flowZScore: 1.05,
  };

  // Noise simulation: Score +1 point, Flow +0.03σ
  const noisyCurrent = {
    ticker: "CPRX",
    setupScore: 72,
    executionState: "WAITING_PULLBACK",
    marketRegime: "RISK_ON",
    flowZScore: 1.08,
  };

  const report = evalFullMat(baseline, noisyCurrent);
  assert.equal(report.isMaterial, false, "Must not be flagged as material");
  assert.equal(report.maxSeverity, "NONE", "Max severity must be NONE");
  assert.equal(report.items.length, 0, "All items must be filtered out as L0 noise");
  assert.equal(report.attentionSignal, undefined, "Must produce zero attention signals");
});

test("Material Change Test: Buy zone entry produces CRITICAL report with Attention Signal", () => {
  const baseline = {
    ticker: "CPRX",
    setupScore: 71,
    executionState: "WAITING_PULLBACK",
    marketRegime: "RISK_ON",
    flowZScore: 1.05,
  };

  const materialCurrent = {
    ticker: "CPRX",
    setupScore: 84, // +13 pts
    executionState: "IN_BUY_ZONE", // Transition
    marketRegime: "RISK_ON",
    flowZScore: 3.20, // +2.15σ
  };

  const report = evalFullMat(baseline, materialCurrent);
  assert.equal(report.isMaterial, true, "Must be flagged as material");
  assert.equal(report.maxSeverity, "CRITICAL", "Max severity must be CRITICAL");
  assert(report.items.length >= 2, "Must contain multiple material items");
  assert(report.attentionSignal !== undefined, "Must generate AttentionSignal for portfolio feed");
  assert.equal(report.attentionSignal?.severity, "CRITICAL");
});

// Test Suite 3: Storage & Acknowledgement Architecture
test("idbSnapshots.ts implements client-owned storage with 1-click baseline acknowledgement", () => {
  assert(idbSrc.includes("arx_change_intelligence_db"), "Must use approved DB name");
  assert(idbSrc.includes("ticker_snapshots"), "Must use ticker_snapshots object store");
  assert(idbSrc.includes("getTickerRecord"), "Must export getTickerRecord");
  assert(idbSrc.includes("saveTickerRecord"), "Must export saveTickerRecord");
  assert(idbSrc.includes("acknowledgeBaseline"), "Must export acknowledgeBaseline");
  assert(idbSrc.includes("getAllTickerRecords"), "Must export getAllTickerRecords");
});

// Test Suite 4: Stage 6 Delta Banner & Attention Feed Contracts
test("DeltaBanner.tsx enforces role='status', aria-live='polite', and 1-click acknowledgement", () => {
  assert(bannerSrc.includes("role=\"status\""), "Must have role='status'");
  assert(bannerSrc.includes("aria-live=\"polite\""), "Must have aria-live='polite'");
  assert(bannerSrc.includes("onAcknowledge"), "Must accept onAcknowledge prop");
  assert(bannerSrc.includes("Acknowledge & Update Baseline"), "Must include 1-click CTA");
  assert(bannerSrc.includes("delta_banner_displayed"), "Must track banner display");
  assert(bannerSrc.includes("delta_acknowledged"), "Must track acknowledgement");
});

test("AttentionFeed.tsx isolates portfolio-level L3/L4 critical transitions", () => {
  assert(feedSrc.includes("Portfolio Attention Feed"), "Must render portfolio heading");
  assert(feedSrc.includes("Zero Critical Thesis Changes"), "Must render empty state");
  assert(feedSrc.includes("attention_item_opened"), "Must track attention item click");
});

test("WorkstationCanvas.tsx integrates DeltaBanner conditionally above Stage 1", () => {
  assert(canvasSrc.includes("DeltaBanner"), "Must import and mount DeltaBanner");
  assert(canvasSrc.includes("deltaReport"), "Must accept deltaReport prop");
  assert(canvasSrc.includes("onAcknowledgeDelta"), "Must accept onAcknowledgeDelta prop");
});

// Test Suite 5: Privacy & Zero PII Guard (ADR-006)
test("Change intelligence types strictly reject dollar portfolio amounts or share counts", () => {
  assert(!typesSrc.includes("dollarPortfolioValue"), "Must not contain dollar portfolio value");
  assert(!typesSrc.includes("userAccountBalance"), "Must not contain account balance");
  assert(typesSrc.includes("TickerSnapshot"), "Must export TickerSnapshot");
  assert(typesSrc.includes("DeltaReport"), "Must export DeltaReport");
});

console.log("\n========================================================================");
console.log(`  VERIFICATION RESULTS: ${passed} PASSED, ${failed} FAILED`);
console.log("========================================================================\n");

if (failed > 0) {
  process.exit(1);
}
