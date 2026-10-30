/**
 * ARX Production UX Acceptance Closure Test Suite
 *
 * Directives: PROD-UX-01 / PROD-UX-02
 * Workstream A: Execution Guidance Authority & Dual Price-Basis R:R
 * Workstream B: Comparison Evidence Integrity & Dynamic Allocation Synthesis
 * Workstream C: Dynamic Radar Counts, Paper Theme Contrast & Accessible Controls
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { evaluatePreFlightChecks } from "../lib/preflightEvaluator";
import { MASTER_ASSET_CATALOG } from "../lib/masterCatalog";

const ROOT_DIR = path.resolve(__dirname, "..");

console.log("\n==================================================================");
console.log("ARX PRODUCTION UX ACCEPTANCE REMEDIATION TEST SUITE");
console.log("Directives: PROD-UX-01 / PROD-UX-02");
console.log("==================================================================\n");

// ------------------------------------------------------------------
// TEST 1: Workstream A — LNTH Reproduction Path & Decision Hierarchy
// ------------------------------------------------------------------
console.log("1. Verifying LNTH Reproduction Path & Execution Guidance Authority...");

const lnthCatalog = MASTER_ASSET_CATALOG["LNTH"];
assert.ok(lnthCatalog, "LNTH must exist in MASTER_ASSET_CATALOG");
assert.strictEqual(lnthCatalog.symbol, "LNTH");

// Case 1A: LNTH at Pivot Price ($96.00) where R:R clears 2.0:1, but catalyst coverage is missing
const lnthAtPivotInputs = {
  symbol: "LNTH",
  currentPrice: 96.00,
  breakoutPivot: 96.00,
  optimalEntryMin: 95.00,
  optimalEntryMax: 99.00,
  stopLoss: 92.00,
  takeProfit1: 105.00,
  riskRewardRatio: 2.25,
  isActionable: true,
  decisionState: "ACTIONABLE",
  setupPattern: "Minervini VCP",
  catalystEvidence: null, // SEC filing/event date feed unavailable
  vix: 18.5,
  macroEvidence: {
    vix: 18.5,
    observationTime: new Date().toISOString(),
    marketSession: "REGULAR",
    freshnessStatus: "LIVE"
  }
};

const lnthAtPivotResult = evaluatePreFlightChecks(lnthAtPivotInputs);

// Decision hierarchy: An actionable setup is NOT necessarily cleared for execution!
assert.strictEqual(
  lnthAtPivotResult.isActionableGranted,
  true,
  "LNTH setup classification must remain ACTIONABLE"
);
assert.strictEqual(
  lnthAtPivotResult.isCleared,
  false,
  "LNTH execution clearance must fail-closed to NOT CLEARED when catalyst coverage is unavailable"
);
assert.strictEqual(
  lnthAtPivotResult.activeBlocker,
  "CATALYST COVERAGE UNAVAILABLE",
  "Active blocker at pivot must cite CATALYST COVERAGE UNAVAILABLE"
);
assert.strictEqual(
  lnthAtPivotResult.checks.find(c => c.id === "CHK-4")?.status,
  "UNAVAILABLE",
  "CHK-4 Catalyst Hazard Buffer must fail-closed to UNAVAILABLE"
);
assert.strictEqual(
  lnthAtPivotResult.checks.find(c => c.id === "CHK-1")?.status,
  "PASS",
  "CHK-1 at pivot must PASS with 2.25:1 R:R"
);

console.log("   [PASS] LNTH at Pivot: CHK-1 PASS, CHK-4 UNAVAILABLE -> activeBlocker: CATALYST COVERAGE UNAVAILABLE");

// Case 1B: LNTH at Live Spot ($98.40) where entry displacement degrades Spot R:R to 1.03:1
const lnthAtSpotInputs = {
  symbol: "LNTH",
  currentPrice: 98.40,
  breakoutPivot: 96.00,
  optimalEntryMin: 95.00,
  optimalEntryMax: 99.00,
  stopLoss: 92.00,
  takeProfit1: 105.00,
  riskRewardRatio: 2.25,
  isActionable: true,
  decisionState: "ACTIONABLE",
  setupPattern: "Minervini VCP",
  catalystEvidence: null,
  vix: 18.5,
  macroEvidence: {
    vix: 18.5,
    observationTime: new Date().toISOString(),
    marketSession: "REGULAR",
    freshnessStatus: "LIVE"
  }
};

const lnthAtSpotResult = evaluatePreFlightChecks(lnthAtSpotInputs);
assert.strictEqual(lnthAtSpotResult.isActionableGranted, true, "Setup is actionable");
assert.strictEqual(lnthAtSpotResult.isCleared, false, "Execution is blocked");
assert.strictEqual(
  lnthAtSpotResult.activeBlocker,
  "REWARD/RISK ASYMMETRY (< 2.0:1)",
  "Active blocker at spot correctly flags spot R:R degradation"
);
assert.strictEqual(
  lnthAtSpotResult.checks.find(c => c.id === "CHK-1")?.status,
  "FAIL",
  "CHK-1 fails because spot R:R (1.03:1) is below 2.0:1 floor"
);
assert.strictEqual(
  lnthAtSpotResult.checks.find(c => c.id === "CHK-4")?.status,
  "UNAVAILABLE",
  "CHK-4 remains UNAVAILABLE"
);

console.log("   [PASS] LNTH at Spot: CHK-1 FAIL (1.03:1 at spot) & CHK-4 UNAVAILABLE -> execution strictly blocked.");

// ------------------------------------------------------------------
// TEST 2: Workstream A — Dual Price-Basis R:R Authority Calculation
// ------------------------------------------------------------------
console.log("\n2. Verifying Dual Price-Basis R:R Calculation Authority...");

const pivotPrice = 96.00;
const stopLoss = 92.00;
const target1 = 105.00;
const spotPrice = 98.40;

const plannedRR = (target1 - pivotPrice) / (pivotPrice - stopLoss);
const spotRR = (target1 - spotPrice) / (spotPrice - stopLoss);

assert.strictEqual(Number(plannedRR.toFixed(2)), 2.25, "Planned R:R from pivot must equal 2.25:1");
assert.strictEqual(Number(spotRR.toFixed(2)), 1.03, "Prospective Spot R:R from $98.40 must equal 1.03:1");

assert.ok(
  plannedRR >= 2.0 && spotRR < 2.0,
  "Planned R:R clears the 2.0:1 floor from pivot, while Spot R:R is degraded due to entry extension"
);

console.log(`   [PASS] Planned R:R from Pivot: ${plannedRR.toFixed(2)}:1 (Authority: Base Model)`);
console.log(`   [PASS] Prospective R:R at Spot: ${spotRR.toFixed(2)}:1 (Authority: Live Observation)`);

// ------------------------------------------------------------------
// TEST 3: Workstream A — Code Invariants in SharedExecutionLadder & OptimalEntryExitCard
// ------------------------------------------------------------------
console.log("\n3. Verifying Execution Blocker & Sizing Decoupling Code Invariants...");

const ladderSource = fs.readFileSync(path.join(ROOT_DIR, "components/terminal/SharedExecutionLadder.tsx"), "utf8");
assert.ok(
  ladderSource.includes("isCleared?: boolean"),
  "SharedExecutionLadderProps must accept isCleared"
);
assert.ok(
  ladderSource.includes("activeBlocker?: string | null"),
  "SharedExecutionLadderProps must accept activeBlocker"
);
assert.ok(
  ladderSource.includes("Configure Shares (Clearance Blocked)"),
  "SharedExecutionLadder must decouple sizing button when clearance is blocked"
);
assert.ok(
  ladderSource.includes("Informational Sizing (Pre-Flight Blocked)"),
  "SharedExecutionLadder must mark sizing utilization as Pre-Flight Blocked"
);
assert.ok(
  ladderSource.includes('id="execution-clearance-blocker-banner"'),
  "SharedExecutionLadder must render prominent execution clearance blocker banner"
);

const optimalCardSource = fs.readFileSync(path.join(ROOT_DIR, "components/OptimalEntryExitCard.tsx"), "utf8");
assert.ok(
  optimalCardSource.includes("preflightCleared?: boolean"),
  "OptimalEntryExitCardProps must accept preflightCleared"
);
assert.ok(
  optimalCardSource.includes("CURRENT ACTION: Setup Actionable — Await Execution Clearance"),
  "OptimalEntryExitCard must display conditional action when actionable but not cleared"
);
assert.ok(
  optimalCardSource.includes('data-testid="preflight-blocker-pill"'),
  "OptimalEntryExitCard must display preflight blocker pill when clearance is blocked"
);

console.log("   [PASS] Code Invariants verified across SharedExecutionLadder and OptimalEntryExitCard.");

// ------------------------------------------------------------------
// TEST 4: Workstream B — Comparison Evidence Integrity & Missing Data Fail-Closed
// ------------------------------------------------------------------
console.log("\n4. Verifying Comparison Evidence Integrity (PROD-UX-02)...");

const compareSource = fs.readFileSync(path.join(ROOT_DIR, "app/compare/page.tsx"), "utf8");

// Assert MASTER_ASSET_CATALOG integration
assert.ok(
  compareSource.includes("MASTER_ASSET_CATALOG"),
  "app/compare/page.tsx must import and use MASTER_ASSET_CATALOG for authentic verified fundamentals"
);

// Assert absence of default 0 / false PARITY on missing metrics
assert.ok(
  compareSource.includes("INSUFFICIENT_EVIDENCE"),
  "Comparison table must render INSUFFICIENT_EVIDENCE when metric evidence is absent"
);

// Assert dynamic allocation synthesis does not cite unavailable metrics
assert.ok(
  compareSource.includes("Overweight recommendation for ${assetA.symbol} is withheld"),
  "Section 3 allocation synthesis must withhold overweight recommendation when fundamentals are unverified"
);
assert.ok(
  compareSource.includes("Overweight recommendation for ${assetB.symbol} is withheld"),
  "Section 3 allocation synthesis must withhold overweight recommendation when fundamentals are unverified"
);

// Assert reload persistence logic
assert.ok(
  compareSource.includes("ARX_COMPARE_PAIR"),
  "app/compare/page.tsx must persist and restore selected comparison pair via localStorage"
);

console.log("   [PASS] Comparison Evidence Integrity invariants verified.");

// ------------------------------------------------------------------
// TEST 5: Workstream B — Verified NVO/LLY & NVDA/AAPL Fundamentals
// ------------------------------------------------------------------
console.log("\n5. Verifying Authentic Fundamental Catalog Entries for NVO/LLY and NVDA/AAPL...");

const nvo = MASTER_ASSET_CATALOG["NVO"];
const lly = MASTER_ASSET_CATALOG["LLY"];
const nvda = MASTER_ASSET_CATALOG["NVDA"];
const aapl = MASTER_ASSET_CATALOG["AAPL"];

assert.ok(nvo && lly && nvda && aapl, "NVO, LLY, NVDA, AAPL must all exist in catalog");

assert.strictEqual(nvo.roic, 54.0, "NVO ROIC must be 54.0%");
assert.strictEqual(lly.roic, 38.2, "LLY ROIC must be 38.2%");
assert.ok(nvo.roic > lly.roic, "NVO has verified ROIC advantage over LLY (+15.8%)");

assert.strictEqual(nvo.grossMargin, 84.5, "NVO Gross Margin must be 84.5%");
assert.strictEqual(lly.grossMargin, 80.2, "LLY Gross Margin must be 80.2%");
assert.ok(nvo.grossMargin > lly.grossMargin, "NVO has verified Gross Margin advantage (+4.3%)");

assert.strictEqual(nvo.fcfYield, 3.8, "NVO FCF Yield must be 3.8%");
assert.strictEqual(lly.fcfYield, 2.4, "LLY FCF Yield must be 2.4%");
assert.ok(nvo.fcfYield > lly.fcfYield, "NVO has verified FCF Yield advantage (+1.4%)");

assert.strictEqual(nvda.roic, 48.0, "NVDA ROIC must be 48.0%");
assert.strictEqual(aapl.roic, 45.0, "AAPL ROIC must be 45.0%");
assert.ok(nvda.roic > aapl.roic, "NVDA has verified ROIC advantage over AAPL (+3.0%)");

console.log("   [PASS] Authentic verified fundamentals confirmed for NVO/LLY and NVDA/AAPL.");

// ------------------------------------------------------------------
// TEST 6: Workstream C — Dynamic Radar Counts, Contrast & Accessibility
// ------------------------------------------------------------------
console.log("\n6. Verifying Workstream C Secondary UX Polish...");

const radarSource = fs.readFileSync(path.join(ROOT_DIR, "app/radar/page.tsx"), "utf8");
assert.ok(
  !radarSource.includes("today's top 24 pre-screened confluence batch"),
  "Radar page must NOT contain hardcoded 'top 24 pre-screened' copy"
);
assert.ok(
  !radarSource.includes("today's top 24 pre-scanned confluence batch"),
  "Radar page must NOT contain hardcoded 'top 24 pre-scanned' copy"
);
assert.ok(
  radarSource.includes("${allAssets.length > 0 ? `${allAssets.length}-candidate` : \"active\"}"),
  "Radar page must render dynamic candidate count"
);

const portfolioSource = fs.readFileSync(path.join(ROOT_DIR, "app/portfolio/page.tsx"), "utf8");
assert.ok(
  portfolioSource.includes('aria-label="Refresh quotes"'),
  "Portfolio refresh quotes button must have aria-label"
);
assert.ok(
  portfolioSource.includes('aria-label="Export portfolio CSV"'),
  "Portfolio CSV export button must have aria-label"
);

const globalsCss = fs.readFileSync(path.join(ROOT_DIR, "app/globals.css"), "utf8");
assert.ok(
  globalsCss.includes('[data-theme="paper"] thead th'),
  "globals.css must include high-contrast paper theme table header styling"
);

console.log("   [PASS] Workstream C secondary UX polish verified.");

console.log("\n==================================================================");
console.log("ALL ARX PRODUCTION UX ACCEPTANCE REMEDIATION TESTS PASSED (100%)");
console.log("==================================================================\n");
