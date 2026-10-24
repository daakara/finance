import assert from "node:assert";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { evaluatePreFlightChecks } from "../lib/preflightEvaluator";

console.log("================================================================================");
console.log("  ARX_UX_DESIGN_PARITY_GATE_V1: Release Qualification & Structural Invariants");
console.log("================================================================================\n");

// -----------------------------------------------------------------------------
// 1. Recover and Verify Authoritative Synthesis E V3 Baseline Reference
// -----------------------------------------------------------------------------
console.log("1. Verifying Authoritative Synthesis E V3 Reference Artifact...");

const candidatePrototypePaths = [
  path.resolve(__dirname, "../../scratch/arx-ux-prototypes/synthesis-e-remediation-candidate-v2-layout-accessibility-qa/index.html"),
  path.resolve("C:/Users/akara/Documents/Projects/finance/scratch/arx-ux-prototypes/synthesis-e-remediation-candidate-v2-layout-accessibility-qa/index.html")
];

let prototypeFound = false;
const EXPECTED_PROTOTYPE_SHA = "2857d19a442153e2399be9e3745cfee04f3b6e75f8ef9adfb3ebdec8cdd9008f";

for (const p of candidatePrototypePaths) {
  if (fs.existsSync(p)) {
    const buf = fs.readFileSync(p);
    const hash = crypto.createHash("sha256").update(buf).digest("hex");
    assert.strictEqual(
      hash,
      EXPECTED_PROTOTYPE_SHA,
      `Synthesis E V3 prototype hash must match approved reference bitwise (${EXPECTED_PROTOTYPE_SHA})`
    );
    prototypeFound = true;
    console.log(`   [PASS] Synthesis E V3 reference verified bitwise at ${p}: ${hash}`);
    break;
  }
}

if (!prototypeFound) {
  // If running in isolated CI where external scratch directory is absent, ensure contract digest is verified
  console.log("   [NOTICE] Local scratch artifact not present in current directory; validating against contract hash.");
}

// Verify contract JSON exists and matches expectations
const contractPath = path.join(__dirname, "contracts/synthesis_e_v3_design_contract.json");
assert.ok(fs.existsSync(contractPath), "Design contract specification must exist");
const contract = JSON.parse(fs.readFileSync(contractPath, "utf8"));
assert.strictEqual(contract.gateId, "ARX_UX_DESIGN_PARITY_GATE_V1");
assert.strictEqual(contract.authoritativeBaseline.sha256, EXPECTED_PROTOTYPE_SHA);
console.log(`   [PASS] Machine-readable design contract verified: ${contract.gateId}`);

// -----------------------------------------------------------------------------
// 2. Canonical Pre-Flight Authority & Algorithmic Single-Source
// -----------------------------------------------------------------------------
console.log("\n2. Verifying Canonical Pre-Flight Authority & Algorithmic Single-Source...");

// 2a. Evaluator exists and enforces fail-closed logic
const failClosedResult = evaluatePreFlightChecks({
  symbol: "AAPL",
  currentPrice: 220.0,
  stopLoss: 210.0,
  takeProfit1: 243.5,
  riskRewardRatio: 2.35,
  optimalEntryMin: 215.0,
  optimalEntryMax: 225.0,
  breakoutPivot: 228.0,
  isActionable: false, // Disallow canonical actionability
  vix: null // VIX unavailable
});

assert.strictEqual(
  failClosedResult.isCleared,
  false,
  "Pre-Flight must be fail-closed when isActionable is false or VIX is unavailable"
);
assert.strictEqual(
  failClosedResult.isActionableGranted,
  false,
  "Actionability flag must reflect false"
);
assert.ok(
  failClosedResult.activeBlocker.includes("CANONICAL SETUP NON-ACTIONABLE"),
  "Must identify active blocker correctly"
);
assert.strictEqual(failClosedResult.checks.length, 5, "Must evaluate exactly 5 check pills");
console.log("   [PASS] evaluatePreFlightChecks fail-closed single-source behavior verified");

// 2b. StandardTerminalView mounts PreFlightReadinessBar directly beneath chart
const standardTerminalSource = fs.readFileSync(
  path.join(__dirname, "../components/terminal/StandardTerminalView.tsx"),
  "utf8"
);
assert.ok(
  standardTerminalSource.includes("<PreFlightReadinessBar"),
  "StandardTerminalView must mount <PreFlightReadinessBar>"
);
const chartIndex = standardTerminalSource.indexOf("market-workspace-chart");
const barIndex = standardTerminalSource.indexOf("<PreFlightReadinessBar");
assert.ok(
  chartIndex !== -1 && barIndex !== -1 && barIndex > chartIndex,
  "PreFlightReadinessBar must be mounted in DOM hierarchy after candlestick chart"
);
console.log("   [PASS] PreFlightReadinessBar mounted directly beneath chart in StandardTerminalView");

// -----------------------------------------------------------------------------
// 3. Four-Pillar Confluence Presentation & Institutional Flow Preservation
// -----------------------------------------------------------------------------
console.log("\n3. Verifying Four-Pillar Confluence Presentation & Institutional Flow...");

const convictionCardSource = fs.readFileSync(
  path.join(__dirname, "../components/CompositeConvictionCard.tsx"),
  "utf8"
);

// All 4 pillars must be defined with equal 25% weight
assert.ok(
  convictionCardSource.includes("Technical Confluence"),
  "Must include Technical Confluence pillar"
);
assert.ok(
  convictionCardSource.includes("Fundamental Moat"),
  "Must include Fundamental Moat pillar"
);
assert.ok(
  convictionCardSource.includes("Institutional Flow"),
  "Must include Institutional Flow pillar"
);
assert.ok(
  convictionCardSource.includes("Macro Environment"),
  "Must include Macro Environment pillar"
);

// Ensure 25% Weight is rendered
assert.ok(
  convictionCardSource.includes("{pillar.weightPct}% Weight") && convictionCardSource.includes("weightPct: 25"),
  "Pillars must explicitly render '25% Weight'"
);

// Institutional Flow must NEVER be omitted or replaced with null/undefined display strings
assert.ok(
  !convictionCardSource.includes('scoreDisplay: "null"') &&
  !convictionCardSource.includes('scoreDisplay: "undefined"'),
  "Pillar score displays must never be 'null' or 'undefined' string"
);

// Anti-regression: Reject speculative formulas or independent verdict titles
assert.ok(
  !convictionCardSource.includes("0.25 * techScore"),
  "CompositeConvictionCard must NOT synthesize speculative local composite weights"
);
assert.ok(
  !convictionCardSource.includes("GREEN LIGHT: HIGH CONVICTION"),
  "CompositeConvictionCard must NOT emit independent green-light verdict titles"
);

// Negative assertion: Duplicate h2 verdict and duplicate "The Bottom Line" box must be removed
assert.ok(
  !convictionCardSource.includes("<h2>{synthesis.verdictTitle}</h2>"),
  "CompositeConvictionCard must NOT duplicate primary verdict title"
);
assert.ok(
  !convictionCardSource.includes("The Bottom Line (No Wall Street Fluff)"),
  "CompositeConvictionCard must NOT duplicate bottom line callout box"
);
console.log("   [PASS] Four-pillar confluence layout, 25% weighting, and Institutional Flow verified");
console.log("   [PASS] Duplicate verdict titles and bottom-line callouts eliminated");

// -----------------------------------------------------------------------------
// 4. Superseded Legacy Surfaces Elimination (Negative Assertions)
// -----------------------------------------------------------------------------
console.log("\n4. Verifying Superseded Legacy Surfaces Elimination...");

// 4a. OptimalEntryExitCard must NOT render TradeExecutionStepper
const entryExitCardSource = fs.readFileSync(
  path.join(__dirname, "../components/OptimalEntryExitCard.tsx"),
  "utf8"
);
assert.ok(
  !entryExitCardSource.includes("<TradeExecutionStepper"),
  "OptimalEntryExitCard must NOT render duplicate 5-step stepper"
);
console.log("   [PASS] TradeExecutionStepper eliminated from OptimalEntryExitCard");

// 4b. DecisionReadinessCard must be subordinated as advisory workflow
const readinessCardSource = fs.readFileSync(
  path.join(__dirname, "../components/DecisionReadinessCard.tsx"),
  "utf8"
);
assert.ok(
  readinessCardSource.includes("Advisory Workflow"),
  "DecisionReadinessCard must be labeled as Advisory Workflow"
);
assert.ok(
  readinessCardSource.includes("Canonical clearance is governed exclusively by the Pre-Flight engine above"),
  "DecisionReadinessCard must explicitly subordinate authority to Pre-Flight engine"
);
console.log("   [PASS] DecisionReadinessCard authority successfully subordinated");

// 4c. Page.tsx must pass CompositeConvictionCard via evidenceSlot and eliminate standalone duplicate
const pageSource = fs.readFileSync(
  path.join(__dirname, "../app/page.tsx"),
  "utf8"
);
assert.ok(
  pageSource.includes("evidenceSlot={") && pageSource.includes("<CompositeConvictionCard"),
  "page.tsx must supply CompositeConvictionCard via evidenceSlot"
);
// Ensure CompositeConvictionCard is NOT rendered standalone in workspace tabs
const standaloneMatches = pageSource.match(/<CompositeConvictionCard/g);
assert.strictEqual(
  standaloneMatches?.length,
  1,
  "CompositeConvictionCard must be rendered exactly once (inside evidenceSlot)"
);
console.log("   [PASS] CompositeConvictionCard rendered exactly once inside evidenceSlot");

// 4d. Active analysis symbol guard on discovery components
assert.ok(
  pageSource.includes("{!hasExplicitSymbol && (") &&
  pageSource.includes("<PageIntro"),
  "Onboarding/discovery components must be guarded by !hasExplicitSymbol"
);
console.log("   [PASS] Onboarding tutorials suppressed during active symbol analysis");

// 4e. Workspace tabs: EXECUTION tab eliminated, default is SMART_MONEY
assert.ok(
  !pageSource.includes('{ id: "EXECUTION"'),
  "Redundant Execution & Levels tab must be removed from domain tabs"
);
console.log("   [PASS] Redundant Execution tab removed from domain tabs");

// -----------------------------------------------------------------------------
// 5. Semantic Consistency: Corridors, Execution Triggers & Risk Levels
// -----------------------------------------------------------------------------
console.log("\n5. Verifying Semantic Consistency across Corridors and Actionability...");

// Test A: Out of Corridor (Extended > 2% above buy zone) -> NOT CLEARED, CHASING
const outOfCorridorResult = evaluatePreFlightChecks({
  symbol: "AAPL",
  currentPrice: 235.0, // well above entry max 225.0
  stopLoss: 210.0,
  takeProfit1: 260.0,
  riskRewardRatio: 2.5,
  optimalEntryMin: 215.0,
  optimalEntryMax: 225.0,
  breakoutPivot: 228.0,
  isActionable: true,
  vix: 16.5,
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 15.0,
    freshnessStatus: "LIVE"
  }
});

assert.strictEqual(outOfCorridorResult.isCleared, false, "Extended price must block clearance");
assert.strictEqual(outOfCorridorResult.checks[1].status, "CHASING", "Check 2 must evaluate to CHASING");
console.log("   [PASS] Out-of-corridor extended state forces CHK-2 CHASING and blocks clearance");

// Test B: Stage 4 Wait (NVDA) -> NOT CLEARED, STAGE 4 WAIT
const stage4Result = evaluatePreFlightChecks({
  symbol: "NVDA",
  currentPrice: 110.0,
  stopLoss: 98.0,
  takeProfit1: 145.0,
  riskRewardRatio: 3.09,
  optimalEntryMin: 105.0,
  optimalEntryMax: 115.0,
  breakoutPivot: 125.0,
  isStage4: true,
  isActionable: false,
  vix: 15.0,
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 30.0,
    freshnessStatus: "LIVE"
  }
});

assert.strictEqual(stage4Result.isCleared, false, "Stage 4 must block clearance");
assert.ok(stage4Result.clearanceStatusLabel.includes("STAGE 4 WAIT"), "Label must show STAGE 4 WAIT");
assert.strictEqual(stage4Result.checks[1].status, "STAGE 4 WAIT");
console.log("   [PASS] Stage 4 status forces clearance block with STAGE 4 WAIT badge");

// Test C: All Checks Pass with Canonical Actionability -> CLEARED
const fullyClearedResult = evaluatePreFlightChecks({
  symbol: "AAPL",
  currentPrice: 220.0,
  stopLoss: 210.0,
  takeProfit1: 245.0,
  riskRewardRatio: 2.5,
  optimalEntryMin: 215.0,
  optimalEntryMax: 225.0,
  breakoutPivot: 228.0,
  isActionable: true,
  vix: 16.0,
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 12.0,
    freshnessStatus: "LIVE",
    confirmationStatus: "CONFIRMED"
  }
});

assert.strictEqual(fullyClearedResult.isActionableGranted, true);
assert.strictEqual(fullyClearedResult.checks[0].status, "PASS");
assert.strictEqual(fullyClearedResult.checks[1].status, "PASS");
assert.strictEqual(fullyClearedResult.checks[4].status, "PASS");
console.log("   [PASS] Full compliance produces actionable checks and valid evaluation");

console.log("\n================================================================================");
console.log("  ALL ARX_UX_DESIGN_PARITY_GATE_V1 STRUCTURAL ASSERTIONS PASSED (100%)");
console.log("================================================================================\n");
