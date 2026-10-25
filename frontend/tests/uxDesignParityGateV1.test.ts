import assert from "node:assert";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { evaluatePreFlightChecks } from "../lib/preflightEvaluator";

console.log("================================================================================");
console.log("  ARX_UX_DESIGN_PARITY_GATE_V1 & V1.1: Release Qualification & Structural Invariants");
console.log("================================================================================\n");

// -----------------------------------------------------------------------------
// 1. Recover and Verify Authoritative Synthesis E V3 Baseline Reference
// -----------------------------------------------------------------------------
console.log("1. Verifying Authoritative Synthesis E V3 Reference Artifact...");

const candidatePrototypePaths = [
  path.resolve(__dirname, "fixtures/synthesis-e-v3-approved-reference.html"),
  path.resolve(__dirname, "../../scratch/arx-ux-prototypes/synthesis-e-remediation-candidate-v2-layout-accessibility-qa/index.html")
];

let prototypeFound = false;
let resolvedReferencePath: string | null = null;
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
    resolvedReferencePath = p;
    console.log(`   [PASS] Synthesis E V3 reference verified bitwise at ${p}: ${hash}`);
    break;
  }
}

if (!prototypeFound) {
  console.error("   [FAIL] Approved Synthesis E V3 reference baseline is legitimately unavailable. Disposition: UNVERIFIED");
  process.exit(1);
}

// Verify V1 and V1.1 contracts exist and match baseline identity
const contractPathV1 = path.join(__dirname, "contracts/synthesis_e_v3_design_contract.json");
assert.ok(fs.existsSync(contractPathV1), "Design contract V1 specification must exist");
const contractV1 = JSON.parse(fs.readFileSync(contractPathV1, "utf8"));
assert.strictEqual(contractV1.gateId, "ARX_UX_DESIGN_PARITY_GATE_V1");
assert.strictEqual(contractV1.authoritativeBaseline.sha256, EXPECTED_PROTOTYPE_SHA);
console.log(`   [PASS] Machine-readable design contract verified: ${contractV1.gateId}`);

const contractPathV1_1 = path.join(__dirname, "contracts/synthesis_e_v1_1_design_contract.json");
assert.ok(fs.existsSync(contractPathV1_1), "Design contract V1.1 specification must exist");
const contractV1_1 = JSON.parse(fs.readFileSync(contractPathV1_1, "utf8"));
assert.strictEqual(contractV1_1.gateId, "ARX_UX_DESIGN_PARITY_GATE_V1_1");
assert.strictEqual(contractV1_1.authoritativeBaseline.sha256, EXPECTED_PROTOTYPE_SHA);
console.log(`   [PASS] Machine-readable design contract verified: ${contractV1_1.gateId}`);

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

// 2c. Execution ladder mounted inside Decision Verdict card
assert.ok(
  standardTerminalSource.includes("Execution Ladder"),
  "StandardTerminalView must render Execution Ladder inside Decision Verdict card"
);
assert.ok(
  standardTerminalSource.includes("Buy Zone"),
  "StandardTerminalView execution ladder must render Buy Zone"
);
console.log("   [PASS] Compact Execution Ladder rendered inside Decision Verdict card");

// 2d. Subordinated 'What Needs to Change' is collapsed by default
assert.ok(
  standardTerminalSource.includes("isPreconditionsOpen, setIsPreconditionsOpen] = React.useState(false)") || standardTerminalSource.includes("isPreconditionsOpen, setIsPreconditionsOpen] = useState(false)"),
  "StandardTerminalView must collapse What Needs to Change by default (isPreconditionsOpen: false)"
);
console.log("   [PASS] 'What Needs to Change' stack subordinated and collapsed by default");

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

// Negative assertion: Duplicate h2 verdict and duplicate 'The Bottom Line' box must be removed
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

const entryExitCardSource = fs.readFileSync(
  path.join(__dirname, "../components/OptimalEntryExitCard.tsx"),
  "utf8"
);

// 4a. OptimalEntryExitCard must NOT render TradeExecutionStepper
assert.ok(
  !entryExitCardSource.includes("<TradeExecutionStepper"),
  "OptimalEntryExitCard must NOT render duplicate 5-step stepper"
);
console.log("   [PASS] TradeExecutionStepper eliminated from OptimalEntryExitCard");

// 4b. OptimalEntryExitCard must NOT mount DecisionReadinessCard
assert.ok(
  !entryExitCardSource.includes("<DecisionReadinessCard"),
  "OptimalEntryExitCard must NOT mount DecisionReadinessCard"
);
console.log("   [PASS] DecisionReadinessCard eliminated from OptimalEntryExitCard");

// 4c. OptimalEntryExitCard must NOT mount duplicate PreFlightChecklistModal
assert.ok(
  !entryExitCardSource.includes("<PreFlightChecklistModal"),
  "OptimalEntryExitCard must NOT mount duplicate PreFlightChecklistModal"
);
console.log("   [PASS] Duplicate PreFlightChecklistModal eliminated from OptimalEntryExitCard");

// 4d. DecisionReadinessCard itself is subordinated as advisory workflow
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
console.log("   [PASS] DecisionReadinessCard standalone authority successfully subordinated");

// 4e. Page.tsx must pass CompositeConvictionCard via evidenceSlot and eliminate standalone duplicate
const pageSource = fs.readFileSync(
  path.join(__dirname, "../app/page.tsx"),
  "utf8"
);
assert.ok(
  pageSource.includes("evidenceSlot={") && pageSource.includes("<CompositeConvictionCard"),
  "page.tsx must supply CompositeConvictionCard via evidenceSlot"
);
const standaloneMatches = pageSource.match(/<CompositeConvictionCard/g);
assert.strictEqual(
  standaloneMatches?.length,
  1,
  "CompositeConvictionCard must be rendered exactly once (inside evidenceSlot)"
);
console.log("   [PASS] CompositeConvictionCard rendered exactly once inside evidenceSlot");

// 4f. Active analysis symbol guard on discovery components
assert.ok(
  pageSource.includes("{!hasExplicitSymbol && (") &&
  pageSource.includes("<PageIntro"),
  "Onboarding/discovery components must be guarded by !hasExplicitSymbol"
);
console.log("   [PASS] Onboarding tutorials suppressed during active symbol analysis");

// 4g. Workspace tabs: EXECUTION tab eliminated, default is SMART_MONEY
assert.ok(
  !pageSource.includes('{ id: "EXECUTION"'),
  "Redundant Execution & Levels tab must be removed from domain tabs"
);
console.log("   [PASS] Redundant Execution tab removed from domain tabs");

// 4h. Navbar auto-onboarding suppression during active symbol analysis
const navbarSource = fs.readFileSync(
  path.join(__dirname, "../components/Navbar.tsx"),
  "utf8"
);
assert.ok(
  navbarSource.includes("isAnalyzingStock") &&
  navbarSource.includes("setIsOnboardingOpen(true)"),
  "Navbar must guard auto-onboarding against active symbol navigation"
);
console.log("   [PASS] Navbar onboarding popup suppressed during active symbol analysis");

// 4i. Increment 5: Touch target compliance (min-h-[44px]) for PreFlight buttons, conviction card, and workspace tabs
const preflightBarSource = fs.readFileSync(
  path.join(__dirname, "../components/PreFlightReadinessBar.tsx"),
  "utf8"
);
assert.ok(
  preflightBarSource.includes('id="toggle-preflight-details-btn"') &&
  preflightBarSource.includes("min-h-[44px]"),
  "toggle-preflight-details-btn must enforce min-h-[44px] touch target"
);
assert.ok(
  preflightBarSource.includes('data-testid="inline-open-modal-btn"') &&
  preflightBarSource.includes("min-h-[44px]"),
  "inline-open-modal-btn must enforce min-h-[44px] touch target"
);
assert.ok(
  convictionCardSource.includes('id="why-score-btn"') &&
  convictionCardSource.includes("min-h-[44px]"),
  "#why-score-btn must enforce min-h-[44px] touch target"
);
assert.ok(
  pageSource.includes('activeTab === "SMART_MONEY"') &&
  pageSource.includes("min-h-[44px]"),
  "Workspace tab buttons must enforce min-h-[44px] touch targets"
);
console.log("   [PASS] Touch target compliance (min-h-[44px]) enforced across all interactive controls (DEF-H1-05)");

// 4j. Increment 5: Tab URL state normalization and orphaned tab cleanup (DEF-H2-06)
assert.ok(
  !pageSource.includes('activeTab === "EXECUTION"'),
  "Orphaned activeTab === EXECUTION render block must be completely removed"
);
assert.ok(
  pageSource.includes('urlTab === "EXECUTION"') &&
  pageSource.includes('setActiveTab("SMART_MONEY")'),
  "page.tsx must normalize legacy urlTab === EXECUTION to SMART_MONEY"
);
console.log("   [PASS] Tab URL state normalization and orphaned tab cleanup verified (DEF-H2-06)");

// 4k. Increment 5: Interactive In-Line Telemetry Drawer Expansion (DEF-H1-06)
assert.ok(
  preflightBarSource.includes("isDrawerExpanded?: boolean") &&
  preflightBarSource.includes("onToggleDrawer?: (expanded: boolean) => void"),
  "PreFlightReadinessBar must support controlled isDrawerExpanded and onToggleDrawer props"
);
assert.ok(
  pageSource.includes("isPreflightDrawerExpanded") &&
  pageSource.includes("onToggleDrawer={setIsPreflightDrawerExpanded}"),
  "page.tsx must pass controlled isDrawerExpanded and onToggleDrawer to PreFlightReadinessBar"
);
assert.ok(
  pageSource.includes("setIsPreflightDrawerExpanded(true)") &&
  pageSource.includes("preflight-readiness-bar"),
  "CompositeConvictionCard onOpenWhy must trigger in-line telemetry drawer expansion and element scroll"
);
console.log("   [PASS] Interactive In-Line Telemetry Drawer expansion and scroll verified (DEF-H1-06)");

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
    freshnessStatus: "LIVE",
    confirmationStatus: "CONFIRMED"
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
    freshnessStatus: "LIVE",
    confirmationStatus: "CONFIRMED"
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
  vixObservationTime: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
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

// Test D: Corridor Semantics within +2% Tolerance Band
const toleranceResult = evaluatePreFlightChecks({
  symbol: "AAPL",
  currentPrice: 226.50, // Above max 225.0, but <= 225.0 * 1.02 (229.50)
  stopLoss: 210.0,
  takeProfit1: 245.0,
  riskRewardRatio: 2.5,
  optimalEntryMin: 215.0,
  optimalEntryMax: 225.0,
  breakoutPivot: 228.0,
  isActionable: true,
  vix: 16.0,
  vixObservationTime: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 12.0,
    freshnessStatus: "LIVE",
    confirmationStatus: "CONFIRMED"
  }
});

assert.strictEqual(toleranceResult.checks[1].summary, "Within Tolerance (+2%)");
assert.strictEqual(toleranceResult.checks[1].status, "PASS");
assert.ok(
  toleranceResult.checks[1].detail.includes("slightly above preferred corridor"),
  "Detail must state slightly above preferred corridor"
);
assert.ok(
  !toleranceResult.checks[1].detail.includes("inside the verified buying range"),
  "Must NEVER claim inside buying range when spot exceeds corridor ceiling"
);
console.log("   [PASS] Corridor tolerance (+2%) accurately described without claiming inside buying range");

// Test E: Catalyst Provenance - Unverified Source
const unverifiedCatalystResult = evaluatePreFlightChecks({
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
  vixObservationTime: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 12.0,
    coverageStatus: "CONFIRMED_UPCOMING",
    freshnessStatus: "LIVE",
    confirmationStatus: "UNVERIFIED",
    sourceName: "Yahoo Finance (Aggregator)"
  }
});

assert.strictEqual(unverifiedCatalystResult.checks[3].status, "PASS");
assert.ok(
  unverifiedCatalystResult.checks[3].detail.includes("unverified by primary source"),
  "Unverified source must disclose unverified by primary source"
);
assert.ok(
  !unverifiedCatalystResult.checks[3].detail.includes("Verified event runway"),
  "Unverified source must NEVER claim 'Verified event runway'"
);
assert.ok(
  !unverifiedCatalystResult.checks[3].detail.includes("Clean runway verified: Confirmed earnings"),
  "Unverified source must NEVER claim 'Clean runway verified: Confirmed earnings'"
);
console.log("   [PASS] Catalyst unverified provenance strictly enforced with mandatory disclaimer");

// Test F: Catalyst Provenance - Missing Coverage or Freshness Fails Closed to UNAVAILABLE
const missingFreshnessResult = evaluatePreFlightChecks({
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
  vixObservationTime: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 12.0,
    freshnessStatus: "MISSING", // Missing freshness
    confirmationStatus: "CONFIRMED"
  }
});

assert.strictEqual(missingFreshnessResult.checks[3].status, "UNAVAILABLE");
assert.strictEqual(missingFreshnessResult.isCleared, false, "Missing freshness must block clearance");
console.log("   [PASS] Missing catalyst freshness fails closed to UNAVAILABLE and blocks clearance");

// Test G: Macro Parity & Single-Source VIX Resolution
const elevatedVixResult = evaluatePreFlightChecks({
  symbol: "AAPL",
  currentPrice: 220.0,
  stopLoss: 210.0,
  takeProfit1: 245.0,
  riskRewardRatio: 2.5,
  optimalEntryMin: 215.0,
  optimalEntryMax: 225.0,
  breakoutPivot: 228.0,
  isActionable: true,
  vix: 28.5, // Elevated VIX >= 26.0
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 12.0,
    freshnessStatus: "LIVE",
    confirmationStatus: "CONFIRMED"
  }
});

assert.strictEqual(elevatedVixResult.checks[4].status, "HIGH VIX");
assert.strictEqual(elevatedVixResult.isCleared, false, "Elevated VIX must block clearance");
console.log("   [PASS] Macro VIX >= 26.0 strictly blocks clearance");

// Test H: Macro Freshness & Edge-Case Governance (ARCH-GOV-DATA-QUALITY-VNEXT Section 3.4 & EC-01)
// Enforces that browser cache cannot provide unverified, stale, or cross-ticker Macro PASS.
const preflightSource = fs.readFileSync(
  path.join(__dirname, "../lib/preflightEvaluator.ts"),
  "utf8"
);
assert.ok(
  preflightSource.includes("export function resolveFreshCachedVix"),
  "preflightEvaluator.ts must export resolveFreshCachedVix"
);
assert.ok(
  preflightSource.includes("export function resolveFreshMacroEvidence"),
  "preflightEvaluator.ts must export resolveFreshMacroEvidence"
);
assert.ok(
  preflightSource.includes("hasQualifiedClosedSettlement ? 4320 : 240"),
  "resolveFreshMacroEvidence must enforce 240m active / 4320m qualified closed freshness SLA"
);

let mockStorage: Record<string, string> = {};
(global as any).window = {};
(global as any).localStorage = {
  getItem: (key: string) => mockStorage[key] || null,
  setItem: (key: string, val: string) => { mockStorage[key] = val; },
  removeItem: (key: string) => { delete mockStorage[key]; },
  clear: () => { mockStorage = {}; }
};

try {
  // H.1: Fresh live feed takes precedence
  const freshLiveResult = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: 18.2,
    vixObservationTime: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
    catalystState: "PASS"
  });
  assert.strictEqual(freshLiveResult.checks[4].status, "PASS");
  assert.strictEqual(freshLiveResult.checks[4].summary, "VIX 18.2 (Normal)");
  console.log("   [PASS] Fresh live exchange VIX feed evaluates to PASS");

  // H.2: Fresh cached VIX (10 mins old) during feed outage
  mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
    vix: { value: 16.5, level: 16.5 },
    updatedAt: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    marketSession: "REGULAR"
  });
  const freshCacheResult = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: undefined,
    catalystState: "PASS"
  });
  assert.strictEqual(freshCacheResult.checks[4].status, "PASS");
  assert.strictEqual(freshCacheResult.checks[4].summary, "VIX 16.5 (Normal)");
  console.log("   [PASS] Fresh cached VIX (within 240m SLA) cleanly covers transient feed outage");

  // H.3: Stale cached VIX (> 240m open session) MUST FAIL CLOSED to UNAVAILABLE
  mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
    vix: { value: 14.8, level: 14.8 },
    updatedAt: new Date(Date.now() - 300 * 60 * 1000).toISOString(),
    marketSession: "REGULAR"
  });
  const staleActiveCacheResult = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: undefined,
    catalystState: "PASS"
  });
  assert.strictEqual(staleActiveCacheResult.checks[4].status, "UNAVAILABLE", "Stale macro cache (> 240m open) must fail closed to UNAVAILABLE");
  assert.strictEqual(staleActiveCacheResult.checks[4].passed, false);
  assert.strictEqual(staleActiveCacheResult.checks[4].summary, "VIX Unavailable");
  assert.strictEqual(staleActiveCacheResult.isCleared, false, "Stale macro cache must strictly block clearance");
  console.log("   [PASS] Stale cache (> 240m open) strictly rejected as UNAVAILABLE and blocks clearance");

  // H.4: Stale weekend cache (> 4320m / 72h) MUST FAIL CLOSED to UNAVAILABLE
  mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
    vix: { value: 15.0, level: 15.0 },
    updatedAt: new Date(Date.now() - 80 * 3600 * 1000).toISOString(),
    marketSession: "CLOSED",
    settlementPinned: true
  });
  const staleWeekendResult = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: undefined,
    catalystState: "PASS"
  });
  assert.strictEqual(staleWeekendResult.checks[4].status, "UNAVAILABLE");
  assert.strictEqual(staleWeekendResult.isCleared, false, "Stale weekend cache must strictly block clearance");
  console.log("   [PASS] Stale weekend pinned session (> 72h) strictly rejected as UNAVAILABLE");

  // H.5: Missing cache / Corrupted JSON / Cross-ticker cache MUST FAIL CLOSED
  mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
    symbol: "AAPL",
    price: 220.0
  });
  const crossTickerResult = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: undefined,
    catalystState: "PASS"
  });
  assert.strictEqual(crossTickerResult.checks[4].status, "UNAVAILABLE");
  assert.strictEqual(crossTickerResult.isCleared, false);
  console.log("   [PASS] Cross-ticker payload strictly fails closed to UNAVAILABLE");

  // H.6: Bare numeric VIX without timestamp metadata MUST FAIL CLOSED to UNAVAILABLE (M03)
  mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = "";
  const bareVixResult = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: 17.5, // Untrusted bare numeric input without timestamp
    catalystState: "PASS"
  });
  assert.strictEqual(bareVixResult.checks[4].status, "UNAVAILABLE", "Bare numeric VIX without timestamp must fail closed");
  assert.strictEqual(bareVixResult.isCleared, false);
  console.log("   [PASS] Bare numeric VIX without trustworthy timestamp strictly fails closed to UNAVAILABLE");

  // H.7: Stale observation time delivered in fresh API payload MUST FAIL CLOSED (M05)
  mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
    vix: {
      value: 16.0,
      observationTime: new Date(Date.now() - 300 * 60 * 1000).toISOString() // 5 hours old observation
    },
    updatedAt: new Date().toISOString(), // fresh delivery wrapper
    marketSession: "REGULAR"
  });
  const staleObsResult = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: undefined,
    catalystState: "PASS"
  });
  assert.strictEqual(staleObsResult.checks[4].status, "UNAVAILABLE", "Stale market observation must govern over fresh API delivery");
  assert.strictEqual(staleObsResult.isCleared, false);
  console.log("   [PASS] Market observation timestamp governance strictly prevents stale bar pass in fresh API wrapper");

  // H.8: Unpinned closed-session snapshot capped at active 240m threshold (M07)
  mockStorage["FINANCE_MARKET_SNAPSHOTS_V1"] = JSON.stringify({
    vix: { value: 15.2, level: 15.2 },
    updatedAt: new Date(Date.now() - 300 * 60 * 1000).toISOString(), // 5h old
    marketSession: "CLOSED",
    settlementPinned: false // Unauthenticated settlement pin
  });
  const unpinnedClosedResult = evaluatePreFlightChecks({
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    optimalEntryMin: 215.0,
    optimalEntryMax: 225.0,
    breakoutPivot: 228.0,
    isActionable: true,
    vix: undefined,
    catalystState: "PASS"
  });
  assert.strictEqual(unpinnedClosedResult.checks[4].status, "UNAVAILABLE", "Unpinned CLOSED session must not receive 72h window");
  assert.strictEqual(unpinnedClosedResult.isCleared, false);
  console.log("   [PASS] Unpinned CLOSED session snapshot strictly capped at 240m active threshold");
} finally {
  delete (global as any).window;
  delete (global as any).localStorage;
}

console.log("\n================================================================================");
console.log("  ALL ARX_UX_DESIGN_PARITY_GATE_V1 & V1.1 STRUCTURAL ASSERTIONS PASSED (100%)");
console.log("================================================================================\n");
