import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { resolveDecisionReadiness } from "../lib/decisionReadiness";
import { resolvePresentationState } from "../lib/decisionPresentation";
import { evaluatePreFlightChecks, PreFlightChecklistInputs } from "../lib/preflightEvaluator";
import { resolveNineEvidenceDomains, EvidenceDomainItem } from "../lib/evidenceDomains";
import { resolveOverallEvidenceBadge } from "../lib/dataProvenance";

const ROOT_DIR = path.resolve(__dirname, "..");

console.log("Starting ARX Phase 3 Governance & Unified Risk Clearance Verification Suite...\n");

// ------------------------------------------------------------------
// 1. Three-Tier Separation Verification
// ------------------------------------------------------------------
console.log("1. Verifying Three-Tier Separation (Data Availability vs Analytical Score vs Execution Clearance)...");

// Tier 1: Evidence Coverage & Provenance (Missing optional data is unassessed, not negative)
const domainsWithMissingOptional = resolveNineEvidenceDomains({
  symbol: "NVDA",
  assetClass: "EQUITY",
  candles: Array(100).fill({ close: 130, high: 132, low: 128, volume: 1000000 }),
  confluence: {
    confluenceScore: 78,
    pillars: [
      { pillar: "TECHNICAL_STRUCTURE", score: 82, status: "positive" },
      { pillar: "FUNDAMENTAL_SOLVENCY", score: 75, status: "positive" },
    ],
  } as any,
  smartMoney: null, // SEC Form 4 and Options Flow missing (e.g. under HOLD_SEC_001)
  macroData: { vix: 17.5, marketSession: "REGULAR" },
  preflightCleared: false,
  activeBlocker: "STAGE 4 DISTRIBUTION",
  isStage4: true,
  currentPrice: 130,
});

assert.strictEqual(domainsWithMissingOptional.length, 9, "Must preserve all 9 discrete evidence domains");

const insiderDomain = domainsWithMissingOptional.find((d) => d.id === "insider_transactions");
assert.ok(insiderDomain, "insider_transactions (Form 4) domain must exist");
assert.strictEqual(insiderDomain.impact, "NON_BLOCKING", "Missing SEC Form 4 must be non-blocking");
assert.strictEqual(insiderDomain.status, "NO_OBSERVATIONS", "Missing filings must report NO_OBSERVATIONS, not negative");

// Tier 2: Analytical Factor Confluence (Setup Score does not clear execution)
const highSetupScore = 88; // Bullish setup score
const stage4Inputs: PreFlightChecklistInputs = {
  symbol: "NVDA",
  currentPrice: 130,
  stopLoss: 120,
  takeProfit1: 155,
  riskRewardRatio: 2.5,
  isStage4: true, // Stage 4 distribution
  setupPattern: "Stage 4 Markdown",
  optimalEntryMin: 128,
  optimalEntryMax: 132,
  vix: 16.5,
  isActionable: true, // Attempting to authorize execution
};

const preflightStage4 = evaluatePreFlightChecks(stage4Inputs);
assert.strictEqual(preflightStage4.isCleared, false, "High setup score (88/100) must NOT clear execution when in Stage 4");
assert.ok(preflightStage4.activeBlocker.includes("STAGE 4") || preflightStage4.activeBlocker.includes("Stage 4"), "Stage 4 blocker must be explicit even when isActionable is passed as true");
assert.strictEqual(preflightStage4.checks[1].status, "STAGE 4 WAIT", "CHK-2 must evaluate to STAGE 4 WAIT");

console.log("   [PASS] Tier 1 & 2 separation verified: Missing optional data is unassessed, high scores cannot override structural blockers");

// ------------------------------------------------------------------
// 2. Authoritative Actionability Contract: isActionable = isExecutionReady && !hasActiveBlockers
// ------------------------------------------------------------------
console.log("\n2. Verifying Authoritative Actionability Contract (Zero setupScore dependency)...");

// Case A: 5/5 Pre-flight checks passed, but isActionable is false -> Remains blocked
const allChecksPassInputs: PreFlightChecklistInputs = {
  symbol: "AAPL",
  currentPrice: 220,
  stopLoss: 210,
  takeProfit1: 245,
  riskRewardRatio: 2.5,
  isStage4: false,
  setupPattern: "Cup and Handle Breakout",
  optimalEntryMin: 218,
  optimalEntryMax: 222,
  vix: 15.2,
  vixObservationTime: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
  catalystState: "PASS",
  catalystEvidence: {
    state: "PASS",
    daysToEarnings: 14.0,
    freshnessStatus: "LIVE",
    confirmationStatus: "CONFIRMED",
    coverageStatus: "CONFIRMED_UPCOMING",
  },
  isActionable: false, // Authoritative decision state says not actionable
};

const evResult = evaluatePreFlightChecks(allChecksPassInputs);
assert.strictEqual(evResult.passedCount, 5, "All 5 individual quantitative checks pass");
assert.strictEqual(evResult.isActionableGranted, false, "isActionableGranted must be false when isActionable input is false");
assert.strictEqual(evResult.isCleared, false, "isCleared must be false without canonical actionability");
assert.strictEqual(evResult.activeBlocker, "CANONICAL SETUP NON-ACTIONABLE");

// Case B: Decision Readiness 3-Gate Progression
const readinessActionable = resolveDecisionReadiness({
  currentPrice: 220,
  optimalEntryMin: 218,
  optimalEntryMax: 222,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 210,
  takeProfit1: 245,
  vix: 15.2,
  isActionable: true,
});

assert.strictEqual(readinessActionable.isExecutionReady, true, "isExecutionReady must be true when all gates pass with canonical actionability");
assert.strictEqual(readinessActionable.activeBlockingGate, "NONE", "activeBlockingGate must be NONE when actionable");

const readinessTriggerPending = resolveDecisionReadiness({
  currentPrice: 220,
  optimalEntryMin: 218,
  optimalEntryMax: 222,
  isConfirmed: false, // Trigger pending
  riskRewardRatio: 2.5,
  stopLoss: 210,
  takeProfit1: 245,
  vix: 15.2,
  isActionable: false,
});

assert.strictEqual(readinessTriggerPending.isExecutionReady, false, "Trigger pending must prevent execution readiness");
assert.strictEqual(readinessTriggerPending.activeBlockingGate, "GATE_2_TRIGGER", "Active blocker must be Gate 2 Trigger");

console.log("   [PASS] Authoritative Actionability Contract verified: actionability strictly requires execution readiness and zero blockers");

// ------------------------------------------------------------------
// 3. Validation of Six Canonical Decision States
// ------------------------------------------------------------------
console.log("\n3. Validating Six Canonical Decision States...");

// State 1: Actionable Setup
const s1 = resolvePresentationState({
  decisionState: "ACTIONABLE_SETUP",
  executionStatus: "READY_TO_BUY",
  isActionable: true,
  readinessResult: readinessActionable,
});
assert.strictEqual(s1.actionabilityLabel, "ACTIONABLE");
assert.strictEqual(s1.badgeLabel, "ACTIONABLE SETUP");
console.log("   [PASS] State 1 (Actionable Setup): Verified");

// State 2: Awaiting Trigger
const s2 = resolvePresentationState({
  decisionState: "VALID_SETUP",
  executionStatus: "IN_BUY_ZONE_AWAITING_TRIGGER",
  isActionable: false,
  readinessResult: readinessTriggerPending,
});
assert.strictEqual(s2.actionabilityLabel, "NOT ACTIONABLE");
assert.strictEqual(s2.badgeLabel, "AWAITING TRIGGER");
console.log("   [PASS] State 2 (Awaiting Trigger): Verified");

// State 3: Invalidated Setup / Stage 4 Defense
const readinessStage4 = resolveDecisionReadiness({
  currentPrice: 130,
  optimalEntryMin: 128,
  optimalEntryMax: 132,
  stagePhase: "Stage 4 Markdown",
  isActionable: false,
});
const s3 = resolvePresentationState({
  decisionState: "DISQUALIFIED",
  executionStatus: "STAGE_4_CORRECTION",
  isActionable: false,
  readinessResult: readinessStage4,
});
assert.strictEqual(s3.actionabilityLabel, "NOT ACTIONABLE");
assert.strictEqual(s3.badgeLabel, "STAGE 4 DEFENSE");
console.log("   [PASS] State 3 (Invalidated / Stage 4): Verified");

// State 4: Missing Optional Evidence (Evidence Incomplete / Partial)
const s4 = resolvePresentationState({
  decisionState: "EVIDENCE_INCOMPLETE",
  isActionable: false,
});
assert.strictEqual(s4.actionabilityLabel, "NOT ACTIONABLE");
assert.strictEqual(s4.badgeLabel, "EVIDENCE INCOMPLETE");
console.log("   [PASS] State 4 (Missing Optional Evidence): Verified");

// State 5: Stale Price Tape
const readinessStale = resolveDecisionReadiness({
  currentPrice: 130,
  optimalEntryMin: 128,
  optimalEntryMax: 132,
  freshnessStatus: "STALE_HISTORICAL",
  isActionable: false,
});
const s5 = resolvePresentationState({
  decisionState: "STALE_DATA",
  executionStatus: "STALE_MARKET_DATA",
  isActionable: false,
  readinessResult: readinessStale,
});
assert.strictEqual(s5.actionabilityLabel, "NOT ACTIONABLE");
assert.strictEqual(s5.badgeLabel, "STALE TAPE");
console.log("   [PASS] State 5 (Stale Price Tape): Verified");

// State 6: Unavailable / Unverified Levels
const readinessUnavailable = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 105,
  optimalEntryMax: 95, // Inverted corridor
  isActionable: false,
});
const s6 = resolvePresentationState({
  decisionState: "UNVERIFIED",
  executionStatus: "UNVERIFIED_ASSET",
  isActionable: false,
  readinessResult: readinessUnavailable,
});
assert.strictEqual(s6.actionabilityLabel, "NOT ACTIONABLE");
assert.strictEqual(s6.badgeLabel, "UNVERIFIED ASSET");
console.log("   [PASS] State 6 (Unavailable / Unverified Levels): Verified");

// ------------------------------------------------------------------
// 4. Consolidated Risk Clearance Presentation AST Invariants
// ------------------------------------------------------------------
console.log("\n4. Verifying Consolidated Risk Clearance Presentation in Components...");

const adaptiveTerminalSrc = fs.readFileSync(path.join(ROOT_DIR, "components/AdaptiveTerminal.tsx"), "utf8");
const evidenceCardSrc = fs.readFileSync(path.join(ROOT_DIR, "components/terminal/EvidenceDomainStatusCard.tsx"), "utf8");
const ladderSrc = fs.readFileSync(path.join(ROOT_DIR, "components/terminal/SharedExecutionLadder.tsx"), "utf8");

// 4.1 AdaptiveTerminal must NOT contain the old standalone floating provenance bar
assert.strictEqual(
  adaptiveTerminalSrc.includes("Evidence State Provenance Bar"),
  false,
  "AdaptiveTerminal must have removed the detached Evidence State Provenance Bar"
);

// 4.2 AdaptiveTerminal must pass evidenceBadge and eligibilityNotice to EvidenceDomainStatusCard
assert.ok(
  adaptiveTerminalSrc.includes("evidenceBadge={evidenceBadge}"),
  "AdaptiveTerminal must pass evidenceBadge to EvidenceDomainStatusCard"
);
assert.ok(
  adaptiveTerminalSrc.includes("eligibilityNotice="),
  "AdaptiveTerminal must pass eligibilityNotice to EvidenceDomainStatusCard"
);

// 4.3 EvidenceDomainStatusCard must integrate evidenceBadge and eligibilityNotice
assert.ok(
  evidenceCardSrc.includes("evidenceBadge?: OverallEvidenceBadge"),
  "EvidenceDomainStatusCard must type evidenceBadge prop"
);
assert.ok(
  evidenceCardSrc.includes('data-testid="evidence-provenance-badge"'),
  "EvidenceDomainStatusCard must render evidence-provenance-badge in header summary"
);
assert.ok(
  evidenceCardSrc.includes('data-testid="evidence-eligibility-notice"'),
  "EvidenceDomainStatusCard must render integrated evidence-eligibility-notice"
);
assert.ok(
  evidenceCardSrc.includes("Missing data is treated as unassessed, not negative"),
  "EvidenceDomainStatusCard must preserve the invariant disclosure"
);

// 4.4 SharedExecutionLadder must reference 5 pre-flight checks and single active blocker
assert.ok(
  ladderSrc.includes("${passedCount}/5 Checks"),
  "SharedExecutionLadder must accurately reference 5 Pre-Flight checks"
);
assert.ok(
  ladderSrc.includes('id="execution-clearance-blocker-banner"'),
  "SharedExecutionLadder must preserve the active blocker banner"
);

console.log("   [PASS] Component consolidation invariants verified across AdaptiveTerminal, EvidenceDomainStatusCard, and SharedExecutionLadder.");

console.log("\nALL PHASE 3 GOVERNANCE AND RISK CLEARANCE TESTS PASSED (100%)!\n");
