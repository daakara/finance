/**
 * ARX TERMINAL — Financial Decision Precedence & Composition Test Matrix
 * Gate A Regression Suite (Execution ID: ARX_VAG01_INDEPENDENT_ACCEPTANCE_RECONCILIATION_003)
 *
 * Verifies that:
 * 1. Stale-data suppression never incorrectly authorizes sizing or trading.
 * 2. Stale data does not silently erase higher-priority financial warnings (breached invalidation floors, exit review, avoid).
 * 3. The authoritative decision state is preserved via `authoritativeDecisionState` and composed with freshness.
 * 4. Precedence hierarchy strictly enforced:
 *    P0 Safety: Invalid price / Unavailable feed -> UNVERIFIED
 *    P0 Emergency: Invalidation Breached -> EXIT_REVIEW (owned) / AVOID (unowned)
 *    P1 Freshness Restriction: Stale tape -> STALE_DATA, sizing prohibited, execution blocked
 *    P2 Incomplete Evidence: Missing trend/health -> INSUFFICIENT_DATA / EVIDENCE_INCOMPLETE
 *    P3 Position Management: Owned assets -> TRIM / HOLD
 *    P4 Canonical Decision: Validated buy zone -> ACQUIRE / ACTIONABLE_SETUP
 */

import assert from "node:assert";
import { deriveAssessmentState } from "../lib/assessmentEngine";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import {
  DecisionTrace,
  DomainAssessment,
  FreshnessInfo,
} from "../types/insight";

console.log("=====================================================================");
console.log("ARX TERMINAL — FINANCIAL DECISION PRECEDENCE MATRIX (DP01 - DP16)");
console.log("=====================================================================\n");

const baseDomainAssessments: DomainAssessment[] = [
  {
    domainId: "trend",
    domainName: "Price Trend",
    availability: "AVAILABLE",
    status: "FAVORABLE",
    pointImpact: 35,
    importanceLevel: "HIGH",
    observation: "Strong constructive trend above 50-day SMA",
    modelRule: "Trend rule",
    evidence: [],
    whatWouldChangeAssessment: "Daily close below 50 SMA",
  },
  {
    domainId: "health",
    domainName: "Company Health",
    availability: "AVAILABLE",
    status: "FAVORABLE",
    pointImpact: 35,
    importanceLevel: "HIGH",
    observation: "Pristine balance sheet and solvency metrics",
    modelRule: "Health rule",
    evidence: [],
    whatWouldChangeAssessment: "Deterioration in interest coverage",
  },
  {
    domainId: "smart_money",
    domainName: "Smart Money Flow",
    availability: "AVAILABLE",
    status: "FAVORABLE",
    pointImpact: 15,
    importanceLevel: "MEDIUM",
    observation: "Net institutional accumulation",
    modelRule: "Smart money rule",
    evidence: [],
    whatWouldChangeAssessment: "Heavy institutional distribution",
  },
  {
    domainId: "macro",
    domainName: "Market Outlook",
    availability: "AVAILABLE",
    status: "FAVORABLE",
    pointImpact: 15,
    importanceLevel: "MEDIUM",
    observation: "Favorable market regime",
    modelRule: "Macro rule",
    evidence: [],
    whatWouldChangeAssessment: "Market regime downgrade",
  },
];

const mockActionableTrace: DecisionTrace = {
  symbol: "NVDA",
  decisionState: "ACTIONABLE_SETUP",
  decisionStateLabel: "Buy Zone Triggered",
  canSizeTrade: true,
  isActionable: true,
  disqualificationReason: null,
  timestamp: new Date().toISOString(),
  allowedActions: ["SIZE_TRADE", "SET_ALERT"],
};

const mockNonActionableTrace: DecisionTrace = {
  symbol: "NVDA",
  decisionState: "VALID_SETUP",
  decisionStateLabel: "Awaiting Trigger",
  canSizeTrade: false,
  isActionable: false,
  disqualificationReason: "Price outside optimal buy corridor",
  timestamp: new Date().toISOString(),
  allowedActions: ["SET_ALERT"],
};

// ── DP01: Fresh Actionable Setup ─────────────────────────────────────────────
console.log("Scenario DP01: Fresh actionable setup allows sizing and acquisition...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: mockActionableTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "ACQUIRE");
  assert.strictEqual(state.decisionState, "ACTIONABLE_SETUP");
  assert.strictEqual(state.authoritativeDecisionState, "ACTIONABLE_SETUP");
  assert.strictEqual(state.isActionable, true);
  assert.strictEqual(state.canSizeTrade, true);
  console.log("  ✓ DP01 PASS: Fresh actionable setup enables sizing.\n");
}

// ── DP02: Fresh Non-Actionable Setup ─────────────────────────────────────────
console.log("Scenario DP02: Fresh non-actionable setup preserves pending status...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: mockNonActionableTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "WATCH");
  assert.strictEqual(state.decisionState, "VALID_SETUP");
  assert.strictEqual(state.authoritativeDecisionState, "VALID_SETUP");
  assert.strictEqual(state.isActionable, false);
  assert.strictEqual(state.canSizeTrade, false);
  console.log("  ✓ DP02 PASS: Non-actionable setup prevents execution.\n");
}

// ── DP03: Stale Suppressed Actionable Setup ──────────────────────────────────
console.log("Scenario DP03: Stale data suppresses actionable setup but preserves authoritative state...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: mockActionableTrace,
    freshness: { status: "STALE_HISTORICAL", stalenessDays: 5 },
  });

  assert.strictEqual(state.posture, "RESEARCH");
  assert.strictEqual(state.decisionState, "STALE_DATA", "Composed state is STALE_DATA");
  assert.strictEqual(state.authoritativeDecisionState, "ACTIONABLE_SETUP", "Authoritative state preserved");
  assert.strictEqual(state.isActionable, false, "Execution strictly blocked");
  assert.strictEqual(state.canSizeTrade, false, "Sizing strictly prohibited");
  console.log("  ✓ DP03 PASS: Stale tape composition blocks sizing while preserving canonical state.\n");
}

// ── DP04: Unavailable Feed Fails Closed ──────────────────────────────────────
console.log("Scenario DP04: Unavailable feed fails closed to UNVERIFIED...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: mockActionableTrace,
    freshnessStatus: "UNAVAILABLE",
    freshness: { status: "UNAVAILABLE" },
  });

  assert.strictEqual(state.posture, "RESEARCH");
  assert.strictEqual(state.decisionState, "UNVERIFIED");
  assert.strictEqual(state.isActionable, false);
  assert.strictEqual(state.canSizeTrade, false);
  console.log("  ✓ DP04 PASS: Unavailable data feed fails closed to UNVERIFIED.\n");
}

// ── DP05: Invalid Price Zero ─────────────────────────────────────────────────
console.log("Scenario DP05: Invalid price zero fails closed to UNVERIFIED...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 0,
    changePct: 0,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: mockActionableTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.decisionState, "UNVERIFIED");
  assert.strictEqual(state.isActionable, false);
  assert.strictEqual(state.canSizeTrade, false);
  console.log("  ✓ DP05 PASS: Zero price fails closed to UNVERIFIED.\n");
}

// ── DP06: Invalid Price NaN ──────────────────────────────────────────────────
console.log("Scenario DP06: Invalid price NaN fails closed to UNVERIFIED...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: NaN,
    changePct: 0,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: mockActionableTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.decisionState, "UNVERIFIED");
  assert.strictEqual(state.isActionable, false);
  assert.strictEqual(state.canSizeTrade, false);
  console.log("  ✓ DP06 PASS: NaN price fails closed to UNVERIFIED.\n");
}

// ── DP07: Breached Setup Floor (Unowned Position) ────────────────────────────
console.log("Scenario DP07: Breached setup floor (unowned) forces AVOID posture...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 165.0, // below invalidationPrice 170.0
    changePct: -6.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    invalidationPrice: 170.0,
    isInvalidationBreached: true,
    decisionTrace: mockActionableTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "AVOID", "Posture must be AVOID when breached");
  assert.strictEqual(state.uiStateLabel, "Setup Invalidated");
  assert.strictEqual(state.isActionable, false, "Breach blocks actionability");
  assert.strictEqual(state.canSizeTrade, false, "Breach blocks sizing");
  console.log("  ✓ DP07 PASS: Breached floor forces AVOID posture and blocks execution.\n");
}

// ── DP08: Breached Setup Floor (Owned Position) ──────────────────────────────
console.log("Scenario DP08: Breached setup floor (owned) forces EXIT_REVIEW posture...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 165.0, // below invalidationPrice 170.0
    changePct: -6.5,
    horizon: "SWING",
    ownershipState: "OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    invalidationPrice: 170.0,
    isInvalidationBreached: true,
    decisionTrace: mockActionableTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "EXIT_REVIEW", "Posture must be EXIT_REVIEW for owned position breached");
  assert.strictEqual(state.uiStateLabel, "Thesis Needs Review");
  assert.strictEqual(state.isActionable, false);
  assert.strictEqual(state.canSizeTrade, false);
  console.log("  ✓ DP08 PASS: Breached floor forces EXIT_REVIEW for owned holdings.\n");
}

// ── DP09: Breached Floor + Stale Tape (Owned Position) ────────────────────────
console.log("Scenario DP09: Breached floor + stale tape (owned) NEVER erases exit review warning...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 165.0, // breached
    changePct: -6.5,
    horizon: "SWING",
    ownershipState: "OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    invalidationPrice: 170.0,
    isInvalidationBreached: true,
    decisionTrace: mockActionableTrace,
    freshness: { status: "STALE_HISTORICAL", stalenessDays: 7 }, // both breached and stale!
  });

  // CRITICAL INVARIANT: Stale data must NOT erase higher-priority financial emergency (breached floor)
  assert.strictEqual(state.posture, "EXIT_REVIEW", "P0 emergency EXIT_REVIEW must NOT be silenced by stale tape");
  assert.strictEqual(state.uiStateLabel, "Thesis Needs Review");
  assert.match(state.headlineExplanation, /breached/i, "Headline must explain breach");
  assert.strictEqual(state.decisionState, "STALE_DATA", "Composed state is STALE_DATA");
  assert.strictEqual(state.isActionable, false, "Execution strictly prohibited");
  assert.strictEqual(state.canSizeTrade, false, "Sizing strictly prohibited");
  console.log("  ✓ DP09 PASS: Breached invalidation emergency takes precedence over stale tape labeling.\n");
}

// ── DP10: Breached Floor + Stale Tape (Unowned Position) ──────────────────────
console.log("Scenario DP10: Breached floor + stale tape (unowned) NEVER erases AVOID warning...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 165.0, // breached
    changePct: -6.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    invalidationPrice: 170.0,
    isInvalidationBreached: true,
    decisionTrace: mockActionableTrace,
    freshness: { status: "STALE_HISTORICAL", stalenessDays: 7 }, // both breached and stale!
  });

  // CRITICAL INVARIANT: AVOID posture must be maintained
  assert.strictEqual(state.posture, "AVOID", "P0 AVOID posture must NOT be silenced by stale tape");
  assert.strictEqual(state.uiStateLabel, "Setup Invalidated");
  assert.match(state.headlineExplanation, /breached/i, "Headline must explain breach");
  assert.strictEqual(state.isActionable, false);
  assert.strictEqual(state.canSizeTrade, false);
  console.log("  ✓ DP10 PASS: AVOID posture preserved under stale tape.\n");
}

// ── DP11: Owned Favorable Position (Fresh Tape) ──────────────────────────────
console.log("Scenario DP11: Owned position with favorable factors resolves to HOLD...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 185.0,
    changePct: 2.1,
    horizon: "SWING",
    ownershipState: "OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "HOLD");
  assert.strictEqual(state.uiStateLabel, "Thesis Intact");
  assert.strictEqual(state.isActionable, false, "Owned position does not trigger new buy action");
  console.log("  ✓ DP11 PASS: Owned holding resolves to HOLD.\n");
}

// ── DP12: Owned Deteriorating Position (Fresh Tape) ──────────────────────────
console.log("Scenario DP12: Owned deteriorating position resolves to TRIM...");
{
  const deterioratingDomains: DomainAssessment[] = baseDomainAssessments.map(d =>
    d.domainId === "trend" || d.domainId === "health"
      ? { ...d, status: "UNFAVORABLE" as const }
      : d
  );

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 175.0,
    changePct: -3.0,
    horizon: "SWING",
    ownershipState: "OWNED",
    ownershipSource: "USER_DECLARED",
    domains: deterioratingDomains,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "TRIM");
  assert.strictEqual(state.uiStateLabel, "Consider Trimming");
  console.log("  ✓ DP12 PASS: Deteriorating holding triggers TRIM.\n");
}

// ── DP13: Conflicting Trace (Actionable Trace with Stale Telemetry) ───────────
console.log("Scenario DP13: Conflicting trace (Actionable Trace with Stale Telemetry) resolves fail-closed...");
{
  const conflictingTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Trigger Fired",
    isActionable: true,
    canSizeTrade: true,
    allowedActions: ["SIZE_TRADE"],
    disqualificationReason: null,
  };

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: conflictingTrace,
    freshness: { status: "STALE", stalenessDays: 6 },
  });

  assert.strictEqual(state.decisionState, "STALE_DATA");
  assert.strictEqual(state.authoritativeDecisionState, "ACTIONABLE_SETUP");
  assert.strictEqual(state.isActionable, false, "Stale telemetry must override nominal actionable trace");
  assert.strictEqual(state.canSizeTrade, false, "Sizing must be disabled");
  console.log("  ✓ DP13 PASS: Stale telemetry fail-closed overrides optimistic trace.\n");
}

// ── DP14: Conflicting Trace (UNVERIFIED State with True Actionable Flag) ─────
console.log("Scenario DP14: Conflicting trace (UNVERIFIED state with True Actionable flag) fails closed...");
{
  const anomalousTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "UNVERIFIED",
    decisionStateLabel: "Unverified Feed",
    isActionable: true, // anomalous contradictory flag!
    canSizeTrade: true,
    allowedActions: [],
    disqualificationReason: "Feed disconnected",
  };

  const insight = generateQuantitativeInsight(
    "NVDA",
    "NVIDIA Corp",
    182.0,
    1.5,
    80,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    undefined,
    "live",
    undefined,
    anomalousTrace,
    undefined,
    "LIVE"
  );

  assert.strictEqual(insight.verdict, "UNVERIFIED", "UNVERIFIED decisionState must force UNVERIFIED verdict");
  assert.strictEqual(insight.terminalState.posture, "RESEARCH", "Degraded decision must demote posture to RESEARCH");
  console.log("  ✓ DP14 PASS: Contradictory UNVERIFIED trace fails closed.\n");
}

// ── DP15: Conflicting Trace (Actionable Trace with Live Price Floor Breach) ──
console.log("Scenario DP15: Actionable trace with live price floor breach fails closed to AVOID...");
{
  const stalePositiveTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Breakout",
    isActionable: true,
    canSizeTrade: true,
    allowedActions: ["SIZE_TRADE"],
    disqualificationReason: null,
  };

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 160.0, // live price has crashed below invalidation floor
    changePct: -12.0,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    invalidationPrice: 172.0,
    isInvalidationBreached: true,
    decisionTrace: stalePositiveTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "AVOID", "Mathematical price breach must override stale positive trace");
  assert.strictEqual(state.isActionable, false, "Actionability must be revoked");
  assert.strictEqual(state.canSizeTrade, false, "Sizing must be revoked");
  console.log("  ✓ DP15 PASS: Live price breach strictly overrides optimistic model trace.\n");
}

// ── DP16: Incomplete Trend Evidence ──────────────────────────────────────────
console.log("Scenario DP16: Incomplete trend evidence (<50 sessions) blocks setup...");
{
  const incompleteDomains: DomainAssessment[] = baseDomainAssessments.map(d =>
    d.domainId === "trend"
      ? { ...d, availability: "UNAVAILABLE" as const, status: "UNAVAILABLE" as const }
      : d
  );

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: incompleteDomains,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "RESEARCH");
  assert.strictEqual(state.decisionState, "INSUFFICIENT_DATA");
  assert.strictEqual(state.isActionable, false);
  assert.strictEqual(state.canSizeTrade, false);
  console.log("  ✓ DP16 PASS: Missing trend evidence blocks setup.\n");
}

console.log("=====================================================================");
console.log("ALL 16 DECISION PRECEDENCE MATRIX SCENARIOS (DP01 - DP16) PASSED");
console.log("=====================================================================\n");
