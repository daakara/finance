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
import { deriveAssessmentState, assertDecisionAuthorityConsistency } from "../lib/assessmentEngine";
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

// ── DP17: Adversarial Contradiction — Zero Evaluated Evidence (agreement.evaluated === 0) ──
console.log("Scenario DP17: Adversarial contradiction — zero evaluated evidence blocks sizing/actionability...");
{
  const contradictoryTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Rogue Actionable Approval",
    canSizeTrade: true, // rogue upstream claim
    isActionable: true, // rogue upstream claim
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0, // valid fresh price
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: [], // agreement.evaluated === 0
    isInvalidationBreached: false, // no floor breach
    freshness: { status: "LIVE", stalenessDays: 0 },
    decisionTrace: contradictoryTrace,
  });

  assert.strictEqual(state.factorAgreement.evaluated, 0, "Evaluated factors must be 0");
  assert.strictEqual(state.posture, "RESEARCH", "Posture must be RESEARCH when evaluated === 0");
  assert.strictEqual(state.decisionState, "UNVERIFIED", "Effective decision state must be UNVERIFIED");
  assert.strictEqual(state.canSizeTrade, false, "canSizeTrade must be blocked fail-closed");
  assert.strictEqual(state.isActionable, false, "isActionable must be blocked fail-closed");
  const sizeAction = state.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false, "size_trade action must be disabled");
  assert.strictEqual(state.authoritativeDecisionState, "ACTIONABLE_SETUP", "Authoritative provenance preserved without authorization override");
  console.log("  ✓ DP17 PASS: Zero evaluated evidence strictly blocks sizing and actionability despite rogue upstream claim.\n");
}

// ── DP18: Adversarial Contradiction — Unavailable Feed Overrides Upstream Claims ──
console.log("Scenario DP18: Adversarial contradiction — unavailable feed blocks sizing/actionability...");
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
    freshnessStatus: "UNAVAILABLE",
    freshness: { status: "UNAVAILABLE" },
    decisionTrace: mockActionableTrace, // claims canSizeTrade: true, isActionable: true
  });

  assert.strictEqual(state.posture, "RESEARCH");
  assert.strictEqual(state.decisionState, "UNVERIFIED");
  assert.strictEqual(state.canSizeTrade, false);
  assert.strictEqual(state.isActionable, false);
  const sizeAction = state.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false);
  console.log("  ✓ DP18 PASS: Unavailable feed strictly blocks sizing and actionability.\n");
}

// ── DP19: Adversarial Contradiction — Invalid Price Zero Overrides Upstream Claims ──
console.log("Scenario DP19: Adversarial contradiction — invalid price zero blocks sizing/actionability...");
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
    freshness: { status: "LIVE", stalenessDays: 0 },
    decisionTrace: mockActionableTrace, // claims canSizeTrade: true, isActionable: true
  });

  assert.strictEqual(state.decisionState, "UNVERIFIED");
  assert.strictEqual(state.canSizeTrade, false);
  assert.strictEqual(state.isActionable, false);
  const sizeAction = state.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false);
  console.log("  ✓ DP19 PASS: Zero price strictly blocks sizing and actionability.\n");
}

// ── DP20: Adversarial Contradiction — Breached Floor Overrides Upstream Claims ──
console.log("Scenario DP20: Adversarial contradiction — breached floor blocks sizing/actionability...");
{
  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 160.0,
    changePct: -8.0,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    invalidationPrice: 170.0,
    isInvalidationBreached: true,
    freshness: { status: "LIVE", stalenessDays: 0 },
    decisionTrace: mockActionableTrace, // claims canSizeTrade: true, isActionable: true
  });

  assert.strictEqual(state.posture, "AVOID");
  assert.strictEqual(state.decisionState, "EVIDENCE_INCOMPLETE");
  assert.strictEqual(state.canSizeTrade, false);
  assert.strictEqual(state.isActionable, false);
  const sizeAction = state.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false);
  console.log("  ✓ DP20 PASS: Breached floor strictly blocks sizing and actionability.\n");
}

// ── DP21: Adversarial Contradiction — Incomplete Mandatory Evidence Overrides Claims ──
console.log("Scenario DP21: Adversarial contradiction — missing trend blocks sizing/actionability...");
{
  const missingTrendDomains: DomainAssessment[] = baseDomainAssessments.map(d =>
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
    domains: missingTrendDomains,
    freshness: { status: "LIVE", stalenessDays: 0 },
    decisionTrace: mockActionableTrace, // claims canSizeTrade: true, isActionable: true
  });

  assert.strictEqual(state.posture, "RESEARCH");
  assert.strictEqual(state.decisionState, "INSUFFICIENT_DATA");
  assert.strictEqual(state.canSizeTrade, false);
  assert.strictEqual(state.isActionable, false);
  const sizeAction = state.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false);
  console.log("  ✓ DP21 PASS: Missing trend strictly blocks sizing and actionability.\n");
}

// ── DP22: Fallback authoritativeDecisionState Construction Without Upstream Trace ──
console.log("Scenario DP22: Fallback authoritativeDecisionState construction never fabricates VALID_SETUP...");
{
  // A. Zero evaluated factors -> UNVERIFIED (never VALID_SETUP)
  const stateZeroEval = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: [],
    freshness: { status: "LIVE", stalenessDays: 0 },
  });
  assert.strictEqual(stateZeroEval.authoritativeDecisionState, "UNVERIFIED");

  // B. Unavailable feed -> UNVERIFIED (never VALID_SETUP)
  const stateUnavail = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    freshnessStatus: "UNAVAILABLE",
    freshness: { status: "UNAVAILABLE" },
  });
  assert.strictEqual(stateUnavail.authoritativeDecisionState, "UNVERIFIED");

  // C. Invalid price zero -> UNVERIFIED (never VALID_SETUP)
  const stateZeroPrice = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 0,
    changePct: 0,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });
  assert.strictEqual(stateZeroPrice.authoritativeDecisionState, "UNVERIFIED");

  // D. Missing trend -> INSUFFICIENT_DATA (never VALID_SETUP)
  const missingTrend: DomainAssessment[] = baseDomainAssessments.map(d =>
    d.domainId === "trend" ? { ...d, availability: "UNAVAILABLE" as const, status: "UNAVAILABLE" as const } : d
  );
  const stateNoTrend = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: missingTrend,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });
  assert.strictEqual(stateNoTrend.authoritativeDecisionState, "INSUFFICIENT_DATA");

  // E. Breached floor -> EVIDENCE_INCOMPLETE (never VALID_SETUP)
  const stateBreached = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 160.0,
    changePct: -8.0,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    invalidationPrice: 170.0,
    isInvalidationBreached: true,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });
  assert.strictEqual(stateBreached.authoritativeDecisionState, "EVIDENCE_INCOMPLETE");

  // F. Genuine unbreached setup awaiting trigger -> VALID_SETUP
  const stateValid = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 182.0,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });
  assert.strictEqual(stateValid.authoritativeDecisionState, "VALID_SETUP");
  console.log("  ✓ DP22 PASS: Fallback authoritativeDecisionState strictly respects data eligibility without fabrication.\n");
}

// ── DP23: Downstream Consumer Non-Override Verification ──
console.log("Scenario DP23: Downstream consumer generateQuantitativeInsight does not override fail-closed state...");
{
  const contradictoryTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Rogue Actionable Approval",
    canSizeTrade: true,
    isActionable: true,
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  const insight = generateQuantitativeInsight(
    "NVDA",
    "NVIDIA Corp",
    182.0,
    1.5,
    90,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    [], // 0 candles -> evaluated === 0
    "live",
    undefined,
    contradictoryTrace,
    undefined,
    "LIVE"
  );

  assert.strictEqual(insight.terminalState.canSizeTrade, false, "Insight terminalState canSizeTrade must be false");
  assert.strictEqual(insight.terminalState.isActionable, false, "Insight terminalState isActionable must be false");
  const sizeAction = insight.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false, "Insight size_trade action must be disabled");
  assert.strictEqual(insight.verdict, "UNVERIFIED", "Insight verdict must be UNVERIFIED (not ACTIONABLE_BUY_ZONE)");
  console.log("  ✓ DP23 PASS: Downstream insightGenerator strictly blocks sizing and actionability on insufficient evidence.\n");
}

// ── DP24: Contradiction: LIMITED Eligibility + Rogue Upstream Actionable Trace ──
console.log("Scenario DP24: LIMITED eligibility strictly blocks sizing and actionability despite rogue upstream claim...");
{
  const rogueTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
    canSizeTrade: true,
    isActionable: true,
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  const limitedDomains: DomainAssessment[] = baseDomainAssessments.map(d =>
    d.domainId === "smart_money" ? { ...d, status: "UNAVAILABLE", availability: "UNAVAILABLE" } : d
  );

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 184.2,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: limitedDomains,
    decisionTrace: rogueTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.overallEligibility, "LIMITED", "Overall eligibility must be LIMITED");
  assert.strictEqual(state.posture, "RESEARCH", "LIMITED eligibility must demote posture to RESEARCH");
  assert.strictEqual(state.decisionState, "EVIDENCE_INCOMPLETE", "LIMITED eligibility decisionState must be EVIDENCE_INCOMPLETE");
  assert.strictEqual(state.authoritativeDecisionState, "ACTIONABLE_SETUP", "Upstream model proposal preserved in authoritativeDecisionState");
  assert.strictEqual(state.canSizeTrade, false, "canSizeTrade must be strictly false on LIMITED eligibility");
  assert.strictEqual(state.isActionable, false, "isActionable must be strictly false on LIMITED eligibility");
  assert.strictEqual(state.uiStateLabel.toLowerCase().includes("actionable"), false, "uiStateLabel must not claim actionable on LIMITED eligibility");
  const sizeAction = state.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false, "size_trade action must be disabled");
  console.log("  ✓ DP24 PASS: LIMITED eligibility strictly blocks sizing and actionability despite rogue upstream claim.\n");
}

// ── DP25: Contradiction: Missing Core Evidence + Rogue Upstream Actionable Trace ──
console.log("Scenario DP25: Missing core evidence strictly blocks sizing and actionability despite rogue upstream claim...");
{
  const rogueTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
    canSizeTrade: true,
    isActionable: true,
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  // Missing core trend domain
  const partialDomains: DomainAssessment[] = baseDomainAssessments.filter(d => d.domainId !== "trend");

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 184.2,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: partialDomains,
    decisionTrace: rogueTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "RESEARCH", "Missing core evidence must demote posture to RESEARCH");
  assert.strictEqual(state.decisionState, "INSUFFICIENT_DATA", "Missing trend decisionState must be INSUFFICIENT_DATA");
  assert.strictEqual(state.canSizeTrade, false, "canSizeTrade must be strictly false on missing core evidence");
  assert.strictEqual(state.isActionable, false, "isActionable must be strictly false on missing core evidence");
  console.log("  ✓ DP25 PASS: Missing core evidence strictly blocks sizing and actionability despite rogue upstream claim.\n");
}

// ── DP26: Contradiction: Missing Health Domain + Rogue Upstream Actionable Trace ──
console.log("Scenario DP26: Missing fundamental health strictly blocks sizing despite rogue upstream claim...");
{
  const rogueTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
    canSizeTrade: true,
    isActionable: true,
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  // Missing financial health domain
  const noHealthDomains: DomainAssessment[] = baseDomainAssessments.filter(d => d.domainId !== "health");

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 184.2,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: noHealthDomains,
    decisionTrace: rogueTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "RESEARCH", "Missing health must demote posture to RESEARCH");
  assert.strictEqual(state.decisionState, "EVIDENCE_INCOMPLETE", "Missing health decisionState must be EVIDENCE_INCOMPLETE");
  assert.strictEqual(state.canSizeTrade, false, "canSizeTrade must be strictly false on missing health");
  assert.strictEqual(state.isActionable, false, "isActionable must be strictly false on missing health");
  console.log("  ✓ DP26 PASS: Missing fundamental health strictly blocks sizing despite rogue upstream claim.\n");
}

// ── DP27: Contradiction: Owned Position (HOLD) + Upstream Buy Permission ──
console.log("Scenario DP27: Owned holding (HOLD) preserves portfolio actions and blocks new trade sizing...");
{
  const rogueBuyTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
    canSizeTrade: true,
    isActionable: true,
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 184.2,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: rogueBuyTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "HOLD", "Owned holding with favorable factors must be HOLD");
  assert.strictEqual(state.canSizeTrade, false, "canSizeTrade for new trade must be strictly false for OWNED holding");
  assert.strictEqual(state.isActionable, false, "isActionable must be strictly false for new trade on OWNED holding");
  assert.strictEqual(state.decisionState, "VALID_SETUP", "decisionState cannot be ACTIONABLE_SETUP on OWNED holding");
  assert.strictEqual(state.authoritativeDecisionState, "ACTIONABLE_SETUP", "authoritativeDecisionState preserves upstream trace");
  const sizeAction = state.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false, "size_trade action must be disabled on OWNED holding");
  const reviewAction = state.availableActions.find(a => a.id === "review_invalidation");
  assert.strictEqual(reviewAction?.enabled, true, "review_invalidation portfolio action must remain enabled on OWNED holding");
  console.log("  ✓ DP27 PASS: Owned holding (HOLD) preserves portfolio actions and blocks new trade sizing.\n");
}

// ── DP28: Contradiction: Owned Position Deteriorating (TRIM) + Upstream Buy Permission ──
console.log("Scenario DP28: Deteriorating holding (TRIM) blocks new trade sizing and enables thesis review...");
{
  const rogueBuyTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
    canSizeTrade: true,
    isActionable: true,
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  const unfavorableDomains: DomainAssessment[] = baseDomainAssessments.map(d =>
    d.domainId === "trend" || d.domainId === "health" ? { ...d, status: "UNFAVORABLE" as const } : d
  );

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 184.2,
    changePct: -3.5,
    horizon: "SWING",
    ownershipState: "OWNED",
    ownershipSource: "USER_DECLARED",
    domains: unfavorableDomains,
    decisionTrace: rogueBuyTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.posture, "TRIM", "Deteriorating owned holding must be TRIM");
  assert.strictEqual(state.canSizeTrade, false, "canSizeTrade must be strictly false for TRIM");
  assert.strictEqual(state.isActionable, false, "isActionable must be strictly false for TRIM");
  const sizeAction = state.availableActions.find(a => a.id === "size_trade");
  assert.strictEqual(sizeAction?.enabled, false, "size_trade action must be disabled for TRIM");
  const reviewAction = state.availableActions.find(a => a.id === "review_invalidation");
  assert.strictEqual(reviewAction?.enabled, true, "review_invalidation portfolio action must remain enabled for TRIM");
  console.log("  ✓ DP28 PASS: Deteriorating holding (TRIM) blocks new trade sizing and enables thesis review.\n");
}

// ── DP29: Contradiction: Invalidation Breached + Upstream ACTIONABLE_SETUP in insightGenerator ──
console.log("Scenario DP29: Breached floor + upstream ACTIONABLE_SETUP emits AVOID_STAGE_4 without actionable override...");
{
  const rogueTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
    canSizeTrade: true,
    isActionable: true,
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  const insight = generateQuantitativeInsight(
    "NVDA",
    "NVIDIA Corp",
    160.0, // breached below stopLoss 170.0
    -7.0,
    88,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    undefined,
    "live",
    undefined,
    rogueTrace,
    {
      current_price: 160.0,
      setup_pattern: "PULLBACK",
      optimal_entry_min: 182.0,
      optimal_entry_max: 184.0,
      breakout_pivot: 182.0,
      stop_loss: 170.0,
      take_profit_1: 210.0,
      take_profit_2: 230.0,
      risk_reward_ratio: 3.2,
      entry_thesis: "Pullback to 50 SMA",
      invalidation_condition: "Stop loss breached",
      stage_phase: "Stage 2",
      vcp_contraction_status: "Forming base",
    },
    "LIVE"
  );

  assert.strictEqual(insight.terminalState.posture, "AVOID", "Breached setup must force AVOID posture");
  assert.strictEqual(insight.terminalState.decisionState, "EVIDENCE_INCOMPLETE", "Breached setup decisionState must be EVIDENCE_INCOMPLETE");
  assert.strictEqual(insight.terminalState.authoritativeDecisionState, "ACTIONABLE_SETUP", "Upstream trace preserved in authoritativeDecisionState");
  assert.strictEqual(insight.terminalState.canSizeTrade, false, "canSizeTrade must be false on breached setup");
  assert.strictEqual(insight.terminalState.isActionable, false, "isActionable must be false on breached setup");
  assert.strictEqual(insight.verdict, "AVOID_STAGE_4", "finalVerdict must be AVOID_STAGE_4");
  assert.strictEqual(insight.verdictLabel, "Setup Invalidated", "verdictLabel must reflect Setup Invalidated (never Actionable)");
  console.log("  ✓ DP29 PASS: Breached floor + upstream ACTIONABLE_SETUP emits AVOID_STAGE_4 without actionable override.\n");
}

// ── DP30: Contradiction: Stale Tape + Upstream ACTIONABLE_SETUP in insightGenerator ──
console.log("Scenario DP30: Stale tape + upstream ACTIONABLE_SETUP strictly preserves STALE_DATA and blocks sizing...");
{
  const rogueTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
    canSizeTrade: true,
    isActionable: true,
    disqualificationReason: null,
    allowedActions: ["SIZE_TRADE", "SET_ALERT"],
  };

  const insight = generateQuantitativeInsight(
    "NVDA",
    "NVIDIA Corp",
    184.0,
    1.5,
    88,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    undefined,
    "historical",
    undefined,
    rogueTrace,
    undefined,
    { status: "STALE_HISTORICAL", stalenessDays: 7 }
  );

  assert.strictEqual(insight.terminalState.posture, "RESEARCH", "Stale tape must demote posture to RESEARCH");
  assert.strictEqual(insight.terminalState.decisionState, "STALE_DATA", "decisionState must be STALE_DATA");
  assert.strictEqual(insight.terminalState.authoritativeDecisionState, "ACTIONABLE_SETUP", "authoritativeDecisionState preserves upstream model state");
  assert.strictEqual(insight.terminalState.canSizeTrade, false, "canSizeTrade must be false on stale tape");
  assert.strictEqual(insight.terminalState.isActionable, false, "isActionable must be false on stale tape");
  assert.strictEqual(insight.verdict, "WAIT_FOR_TRIGGER", "finalVerdict must be WAIT_FOR_TRIGGER (never ACTIONABLE_BUY_ZONE)");
  assert.strictEqual(insight.verdictLabel, "Stale Historical Tape (> 4 Days)", "verdictLabel must report Stale Tape");
  console.log("  ✓ DP30 PASS: Stale tape + upstream ACTIONABLE_SETUP strictly preserves STALE_DATA and blocks sizing.\n");
}

// ── DP31: Contradiction: Conflicting Actionability and Sizing Flags ──
console.log("Scenario DP31: Conflicting actionability and sizing flags strictly fail closed...");
{
  const conflictingTrace: DecisionTrace = {
    symbol: "NVDA",
    decisionState: "ACTIONABLE_SETUP",
    decisionStateLabel: "Actionable Setup",
    canSizeTrade: false, // contradiction: actionable but sizing forbidden!
    isActionable: true,
    disqualificationReason: "Volatility threshold exceeded",
    allowedActions: ["SET_ALERT"],
  };

  const state = deriveAssessmentState({
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    currentPrice: 184.2,
    changePct: 1.5,
    horizon: "SWING",
    ownershipState: "NOT_OWNED",
    ownershipSource: "USER_DECLARED",
    domains: baseDomainAssessments,
    decisionTrace: conflictingTrace,
    freshness: { status: "LIVE", stalenessDays: 0 },
  });

  assert.strictEqual(state.canSizeTrade, false, "canSizeTrade must be false when trace forbids sizing");
  assert.strictEqual(state.isActionable, false, "isActionable must fail closed when sizing is blocked");
  assert.strictEqual(state.posture, "WATCH", "Posture must resolve to WATCH (not ACQUIRE)");
  assert.strictEqual(state.decisionState, "VALID_SETUP", "decisionState must resolve to VALID_SETUP (not ACTIONABLE_SETUP)");
  console.log("  ✓ DP31 PASS: Conflicting actionability and sizing flags strictly fail closed.\n");
}

// ── DP32: Consistency Invariant Rigorous Unit Verification ──
console.log("Scenario DP32: assertDecisionAuthorityConsistency rejects every contradiction vector...");
{
  // Vector A: canSizeTrade true on non-ACQUIRE posture
  assert.throws(
    () => {
      assertDecisionAuthorityConsistency({
        posture: "RESEARCH",
        decisionState: "VALID_SETUP",
        canSizeTrade: true,
        isActionable: false,
        uiStateLabel: "Research State",
      });
    },
    /canSizeTrade is true while posture is RESEARCH/,
    "Invariant must throw on canSizeTrade true with RESEARCH posture"
  );

  // Vector B: canSizeTrade true on OWNED position
  assert.throws(
    () => {
      assertDecisionAuthorityConsistency({
        posture: "ACQUIRE",
        decisionState: "ACTIONABLE_SETUP",
        canSizeTrade: true,
        isActionable: true,
        uiStateLabel: "Actionable Setup",
        ownershipState: "OWNED",
      });
    },
    /canSizeTrade is true on an OWNED position/,
    "Invariant must throw on canSizeTrade true with OWNED position"
  );

  // Vector C: isActionable true while canSizeTrade false
  assert.throws(
    () => {
      assertDecisionAuthorityConsistency({
        posture: "WATCH",
        decisionState: "VALID_SETUP",
        canSizeTrade: false,
        isActionable: true,
        uiStateLabel: "Valid Setup",
      });
    },
    /isActionable is true while canSizeTrade is false/,
    "Invariant must throw on isActionable true with canSizeTrade false"
  );

  // Vector D: decisionState ACTIONABLE_SETUP while canSizeTrade false
  assert.throws(
    () => {
      assertDecisionAuthorityConsistency({
        posture: "WATCH",
        decisionState: "ACTIONABLE_SETUP",
        canSizeTrade: false,
        isActionable: false,
        uiStateLabel: "Valid Setup",
      });
    },
    /decisionState is ACTIONABLE_SETUP while canSizeTrade is false/,
    "Invariant must throw on decisionState ACTIONABLE_SETUP with canSizeTrade false"
  );

  // Vector E: finalVerdict ACTIONABLE_BUY_ZONE while canSizeTrade false
  assert.throws(
    () => {
      assertDecisionAuthorityConsistency({
        posture: "WATCH",
        decisionState: "VALID_SETUP",
        canSizeTrade: false,
        isActionable: false,
        uiStateLabel: "Valid Setup",
        finalVerdict: "ACTIONABLE_BUY_ZONE",
      });
    },
    /finalVerdict is ACTIONABLE_BUY_ZONE while canSizeTrade is false/,
    "Invariant must throw on finalVerdict ACTIONABLE_BUY_ZONE with canSizeTrade false"
  );

  // Vector F: uiStateLabel claims actionable while canSizeTrade false
  assert.throws(
    () => {
      assertDecisionAuthorityConsistency({
        posture: "WATCH",
        decisionState: "VALID_SETUP",
        canSizeTrade: false,
        isActionable: false,
        uiStateLabel: "Actionable Setup — Buy Zone Confirmed",
      });
    },
    /uiStateLabel claims actionable/,
    "Invariant must throw on uiStateLabel claiming actionable with canSizeTrade false"
  );

  // Vector G: size_trade action enabled while canSizeTrade false
  assert.throws(
    () => {
      assertDecisionAuthorityConsistency({
        posture: "WATCH",
        decisionState: "VALID_SETUP",
        canSizeTrade: false,
        isActionable: false,
        uiStateLabel: "Valid Setup",
        availableActions: [
          { id: "size_trade", type: "SIZE_POSITION", label: "Size Trade", enabled: true },
        ],
      });
    },
    /size_trade action is enabled while canSizeTrade is false/,
    "Invariant must throw on size_trade action enabled with canSizeTrade false"
  );

  console.log("  ✓ DP32 PASS: assertDecisionAuthorityConsistency strictly rejects every contradiction vector.\n");
}

console.log("=====================================================================");
console.log("ALL 32 DECISION PRECEDENCE MATRIX SCENARIOS (DP01 - DP32) PASSED");
console.log("=====================================================================\n");
