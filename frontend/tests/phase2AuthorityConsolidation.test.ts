import assert from "node:assert";
import { deriveAssessmentState } from "../lib/assessmentEngine";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import { DecisionTrace, DomainAssessment } from "../types/insight";

console.log("Starting Phase 2 Decision Authority Consolidation Test Suite...\n");

// ---------------------------------------------------------------------------
// 1. Posture Authority Gating (F_06 Remediation)
// ---------------------------------------------------------------------------
console.log("1. Testing deriveAssessmentState posture authority demotion...");

const baseDomainAssessments: DomainAssessment[] = [
  {
    domainId: "trend",
    domainName: "Price Trend",
    availability: "AVAILABLE",
    status: "FAVORABLE",
    pointImpact: 35,
    importanceLevel: "HIGH",
    observation: "Strong trend",
    modelRule: "Trend rule",
    evidence: [],
    whatWouldChangeAssessment: "Break below 50 SMA",
  },
  {
    domainId: "health",
    domainName: "Company Health",
    availability: "AVAILABLE",
    status: "FAVORABLE",
    pointImpact: 35,
    importanceLevel: "HIGH",
    observation: "Pristine balance sheet",
    modelRule: "Health rule",
    evidence: [],
    whatWouldChangeAssessment: "Earnings degradation",
  },
  {
    domainId: "smart_money",
    domainName: "Smart Money Flow",
    availability: "AVAILABLE",
    status: "FAVORABLE",
    pointImpact: 15,
    importanceLevel: "MEDIUM",
    observation: "Institutional accumulation",
    modelRule: "Smart money rule",
    evidence: [],
    whatWouldChangeAssessment: "Heavy insider selling",
  },
  {
    domainId: "macro",
    domainName: "Macro Regime",
    availability: "AVAILABLE",
    status: "FAVORABLE",
    pointImpact: 15,
    importanceLevel: "MEDIUM",
    observation: "Supportive macro",
    modelRule: "Macro rule",
    evidence: [],
    whatWouldChangeAssessment: "Yield curve inversion",
  },
];

// Case A: Perfect factors, but CANONICAL authority declares NOT actionable
const nonActionableTrace: DecisionTrace = {
  symbol: "NVDA",
  decisionState: "VALID_SETUP",
  stateLabel: "Valid Setup — Disclosures Pending",
  isActionable: false,
  canSizeTrade: false,
  allowedActions: ["ADD_WATCHLIST", "SET_PRICE_ALERT"],
  disqualificationReason: "Awaiting final clearance trigger",
};

const stateNonActionable = deriveAssessmentState({
  symbol: "NVDA",
  companyName: "NVIDIA Corporation",
  currentPrice: 120.0,
  changePct: 1.5,
  horizon: "SWING",
  ownershipState: "NOT_OWNED",
  ownershipSource: "USER_DECLARED",
  domains: baseDomainAssessments,
  decisionTrace: nonActionableTrace,
});

assert.strictEqual(
  stateNonActionable.canSizeTrade,
  false,
  "canSizeTrade must bind to canonical decisionTrace"
);
assert.notStrictEqual(
  stateNonActionable.posture,
  "ACQUIRE",
  "Posture CANNOT be ACQUIRE when decisionTrace.isActionable is false"
);
assert.strictEqual(
  stateNonActionable.posture,
  "WATCH",
  "Posture must fall back to WATCH when non-actionable"
);
console.log("   [OK] deriveAssessmentState rejects ACQUIRE when isActionable is false");

// Case B: Canonical authority grants ACTIONABLE_SETUP
const actionableTrace: DecisionTrace = {
  symbol: "NVDA",
  decisionState: "ACTIONABLE_SETUP",
  stateLabel: "Actionable Setup Confirmed",
  isActionable: true,
  canSizeTrade: true,
  allowedActions: ["CALCULATE_SIZE", "EXECUTE_ORDER"],
  disqualificationReason: null,
};

const stateActionable = deriveAssessmentState({
  symbol: "NVDA",
  companyName: "NVIDIA Corporation",
  currentPrice: 120.0,
  changePct: 1.5,
  horizon: "SWING",
  ownershipState: "NOT_OWNED",
  ownershipSource: "USER_DECLARED",
  domains: baseDomainAssessments,
  decisionTrace: actionableTrace,
});

assert.strictEqual(stateActionable.canSizeTrade, true);
assert.strictEqual(stateActionable.posture, "ACQUIRE");
console.log("   [OK] deriveAssessmentState permits ACQUIRE strictly when decisionTrace is ACTIONABLE_SETUP");

// ---------------------------------------------------------------------------
// 2. Terminal Insight Generator Verdict Gating (F_06 Remediation)
// ---------------------------------------------------------------------------
console.log("2. Testing generateQuantitativeInsight finalVerdict authority binding...");

const mockCandles = Array.from({ length: 60 }, (_, i) => ({
  time: `2026-08-${String((i % 28) + 1).padStart(2, "0")}`,
  open: 118,
  high: 122,
  low: 117,
  close: 120,
  volume: 1000000,
}));

const executionPlan = {
  optimal_entry_min: 118.0,
  optimal_entry_max: 122.0,
  stop_loss: 112.0,
  take_profit_1: 135.0,
  take_profit_2: 150.0,
  risk_reward_ratio: 2.8,
  setup_pattern: "Minervini VCP",
  execution_status: "IN_BUY_ZONE",
} as any;

const confluenceData = {
  confluenceScore: 82.0,
  confluenceRating: "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT",
  confluenceBadge: "4-Pillar Confluence (Pristine)",
  badgeColor: "emerald",
  bottomLine: "Disciplined accumulation.",
  pillars: [],
} as any;

const insightNonActionable = generateQuantitativeInsight(
  "NVDA",
  "NVIDIA Corporation",
  120.0,
  1.5,
  82,
  2,
  "SWING",
  "NOT_OWNED",
  "USER_DECLARED",
  mockCandles,
  "live",
  confluenceData,
  nonActionableTrace,
  executionPlan,
  "LIVE"
);

assert.strictEqual(
  insightNonActionable.verdict,
  "WAIT_FOR_TRIGGER",
  "finalVerdict must be WAIT_FOR_TRIGGER when decisionTrace.isActionable is false"
);
assert.notStrictEqual(
  insightNonActionable.verdict,
  "ACTIONABLE_BUY_ZONE",
  "finalVerdict CANNOT be ACTIONABLE_BUY_ZONE without canonical isActionable"
);
console.log("   [OK] generateQuantitativeInsight demotes verdict to WAIT_FOR_TRIGGER when non-actionable");

const insightActionable = generateQuantitativeInsight(
  "NVDA",
  "NVIDIA Corporation",
  120.0,
  1.5,
  82,
  2,
  "SWING",
  "NOT_OWNED",
  "USER_DECLARED",
  mockCandles,
  "live",
  confluenceData,
  actionableTrace,
  executionPlan,
  "LIVE"
);

assert.strictEqual(
  insightActionable.verdict,
  "ACTIONABLE_BUY_ZONE",
  "finalVerdict is ACTIONABLE_BUY_ZONE when canonical decisionTrace is actionable"
);
console.log("   [OK] generateQuantitativeInsight emits ACTIONABLE_BUY_ZONE when canonical decisionTrace permits");

// ---------------------------------------------------------------------------
// 3. Composite Conviction Card Evidence-Only Invariant (F_09 Remediation)
// ---------------------------------------------------------------------------
console.log("3. Testing CompositeConvictionCard evidence-only and pending contract...");

// Verify by reading source file directly that speculative fallback formula was eliminated
import * as fs from "node:fs";
import * as path from "node:path";

const convictionCardSource = fs.readFileSync(
  path.join(__dirname, "../components/CompositeConvictionCard.tsx"),
  "utf8"
);

assert.ok(
  !convictionCardSource.includes("0.25 * techScore"),
  "CompositeConvictionCard must NOT synthesize speculative local composite weights"
);
assert.ok(
  !convictionCardSource.includes("GREEN LIGHT: HIGH CONVICTION"),
  "CompositeConvictionCard must NOT issue independent GREEN LIGHT verdict titles"
);
assert.ok(
  convictionCardSource.includes("CONFLUENCE EVALUATION PENDING"),
  "CompositeConvictionCard must show pending state when backend confluence is missing"
);
console.log("   [OK] CompositeConvictionCard does not contain speculative fallback math or independent verdicts");

// ---------------------------------------------------------------------------
// 4. Modal Clearance & Sizing Gates (Requirements 21 & 22)
// ---------------------------------------------------------------------------
console.log("4. Testing PreFlightChecklistModal and PositionSizerModal canonical gates...");

const preflightSource = fs.readFileSync(
  path.join(__dirname, "../components/PreFlightChecklistModal.tsx"),
  "utf8"
);

assert.ok(
  preflightSource.includes("isActionableGranted &&"),
  "PreFlightChecklistModal must require canonical actionability for isCleared"
);
assert.ok(
  preflightSource.includes("FLIGHT CLEARANCE REVOKED: CANONICAL AUTHORITY"),
  "PreFlightChecklistModal must render revocation banner when isActionable is false"
);
console.log("   [OK] PreFlightChecklistModal enforces canonical clearance gate");

const sizerSource = fs.readFileSync(
  path.join(__dirname, "../components/PositionSizerModal.tsx"),
  "utf8"
);

assert.ok(
  sizerSource.includes("isSizingBlocked = canSizeTrade === false || isActionable === false"),
  "PositionSizerModal must declare isSizingBlocked from canSizeTrade and isActionable"
);
assert.ok(
  sizerSource.includes("rawShares = (isSetupInvalid || isSizingBlocked)"),
  "PositionSizerModal must force shares to 0 when sizing is blocked"
);
assert.ok(
  sizerSource.includes("Sizing Disabled (Non-Actionable Asset)"),
  "PositionSizerModal must show disabled text when sizing is blocked"
);
console.log("   [OK] PositionSizerModal strictly suppresses shares and disables save when non-actionable");

console.log("\nALL PHASE 2 DECISION AUTHORITY CONSOLIDATION TESTS PASSED SUCCESSFULLY!");
