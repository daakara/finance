import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import {
  DecisionState,
  isDecisionActionable,
  isStatusActionable,
} from "../types/decisionContract";

console.log("Starting Pre-Flight & Radar Copy Remediation Forensic Verification Suite...\n");

// ---------------------------------------------------------------------------
// 1. Verify PreFlightChecklistModal Copy Quality Acceptance
// ---------------------------------------------------------------------------
console.log("1. Verifying PreFlightChecklistModal copy quality and architecture leakage...");

const preflightPath = path.join(__dirname, "../components/PreFlightChecklistModal.tsx");
const preflightSource = fs.readFileSync(preflightPath, "utf8");

// A. Elimination of emotional/paternalistic copy
const hasHardEarnedMoney = preflightSource.toLowerCase().includes("hard-earned money");
assert.strictEqual(
  hasHardEarnedMoney,
  false,
  "HARD_EARNED_MONEY_COPY must be completely removed from PreFlightChecklistModal"
);
assert.ok(
  preflightSource.includes("5-point risk and confirmation checklist before entering this position."),
  "Pre-Flight plain subtitle must be: '5-point risk and confirmation checklist before entering this position.'"
);
console.log("   [OK] Emotional 'hard-earned money' copy removed; professional subtitle verified");

// B. Elimination of false revocation language
const hasRevoked = preflightSource.includes("FLIGHT CLEARANCE REVOKED");
assert.strictEqual(
  hasRevoked,
  false,
  "PRE_FLIGHT_FALSE_REVOCATION_LANGUAGE must be 0 (no 'FLIGHT CLEARANCE REVOKED')"
);
console.log("   [OK] False revocation language eliminated (0 instances of FLIGHT CLEARANCE REVOKED)");

// C. Elimination of internal engine leakage in primary UI
const hasEngineLeakageInPrimary =
  preflightSource.includes("DecisionHierarchyEngine reports") ||
  preflightSource.includes("CANONICAL AUTHORITY") ||
  preflightSource.includes("Local checklist cannot override");
assert.strictEqual(
  hasEngineLeakageInPrimary,
  false,
  "PRE_FLIGHT_INTERNAL_ENGINE_LEAKAGE must be 0 in primary UI copy"
);
console.log("   [OK] Internal engine terminology leakage eliminated from primary UI");

// D. Plain English Banner Verification
assert.ok(
  preflightSource.includes("TRADE NOT CLEARED: AWAITING CONFIRMATION"),
  "Plain English banner title must be 'TRADE NOT CLEARED: AWAITING CONFIRMATION'"
);
assert.ok(
  preflightSource.includes("A valid setup is forming for ${symbol}, but the entry trigger has not confirmed yet."),
  "Plain English banner must explain setup forming and trigger awaiting confirmation"
);
assert.ok(
  preflightSource.includes("This checklist reviews setup and risk conditions; it does not bypass the required confirmation trigger."),
  "Plain English banner must state checklist does not bypass required confirmation trigger"
);
console.log("   [OK] Plain English non-actionable banner verified");

// E. Pro Quant Banner Verification
assert.ok(
  preflightSource.includes("NON-ACTIONABLE STATE: AWAITING TRIGGER CONFIRMATION"),
  "Pro Quant banner title must be 'NON-ACTIONABLE STATE: AWAITING TRIGGER CONFIRMATION'"
);
assert.ok(
  preflightSource.includes("Valid setup structure is present (${decisionState || \"VALID_SETUP\"}), but execution criteria remain incomplete."),
  "Pro Quant banner must explain valid setup structure with incomplete execution criteria"
);
assert.ok(
  preflightSource.includes("Current state: WAIT FOR TRIGGER. Sizing and execution remain locked until confirmation conditions are satisfied."),
  "Pro Quant banner must explain execution and sizing remain locked until trigger conditions are met"
);
console.log("   [OK] Pro Quant non-actionable banner verified");

// F. Optional Technical Provenance in Collapsed Details
assert.ok(
  preflightSource.includes("<details") &&
  preflightSource.includes("Technical Provenance") &&
  preflightSource.includes("Authority: Canonical Decision Engine"),
  "Technical provenance must be available in secondary collapsed details UI"
);
console.log("   [OK] Technical provenance encapsulated in secondary collapsed details");

// ---------------------------------------------------------------------------
// 2. Verify Radar Page Copy & Split Capability Semantics
// ---------------------------------------------------------------------------
console.log("\n2. Verifying Radar page capability copy and split capability semantics...");

const radarPath = path.join(__dirname, "../app/radar/page.tsx");
const radarSource = fs.readFileSync(radarPath, "utf8");

// A. Badge remediation: "Universe Scanner Pending" replaces ambiguous "Pipeline Pending"
assert.ok(
  radarSource.includes("badge: isAvailable('SMART_MONEY') ? `${allAssets.filter((a) => a.categories.includes('SMART_MONEY')).length}` : 'Universe Scanner Pending'"),
  "Smart Money badge must be 'Universe Scanner Pending' when not available"
);
assert.ok(
  radarSource.includes("badge: isAvailable('VCP') ? `${allAssets.filter((a) => a.categories.includes('VCP')).length}` : 'Universe Scanner Pending'"),
  "VCP badge must be 'Universe Scanner Pending' when not available"
);
console.log("   [OK] Smart Money & VCP badges updated to 'Universe Scanner Pending'");

// B. Empty State Title Remediation
assert.ok(
  radarSource.includes('activeFilter === \'SMART_MONEY\' && categoryMeta.SMART_MONEY.status === \'PIPELINE_PENDING\'\n                              ? "Smart Money — Universe Scanner Pending"'),
  "Smart Money empty state title must be 'Smart Money — Universe Scanner Pending'"
);
assert.ok(
  radarSource.includes('activeFilter === \'VCP\' && categoryMeta.VCP.status === \'PIPELINE_PENDING\'\n                              ? "Minervini VCP — Universe Scanner Pending"'),
  "VCP empty state title must be 'Minervini VCP — Universe Scanner Pending'"
);
console.log("   [OK] Smart Money & VCP empty state titles verified");

// C. Explanatory Body: Single-Asset Availability Communicated
assert.ok(
  radarSource.includes("Automated market-wide institutional-flow screening is not active yet. Single-asset insider and congressional filing analysis is still available for individual symbols on /smart-money."),
  "Smart Money empty state body must explain single-asset availability on /smart-money"
);
assert.ok(
  radarSource.includes("Automated market-wide volatility contraction screening is not active yet. Single-asset VCP geometry is still available in the Analysis (/) and Setups (/setups) hubs."),
  "VCP empty state body must explain single-asset availability in Analysis and Setups"
);
console.log("   [OK] Split capability model verified: single-asset availability clearly communicated");

// D. Tab Button Tooltips
assert.ok(
  radarSource.includes("title={categoryMeta.SMART_MONEY.status === 'PIPELINE_PENDING' ? \"Automated market-wide institutional-flow screening is not active yet. Single-asset insider and congressional filing analysis is still available for individual symbols on /smart-money.\" : undefined}"),
  "Smart Money tab button must have informative tooltip"
);
assert.ok(
  radarSource.includes("title={categoryMeta.VCP.status === 'PIPELINE_PENDING' ? \"Automated market-wide volatility contraction screening is not active yet. Single-asset VCP geometry is still available in the Analysis (/) and Setups (/setups) hubs.\" : undefined}"),
  "VCP tab button must have informative tooltip"
);
console.log("   [OK] Smart Money & VCP tab tooltips verified");

// ---------------------------------------------------------------------------
// 3. Invariant Preservation: Decision Contract & Actionability Logic
// ---------------------------------------------------------------------------
console.log("\n3. Verifying authoritative decision contract invariants...");

// Invariant 1: VALID_SETUP + IN_BUY_ZONE is strictly NOT actionable
assert.strictEqual(
  isDecisionActionable(DecisionState.VALID_SETUP, "IN_BUY_ZONE"),
  false,
  "INVARIANT BREACH: VALID_SETUP must NEVER be actionable in buy zone"
);

// Invariant 2: ACTIONABLE_SETUP + IN_BUY_ZONE is actionable
assert.strictEqual(
  isDecisionActionable(DecisionState.ACTIONABLE_SETUP, "IN_BUY_ZONE"),
  true,
  "INVARIANT BREACH: ACTIONABLE_SETUP + IN_BUY_ZONE must be actionable"
);

// Invariant 3: ACTIONABLE_SETUP with non-actionable status is NOT actionable
assert.strictEqual(
  isDecisionActionable(DecisionState.ACTIONABLE_SETUP, "WAITING_PULLBACK"),
  false,
  "INVARIANT BREACH: ACTIONABLE_SETUP + WAITING_PULLBACK must not be actionable"
);

// Invariant 4: Status actionability
assert.strictEqual(isStatusActionable("IN_BUY_ZONE"), true);
assert.strictEqual(isStatusActionable("READY_TO_BUY"), true);
assert.strictEqual(isStatusActionable("WAITING_PULLBACK"), false);

console.log("   [OK] All decision contract invariants preserved with zero modification");

console.log("\n===============================================================================");
console.log("ALL PRE-FLIGHT & RADAR COPY REMEDIATION TESTS PASSED SUCCESSFULLY!");
console.log("===============================================================================");
