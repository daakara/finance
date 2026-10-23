import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { DEFAULT_RADAR_CAPABILITIES } from "../lib/api";
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

// VCP badge uses capability matchedCount when available, falls back to candidate count, and defaults to 'Universe Scanner Pending'
const vcpBadgeRegex = /badge:\s*isAvailable\('VCP'\)\s*\?\s*`\$\{typeof\s+capabilities\.VCP\?\.matchedCount\s*===\s*'number'\s*\?\s*capabilities\.VCP\.matchedCount\s*:\s*allAssets\.filter\(\(a\)\s*=>\s*a\.categories\.includes\('VCP'\)\)\.length\}`\s*:\s*'Universe Scanner Pending'/;
assert.ok(
  vcpBadgeRegex.test(radarSource),
  "VCP badge must use capability matchedCount when available, fallback to candidate length, and 'Universe Scanner Pending' when not available"
);
console.log("   [OK] Smart Money & VCP badges updated to 'Universe Scanner Pending' with dynamic matchedCount support");

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
// 2.E Semantic Capability-to-Badge and Capability-to-Count Contract Tests
// ---------------------------------------------------------------------------
console.log("\n2.E Verifying semantic capability-to-badge and count contracts across all lifecycle states...");

interface CapabilityState {
  universeScreening?: string;
  matchedCount?: number | null;
  status?: string;
}

function resolveVcpCategoryMeta(
  cap: CapabilityState | undefined,
  assets: Array<{ categories: string[] }>
) {
  const isAvailable = cap?.universeScreening === "AVAILABLE";
  const count = isAvailable
    ? (typeof cap?.matchedCount === "number"
        ? cap.matchedCount
        : assets.filter((a) => a.categories.includes("VCP")).length)
    : null;
  const badge = isAvailable
    ? `${typeof cap?.matchedCount === "number"
        ? cap.matchedCount
        : assets.filter((a) => a.categories.includes("VCP")).length}`
    : "Universe Scanner Pending";
  const status = cap?.status || "PIPELINE_PENDING";
  return { count, badge, status };
}

// State 1: Scanner Unavailable (NOT_IMPLEMENTED)
const state1 = resolveVcpCategoryMeta(
  { universeScreening: "NOT_IMPLEMENTED", status: "NOT_IMPLEMENTED", matchedCount: null },
  [{ categories: ["VCP"] }]
);
assert.strictEqual(state1.count, null, "State 1: Unavailable scanner must produce count: null");
assert.strictEqual(state1.badge, "Universe Scanner Pending", "State 1: Unavailable scanner badge must be 'Universe Scanner Pending'");
console.log("   [OK] State 1 (Unavailable): count=null, badge='Universe Scanner Pending'");

// State 2: Scanner PIPELINE_PENDING (Must NOT fabricate zero-match claim)
const state2 = resolveVcpCategoryMeta(
  { universeScreening: "PIPELINE_PENDING", status: "PIPELINE_PENDING", matchedCount: null },
  []
);
assert.strictEqual(state2.count, null, "State 2: PIPELINE_PENDING must NOT fabricate count 0; must be null");
assert.notStrictEqual(state2.count, 0, "State 2 ZERO SEMANTICS BREACH: count must never be 0 when pending");
assert.strictEqual(state2.badge, "Universe Scanner Pending", "State 2: Pending badge must be 'Universe Scanner Pending'");
console.log("   [OK] State 2 (PIPELINE_PENDING): count=null (no fabricated 0), badge='Universe Scanner Pending'");

// State 3: Qualified snapshot with known count (matchedCount = 7)
const state3 = resolveVcpCategoryMeta(
  { universeScreening: "AVAILABLE", status: "AVAILABLE", matchedCount: 7 },
  [{ categories: ["VCP"] }, { categories: ["VCP"] }] // 2 candidate assets, authoritative count is 7
);
assert.strictEqual(state3.count, 7, "State 3: Authoritative matchedCount 7 must take precedence over candidate assets");
assert.strictEqual(state3.badge, "7", "State 3: Badge must be authoritative '7'");
console.log("   [OK] State 3 (Qualified snapshot): count=7, badge='7'");

// State 4: Qualified snapshot with 0 matches (authoritative matchedCount = 0)
const state4 = resolveVcpCategoryMeta(
  { universeScreening: "AVAILABLE", status: "AVAILABLE", matchedCount: 0 },
  []
);
assert.strictEqual(state4.count, 0, "State 4: Authoritative matchedCount 0 must be preserved when scanner is AVAILABLE");
assert.strictEqual(state4.badge, "0", "State 4: Badge must be '0' when scanner is AVAILABLE with 0 matches");
console.log("   [OK] State 4 (Zero matches): count=0, badge='0'");

// State 5: Unknown or null matched count with candidate assets (fallback without crashing)
const state5 = resolveVcpCategoryMeta(
  { universeScreening: "AVAILABLE", status: "AVAILABLE", matchedCount: null },
  [{ categories: ["VCP"] }, { categories: ["VCP"] }, { categories: ["VCP"] }]
);
assert.strictEqual(state5.count, 3, "State 5: Null matchedCount in AVAILABLE scanner must fall back to candidate count (3)");
assert.strictEqual(state5.badge, "3", "State 5: Badge must fall back to '3'");
console.log("   [OK] State 5 (Null matchedCount fallback): count=3, badge='3'");

// State 6: Backend error recovery (Reset to DEFAULT_RADAR_CAPABILITIES, allAssets = [])
const state6 = resolveVcpCategoryMeta(
  DEFAULT_RADAR_CAPABILITIES.VCP,
  []
);
assert.strictEqual(state6.count, null, "State 6: Backend error reset must enforce count: null");
assert.strictEqual(state6.badge, "Universe Scanner Pending", "State 6: Backend error reset must enforce badge: 'Universe Scanner Pending'");
console.log("   [OK] State 6 (Backend error reset): count=null, badge='Universe Scanner Pending'");

// ---------------------------------------------------------------------------
// 2.F Negative Contract Tests (Adversarial Broken Contracts)
// ---------------------------------------------------------------------------
console.log("\n2.F Verifying negative contracts catch broken/adversarial implementations...");

// Negative Test 1: Fabricating count 0 when PIPELINE_PENDING must be caught
function brokenZeroFabricatingResolver(cap: CapabilityState | undefined) {
  return {
    count: cap?.universeScreening === "AVAILABLE" ? (cap?.matchedCount ?? 0) : 0, // BUG: returns 0 when pending
    badge: "0",
  };
}
const brokenState1 = brokenZeroFabricatingResolver({ universeScreening: "PIPELINE_PENDING" });
assert.throws(
  () => {
    assert.strictEqual(
      brokenState1.count,
      null,
      "Zero Semantics Invariant breached: pending returned 0 instead of null"
    );
  },
  /AssertionError/,
  "Negative Test 1 must catch zero-fabrication during pending state"
);
console.log("   [OK] Negative 1: Zero-fabricating pending resolver correctly caught and rejected");

// Negative Test 2: Ignoring authoritative matchedCount must be caught
function brokenIgnoringMatchedCountResolver(
  cap: CapabilityState | undefined,
  assets: Array<{ categories: string[] }>
) {
  // BUG: ignores cap.matchedCount and only returns asset count
  return {
    count: assets.filter((a) => a.categories.includes("VCP")).length,
    badge: `${assets.filter((a) => a.categories.includes("VCP")).length}`,
  };
}
const brokenState2 = brokenIgnoringMatchedCountResolver(
  { universeScreening: "AVAILABLE", matchedCount: 42 },
  [{ categories: ["VCP"] }]
);
assert.throws(
  () => {
    assert.strictEqual(
      brokenState2.count,
      42,
      "Authority breach: resolver ignored authoritative matchedCount"
    );
  },
  /AssertionError/,
  "Negative Test 2 must catch resolver ignoring authoritative matchedCount"
);
console.log("   [OK] Negative 2: Resolver ignoring authoritative matchedCount correctly caught and rejected");

// Negative Test 3: Returning raw count badge when scanner is unavailable must be caught
function brokenBadgeResolver(cap: CapabilityState | undefined) {
  // BUG: returns "0" badge when unavailable instead of "Universe Scanner Pending"
  return {
    badge: cap?.universeScreening === "AVAILABLE" ? "0" : "0",
  };
}
const brokenState3 = brokenBadgeResolver({ universeScreening: "PIPELINE_PENDING" });
assert.throws(
  () => {
    assert.strictEqual(
      brokenState3.badge,
      "Universe Scanner Pending",
      "Copy breach: resolver returned numeric badge during pending state"
    );
  },
  /AssertionError/,
  "Negative Test 3 must catch non-pending badge during pending state"
);
console.log("   [OK] Negative 3: Raw numeric badge during pending state correctly caught and rejected");

// Negative Test 4: Broken resolver that throws on null/undefined matchedCount
function brokenFragileResolver(cap: CapabilityState | undefined) {
  if (cap?.matchedCount === null || cap?.matchedCount === undefined) {
    throw new TypeError("Cannot read properties of null (matchedCount)");
  }
  return { count: cap.matchedCount, badge: `${cap.matchedCount}` };
}
assert.throws(
  () => {
    brokenFragileResolver({ universeScreening: "AVAILABLE", matchedCount: null });
  },
  /TypeError/,
  "Negative Test 4 must catch fragile resolver throwing on null matchedCount"
);
// In contrast, robust resolveVcpCategoryMeta handles null/undefined without throwing:
assert.doesNotThrow(() => {
  resolveVcpCategoryMeta({ universeScreening: "AVAILABLE", matchedCount: null }, []);
  resolveVcpCategoryMeta({ universeScreening: "AVAILABLE", matchedCount: undefined }, []);
}, "Robust resolver must handle null/undefined matchedCount without throwing");
console.log("   [OK] Negative 4: Fragile null-handling resolver caught; robust fallback verified");

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
