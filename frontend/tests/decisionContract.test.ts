import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import {
  ACTIONABLE_EXECUTION_STATUSES,
  NON_ACTIONABLE_EXECUTION_STATUSES,
  isStatusActionable,
  isDecisionActionable,
  DecisionState,
  EvidenceQualityState,
  canQualityCreateActionability,
  canQualityContributeEvidence,
  ARXDecision,
  FundamentalEvidenceContract,
  MacroEvidenceContract,
} from "../types/decisionContract";

console.log("Starting Decision Contract & Taxonomy Regression Suite...");

// 1. Verify Actionable Statuses
assert.strictEqual(isStatusActionable("IN_BUY_ZONE"), true);
assert.strictEqual(isStatusActionable("READY_TO_BUY"), true);
console.log("[OK] Actionable execution statuses verified");

// 2. Verify Non-Actionable Statuses
const nonActionable = [
  "WAITING_PULLBACK",
  "IN_BUY_ZONE_AWAITING_TRIGGER",
  "APPROACHING_TARGET",
  "STOPPED_OUT",
  "INSUFFICIENT_HISTORY",
  "UNVERIFIED_ASSET",
  "STALE_MARKET_DATA",
  "UNKNOWN",
  "",
  null,
  undefined,
];

for (const status of nonActionable) {
  assert.strictEqual(
    isStatusActionable(status),
    false,
    `Status ${status} must NOT be actionable`
  );
}
console.log("[OK] Non-actionable execution statuses verified");

// 3. Verify Decision Hierarchy + Execution Status Joint Actionability
assert.strictEqual(
  isDecisionActionable(DecisionState.ACTIONABLE_SETUP, "IN_BUY_ZONE"),
  true,
  "ACTIONABLE_SETUP + IN_BUY_ZONE must be actionable"
);

assert.strictEqual(
  isDecisionActionable(DecisionState.EVIDENCE_INCOMPLETE, "IN_BUY_ZONE"),
  false,
  "EVIDENCE_INCOMPLETE + IN_BUY_ZONE must NOT be actionable"
);

assert.strictEqual(
  isDecisionActionable(DecisionState.VALID_SETUP, "IN_BUY_ZONE"),
  false,
  "VALID_SETUP + IN_BUY_ZONE must NOT be actionable"
);

assert.strictEqual(
  isDecisionActionable(DecisionState.ACTIONABLE_SETUP, "WAITING_PULLBACK"),
  false,
  "ACTIONABLE_SETUP + WAITING_PULLBACK must NOT be actionable"
);

// 4. Verify Strict Fail-Closed (Missing, null, or undefined decisionState MUST return false)
assert.strictEqual(
  isDecisionActionable(undefined, "IN_BUY_ZONE"),
  false,
  "Undefined decisionState must fail closed (false) even in buy zone"
);

assert.strictEqual(
  isDecisionActionable(null, "IN_BUY_ZONE"),
  false,
  "Null decisionState must fail closed (false) even in buy zone"
);

assert.strictEqual(
  isDecisionActionable("", "IN_BUY_ZONE"),
  false,
  "Empty decisionState must fail closed (false) even in buy zone"
);

assert.strictEqual(
  isDecisionActionable("UNKNOWN_STATE", "IN_BUY_ZONE"),
  false,
  "Unknown decisionState must fail closed (false)"
);

console.log("[OK] Decision state + execution status joint actionability verified (fail-closed)");

// 5. Evidence Quality Governance Rules
assert.strictEqual(canQualityCreateActionability("AUTHORITATIVE"), true);
assert.strictEqual(canQualityCreateActionability("PROVISIONAL"), false);
assert.strictEqual(canQualityCreateActionability("FALLBACK"), false);
assert.strictEqual(canQualityCreateActionability("UNAVAILABLE"), false);
assert.strictEqual(canQualityCreateActionability("STALE"), false);

assert.strictEqual(canQualityContributeEvidence("AUTHORITATIVE"), true);
assert.strictEqual(canQualityContributeEvidence("PROVISIONAL"), true);
assert.strictEqual(canQualityContributeEvidence("FALLBACK"), true);
assert.strictEqual(canQualityContributeEvidence("UNAVAILABLE"), false);
assert.strictEqual(canQualityContributeEvidence("STALE"), false);
console.log("[OK] Evidence quality governance rules verified");

// 6. Cross-Language Schema Parity with Canonical Fixture
const fixturePath = path.resolve(__dirname, "../../tests/fixtures/canonical_decision_fixture.json");
assert(fs.existsSync(fixturePath), `Fixture file must exist at ${fixturePath}`);
const rawFixture = JSON.parse(fs.readFileSync(fixturePath, "utf-8"));

const fullDecision: ARXDecision = rawFixture.full_decision;
assert.strictEqual(fullDecision.authority, "BACKEND_CANONICAL");
assert.strictEqual(fullDecision.verdict.isActionable, true);
assert.strictEqual(fullDecision.verdict.decisionState, "ACTIONABLE_SETUP");
assert.strictEqual(fullDecision.context.marketEvidence.quality, "AUTHORITATIVE");

const fundPayload: FundamentalEvidenceContract = fullDecision.context.fundamentalEvidence.payload;
assert.strictEqual(fundPayload.pointInTimeStatus, "POINT_IN_TIME");
assert.strictEqual(fundPayload.source, "sec_edgar");
assert.strictEqual(fundPayload.quality, "AUTHORITATIVE");

const macroPayload: MacroEvidenceContract = fullDecision.context.macroEvidence.payload;
assert.strictEqual(macroPayload.tacticalEquityRegime, "BULL_TRENDING");
assert.strictEqual(macroPayload.structuralMacroRegime, "EXPANSION");
assert.strictEqual(macroPayload.macroRiskFriction, "STABLE");

const degradedDecision: ARXDecision = rawFixture.degraded_decision;
assert.strictEqual(degradedDecision.authority, "DISPLAY_ONLY_MARKET_DATA");
assert.strictEqual(degradedDecision.verdict.isActionable, false);
assert.strictEqual(degradedDecision.verdict.decisionState, "UNVERIFIED");
assert.strictEqual(degradedDecision.verdict.levels.entryMin, null);
assert.strictEqual(degradedDecision.verdict.levels.stopLoss, null);
assert.strictEqual(degradedDecision.context.isDegraded, true);
assert.strictEqual(degradedDecision.context.evidenceCompleteness, "DEGRADED");
console.log("[OK] Cross-language schema parity verified against canonical fixture");

console.log("ALL DECISION CONTRACT TESTS PASSED!");
