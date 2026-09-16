import assert from "node:assert";
import {
  ACTIONABLE_EXECUTION_STATUSES,
  NON_ACTIONABLE_EXECUTION_STATUSES,
  isStatusActionable,
  isDecisionActionable,
  DecisionState,
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

console.log("[OK] Decision state + execution status joint actionability verified");
console.log("ALL DECISION CONTRACT TESTS PASSED!");
