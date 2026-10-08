import assert from "node:assert";
import {
  resolvePresentationState,
  PresentationResolverParams,
} from "../lib/decisionPresentation";

/**
 * ARX Terminal Synthesis E Wave 4: Universal Shared Presentation Authority Suite
 *
 * Verifies:
 * 1. Anti-Collapse Invariant: isActionable === false strictly yields actionabilityLabel: "NOT ACTIONABLE",
 *    NEVER "WAIT FOR TRIGGER".
 * 2. Cross-State Matrix: AVOID, HOLD, UNVERIFIED, INSUFFICIENT_HISTORY, EVIDENCE_INCOMPLETE, WAITING_PULLBACK
 *    never display "WAIT FOR TRIGGER" badge.
 * 3. Zero Headline/Badge Duplication: Headline and Badge are decoupled.
 * 4. Fail-Closed Fallback: Null/empty decision state defaults to "Setup Evaluation Pending" (Neutral).
 * 5. Full coverage of the 8 canonical presentation states.
 */

console.log("Starting Synthesis E Wave 4 Decision Presentation Test Suite...\n");

// ── 1. Anti-Collapse & Binary Actionability Domain ───────────────────────────
console.log("1. Verifying Strict Anti-Collapse Rules (QA-ESC-011 Invariant)...");

const nonActionableCases: PresentationResolverParams[] = [
  { decisionState: "AVOID", isActionable: false },
  { decisionState: "HOLD", isActionable: false },
  { decisionState: "UNVERIFIED", isActionable: false },
  { decisionState: "INSUFFICIENT_DATA", isActionable: false },
  { decisionState: "EVIDENCE_INCOMPLETE", isActionable: false },
  { executionStatus: "WAITING_PULLBACK", isActionable: false },
  { executionStatus: "STAGE_4_CORRECTION", isActionable: false },
];

for (const tc of nonActionableCases) {
  const result = resolvePresentationState(tc);
  assert.strictEqual(
    result.actionabilityLabel,
    "NOT ACTIONABLE",
    `Violation: ${tc.decisionState || tc.executionStatus} produced actionability '${result.actionabilityLabel}' instead of 'NOT ACTIONABLE'`
  );
  assert.notStrictEqual(
    result.badgeLabel,
    "WAIT FOR TRIGGER",
    `Violation: Non-actionable state collapsed into generic 'WAIT FOR TRIGGER'!`
  );
  assert.notStrictEqual(
    result.actionabilityLabel,
    "WAIT FOR TRIGGER",
    `Violation: Actionability domain collapsed into 'WAIT FOR TRIGGER'!`
  );
}
console.log("   [PASS] 1. All non-actionable states strictly map to 'NOT ACTIONABLE' and never collapse into 'WAIT FOR TRIGGER'");

// ── 2. Decoupled Headline and Badge (Zero Duplication) ───────────────────────
console.log("\n2. Verifying Zero Headline/Badge Duplication...");

const actionableState = resolvePresentationState({
  decisionState: "ACTIONABLE_SETUP",
  executionStatus: "IN_BUY_ZONE",
  isActionable: true,
});
assert.strictEqual(actionableState.badgeLabel, "ACTIONABLE SETUP");
assert.strictEqual(actionableState.actionabilityLabel, "ACTIONABLE");
assert.notStrictEqual(actionableState.headlineLabel, actionableState.badgeLabel);
console.log("   [PASS] 2. Headline and Badge labels are cleanly decoupled (no verbatim duplication)");

// ── 3. Canonical State Matrix Verification ───────────────────────────────────
console.log("\n3. Verifying Canonical State Matrix (PRD Section 9.1)...");

// 3.1 ACTIONABLE_SETUP
assert.strictEqual(
  resolvePresentationState({ decisionState: "ACTIONABLE_SETUP", isActionable: true }).badgeLabel,
  "ACTIONABLE SETUP"
);

// 3.2 WAITING_PULLBACK
assert.strictEqual(
  resolvePresentationState({ executionStatus: "WAITING_PULLBACK", isActionable: false }).badgeLabel,
  "AWAITING PULLBACK"
);

// 3.3 IN_BUY_ZONE_AWAITING_TRIGGER
assert.strictEqual(
  resolvePresentationState({ executionStatus: "IN_BUY_ZONE_AWAITING_TRIGGER", isActionable: false }).badgeLabel,
  "AWAITING TRIGGER"
);

// 3.4 APPROACHING_TARGET
assert.strictEqual(
  resolvePresentationState({ executionStatus: "APPROACHING_TARGET", isActionable: false }).badgeLabel,
  "APPROACHING TARGET"
);

// 3.5 STAGE_4_CORRECTION
const stage4Readiness = {
  activeBlockingGate: "GATE_1_LOCATION",
  gates: [{ explanation: "Asset is in Stage 4 distribution" }],
} as any;
assert.strictEqual(
  resolvePresentationState({ executionStatus: "STAGE_4_CORRECTION", isActionable: false, readinessResult: stage4Readiness }).badgeLabel,
  "STAGE 4 DEFENSE"
);

// 3.6 EVIDENCE_INCOMPLETE
assert.strictEqual(
  resolvePresentationState({ decisionState: "EVIDENCE_INCOMPLETE", isActionable: false }).badgeLabel,
  "EVIDENCE INCOMPLETE"
);

// 3.7 INSUFFICIENT_HISTORY
assert.strictEqual(
  resolvePresentationState({ executionStatus: "INSUFFICIENT_HISTORY", isActionable: false }).badgeLabel,
  "INSUFFICIENT HISTORY"
);

// 3.8 STALE_MARKET_DATA
assert.strictEqual(
  resolvePresentationState({ executionStatus: "STALE_MARKET_DATA", isActionable: false }).badgeLabel,
  "STALE TAPE"
);

// 3.9 UNVERIFIED_ASSET
assert.strictEqual(
  resolvePresentationState({ executionStatus: "UNVERIFIED_ASSET", isActionable: false }).badgeLabel,
  "UNVERIFIED ASSET"
);

console.log("   [PASS] 3. All 9 canonical states resolve to ratified, distinct badges");

// ── 4. Fail-Closed Fallback ──────────────────────────────────────────────────
console.log("\n4. Verifying Fail-Closed Fallback...");

const emptyState = resolvePresentationState({});
assert.strictEqual(emptyState.headlineLabel, "Setup Evaluation Pending");
assert.strictEqual(emptyState.badgeLabel, "SETUP EVALUATION PENDING");
assert.strictEqual(emptyState.actionabilityLabel, "NOT ACTIONABLE");
assert.strictEqual(emptyState.badgeStyle.text, "text-slate-400");
console.log("   [PASS] 4. Missing inputs fail closed to SETUP EVALUATION PENDING (Neutral slate), NEVER WAIT FOR TRIGGER");

console.log("\nALL DECISION PRESENTATION TESTS PASSED!\n");
