import assert from "node:assert";
import {
  evaluateCorridorLocation,
  resolveDecisionReadiness,
  resolveTacticalExecutionHint,
} from "../lib/decisionReadiness";
import { resolvePresentationState } from "../lib/decisionPresentation";
import { deriveUnmetConditions } from "../lib/decisionHierarchyUtils";
import { QuantitativeInsight } from "../types/insight";

/**
 * ARX Terminal Workstream D — Slice 1 Regression Suite
 * Decision Text Truthfulness & Blocked-Action Consistency Acceptance Matrix
 *
 * Verifies:
 * Part 1: Corridor Location & Geometry Invariants (9 cases)
 * Part 2: Decision Guidance & Blocked-Action Consistency (6 cases)
 * Part 3: LNTH Live Fixture Inversion Reproduction & Verification
 */

console.log("Starting Workstream D Slice 1 Acceptance Suite...\n");

// ═════════════════════════════════════════════════════════════════════════════
// PART 1: Corridor Location & Geometry Invariants (9 Cases)
// ═════════════════════════════════════════════════════════════════════════════
console.log("Part 1: Verifying Corridor Location & Geometry Invariants (9 Cases)...");

// Case 1.1: Below lower boundary ($95 vs $98–$102)
const c1 = evaluateCorridorLocation(95, 98, 102, 0.02);
assert.strictEqual(c1.isValid, true);
assert.strictEqual(c1.spatialRelation, "BELOW");
assert.strictEqual(c1.isBelowFloor, true);
assert.strictEqual(c1.isInside, false);
assert.strictEqual(c1.isAboveCeiling, false);
assert.strictEqual(c1.isWithinTolerance, false);
assert.ok(c1.description.includes("below the accumulation corridor"));
assert.ok(!c1.description.includes("positioned inside"));

const g1_1 = resolveDecisionReadiness({ currentPrice: 95, optimalEntryMin: 98, optimalEntryMax: 102 });
assert.strictEqual(g1_1.gates[0].state, "BLOCKING");
assert.ok(g1_1.gates[0].explanation.includes("below the accumulation corridor"));
console.log("   [PASS] 1.1: Price below lower boundary ($95 vs $98–$102) -> BELOW / BLOCKING");

// Case 1.2: Exact lower boundary ($98 vs $98–$102) — Inclusive boundary convention
const c2 = evaluateCorridorLocation(98, 98, 102, 0.02);
assert.strictEqual(c2.isValid, true);
assert.strictEqual(c2.spatialRelation, "INSIDE");
assert.strictEqual(c2.isInside, true);
assert.strictEqual(c2.isBelowFloor, false);
assert.strictEqual(c2.isAboveCeiling, false);
assert.strictEqual(c2.isWithinTolerance, false);
assert.ok(c2.description.includes("positioned inside the institutional accumulation corridor"));

const g1_2 = resolveDecisionReadiness({ currentPrice: 98, optimalEntryMin: 98, optimalEntryMax: 102 });
assert.strictEqual(g1_2.gates[0].state, "PASSED");
console.log("   [PASS] 1.2: Exact lower boundary ($98 vs $98–$102) -> INSIDE / PASSED (Inclusive boundary)");

// Case 1.3: Inside corridor ($100 vs $98–$102)
const c3 = evaluateCorridorLocation(100, 98, 102, 0.02);
assert.strictEqual(c3.isValid, true);
assert.strictEqual(c3.spatialRelation, "INSIDE");
assert.strictEqual(c3.isInside, true);
assert.strictEqual(c3.isBelowFloor, false);
assert.strictEqual(c3.isAboveCeiling, false);
assert.strictEqual(c3.isWithinTolerance, false);
assert.ok(c3.description.includes("positioned inside the institutional accumulation corridor"));

const g1_3 = resolveDecisionReadiness({ currentPrice: 100, optimalEntryMin: 98, optimalEntryMax: 102 });
assert.strictEqual(g1_3.gates[0].state, "PASSED");
console.log("   [PASS] 1.3: Inside corridor ($100 vs $98–$102) -> INSIDE / PASSED");

// Case 1.4: Exact upper boundary ($102 vs $98–$102) — Inclusive boundary convention
const c4 = evaluateCorridorLocation(102, 98, 102, 0.02);
assert.strictEqual(c4.isValid, true);
assert.strictEqual(c4.spatialRelation, "INSIDE");
assert.strictEqual(c4.isInside, true);
assert.strictEqual(c4.isBelowFloor, false);
assert.strictEqual(c4.isAboveCeiling, false);
assert.strictEqual(c4.isWithinTolerance, false);
assert.ok(c4.description.includes("positioned inside the institutional accumulation corridor"));

const g1_4 = resolveDecisionReadiness({ currentPrice: 102, optimalEntryMin: 98, optimalEntryMax: 102 });
assert.strictEqual(g1_4.gates[0].state, "PASSED");
console.log("   [PASS] 1.4: Exact upper boundary ($102 vs $98–$102) -> INSIDE / PASSED (Inclusive boundary)");

// Case 1.5: Just above upper boundary ($102.01 vs $98–$102) — Must NEVER be described as inside
const c5 = evaluateCorridorLocation(102.01, 98, 102, 0.02);
assert.strictEqual(c5.isValid, true);
assert.strictEqual(c5.spatialRelation, "ABOVE");
assert.strictEqual(c5.isInside, false, "Price > upper bound must NEVER be isInside: true");
assert.strictEqual(c5.isAboveCeiling, true);
assert.strictEqual(c5.isBelowFloor, false);
assert.strictEqual(c5.isWithinTolerance, true);
assert.ok(c5.description.includes("above the entry corridor"));
assert.ok(!c5.description.includes("positioned inside"));

const g1_5 = resolveDecisionReadiness({ currentPrice: 102.01, optimalEntryMin: 98, optimalEntryMax: 102 });
assert.strictEqual(g1_5.gates[0].state, "PASSED", "Preserves existing 2% tolerance decision pass");
assert.ok(g1_5.gates[0].explanation.includes("above the entry corridor"));
assert.ok(!g1_5.gates[0].explanation.includes("positioned inside"));
console.log("   [PASS] 1.5: Just above upper boundary ($102.01 vs $98–$102) -> ABOVE / isInside=false / PASSED (Tolerance)");

// Case 1.6: Above upper boundary within 2% tolerance ($103.50 vs $98–$102)
// Upper bound tolerance limit: 102 * 1.02 = 104.04
const c6 = evaluateCorridorLocation(103.50, 98, 102, 0.02);
assert.strictEqual(c6.isValid, true);
assert.strictEqual(c6.spatialRelation, "ABOVE");
assert.strictEqual(c6.isInside, false);
assert.strictEqual(c6.isAboveCeiling, true);
assert.strictEqual(c6.isWithinTolerance, true);
assert.ok(c6.description.includes("above the entry corridor"));
assert.ok(c6.description.includes("Existing extension tolerance applies"));
assert.ok(!c6.description.includes("positioned inside"));

const g1_6 = resolveDecisionReadiness({ currentPrice: 103.50, optimalEntryMin: 98, optimalEntryMax: 102 });
assert.strictEqual(g1_6.gates[0].state, "PASSED");
assert.ok(g1_6.gates[0].explanation.includes("above the entry corridor"));
assert.ok(!g1_6.gates[0].explanation.includes("positioned inside"));
console.log("   [PASS] 1.6: Extended within 2% tolerance ($103.50 vs $98–$102) -> ABOVE / PASSED (Truthful text)");

// Case 1.7: Above tolerance ($105 vs $98–$102)
const c7 = evaluateCorridorLocation(105, 98, 102, 0.02);
assert.strictEqual(c7.isValid, true);
assert.strictEqual(c7.spatialRelation, "ABOVE");
assert.strictEqual(c7.isInside, false);
assert.strictEqual(c7.isAboveCeiling, true);
assert.strictEqual(c7.isWithinTolerance, false);
assert.ok(c7.description.includes("extended >2% past the accumulation corridor"));

const g1_7 = resolveDecisionReadiness({ currentPrice: 105, optimalEntryMin: 98, optimalEntryMax: 102 });
assert.strictEqual(g1_7.gates[0].state, "BLOCKING");
assert.ok(g1_7.gates[0].explanation.includes("extended >2% past the accumulation corridor"));
console.log("   [PASS] 1.7: Above tolerance ($105 vs $98–$102) -> ABOVE / BLOCKING");

// Case 1.8: Missing/non-finite price or bounds ($null, NaN, non-positive)
const c8a = evaluateCorridorLocation(null, 98, 102);
assert.strictEqual(c8a.isValid, false);
assert.strictEqual(c8a.spatialRelation, "UNAVAILABLE");
assert.strictEqual(c8a.isInside, false);

const c8b = evaluateCorridorLocation(100, NaN, 102);
assert.strictEqual(c8b.isValid, false);
assert.strictEqual(c8b.spatialRelation, "UNAVAILABLE");

const c8c = evaluateCorridorLocation(100, 98, -5);
assert.strictEqual(c8c.isValid, false);
assert.strictEqual(c8c.spatialRelation, "UNAVAILABLE");

const g1_8 = resolveDecisionReadiness({ currentPrice: 100, optimalEntryMin: null, optimalEntryMax: 102 });
assert.strictEqual(g1_8.gates[0].state, "UNAVAILABLE");
console.log("   [PASS] 1.8: Missing or non-finite inputs -> UNAVAILABLE (Fail-closed)");

// Case 1.9: Inverted bounds ($102 min vs $98 max) — Invariant enforcement
const c9 = evaluateCorridorLocation(100, 102, 98);
assert.strictEqual(c9.isValid, false, "Inverted bounds must strictly yield isValid: false");
assert.strictEqual(c9.spatialRelation, "UNAVAILABLE");
assert.strictEqual(c9.isInside, false);
assert.ok(c9.description.includes("inverted bounds"));

const g1_9 = resolveDecisionReadiness({ currentPrice: 100, optimalEntryMin: 102, optimalEntryMax: 98 });
assert.strictEqual(g1_9.gates[0].state, "UNAVAILABLE");
assert.ok(g1_9.gates[0].explanation.includes("inverted bounds"));
console.log("   [PASS] 1.9: Inverted bounds (102 > 98) -> UNAVAILABLE / Fail-closed (Never fabricated)");

// ═════════════════════════════════════════════════════════════════════════════
// PART 2: Decision Guidance & Blocked-Action Consistency (6 Cases)
// ═════════════════════════════════════════════════════════════════════════════
console.log("\nPart 2: Verifying Decision Guidance & Blocked-Action Consistency (6 Cases)...");

// Case 2.1: Inside corridor + trigger blocked
const g2_1 = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: false,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 18.0,
});
assert.strictEqual(g2_1.gates[0].state, "PASSED");
assert.strictEqual(g2_1.gates[1].state, "BLOCKING");
assert.strictEqual(g2_1.activeBlockingGate, "GATE_2_TRIGGER");
assert.strictEqual(g2_1.isExecutionReady, false);
assert.ok(g2_1.gates[1].explanation.includes("Price is inside the accumulation corridor, but confirmation trigger is pending"));

const hint2_1 = resolveTacticalExecutionHint({
  evalPrice: 100,
  entryMin: 98,
  entryMax: 102,
  corridorLoc: evaluateCorridorLocation(100, 98, 102),
  isActionable: false,
  isTriggerConfirmed: false,
  readinessResult: g2_1,
});
assert.ok(hint2_1 !== null);
assert.strictEqual(hint2_1?.label, "⏳ Trigger Pending");
assert.ok(hint2_1?.advice.includes("entry remains conditional"));
assert.ok(!hint2_1?.advice.includes("Scale in"), "Actionable scale-in advice must be suppressed when trigger is blocked");
assert.ok(!hint2_1?.advice.includes("30% initial size"));
console.log("   [PASS] 2.1: Inside corridor + trigger blocked -> Scale-in suppressed, conditional guidance active");

// Case 2.2: Above corridor within tolerance + trigger blocked
const g2_2 = resolveDecisionReadiness({
  currentPrice: 103.50,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: false,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 18.0,
});
assert.strictEqual(g2_2.gates[0].state, "PASSED");
assert.strictEqual(g2_2.gates[1].state, "BLOCKING");
assert.strictEqual(g2_2.activeBlockingGate, "GATE_2_TRIGGER");
assert.strictEqual(g2_2.isExecutionReady, false);
assert.ok(g2_2.gates[0].explanation.includes("above the entry corridor"));
assert.ok(!g2_2.gates[0].explanation.includes("positioned inside"));
assert.ok(g2_2.gates[1].explanation.includes("Price is above the entry corridor (within extension tolerance), but confirmation trigger is pending"));

const hint2_2 = resolveTacticalExecutionHint({
  evalPrice: 103.50,
  entryMin: 98,
  entryMax: 102,
  corridorLoc: evaluateCorridorLocation(103.50, 98, 102),
  isActionable: false,
  isTriggerConfirmed: false,
  readinessResult: g2_2,
});
assert.ok(hint2_2 !== null);
assert.strictEqual(hint2_2?.label, "⚠️ Above Corridor Ceiling");
assert.ok(hint2_2?.advice.includes("above the entry corridor"));
assert.ok(hint2_2?.advice.includes("Existing extension tolerance applies, but entry remains conditional"));
assert.ok(!hint2_2?.advice.includes("Scale in"), "Actionable scale-in advice must be suppressed when trigger is blocked");
assert.ok(!hint2_2?.advice.includes("inside the accumulation corridor"), "Above-ceiling price must NEVER claim to be inside");
console.log("   [PASS] 2.2: Above corridor within tolerance + trigger blocked -> Truthful location + scale-in suppressed");

// Case 2.3: Canonical action Wait despite locally passing gate
const g2_3 = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: false, // Canonical trigger pending
});
const pres2_3 = resolvePresentationState({
  decisionState: "VALID_SETUP",
  decisionStateLabel: "Valid Setup — Awaiting Trigger",
  isActionable: false, // Authoritative canonical wait posture
  readinessResult: g2_3,
});
assert.strictEqual(pres2_3.actionabilityLabel, "NOT ACTIONABLE");
assert.strictEqual(pres2_3.headlineLabel, "Valid Setup — Awaiting Trigger");

const hint2_3 = resolveTacticalExecutionHint({
  evalPrice: 100,
  entryMin: 98,
  entryMax: 102,
  corridorLoc: evaluateCorridorLocation(100, 98, 102),
  isActionable: false,
  isTriggerConfirmed: false,
  readinessResult: g2_3,
});
assert.ok(!hint2_3?.advice.includes("Scale in with 30% initial size"));
console.log("   [PASS] 2.3: Canonical action Wait authoritative -> NOT ACTIONABLE preserved, scale-in suppressed");

// Case 2.4: Missing/unknown required confirmation
const g2_4 = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: null, // Unknown/missing telemetry
});
assert.strictEqual(g2_4.gates[0].state, "PASSED");
assert.strictEqual(g2_4.gates[1].state, "UNAVAILABLE");
assert.strictEqual(g2_4.activeBlockingGate, "GATE_2_TRIGGER");
assert.strictEqual(g2_4.isExecutionReady, false);
assert.strictEqual(g2_4.gates[1].explanation, "Trigger telemetry unavailable.");

const hint2_4 = resolveTacticalExecutionHint({
  evalPrice: 100,
  entryMin: 98,
  entryMax: 102,
  corridorLoc: evaluateCorridorLocation(100, 98, 102),
  isActionable: false,
  isTriggerConfirmed: null,
  readinessResult: g2_4,
});
assert.ok(!hint2_4?.advice.includes("Scale in with 30% initial size"));
console.log("   [PASS] 2.4: Missing trigger confirmation -> Gate 2 UNAVAILABLE / Fail-closed");

// Case 2.5: Fully satisfied established canonical state
const g2_5 = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  isConfirmed: true,
  riskRewardRatio: 2.5,
  stopLoss: 95,
  takeProfit1: 112.5,
  vix: 17.5,
});
assert.strictEqual(g2_5.gates[0].state, "PASSED");
assert.strictEqual(g2_5.gates[1].state, "PASSED");
assert.strictEqual(g2_5.gates[2].state, "PASSED");
assert.strictEqual(g2_5.activeBlockingGate, "NONE");
assert.strictEqual(g2_5.isExecutionReady, true);

const pres2_5 = resolvePresentationState({
  decisionState: "ACTIONABLE_SETUP",
  isActionable: true,
  readinessResult: g2_5,
});
assert.strictEqual(pres2_5.actionabilityLabel, "ACTIONABLE");

// With price at $101.50 (zonePositionPct ~87.5% > 65%), actionable guidance should be provided
const hint2_5 = resolveTacticalExecutionHint({
  evalPrice: 101.50,
  entryMin: 98,
  entryMax: 102,
  corridorLoc: evaluateCorridorLocation(101.50, 98, 102),
  zonePositionPct: 87.5,
  riskRewardRatio: 2.5,
  isActionable: true,
  isTriggerConfirmed: true,
  readinessResult: g2_5,
});
assert.ok(hint2_5 !== null);
assert.strictEqual(hint2_5?.label, "⚠️ Near Zone Ceiling");
assert.ok(hint2_5?.advice.includes("Scale in with 30% initial size"));
console.log("   [PASS] 2.5: Fully satisfied canonical state -> ACTIONABLE, tactical guidance unlocked");

// Case 2.6: Explicitly not-applicable condition (Stage 4 Markdown)
const g2_6 = resolveDecisionReadiness({
  currentPrice: 100,
  optimalEntryMin: 98,
  optimalEntryMax: 102,
  stagePhase: "Stage 4 Markdown",
  candleCount: 100,
});
assert.strictEqual(g2_6.gates[0].state, "BLOCKING");
assert.strictEqual(g2_6.gates[1].state, "PENDING_DEPENDENCY");
assert.strictEqual(g2_6.gates[2].state, "PENDING_DEPENDENCY");
assert.strictEqual(g2_6.activeBlockingGate, "GATE_1_LOCATION");
assert.strictEqual(g2_6.isExecutionReady, false);

const hint2_6 = resolveTacticalExecutionHint({
  evalPrice: 100,
  entryMin: 98,
  entryMax: 102,
  isStage4: true,
  corridorLoc: evaluateCorridorLocation(100, 98, 102),
  isActionable: false,
  isTriggerConfirmed: false,
  readinessResult: g2_6,
});
assert.strictEqual(hint2_6, null, "Tactical hints must be completely suppressed during Stage 4 breakdown");
console.log("   [PASS] 2.6: Stage 4 distribution -> Gate 1 BLOCKING, downstream PENDING, tactical hints suppressed");

// ═════════════════════════════════════════════════════════════════════════════
// PART 3: LNTH Live Fixture Audit & Verification
// ═════════════════════════════════════════════════════════════════════════════
console.log("\nPart 3: Verifying LNTH Live Fixture Audit ($341.91 vs $325.07–$339.38)...");

const lnthPrice = 341.91;
const lnthMin = 325.07;
const lnthMax = 339.38;

// 1. Evaluate pure location
const lnthLoc = evaluateCorridorLocation(lnthPrice, lnthMin, lnthMax, 0.02);
assert.strictEqual(lnthLoc.isValid, true);
assert.strictEqual(lnthLoc.spatialRelation, "ABOVE", "LNTH $341.91 is above ceiling $339.38");
assert.strictEqual(lnthLoc.isInside, false, "LNTH isInside must strictly be false");
assert.strictEqual(lnthLoc.isAboveCeiling, true);
assert.strictEqual(lnthLoc.isWithinTolerance, true, "LNTH is within 2% tolerance (339.38 * 1.02 = 346.1676)");
assert.ok(lnthLoc.description.includes("Price ($341.91) is above the entry corridor ($325.07–$339.38)"));
assert.ok(lnthLoc.description.includes("Existing extension tolerance applies"));
assert.ok(!lnthLoc.description.includes("positioned inside"), "Must NEVER claim LNTH is inside corridor");

// 2. Evaluate Decision Readiness (Trigger Pending)
const lnthReadiness = resolveDecisionReadiness({
  currentPrice: lnthPrice,
  optimalEntryMin: lnthMin,
  optimalEntryMax: lnthMax,
  isConfirmed: false,
  candleCount: 100,
  riskRewardRatio: 2.4,
  stopLoss: 315.00,
  takeProfit1: 395.00,
  vix: 18.2,
});

assert.strictEqual(lnthReadiness.gates[0].state, "PASSED", "Gate 1 preserves tolerance pass");
assert.ok(lnthReadiness.gates[0].explanation.includes("Price ($341.91) is above the entry corridor ($325.07–$339.38)"));
assert.ok(!lnthReadiness.gates[0].explanation.includes("positioned inside"));

assert.strictEqual(lnthReadiness.gates[1].state, "BLOCKING", "Gate 2 blocks on unconfirmed trigger");
assert.ok(lnthReadiness.gates[1].explanation.includes("Price is above the entry corridor (within extension tolerance), but confirmation trigger is pending"));
assert.ok(!lnthReadiness.gates[1].explanation.includes("Price is inside the accumulation corridor"));

assert.strictEqual(lnthReadiness.gates[2].state, "PENDING_DEPENDENCY");
assert.strictEqual(lnthReadiness.activeBlockingGate, "GATE_2_TRIGGER");
assert.strictEqual(lnthReadiness.isExecutionReady, false);

// 3. Evaluate Tactical Guidance (Contradiction Elimination)
const lnthHint = resolveTacticalExecutionHint({
  evalPrice: lnthPrice,
  entryMin: lnthMin,
  entryMax: lnthMax,
  corridorLoc: lnthLoc,
  inZone: false,
  zonePositionPct: 50,
  riskRewardRatio: 2.4,
  isActionable: false,
  isTriggerConfirmed: false,
  readinessResult: lnthReadiness,
});

assert.ok(lnthHint !== null);
assert.strictEqual(lnthHint?.label, "⚠️ Above Corridor Ceiling");
assert.ok(lnthHint?.advice.includes("Spot ($341.91) is above the entry corridor ($325.07–$339.38)"));
assert.ok(lnthHint?.advice.includes("Existing extension tolerance applies, but entry remains conditional"));
assert.ok(lnthHint?.advice.includes("Wait for required trigger confirmation"));
assert.ok(!lnthHint?.advice.includes("Scale in with 30% initial size"), "Defect resolved: contradictory scale-in advice suppressed!");
assert.ok(!lnthHint?.advice.includes("inside the accumulation corridor"), "Defect resolved: spatial contradiction eliminated!");

// 4. Evaluate Unmet Conditions
const mockLnthInsight: QuantitativeInsight = {
  symbol: "LNTH",
  price: lnthPrice,
  setupScore: 82,
  terminalState: {
    isActionable: false,
    overallEligibility: "ELIGIBLE",
    headlineExplanation: "Awaiting volume breakout confirmation.",
  },
  standard: {
    keyLevels: {
      entryMin: lnthMin,
      entryMax: lnthMax,
      watchZone: "$325.07 – $339.38",
      sma50: 310.0,
    },
  },
};

const unmetConditions = deriveUnmetConditions(mockLnthInsight);
const corridorCondition = unmetConditions.find((c) => c.id === "corridor");
assert.ok(corridorCondition !== undefined);
assert.strictEqual(corridorCondition?.status, "UNMET");
assert.strictEqual(corridorCondition?.title, "Pullback into Accumulation Corridor");
assert.ok(corridorCondition?.description.includes("extended above preferred entry corridor"));

console.log("   [PASS] 3.1: LNTH spatial classification: ABOVE / isInside=false");
console.log("   [PASS] 3.2: LNTH Gate 1: PASSED (Truthful explanation with tolerance disclaimer)");
console.log("   [PASS] 3.3: LNTH Gate 2: BLOCKING (Truthful above-corridor explanation)");
console.log("   [PASS] 3.4: LNTH Tactical Hint: Contradictory scale-in instructions eliminated");
console.log("   [PASS] 3.5: LNTH Unmet Conditions: Correctly reports UNMET pullback condition");

console.log("\nALL WORKSTREAM D SLICE 1 REGRESSION TESTS PASSED (100% SUCCESS)!\n");
