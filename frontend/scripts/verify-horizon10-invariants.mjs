/**
 * Verification Suite: Horizon 10 Behavioral Safety & Decision Reduction Invariants
 *
 * Tests fail-closed enforcement of:
 * - INV-OI97-P: Cognitive Trading Discipline
 * - INV-OI98-P: Household Capital Protection
 * - INV-OI99-P: Recommendation Overload Prevention
 * - INV-OI100-P: Human Agency Preservation
 * - INV-OI101-P: Decision Simplicity
 */

import { strict as assert } from 'node:assert';

let passed = 0;
let failed = 0;

function testAssert(condition, message) {
  if (condition) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL: ${message}`);
  }
}

function testEqual(actual, expected, message) {
  if (actual === expected) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL: ${message} (expected: ${expected}, got: ${actual})`);
  }
}

// -----------------------------------------------------------------------------
// INVARIANT IMPLEMENTATIONS
// -----------------------------------------------------------------------------

function verifyCognitiveTradingDiscipline(recoveryScore, recentLossStreak, dailyDrawdownPct) {
  const violations = [];
  let enforcementAction = "PROCEED";
  let reason = "Cognitive and risk parameters nominal.";

  if (recoveryScore < 55) {
    violations.push(`INV-OI97-P VIOLATION: Recovery score ${recoveryScore}% below cognitive safety floor (55%). High tilt probability.`);
    enforcementAction = "LOCKOUT";
    reason = "Recovery score degraded. Impulse trading shield engaged.";
  } else if (dailyDrawdownPct > 0.03) {
    violations.push(`INV-OI97-P VIOLATION: Daily portfolio drawdown ${(dailyDrawdownPct * 100).toFixed(1)}% exceeds 3.0% safety threshold.`);
    enforcementAction = "LOCKOUT";
    reason = "Intra-day drawdown limit breached. Capital preservation lockout active.";
  } else if (recentLossStreak >= 2) {
    violations.push(`INV-OI97-P VIOLATION: Consecutive loss streak of ${recentLossStreak} detected (>= 2). Revenge trading circuit breaker tripped.`);
    enforcementAction = "PAPER_ONLY";
    reason = "Consecutive losses detected. Trading restricted to paper mode.";
  }

  const compliant = violations.length === 0;
  return {
    compliant,
    safeToTrade: enforcementAction === "PROCEED",
    enforcementAction,
    reason,
    violations
  };
}

function verifyHouseholdCapitalProtection(currentLiquidCash, monthlyEssentialBurn, proposedCapitalDeployment) {
  const violations = [];
  const safeBurn = Math.max(1, monthlyEssentialBurn);
  const currentRunway = currentLiquidCash / safeBurn;
  const remainingCash = currentLiquidCash - proposedCapitalDeployment;
  const postDeploymentRunway = remainingCash / safeBurn;

  if (postDeploymentRunway < 6.0) {
    violations.push(
      `INV-OI98-P VIOLATION: Proposed capital deployment of $${proposedCapitalDeployment.toLocaleString()} reduces household runway to ${postDeploymentRunway.toFixed(1)} months (strict minimum floor is 6.0 months).`
    );
  }

  if (proposedCapitalDeployment > currentLiquidCash) {
    violations.push(
      `INV-OI98-P VIOLATION: Proposed capital deployment ($${proposedCapitalDeployment}) exceeds available liquid cash ($${currentLiquidCash}).`
    );
  }

  return {
    compliant: violations.length === 0,
    currentRunwayMonths: Number(currentRunway.toFixed(2)),
    postDeploymentRunwayMonths: Number(postDeploymentRunway.toFixed(2)),
    violations
  };
}

function verifyRecommendationOverloadPrevention(actions) {
  const violations = [];
  const visibleCount = actions.length;

  if (visibleCount > 3) {
    violations.push(
      `INV-OI99-P VIOLATION: Surfaced ${visibleCount} recommendations simultaneously (maximum allowable is 3 to prevent choice paralysis).`
    );
  }

  return {
    compliant: violations.length === 0,
    visibleCount,
    violations
  };
}

function verifyHumanAgencyPreservation(actionType, hasExplicitUserConsent) {
  const violations = [];

  if (actionType === "AUTONOMOUS_EXECUTION") {
    violations.push(
      "INV-OI100-P FATAL VIOLATION: Autonomous financial or career mutation attempted. Platform strictly forbids unconfirmed mutations."
    );
  } else if (!hasExplicitUserConsent) {
    violations.push(
      "INV-OI100-P VIOLATION: Pending explicit two-factor human confirmation before execution."
    );
  }

  return {
    compliant: violations.length === 0,
    violationRisk: violations.length > 0 ? violations[0] : "NONE",
    violations
  };
}

function verifyDecisionSimplicity(actions) {
  const violations = [];
  const primaryCount = actions.filter(a => a.isPrimary).length;
  const secondaryCount = actions.filter(a => !a.isPrimary).length;

  if (primaryCount !== 1) {
    violations.push(
      `INV-OI101-P VIOLATION: Found ${primaryCount} primary actions (exactly 1 is required for decision clarity).`
    );
  }

  if (secondaryCount > 2) {
    violations.push(
      `INV-OI101-P VIOLATION: Found ${secondaryCount} secondary actions (maximum allowable is 2).`
    );
  }

  return {
    compliant: violations.length === 0,
    primaryCount,
    secondaryCount,
    violations
  };
}

console.log("");
console.log("===============================================================================");
console.log("HORIZON 10: BEHAVIORAL SAFETY & DECISION REDUCTION INVARIANTS VERIFICATION");
console.log("===============================================================================");

// -----------------------------------------------------------------------------
// SUITE 1: INV-OI97-P Cognitive Trading Discipline Invariant
// -----------------------------------------------------------------------------
console.log("\n--- Suite 1: INV-OI97-P Cognitive Trading Discipline Invariant ---");

// Nominal baseline
const r1 = verifyCognitiveTradingDiscipline(85, 0, 0.005);
testAssert(r1.compliant === true, "Nominal parameters pass INV-OI97-P");
testEqual(r1.enforcementAction, "PROCEED", "Nominal parameters yield PROCEED action");
testAssert(r1.safeToTrade === true, "Nominal parameters allow safe trade execution");
testEqual(r1.violations.length, 0, "No violations reported in nominal state");

// Low recovery lockout
const r2 = verifyCognitiveTradingDiscipline(54, 0, 0.005);
testAssert(r2.compliant === false, "Recovery 54% fails INV-OI97-P fail-closed");
testEqual(r2.enforcementAction, "LOCKOUT", "Low recovery triggers LOCKOUT");
testAssert(r2.safeToTrade === false, "Low recovery forbids trade execution");
testAssert(r2.violations.some(v => v.includes("Recovery score 54%")), "Explicit violation message for low recovery");

// Boundary recovery (55%)
const r3 = verifyCognitiveTradingDiscipline(55, 0, 0.005);
testAssert(r3.compliant === true, "Recovery at boundary 55% passes INV-OI97-P");
testEqual(r3.enforcementAction, "PROCEED", "Boundary recovery yields PROCEED");

// Consecutive loss streak (>= 2)
const r4 = verifyCognitiveTradingDiscipline(80, 2, 0.01);
testAssert(r4.compliant === false, "Loss streak of 2 fails INV-OI97-P");
testEqual(r4.enforcementAction, "PAPER_ONLY", "Loss streak of 2 triggers PAPER_ONLY mode");
testAssert(r4.safeToTrade === false, "Loss streak prohibits real trade execution");
testAssert(r4.violations.some(v => v.includes("streak of 2")), "Revenge trading breaker message generated");

// Single loss streak (nominal)
const r5 = verifyCognitiveTradingDiscipline(80, 1, 0.01);
testAssert(r5.compliant === true, "Loss streak of 1 passes INV-OI97-P");
testEqual(r5.enforcementAction, "PROCEED", "Single loss permits trading");

// Excessive intra-day drawdown (> 3.0%)
const r6 = verifyCognitiveTradingDiscipline(80, 0, 0.035);
testAssert(r6.compliant === false, "Drawdown of 3.5% fails INV-OI97-P");
testEqual(r6.enforcementAction, "LOCKOUT", "Drawdown > 3% triggers LOCKOUT");
testAssert(r6.safeToTrade === false, "Excessive drawdown stops trading");

// Drawdown boundary (3.0%)
const r7 = verifyCognitiveTradingDiscipline(80, 0, 0.030);
testAssert(r7.compliant === true, "Drawdown of 3.0% passes INV-OI97-P");

// -----------------------------------------------------------------------------
// SUITE 2: INV-OI98-P Household Capital Protection Invariant
// -----------------------------------------------------------------------------
console.log("\n--- Suite 2: INV-OI98-P Household Capital Protection Invariant ---");

// Nominal capital deployment (runway remains 9.0 mo >= 6.0 mo)
const c1 = verifyHouseholdCapitalProtection(50000, 5000, 5000);
testAssert(c1.compliant === true, "Capital deployment leaving 9 months runway passes INV-OI98-P");
testEqual(c1.currentRunwayMonths, 10.0, "Current runway correctly calculated as 10.0 months");
testEqual(c1.postDeploymentRunwayMonths, 9.0, "Post-deployment runway correctly calculated as 9.0 months");
testEqual(c1.violations.length, 0, "Zero violations on safe deployment");

// Capital deployment violating 6-month floor (runway drops to 5.0 mo)
const c2 = verifyHouseholdCapitalProtection(35000, 5000, 10000);
testAssert(c2.compliant === false, "Capital deployment leaving 5 months runway fails INV-OI98-P");
testEqual(c2.postDeploymentRunwayMonths, 5.0, "Post-deployment runway reflects 5.0 months");
testAssert(c2.violations.some(v => v.includes("strict minimum floor is 6.0 months")), "Clear warning about 6.0 month floor");

// Exact 6-month boundary
const c3 = verifyHouseholdCapitalProtection(35000, 5000, 5000);
testAssert(c3.compliant === true, "Capital deployment leaving exactly 6.0 months passes INV-OI98-P");
testEqual(c3.postDeploymentRunwayMonths, 6.0, "Post-deployment runway is exactly 6.0");

// Deployment exceeding available cash
const c4 = verifyHouseholdCapitalProtection(30000, 5000, 35000);
testAssert(c4.compliant === false, "Deploying more cash than exists fails INV-OI98-P");
testAssert(c4.violations.some(v => v.includes("exceeds available liquid cash")), "Exceeds cash violation captured");

// -----------------------------------------------------------------------------
// SUITE 3: INV-OI99-P Recommendation Overload Prevention Invariant
// -----------------------------------------------------------------------------
console.log("\n--- Suite 3: INV-OI99-P Recommendation Overload Prevention Invariant ---");

const actions1 = [{ id: 'a1' }];
const o1 = verifyRecommendationOverloadPrevention(actions1);
testAssert(o1.compliant === true, "1 action passes INV-OI99-P");
testEqual(o1.visibleCount, 1, "Visible count matches 1");

const actions3 = [{ id: 'a1' }, { id: 'a2' }, { id: 'a3' }];
const o3 = verifyRecommendationOverloadPrevention(actions3);
testAssert(o3.compliant === true, "3 actions pass INV-OI99-P");
testEqual(o3.visibleCount, 3, "Visible count matches 3");

const actions4 = [{ id: 'a1' }, { id: 'a2' }, { id: 'a3' }, { id: 'a4' }];
const o4 = verifyRecommendationOverloadPrevention(actions4);
testAssert(o4.compliant === false, "4 actions fail INV-OI99-P overload test");
testAssert(o4.violations.some(v => v.includes("maximum allowable is 3")), "Choice paralysis violation flagged");

const actions8 = Array.from({ length: 8 }, (_, i) => ({ id: `act-${i}` }));
const o8 = verifyRecommendationOverloadPrevention(actions8);
testAssert(o8.compliant === false, "8 actions fail INV-OI99-P overload test");
testEqual(o8.visibleCount, 8, "Count accurately reports 8");

// -----------------------------------------------------------------------------
// SUITE 4: INV-OI100-P Human Agency Preservation Invariant
// -----------------------------------------------------------------------------
console.log("\n--- Suite 4: INV-OI100-P Human Agency Preservation Invariant ---");

// Suggestion with explicit user consent
const h1 = verifyHumanAgencyPreservation("SUGGESTION", true);
testAssert(h1.compliant === true, "Suggestion with explicit consent passes INV-OI100-P");
testEqual(h1.violationRisk, "NONE", "Zero violation risk when confirmed");

// Suggestion without explicit user consent
const h2 = verifyHumanAgencyPreservation("SUGGESTION", false);
testAssert(h2.compliant === false, "Suggestion without user consent cannot execute");
testAssert(h2.violations.some(v => v.includes("Pending explicit two-factor human confirmation")), "Human confirmation required");

// Autonomous execution attempt (FATAL violation)
const h3 = verifyHumanAgencyPreservation("AUTONOMOUS_EXECUTION", true);
testAssert(h3.compliant === false, "Autonomous execution is FATAL violation under INV-OI100-P");
testAssert(h3.violations.some(v => v.includes("FATAL VIOLATION: Autonomous financial or career mutation attempted")), "Fatal autonomous block enforced");

const h4 = verifyHumanAgencyPreservation("AUTONOMOUS_EXECUTION", false);
testAssert(h4.compliant === false, "Autonomous execution without consent is strictly rejected");

// -----------------------------------------------------------------------------
// SUITE 5: INV-OI101-P Decision Simplicity Invariant
// -----------------------------------------------------------------------------
console.log("\n--- Suite 5: INV-OI101-P Decision Simplicity Invariant ---");

// Valid: 1 Primary, 1 Secondary
const d1 = verifyDecisionSimplicity([
  { id: 'p1', isPrimary: true },
  { id: 's1', isPrimary: false }
]);
testAssert(d1.compliant === true, "1 Primary + 1 Secondary passes INV-OI101-P");
testEqual(d1.primaryCount, 1, "Primary count is 1");
testEqual(d1.secondaryCount, 1, "Secondary count is 1");

// Valid: 1 Primary, 2 Secondary
const d2 = verifyDecisionSimplicity([
  { id: 'p1', isPrimary: true },
  { id: 's1', isPrimary: false },
  { id: 's2', isPrimary: false }
]);
testAssert(d2.compliant === true, "1 Primary + 2 Secondary passes INV-OI101-P");

// Invalid: 0 Primary
const d3 = verifyDecisionSimplicity([
  { id: 's1', isPrimary: false },
  { id: 's2', isPrimary: false }
]);
testAssert(d3.compliant === false, "0 Primary fails INV-OI101-P");
testAssert(d3.violations.some(v => v.includes("Found 0 primary actions")), "Flagged missing primary action");

// Invalid: 2 Primary actions
const d4 = verifyDecisionSimplicity([
  { id: 'p1', isPrimary: true },
  { id: 'p2', isPrimary: true }
]);
testAssert(d4.compliant === false, "2 Primary actions fail INV-OI101-P");
testAssert(d4.violations.some(v => v.includes("Found 2 primary actions")), "Flagged multiple primary actions");

// Invalid: 3 Secondary actions (> 2)
const d5 = verifyDecisionSimplicity([
  { id: 'p1', isPrimary: true },
  { id: 's1', isPrimary: false },
  { id: 's2', isPrimary: false },
  { id: 's3', isPrimary: false }
]);
testAssert(d5.compliant === false, "3 Secondary actions fail INV-OI101-P");
testAssert(d5.violations.some(v => v.includes("Found 3 secondary actions")), "Flagged excessive secondary actions");

console.log(`\n===============================================================================`);
console.log(`VERIFICATION SUMMARY: ${passed} assertions passed, ${failed} failed`);
console.log(`===============================================================================\n`);

if (failed > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
