/**
 * ARX Terminal vNext - Phase 28 Milestone 2B Verification Suite
 * My Evolution Workspace & Personal Decision Development Platform
 * 
 * Verifies:
 * - Suite 1: Contracts & Data Models (Milestones, Ledger, CRI Leaderboard, Journey Profile)
 * - Suite 2: INV-B9 Improvement Traceability Invariant (6 Validation Criteria)
 * - Suite 3: INV-B10 Behavior Conservation & Cohort Invariants (7 Validation Criteria)
 * - Suite 4: Longitudinal Evolution Milestones (Q4 2025 [62] -> Current [74] -> Target [80])
 * - Suite 5: Behavior Ledger & Capability Leaderboard (Adopted/Removed habits, CRI scores)
 * - Suite 6: UI Component Integrity, 4 Interaction Modes & Master Dashboard Integration
 * 
 * Target: >= 80 Assertions with 100% Pass Rate
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendRoot = path.resolve(__dirname, '..');
const projectRoot = path.resolve(frontendRoot, '..');

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function assert(condition, message) {
  totalTests++;
  if (condition) {
    passedTests++;
    console.log(`  ✓ ${message}`);
  } else {
    failedTests++;
    console.error(`  ✗ FAIL: ${message}`);
  }
}

function assertEqual(actual, expected, message) {
  totalTests++;
  if (actual === expected) {
    passedTests++;
    console.log(`  ✓ ${message}`);
  } else {
    failedTests++;
    console.error(`  ✗ FAIL: ${message} (expected: ${expected}, got: ${actual})`);
  }
}

console.log('\n========================================================================');
console.log('  ARX Terminal vNext: Phase 28 Milestone 2B Verification Suite');
console.log('  (My Evolution Workspace, INV-B9 Traceability & INV-B10 Conservation)');
console.log('========================================================================');

// Dynamic Imports of Engines & Fixtures
const {
  CANONICAL_EVOLUTION_MILESTONES,
  CANONICAL_BEHAVIOR_LEDGER,
  CANONICAL_CAPABILITY_LEADERBOARD,
  getCanonicalEvolutionJourney,
  verifyImprovementTraceability,
  verifyBehaviorConservation,
} = await import('../lib/telemetry/decisionEvolutionEngine.ts');

// ------------------------------------------------------------------------
// SUITE 1: CONTRACTS & DATA MODELS
// ------------------------------------------------------------------------
console.log('\nSUITE 1: Contracts & Data Models');

const journey = getCanonicalEvolutionJourney();
assert(Boolean(journey), 'Loads canonical evolution journey');
assertEqual(journey.userId, 'usr_exec_david', 'Matches canonical executive David identifier');
assertEqual(journey.currentDIR, 74, 'Current DIR score is strictly 74 points');
assertEqual(journey.startingDIR, 62, 'Starting baseline DIR is strictly 62 points (Q4 2025)');
assertEqual(journey.targetDIR, 80, 'Target DIR score is strictly 80 points (Q1 2027)');
assertEqual(journey.fourQuarterGain, 12, 'Four quarter net DIR gain is +12 points (62 -> 74)');
assertEqual(journey.learningVelocity, 84, 'Learning velocity is 84 (High)');
assertEqual(journey.percentileRank, 82, 'Percentile rank is 82nd percentile (Top 18%)');
assertEqual(journey.projectedMonthsToTarget, 4, 'Projected time to target is 4 months');
assertEqual(journey.projectionConfidence, 87, 'Projection confidence is 87%');

// Milestone contract structure
assert(Array.isArray(journey.milestones), 'Journey contains milestones array');
assertEqual(journey.milestones.length, 5, 'Journey has exactly 5 canonical milestones');
journey.milestones.forEach((m, idx) => {
  assert(Boolean(m.id), `Milestone [${idx}] has unique ID: ${m.id}`);
  assert(Boolean(m.quarter), `Milestone [${idx}] has quarter label: ${m.quarter}`);
  assert(typeof m.dirScore === 'number' && m.dirScore > 0, `Milestone [${idx}] has positive DIR score`);
  assert(Boolean(m.status), `Milestone [${idx}] has status`);
  assert(Array.isArray(m.adoptedHabits), `Milestone [${idx}] lists adopted habits`);
  assert(Array.isArray(m.stoppedHabits), `Milestone [${idx}] lists stopped habits`);
  assert(Boolean(m.problemStatement), `Milestone [${idx}] defines problem statement`);
  assert(Boolean(m.actionTaken), `Milestone [${idx}] defines action taken`);
  assert(Boolean(m.outcomeImpact), `Milestone [${idx}] defines outcome impact`);
  assert(Boolean(m.evidenceTrace), `Milestone [${idx}] has evidence trace`);
});

// Behavior Ledger items
assert(Array.isArray(journey.behaviorLedger), 'Journey contains behavior ledger');
assert(journey.behaviorLedger.length >= 8, 'Behavior ledger contains at least 8 habits');
journey.behaviorLedger.forEach((h, idx) => {
  assert(Boolean(h.id), `Ledger item [${idx}] has ID`);
  assert(Boolean(h.habitName), `Ledger item [${idx}] has habitName`);
  assert(['ADOPTED', 'REMOVED'].includes(h.type), `Ledger item [${idx}] has valid type (ADOPTED/REMOVED)`);
  assert(typeof h.dqImpactPoints === 'number', `Ledger item [${idx}] has numeric dqImpactPoints`);
  assert(Boolean(h.category), `Ledger item [${idx}] has category`);
  assert(Boolean(h.frequency), `Ledger item [${idx}] has frequency`);
});

// Capability ROI Leaderboard
assert(Array.isArray(journey.capabilityLeaderboard), 'Journey contains capability leaderboard');
assertEqual(journey.capabilityLeaderboard.length, 4, 'Capability leaderboard contains 4 capabilities');
journey.capabilityLeaderboard.forEach((c, idx) => {
  assert(Boolean(c.capabilityId), `Capability [${idx}] has capabilityId`);
  assert(Boolean(c.capabilityName), `Capability [${idx}] has capabilityName`);
  assert(typeof c.capabilityRoiIndex === 'number' && c.capabilityRoiIndex > 0, `Capability [${idx}] has positive CRI`);
  assert(typeof c.marginalDIRPoints === 'number', `Capability [${idx}] has marginalDIRPoints`);
  assert(Boolean(c.efficiencyBadge), `Capability [${idx}] has efficiencyBadge`);
});

// ------------------------------------------------------------------------
// SUITE 2: INV-B9 IMPROVEMENT TRACEABILITY INVARIANT
// ------------------------------------------------------------------------
console.log('\nSUITE 2: INV-B9 Improvement Traceability Invariant');

const invB9 = verifyImprovementTraceability(journey);
assert(invB9.isCompliant, 'INV-B9 overall compliance satisfied');
assertEqual(invB9.unexplainedResidualDrift, 0.6, 'Unexplained residual drift is strictly 0.6 DQ points');
assertEqual(invB9.attributionCoveragePercent, 95.0, 'Attribution coverage is strictly 95.0% (>= 95% threshold)');

// Validation criteria details
const b9Criteria = invB9.criteria;
assertEqual(b9Criteria.length, 6, 'INV-B9 checks exactly 6 validation criteria');

const cBar = b9Criteria.find(c => c.criterionId === 'INV-B9-001');
assert(Boolean(cBar) && cBar.passed, 'INV-B9-001 passed: BAR consistency check');

const cMistake = b9Criteria.find(c => c.criterionId === 'INV-B9-002');
assert(Boolean(cMistake) && cMistake.passed, 'INV-B9-002 passed: Repeat mistake rate monotonically decreases');

const cDrift = b9Criteria.find(c => c.criterionId === 'INV-B9-003');
assert(Boolean(cDrift) && cDrift.passed, 'INV-B9-003 passed: Drift stability is strictly < 1.0 DQ points');

const cLvi = b9Criteria.find(c => c.criterionId === 'INV-B9-004');
assert(Boolean(cLvi) && cLvi.passed, 'INV-B9-004 passed: Learning Velocity Index validity');

const cTimeline = b9Criteria.find(c => c.criterionId === 'INV-B9-005');
assert(Boolean(cTimeline) && cTimeline.passed, 'INV-B9-005 passed: Timeline strictly chronological');

const cCoverage = b9Criteria.find(c => c.criterionId === 'INV-B9-006');
assert(Boolean(cCoverage) && cCoverage.passed, 'INV-B9-006 passed: Attribution coverage >= 95%');

// ------------------------------------------------------------------------
// SUITE 3: INV-B10 BEHAVIOR CONSERVATION & COHORT INVARIANTS
// ------------------------------------------------------------------------
console.log('\nSUITE 3: INV-B10 Behavior Conservation & Cohort Invariants');

const invB10 = verifyBehaviorConservation(journey);
assert(invB10.isCompliant, 'INV-B10 overall compliance satisfied');
assertEqual(invB10.totalAttributedPoints, 11.4, 'Total attributed points strictly sum to 11.4 DQ points');
assertEqual(invB10.conservationDiscrepancy, 0, 'Conservation discrepancy is strictly 0 (no double-counting)');

// Validation criteria details
const b10Criteria = invB10.criteria;
assertEqual(b10Criteria.length, 7, 'INV-B10 checks exactly 7 validation criteria');

const cCohort = b10Criteria.find(c => c.criterionId === 'INV-B10-001');
assert(Boolean(cCohort) && cCohort.passed, 'INV-B10-001 passed: Single cohort assignment (Advanced Improvers)');

const cConsumer = b10Criteria.find(c => c.criterionId === 'INV-B10-002');
assert(Boolean(cConsumer) && cConsumer.passed, 'INV-B10-002 passed: Consumer accuracy validated');

const cOptimizer = b10Criteria.find(c => c.criterionId === 'INV-B10-003');
assert(Boolean(cOptimizer) && cOptimizer.passed, 'INV-B10-003 passed: Optimizer accuracy validated');

const cBounds = b10Criteria.find(c => c.criterionId === 'INV-B10-004');
assert(Boolean(cBounds) && cBounds.passed, 'INV-B10-004 passed: Confidence bounds on projection (78-83)');

const cSample = b10Criteria.find(c => c.criterionId === 'INV-B10-005');
assert(Boolean(cSample) && cSample.passed, 'INV-B10-005 passed: Sample size guard active (N >= 20)');

const cTrend = b10Criteria.find(c => c.criterionId === 'INV-B10-006');
assert(Boolean(cTrend) && cTrend.passed, 'INV-B10-006 passed: Trend consistency across milestones');

const cDoubleCount = b10Criteria.find(c => c.criterionId === 'INV-B10-007');
assert(Boolean(cDoubleCount) && cDoubleCount.passed, 'INV-B10-007 passed: Zero double-counting verified');

// ------------------------------------------------------------------------
// SUITE 4: LONGITUDINAL EVOLUTION MILESTONES
// ------------------------------------------------------------------------
console.log('\nSUITE 4: Longitudinal Evolution Milestones');

const m1 = CANONICAL_EVOLUTION_MILESTONES[0];
assertEqual(m1.quarter, '2025 Q4', 'Milestone 1 quarter is 2025 Q4');
assertEqual(m1.dirScore, 62, 'Milestone 1 score is 62');
assertEqual(m1.status, 'COMPLETED', 'Milestone 1 status is COMPLETED');
assert(m1.actionTaken.includes('Hard stop-loss rule'), 'Milestone 1 action reflects stop-loss rule');
assert(m1.evidenceTrace.includes('TRACE-2025Q4'), 'Milestone 1 trace ID matches format');

const m2 = CANONICAL_EVOLUTION_MILESTONES[1];
assertEqual(m2.quarter, '2026 Q1', 'Milestone 2 quarter is 2026 Q1');
assertEqual(m2.dirScore, 66, 'Milestone 2 score is 66 (+4 pts)');
assertEqual(m2.status, 'COMPLETED', 'Milestone 2 status is COMPLETED');
assert(m2.actionTaken.includes('Weekly outcome reviews'), 'Milestone 2 action reflects weekly outcome reviews');

const m3 = CANONICAL_EVOLUTION_MILESTONES[2];
assertEqual(m3.quarter, '2026 Q2', 'Milestone 3 quarter is 2026 Q2');
assertEqual(m3.dirScore, 70, 'Milestone 3 score is 70 (+4 pts)');
assertEqual(m3.status, 'COMPLETED', 'Milestone 3 status is COMPLETED');
assert(m3.actionTaken.includes('Pre-mortem requirement'), 'Milestone 3 action reflects pre-mortem friction');

const m4 = CANONICAL_EVOLUTION_MILESTONES[3];
assertEqual(m4.quarter, 'Current', 'Milestone 4 quarter is Current');
assertEqual(m4.dirScore, 74, 'Milestone 4 score is 74 (+4 pts, +12 pts total)');
assertEqual(m4.status, 'CURRENT', 'Milestone 4 status is CURRENT');
assert(m4.actionTaken.includes('AI Learning Coach integration'), 'Milestone 4 action reflects AI coach integration');

const m5 = CANONICAL_EVOLUTION_MILESTONES[4];
assertEqual(m5.quarter, 'Target (Q1 2027)', 'Milestone 5 quarter is Target (Q1 2027)');
assertEqual(m5.dirScore, 80, 'Milestone 5 score is 80 (+6 pts)');
assertEqual(m5.status, 'PROJECTED', 'Milestone 5 status is PROJECTED');
assert(m5.actionTaken.includes('Systematic Macro Hedging'), 'Milestone 5 action reflects systematic macro hedging');
assert(m5.confidenceInterval !== undefined, 'Milestone 5 contains confidence interval');
assertEqual(m5.confidenceInterval.lower, 78, 'Target lower bound is 78');
assertEqual(m5.confidenceInterval.upper, 83, 'Target upper bound is 83');

// ------------------------------------------------------------------------
// SUITE 5: BEHAVIOR LEDGER & CAPABILITY LEADERBOARD
// ------------------------------------------------------------------------
console.log('\nSUITE 5: Behavior Ledger & Capability Leaderboard');

const adoptedHabits = CANONICAL_BEHAVIOR_LEDGER.filter(h => h.type === 'ADOPTED');
const removedHabits = CANONICAL_BEHAVIOR_LEDGER.filter(h => h.type === 'REMOVED');
assert(adoptedHabits.length >= 5, 'Adopted habits count is at least 5');
assert(removedHabits.length >= 4, 'Removed habits count is at least 4');

// Specific habits presence
const preMortem = adoptedHabits.find(h => h.habitName.includes('Pre-Mortem'));
assert(Boolean(preMortem), 'Ledger includes Pre-Mortem Friction habit');
assertEqual(preMortem.dqImpactPoints, 3.8, 'Pre-Mortem habit impact is +3.8 DQ points');

const revengeTrade = removedHabits.find(h => h.habitName.includes('Revenge'));
assert(Boolean(revengeTrade), 'Ledger includes Revenge Trading Elimination');
assertEqual(revengeTrade.dqImpactPoints, 4.2, 'Revenge Trading elimination saved 4.2 DQ points');

// Capability Leaderboard order and CRI values
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[0].capabilityId, 'outcome_reviews', 'Rank 1 capability is Outcome Reviews');
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[0].capabilityRoiIndex, 6.0, 'Rank 1 CRI is 6.0');
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[0].marginalDIRPoints, 4.7, 'Rank 1 marginal gain is +4.7 pts');

assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[1].capabilityId, 'ai_coach', 'Rank 2 capability is AI Learning Coach');
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[1].capabilityRoiIndex, 4.1, 'Rank 2 CRI is 4.1');
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[1].marginalDIRPoints, 3.4, 'Rank 2 marginal gain is +3.4 pts');

assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[2].capabilityId, 'decision_journal', 'Rank 3 capability is Decision Journal');
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[2].capabilityRoiIndex, 2.8, 'Rank 3 CRI is 2.8');
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[2].marginalDIRPoints, 2.1, 'Rank 3 marginal gain is +2.1 pts');

assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[3].capabilityId, 'committee_governance', 'Rank 4 capability is Committee Governance');
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[3].capabilityRoiIndex, 1.2, 'Rank 4 CRI is 1.2');
assertEqual(CANONICAL_CAPABILITY_LEADERBOARD[3].marginalDIRPoints, 1.2, 'Rank 4 marginal gain is +1.2 pts');

// Verify CRI is strictly decreasing
for (let i = 0; i < CANONICAL_CAPABILITY_LEADERBOARD.length - 1; i++) {
  assert(
    CANONICAL_CAPABILITY_LEADERBOARD[i].capabilityRoiIndex >= CANONICAL_CAPABILITY_LEADERBOARD[i + 1].capabilityRoiIndex,
    `CRI decreases monotonically: rank ${i + 1} (${CANONICAL_CAPABILITY_LEADERBOARD[i].capabilityRoiIndex}) >= rank ${i + 2} (${CANONICAL_CAPABILITY_LEADERBOARD[i + 1].capabilityRoiIndex})`
  );
}

// ------------------------------------------------------------------------
// SUITE 6: UI COMPONENT INTEGRITY & MASTER DASHBOARD INTEGRATION
// ------------------------------------------------------------------------
console.log('\nSUITE 6: UI Component Integrity & Master Dashboard Integration');

const mewPath = path.join(frontendRoot, 'components', 'behavioral', 'MyEvolutionWorkspace.tsx');
assert(fs.existsSync(mewPath), 'MyEvolutionWorkspace.tsx exists');
const mewContent = fs.readFileSync(mewPath, 'utf8');

// Test IDs verification
assert(mewContent.includes('data-testid="my-evolution-workspace"'), 'Renders data-testid="my-evolution-workspace"');
assert(mewContent.includes('data-testid="evolution-hero"'), 'Renders data-testid="evolution-hero"');
assert(mewContent.includes('data-testid="interaction-mode-selector"'), 'Renders data-testid="interaction-mode-selector"');
assert(mewContent.includes('data-testid="progression-timeline"'), 'Renders data-testid="progression-timeline"');
assert(mewContent.includes('data-testid="behavior-ledger"'), 'Renders data-testid="behavior-ledger"');
assert(mewContent.includes('data-testid="capability-roi-leaderboard"'), 'Renders data-testid="capability-roi-leaderboard"');
assert(mewContent.includes('data-testid="ai-evolution-coach"'), 'Renders data-testid="ai-evolution-coach"');

// 4 Interaction Modes
assert(mewContent.includes("'SUMMARY'"), 'Supports SUMMARY mode (<5s fast read)');
assert(mewContent.includes("'EXPLORATION'"), 'Supports EXPLORATION mode (interactive timeline)');
assert(mewContent.includes("'ANALYSIS'"), 'Supports ANALYSIS mode (deep behavior ledger & CRI)');
assert(mewContent.includes("'PROJECTION'"), 'Supports PROJECTION mode (forward forecast to 80)');

// Design system & A11y invariants
assert(mewContent.includes('min-h-[44px]'), 'Enforces touch target floor min-h-[44px]');
assert(!mewContent.includes('text-cyan-500'), 'Adheres strictly to Anti-Cyan palette (no text-cyan-500)');
assert(mewContent.includes('aria-label='), 'Includes accessible ARIA labels');

// Master Dashboard Mounting
const masterPath = path.join(frontendRoot, 'components', 'behavioral', 'Phase28MasterDashboard.tsx');
assert(fs.existsSync(masterPath), 'Phase28MasterDashboard.tsx exists');
const masterContent = fs.readFileSync(masterPath, 'utf8');
assert(masterContent.includes("import MyEvolutionWorkspace from './MyEvolutionWorkspace';"), 'Master dashboard imports MyEvolutionWorkspace');
assert(masterContent.includes("<MyEvolutionWorkspace />"), 'Master dashboard renders <MyEvolutionWorkspace />');
assert(masterContent.includes("'evolution'"), 'Master dashboard includes evolution tab state');

console.log('\n========================================================================');
console.log(`  Phase 28 Milestone 2B Verification Completed: ${passedTests} Passed, ${failedTests} Failed`);
console.log('========================================================================\n');

if (failedTests > 0) {
  process.exit(1);
}
