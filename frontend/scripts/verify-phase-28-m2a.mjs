/**
 * ARX Terminal vNext - Phase 28 Milestone 2A Verification Suite
 * Executive Narrative Experience & Behavioral Intelligence Layer
 * 
 * Verifies:
 * - Suite 1: Contracts & Governance Fixtures (INV-B7, INV-B8)
 * - Suite 2: INV-B7 Narrative Determinism (100 Iterations Audit, Zero Variance)
 * - Suite 3: INV-B8 Actionability Invariant (Observation -> Learning -> Action -> Trace)
 * - Suite 4: All 7 Narrative State Resolvers (Healthy, Improving, Plateau, Declining, New User, Inactive, Low Confidence)
 * - Suite 5: Capability Impact Attribution & Capability ROI Index (CRI)
 * - Suite 6: Executive Home UI Surfaces & Invariants
 * 
 * Target: >= 50 Assertions with 100% Pass Rate
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
console.log('  ARX Terminal vNext: Phase 28 Milestone 2A Verification Suite');
console.log('  (Executive Narrative Experience, INV-B7 Determinism & INV-B8 Actionability)');
console.log('========================================================================');

// Dynamic Imports of Engines & Fixtures
const {
  resolveNarrativeState,
  generateExecutiveNarrative,
  generateCanonicalExecutiveNarrative,
} = await import('../lib/telemetry/executiveNarrativeEngine.ts');

const {
  evaluateCapabilityAttribution,
  CANONICAL_CAPABILITY_ATTRIBUTION,
} = await import('../lib/telemetry/capabilityImpactEngine.ts');

const {
  INV_B7_FIXTURE,
  INV_B8_FIXTURE,
  CANONICAL_NARRATIVE_FIXTURES,
} = await import('../fixtures/behavioral-fixtures.ts');

// ------------------------------------------------------------------------
// SUITE 1: CONTRACTS & GOVERNANCE FIXTURES (INV-B7 & INV-B8)
// ------------------------------------------------------------------------
console.log('\nSUITE 1: Contracts & Governance Fixtures');

const calculatedBAR = Number(
  ((INV_B7_FIXTURE.recommendationsFollowed / INV_B7_FIXTURE.recommendationsIssued) * 100).toFixed(1)
);
assertEqual(
  calculatedBAR,
  INV_B7_FIXTURE.expected.behavioralAdoptionRate,
  'INV-B7 Fixture: Behavioral Adoption Rate equals strictly 70.5% (79/112)'
);

assertEqual(
  INV_B8_FIXTURE.expected.lvi,
  84.0,
  'INV-B8 Fixture: Learning Velocity Index equals strictly 84.0'
);

assert(Boolean(CANONICAL_NARRATIVE_FIXTURES.HEALTHY), 'Exports CANONICAL_NARRATIVE_FIXTURES.HEALTHY');
assert(Boolean(CANONICAL_NARRATIVE_FIXTURES.PLATEAU), 'Exports CANONICAL_NARRATIVE_FIXTURES.PLATEAU');
assert(Boolean(CANONICAL_NARRATIVE_FIXTURES.DECLINING), 'Exports CANONICAL_NARRATIVE_FIXTURES.DECLINING');
assert(Boolean(CANONICAL_NARRATIVE_FIXTURES.NEW_USER), 'Exports CANONICAL_NARRATIVE_FIXTURES.NEW_USER');
assert(Boolean(CANONICAL_NARRATIVE_FIXTURES.INACTIVE), 'Exports CANONICAL_NARRATIVE_FIXTURES.INACTIVE');
assert(Boolean(CANONICAL_NARRATIVE_FIXTURES.LOW_CONFIDENCE), 'Exports CANONICAL_NARRATIVE_FIXTURES.LOW_CONFIDENCE');

// ------------------------------------------------------------------------
// SUITE 2: INV-B7 NARRATIVE DETERMINISM (100 Iterations Audit)
// ------------------------------------------------------------------------
console.log('\nSUITE 2: INV-B7 Narrative Determinism (100 Iterations Audit)');

const baseInputs = {
  dir: 84,
  dirTrend: 6.2,
  learningVelocity: 84,
  confidence: 92,
  decisionCount: 42,
  daysSinceLastActivity: 1,
  topDriver: 'Institutional Accumulation',
  topWeakness: 'Regime Deterioration',
  userName: 'David',
  dateString: 'Monday 07 September 2026',
};

const initialRun = generateExecutiveNarrative(baseInputs);
const initialSerialized = JSON.stringify(initialRun);
let varianceDetected = false;

for (let i = 0; i < 100; i++) {
  const currentRun = generateExecutiveNarrative(baseInputs);
  if (JSON.stringify(currentRun) !== initialSerialized) {
    varianceDetected = true;
    break;
  }
}

assert(!varianceDetected, 'INV-B7-01: 100 deterministic narrative generations produce 100 identical outputs (0 variance)');

// VERIFY-B7-02: Predictable state transitions
const testHealthy = resolveNarrativeState({ dir: 84, dirTrend: 6.2, confidence: 92, decisionCount: 40 });
assertEqual(testHealthy, 'HEALTHY', 'INV-B7-02a: DIR 84 + positive trend maps to HEALTHY');

const testPlateau = resolveNarrativeState({ dir: 78, dirTrend: 0.2, confidence: 89, decisionCount: 35 });
assertEqual(testPlateau, 'PLATEAU', 'INV-B7-02b: DIR 78 + flat trend maps to PLATEAU');

const testDeclining = resolveNarrativeState({ dir: 65, dirTrend: -11.0, confidence: 88, decisionCount: 30 });
assertEqual(testDeclining, 'DECLINING', 'INV-B7-02c: DIR 65 + negative trend maps to DECLINING');

// VERIFY-B7-03: Observable facts only (No ungrounded speculative claims)
const narrativeText = `${initialRun.executiveSummary.observation} ${initialRun.executiveSummary.learning} ${initialRun.executiveSummary.recommendedAction}`;
assert(!narrativeText.toLowerCase().includes('you will outperform next month'), 'INV-B7-03a: Narrative avoids ungrounded future outperformance claims');
assert(!narrativeText.toLowerCase().includes('markets should rally'), 'INV-B7-03b: Narrative avoids speculative macro directional claims');
assert(narrativeText.includes('Institutional Accumulation'), 'INV-B7-03c: Narrative grounds claims in verified behavioral drivers');

// ------------------------------------------------------------------------
// SUITE 3: INV-B8 ACTIONABILITY INVARIANT
// ------------------------------------------------------------------------
console.log('\nSUITE 3: INV-B8 Actionability Invariant');

// VERIFY-B8-01: Every narrative contains Observation, Learning, and Action
assert(Boolean(initialRun.executiveSummary.observation), 'INV-B8-01a: Contains Observation');
assert(Boolean(initialRun.executiveSummary.learning), 'INV-B8-01b: Contains Learning');
assert(Boolean(initialRun.executiveSummary.recommendedAction), 'INV-B8-01c: Contains Recommended Action');

// VERIFY-B8-02: Action traceability
assert(Boolean(initialRun.executiveSummary.evidenceTrace), 'INV-B8-02a: Action resolves to Evidence Trace');
assert(initialRun.executiveSummary.evidenceTrace.includes('EV-'), 'INV-B8-02b: Evidence trace has formal identifier');
assert(Boolean(initialRun.topOpportunity.actionableDirective), 'INV-B8-02c: Top Opportunity provides actionable directive');
assert(Boolean(initialRun.topRisk.mitigationDirective), 'INV-B8-02d: Top Risk provides mitigation directive');

// VERIFY-B8-03: Action confidence
assert(typeof initialRun.executiveSummary.actionConfidence === 'number', 'INV-B8-03a: Action confidence is numeric');
assert(initialRun.executiveSummary.actionConfidence >= 50 && initialRun.executiveSummary.actionConfidence <= 100, 'INV-B8-03b: Action confidence is within [50, 100]% bounds');
assertEqual(initialRun.executiveSummary.actionConfidence, 91, 'INV-B8-03c: Canonical action confidence is strictly 91%');

// ------------------------------------------------------------------------
// SUITE 4: ALL 7 NARRATIVE STATE RESOLVERS
// ------------------------------------------------------------------------
console.log('\nSUITE 4: All 7 Narrative State Resolvers');

// State 1: Healthy
const sHealthy = generateExecutiveNarrative(CANONICAL_NARRATIVE_FIXTURES.HEALTHY);
assertEqual(sHealthy.state, 'HEALTHY', 'State 1: Healthy performance resolved');
assert(sHealthy.executiveSummary.recommendedAction.includes('accumulation setups'), 'State 1 Action valid');

// State 2: Plateau
const sPlateau = generateExecutiveNarrative(CANONICAL_NARRATIVE_FIXTURES.PLATEAU);
assertEqual(sPlateau.state, 'PLATEAU', 'State 2: Plateau state resolved');
assert(sPlateau.executiveSummary.recommendedAction.includes('journal reviews'), 'State 2 Action valid');

// State 3: Declining
const sDeclining = generateExecutiveNarrative(CANONICAL_NARRATIVE_FIXTURES.DECLINING);
assertEqual(sDeclining.state, 'DECLINING', 'State 3: Declining state resolved');
assert(sDeclining.executiveSummary.recommendedAction.includes('Reduce conviction'), 'State 3 Action valid');

// State 4: New User
const sNewUser = generateExecutiveNarrative(CANONICAL_NARRATIVE_FIXTURES.NEW_USER);
assertEqual(sNewUser.state, 'NEW_USER', 'State 4: New user onboarding state resolved');
assert(sNewUser.executiveSummary.recommendedAction.includes('Complete six additional decisions'), 'State 4 Action valid');

// State 5: Inactive
const sInactive = generateExecutiveNarrative(CANONICAL_NARRATIVE_FIXTURES.INACTIVE);
assertEqual(sInactive.state, 'INACTIVE', 'State 5: Inactive user state resolved');
assert(sInactive.executiveSummary.recommendedAction.includes('Review open predictions'), 'State 5 Action valid');

// State 6: Low Confidence
const sLowConf = generateExecutiveNarrative(CANONICAL_NARRATIVE_FIXTURES.LOW_CONFIDENCE);
assertEqual(sLowConf.state, 'LOW_CONFIDENCE', 'State 6: Low confidence state resolved');
assert(sLowConf.executiveSummary.recommendedAction.includes('Verify historical decisions'), 'State 6 Action valid');

// State 7: Improving (Intermediate score with positive trend)
const sImproving = generateExecutiveNarrative({
  dir: 74,
  dirTrend: 3.2,
  confidence: 88,
  decisionCount: 30,
  daysSinceLastActivity: 2,
});
assertEqual(sImproving.state, 'IMPROVING', 'State 7: Improving performance state resolved');

// Canonical institutional narrative
const canonicalNarrative = generateCanonicalExecutiveNarrative();
assertEqual(canonicalNarrative.dirScore, 84, 'Canonical DIR score is 84');
assertEqual(canonicalNarrative.headline, 'GOOD MORNING DAVID', 'Canonical greeting matches executive David');

// ------------------------------------------------------------------------
// SUITE 5: CAPABILITY IMPACT ATTRIBUTION & CAPABILITY ROI INDEX (CRI)
// ------------------------------------------------------------------------
console.log('\nSUITE 5: Capability Impact Attribution & Capability ROI Index (CRI)');

const capResult = evaluateCapabilityAttribution();

// Contributions
const orCap = capResult.capabilities.find(c => c.capabilityId === 'outcome_reviews');
assert(Boolean(orCap), 'Evaluates Outcome Reviews capability');
assertEqual(orCap.estimatedContribution, 4.7, 'Outcome Reviews contribution is +4.7 DQ points');
assertEqual(orCap.capabilityRoiIndex, 6.0, 'Outcome Reviews CRI is strictly 6.0 (highest return per usage)');

const coachCap = capResult.capabilities.find(c => c.capabilityId === 'ai_coach');
assert(Boolean(coachCap), 'Evaluates AI Learning Coach capability');
assertEqual(coachCap.estimatedContribution, 3.4, 'AI Learning Coach contribution is +3.4 DQ points');
assertEqual(coachCap.capabilityRoiIndex, 4.1, 'AI Learning Coach CRI is strictly 4.1');

const journalCap = capResult.capabilities.find(c => c.capabilityId === 'decision_journal');
assert(Boolean(journalCap), 'Evaluates Decision Journal capability');
assertEqual(journalCap.estimatedContribution, 2.1, 'Decision Journal contribution is +2.1 DQ points');
assertEqual(journalCap.capabilityRoiIndex, 2.8, 'Decision Journal CRI is strictly 2.8');

const commCap = capResult.capabilities.find(c => c.capabilityId === 'committee_governance');
assert(Boolean(commCap), 'Evaluates Committee Governance capability');
assertEqual(commCap.estimatedContribution, 1.2, 'Committee Governance contribution is +1.2 DQ points');
assertEqual(commCap.capabilityRoiIndex, 1.2, 'Committee Governance CRI is strictly 1.2');

// Improvement Conservation Invariant
assertEqual(capResult.totalImprovementPoints, 12.0, 'Total DQ improvement is 12.0 points');
assertEqual(capResult.explainedImprovementPoints, 11.4, 'Explained capability improvement is 11.4 points');
assertEqual(capResult.residualDriftPoints, 0.6, 'Residual drift is 0.6 points');
assert(capResult.isConservationSatisfied, 'Improvement Conservation Invariant satisfied (|explained + residual - total| <= 0.5)');
assertEqual(capResult.highestRoiCapability, 'Outcome Reviews & Resolution', 'Highest leverage feature correctly identified');

// ------------------------------------------------------------------------
// SUITE 6: UI COMPONENT INTEGRITY & DESIGN SYSTEM INVARIANTS
// ------------------------------------------------------------------------
console.log('\nSUITE 6: UI Component Integrity & Design System Invariants');

const enhPath = path.join(frontendRoot, 'components', 'behavioral', 'ExecutiveNarrativeHome.tsx');
assert(fs.existsSync(enhPath), 'ExecutiveNarrativeHome.tsx exists');
const enhContent = fs.readFileSync(enhPath, 'utf8');

assert(enhContent.includes('data-testid="executive-narrative-home"'), 'Renders data-testid="executive-narrative-home"');
assert(enhContent.includes('data-testid="executive-narrative-hero"'), 'Renders data-testid="executive-narrative-hero"');
assert(enhContent.includes('data-testid="recommended-action-box"'), 'Renders data-testid="recommended-action-box"');
assert(enhContent.includes('data-testid="top-opportunity-card"'), 'Renders data-testid="top-opportunity-card"');
assert(enhContent.includes('data-testid="top-risk-card"'), 'Renders data-testid="top-risk-card"');
assert(enhContent.includes('data-testid="capability-roi-section"'), 'Renders data-testid="capability-roi-section"');
assert(enhContent.includes('min-h-[44px]'), 'Enforces touch target floor (min-h-[44px])');
assert(!enhContent.includes('text-cyan-500'), 'Strictly adheres to Anti-Cyan palette (no raw cyan-500 text)');

// Master Dashboard Integration
const masterPath = path.join(frontendRoot, 'components', 'behavioral', 'Phase28MasterDashboard.tsx');
assert(fs.existsSync(masterPath), 'Phase28MasterDashboard.tsx exists');
const masterContent = fs.readFileSync(masterPath, 'utf8');
assert(masterContent.includes("import ExecutiveNarrativeHome from './ExecutiveNarrativeHome';"), 'Master dashboard imports ExecutiveNarrativeHome');
assert(masterContent.includes("<ExecutiveNarrativeHome />"), 'Master dashboard embeds ExecutiveNarrativeHome');
assert(masterContent.includes("'executive-narrative'"), 'Master dashboard includes executive-narrative subtab');

console.log('\n========================================================================');
console.log(`  Phase 28 Milestone 2A Verification Completed: ${passedTests} Passed, ${failedTests} Failed`);
console.log('========================================================================\n');

if (failedTests > 0) {
  process.exit(1);
}
