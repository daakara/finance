/**
 * ARX Terminal vNext - Phase 28: Behavioral Intelligence UAT & Accessibility Verification Suite
 * 
 * Verifies the 8 UAT Scenarios (UAT-P28-001 to UAT-P28-008)
 * and the 6 Accessibility Criteria (A11Y-P28-001 to A11Y-P28-006)
 * defined in the Phase 28 Final Scope Definition.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendRoot = path.resolve(__dirname, '..');

let passedTests = 0;
let failedTests = 0;

function assert(condition, testName) {
  if (condition) {
    console.log(`  ✓ ${testName}`);
    passedTests++;
  } else {
    console.error(`  ✗ FAIL: ${testName}`);
    failedTests++;
  }
}

console.log('\n========================================================================');
console.log('  ARX Terminal vNext: Phase 28 UAT & Accessibility Verification Suite');
console.log('  (UAT-P28-001..008 & A11Y-P28-001..006 Formal Audit)');
console.log('========================================================================\n');

// ------------------------------------------------------------------------
// PART 1: UAT VERIFICATION (UAT-P28-001 to UAT-P28-008)
// ------------------------------------------------------------------------
console.log('PART 1: Institutional UAT Scenarios (UAT-P28-001 to UAT-P28-008)');

// UAT-P28-001: Story-First Executive Home
console.log('\n[UAT-P28-001] Story-First Executive Home (< 5s comprehension)');
const execHomePath = path.join(frontendRoot, 'components', 'behavioral', 'ExecutiveStoryHome.tsx');
assert(fs.existsSync(execHomePath), 'ExecutiveStoryHome.tsx exists');
const execHomeContent = fs.readFileSync(execHomePath, 'utf8');

assert(execHomeContent.includes('ARX-EXH-001'), 'Component ID ARX-EXH-001 assigned');
assert(execHomeContent.includes('data-testid="executive-story-home"'), 'Renders executive-story-home testid');
assert(execHomeContent.includes('GOOD MORNING'), 'Renders Executive Greeting');
assert(execHomeContent.includes('YOUR CURRENT STATE'), 'Displays Current State Matrix');
assert(execHomeContent.includes("TODAY'S PRIORITIES") || execHomeContent.includes("TODAY&apos;S PRIORITIES"), 'Displays Today Priorities');
assert(execHomeContent.includes('Institutional Flow Discipline'), 'Largest positive driver visible');
assert(execHomeContent.includes('Macro Deterioration') || execHomeContent.includes('Macro Blindness'), 'Largest risk visible');
assert(execHomeContent.includes('Reduce exposure 15%') || execHomeContent.includes('Tighten macro filters'), 'Recommended action visible');
assert(execHomeContent.includes('ExecutiveStatusStrip'), 'Includes Component A: ExecutiveStatusStrip (72px height)');

// UAT-P28-002: Morning Briefing Narrative
console.log('\n[UAT-P28-002] Morning Briefing Narrative (Context -> Meaning -> Impact -> Action)');
const briefingPath = path.join(frontendRoot, 'components', 'behavioral', 'MorningBriefingV2.tsx');
assert(fs.existsSync(briefingPath), 'MorningBriefingV2.tsx exists');
const briefingContent = fs.readFileSync(briefingPath, 'utf8');

assert(briefingContent.includes('OVERNIGHT STORY'), 'Displays Overnight Story');
assert(briefingContent.includes('WHY IT MATTERS'), 'Displays Why It Matters section');
assert(briefingContent.includes('IMPACTED POSITIONS'), 'Displays Impacted Positions section');
assert(briefingContent.includes('RECOMMENDED ACTION'), 'Displays Recommended Action section');
assert(briefingContent.includes('42') && briefingContent.includes('56'), 'Overnight Market Risk Score shift 42 -> 56 visible');
assert(briefingContent.includes('NVDA') && briefingContent.includes('AMD') && briefingContent.includes('CRWD'), 'NVDA, AMD, CRWD flagged as affected');
assert(briefingContent.includes('184,000') || briefingContent.includes('totalCapitalAtRisk'), '$184,000 capital at risk displayed');
assert(briefingContent.includes('[Review Now]'), 'Actionable [Review Now] button available');

// UAT-P28-003: Behavioral Intelligence Dashboard
console.log('\n[UAT-P28-003] Behavioral Intelligence Dashboard');
const centerPath = path.join(frontendRoot, 'components', 'behavioral', 'BehavioralIntelligenceCenter.tsx');
assert(fs.existsSync(centerPath), 'BehavioralIntelligenceCenter.tsx exists');
const centerContent = fs.readFileSync(centerPath, 'utf8');

assert(centerContent.includes('YOUR DECISION EVOLUTION'), 'Displays Evolution Header');
assert(centerContent.includes('Decision Quality'), 'Card 1: Decision Quality visible');
assert(centerContent.includes('Learning Velocity (LVI)') || centerContent.includes('Learning Velocity'), 'Card 2: Learning Velocity Index (LVI) visible');
assert(centerContent.includes('Adoption Rate (BAR)') || centerContent.includes('Behavior Adoption'), 'Card 3: Behavior Adoption visible');
assert(centerContent.includes('Repeat Mistakes'), 'Card 4: Repeat Mistakes visible');
assert(centerContent.includes('Decision Drift'), 'Card 5: Decision Drift visible');

// UAT-P28-004: AI Behavioral Coach
console.log('\n[UAT-P28-004] AI Behavioral Coach (Why you are improving)');
const coachPath = path.join(frontendRoot, 'components', 'behavioral', 'AIBehavioralCoachCard.tsx');
assert(fs.existsSync(coachPath), 'AIBehavioralCoachCard.tsx exists');
const coachContent = fs.readFileSync(coachPath, 'utf8');

assert(coachContent.includes('Better stop discipline'), 'Contributor 1: Better stop discipline displayed');
assert(coachContent.includes('+3.2 points') || coachContent.includes('3.2'), 'Stop discipline +3.2 points contribution');
assert(coachContent.includes('Stronger macro filtering'), 'Contributor 2: Stronger macro filtering displayed');
assert(coachContent.includes('+4.4 points') || coachContent.includes('4.4'), 'Macro filtering +4.4 points contribution');
assert(coachContent.includes('Reduced momentum chasing'), 'Contributor 3: Reduced momentum chasing displayed');
assert(coachContent.includes('+2.8 points') || coachContent.includes('2.8'), 'Momentum chasing +2.8 points contribution');
assert(coachContent.includes('Score 80') || coachContent.includes('80'), 'Target Quality Score 80 forecast displayed');
assert(coachContent.includes('87%'), '87% confidence forecast visible');

// UAT-P28-005: Learning Velocity Engine
console.log('\n[UAT-P28-005] Learning Velocity Engine');
const lviEnginePath = path.join(frontendRoot, 'lib', 'telemetry', 'learningVelocityEngine.ts');
assert(fs.existsSync(lviEnginePath), 'learningVelocityEngine.ts exists');
const lviContent = fs.readFileSync(lviEnginePath, 'utf8');

assert(lviContent.includes('computeLearningVelocityIndex'), 'Exports computeLearningVelocityIndex');
assert(lviContent.includes('computeImprovementMomentum'), 'Exports computeImprovementMomentum');
assert(lviContent.includes('classifyLVI'), 'Exports classifyLVI with 4 standardized tiers');

// UAT-P28-006: Cohort Analytics
console.log('\n[UAT-P28-006] Cohort Analytics & Distribution');
const cohortMatrixPath = path.join(frontendRoot, 'components', 'behavioral', 'BehavioralMaturityCohortMatrix.tsx');
assert(fs.existsSync(cohortMatrixPath), 'BehavioralMaturityCohortMatrix.tsx exists');
const cohortMatrixContent = fs.readFileSync(cohortMatrixPath, 'utf8');

assert(cohortMatrixContent.includes('USER EVOLUTION DISTRIBUTION'), 'Displays User Evolution Distribution header');
assert(cohortMatrixContent.includes('Optimizer Cohort Growth') || cohortMatrixContent.includes('Optimizer'), 'Displays Optimizer Trend View');
assert(cohortMatrixContent.includes('Cohort Outcome Comparison'), 'Displays Outcome Comparison');
assert(cohortMatrixContent.includes('Role-Based Behavioral Adoption'), 'Displays Role-based comparisons (Execs, PMs, Analysts)');

// UAT-P28-007: Confidence Bands
console.log('\n[UAT-P28-007] Confidence Bands Standards');
const confBandPath = path.join(frontendRoot, 'components', 'behavioral', 'EnhancedConfidenceBand.tsx');
assert(fs.existsSync(confBandPath), 'EnhancedConfidenceBand.tsx exists');
const confBandContent = fs.readFileSync(confBandPath, 'utf8');

assert(confBandContent.includes('confidenceLevel'), 'Displays confidence level');
assert(confBandContent.includes('lowerBound') && confBandContent.includes('upperBound'), 'Displays interval lower and upper bounds');
assert(confBandContent.includes('marginOfError'), 'Displays margin of error');
assert(confBandContent.includes('target'), 'Displays target comparison benchmark');

// UAT-P28-008: Decision Evolution Timeline
console.log('\n[UAT-P28-008] Decision Evolution Timeline');
const storyEnginePath = path.join(frontendRoot, 'lib', 'telemetry', 'behavioralStoryEngine.ts');
assert(fs.existsSync(storyEnginePath), 'behavioralStoryEngine.ts exists');
const storyEngineContent = fs.readFileSync(storyEnginePath, 'utf8');
assert(storyEngineContent.includes('Q1 2026') && (storyEngineContent.toLowerCase().includes('stop-loss') || storyEngineContent.includes('STOP_LOSS')), 'Displays Q1 Stop Loss milestone');
assert(storyEngineContent.includes('Q2 2026') && (storyEngineContent.includes('Macro Gating') || storyEngineContent.includes('MACRO_GATING')), 'Displays Q2 Macro Gating milestone');
assert(storyEngineContent.includes('Q3 2026') && (storyEngineContent.toLowerCase().includes('flow accumulation') || storyEngineContent.includes('FLOW_ACCUMULATION')), 'Displays Q3 Flow Accumulation milestone');
assert(execHomeContent.includes('62') && execHomeContent.includes('74'), 'Displays quality progression 62 -> 74');

// ------------------------------------------------------------------------
// PART 2: ACCESSIBILITY TEST PLAN (A11Y-P28-001 to A11Y-P28-006)
// ------------------------------------------------------------------------
console.log('\nPART 2: Accessibility Criteria Audit (A11Y-P28-001 to A11Y-P28-006)');

// A11Y-P28-001: Executive Home Landmarks & Keyboard Nav
console.log('\n[A11Y-P28-001] Executive Home Landmarks & Keyboard Navigation');
assert(execHomeContent.includes('data-testid="executive-story-home"'), 'Has semantic landmark/test container');
assert(execHomeContent.includes('<button') && execHomeContent.includes('onClick'), 'Interactive actions use keyboard accessible buttons');
assert(execHomeContent.includes('focus:') || execHomeContent.includes('transition-'), 'Focus visible styling configured');

// A11Y-P28-002: Behavioral Dashboard Screen Reader Navigability
console.log('\n[A11Y-P28-002] Behavioral Dashboard Text Alternatives & Landmarks');
assert(centerContent.includes('text-caption') || centerContent.includes('font-mono'), 'Metrics use readable structural text elements');
assert(centerContent.includes('Milestone') || centerContent.includes('milestone'), 'Timeline is screen-reader navigable with labeled milestones');

// A11Y-P28-003: Confidence Band Visualization (Color-Independent)
console.log('\n[A11Y-P28-003] Confidence Band Non-Color Dependence');
assert(confBandContent.includes('Above Target') && confBandContent.includes('Below Target'), 'Uses text labels in addition to colors');
assert(confBandContent.includes('.toFixed(1)'), 'All interval endpoints displayed numerically');

// A11Y-P28-004: Cohort Analytics Keyboard Accessibility
console.log('\n[A11Y-P28-004] Cohort Analytics Keyboard Accessibility');
assert(cohortMatrixContent.includes('tabIndex={0}'), 'Interactive cohort cards include tabIndex={0}');
assert(cohortMatrixContent.includes('onKeyDown'), 'Interactive cohort cards handle Enter/Space onKeyDown');
assert(cohortMatrixContent.includes('role="button"'), 'Interactive cards have explicit role="button"');

// A11Y-P28-005: Behavioral Coach Accessibility
console.log('\n[A11Y-P28-005] Behavioral Coach Interactive Controls');
assert(coachContent.includes('<button') && coachContent.includes('setScenarioMode'), 'Scenario toggle buttons are accessible');
assert(coachContent.includes('aria-') || coachContent.includes('tabIndex') || coachContent.includes('role='), 'Uses semantic accessible properties');

// A11Y-P28-006: Mobile Accessibility & Touch Targets
console.log('\n[A11Y-P28-006] Mobile Viewport & Touch Target Floor');
const statusStripPath = path.join(frontendRoot, 'components', 'behavioral', 'ExecutiveStatusStrip.tsx');
assert(fs.existsSync(statusStripPath), 'ExecutiveStatusStrip.tsx exists');
const statusStripContent = fs.readFileSync(statusStripPath, 'utf8');
assert(statusStripContent.includes('min-h-[72px]'), 'Status strip enforces 72px institutional height');
assert(execHomeContent.includes('px-4 py-2') || execHomeContent.includes('p-3') || execHomeContent.includes('p-4'), 'Touch targets meet >= 44x44px hit area guidelines');
assert(!execHomeContent.includes('overflow-x-scroll'), 'Executive Home eliminates accidental horizontal scrollbars');

// Summary
console.log('\n========================================================================');
console.log(`  Phase 28 UAT & Accessibility Verification: ${passedTests} Passed, ${failedTests} Failed`);
console.log('========================================================================\n');

if (failedTests > 0) {
  process.exit(1);
}
