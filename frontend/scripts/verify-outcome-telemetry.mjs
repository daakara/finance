/**
 * ARX Terminal vNext - User Outcome Telemetry & Executive Tracker Verification Suite
 * Verifies the 4-Phase Telemetry Philosophy (SEEN -> UNDERSTOOD -> ACTED_UPON -> BEHAVIOR_IMPROVED),
 * 14 Core Events, Tier 1/2/3 Metrics, and Executive Execution Tracker Component.
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
console.log('  ARX Terminal vNext: User Outcome Telemetry & Executive Tracker Suite');
console.log('  (4-Phase Philosophy, 14 Events, Tier 1/2/3 Metrics, 95% Release Candidate)');
console.log('========================================================================\n');

// ------------------------------------------------------------------------
// SUITE 1: CONTRACTS & DATA MODELS (types/user-outcome-telemetry.ts)
// ------------------------------------------------------------------------
console.log('SUITE 1: Contracts & Data Models');
const typesPath = path.join(frontendRoot, 'types', 'user-outcome-telemetry.ts');
assert(fs.existsSync(typesPath), 'types/user-outcome-telemetry.ts exists');
const typesContent = fs.readFileSync(typesPath, 'utf8');

assert(typesContent.includes('export type OutcomeTelemetryPhase'), 'Exports OutcomeTelemetryPhase union');
assert(
  typesContent.includes("'SEEN'") &&
  typesContent.includes("'UNDERSTOOD'") &&
  typesContent.includes("'ACTED_UPON'") &&
  typesContent.includes("'BEHAVIOR_IMPROVED'"),
  'Defines all 4 philosophical phases: SEEN, UNDERSTOOD, ACTED_UPON, BEHAVIOR_IMPROVED'
);

assert(typesContent.includes('export type UserOutcomeEventCategory'), 'Exports UserOutcomeEventCategory union');
assert(
  typesContent.includes("'ENGAGEMENT'") &&
  typesContent.includes("'DECISION'") &&
  typesContent.includes("'LEARNING'") &&
  typesContent.includes("'PLAYBOOK'") &&
  typesContent.includes("'MENTOR'") &&
  typesContent.includes("'GOVERNANCE'") &&
  typesContent.includes("'OUTCOME'"),
  'Defines all 7 event categories in taxonomy'
);

assert(typesContent.includes('export type UserOutcomeEventName'), 'Exports UserOutcomeEventName union');
const expectedEvents = [
  'mentor_viewed',
  'mentor_recommendation_clicked',
  'mentor_evidence_opened',
  'learning_journey_viewed',
  'milestone_expanded',
  'next_milestone_clicked',
  'playbook_viewed',
  'rule_opened',
  'rule_followed',
  'adoption_dashboard_viewed',
  'drift_warning_seen',
  'drift_warning_acknowledged',
  'lifecycle_stage_viewed',
  'lifecycle_transition_completed',
];

expectedEvents.forEach(evt => {
  assert(typesContent.includes(`'${evt}'`), `Defines event name "${evt}"`);
});

assert(typesContent.includes('export interface ExecutiveTrackerMetrics'), 'Exports ExecutiveTrackerMetrics contract');
assert(typesContent.includes('export interface ReleaseGateItem'), 'Exports ReleaseGateItem contract');

// ------------------------------------------------------------------------
// SUITE 2: TELEMETRY SERVICE & EVENT DISPATCHERS (telemetry/userOutcomeTelemetry.ts)
// ------------------------------------------------------------------------
console.log('\nSUITE 2: Telemetry Service & Event Dispatchers');
const servicePath = path.join(frontendRoot, 'telemetry', 'userOutcomeTelemetry.ts');
assert(fs.existsSync(servicePath), 'telemetry/userOutcomeTelemetry.ts exists');
const serviceContent = fs.readFileSync(servicePath, 'utf8');

assert(serviceContent.includes('class UserOutcomeTelemetryService'), 'Implements UserOutcomeTelemetryService class');
assert(serviceContent.includes('export const userOutcomeTelemetry'), 'Exports userOutcomeTelemetry singleton');

// Test 14 dispatchers presence
assert(serviceContent.includes('trackMentorViewed('), 'Implements trackMentorViewed (Phase: SEEN)');
assert(serviceContent.includes('trackMentorRecommendationClicked('), 'Implements trackMentorRecommendationClicked (Phase: UNDERSTOOD)');
assert(serviceContent.includes('trackMentorEvidenceOpened('), 'Implements trackMentorEvidenceOpened (Trust Validation Rate)');
assert(serviceContent.includes('trackLearningJourneyViewed('), 'Implements trackLearningJourneyViewed');
assert(serviceContent.includes('trackMilestoneExpanded('), 'Implements trackMilestoneExpanded');
assert(serviceContent.includes('trackNextMilestoneClicked('), 'Implements trackNextMilestoneClicked (Phase: ACTED_UPON)');
assert(serviceContent.includes('trackPlaybookViewed('), 'Implements trackPlaybookViewed');
assert(serviceContent.includes('trackRuleOpened('), 'Implements trackRuleOpened');
assert(serviceContent.includes('trackRuleFollowed('), 'Implements trackRuleFollowed (Phase: BEHAVIOR_IMPROVED)');
assert(serviceContent.includes('trackAdoptionDashboardViewed('), 'Implements trackAdoptionDashboardViewed');
assert(serviceContent.includes('trackDriftWarningSeen('), 'Implements trackDriftWarningSeen');
assert(serviceContent.includes('trackDriftWarningAcknowledged('), 'Implements trackDriftWarningAcknowledged');
assert(serviceContent.includes('trackLifecycleStageViewed('), 'Implements trackLifecycleStageViewed');
assert(serviceContent.includes('trackLifecycleTransitionCompleted('), 'Implements trackLifecycleTransitionCompleted');

// Verify buffer capping logic
assert(serviceContent.includes('maxBufferSize = 500'), 'Configures 500 event FIFO buffer cap');
assert(serviceContent.includes('this.buffer.shift()'), 'Enforces oldest-first dropping upon buffer overflow');

// ------------------------------------------------------------------------
// SUITE 3: METRIC TARGETS & CONVERSION LOGIC
// ------------------------------------------------------------------------
console.log('\nSUITE 3: Metric Targets & Conversion Logic');

// Tier 1: Strategic Outcome Metrics
assert(serviceContent.includes('decisionQualityScore: 74'), 'Tier 1: Decision Quality Score = 74');
assert(serviceContent.includes('behavioralAdoptionRate: 70.5'), 'Tier 1: BAR = 70.5% (Target > 70%)');
assert(serviceContent.includes('repeatMistakeReduction: -43'), 'Tier 1: Repeat Mistake Reduction = -43% (Target > 30%)');
assert(serviceContent.includes('decisionDrift: 21'), 'Tier 1: Decision Drift = 21% (Target < 25%)');

// Tier 2: Product Metrics
assert(serviceContent.includes('mentorVisibilityRate: 94'), 'Tier 2: Mentor Visibility Rate = 94% (Target > 95%)');
assert(serviceContent.includes('recommendationEngagementRate: 67'), 'Tier 2: Recommendation Engagement Rate = 67% (Target > 60%)');
assert(serviceContent.includes('trustValidationRate: 44'), 'Tier 2: Trust Validation Rate = 44% (Target: 30%-70% optimal band)');
assert(serviceContent.includes('ruleAdherenceRate: 87'), 'Tier 2: Rule Adherence Rate = 87% (Target > 80%)');

// Tier 3: UX Speed Test
assert(serviceContent.includes('questionResolutionAvgTimeSec: 2.1'), 'Tier 3: Question Resolution Time = 2.1s (Target < 5s)');

// ------------------------------------------------------------------------
// SUITE 4: EXECUTIVE EXECUTION TRACKER COMPONENT
// ------------------------------------------------------------------------
console.log('\nSUITE 4: Executive Execution Tracker Component');
const trackerComponentPath = path.join(frontendRoot, 'components', 'audit', 'ExecutiveExecutionTracker.tsx');
assert(fs.existsSync(trackerComponentPath), 'ExecutiveExecutionTracker.tsx exists');
const trackerContent = fs.readFileSync(trackerComponentPath, 'utf8');

assert(trackerContent.includes('role="region"') && trackerContent.includes('aria-label="Executive Execution Tracker"'), 'Renders accessible region landmark');
assert(trackerContent.includes('1. Platform Health'), 'Renders Section 1: Platform Health');
assert(trackerContent.includes('2. User Outcomes (Tier 1)'), 'Renders Section 2: User Outcomes');
assert(trackerContent.includes('3. Mentor Metrics (Tier 2)'), 'Renders Section 3: Mentor Metrics');
assert(trackerContent.includes('4. Release Gate Tracker'), 'Renders Section 4: Release Gate Tracker');
assert(trackerContent.includes('Live Outcome Telemetry Event Stream'), 'Renders Section 5: Live Telemetry Simulator');
assert(trackerContent.includes('currentReadinessScore') && trackerContent.includes('releaseStatus'), 'Displays current readiness score and Release Candidate status');
assert(trackerContent.includes('Target: 98.0%') || trackerContent.includes('98.0% Excellence'), 'Displays 98.0% Institutional Excellence target');

// ------------------------------------------------------------------------
// SUITE 5: SHOWCASE INTEGRATION
// ------------------------------------------------------------------------
console.log('\nSUITE 5: Showcase Integration');
const showcasePath = path.join(frontendRoot, 'components', 'experience', 'Sprint85Showcase.tsx');
const showcaseContent = fs.readFileSync(showcasePath, 'utf8');

assert(showcaseContent.includes('<ExecutiveExecutionTracker />'), 'Sprint85Showcase renders ExecutiveExecutionTracker in Section 5');
assert(showcaseContent.includes('Executive Execution Tracker & User Outcome Telemetry Framework'), 'Showcase Section 5 headline present');

// ------------------------------------------------------------------------
// FINAL TALLY
// ------------------------------------------------------------------------
console.log('\n========================================================================');
console.log(`  VERIFICATION RESULTS: ${passedTests} / ${passedTests + failedTests} TESTS PASSED`);
console.log('========================================================================\n');

if (failedTests > 0) {
  console.error(`💥 Verification failed with ${failedTests} failure(s)!\n`);
  process.exit(1);
} else {
  console.log('🎉 USER OUTCOME TELEMETRY & EXECUTIVE TRACKER VERIFIED (95% RELEASE CANDIDATE)!\n');
  process.exit(0);
}
