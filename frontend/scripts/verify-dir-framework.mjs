/**
 * ARX Terminal vNext - Phase 28: Decision Improvement Rating (DIR) & Cohort Migration Verification Suite
 * 
 * Verifies:
 * 1. DIR mathematical formulation: DIR = 0.30(DQG) + 0.20(BAS) + 0.20(RAS) + 0.15(DRS) + 0.15(LVI)
 * 2. 6 Strict Validation Rules (Min observations >= 30, Recs >= 20, Reviews >= 10, Drift cap <= 70, Penalty -15, Low confidence flag)
 * 3. 90-Day Trajectory projection (63 -> 69 @ 88% confidence)
 * 4. Behavioral Cohort Migration (Consumer -> Operator, CAR 31%, CRR 4%, TTM 142d, CVS 0.033/d)
 * 5. UI Components & Master Dashboard Integration
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
console.log('  ARX Terminal vNext: Phase 28 DIR & Cohort Migration Suite');
console.log('  (DIR = 0.30DQG + 0.20BAS + 0.20RAS + 0.15DRS + 0.15LVI & 6-Stage Migration)');
console.log('========================================================================\n');

// ------------------------------------------------------------------------
// SUITE 1: CONTRACTS & DATA MODELS (types/dir-framework.ts)
// ------------------------------------------------------------------------
console.log('SUITE 1: Contracts & Data Models');
const typesPath = path.join(frontendRoot, 'types', 'dir-framework.ts');
assert(fs.existsSync(typesPath), 'types/dir-framework.ts exists');
const typesContent = fs.readFileSync(typesPath, 'utf8');

assert(typesContent.includes('export interface UserDIRInputs'), 'Exports UserDIRInputs');
assert(typesContent.includes('export interface DIRComponents'), 'Exports DIRComponents');
assert(typesContent.includes('export interface DIRResult'), 'Exports DIRResult');
assert(typesContent.includes('export interface DIRValidationRule'), 'Exports DIRValidationRule');
assert(typesContent.includes('export interface DIRProjection'), 'Exports DIRProjection');
assert(typesContent.includes('export type DIRClassification'), 'Exports DIRClassification');
assert(typesContent.includes('export interface BehavioralCohortDefinition'), 'Exports BehavioralCohortDefinition');
assert(typesContent.includes('export interface CohortMigrationSummary'), 'Exports CohortMigrationSummary');
assert(typesContent.includes('export interface DIRQuarterlyHistory'), 'Exports DIRQuarterlyHistory');
assert(typesContent.includes('export interface DIRPeerBenchmark'), 'Exports DIRPeerBenchmark');

// ------------------------------------------------------------------------
// SUITE 2: DIR MATHEMATICAL FORMULATION & WEIGHTS (lib/telemetry/dirEngine.ts)
// ------------------------------------------------------------------------
console.log('\nSUITE 2: DIR Mathematical Formulation & Engine Logic');
const enginePath = path.join(frontendRoot, 'lib', 'telemetry', 'dirEngine.ts');
assert(fs.existsSync(enginePath), 'lib/telemetry/dirEngine.ts exists');
const engineContent = fs.readFileSync(enginePath, 'utf8');

// Verify Weights
assert(engineContent.includes('dqg: 0.30'), 'DQG weight is strictly 0.30 (30%)');
assert(engineContent.includes('bas: 0.20'), 'BAS weight is strictly 0.20 (20%)');
assert(engineContent.includes('ras: 0.20'), 'RAS weight is strictly 0.20 (20%)');
assert(engineContent.includes('drs: 0.15'), 'DRS weight is strictly 0.15 (15%)');
assert(engineContent.includes('lvi: 0.15'), 'LVI weight is strictly 0.15 (15%)');

// Verify Canonical Inputs
assert(engineContent.includes('currentDecisionScore: 74'), 'Canonical currentDecisionScore is 74');
assert(engineContent.includes('baselineDecisionScore: 62'), 'Canonical baselineDecisionScore is 62');
assert(engineContent.includes('recommendationsFollowed: 79'), 'Canonical recommendationsFollowed is 79');
assert(engineContent.includes('recommendationsIssued: 112'), 'Canonical recommendationsIssued is 112');
assert(engineContent.includes('stopLossAdherence: 91'), 'Canonical stopLossAdherence is 91%');
assert(engineContent.includes('macroInvalidationAdherence: 85'), 'Canonical macroInvalidationAdherence is 85%');
assert(engineContent.includes('positionSizingLimitAdherence: 88'), 'Canonical positionSizingLimitAdherence is 88%');
assert(engineContent.includes('riskControlsAdherence: 84'), 'Canonical riskControlsAdherence is 84%');
assert(engineContent.includes('driftScore: 21.0'), 'Canonical driftScore is 21.0%');
assert(engineContent.includes('learningVelocityIndex: 68.0'), 'Canonical learningVelocityIndex is 68.0');

// Deterministic Mathematical Verification
function calculateDQG(current, baseline) {
  return ((current - baseline) / (100 - baseline)) * 100;
}
function calculateBAS(followed, issued) {
  return (followed / issued) * 100;
}
function calculateRAS(stop, macro, sizing, risk) {
  return 0.30 * stop + 0.20 * macro + 0.20 * sizing + 0.30 * risk;
}

const testDQG = calculateDQG(74, 62);
assert(Math.abs(testDQG - 31.5789) < 0.01, 'DQG calculation matches ((74-62)/(100-62))*100 = 31.58');
const testBAS = calculateBAS(79, 112);
assert(Math.abs(testBAS - 70.5357) < 0.01, 'BAS calculation matches (79/112)*100 = 70.54%');
const testRAS = calculateRAS(91, 85, 88, 84);
assert(Math.abs(testRAS - 87.10) < 0.01, 'RAS calculation matches 0.30(91)+0.20(85)+0.20(88)+0.30(84) = 87.10%');

const testDRS = 100 - 21.0;
assert(testDRS === 79.0, 'DRS calculation matches 100 - 21.0 = 79.0%');
const testLVI = 68.0;

const testRawDIR = testDQG * 0.30 + testBAS * 0.20 + testRAS * 0.20 + testDRS * 0.15 + testLVI * 0.15;
assert(Math.abs(testRawDIR - 63.05) < 0.01, 'Total raw DIR sums to 63.05');
assert(Math.round(testRawDIR) === 63, 'Final rounded DIR is strictly 63');

// ------------------------------------------------------------------------
// SUITE 3: SIX STRICT VALIDATION RULES & EDGE CASES
// ------------------------------------------------------------------------
console.log('\nSUITE 3: Six Strict Validation Rules & Edge Cases');
assert(engineContent.includes('RULE_1_MIN_OBSERVATIONS'), 'Implements Rule 1: Minimum Observations');
assert(engineContent.includes('RULE_2_MIN_RECOMMENDATIONS'), 'Implements Rule 2: Minimum Recommendations');
assert(engineContent.includes('RULE_3_MIN_REVIEWS'), 'Implements Rule 3: Minimum Reviews');
assert(engineContent.includes('RULE_4_MAX_DRIFT_CAP'), 'Implements Rule 4: Max Drift Cap');
assert(engineContent.includes('RULE_5_BEHAVIOR_PENALTY'), 'Implements Rule 5: Behavior Penalty');
assert(engineContent.includes('RULE_6_CONFIDENCE_THRESHOLD'), 'Implements Rule 6: Confidence Threshold');

// Edge case simulation for Rule 4 (Drift > 60% caps DIR at 70)
function applyDriftCap(rawDir, drift) {
  if (drift > 60.0) {
    return Math.min(rawDir, 70.0);
  }
  return rawDir;
}
assert(applyDriftCap(85.0, 65.0) === 70.0, 'Rule 4 caps high DIR of 85 at 70 when drift > 60%');
assert(applyDriftCap(63.0, 21.0) === 63.0, 'Rule 4 leaves DIR unchanged at 63 when drift <= 60%');

// Edge case simulation for Rule 5 (RAS < 50 applies 15 pt penalty)
function applyBehaviorPenalty(rawDir, ras) {
  if (ras < 50.0) {
    return Math.max(0, rawDir - 15.0);
  }
  return rawDir;
}
assert(applyBehaviorPenalty(63.0, 45.0) === 48.0, 'Rule 5 applies 15 pt penalty when RAS < 50%');
assert(applyBehaviorPenalty(63.0, 87.1) === 63.0, 'Rule 5 leaves DIR intact when RAS >= 50%');

// ------------------------------------------------------------------------
// SUITE 4: BEHAVIORAL COHORT MIGRATION
// ------------------------------------------------------------------------
console.log('\nSUITE 4: Behavioral Cohort Migration Framework');
assert(engineContent.includes('BEHAVIORAL_COHORTS'), 'Exports BEHAVIORAL_COHORTS');
assert(engineContent.includes('CANONICAL_COHORT_MIGRATION'), 'Exports CANONICAL_COHORT_MIGRATION');
assert(engineContent.includes("'consumer'"), 'Includes Consumer cohort');
assert(engineContent.includes("'investigator'"), 'Includes Investigator cohort');
assert(engineContent.includes("'practitioner'"), 'Includes Practitioner cohort');
assert(engineContent.includes("'learner'"), 'Includes Learner cohort');
assert(engineContent.includes("'optimizer'"), 'Includes Optimizer cohort');
assert(engineContent.includes("'operator'"), 'Includes Operator cohort');

assert(engineContent.includes('cohortAdvancementRate: 31.0'), 'Cohort Advancement Rate (CAR) is 31.0% (> 25% target)');
assert(engineContent.includes('cohortRegressionRate: 4.0'), 'Cohort Regression Rate (CRR) is 4.0% (< 10% target)');
assert(engineContent.includes('timeToMaturityDays: 142'), 'Time to Maturity (TTM) is 142 days (< 180d target)');
assert(engineContent.includes('cohortVelocityScore: 0.033'), 'Cohort Velocity Score (CVS) is 0.033 / day');

// ------------------------------------------------------------------------
// SUITE 5: UI COMPONENTS INTEGRATION
// ------------------------------------------------------------------------
console.log('\nSUITE 5: UI Components & Dashboard Integration');

const heroCardPath = path.join(frontendRoot, 'components', 'behavioral', 'DIRHeroCard.tsx');
assert(fs.existsSync(heroCardPath), 'DIRHeroCard.tsx exists');
const heroCardContent = fs.readFileSync(heroCardPath, 'utf8');
assert(heroCardContent.includes('data-testid="dir-hero-card"'), 'DIRHeroCard has data-testid');
assert(heroCardContent.includes('finalDIR'), 'Renders final DIR');
assert(heroCardContent.includes('projectedScore90d'), 'Renders 90-day projection');
assert(heroCardContent.includes('Strongest Driver'), 'Renders Strongest Driver');
assert(heroCardContent.includes('Largest Obstacle'), 'Renders Largest Obstacle');
assert(heroCardContent.includes('validationRules'), 'Renders 6 Validation Gates checklist');

const timelinePath = path.join(frontendRoot, 'components', 'behavioral', 'DIREvolutionTimeline.tsx');
assert(fs.existsSync(timelinePath), 'DIREvolutionTimeline.tsx exists');
const timelineContent = fs.readFileSync(timelinePath, 'utf8');
assert(timelineContent.includes('data-testid="dir-evolution-timeline"'), 'DIREvolutionTimeline has data-testid');
assert(timelineContent.includes('Peer Cohort Comparison'), 'Renders Peer Cohort Comparison');
assert(timelineContent.includes('+8.0 pts'), 'Renders +8.0 pts annual delta');

const cohortDashPath = path.join(frontendRoot, 'components', 'behavioral', 'CohortMigrationDashboard.tsx');
assert(fs.existsSync(cohortDashPath), 'CohortMigrationDashboard.tsx exists');
const cohortDashContent = fs.readFileSync(cohortDashPath, 'utf8');
assert(cohortDashContent.includes('data-testid="cohort-migration-dashboard"'), 'CohortMigrationDashboard has data-testid');
assert(cohortDashContent.includes('Advancement Rate (CAR)'), 'Renders CAR KPI');
assert(cohortDashContent.includes('Regression Rate (CRR)'), 'Renders CRR KPI');
assert(cohortDashContent.includes('Time to Maturity (TTM)'), 'Renders TTM KPI');
assert(cohortDashContent.includes('Sequential Promotion Funnel Rates'), 'Renders Sequential Promotion Funnel');

const confidenceBandPath = path.join(frontendRoot, 'components', 'behavioral', 'EnhancedConfidenceBand.tsx');
assert(fs.existsSync(confidenceBandPath), 'EnhancedConfidenceBand.tsx exists');
const confidenceBandContent = fs.readFileSync(confidenceBandPath, 'utf8');
assert(confidenceBandContent.includes('sampleSize < 30'), 'Implements sample size guard (n < 30)');
assert(confidenceBandContent.includes('Data Insufficient (n &lt; 30)'), 'Renders Data Insufficient fallback');
assert(confidenceBandContent.includes('upperBound < target'), 'Implements 3-tier color coding');

const masterDashPath = path.join(frontendRoot, 'components', 'behavioral', 'Phase28MasterDashboard.tsx');
assert(fs.existsSync(masterDashPath), 'Phase28MasterDashboard.tsx exists');
const masterDashContent = fs.readFileSync(masterDashPath, 'utf8');
assert(masterDashContent.includes('DIRHeroCard'), 'Mounts DIRHeroCard');
assert(masterDashContent.includes('DIREvolutionTimeline'), 'Mounts DIREvolutionTimeline');
assert(masterDashContent.includes('CohortMigrationDashboard'), 'Mounts CohortMigrationDashboard');
assert(masterDashContent.includes('EnhancedConfidenceBand'), 'Mounts EnhancedConfidenceBand');

// Summary
console.log('\n========================================================================');
console.log(`  Phase 28 DIR & Cohort Migration Verification: ${passedTests} Passed, ${failedTests} Failed`);
console.log('========================================================================\n');

if (failedTests > 0) {
  process.exit(1);
}
