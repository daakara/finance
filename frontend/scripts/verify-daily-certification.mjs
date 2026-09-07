/**
 * ARX Terminal vNext - Phase 28 Daily Production Certification Verification Suite
 * Daily Automated Production Protection Framework & Outcome Intelligence Attribution
 * 
 * Verifies:
 * - Suite 1: Contracts & Data Models (DailyAuditItem, DIRatio, ValueAttribution, Cohorts)
 * - Suite 2: Daily 6-Audit Integrity (all 6 pass, 99.8% health score, failure behavior)
 * - Suite 3: North Star Metric: Decision Impact Ratio (DIRatio +24.0%, 95% CI, N=4,218)
 * - Suite 4: Outcome Intelligence Value Attribution ($2.4M preserved, +3.8% excess return, 74 mistakes, 1,247 recommendations)
 * - Suite 5: Behavioral Maturity Cohorts Distribution (100% sum, 12% to 20% optimizer growth)
 * - Suite 6: UI Component Integrity & Master Dashboard Mounting
 * - Suite 7: Governance Documentation Integrity (DAILY_PRODUCTION_CERTIFICATION_CHECKLIST.md)
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
console.log('  ARX Terminal vNext: Phase 28 Daily Production Certification Verification');
console.log('  (Production Protection Framework & Outcome Intelligence Attribution)');
console.log('========================================================================\n');

// Dynamic Imports of Engines & Canonical Fixtures
const {
  CANONICAL_DAILY_AUDITS,
  CANONICAL_DIR_RATIO,
  CANONICAL_VALUE_ATTRIBUTION,
  CANONICAL_BEHAVIORAL_MATURITY_COHORTS,
  getCanonicalDailyCertification,
  evaluateProductionCertification,
} = await import('../lib/telemetry/dailyCertificationEngine.ts');

// -----------------------------------------------------------------------------
// Suite 1: Contracts & Data Models
// -----------------------------------------------------------------------------
console.log('--- Suite 1: Contracts & Data Models ---');
assert(Array.isArray(CANONICAL_DAILY_AUDITS), 'CANONICAL_DAILY_AUDITS is an array');
assertEqual(CANONICAL_DAILY_AUDITS.length, 6, 'CANONICAL_DAILY_AUDITS contains exactly 6 audits');

const expectedCategories = ['TELEMETRY', 'ATTRIBUTION', 'PERFORMANCE', 'PLAYBOOK', 'AI_CONFIDENCE', 'GOVERNANCE'];
expectedCategories.forEach((cat) => {
  const exists = CANONICAL_DAILY_AUDITS.some(a => a.category === cat);
  assert(exists, `Audit category '${cat}' is represented in canonical audits`);
});

CANONICAL_DAILY_AUDITS.forEach((audit, idx) => {
  assert(Boolean(audit.id && audit.id.startsWith('AUD-0')), `Audit ${idx + 1} has valid ID ${audit.id}`);
  assert(Boolean(audit.name && audit.name.length > 3), `Audit ${audit.id} has valid name`);
  assert(audit.status === 'PASS', `Audit ${audit.id} status is strictly PASS`);
  assert(Boolean(audit.target && audit.actual), `Audit ${audit.id} specifies both target and actual values`);
  assert(Boolean(audit.details && audit.details.length > 10), `Audit ${audit.id} includes detailed verification rationale`);
});

assert(typeof CANONICAL_DIR_RATIO === 'object', 'CANONICAL_DIR_RATIO object exists');
assert(typeof CANONICAL_VALUE_ATTRIBUTION === 'object', 'CANONICAL_VALUE_ATTRIBUTION object exists');
assert(typeof CANONICAL_BEHAVIORAL_MATURITY_COHORTS === 'object', 'CANONICAL_BEHAVIORAL_MATURITY_COHORTS object exists');

// -----------------------------------------------------------------------------
// Suite 2: Daily 6-Audit Integrity & Health Scoring
// -----------------------------------------------------------------------------
console.log('\n--- Suite 2: Daily 6-Audit Integrity & Health Scoring ---');
const evalResult = evaluateProductionCertification(CANONICAL_DAILY_AUDITS);
assert(evalResult.isCertified === true, 'All 6 audits passing yields isCertified === true');
assertEqual(evalResult.passedCount, 6, 'Passed audit count is exactly 6');
assertEqual(evalResult.totalCount, 6, 'Total audit count is exactly 6');
assertEqual(evalResult.healthScore, 99.8, 'Canonical production health score is strictly 99.8%');

// Test degraded behavior
const degradedAudits = [
  ...CANONICAL_DAILY_AUDITS.slice(0, 5),
  { ...CANONICAL_DAILY_AUDITS[5], status: 'FAIL' },
];
const degradedResult = evaluateProductionCertification(degradedAudits);
assert(degradedResult.isCertified === false, 'A single audit failure sets isCertified to false');
assertEqual(degradedResult.passedCount, 5, 'Degraded evaluation reports 5 passing audits');
assertEqual(degradedResult.healthScore, 83.3, 'Degraded evaluation scores 83.3% health');

const report = getCanonicalDailyCertification();
assertEqual(report.status, 'CERTIFIED', 'Canonical report status is CERTIFIED');
assertEqual(report.overallHealthScore, 99.8, 'Canonical report health score is 99.8%');
assert(report.releaseTrain.includes('Phase 28'), 'Canonical report specifies Phase 28 release train');
assert(report.certifiedAt.length >= 10, 'Canonical report includes ISO date stamp');

// -----------------------------------------------------------------------------
// Suite 3: North Star Metric: Decision Impact Ratio (DIRatio)
// -----------------------------------------------------------------------------
console.log('\n--- Suite 3: North Star Metric: Decision Impact Ratio (DIRatio) ---');
assertEqual(CANONICAL_DIR_RATIO.highAdoptionWinRate, 68.0, 'High adoption win rate is 68.0%');
assertEqual(CANONICAL_DIR_RATIO.lowAdoptionWinRate, 44.0, 'Low adoption win rate is 44.0%');
const calculatedDelta = CANONICAL_DIR_RATIO.highAdoptionWinRate - CANONICAL_DIR_RATIO.lowAdoptionWinRate;
assertEqual(CANONICAL_DIR_RATIO.dirRatio, 24.0, 'DIRatio outperformance delta is strictly +24.0%');
assertEqual(calculatedDelta, CANONICAL_DIR_RATIO.dirRatio, 'High - Low adoption equals dirRatio');
assertEqual(CANONICAL_DIR_RATIO.confidence, 95.0, 'Statistical confidence is 95.0% (p < 0.001)');
assertEqual(CANONICAL_DIR_RATIO.sampleSize, 4218, 'Empirical sample size is institutional N = 4,218');
assert(CANONICAL_DIR_RATIO.description.includes('+24.0%'), 'DIRatio description includes +24.0% impact');

// -----------------------------------------------------------------------------
// Suite 4: Outcome Intelligence Value Attribution (P28-400)
// -----------------------------------------------------------------------------
console.log('\n--- Suite 4: Outcome Intelligence Value Attribution ---');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.capitalPreservedFormatted, '$2.4M', 'Capital preserved is formatted as $2.4M');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.capitalPreservedDollars, 2400000, 'Capital preserved is exactly $2,400,000 USD');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.excessReturnPct, 3.8, 'Excess return generated is +3.8%');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.mistakesPrevented, 74, 'Total mistakes prevented is 74');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.recommendationsAdopted, 1247, 'Recommendations adopted count is 1,247');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.topDriverName, 'Institutional Flow Filter', 'Top driver is Institutional Flow Filter');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.topDriverContributionPct, 28.0, 'Top driver contribution is 28.0%');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.sources.stopDiscipline, '$1.1M', 'Stop discipline preserved $1.1M');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.sources.macroFilters, '$850K', 'Macro filters preserved $850K');
assertEqual(CANONICAL_VALUE_ATTRIBUTION.sources.riskReductions, '$450K', 'Risk reductions preserved $450K');

// -----------------------------------------------------------------------------
// Suite 5: Behavioral Maturity Cohort Distribution
// -----------------------------------------------------------------------------
console.log('\n--- Suite 5: Behavioral Maturity Cohorts Distribution ---');
const cohorts = CANONICAL_BEHAVIORAL_MATURITY_COHORTS;
const cohortSum = cohorts.nonAdoptersPct + cohorts.explorersPct + cohorts.practitionersPct + cohorts.learnersPct + cohorts.optimizersPct;
assertEqual(cohortSum, 100.0, 'Behavioral maturity cohorts sum exactly to 100.0%');
assertEqual(cohorts.nonAdoptersPct, 12.0, 'Non-adopters proportion is 12.0%');
assertEqual(cohorts.explorersPct, 24.0, 'Explorers proportion is 24.0%');
assertEqual(cohorts.practitionersPct, 31.0, 'Practitioners proportion is 31.0%');
assertEqual(cohorts.learnersPct, 21.0, 'Learners proportion is 21.0%');
assertEqual(cohorts.optimizersPct, 12.0, 'Optimizers proportion is 12.0%');
assertEqual(cohorts.optimizerTargetPct, 20.0, 'Optimizer expansion target is 20.0%');
assert(cohorts.practitionersPct + cohorts.learnersPct + cohorts.optimizersPct >= 60.0, 'Active adopters exceed 60% threshold');

// -----------------------------------------------------------------------------
// Suite 6: UI Component Integrity & Master Dashboard Mounting
// -----------------------------------------------------------------------------
console.log('\n--- Suite 6: UI Component Integrity & Master Dashboard Mounting ---');
const uiFile = path.join(frontendRoot, 'components', 'behavioral', 'DailyCertificationDashboard.tsx');
assert(fs.existsSync(uiFile), 'DailyCertificationDashboard.tsx component file exists');
const uiContent = fs.readFileSync(uiFile, 'utf8');

const requiredTestIds = [
  'data-testid="daily-certification-dashboard"',
  'data-testid="certification-banner"',
  'data-testid="dir-ratio-card"',
  'data-testid="value-attribution-card"',
  'data-testid="behavioral-maturity-cohorts"',
  'data-testid="daily-audit-table"',
];
requiredTestIds.forEach((tid) => {
  assert(uiContent.includes(tid), `Component contains ${tid}`);
});

assert(uiContent.includes('role="region"'), 'Component enforces accessibility role="region"');
assert(uiContent.includes('aria-label='), 'Component specifies aria-label attributes');
assert(uiContent.includes('certData.overallHealthScore'), 'Component displays production health score');

const masterDashboardFile = path.join(frontendRoot, 'components', 'behavioral', 'Phase28MasterDashboard.tsx');
assert(fs.existsSync(masterDashboardFile), 'Phase28MasterDashboard.tsx exists');
const masterContent = fs.readFileSync(masterDashboardFile, 'utf8');
assert(masterContent.includes("import DailyCertificationDashboard from './DailyCertificationDashboard';"), 'Master dashboard imports DailyCertificationDashboard');
assert(masterContent.includes("'certification'"), 'Master dashboard includes \'certification\' activeTab state');
assert(masterContent.includes('★ Daily Certification & Value (99.8%)'), 'Master dashboard contains Certification tab item');
assert(masterContent.includes('<DailyCertificationDashboard />'), 'Master dashboard renders DailyCertificationDashboard component');

// -----------------------------------------------------------------------------
// Suite 7: Governance Documentation Integrity
// -----------------------------------------------------------------------------
console.log('\n--- Suite 7: Governance Documentation Integrity ---');
const checklistFile = path.join(projectRoot, 'docs', 'governance', 'DAILY_PRODUCTION_CERTIFICATION_CHECKLIST.md');
assert(fs.existsSync(checklistFile), 'DAILY_PRODUCTION_CERTIFICATION_CHECKLIST.md exists in docs/governance/');
if (fs.existsSync(checklistFile)) {
  const checklistContent = fs.readFileSync(checklistFile, 'utf8');
  assert(checklistContent.includes('Section A: Telemetry Health Audit'), 'Checklist documents Section A: Telemetry Health');
  assert(checklistContent.includes('Section B: Attribution Traceability Audit'), 'Checklist documents Section B: Attribution Traceability');
  assert(checklistContent.includes('Section C: Platform Performance Budget Audit'), 'Checklist documents Section C: Performance Budget');
  assert(checklistContent.includes('Section D: Playbook Freshness Audit'), 'Checklist documents Section D: Playbook Freshness');
  assert(checklistContent.includes('Section E: AI Confidence Calibration Audit'), 'Checklist documents Section E: AI Confidence Calibration');
  assert(checklistContent.includes('Section F: Governance & Audit Trail Audit'), 'Checklist documents Section F: Governance Audit Trail');
  assert(checklistContent.includes('Decision Impact Ratio (DIRatio)'), 'Checklist documents DIRatio North Star (+24.0%)');
  assert(checklistContent.includes('$2.4M'), 'Checklist documents $2.4M capital preservation');
}

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------
console.log('\n========================================================================');
console.log(`  Daily Production Certification Verification Completed: ${passedTests}/${totalTests} Passed (${failedTests} Failed)`);
console.log('========================================================================\n');

if (failedTests > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
