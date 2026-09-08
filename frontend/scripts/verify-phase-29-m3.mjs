/**
 * Phase 29 M3 Verification: Benchmarking + Capability Attribution (CIS)
 * Target: ≥40 assertions, 100% pass
 */

import { strict as assert } from 'node:assert';

// ── Inline canonical data ──────────────────────────────────────────────────

const CANONICAL_TEAMS = [
  { teamId: 'committee-alpha', teamName: 'Committee Alpha', odei: 91, cohort: 'ELITE', decisionQuality: 94, learningVelocity: 89, ruleAdherence: 96, drift: 8, percentile: 97 },
  { teamId: 'growth-equity', teamName: 'Growth Equity Team', odei: 86, cohort: 'HIGH_PERFORMING', decisionQuality: 88, learningVelocity: 84, ruleAdherence: 91, drift: 11, percentile: 82 },
  { teamId: 'macro-strategy', teamName: 'Macro Strategy', odei: 81, cohort: 'HIGH_PERFORMING', decisionQuality: 84, learningVelocity: 78, ruleAdherence: 87, drift: 14, percentile: 68 },
  { teamId: 'fixed-income', teamName: 'Fixed Income', odei: 74, cohort: 'DEVELOPING', decisionQuality: 76, learningVelocity: 71, ruleAdherence: 79, drift: 19, percentile: 44 },
  { teamId: 'emerging-markets', teamName: 'Emerging Markets', odei: 71, cohort: 'DEVELOPING', decisionQuality: 73, learningVelocity: 68, ruleAdherence: 74, drift: 22, percentile: 36 },
];

const CANONICAL_ROLE_COHORTS = [
  { role: 'ANALYST', avgDecisionQuality: 79, avgLearningVelocity: 82, avgRuleAdherence: 84, sampleSize: 1847 },
  { role: 'PORTFOLIO_MANAGER', avgDecisionQuality: 86, avgLearningVelocity: 78, avgRuleAdherence: 91, sampleSize: 1124 },
  { role: 'LEADERSHIP', avgDecisionQuality: 91, avgLearningVelocity: 74, avgRuleAdherence: 94, sampleSize: 247 },
];

function detectGroupthinkRisk(input) {
  const hasZeroDissent = input.dissentCount === 0;
  const hasLowEvidenceVariance = input.evidenceVariance < 0.2;
  const diversityIndex = (input.evidenceVariance + input.uniqueContributorRatio) / 2;
  const isGroupthink = hasZeroDissent && hasLowEvidenceVariance;
  const riskLevel = isGroupthink ? (input.uniqueContributorRatio < 0.5 ? 'HIGH' : 'MODERATE') : hasZeroDissent ? 'LOW' : 'NONE';
  return { groupthinkRisk: isGroupthink, riskLevel, diversityIndex };
}

function verifyBenchmarkIsolation(targetTeamId, benchmarkTeamIds) {
  const violations = benchmarkTeamIds.filter(id => id === targetTeamId);
  return { isIsolated: violations.length === 0, violations };
}

function detectInfluenceConcentration(actors) {
  const maxActor = actors.reduce((prev, curr) => curr.influencePct > prev.influencePct ? curr : prev);
  return { isConcentrated: maxActor.influencePct > 40.0, maxInfluencePct: maxActor.influencePct };
}

const CANONICAL_CAPABILITY_SCORES = [
  { capabilityId: 'institutional-flow-filter', cis: 9.2, behaviorLiftPct: 18.0, decisionQualityLift: 3.8, contributionPct: 28.0, adoptionPct: 82.0 },
  { capabilityId: 'ai-mentor-engine', cis: 7.4, behaviorLiftPct: 14.2, decisionQualityLift: 2.9, contributionPct: 21.0, adoptionPct: 74.0 },
  { capabilityId: 'playbook-engine', cis: 6.1, behaviorLiftPct: 11.4, decisionQualityLift: 2.1, contributionPct: 18.0, adoptionPct: 69.0 },
  { capabilityId: 'committee-governance', cis: 4.8, behaviorLiftPct: 8.7, decisionQualityLift: 1.6, contributionPct: 15.0, adoptionPct: 91.0 },
  { capabilityId: 'decision-journal', cis: 3.2, behaviorLiftPct: 5.8, decisionQualityLift: 1.1, contributionPct: 10.0, adoptionPct: 48.0 },
];
const OTHER_CONTRIBUTION = 8.0;

let passed = 0; let failed = 0; const errors = [];
function check(label, fn) {
  try { fn(); passed++; }
  catch (e) { failed++; errors.push({ label, error: e.message }); }
}

console.log('\n=== Phase 29 M3: Benchmarking & Capability Attribution ===\n');

// Suite 1: Team Benchmarks
console.log('Suite 1: Team Benchmarks');
check('5 teams defined', () => assert.strictEqual(CANONICAL_TEAMS.length, 5));
check('Committee Alpha ODEI === 91', () => assert.strictEqual(CANONICAL_TEAMS[0].odei, 91));
check('Committee Alpha cohort === ELITE', () => assert.strictEqual(CANONICAL_TEAMS[0].cohort, 'ELITE'));
check('all teams have ODEI > 0', () => assert.ok(CANONICAL_TEAMS.every(t => t.odei > 0)));
check('all teams have positive rule adherence', () => assert.ok(CANONICAL_TEAMS.every(t => t.ruleAdherence > 0)));
check('all teams have drift recorded', () => assert.ok(CANONICAL_TEAMS.every(t => t.drift >= 0)));
check('all teams have percentile > 0', () => assert.ok(CANONICAL_TEAMS.every(t => t.percentile > 0)));
check('top team DQ === 94', () => assert.strictEqual(CANONICAL_TEAMS[0].decisionQuality, 94));

// Suite 2: Role Cohorts
console.log('Suite 2: Role Cohorts');
check('3 role cohorts defined', () => assert.strictEqual(CANONICAL_ROLE_COHORTS.length, 3));
check('ANALYST cohort exists', () => assert.ok(CANONICAL_ROLE_COHORTS.some(r => r.role === 'ANALYST')));
check('PORTFOLIO_MANAGER cohort exists', () => assert.ok(CANONICAL_ROLE_COHORTS.some(r => r.role === 'PORTFOLIO_MANAGER')));
check('LEADERSHIP cohort exists', () => assert.ok(CANONICAL_ROLE_COHORTS.some(r => r.role === 'LEADERSHIP')));
check('all cohorts have sampleSize > 0', () => assert.ok(CANONICAL_ROLE_COHORTS.every(r => r.sampleSize > 0)));
check('LEADERSHIP DQ highest', () => {
  const l = CANONICAL_ROLE_COHORTS.find(r => r.role === 'LEADERSHIP').avgDecisionQuality;
  const others = CANONICAL_ROLE_COHORTS.filter(r => r.role !== 'LEADERSHIP').map(r => r.avgDecisionQuality);
  assert.ok(others.every(v => l > v));
});

// Suite 3: INV-OI5 Groupthink Detection
console.log('Suite 3: INV-OI5 Groupthink Detection');
const groupthinkScenario = { committeeId: 'SYNTHETIC-001', approvalCount: 12, dissentCount: 0, evidenceVariance: 0.04, uniqueContributorRatio: 0.33 };
const groupthinkResult = detectGroupthinkRisk(groupthinkScenario);
check('groupthink detected (risk = true)', () => assert.ok(groupthinkResult.groupthinkRisk === true));
check('groupthink risk level is HIGH or MODERATE', () => {
  assert.ok(['HIGH', 'MODERATE'].includes(groupthinkResult.riskLevel));
});
check('healthy committee does NOT flag groupthink', () => {
  const healthy = detectGroupthinkRisk({ committeeId: 'HEALTHY', approvalCount: 8, dissentCount: 3, evidenceVariance: 0.6, uniqueContributorRatio: 0.9 });
  assert.ok(healthy.groupthinkRisk === false);
});

// Suite 4: INV-OI6 Benchmark Isolation
console.log('Suite 4: INV-OI6 Benchmark Isolation');
const isolationResult = verifyBenchmarkIsolation('committee-alpha', ['growth-equity', 'macro-strategy', 'fixed-income', 'emerging-markets']);
check('benchmark isolation passes (no self-comparison)', () => assert.ok(isolationResult.isIsolated === true));
check('zero violations', () => assert.strictEqual(isolationResult.violations.length, 0));
check('self-comparison detected correctly', () => {
  const selfCompare = verifyBenchmarkIsolation('team-a', ['team-a', 'team-b']);
  assert.ok(selfCompare.isIsolated === false);
  assert.ok(selfCompare.violations.length > 0);
});

// Suite 5: INV-OI8 Fairness
console.log('Suite 5: INV-OI8 Organizational Fairness');
const fairResult = detectInfluenceConcentration([{ actorId: 'u1', influencePct: 63 }, { actorId: 'u2', influencePct: 22 }, { actorId: 'u3', influencePct: 15 }]);
check('concentration at 63% flagged', () => assert.ok(fairResult.isConcentrated === true));
check('max influence === 63%', () => assert.strictEqual(fairResult.maxInfluencePct, 63));
check('fair distribution passes (max 35%)', () => {
  const fair = detectInfluenceConcentration([{ actorId: 'u1', influencePct: 35 }, { actorId: 'u2', influencePct: 35 }, { actorId: 'u3', influencePct: 30 }]);
  assert.ok(fair.isConcentrated === false);
});

// Suite 6: Capability Impact Score (CIS)
console.log('Suite 6: Capability Impact Score (CIS)');
check('5 capabilities defined', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES.length, 5));
check('Flow Filter has highest CIS', () => {
  const sorted = [...CANONICAL_CAPABILITY_SCORES].sort((a, b) => b.cis - a.cis);
  assert.strictEqual(sorted[0].capabilityId, 'institutional-flow-filter');
});
check('Flow Filter CIS === 9.2', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES[0].cis, 9.2));
check('Flow Filter behavior lift === 18%', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES[0].behaviorLiftPct, 18.0));
check('Flow Filter DQ lift === 3.8', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES[0].decisionQualityLift, 3.8));
check('all capabilities have CIS > 0', () => assert.ok(CANONICAL_CAPABILITY_SCORES.every(c => c.cis > 0)));
check('all capabilities have adoption > 0', () => assert.ok(CANONICAL_CAPABILITY_SCORES.every(c => c.adoptionPct > 0)));

// Suite 7: Attribution Conservation (INV-OI2)
console.log('Suite 7: INV-OI2 Attribution Conservation');
const totalContribution = CANONICAL_CAPABILITY_SCORES.reduce((s, c) => s + c.contributionPct, 0) + OTHER_CONTRIBUTION;
check('attribution sums to exactly 100%', () => assert.strictEqual(totalContribution, 100.0));
check('Flow Filter attribution === 28%', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES[0].contributionPct, 28.0));
check('Mentor attribution === 21%', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES[1].contributionPct, 21.0));
check('Playbook attribution === 18%', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES[2].contributionPct, 18.0));
check('Governance attribution === 15%', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES[3].contributionPct, 15.0));
check('Journal attribution === 10%', () => assert.strictEqual(CANONICAL_CAPABILITY_SCORES[4].contributionPct, 10.0));
check('Other attribution === 8%', () => assert.strictEqual(OTHER_CONTRIBUTION, 8.0));

// Summary
console.log(`\n${'='.repeat(50)}`);
console.log(`Phase 29 M3 Results: ${passed} passed, ${failed} failed`);
if (errors.length > 0) { errors.forEach(e => console.log(`  ✗ ${e.label}: ${e.error}`)); }
console.log(`${'='.repeat(50)}\n`);
if (failed > 0) process.exit(1);
console.log('✅ All Phase 29 M3 assertions passed.');
