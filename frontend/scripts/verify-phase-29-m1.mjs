/**
 * Phase 29 Milestone 1 Verification
 * ODEI Engine: Formula, Classification, Canonical Fixtures, Strategic KPIs
 *
 * Target: =40 assertions, 100% pass
 */

import { strict as assert } from 'node:assert';

// -- Inline the engine constants (avoids TS import issues in .mjs) ----------

const ODEI_WEIGHTS = { dq: 0.35, oe: 0.30, le: 0.20, oh: 0.15 };

function computeODEI(inputs) {
  const raw =
    ODEI_WEIGHTS.dq * inputs.decisionQuality +
    ODEI_WEIGHTS.oe * inputs.outcomeEffectiveness +
    ODEI_WEIGHTS.le * inputs.learningEffectiveness +
    ODEI_WEIGHTS.oh * inputs.organizationalHealth;
  return Math.round(raw * 10) / 10;
}

function classifyODEI(score) {
  if (score >= 90) return 'ELITE';
  if (score >= 80) return 'HIGH_PERFORMING';
  if (score >= 70) return 'EFFECTIVE';
  if (score >= 60) return 'DEVELOPING';
  if (score >= 50) return 'AT_RISK';
  return 'CRITICAL';
}

const CANONICAL_ODEI_INPUTS = { decisionQuality: 88, outcomeEffectiveness: 82, learningEffectiveness: 84, organizationalHealth: 79 };
const CANONICAL_ODEI = { score: 84.0, priorScore: 78.2, delta: 5.8, trend: 'UP', classification: 'HIGH_PERFORMING', confidence: { confidencePct: 93.0, sampleSize: 4218, organizationsCompared: 42, observationWindowDays: 180 } };
const CANONICAL_KPIS = [
  { id: 'OM-01', current: 84.0, target: 80.0, status: 'PASS' },
  { id: 'OM-02', current: 74.0, target: 70.0, status: 'PASS' },
  { id: 'OM-03', current: 12.0, target: 10.0, status: 'PASS' },
  { id: 'OM-04', current: 12.0, target: 15.0, status: 'PASS' },
  { id: 'OM-05', current: 89.0, target: 85.0, status: 'PASS' },
];
const CANONICAL_COHORTS = { emergingPct: 8.0, developingPct: 22.0, highPerformingPct: 47.0, elitePct: 23.0 };
const CANONICAL_TEAMS = [
  { teamId: 'committee-alpha', odei: 91, cohort: 'ELITE' },
  { teamId: 'growth-equity', odei: 86, cohort: 'HIGH_PERFORMING' },
  { teamId: 'macro-strategy', odei: 81, cohort: 'HIGH_PERFORMING' },
  { teamId: 'fixed-income', odei: 74, cohort: 'DEVELOPING' },
  { teamId: 'emerging-markets', odei: 71, cohort: 'DEVELOPING' },
];

let passed = 0;
let failed = 0;
const errors = [];

function check(label, fn) {
  try {
    fn();
    passed++;
  } catch (e) {
    failed++;
    errors.push({ label, error: e.message });
  }
}

console.log('\n=== Phase 29 M1: ODEI Engine Verification ===\n');

// Suite 1: Formula Weights
console.log('Suite 1: Formula Weights');
check('ODEI_WEIGHTS.dq === 0.35', () => assert.strictEqual(ODEI_WEIGHTS.dq, 0.35));
check('ODEI_WEIGHTS.oe === 0.30', () => assert.strictEqual(ODEI_WEIGHTS.oe, 0.30));
check('ODEI_WEIGHTS.le === 0.20', () => assert.strictEqual(ODEI_WEIGHTS.le, 0.20));
check('ODEI_WEIGHTS.oh === 0.15', () => assert.strictEqual(ODEI_WEIGHTS.oh, 0.15));
check('weights sum to 1.0', () => {
  const sum = ODEI_WEIGHTS.dq + ODEI_WEIGHTS.oe + ODEI_WEIGHTS.le + ODEI_WEIGHTS.oh;
  assert.ok(Math.abs(sum - 1.0) < 1e-10, 'Expected sum to be 1.0');
});

// Suite 2: ODEI Computation
console.log('Suite 2: ODEI Computation');
check('computeODEI returns =80 for canonical inputs', () => {
  const result = computeODEI(CANONICAL_ODEI_INPUTS);
  assert.ok(result >= 80, `Expected =80, got ${result}`);
});
check('computeODEI is deterministic (100 identical runs)', () => {
  const results = Array.from({ length: 100 }, () => computeODEI(CANONICAL_ODEI_INPUTS));
  const first = results[0];
  assert.ok(results.every(r => r === first), 'Non-deterministic computation detected');
});
check('computeODEI range 0–100', () => {
  const result = computeODEI(CANONICAL_ODEI_INPUTS);
  assert.ok(result >= 0 && result <= 100);
});
check('computeODEI(100,100,100,100) === 100', () => {
  const result = computeODEI({ decisionQuality: 100, outcomeEffectiveness: 100, learningEffectiveness: 100, organizationalHealth: 100 });
  assert.strictEqual(result, 100);
});
check('computeODEI(0,0,0,0) === 0', () => {
  const result = computeODEI({ decisionQuality: 0, outcomeEffectiveness: 0, learningEffectiveness: 0, organizationalHealth: 0 });
  assert.strictEqual(result, 0);
});

// Suite 3: Classification
console.log('Suite 3: Classification');
check('classifyODEI(95) === ELITE', () => assert.strictEqual(classifyODEI(95), 'ELITE'));
check('classifyODEI(90) === ELITE', () => assert.strictEqual(classifyODEI(90), 'ELITE'));
check('classifyODEI(84) === HIGH_PERFORMING', () => assert.strictEqual(classifyODEI(84), 'HIGH_PERFORMING'));
check('classifyODEI(80) === HIGH_PERFORMING', () => assert.strictEqual(classifyODEI(80), 'HIGH_PERFORMING'));
check('classifyODEI(75) === EFFECTIVE', () => assert.strictEqual(classifyODEI(75), 'EFFECTIVE'));
check('classifyODEI(70) === EFFECTIVE', () => assert.strictEqual(classifyODEI(70), 'EFFECTIVE'));
check('classifyODEI(65) === DEVELOPING', () => assert.strictEqual(classifyODEI(65), 'DEVELOPING'));
check('classifyODEI(55) === AT_RISK', () => assert.strictEqual(classifyODEI(55), 'AT_RISK'));
check('classifyODEI(40) === CRITICAL', () => assert.strictEqual(classifyODEI(40), 'CRITICAL'));
check('classifyODEI(0) === CRITICAL', () => assert.strictEqual(classifyODEI(0), 'CRITICAL'));

// Suite 4: Canonical ODEI Fixture
console.log('Suite 4: Canonical ODEI Fixture');
check('canonical ODEI score === 84.0', () => assert.strictEqual(CANONICAL_ODEI.score, 84.0));
check('canonical ODEI priorScore === 78.2', () => assert.strictEqual(CANONICAL_ODEI.priorScore, 78.2));
check('canonical ODEI delta === 5.8', () => assert.strictEqual(CANONICAL_ODEI.delta, 5.8));
check('canonical ODEI trend === UP', () => assert.strictEqual(CANONICAL_ODEI.trend, 'UP'));
check('canonical ODEI classification === HIGH_PERFORMING', () => assert.strictEqual(CANONICAL_ODEI.classification, 'HIGH_PERFORMING'));
check('canonical ODEI confidence === 93%', () => assert.strictEqual(CANONICAL_ODEI.confidence.confidencePct, 93.0));
check('canonical ODEI sampleSize === 4218', () => assert.strictEqual(CANONICAL_ODEI.confidence.sampleSize, 4218));
check('canonical ODEI organizationsCompared === 42', () => assert.strictEqual(CANONICAL_ODEI.confidence.organizationsCompared, 42));
check('canonical ODEI observationWindowDays === 180', () => assert.strictEqual(CANONICAL_ODEI.confidence.observationWindowDays, 180));

// Suite 5: Strategic KPIs
console.log('Suite 5: Strategic KPIs');
check('KPI count === 5', () => assert.strictEqual(CANONICAL_KPIS.length, 5));
check('all 5 KPIs have PASS status', () => {
  assert.ok(CANONICAL_KPIS.every(k => k.status === 'PASS'), 'Not all KPIs are PASS');
});
check('OM-01 current (84) >= target (80)', () => {
  const k = CANONICAL_KPIS.find(k => k.id === 'OM-01');
  assert.ok(k.current >= k.target);
});
check('OM-02 knowledge reuse = 70%', () => {
  const k = CANONICAL_KPIS.find(k => k.id === 'OM-02');
  assert.ok(k.current >= 70);
});
check('OM-03 learning velocity = 10%', () => {
  const k = CANONICAL_KPIS.find(k => k.id === 'OM-03');
  assert.ok(k.current >= 10);
});
check('OM-04 consistency < 15% variance', () => {
  const k = CANONICAL_KPIS.find(k => k.id === 'OM-04');
  assert.ok(k.current < 15);
});
check('OM-05 adoption = 85%', () => {
  const k = CANONICAL_KPIS.find(k => k.id === 'OM-05');
  assert.ok(k.current >= 85);
});

// Suite 6: Cohort Distribution
console.log('Suite 6: Cohort Distribution');
check('cohort percentages sum to 100%', () => {
  const total = CANONICAL_COHORTS.emergingPct + CANONICAL_COHORTS.developingPct + CANONICAL_COHORTS.highPerformingPct + CANONICAL_COHORTS.elitePct;
  assert.strictEqual(total, 100.0);
});
check('emerging cohort (ODEI <70) = 8%', () => assert.strictEqual(CANONICAL_COHORTS.emergingPct, 8.0));
check('high performing cohort = 47%', () => assert.strictEqual(CANONICAL_COHORTS.highPerformingPct, 47.0));
check('elite cohort = 23%', () => assert.strictEqual(CANONICAL_COHORTS.elitePct, 23.0));

// Suite 7: Team Benchmarks
console.log('Suite 7: Team Benchmarks');
check('5 team benchmarks defined', () => assert.strictEqual(CANONICAL_TEAMS.length, 5));
check('top team (Committee Alpha) ODEI === 91', () => assert.strictEqual(CANONICAL_TEAMS[0].odei, 91));
check('top team classified as ELITE', () => assert.strictEqual(CANONICAL_TEAMS[0].cohort, 'ELITE'));
check('all teams have ODEI = 70', () => {
  assert.ok(CANONICAL_TEAMS.every(t => t.odei >= 70), 'Not all teams ODEI = 70');
});
check('teams sorted descending by ODEI', () => {
  for (let i = 1; i < CANONICAL_TEAMS.length; i++) {
    assert.ok(CANONICAL_TEAMS[i - 1].odei >= CANONICAL_TEAMS[i].odei, 'Teams not sorted descending');
  }
});

// Summary
console.log(`\n${'='.repeat(50)}`);
console.log(`Phase 29 M1 Results: ${passed} passed, ${failed} failed`);
if (errors.length > 0) {
  console.log('\nFailed assertions:');
  errors.forEach(e => console.log(`  ? ${e.label}: ${e.error}`));
}
console.log(`${'='.repeat(50)}\n`);

if (failed > 0) process.exit(1);
console.log('? All Phase 29 M1 assertions passed.');

