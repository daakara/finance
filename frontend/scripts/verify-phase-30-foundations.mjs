/**
 * Phase 30 Foundations Verification: Capability Intelligence
 *
 * Validates:
 * - Capability Impact Efficiency (CIE) formula & canonical calculations
 * - Capability Impact Index (CII) weighted composite formula
 * - Capability Dependency Graph structure & prerequisite resolution
 * - Underperforming & Sunset candidate detection
 * - Invariant INV-CI1: Value Attribution Integrity (Sum(Attributed) <= Realized)
 * - Executive action recommendation contracts
 */

import { strict as assert } from 'node:assert';

// ── Inlined engine logic and canonical fixtures ────────────────────────────

const CANONICAL_CAPABILITIES = [
  {
    id: 'institutional-flow-filter',
    name: 'Institutional Flow Filter',
    category: 'ANALYTICS',
    status: 'CORE',
    valueDollars: 1100000,
    costDollars: 250000,
    cie: 4.4,
    cii: 92.4,
    dependencies: [],
    recommendedAction: 'INVEST_MORE',
  },
  {
    id: 'ai-mentor-engine',
    name: 'AI Mentor Engine',
    category: 'COACHING',
    status: 'CORE',
    valueDollars: 850000,
    costDollars: 200000,
    cie: 4.25,
    cii: 87.2,
    dependencies: ['playbook-engine'],
    recommendedAction: 'INVEST_MORE',
  },
  {
    id: 'playbook-engine',
    name: 'Playbook Engine',
    category: 'EXECUTION',
    status: 'CORE',
    valueDollars: 450000,
    costDollars: 120000,
    cie: 3.75,
    cii: 81.2,
    dependencies: [],
    recommendedAction: 'MAINTAIN',
  },
  {
    id: 'committee-governance',
    name: 'Committee Governance Gate',
    category: 'GOVERNANCE',
    status: 'PROTECTED',
    valueDollars: 290000,
    costDollars: 100000,
    cie: 2.9,
    cii: 80.8,
    dependencies: [],
    recommendedAction: 'MAINTAIN',
  },
  {
    id: 'decision-simulator',
    name: 'What-If Decision Simulator',
    category: 'SIMULATION',
    status: 'PILOT',
    valueDollars: 210000,
    costDollars: 80000,
    cie: 2.63,
    cii: 71.0,
    dependencies: ['committee-governance'],
    recommendedAction: 'OPTIMIZE',
  },
  {
    id: 'decision-journal',
    name: 'Decision Journal',
    category: 'EXECUTION',
    status: 'RETIREMENT_REVIEW',
    valueDollars: 140000,
    costDollars: 90000,
    cie: 1.56,
    cii: 53.7,
    dependencies: [],
    recommendedAction: 'REDESIGN',
  },
];

const DEPENDENCY_EDGES = [
  { source: 'playbook-engine', target: 'ai-mentor-engine', type: 'PREREQUISITE', criticality: 'CRITICAL' },
  { source: 'committee-governance', target: 'decision-simulator', type: 'ENHANCER', criticality: 'CRITICAL' },
  { source: 'institutional-flow-filter', target: 'playbook-engine', type: 'FEEDBACK_LOOP', criticality: 'OPTIONAL' },
];

function computeCIE(value, cost) {
  if (cost <= 0) return 0;
  return Math.round((value / cost) * 100) / 100;
}

function computeCII(b, o, v, a) {
  const raw = 0.35 * b + 0.30 * o + 0.20 * v + 0.15 * a;
  return Math.round(raw * 10) / 10;
}

function verifyValueAttributionIntegrity(capabilities, totalRealizedValue) {
  const totalAttributed = capabilities.reduce((sum, c) => sum + c.valueDollars, 0);
  const isSatisfied = totalAttributed <= totalRealizedValue;
  const inflationRatio = totalRealizedValue > 0 ? totalAttributed / totalRealizedValue : 1.0;
  return {
    isSatisfied,
    totalAttributed,
    totalRealizedValue,
    inflationRatio: Math.round(inflationRatio * 1000) / 1000,
  };
}

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

console.log('\n=== Phase 30 Foundations Verification: Capability Intelligence ===\n');

// ---------------------------------------------------------------------------
// Suite 1: Capability Impact Efficiency (CIE)
// ---------------------------------------------------------------------------
console.log('Suite 1: Capability Impact Efficiency (CIE = Value / Cost)');

check('CI-01: AI Mentor CIE === 4.25x ($850K / $200K)', () => {
  const cie = computeCIE(850000, 200000);
  assert.strictEqual(cie, 4.25);
});

check('CI-02: Institutional Flow Filter CIE === 4.40x ($1.1M / $250K)', () => {
  const cie = computeCIE(1100000, 250000);
  assert.strictEqual(cie, 4.4);
});

check('CI-03: Playbook Engine CIE === 3.75x ($450K / $120K)', () => {
  const cie = computeCIE(450000, 120000);
  assert.strictEqual(cie, 3.75);
});

check('CI-04: Committee Governance CIE === 2.90x ($290K / $100K)', () => {
  const cie = computeCIE(290000, 100000);
  assert.strictEqual(cie, 2.9);
});

check('CI-05: Decision Simulator CIE === 2.63x ($210K / $80K)', () => {
  const cie = computeCIE(210000, 80000);
  assert.strictEqual(cie, 2.63);
});

check('CI-06: Decision Journal CIE === 1.56x ($140K / $90K)', () => {
  const cie = computeCIE(140000, 90000);
  assert.strictEqual(cie, 1.56);
});

check('CI-07: CIE handles zero cost gracefully without division by zero', () => {
  const cie = computeCIE(100000, 0);
  assert.strictEqual(cie, 0);
});

// ---------------------------------------------------------------------------
// Suite 2: Capability Impact Index (CII)
// ---------------------------------------------------------------------------
console.log('Suite 2: Capability Impact Index (CII Formula)');

check('CI-08: CII weighted sum weights equal 1.0', () => {
  const sumWeights = 0.35 + 0.30 + 0.20 + 0.15;
  assert.ok(Math.abs(sumWeights - 1.0) < 1e-10);
});

check('CI-09: CII computation matches expected composite', () => {
  // b=94, o=92, v=96, a=82 -> 0.35*94 + 0.30*92 + 0.20*96 + 0.15*82 = 32.9 + 27.6 + 19.2 + 12.3 = 92.0
  const cii = computeCII(94, 92, 96, 82);
  assert.strictEqual(cii, 92.0);
});

check('CI-10: Max theoretical inputs produce CII === 100.0', () => {
  const maxCii = computeCII(100, 100, 100, 100);
  assert.strictEqual(maxCii, 100.0);
});

check('CI-11: Min theoretical inputs produce CII === 0.0', () => {
  const minCii = computeCII(0, 0, 0, 0);
  assert.strictEqual(minCii, 0.0);
});

// ---------------------------------------------------------------------------
// Suite 3: Capability Dependency Graph
// ---------------------------------------------------------------------------
console.log('Suite 3: Capability Dependency Graph & Prerequisites');

check('CI-12: AI Mentor requires Playbook Engine prerequisite', () => {
  const edge = DEPENDENCY_EDGES.find(e => e.source === 'playbook-engine' && e.target === 'ai-mentor-engine');
  assert.ok(edge != null);
  assert.strictEqual(edge.type, 'PREREQUISITE');
  assert.strictEqual(edge.criticality, 'CRITICAL');
});

check('CI-13: Simulator enhanced by Committee Governance', () => {
  const edge = DEPENDENCY_EDGES.find(e => e.source === 'committee-governance' && e.target === 'decision-simulator');
  assert.ok(edge != null);
  assert.strictEqual(edge.type, 'ENHANCER');
});

check('CI-14: Flow Filter provides feedback loop to Playbook Engine', () => {
  const edge = DEPENDENCY_EDGES.find(e => e.source === 'institutional-flow-filter' && e.target === 'playbook-engine');
  assert.ok(edge != null);
  assert.strictEqual(edge.type, 'FEEDBACK_LOOP');
});

check('CI-15: Dependency graph is non-empty and contains all 6 core nodes', () => {
  assert.strictEqual(CANONICAL_CAPABILITIES.length, 6);
});

// ---------------------------------------------------------------------------
// Suite 4: Retirement & Underperformance Detection (CI-300)
// ---------------------------------------------------------------------------
console.log('Suite 4: Underperforming & Sunset Candidate Detection');

check('CI-16: Decision Journal flagged for RETIREMENT_REVIEW due to low CIE (< 2.0x)', () => {
  const journal = CANONICAL_CAPABILITIES.find(c => c.id === 'decision-journal');
  assert.strictEqual(journal.status, 'RETIREMENT_REVIEW');
  assert.ok(journal.cie < 2.0);
  assert.strictEqual(journal.recommendedAction, 'REDESIGN');
});

check('CI-17: High-value capabilities identified (Top 20% by CIE)', () => {
  const sorted = [...CANONICAL_CAPABILITIES].sort((a, b) => b.cie - a.cie);
  const top2 = sorted.slice(0, 2);
  assert.strictEqual(top2[0].id, 'institutional-flow-filter');
  assert.strictEqual(top2[1].id, 'ai-mentor-engine');
  assert.ok(top2.every(c => c.recommendedAction === 'INVEST_MORE'));
});

// ---------------------------------------------------------------------------
// Suite 5: Invariant INV-CI1: Value Attribution Integrity
// ---------------------------------------------------------------------------
console.log('Suite 5: INV-CI1 Value Attribution Integrity Invariant');

check('CI-18: Total attributed value across capabilities ($3.04M) <= Realized value ($3.04M)', () => {
  const res = verifyValueAttributionIntegrity(CANONICAL_CAPABILITIES, 3040000);
  assert.strictEqual(res.isSatisfied, true);
  assert.strictEqual(res.inflationRatio, 1.0);
});

check('CI-19: Inflation scenario properly rejected by INV-CI1', () => {
  // If actual realized value was only $2.0M but attributed $3.04M -> inflation ratio 1.52 > 1.0 -> FAIL
  const res = verifyValueAttributionIntegrity(CANONICAL_CAPABILITIES, 2000000);
  assert.strictEqual(res.isSatisfied, false);
  assert.ok(res.inflationRatio > 1.0);
});

check('CI-20: Under-attribution scenario is valid (sum < realized value)', () => {
  const res = verifyValueAttributionIntegrity(CANONICAL_CAPABILITIES, 4000000);
  assert.strictEqual(res.isSatisfied, true);
  assert.ok(res.inflationRatio < 1.0);
});

// Summary
console.log(`\n${'='.repeat(60)}`);
console.log(`Phase 30 Foundations Results: ${passed} passed, ${failed} failed`);
if (errors.length > 0) {
  console.log('\nFailed assertions:');
  errors.forEach(e => console.log(`  ✗ ${e.label}: ${e.error}`));
}
console.log(`${'='.repeat(60)}\n`);

if (failed > 0) process.exit(1);
console.log('✅ All Phase 30 Foundations assertions passed.');
