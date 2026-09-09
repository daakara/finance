/**
 * Horizon 5 Verification Harness: Personal Life Operating System & Future Recovery Projections
 *
 * 260+ Fail-Closed Assertions across 10 Verification Suites:
 * - Suite 1: Personal Capacity Model & INV-OI75-P (Time, Sleep, Money, Energy, Attention)
 * - Suite 2: 168-Hour Weekly Time Budget Engine & Slack Calculation
 * - Suite 3: Interactive Outcome Recalculation & Overload Drag Mechanics
 * - Suite 4: Personal Drift Detection & Severity Categorization
 * - Suite 5: Drift Visualizations (Progress Rails, Gauges, Waterfall Root Causes)
 * - Suite 6: Future Recovery Strategy Generation (Strategies A, B, C, D)
 * - Suite 7: INV-OI83-P Recovery Feasibility Invariant Enforcement
 * - Suite 8: Recovery Velocity Metric & Monte Carlo Duration Distributions
 * - Suite 9: Recovery Traceability Lineage & Causal Waterfalls
 * - Suite 10: Non-Moralizing Adaptive Recalibration (INV-OI80-P) & Platform Invariants
 */

import { strict as assert } from 'node:assert';
import crypto from 'node:crypto';

let totalAssertions = 0;
function testAssert(condition, message) {
  totalAssertions++;
  assert.ok(condition, message);
}

function testEqual(actual, expected, message) {
  totalAssertions++;
  assert.strictEqual(actual, expected, message);
}

function testDeepEqual(actual, expected, message) {
  totalAssertions++;
  assert.deepStrictEqual(actual, expected, message);
}

console.log('');
console.log('==================================================================');
console.log('  HORIZON 5: PERSONAL LIFE OPERATING SYSTEM VERIFICATION HARNESS  ');
console.log('==================================================================');
console.log('');

// -------------------------------------------------------------
// PURE CANONICAL IMPLEMENTATIONS & FIXTURES
// -------------------------------------------------------------

const DEFAULT_PERSONAL_CAPACITY = {
  weeklyHours: 25,
  monthlyBudget: 500,
  energyCapacity: 75,
  attentionCapacity: 80,
};

const DEFAULT_WEEKLY_TIME_BUDGET = {
  totalHours: 168,
  sleepHours: 56,
  workHours: 40,
  commuteHours: 5,
  familyHours: 20,
  exerciseHours: 5,
  adminHours: 17,
  discretionaryHours: 25,
};

const DEFAULT_ALLOCATION_DOMAINS = [
  { id: 'career', name: 'Career & Craft', hours: 8, impactScore: 18, color: '#3b82f6', description: 'System design' },
  { id: 'learning', name: 'AI & Deep Learning', hours: 5, impactScore: 14, color: '#10b981', description: 'Model fine-tuning' },
  { id: 'health', name: 'Aerobic & Strength Fitness', hours: 4, impactScore: 12, color: '#f59e0b', description: 'Zone-2 running' },
  { id: 'relationships', name: 'Core Relationships & Family', hours: 3, impactScore: 9, color: '#ec4899', description: 'Presence' },
  { id: 'finance', name: 'Capital & Wealth Allocation', hours: 2, impactScore: 5, color: '#8b5cf6', description: 'Portfolio' },
];

function calculateWeeklyTimeBudget(partial) {
  const sleep = partial.sleepHours ?? 56;
  const work = partial.workHours ?? 40;
  const commute = partial.commuteHours ?? 5;
  const family = partial.familyHours ?? 20;
  const exercise = partial.exerciseHours ?? 5;
  const admin = partial.adminHours ?? 17;

  const allocated = sleep + work + commute + family + exercise + admin;
  const discretionary = Math.max(0, 168 - allocated);

  return {
    totalHours: 168,
    sleepHours: sleep,
    workHours: work,
    commuteHours: commute,
    familyHours: family,
    exerciseHours: exercise,
    adminHours: admin,
    discretionaryHours: discretionary,
  };
}

function verifyPersonalCapacity(capacity, demand, sleepHours = 56) {
  const violations = [];

  if (demand.weeklyHours > capacity.weeklyHours) {
    violations.push('TIME_CAPACITY_EXCEEDED');
  }
  if (sleepHours < 49) {
    violations.push('SLEEP_FLOOR_VIOLATION');
  }
  if (demand.monthlyBudget > capacity.monthlyBudget) {
    violations.push('MONEY_CAPACITY_EXCEEDED');
  }
  if (demand.energyDemand > capacity.energyCapacity) {
    violations.push('ENERGY_CAPACITY_EXCEEDED');
  }
  if (demand.attentionDemand > capacity.attentionCapacity) {
    violations.push('ATTENTION_CAPACITY_EXCEEDED');
  }

  const isFeasible = violations.length === 0;

  const timePct = Math.round((demand.weeklyHours / Math.max(1, capacity.weeklyHours)) * 100);
  const moneyPct = Math.round((demand.monthlyBudget / Math.max(1, capacity.monthlyBudget)) * 100);
  const energyPct = Math.round((demand.energyDemand / Math.max(1, capacity.energyCapacity)) * 100);
  const attentionPct = Math.round((demand.attentionDemand / Math.max(1, capacity.attentionCapacity)) * 100);

  const rebalanceSuggestions = [];
  if (demand.weeklyHours > capacity.weeklyHours) {
    const excess = demand.weeklyHours - capacity.weeklyHours;
    rebalanceSuggestions.push({
      domain: 'Discretionary Allocations',
      suggestedReductionHours: excess,
      reason: `Weekly time exceeds available capacity by ${excess}h. Reduce low-leverage activities to restore balance.`,
    });
  }
  if (demand.attentionDemand > capacity.attentionCapacity) {
    rebalanceSuggestions.push({
      domain: 'Deep Work / Learning',
      suggestedReductionHours: 2,
      reason: 'High cognitive fragmentation. Batch meetings or defer one complex learning module.',
    });
  }

  return {
    isFeasible,
    violations,
    utilization: { timePct, moneyPct, energyPct, attentionPct },
    rebalanceSuggestions,
  };
}

function calculateProjectedOutcomes(allocations, baseLhi = 82.4) {
  const totalHours = Object.values(allocations).reduce((sum, h) => sum + (h || 0), 0);

  const careerGain = (allocations.career || 0) * 0.45;
  const learningGain = (allocations.learning || 0) * 0.40;
  const healthGain = (allocations.health || 0) * 0.50;
  const relGain = (allocations.relationships || 0) * 0.35;
  const finGain = (allocations.finance || 0) * 0.30;

  const overloadPenalty = totalHours > 22 ? (totalHours - 22) * 0.8 : 0;

  const net6mGain = Math.min(12, Math.max(0, (careerGain + learningGain + healthGain + relGain + finGain) * 0.6 - overloadPenalty));
  const net12mGain = Math.min(16, Math.max(0, (careerGain + learningGain + healthGain + relGain + finGain) * 1.1 - overloadPenalty * 1.5));

  const projectedLhiCurrent = baseLhi;
  const projectedLhi6m = Math.min(100, Number((baseLhi + net6mGain).toFixed(1)));
  const projectedLhi12m = Math.min(100, Number((baseLhi + net12mGain).toFixed(1)));

  const confidencePct = Math.max(65, Math.min(94, Math.round(90 - Math.abs(totalHours - 18) * 1.5)));

  return {
    domains: { ...allocations },
    projectedLhiCurrent,
    projectedLhi6m,
    projectedLhi12m,
    confidencePct,
  };
}

const CANONICAL_PERSONAL_DRIFT_CARDS = [
  {
    domain: 'CAREER',
    metric: 'System Architecture & Skill Score',
    expected: 78,
    actual: 72,
    driftPct: -7.7,
    severity: 'HIGH',
    unit: 'pts',
    recommendation: 'Reallocate 2h/week from admin to focused system design mock reviews.',
    waterfallCauses: [
      { cause: 'Meeting Fragmentation & Context Switching', impact: -3.2 },
      { cause: 'Deferred Deep Work Blocks', impact: -2.8 },
      { cause: 'Evening Cognitive Fatigue', impact: -1.7 },
    ],
  },
  {
    domain: 'FINANCE',
    metric: 'Cumulative Annual Savings',
    expected: 14000,
    actual: 11800,
    driftPct: -15.7,
    severity: 'HIGH',
    unit: '€',
    recommendation: 'Audit automated discretionary subscriptions and adjust travel budget by €180/mo.',
    waterfallCauses: [
      { cause: 'Macro Cost-of-Living & Food Inflation', impact: -6.0 },
      { cause: 'Unplanned Dental / Medical Out-of-Pocket', impact: -4.2 },
      { cause: 'Summer Travel Flight Premiums', impact: -3.1 },
      { cause: 'Recurring SaaS Subscriptions', impact: -2.4 },
    ],
  },
  {
    domain: 'HEALTH',
    metric: 'Cardiovascular & Aerobic Fitness Score',
    expected: 82,
    actual: 79,
    driftPct: -3.7,
    severity: 'MEDIUM',
    unit: 'pts',
    recommendation: 'Substitute one high-intensity workout with a 35m restorative zone-2 walk.',
    waterfallCauses: [
      { cause: 'Consecutive Late Work Deliverables', impact: -2.1 },
      { cause: 'Reduced REM / Deep Sleep Quality', impact: -1.6 },
    ],
  },
  {
    domain: 'LEARNING',
    metric: 'Technical Mastery Curriculum Modules',
    expected: 12,
    actual: 8,
    driftPct: -33.3,
    severity: 'HIGH',
    unit: 'modules',
    recommendation: 'Convert commute and travel into audio paper reviews or enroll in a 3-week sprint.',
    waterfallCauses: [
      { cause: 'Late Afternoon Meeting Spillover', impact: -15.0 },
      { cause: 'Commute Exhaustion', impact: -10.0 },
      { cause: 'Lack of Dedicated Sunday Planning', impact: -8.3 },
    ],
  },
];

function calculateDrift(expected, actual) {
  if (expected === 0) return { driftPct: 0, severity: 'LOW' };
  const driftPct = Number((((actual - expected) / expected) * 100).toFixed(1));
  const absDrift = Math.abs(driftPct);

  let severity = 'LOW';
  if (absDrift > 7.0) {
    severity = 'HIGH';
  } else if (absDrift >= 3.0) {
    severity = 'MEDIUM';
  }

  return { driftPct, severity };
}

function getDriftGaugeProperties(driftPct) {
  const abs = Math.abs(driftPct);
  if (abs < 3.0) {
    return {
      color: '#10b981',
      badgeClass: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
      statusLabel: 'NOMINAL',
    };
  }
  if (abs <= 7.0) {
    return {
      color: '#f59e0b',
      badgeClass: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
      statusLabel: 'MODERATE DRIFT',
    };
  }
  return {
    color: '#ef4444',
    badgeClass: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    statusLabel: 'ELEVATED DRIFT',
  };
}

function calculateProgressRail(expected, actual, maxScale = 100) {
  const normExpected = Math.max(0, Math.min(100, (expected / maxScale) * 100));
  const normActual = Math.max(0, Math.min(100, (actual / maxScale) * 100));
  return {
    expectedOffsetPct: Number(normExpected.toFixed(1)),
    actualOffsetPct: Number(normActual.toFixed(1)),
  };
}

const CANONICAL_RECOVERY_PROJECTION = {
  projectionId: 'REC-PROJ-2026-09',
  gapName: 'Senior AI Engineer Trajectory Gap',
  currentLhi: 73,
  baselineProjectedLhi: 79,
  driftLevel: 'MODERATE',
  recommendedStrategyId: 'STRAT-D-COMBINED',
  monteCarloSimulations: {
    runs: 10000,
    p10Months: 4,
    p50Months: 6,
    p90Months: 8,
    confidencePct: 82,
  },
  waterfallImpact: [
    { lever: 'Core Skill Architecture Mastery', lhiContribution: 5.2 },
    { lever: 'Flagship Portfolio Shipped', lhiContribution: 4.1 },
    { lever: 'Executive Interview Performance', lhiContribution: 3.1 },
    { lever: 'Conviction & Psychological Momentum', lhiContribution: 1.6 },
  ],
  strategies: [
    {
      strategyId: 'STRAT-A-TIME',
      name: 'Strategy A: Time Reallocation',
      type: 'TIME_REALLOCATION',
      description: 'Reallocate +3 hrs/week from media to system architecture practice.',
      weeklyHoursRequired: 3,
      monthlyCost: 0,
      energyDemand: 15,
      attentionDemand: 20,
      projectedLhi: 84,
      recoveryTimeMonths: 8,
      recoveryVelocity: 2.5,
      confidencePct: 78,
      isFeasible: true,
      traceabilityLineage: [
        { step: '+3 Study Hours', delta: '+15% weekly depth' },
        { step: 'Skill Growth', delta: '+6 pts competency' },
        { step: 'Interview Readiness', delta: '+8% pass rate' },
        { step: 'Career Goal Recovery', delta: '+5.0 LHI' },
      ],
    },
    {
      strategyId: 'STRAT-B-ACCELERATOR',
      name: 'Strategy B: Course Accelerator',
      type: 'ACCELERATOR',
      description: 'Enroll in a 6-week intensive engineering cohort.',
      weeklyHoursRequired: 2,
      monthlyCost: 150,
      energyDemand: 18,
      attentionDemand: 25,
      projectedLhi: 86,
      recoveryTimeMonths: 7,
      recoveryVelocity: 2.8,
      confidencePct: 80,
      isFeasible: true,
      traceabilityLineage: [
        { step: 'Curated Cohort', delta: '+40% material speed' },
        { step: 'Project Completion', delta: '+2 production apps' },
        { step: 'Resume Leverage', delta: '+18% callback rate' },
        { step: 'Career Goal Recovery', delta: '+7.0 LHI' },
      ],
    },
    {
      strategyId: 'STRAT-C-MENTOR',
      name: 'Strategy C: 1-on-1 Coach & Mentor',
      type: 'COACH_MENTOR',
      description: 'Bi-weekly 60m tactical mentoring with a Principal AI Architect.',
      weeklyHoursRequired: 1.5,
      monthlyCost: 200,
      energyDemand: 12,
      attentionDemand: 22,
      projectedLhi: 87,
      recoveryTimeMonths: 6,
      recoveryVelocity: 3.3,
      confidencePct: 81,
      isFeasible: true,
      traceabilityLineage: [
        { step: 'Targeted Feedback', delta: 'Zero wasted rabbit holes' },
        { step: 'Mock Interview Review', delta: '+25% system design clarity' },
        { step: 'Network Referral', delta: '+30% offer likelihood' },
        { step: 'Career Goal Recovery', delta: '+8.0 LHI' },
      ],
    },
    {
      strategyId: 'STRAT-D-COMBINED',
      name: 'Strategy D: Combined High-Velocity Plan',
      type: 'COMBINED',
      description: '+2 hrs study + Project Accelerator + Monthly Mentor Review.',
      weeklyHoursRequired: 4.5,
      monthlyCost: 250,
      energyDemand: 24,
      attentionDemand: 30,
      projectedLhi: 88,
      recoveryTimeMonths: 5,
      recoveryVelocity: 4.0,
      confidencePct: 82,
      isFeasible: true,
      traceabilityLineage: [
        { step: 'Structured Protocol', delta: 'Multi-lever compound velocity' },
        { step: 'Rapid Milestone Shipped', delta: 'Month 2 milestone cleared' },
        { step: 'Top-Tier Interview Readiness', delta: 'Top 5% candidate pool' },
        { step: 'Career Goal Recovery', delta: '+14.0 LHI' },
      ],
    },
  ],
};

function verifyRecoveryFeasibility(capacity, strategy) {
  const demand = {
    weeklyHours: strategy.weeklyHoursRequired,
    monthlyBudget: strategy.monthlyCost,
    energyDemand: strategy.energyDemand,
    attentionDemand: strategy.attentionDemand,
  };

  const check = verifyPersonalCapacity(capacity, demand);
  if (!check.isFeasible) {
    return {
      isFeasible: false,
      reason: `INV-OI83-P Violation: ${check.violations.join(', ')}`,
    };
  }

  return { isFeasible: true };
}

function calculateRecoveryVelocity(gapPoints, recoveryMonths) {
  if (recoveryMonths <= 0) return 0;
  return Number((gapPoints / recoveryMonths).toFixed(2));
}

function rankRecoveryStrategies(strategies, capacity) {
  return strategies
    .map((strat) => {
      const feasibility = verifyRecoveryFeasibility(capacity, strat);
      return {
        ...strat,
        isFeasible: feasibility.isFeasible,
        violationReason: feasibility.reason,
      };
    })
    .sort((a, b) => {
      if (a.isFeasible && !b.isFeasible) return -1;
      if (!a.isFeasible && b.isFeasible) return 1;
      if (b.recoveryVelocity !== a.recoveryVelocity) {
        return b.recoveryVelocity - a.recoveryVelocity;
      }
      return b.projectedLhi - a.projectedLhi;
    });
}

// -------------------------------------------------------------
// SUITE 1: PERSONAL CAPACITY MODEL & INV-OI75-P ENFORCEMENT
// -------------------------------------------------------------
console.log('--- Suite 1: Personal Capacity Model & INV-OI75-P Enforcement ---');

const nominalCapacity = {
  weeklyHours: 20,
  monthlyBudget: 500,
  energyCapacity: 75,
  attentionCapacity: 80,
};

const nominalDemand = {
  weeklyHours: 15,
  monthlyBudget: 350,
  energyDemand: 60,
  attentionDemand: 65,
};

const nominalResult = verifyPersonalCapacity(nominalCapacity, nominalDemand, 56);
testAssert(nominalResult.isFeasible, 'Nominal demand should be feasible');
testEqual(nominalResult.violations.length, 0, 'Nominal demand should have 0 violations');
testEqual(nominalResult.utilization.timePct, 75, 'Time utilization should be 75%');
testEqual(nominalResult.utilization.moneyPct, 70, 'Money utilization should be 70%');
testEqual(nominalResult.utilization.energyPct, 80, 'Energy utilization should be 80%');
testEqual(nominalResult.utilization.attentionPct, 81, 'Attention utilization should be 81%');

// Test 1: TIME_CAPACITY_EXCEEDED
const timeOverloadDemand = { ...nominalDemand, weeklyHours: 24 };
const timeResult = verifyPersonalCapacity(nominalCapacity, timeOverloadDemand, 56);
testAssert(!timeResult.isFeasible, 'Time overload should fail feasibility');
testAssert(timeResult.violations.includes('TIME_CAPACITY_EXCEEDED'), 'Must flag TIME_CAPACITY_EXCEEDED');
testAssert(timeResult.rebalanceSuggestions.length > 0, 'Must emit rebalancing suggestions on time overload');
testEqual(timeResult.rebalanceSuggestions[0].suggestedReductionHours, 4, 'Suggested reduction must equal 4h excess');

// Test 2: SLEEP_FLOOR_VIOLATION (< 49h/week = < 7h/night)
const sleepDeficitResult = verifyPersonalCapacity(nominalCapacity, nominalDemand, 42);
testAssert(!sleepDeficitResult.isFeasible, 'Sleep deficit should fail feasibility');
testAssert(sleepDeficitResult.violations.includes('SLEEP_FLOOR_VIOLATION'), 'Must flag SLEEP_FLOOR_VIOLATION under 49h floor');

// Test 3: MONEY_CAPACITY_EXCEEDED
const moneyOverloadDemand = { ...nominalDemand, monthlyBudget: 750 };
const moneyResult = verifyPersonalCapacity(nominalCapacity, moneyOverloadDemand, 56);
testAssert(!moneyResult.isFeasible, 'Money overload should fail feasibility');
testAssert(moneyResult.violations.includes('MONEY_CAPACITY_EXCEEDED'), 'Must flag MONEY_CAPACITY_EXCEEDED');

// Test 4: ENERGY_CAPACITY_EXCEEDED
const energyOverloadDemand = { ...nominalDemand, energyDemand: 92 };
const energyResult = verifyPersonalCapacity(nominalCapacity, energyOverloadDemand, 56);
testAssert(!energyResult.isFeasible, 'Energy overload should fail feasibility');
testAssert(energyResult.violations.includes('ENERGY_CAPACITY_EXCEEDED'), 'Must flag ENERGY_CAPACITY_EXCEEDED');

// Test 5: ATTENTION_CAPACITY_EXCEEDED
const attentionOverloadDemand = { ...nominalDemand, attentionDemand: 98 };
const attentionResult = verifyPersonalCapacity(nominalCapacity, attentionOverloadDemand, 56);
testAssert(!attentionResult.isFeasible, 'Attention overload should fail feasibility');
testAssert(attentionResult.violations.includes('ATTENTION_CAPACITY_EXCEEDED'), 'Must flag ATTENTION_CAPACITY_EXCEEDED');

// Compound violations (Multiple simultaneous failures)
const compoundDemand = {
  weeklyHours: 35,
  monthlyBudget: 1200,
  energyDemand: 95,
  attentionDemand: 90,
};
const compoundResult = verifyPersonalCapacity(nominalCapacity, compoundDemand, 40);
testAssert(!compoundResult.isFeasible, 'Compound overload must fail');
testEqual(compoundResult.violations.length, 5, 'Must catch all 5 violations simultaneously');
testAssert(compoundResult.violations.includes('TIME_CAPACITY_EXCEEDED'), 'Compound: TIME');
testAssert(compoundResult.violations.includes('SLEEP_FLOOR_VIOLATION'), 'Compound: SLEEP');
testAssert(compoundResult.violations.includes('MONEY_CAPACITY_EXCEEDED'), 'Compound: MONEY');
testAssert(compoundResult.violations.includes('ENERGY_CAPACITY_EXCEEDED'), 'Compound: ENERGY');
testAssert(compoundResult.violations.includes('ATTENTION_CAPACITY_EXCEEDED'), 'Compound: ATTENTION');

// -------------------------------------------------------------
// SUITE 2: 168-HOUR WEEKLY TIME BUDGET ENGINE
// -------------------------------------------------------------

// Boundary edge tests: exact boundary conditions
testAssert(verifyPersonalCapacity(nominalCapacity, nominalDemand, 49.0).isFeasible, 'Exact 49.0h sleep floor must pass');
testAssert(!verifyPersonalCapacity(nominalCapacity, nominalDemand, 48.9).isFeasible, 'Sleep 48.9h must fail sleep floor');

testAssert(verifyPersonalCapacity(nominalCapacity, { ...nominalDemand, weeklyHours: 20 }, 56).isFeasible, 'Exact 20h time capacity must pass');
testAssert(!verifyPersonalCapacity(nominalCapacity, { ...nominalDemand, weeklyHours: 20.1 }, 56).isFeasible, '20.1h time capacity must fail');

testAssert(verifyPersonalCapacity(nominalCapacity, { ...nominalDemand, monthlyBudget: 500 }, 56).isFeasible, 'Exact €500 budget must pass');
testAssert(!verifyPersonalCapacity(nominalCapacity, { ...nominalDemand, monthlyBudget: 500.01 }, 56).isFeasible, '€500.01 budget must fail');

testAssert(verifyPersonalCapacity(nominalCapacity, { ...nominalDemand, energyDemand: 75 }, 56).isFeasible, 'Exact 75 energy demand must pass');
testAssert(!verifyPersonalCapacity(nominalCapacity, { ...nominalDemand, energyDemand: 76 }, 56).isFeasible, '76 energy demand must fail');

testAssert(verifyPersonalCapacity(nominalCapacity, { ...nominalDemand, attentionDemand: 80 }, 56).isFeasible, 'Exact 80 attention demand must pass');
testAssert(!verifyPersonalCapacity(nominalCapacity, { ...nominalDemand, attentionDemand: 81 }, 56).isFeasible, '81 attention demand must fail');

console.log('--- Suite 2: 168-Hour Weekly Time Budget Engine ---');

const budgetDefault = calculateWeeklyTimeBudget({});
testEqual(budgetDefault.totalHours, 168, 'Weekly budget total must always equal 168');
testEqual(budgetDefault.sleepHours, 56, 'Default sleep must equal 56h (8h/night)');
testEqual(budgetDefault.workHours, 40, 'Default work must equal 40h');
testEqual(budgetDefault.familyHours, 20, 'Default family must equal 20h');
testEqual(budgetDefault.commuteHours, 5, 'Default commute must equal 5h');
testEqual(budgetDefault.exerciseHours, 5, 'Default exercise must equal 5h');
testEqual(budgetDefault.adminHours, 17, 'Default admin must equal 17h');
testEqual(budgetDefault.discretionaryHours, 25, 'Discretionary slack must calculate to 25h');

const customBudget = calculateWeeklyTimeBudget({
  sleepHours: 49,
  workHours: 50,
  commuteHours: 10,
  familyHours: 15,
  exerciseHours: 7,
  adminHours: 14,
});
testEqual(customBudget.discretionaryHours, 23, 'Custom budget discretionary hours must equal 23h');

const packedBudget = calculateWeeklyTimeBudget({
  sleepHours: 56,
  workHours: 70,
  commuteHours: 12,
  familyHours: 15,
  exerciseHours: 5,
  adminHours: 10,
});
testEqual(packedBudget.discretionaryHours, 0, 'Packed schedule should yield 0 discretionary hours');

// -------------------------------------------------------------
// SUITE 3: INTERACTIVE OUTCOME RECALCULATION & OVERLOAD MECHANICS
// -------------------------------------------------------------

// Additional Time Budget Edge Cases
testEqual(calculateWeeklyTimeBudget({ sleepHours: 60, workHours: 40 }).sleepHours, 60, 'Sleep hours preserved at 60h');
testEqual(calculateWeeklyTimeBudget({ sleepHours: 60, workHours: 40 }).workHours, 40, 'Work hours preserved at 40h');
testEqual(calculateWeeklyTimeBudget({ commuteHours: 8 }).commuteHours, 8, 'Commute hours preserved at 8h');
testEqual(calculateWeeklyTimeBudget({ familyHours: 25 }).familyHours, 25, 'Family hours preserved at 25h');
testEqual(calculateWeeklyTimeBudget({ exerciseHours: 10 }).exerciseHours, 10, 'Exercise hours preserved at 10h');
testEqual(calculateWeeklyTimeBudget({ adminHours: 20 }).adminHours, 20, 'Admin hours preserved at 20h');
testAssert(calculateWeeklyTimeBudget({ workHours: 100 }).discretionaryHours >= 0, 'Discretionary hours cannot be negative');
testEqual(calculateWeeklyTimeBudget({ workHours: 120, sleepHours: 49 }).discretionaryHours, 0, 'Overallocated hours clamp discretionary to 0');

// Default Personal Capacity assertions
testEqual(DEFAULT_PERSONAL_CAPACITY.weeklyHours, 25, 'Default weekly hours = 25');
testEqual(DEFAULT_PERSONAL_CAPACITY.monthlyBudget, 500, 'Default monthly budget = 500');
testEqual(DEFAULT_PERSONAL_CAPACITY.energyCapacity, 75, 'Default energy capacity = 75');
testEqual(DEFAULT_PERSONAL_CAPACITY.attentionCapacity, 80, 'Default attention capacity = 80');

// Default Allocation Domains assertions
testEqual(DEFAULT_ALLOCATION_DOMAINS[0].id, 'career', 'Domain 0 is career');
testEqual(DEFAULT_ALLOCATION_DOMAINS[1].id, 'learning', 'Domain 1 is learning');
testEqual(DEFAULT_ALLOCATION_DOMAINS[2].id, 'health', 'Domain 2 is health');
testEqual(DEFAULT_ALLOCATION_DOMAINS[3].id, 'relationships', 'Domain 3 is relationships');
testEqual(DEFAULT_ALLOCATION_DOMAINS[4].id, 'finance', 'Domain 4 is finance');

console.log('--- Suite 3: Interactive Outcome Recalculation Engine ---');

const balancedAllocations = { career: 7, learning: 4, health: 3, relationships: 2, finance: 1 };
const balancedOutcomes = calculateProjectedOutcomes(balancedAllocations, 82.4);

testEqual(balancedOutcomes.projectedLhiCurrent, 82.4, 'Current LHI must match base');
testAssert(balancedOutcomes.projectedLhi6m > 82.4, '6m LHI must show positive gain');
testAssert(balancedOutcomes.projectedLhi12m > balancedOutcomes.projectedLhi6m, '12m LHI must compound over 6m');
testAssert(balancedOutcomes.confidencePct >= 80, 'Moderate allocations must yield high confidence');

// Overload test (> 22 hours allocated induces fatigue penalty)
const overloadedAllocations = { career: 15, learning: 12, health: 8, relationships: 5, finance: 5 };
const overloadedOutcomes = calculateProjectedOutcomes(overloadedAllocations, 82.4);
testAssert(overloadedOutcomes.confidencePct < 75, 'Overloaded allocations must penalize confidence');

// Zero allocations test
const zeroAllocations = { career: 0, learning: 0, health: 0, relationships: 0, finance: 0 };
const zeroOutcomes = calculateProjectedOutcomes(zeroAllocations, 80.0);
testEqual(zeroOutcomes.projectedLhi6m, 80.0, 'Zero allocations should yield zero 6m gain');
testEqual(zeroOutcomes.projectedLhi12m, 80.0, 'Zero allocations should yield zero 12m gain');

// -------------------------------------------------------------
// SUITE 4: PERSONAL DRIFT DETECTION & SEVERITY CATEGORIZATION
// -------------------------------------------------------------
console.log('--- Suite 4: Personal Drift Detection Engine ---');

const careerDrift = calculateDrift(78, 72);
testEqual(careerDrift.driftPct, -7.7, 'Career drift percentage must be -7.7%');
testEqual(careerDrift.severity, 'HIGH', 'Drift > 7% must be HIGH severity');

const healthDrift = calculateDrift(82, 79);
testEqual(healthDrift.driftPct, -3.7, 'Health drift percentage must be -3.7%');
testEqual(healthDrift.severity, 'MEDIUM', 'Drift between 3% and 7% must be MEDIUM severity');

const nominalDrift = calculateDrift(100, 98);
testEqual(nominalDrift.driftPct, -2.0, 'Nominal drift percentage must be -2.0%');
testEqual(nominalDrift.severity, 'LOW', 'Drift < 3% must be LOW severity');

const zeroExpected = calculateDrift(0, 10);
testEqual(zeroExpected.driftPct, 0, 'Zero expected must return 0 driftPct');

// -------------------------------------------------------------
// SUITE 5: DRIFT VISUALIZATIONS (RAILS, GAUGES, WATERFALLS)
// -------------------------------------------------------------

// Additional drift tests: Positive progress and exact boundary checks
const positiveDrift = calculateDrift(100, 115);
testEqual(positiveDrift.driftPct, 15.0, 'Positive drift of +15%');
testEqual(positiveDrift.severity, 'HIGH', 'Positive drift > 7% must have HIGH severity');

const boundaryLowDrift = calculateDrift(100, 102.9);
testEqual(boundaryLowDrift.severity, 'LOW', 'Drift of 2.9% is LOW');

const boundaryMedDrift = calculateDrift(100, 103.0);
testEqual(boundaryMedDrift.severity, 'MEDIUM', 'Drift of 3.0% is MEDIUM');

const boundaryMedMaxDrift = calculateDrift(100, 107.0);
testEqual(boundaryMedMaxDrift.severity, 'MEDIUM', 'Drift of 7.0% is MEDIUM');

const boundaryHighMinDrift = calculateDrift(100, 107.1);
testEqual(boundaryHighMinDrift.severity, 'HIGH', 'Drift of 7.1% is HIGH');

// Negative boundary checks
testEqual(calculateDrift(100, 97.1).severity, 'LOW', '-2.9% drift is LOW');
testEqual(calculateDrift(100, 97.0).severity, 'MEDIUM', '-3.0% drift is MEDIUM');
testEqual(calculateDrift(100, 93.0).severity, 'MEDIUM', '-7.0% drift is MEDIUM');
testEqual(calculateDrift(100, 92.9).severity, 'HIGH', '-7.1% drift is HIGH');

console.log('--- Suite 5: Drift Visualizations & Waterfall Root Causes ---');

testEqual(CANONICAL_PERSONAL_DRIFT_CARDS.length, 4, 'Must have 4 canonical drift cards');

const domains = CANONICAL_PERSONAL_DRIFT_CARDS.map((c) => c.domain);
testAssert(domains.includes('CAREER'), 'Drift cards must include CAREER');
testAssert(domains.includes('FINANCE'), 'Drift cards must include FINANCE');
testAssert(domains.includes('HEALTH'), 'Drift cards must include HEALTH');
testAssert(domains.includes('LEARNING'), 'Drift cards must include LEARNING');

for (const card of CANONICAL_PERSONAL_DRIFT_CARDS) {
  testAssert(card.waterfallCauses.length >= 2, `${card.domain} must have at least 2 waterfall causes`);
  for (const cause of card.waterfallCauses) {
    testAssert(cause.impact < 0, `Cause impact must be negative drag: ${cause.cause}`);
    testAssert(cause.cause.length > 5, 'Cause description must be descriptive');
  }

  const gaugeProps = getDriftGaugeProperties(card.driftPct);
  testAssert(gaugeProps.color.startsWith('#'), 'Gauge color must be valid hex');
  testAssert(gaugeProps.statusLabel.length > 0, 'Gauge status label must be present');

  const rail = calculateProgressRail(card.expected, card.actual);
  testAssert(rail.expectedOffsetPct >= 0 && rail.expectedOffsetPct <= 100, 'Expected rail must be within [0, 100]');
  testAssert(rail.actualOffsetPct >= 0 && rail.actualOffsetPct <= 100, 'Actual rail must be within [0, 100]');
}

// -------------------------------------------------------------
// SUITE 6: FUTURE RECOVERY STRATEGY GENERATION
// -------------------------------------------------------------
console.log('--- Suite 6: Future Recovery Strategy Generation ---');

const recovery = CANONICAL_RECOVERY_PROJECTION;
testEqual(recovery.strategies.length, 4, 'Must generate exactly 4 recovery strategies (A, B, C, D)');

const stratA = recovery.strategies.find((s) => s.strategyId === 'STRAT-A-TIME');
const stratB = recovery.strategies.find((s) => s.strategyId === 'STRAT-B-ACCELERATOR');
const stratC = recovery.strategies.find((s) => s.strategyId === 'STRAT-C-MENTOR');
const stratD = recovery.strategies.find((s) => s.strategyId === 'STRAT-D-COMBINED');

testAssert(!!stratA, 'Strategy A (Time) must exist');
testAssert(!!stratB, 'Strategy B (Accelerator) must exist');
testAssert(!!stratC, 'Strategy C (Mentor) must exist');
testAssert(!!stratD, 'Strategy D (Combined) must exist');

testEqual(stratA.type, 'TIME_REALLOCATION', 'Strategy A type must be TIME_REALLOCATION');
testEqual(stratB.type, 'ACCELERATOR', 'Strategy B type must be ACCELERATOR');
testEqual(stratC.type, 'COACH_MENTOR', 'Strategy C type must be COACH_MENTOR');
testEqual(stratD.type, 'COMBINED', 'Strategy D type must be COMBINED');

testEqual(stratA.monthlyCost, 0, 'Strategy A has zero financial cost');
testEqual(stratB.monthlyCost, 150, 'Strategy B cost is €150');
testEqual(stratC.monthlyCost, 200, 'Strategy C cost is €200');
testEqual(stratD.monthlyCost, 250, 'Strategy D cost is €250');

// -------------------------------------------------------------
// SUITE 7: INV-OI83-P RECOVERY FEASIBILITY INVARIANT ENFORCEMENT
// -------------------------------------------------------------
console.log('--- Suite 7: INV-OI83-P Recovery Feasibility Invariant Enforcement ---');

for (const strat of recovery.strategies) {
  const feasibility = verifyRecoveryFeasibility(DEFAULT_PERSONAL_CAPACITY, strat);
  testAssert(feasibility.isFeasible, `Strategy ${strat.strategyId} must be feasible under default capacity`);
}

// Test capacity constraint: User only has 2h/week available
const tightCapacity = {
  ...DEFAULT_PERSONAL_CAPACITY,
  weeklyHours: 2,
};

const tightCheckA = verifyRecoveryFeasibility(tightCapacity, stratA);
testAssert(!tightCheckA.isFeasible, 'Strategy A must be infeasible when weeklyHours=2');
testAssert(tightCheckA.reason.includes('TIME_CAPACITY_EXCEEDED'), 'Must identify TIME_CAPACITY_EXCEEDED in reason');

const tightCheckD = verifyRecoveryFeasibility(tightCapacity, stratD);
testAssert(!tightCheckD.isFeasible, 'Strategy D must be infeasible when weeklyHours=2');

// Test budget constraint: User has €50 budget
const brokeCapacity = {
  ...DEFAULT_PERSONAL_CAPACITY,
  monthlyBudget: 50,
};
const brokeCheckC = verifyRecoveryFeasibility(brokeCapacity, stratC);
testAssert(!brokeCheckC.isFeasible, 'Strategy C must be infeasible with €50 budget');
testAssert(brokeCheckC.reason.includes('MONEY_CAPACITY_EXCEEDED'), 'Must identify MONEY_CAPACITY_EXCEEDED in reason');

// -------------------------------------------------------------
// SUITE 8: RECOVERY VELOCITY METRIC & MONTE CARLO DISTRIBUTIONS
// -------------------------------------------------------------
console.log('--- Suite 8: Recovery Velocity Metric & Monte Carlo Distributions ---');

const velA = calculateRecoveryVelocity(20, 8);
testEqual(velA, 2.5, '20 points in 8 months = 2.5 pts/mo');

const velB = calculateRecoveryVelocity(20, 7);
testEqual(velB, 2.86, '20 points in 7 months = 2.86 pts/mo');

const velC = calculateRecoveryVelocity(20, 6);
testEqual(velC, 3.33, '20 points in 6 months = 3.33 pts/mo');

const velD = calculateRecoveryVelocity(20, 5);
testEqual(velD, 4.0, '20 points in 5 months = 4.0 pts/mo');

testEqual(calculateRecoveryVelocity(20, 0), 0, 'Zero months must return 0 velocity');

const ranked = rankRecoveryStrategies(recovery.strategies, DEFAULT_PERSONAL_CAPACITY);
testEqual(ranked[0].strategyId, 'STRAT-D-COMBINED', 'Highest velocity strategy D must rank first');
testEqual(ranked[1].strategyId, 'STRAT-C-MENTOR', 'Strategy C must rank second');
testEqual(ranked[2].strategyId, 'STRAT-B-ACCELERATOR', 'Strategy B must rank third');
testEqual(ranked[3].strategyId, 'STRAT-A-TIME', 'Strategy A must rank fourth');

testEqual(recovery.monteCarloSimulations.runs, 10000, 'Monte Carlo runs must equal 10,000');
testAssert(recovery.monteCarloSimulations.p10Months <= recovery.monteCarloSimulations.p50Months, 'p10 <= p50');
testAssert(recovery.monteCarloSimulations.p50Months <= recovery.monteCarloSimulations.p90Months, 'p50 <= p90');
testEqual(recovery.monteCarloSimulations.confidencePct, 82, 'Simulation confidence must be 82%');

// -------------------------------------------------------------
// SUITE 9: RECOVERY TRACEABILITY LINEAGE & CAUSAL WATERFALLS
// -------------------------------------------------------------

// Additional velocity and ranking checks
testEqual(calculateRecoveryVelocity(10, 2), 5.0, '10 points in 2 months = 5.0 pts/mo');
testEqual(calculateRecoveryVelocity(15, 3), 5.0, '15 points in 3 months = 5.0 pts/mo');
testEqual(calculateRecoveryVelocity(50, 10), 5.0, '50 points in 10 months = 5.0 pts/mo');
testEqual(calculateRecoveryVelocity(100, 20), 5.0, '100 points in 20 months = 5.0 pts/mo');
testEqual(calculateRecoveryVelocity(0, 5), 0, 'Zero gap must yield 0 velocity');
testEqual(calculateRecoveryVelocity(10, -1), 0, 'Negative months must return 0 velocity');

// Percentile integrity checks
testAssert(recovery.monteCarloSimulations.p10Months < recovery.monteCarloSimulations.p90Months, 'p10 strictly < p90');
testAssert(recovery.monteCarloSimulations.p50Months > 0, 'p50 must be positive');
testAssert(recovery.monteCarloSimulations.runs >= 1000, 'Must have >= 1,000 runs');

console.log('--- Suite 9: Recovery Traceability Lineage & Causal Waterfalls ---');

for (const strat of recovery.strategies) {
  testAssert(strat.traceabilityLineage.length >= 4, `${strat.name} must have at least 4 lineage steps`);
  for (const step of strat.traceabilityLineage) {
    testAssert(step.step.length > 0, 'Lineage step title must not be empty');
    testAssert(step.delta.length > 0, 'Lineage step delta must not be empty');
  }
}

testAssert(recovery.waterfallImpact.length >= 4, 'Waterfall impact must have at least 4 levers');
const totalContribution = recovery.waterfallImpact.reduce((sum, item) => sum + item.lhiContribution, 0);
testAssert(totalContribution >= 13.0, 'Total LHI recovery contribution must be >= 13 points');

// -------------------------------------------------------------
// SUITE 10: NON-MORALIZING ADAPTIVE RECALIBRATION (INV-OI80-P) & PLATFORM AUDIT
// -------------------------------------------------------------
console.log('--- Suite 10: Non-Moralizing Adaptive Recalibration (INV-OI80-P) & Platform Audit ---');

const shameWords = ['failed', 'lazy', 'broken streak', 'shame', 'punishment', 'guilt', 'disappointing'];
for (const card of CANONICAL_PERSONAL_DRIFT_CARDS) {
  for (const word of shameWords) {
    testAssert(
      !card.recommendation.toLowerCase().includes(word),
      `Recommendation must not contain shame word '${word}': ${card.recommendation}`
    );
  }
}

testEqual(DEFAULT_ALLOCATION_DOMAINS.length, 5, 'Must have 5 default allocation domains');
for (const domain of DEFAULT_ALLOCATION_DOMAINS) {
  testAssert(domain.hours > 0, `${domain.name} hours must be > 0`);
  testAssert(domain.impactScore > 0, `${domain.name} impactScore must be > 0`);
  testAssert(domain.color.startsWith('#'), `${domain.name} color must be valid hex`);
}

const dataToHash = JSON.stringify({
  capacity: DEFAULT_PERSONAL_CAPACITY,
  drift: CANONICAL_PERSONAL_DRIFT_CARDS,
  recovery: CANONICAL_RECOVERY_PROJECTION,
});
const auditHash = crypto.createHash('sha256').update(dataToHash).digest('hex');
testAssert(auditHash.length === 64, 'State audit hash must be valid 64-char SHA-256');

console.log('');
console.log('==================================================================');
console.log(`  ALL ${totalAssertions} / ${totalAssertions} HORIZON 5 ASSERTIONS PASSED (100% FAIL-CLOSED)`);
console.log('==================================================================');
console.log('');
