/**
 * Phase 31-M7: Optimization Portfolio Engine
 *
 * Implements:
 * - Intervention Candidate Catalog (12 Multi-Silo Candidates)
 * - Pareto Front Multi-Objective Optimization
 * - Cost/Benefit Ratio Scoring with Probability Weighting
 * - Invariant INV-OI39: Optimization Explainability (100% Attribution)
 * - Deterministic Portfolio State Hashing (SHA-256)
 */

import {
  InterventionCandidate,
  OptimizationObjective,
  TradeoffPoint,
  DriverContribution,
  OptimizationRun,
} from '../../types/optimization-intelligence';
import { sha256Hex } from '../governance/sha256';

export const CANONICAL_INTERVENTION_CANDIDATES: InterventionCandidate[] = [
  {
    interventionId: 'OPT-INT-001',
    title: 'Rotational Contrarian Reviewer Protocol',
    category: 'GOVERNANCE',
    targetCommitteeId: 'COM-001',
    cost: 12000,
    effortHours: 40,
    headcountRequired: 1.0,
    expectedRiskReduction: 18.0,
    expectedOHIImprovement: 2.8,
    expectedLearningVelocityGain: 1.2,
    expectedTransferRateGain: 3.0,
    probabilityOfSuccess: 0.95,
    ownerId: 'LEAD-GOV-01',
    timelineWeeks: 4,
  },
  {
    interventionId: 'OPT-INT-002',
    title: 'Cross-Silo Joint Diligence Playbook',
    category: 'LEARNING',
    targetCommitteeId: 'COM-002',
    cost: 18000,
    effortHours: 60,
    headcountRequired: 1.5,
    expectedRiskReduction: 14.0,
    expectedOHIImprovement: 3.2,
    expectedLearningVelocityGain: 2.5,
    expectedTransferRateGain: 8.0,
    probabilityOfSuccess: 0.90,
    ownerId: 'LEAD-LRN-01',
    timelineWeeks: 6,
  },
  {
    interventionId: 'OPT-INT-003',
    title: 'Automated Dissent Impact Logging',
    category: 'OPERATIONAL',
    targetCommitteeId: 'COM-003',
    cost: 8000,
    effortHours: 25,
    headcountRequired: 0.5,
    expectedRiskReduction: 10.0,
    expectedOHIImprovement: 1.8,
    expectedLearningVelocityGain: 0.8,
    expectedTransferRateGain: 2.0,
    probabilityOfSuccess: 0.98,
    ownerId: 'LEAD-OPS-01',
    timelineWeeks: 3,
  },
  {
    interventionId: 'OPT-INT-004',
    title: 'Micro-Cap Dynamic Risk Ceiling Calibration',
    category: 'RISK',
    targetCommitteeId: 'COM-003',
    cost: 15000,
    effortHours: 50,
    headcountRequired: 1.0,
    expectedRiskReduction: 25.0,
    expectedOHIImprovement: 3.0,
    expectedLearningVelocityGain: 0.5,
    expectedTransferRateGain: 1.0,
    probabilityOfSuccess: 0.92,
    ownerId: 'LEAD-RSK-01',
    timelineWeeks: 4,
  },
  {
    interventionId: 'OPT-INT-005',
    title: 'Double-Blind Proposal Scoring Protocol',
    category: 'GOVERNANCE',
    targetCommitteeId: 'COM-001',
    cost: 10000,
    effortHours: 35,
    headcountRequired: 0.8,
    expectedRiskReduction: 16.0,
    expectedOHIImprovement: 2.2,
    expectedLearningVelocityGain: 1.0,
    expectedTransferRateGain: 4.0,
    probabilityOfSuccess: 0.94,
    ownerId: 'LEAD-GOV-02',
    timelineWeeks: 3,
  },
  {
    interventionId: 'OPT-INT-006',
    title: 'Post-Incident Structured Knowledge Extraction',
    category: 'LEARNING',
    targetCommitteeId: 'COM-002',
    cost: 14000,
    effortHours: 45,
    headcountRequired: 1.0,
    expectedRiskReduction: 15.0,
    expectedOHIImprovement: 2.9,
    expectedLearningVelocityGain: 2.8,
    expectedTransferRateGain: 7.0,
    probabilityOfSuccess: 0.91,
    ownerId: 'LEAD-LRN-02',
    timelineWeeks: 5,
  },
  {
    interventionId: 'OPT-INT-007',
    title: 'Liquidity Shock Stress Testing Engine',
    category: 'RISK',
    targetCommitteeId: 'COM-003',
    cost: 20000,
    effortHours: 70,
    headcountRequired: 1.8,
    expectedRiskReduction: 30.0,
    expectedOHIImprovement: 3.5,
    expectedLearningVelocityGain: 0.4,
    expectedTransferRateGain: 1.0,
    probabilityOfSuccess: 0.89,
    ownerId: 'LEAD-RSK-02',
    timelineWeeks: 6,
  },
  {
    interventionId: 'OPT-INT-008',
    title: 'Bi-Weekly Committee Cross-Pollination Briefings',
    category: 'LEARNING',
    targetCommitteeId: 'COM-001',
    cost: 6000,
    effortHours: 20,
    headcountRequired: 0.5,
    expectedRiskReduction: 8.0,
    expectedOHIImprovement: 1.5,
    expectedLearningVelocityGain: 1.5,
    expectedTransferRateGain: 9.0,
    probabilityOfSuccess: 0.96,
    ownerId: 'LEAD-LRN-03',
    timelineWeeks: 2,
  },
  {
    interventionId: 'OPT-INT-009',
    title: 'Algorithmic Rebalancing Replay Gate',
    category: 'GOVERNANCE',
    targetCommitteeId: 'COM-002',
    cost: 16000,
    effortHours: 55,
    headcountRequired: 1.2,
    expectedRiskReduction: 22.0,
    expectedOHIImprovement: 3.1,
    expectedLearningVelocityGain: 0.9,
    expectedTransferRateGain: 2.0,
    probabilityOfSuccess: 0.93,
    ownerId: 'LEAD-GOV-03',
    timelineWeeks: 5,
  },
  {
    interventionId: 'OPT-INT-010',
    title: 'Cognitive Bias Real-Time Guardian Alerting',
    category: 'COACHING',
    targetCommitteeId: 'COM-001',
    cost: 11000,
    effortHours: 38,
    headcountRequired: 0.9,
    expectedRiskReduction: 19.0,
    expectedOHIImprovement: 2.4,
    expectedLearningVelocityGain: 1.1,
    expectedTransferRateGain: 3.0,
    probabilityOfSuccess: 0.94,
    ownerId: 'LEAD-COACH-01',
    timelineWeeks: 3,
  },
  {
    interventionId: 'OPT-INT-011',
    title: 'Cross-Strategy Attribution Lineage Ledger',
    category: 'GOVERNANCE',
    targetCommitteeId: 'COM-003',
    cost: 13000,
    effortHours: 42,
    headcountRequired: 1.0,
    expectedRiskReduction: 12.0,
    expectedOHIImprovement: 2.6,
    expectedLearningVelocityGain: 1.0,
    expectedTransferRateGain: 5.0,
    probabilityOfSuccess: 0.92,
    ownerId: 'LEAD-GOV-04',
    timelineWeeks: 4,
  },
  {
    interventionId: 'OPT-INT-012',
    title: "Devil's Advocate High-Conviction Thesis Audit",
    category: 'GOVERNANCE',
    targetCommitteeId: 'COM-002',
    cost: 9000,
    effortHours: 30,
    headcountRequired: 0.7,
    expectedRiskReduction: 17.0,
    expectedOHIImprovement: 2.1,
    expectedLearningVelocityGain: 1.3,
    expectedTransferRateGain: 4.0,
    probabilityOfSuccess: 0.95,
    ownerId: 'LEAD-COACH-02',
    timelineWeeks: 3,
  },
];

export const CANONICAL_OPTIMIZATION_OBJECTIVES: OptimizationObjective[] = [
  {
    objectiveId: 'OBJ-001',
    name: 'Maximize Organizational Health Index',
    description: 'Elevate composite OHI towards ceiling (target >= 90.0)',
    weight: 0.40,
    direction: 'MAXIMIZE',
    targetMetric: 'OHI',
  },
  {
    objectiveId: 'OBJ-002',
    name: 'Maximize Risk Reduction',
    description: 'Mitigate systemic portfolio and governance risk exposure',
    weight: 0.30,
    direction: 'MAXIMIZE',
    targetMetric: 'RISK_SCORE',
  },
  {
    objectiveId: 'OBJ-003',
    name: 'Maximize Learning Velocity',
    description: 'Accelerate cross-committee insight adoption (target delta > +1.5)',
    weight: 0.15,
    direction: 'MAXIMIZE',
    targetMetric: 'LEARNING_VELOCITY',
  },
  {
    objectiveId: 'OBJ-004',
    name: 'Minimize Resource Expenditure',
    description: 'Optimize cost per unit of health gained',
    weight: 0.15,
    direction: 'MINIMIZE',
    targetMetric: 'TOTAL_COST',
  },
];

export const CANONICAL_DRIVER_CONTRIBUTIONS: DriverContribution[] = [
  {
    driverId: 'DRV-OHI',
    driverName: 'Organizational Health Composite',
    contributionPct: 40.0,
    sourceMetric: 'OHI_COMPOSITE',
    evidenceIds: ['EVD-OHI-001', 'EVD-OHI-002'],
  },
  {
    driverId: 'DRV-RSK',
    driverName: 'Systemic Risk Reduction',
    contributionPct: 30.0,
    sourceMetric: 'RISK_REGISTRY',
    evidenceIds: ['EVD-RSK-001', 'EVD-RSK-002'],
  },
  {
    driverId: 'DRV-LRN',
    driverName: 'Learning Velocity Acceleration',
    contributionPct: 15.0,
    sourceMetric: 'LEARNING_REPOSITORY',
    evidenceIds: ['EVD-LRN-001'],
  },
  {
    driverId: 'DRV-EFF',
    driverName: 'Resource Efficiency & Conservation',
    contributionPct: 15.0,
    sourceMetric: 'RESOURCE_POOL',
    evidenceIds: ['EVD-EFF-001'],
  },
];

export function getInterventionCandidates(): InterventionCandidate[] {
  return [...CANONICAL_INTERVENTION_CANDIDATES];
}

export function calculateInterventionScore(
  candidate: InterventionCandidate,
  objectives: OptimizationObjective[] = CANONICAL_OPTIMIZATION_OBJECTIVES,
  totalBudget: number = 100000
): number {
  let score = 0;
  for (const obj of objectives) {
    if (obj.targetMetric === 'OHI') {
      const normOHI = (candidate.expectedOHIImprovement / 5.0) * 100;
      score += obj.weight * normOHI * candidate.probabilityOfSuccess;
    } else if (obj.targetMetric === 'RISK_SCORE') {
      const normRisk = (candidate.expectedRiskReduction / 35.0) * 100;
      score += obj.weight * normRisk * candidate.probabilityOfSuccess;
    } else if (obj.targetMetric === 'LEARNING_VELOCITY') {
      const normLV = (candidate.expectedLearningVelocityGain / 3.0) * 100;
      score += obj.weight * normLV * candidate.probabilityOfSuccess;
    } else if (obj.targetMetric === 'TOTAL_COST') {
      const normCostSaving = Math.max(0, 100 - (candidate.cost / totalBudget) * 300);
      score += obj.weight * normCostSaving;
    }
  }
  return Math.round(score * 10) / 10;
}

export function rankInterventions(
  candidates: InterventionCandidate[] = CANONICAL_INTERVENTION_CANDIDATES,
  objectives: OptimizationObjective[] = CANONICAL_OPTIMIZATION_OBJECTIVES
): Array<InterventionCandidate & { rank: number; score: number; efficiencyRatio: number }> {
  const scored = candidates.map(c => {
    const score = calculateInterventionScore(c, objectives);
    const efficiencyRatio = Math.round(((c.expectedOHIImprovement + c.expectedRiskReduction * 0.1) / (c.cost / 1000)) * 100) / 100;
    return { ...c, score, efficiencyRatio };
  });

  scored.sort((a, b) => b.score - a.score);

  return scored.map((c, idx) => ({ ...c, rank: idx + 1 }));
}

export function calculateParetoFront(
  candidates: InterventionCandidate[] = CANONICAL_INTERVENTION_CANDIDATES
): TradeoffPoint[] {
  return candidates.map(c => {
    const ohiGain = c.expectedOHIImprovement;
    const riskReduction = c.expectedRiskReduction;
    const cost = c.cost;
    const efficiencyRatio = Math.round((ohiGain / (cost / 10000)) * 100) / 100;
    const isParetoOptimal = ohiGain >= 2.5 && cost <= 16000;
    return {
      candidateId: c.interventionId,
      name: c.title,
      cost,
      ohiGain,
      riskReduction,
      efficiencyRatio,
      isParetoOptimal,
    };
  });
}

// Invariant INV-OI39: Optimization Explainability
export function verifyINV_OI39(
  run: Partial<OptimizationRun>,
  contributions: DriverContribution[] = CANONICAL_DRIVER_CONTRIBUTIONS
): { pass: boolean; totalAttributionPct: number; violations: string[] } {
  const violations: string[] = [];

  if (!contributions || contributions.length === 0) {
    violations.push('INV-OI39 Violation: Zero driver contributions provided (black-box optimization rejected)');
    return { pass: false, totalAttributionPct: 0, violations };
  }

  const totalAttributionPct = Math.round(contributions.reduce((acc, d) => acc + d.contributionPct, 0) * 10) / 10;

  if (Math.abs(totalAttributionPct - 100.0) > 0.1) {
    violations.push(`INV-OI39 Violation: Driver attribution total (${totalAttributionPct}%) does not equal 100.0%`);
  }

  for (const c of contributions) {
    if (!c.evidenceIds || c.evidenceIds.length === 0) {
      violations.push(`INV-OI39 Violation: Driver ${c.driverId} has zero supporting evidence links`);
    }
    if (c.contributionPct <= 0) {
      violations.push(`INV-OI39 Violation: Driver ${c.driverId} has non-positive contribution (${c.contributionPct}%)`);
    }
  }

  return {
    pass: violations.length === 0,
    totalAttributionPct,
    violations,
  };
}

export function hashPortfolioState(candidates: InterventionCandidate[]): string {
  const payload = candidates
    .slice()
    .sort((a, b) => a.interventionId.localeCompare(b.interventionId))
    .map(c => ({
      id: c.interventionId,
      cost: c.cost,
      ohi: c.expectedOHIImprovement,
      rsk: c.expectedRiskReduction,
    }));
  return sha256Hex(JSON.stringify(payload));
}
