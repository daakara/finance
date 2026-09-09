/**
 * Horizon 3: Strategy Portfolio Intelligence & Survivability Engine
 *
 * Implements:
 * - Multi-strategy candidate evaluation (INV-OI61: Portfolio Completeness)
 * - Comparable baseline snapshot anchoring (INV-OI62: Strategy Comparability)
 * - Multi-scenario stress testing (Baseline, Optimistic, Adverse, Stress)
 * - Robustness scoring (Mean Outcome / StdDev)
 * - Survivability analysis (Rollback SLA, Recovery Hours, Failure Risk)
 * - Weighted portfolio ranking & deterministic recommendation (INV-OI63)
 */

import type {
  OrganizationalSnapshot,
  Strategy,
  StrategyEvaluation,
  StrategyPortfolioResult,
  ScenarioOutcome,
} from '../../types/simulation-digital-twin';
import { createSnapshot } from './digitalTwinEngine';

export const CANONICAL_STRATEGIES: Strategy[] = [
  {
    strategyId: 'STRAT-A-TRN',
    name: 'Strategic Training & Curriculum Scaling',
    description: 'Targeted 15% expansion in executive development budgets to accelerate insight acquisition and transfer velocity.',
    interventions: [
      {
        targetMetric: 'TRAINING_BUDGET',
        interventionType: 'BUDGET_INCREASE',
        parameterDeltaPct: 15.0,
        costUSD: 180000,
        implementationWeeks: 4,
      },
    ],
    assumptions: [
      'Stable committee leadership with low near-term turnover',
      'High knowledge transfer across cross-functional charters',
    ],
    constraints: [
      'Total implementation cost capped at $200,000',
      'No net headcount additions',
    ],
  },
  {
    strategyId: 'STRAT-B-DUAL',
    name: 'Dual Curriculum & Autonomous Governance Rollout',
    description: 'Balanced investment combining 15% training development with 25% automated fail-closed governance rule enforcement.',
    interventions: [
      {
        targetMetric: 'TRAINING_BUDGET',
        interventionType: 'BUDGET_INCREASE',
        parameterDeltaPct: 15.0,
        costUSD: 180000,
        implementationWeeks: 4,
      },
      {
        targetMetric: 'GOVERNANCE_ADHERENCE',
        interventionType: 'GOVERNANCE_RULE',
        parameterDeltaPct: 25.0,
        costUSD: 95000,
        implementationWeeks: 6,
      },
    ],
    assumptions: [
      'Executive consensus on automated policy guardrails',
      'Fail-closed committee enforcement eliminates compliance drift',
    ],
    constraints: [
      'Requires dual-committee sponsorship',
      'Total budget under $300,000',
    ],
  },
  {
    strategyId: 'STRAT-C-CONSV',
    name: 'Conservative Capital Freeze & Risk Containment',
    description: 'Prioritizes institutional survival and risk reduction via contrarian dissent integration and discretionary freeze.',
    interventions: [
      {
        targetMetric: 'DISSENT_INTEGRATION',
        interventionType: 'RISK_DAMPENING',
        parameterDeltaPct: 35.0,
        costUSD: 45000,
        implementationWeeks: 2,
      },
      {
        targetMetric: 'BASE_RISK_FLOOR',
        interventionType: 'RISK_DAMPENING',
        parameterDeltaPct: -20.0,
        costUSD: 30000,
        implementationWeeks: 2,
      },
    ],
    assumptions: [
      'Macroeconomic volatility persists over 180-day horizon',
      'Preservation of capital overrides aggressive growth',
    ],
    constraints: [
      'Minimal capex spend (< $100,000)',
      'Zero new recurring operational commitments',
    ],
  },
  {
    strategyId: 'STRAT-D-RESIL',
    name: 'Resilient Infrastructure & RTO Compression',
    description: 'Deploys multi-level fallback automation and runbooks to compress organizational Recovery Time Objective to under 15 minutes.',
    interventions: [
      {
        targetMetric: 'RESILIENCE_INVESTMENT',
        interventionType: 'CAPITAL_ALLOCATION',
        parameterDeltaPct: 50.0,
        costUSD: 140000,
        implementationWeeks: 8,
      },
    ],
    assumptions: [
      'Operational circuit-breakers can be executed without manual intervention',
      'L1-L4 rollback runbooks tested against simulated outages',
    ],
    constraints: [
      'Requires cross-system telemetry integration',
      'Target recovery SLA strictly <= 30 minutes',
    ],
  },
];

export function getCanonicalStrategies(): Strategy[] {
  return JSON.parse(JSON.stringify(CANONICAL_STRATEGIES));
}

/**
 * Computes robustness score: Mean OHI / Standard Deviation across multi-scenario testing.
 */
export function calculateRobustnessScore(outcomes: ScenarioOutcome[]): number {
  if (outcomes.length === 0) return 0;
  const ohiValues = outcomes.map(o => o.projectedOhi);
  const mean = ohiValues.reduce((a, b) => a + b, 0) / ohiValues.length;
  const variance = ohiValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / ohiValues.length;
  const stdDev = Math.sqrt(variance);

  if (stdDev < 0.01) return 100.0; // Perfect stability
  return Number((mean / stdDev).toFixed(2));
}

/**
 * Computes survivability score based on rollback coverage, recovery SLA, and failure risk.
 */
export function calculateSurvivabilityScore(
  recoveryHours: number,
  rollbackCoveragePct: number,
  failureProbabilityPct: number
): number {
  const recoveryScore = Math.max(0, Math.min(100, 100 - recoveryHours * 10));
  const riskDampening = Math.max(0, 100 - failureProbabilityPct);

  const weighted =
    0.35 * rollbackCoveragePct +
    0.25 * recoveryScore +
    0.20 * 92.0 + // Base organizational resilience
    0.20 * riskDampening;

  return Number(weighted.toFixed(1));
}

/**
 * Evaluates a single candidate strategy across all 4 required scenarios (INV-OI61).
 * Operates against an immutable baseline snapshot (INV-OI62).
 */
export function evaluateStrategy(
  strategy: Strategy,
  baselineSnapshot: OrganizationalSnapshot
): StrategyEvaluation {
  let baseOhiDelta = 0;
  let baseRiskDelta = 0;
  let totalCost = 0;
  let recoveryHours = 1.0;
  let rollbackCoveragePct = 100.0;
  let failureProbabilityPct = 4.0;

  for (const intervention of strategy.interventions) {
    totalCost += intervention.costUSD;

    if (intervention.targetMetric === 'TRAINING_BUDGET') {
      baseOhiDelta += Number(((intervention.parameterDeltaPct / 15.0) * 4.2).toFixed(2));
      baseRiskDelta -= 2.0;
      recoveryHours = 0.5;
    } else if (intervention.targetMetric === 'GOVERNANCE_ADHERENCE') {
      baseOhiDelta += Number(((intervention.parameterDeltaPct / 25.0) * 2.4).toFixed(2));
      baseRiskDelta -= 5.5;
      recoveryHours = 1.5;
    } else if (intervention.targetMetric === 'DISSENT_INTEGRATION' || intervention.targetMetric === 'BASE_RISK_FLOOR') {
      baseOhiDelta += 1.8;
      baseRiskDelta -= 12.0;
      recoveryHours = 0.25;
      failureProbabilityPct = 1.5;
    } else if (intervention.targetMetric === 'RESILIENCE_INVESTMENT') {
      baseOhiDelta += 2.0;
      baseRiskDelta -= 8.0;
      recoveryHours = 0.3;
      failureProbabilityPct = 2.0;
    }
  }

  const baselineOhi = Number((baselineSnapshot.ohi + baseOhiDelta).toFixed(2));
  const baselineRisk = Number((baselineSnapshot.riskScore + baseRiskDelta).toFixed(2));

  // Multi-scenario stress evaluations: Baseline, Optimistic, Adverse, Stress (INV-OI61)
  const scenarioOutcomes: ScenarioOutcome[] = [
    {
      scenarioType: 'BASELINE',
      projectedOhi: baselineOhi,
      projectedRisk: baselineRisk,
      projectedVelocity: Number((baselineSnapshot.learningVelocity + 6.0).toFixed(1)),
    },
    {
      scenarioType: 'OPTIMISTIC',
      projectedOhi: Number((baselineOhi + 3.2).toFixed(2)),
      projectedRisk: Number(Math.max(10, baselineRisk - 3.0).toFixed(2)),
      projectedVelocity: Number((baselineSnapshot.learningVelocity + 9.5).toFixed(1)),
    },
    {
      scenarioType: 'ADVERSE',
      projectedOhi: Number((baselineOhi - 2.8).toFixed(2)),
      projectedRisk: Number((baselineRisk + 6.5).toFixed(2)),
      projectedVelocity: Number((baselineSnapshot.learningVelocity + 2.0).toFixed(1)),
    },
    {
      scenarioType: 'STRESS',
      projectedOhi: Number((baselineOhi - 6.5).toFixed(2)),
      projectedRisk: Number((baselineRisk + 14.0).toFixed(2)),
      projectedVelocity: Number((baselineSnapshot.learningVelocity - 4.0).toFixed(1)),
    },
  ];

  const robustnessScore = calculateRobustnessScore(scenarioOutcomes);
  const survivabilityScore = calculateSurvivabilityScore(recoveryHours, rollbackCoveragePct, failureProbabilityPct);

  // Expected ROI calculation
  const productivityGainsUSD = Math.round(baseOhiDelta * 125000);
  const expectedRoi = Number((productivityGainsUSD / Math.max(1, totalCost)).toFixed(2));

  // Risk reduction point difference
  const riskReduction = Number((baselineSnapshot.riskScore - baselineRisk).toFixed(2));

  // Weighted Portfolio Score
  // 0.35 * OHI + 0.25 * RiskReduction + 0.20 * (Robustness * 2) + 0.10 * Survivability + 0.10 * (ROI * 10)
  const weightedScore = Number(
    (
      0.35 * baselineOhi +
      0.25 * Math.max(0, riskReduction * 3.5) +
      0.20 * Math.min(100, robustnessScore * 3.2) +
      0.10 * survivabilityScore +
      0.10 * Math.min(100, expectedRoi * 25.0)
    ).toFixed(2)
  );

  return {
    strategyId: strategy.strategyId,
    strategyName: strategy.name,
    projectedOhi: baselineOhi,
    projectedRisk: baselineRisk,
    implementationCost: totalCost,
    confidencePct: 94.5,
    robustnessScore,
    survivabilityScore,
    expectedRoi,
    weightedScore,
    scenarioOutcomes,
    overallRank: 0, // Assigned during portfolio ranking
    rankingRationale: '', // Assigned during portfolio ranking
    rollbackCoveragePct,
    recoveryHours,
    failureProbabilityPct,
  };
}

/**
 * Ranks strategies and attaches deterministic ranking rationale (INV-OI63).
 */
export function rankStrategies(evaluations: StrategyEvaluation[]): StrategyEvaluation[] {
  const sorted = [...evaluations].sort((a, b) => b.weightedScore - a.weightedScore);

  return sorted.map((item, index) => {
    const rank = index + 1;
    let rationale = '';

    if (rank === 1) {
      rationale = `${item.strategyName} ranks #1 with top composite score (${item.weightedScore.toFixed(1)}), delivering strong OHI lift (${item.projectedOhi.toFixed(1)}) and superior cross-scenario robustness (${item.robustnessScore.toFixed(1)}).`;
    } else if (rank === 2) {
      rationale = `${item.strategyName} ranks #2: competitive upside with high ROI (${item.expectedRoi}x), but slightly elevated variance in adverse stress regimes.`;
    } else if (rank === 3) {
      rationale = `${item.strategyName} ranks #3: defensive posture with excellent survivability (${item.survivabilityScore.toFixed(1)}) at the cost of lower organizational growth.`;
    } else {
      rationale = `${item.strategyName} ranks #${rank}: specialized resilience optimization with higher implementation complexity and longer deployment timeline.`;
    }

    return {
      ...item,
      overallRank: rank,
      rankingRationale: rationale,
    };
  });
}

/**
 * Full portfolio evaluator: evaluates all candidates against baseline snapshot and generates ranking.
 */
export function evaluateStrategyPortfolio(
  strategies: Strategy[] = CANONICAL_STRATEGIES,
  baselineSnapshot?: OrganizationalSnapshot
): StrategyPortfolioResult {
  const baseline = baselineSnapshot || createSnapshot();
  const rawEvaluations = strategies.map(s => evaluateStrategy(s, baseline));
  const rankedEvaluations = rankStrategies(rawEvaluations);

  const top = rankedEvaluations[0];
  const hashPayload = `${baseline.snapshotId}|${rankedEvaluations.map(e => `${e.strategyId}:${e.weightedScore}`).join(',')}`;

  let h = 0x811c9dc5;
  for (let i = 0; i < hashPayload.length; i++) {
    h ^= hashPayload.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  const replayHash = `STRAT-PORTFOLIO-0x${(h >>> 0).toString(16).padStart(8, '0').toUpperCase()}`;

  return {
    portfolioId: 'PORTFOLIO-EXEC-2026',
    snapshotId: baseline.snapshotId,
    evaluations: rankedEvaluations,
    topRecommendedStrategyId: top?.strategyId || '',
    recommendedRationale: top?.rankingRationale || 'No strategies evaluated',
    evaluatedAtUtc: new Date().toISOString(),
    deterministicReplayHash: replayHash,
  };
}
