/**
 * Phase 31-M8: Strategy Survivability Engine
 *
 * Implements:
 * - Multi-Scenario Strategy Stress Testing (Base, Optimistic, Adverse, Stress)
 * - Survivability Rating Assignment (LOW, MEDIUM, HIGH, CERTIFIED)
 * - Failure Probability Quantification
 * - Robustness Index Computation
 */

import {
  StrategySurvivability,
  SurvivabilityRating,
} from '../../types/resilience-intelligence';
import { sha256Hex } from '../governance/sha256';

export const CANONICAL_STRATEGIES: StrategySurvivability[] = [
  {
    strategyId: 'STRAT-SURV-001',
    primaryScenarioScore: 92.8,
    optimisticScenarioScore: 96.4,
    adverseScenarioScore: 84.5,
    stressScenarioScore: 74.2,
    robustnessScore: 89.6,
    failureProbabilityPct: 3.2,
    survivabilityRating: 'CERTIFIED',
  },
  {
    strategyId: 'STRAT-SURV-002',
    primaryScenarioScore: 88.0,
    optimisticScenarioScore: 94.0,
    adverseScenarioScore: 78.0,
    stressScenarioScore: 66.0,
    robustnessScore: 83.9,
    failureProbabilityPct: 7.5,
    survivabilityRating: 'HIGH',
  },
  {
    strategyId: 'STRAT-SURV-003',
    primaryScenarioScore: 82.5,
    optimisticScenarioScore: 91.0,
    adverseScenarioScore: 69.0,
    stressScenarioScore: 54.0,
    robustnessScore: 74.5,
    failureProbabilityPct: 18.4,
    survivabilityRating: 'MEDIUM',
  },
];

export function getCanonicalStrategies(): StrategySurvivability[] {
  return [...CANONICAL_STRATEGIES];
}

export function evaluateStrategySurvivability(
  primaryScore: number,
  optimisticScore: number,
  adverseScore: number,
  stressScore: number
): {
  robustnessScore: number;
  failureProbabilityPct: number;
  survivabilityRating: SurvivabilityRating;
  isCertified: boolean;
} {
  // Robustness formula: 40% Base, 25% Opt, 20% Adv, 15% Str
  const robustnessScore = Math.round(
    (0.40 * primaryScore + 0.25 * optimisticScore + 0.20 * adverseScore + 0.15 * stressScore) * 10
  ) / 10;

  // Failure probability estimation
  const adverseDeficit = Math.max(0, 75.0 - adverseScore);
  const stressDeficit = Math.max(0, 65.0 - stressScore);
  const failureProbabilityPct = Math.min(
    100,
    Math.round((adverseDeficit * 0.5 + stressDeficit * 1.2) * 10) / 10
  );

  let survivabilityRating: SurvivabilityRating = 'LOW';
  if (robustnessScore >= 85.0 && failureProbabilityPct <= 5.0 && stressScore >= 70.0) {
    survivabilityRating = 'CERTIFIED';
  } else if (robustnessScore >= 78.0 && failureProbabilityPct <= 12.0) {
    survivabilityRating = 'HIGH';
  } else if (robustnessScore >= 65.0) {
    survivabilityRating = 'MEDIUM';
  }

  const isCertified = survivabilityRating === 'CERTIFIED';

  return {
    robustnessScore,
    failureProbabilityPct,
    survivabilityRating,
    isCertified,
  };
}

export function hashStrategySurvivability(strategies: StrategySurvivability[] = CANONICAL_STRATEGIES): string {
  const serialized = strategies
    .map(s => `${s.strategyId}:${s.robustnessScore}:${s.failureProbabilityPct}:${s.survivabilityRating}`)
    .sort()
    .join('|');
  return sha256Hex(`SURVIVABILITY:${serialized}`);
}
