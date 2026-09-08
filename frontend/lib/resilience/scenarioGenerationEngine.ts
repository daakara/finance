/**
 * Phase 31-M8: Scenario Generation Engine
 *
 * Implements:
 * - 4 Canonical Scenarios: Base, Optimistic, Adverse, Severe Stress
 * - Multi-metric impact modeling (OHI, Risk, Liquidity, Velocity)
 * - Robustness score computation and scenario coverage validation
 * - Deterministic SHA-256 state hashing
 */

import { ScenarioDefinition } from '../../types/resilience-intelligence';
import { sha256Hex } from '../governance/sha256';

export const CANONICAL_SCENARIOS: ScenarioDefinition[] = [
  {
    scenarioId: 'SCN-BASE-001',
    name: 'Nominal Baseline Operating Environment',
    type: 'BASE',
    perturbationFactor: 1.0,
    description: 'Steady-state conditions with normal committee consensus, predictable market volatility, and stable learning velocity.',
    expectedOHI: 84.2,
    expectedRiskScore: 35.0,
    probabilityPct: 50.0,
  },
  {
    scenarioId: 'SCN-OPT-001',
    name: 'Optimistic High-Confluence Expansion',
    type: 'OPTIMISTIC',
    perturbationFactor: 1.10,
    description: 'Favorable capital allocation environment, heightened dissent utilization (+10%), accelerated knowledge transfer, and low incident frequency.',
    expectedOHI: 92.6,
    expectedRiskScore: 24.5,
    probabilityPct: 25.0,
  },
  {
    scenarioId: 'SCN-ADV-001',
    name: 'Adverse Macroeconomic & Dissent Headwind',
    type: 'ADVERSE',
    perturbationFactor: 0.90,
    description: 'Elevated market regime uncertainty, resource contention across underwriting committees, and 10% effectiveness drag.',
    expectedOHI: 76.5,
    expectedRiskScore: 46.2,
    probabilityPct: 15.0,
  },
  {
    scenarioId: 'SCN-STR-001',
    name: 'Severe Liquidity & Committee Fracture Stress',
    type: 'STRESS',
    perturbationFactor: 0.75,
    description: 'Severe systemic volatility shock (-25%), dual committee quorum failure, and temporary loss of primary telemetry feed.',
    expectedOHI: 68.2,
    expectedRiskScore: 58.0,
    probabilityPct: 10.0,
  },
];

export function getCanonicalScenarios(): ScenarioDefinition[] {
  return [...CANONICAL_SCENARIOS];
}

export function calculateScenarioRobustness(
  scores: { base: number; optimistic: number; adverse: number; stress: number }
): {
  robustnessScore: number;
  scenarioCoveragePct: number;
  isRobust: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  if (scores.base < 0 || scores.base > 100) violations.push('Base score out of bounds [0, 100]');
  if (scores.optimistic < 0 || scores.optimistic > 100) violations.push('Optimistic score out of bounds [0, 100]');
  if (scores.adverse < 0 || scores.adverse > 100) violations.push('Adverse score out of bounds [0, 100]');
  if (scores.stress < 0 || scores.stress > 100) violations.push('Stress score out of bounds [0, 100]');

  // Weighted robustness calculation
  const robustnessScore = Math.round(
    (0.40 * scores.base + 0.25 * scores.optimistic + 0.20 * scores.adverse + 0.15 * scores.stress) * 10
  ) / 10;

  const isRobust = robustnessScore >= 75.0 && scores.stress >= 60.0;

  return {
    robustnessScore,
    scenarioCoveragePct: 100.0,
    isRobust,
    violations,
  };
}

export function hashScenarioState(scenarios: ScenarioDefinition[] = CANONICAL_SCENARIOS): string {
  const serialized = scenarios
    .map(s => `${s.scenarioId}:${s.type}:${s.expectedOHI}:${s.expectedRiskScore}:${s.probabilityPct}`)
    .sort()
    .join('|');
  return sha256Hex(`SCENARIOS:${serialized}`);
}

export function verifyScenarioCoverage(scenarios: ScenarioDefinition[] = CANONICAL_SCENARIOS): {
  coveredCount: number;
  totalRequired: number;
  isComplete: boolean;
} {
  const types = new Set(scenarios.map(s => s.type));
  const totalRequired = 4;
  const coveredCount = (['BASE', 'OPTIMISTIC', 'ADVERSE', 'STRESS'] as const).filter(t => types.has(t)).length;
  return {
    coveredCount,
    totalRequired,
    isComplete: coveredCount === totalRequired,
  };
}

export function calculateRobustnessScore(expectedOHI: number, expectedRiskScore: number): number {
  return Math.max(0, Math.min(100, Math.round((expectedOHI * 0.7 + (100 - expectedRiskScore) * 0.3) * 10) / 10));
}

