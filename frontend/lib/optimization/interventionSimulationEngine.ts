/**
 * Phase 31-M7: Intervention Simulation Engine
 *
 * Implements:
 * - Monte Carlo Simulation (1,000 Iterations per Intervention)
 * - Sensitivity Analysis (Perturbation Testing +/-10%)
 * - Intervention Stability & Robustness Scoring
 * - Invariant INV-OI43: Scenario Determinism (100 Replays -> 1 SHA-256 Hash)
 * - Comparative Outcome Ranking & Confidence Intervals
 */

import {
  InterventionCandidate,
  InterventionSimulationResult,
  SensitivityScenario,
} from '../../types/optimization-intelligence';
import { CANONICAL_INTERVENTION_CANDIDATES } from './optimizationPortfolioEngine';
import { sha256Hex } from '../governance/sha256';

export function simulateIntervention(
  candidate: InterventionCandidate,
  iterations: number = 1000,
  baselineOHI: number = 84.2,
  baselineRisk: number = 40.0
): InterventionSimulationResult {
  // Deterministic pseudo-random seed generator for replay reproducibility
  let seed = 123456789;
  const pseudoRandom = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };

  const ohiGains: number[] = [];
  const riskReductions: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const success = pseudoRandom() <= candidate.probabilityOfSuccess;
    const factor = success ? 0.85 + pseudoRandom() * 0.3 : 0.2 + pseudoRandom() * 0.3;

    const ohiDelta = candidate.expectedOHIImprovement * factor;
    const riskDelta = candidate.expectedRiskReduction * factor;

    ohiGains.push(ohiDelta);
    riskReductions.push(riskDelta);
  }

  ohiGains.sort((a, b) => a - b);
  const avgOHIGain = ohiGains.reduce((a, b) => a + b, 0) / iterations;
  const avgRiskRed = riskReductions.reduce((a, b) => a + b, 0) / iterations;

  const p05 = ohiGains[Math.floor(iterations * 0.05)];
  const p95 = ohiGains[Math.floor(iterations * 0.95)];

  const expectedOHI = Math.round((baselineOHI + avgOHIGain) * 10) / 10;
  const expectedRiskScore = Math.round(Math.max(0, baselineRisk - avgRiskRed) * 10) / 10;
  const expectedLearningVelocity = Math.round((12.0 + candidate.expectedLearningVelocityGain) * 10) / 10;

  const variance = Math.round((p95 - p05) * 100) / 100;
  const stabilityIndex: 'HIGH' | 'MEDIUM' | 'LOW' = variance < 1.5 ? 'HIGH' : variance < 3.0 ? 'MEDIUM' : 'LOW';

  return {
    simulationId: `SIM-${candidate.interventionId}`,
    interventionId: candidate.interventionId,
    iterations,
    expectedOHI,
    expectedRiskScore,
    expectedLearningVelocity,
    confidenceLowerBound: Math.round((baselineOHI + p05) * 10) / 10,
    confidenceUpperBound: Math.round((baselineOHI + p95) * 10) / 10,
    recommendationRank: 1,
    stabilityIndex,
    variance,
  };
}

export function runSensitivityAnalysis(
  candidate: InterventionCandidate,
  perturbationPct: number = 0.10,
  baselineOHI: number = 84.2
): SensitivityScenario[] {
  const scenarios: SensitivityScenario[] = [];

  const variations = [
    { name: 'Baseline Optimistic (+10% performance)', factor: 1 + perturbationPct },
    { name: 'Nominal Case (0% drift)', factor: 1.0 },
    { name: 'Adverse Headwind (-10% cost/effectiveness drag)', factor: 1 - perturbationPct },
    { name: 'Severe Stress Case (-25% shock)', factor: 0.75 },
  ];

  for (const v of variations) {
    const sim = simulateIntervention(
      {
        ...candidate,
        expectedOHIImprovement: candidate.expectedOHIImprovement * v.factor,
        expectedRiskReduction: candidate.expectedRiskReduction * v.factor,
      },
      500,
      baselineOHI
    );

    scenarios.push({
      scenarioId: `SCEN-${candidate.interventionId}-${Math.round(v.factor * 100)}`,
      name: v.name,
      perturbationPct: Math.round((v.factor - 1) * 100),
      resultingOHI: sim.expectedOHI,
      resultingRisk: sim.expectedRiskScore,
      isStable: sim.stabilityIndex !== 'LOW',
    });
  }

  return scenarios;
}

// Invariant INV-OI43: Scenario Determinism
export function verifyINV_OI43(
  candidate: InterventionCandidate = CANONICAL_INTERVENTION_CANDIDATES[0],
  replays: number = 100
): { pass: boolean; uniqueHashes: number; sampleHash: string } {
  const hashes = new Set<string>();

  for (let i = 0; i < replays; i++) {
    const sim = simulateIntervention(candidate, 250);
    const hash = sha256Hex(JSON.stringify(sim));
    hashes.add(hash);
  }

  const uniqueHashes = hashes.size;
  const sampleHash = hashes.values().next().value ?? '';

  return {
    pass: uniqueHashes === 1,
    uniqueHashes,
    sampleHash,
  };
}

export function hashSimulationState(sim: InterventionSimulationResult): string {
  const payload = {
    id: sim.simulationId,
    ohi: sim.expectedOHI,
    rsk: sim.expectedRiskScore,
    ci: [sim.confidenceLowerBound, sim.confidenceUpperBound],
  };
  return sha256Hex(JSON.stringify(payload));
}
