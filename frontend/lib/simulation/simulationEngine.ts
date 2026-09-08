/**
 * M12.1 Simulation Engine (Phase 31-M12)
 *
 * Implements:
 * - Deterministic scenario execution (INV-OI64)
 * - State isolation & zero production mutation (INV-OI66)
 * - Multi-regime shock propagation (BASE, OPTIMISTIC, ADVERSE, STRESS, INV-OI68)
 * - 100% Driver attribution & scenario lineage (INV-OI65)
 * - SHA-256 Replay determinism verification (100 Replays = 1 Hash)
 */

import { sha256Hex } from '../governance/sha256';
import type {
  SimulationRequest,
  SimulationResult,
  ScenarioResult,
  GovernanceForecast,
  ForecastDriver,
  ScenarioDefinition,
  ReplayVerificationResult,
  ScenarioType,
} from '../../types/simulation-intelligence';
import { CANONICAL_SCENARIOS } from '../../types/simulation-intelligence';

export const CANONICAL_SIMULATION_BASELINE = {
  baselineId: 'BASE-2026-Q3',
  ohi: 84.2,
  odei: 86.4,
  riskScore: 22.0,
  stressProbability: 0.12,
  groupthinkScore: 18.5,
  survivabilityScore: 89.4,
};

export function executeSimulation(
  request: SimulationRequest,
  baseline = CANONICAL_SIMULATION_BASELINE
): SimulationResult {
  if (!request.simulationId) {
    throw new Error('SimulationRequest must provide a unique simulationId');
  }

  // Multi-Scenario Execution (BASE, OPTIMISTIC, ADVERSE, STRESS)
  const scenarioResults: ScenarioResult[] = CANONICAL_SCENARIOS.map((sc) => {
    let ohiDelta = 0;
    let odeiDelta = 0;
    let riskDelta = 0;
    let survivabilityDelta = 0;

    switch (sc.scenarioType) {
      case 'BASE':
        ohiDelta = 0.5;
        odeiDelta = 0.8;
        riskDelta = -1.2;
        survivabilityDelta = 1.0;
        break;
      case 'OPTIMISTIC':
        ohiDelta = 3.2;
        odeiDelta = 4.1;
        riskDelta = -4.5;
        survivabilityDelta = 4.0;
        break;
      case 'ADVERSE':
        ohiDelta = -6.8;
        odeiDelta = -5.4;
        riskDelta = 8.6;
        survivabilityDelta = -7.5;
        break;
      case 'STRESS':
        ohiDelta = -14.2;
        odeiDelta = -11.8;
        riskDelta = 18.4;
        survivabilityDelta = -16.8;
        break;
    }

    // Apply specific assumptions if provided
    request.assumptions.forEach((asm) => {
      const diff = asm.projectedValue - asm.currentValue;
      if (asm.category === 'MARKET') ohiDelta += diff * 0.15;
      if (asm.category === 'GOVERNANCE') odeiDelta += diff * 0.10;
      if (asm.category === 'RISK') riskDelta -= diff * 0.20;
    });

    const projectedOHI = Number((baseline.ohi + ohiDelta).toFixed(1));
    const projectedODEI = Number((baseline.odei + odeiDelta).toFixed(1));
    const projectedRisk = Number(Math.max(0, baseline.riskScore + riskDelta).toFixed(1));
    const survivabilityScore = Number(Math.max(0, baseline.survivabilityScore + survivabilityDelta).toFixed(1));
    const projectedGroupthinkScore = Number((baseline.groupthinkScore + (sc.scenarioType === 'STRESS' ? 12 : 0)).toFixed(1));

    const drivers: ForecastDriver[] = [
      {
        driverId: `DRV-${sc.scenarioType}-01`,
        name: 'Macro Liquidity Spread',
        weightPct: 35,
        deltaImpact: Number((ohiDelta * 0.35).toFixed(2)),
        attributionCategory: 'MARKET',
      },
      {
        driverId: `DRV-${sc.scenarioType}-02`,
        name: 'Committee Decision Velocity',
        weightPct: 30,
        deltaImpact: Number((odeiDelta * 0.30).toFixed(2)),
        attributionCategory: 'GOVERNANCE',
      },
      {
        driverId: `DRV-${sc.scenarioType}-03`,
        name: 'Cross-Domain Knowledge Transfer',
        weightPct: 20,
        deltaImpact: Number((ohiDelta * 0.20).toFixed(2)),
        attributionCategory: 'LEARNING',
      },
      {
        driverId: `DRV-${sc.scenarioType}-04`,
        name: 'Dissent Friction & Absorption',
        weightPct: 15,
        deltaImpact: Number((riskDelta * 0.15).toFixed(2)),
        attributionCategory: 'RISK',
      },
    ];

    const certificationStatus: 'PASS' | 'FAIL' = projectedOHI >= sc.ohiFloor ? 'PASS' : 'FAIL';

    return {
      scenarioId: sc.scenarioId,
      scenarioType: sc.scenarioType,
      probability: sc.probability,
      projectedOHI,
      projectedODEI,
      projectedRisk,
      projectedGroupthinkScore,
      survivabilityScore,
      certificationStatus,
      drivers,
    };
  });

  // Calculate overall probability-weighted forecast
  let weightedOHI = 0;
  let weightedODEI = 0;
  let weightedRisk = 0;
  let weightedSurvivability = 0;

  scenarioResults.forEach((sr) => {
    weightedOHI += sr.projectedOHI * sr.probability;
    weightedODEI += sr.projectedODEI * sr.probability;
    weightedRisk += sr.projectedRisk * sr.probability;
    weightedSurvivability += sr.survivabilityScore * sr.probability;
  });

  const overallForecast: GovernanceForecast = {
    projectedOHI: Number(weightedOHI.toFixed(1)),
    projectedODEI: Number(weightedODEI.toFixed(1)),
    projectedRiskScore: Number(weightedRisk.toFixed(1)),
    projectedSurvivability: Number(weightedSurvivability.toFixed(1)),
    stressProbability: 0.10,
    confidencePct: 98.5,
  };

  // Canonical Deterministic Hash (INV-OI64)
  const canonicalPayload = JSON.stringify({
    simulationId: request.simulationId,
    simulationType: request.simulationType,
    forecastPeriod: request.forecastPeriod,
    scenarioResults: scenarioResults.map((s) => ({
      id: s.scenarioId,
      type: s.scenarioType,
      ohi: s.projectedOHI,
      odei: s.projectedODEI,
      risk: s.projectedRisk,
      surv: s.survivabilityScore,
    })),
    overallForecast,
  });

  const replayHash = sha256Hex(canonicalPayload);

  return {
    simulationId: request.simulationId,
    completedAtUtc: new Date().toISOString(),
    status: 'COMPLETED',
    scenarioResults,
    overallForecast,
    replayHash,
    deterministic: true,
    productionMutated: false, // Strict INV-OI66 guarantee
  };
}

export function verifyReplayDeterminism(
  request: SimulationRequest,
  iterations = 100
): ReplayVerificationResult {
  const initial = executeSimulation(request);
  const hashes = new Set<string>();

  for (let i = 0; i < iterations; i++) {
    const run = executeSimulation(request);
    hashes.add(run.replayHash);
  }

  const isDeterministic = hashes.size === 1 && hashes.has(initial.replayHash);

  return {
    scenarioId: request.simulationId,
    replayHash: initial.replayHash,
    deterministic: isDeterministic,
    replayCount: iterations,
    driftCount: hashes.size - 1,
    status: isDeterministic ? 'PASS' : 'FAIL',
  };
}
