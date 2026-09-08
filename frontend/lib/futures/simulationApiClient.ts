/**
 * Phase 31-M14: Simulation API Client & Service Layer
 *
 * Implements typed in-memory REST service endpoints matching:
 * - POST /api/simulation (create)
 * - POST /api/simulation/{simulationId}/execute (execute)
 * - GET /api/simulation/{simulationId} (get)
 * - POST /api/simulation/counterfactual (counterfactual)
 * - GET /api/simulation/{simulationId}/certification (certification)
 */

import {
  SimulationRequest,
  SimulationOutcome,
  CounterfactualResult,
  SimulationCertificationResult,
  CANONICAL_ASSUMPTIONS_FIXTURE,
  CANONICAL_CANDIDATE_STRATEGIES,
} from '@/types/simulation-futures';
import { runScenarioSimulation } from './scenarioSimulationEngine';
import { evaluateCounterfactualDecision } from './counterfactualEngine';
import { certifySimulationOutcome } from './futuresCertificationEngine';

const inMemorySimulations = new Map<string, { request: SimulationRequest; outcome?: SimulationOutcome }>();

export const CANONICAL_SIMULATION_REQUEST: SimulationRequest = {
  simulationId: 'SIM-FUT-2026-001',
  committeeId: 'COM-001',
  createdAtUtc: '2026-09-08T20:00:00Z',
  simulationType: 'STRATEGIC',
  horizon: '90D',
  assumptions: CANONICAL_ASSUMPTIONS_FIXTURE,
  candidateStrategies: CANONICAL_CANDIDATE_STRATEGIES.map((s) => s.strategyId),
};

export async function createSimulation(
  request: SimulationRequest
): Promise<{ simulationId: string; status: 'QUEUED' }> {
  inMemorySimulations.set(request.simulationId, { request });
  return { simulationId: request.simulationId, status: 'QUEUED' };
}

export async function executeSimulation(
  simulationId: string
): Promise<SimulationOutcome> {
  const entry = inMemorySimulations.get(simulationId) || {
    request: { ...CANONICAL_SIMULATION_REQUEST, simulationId },
  };

  const outcome = runScenarioSimulation(entry.request);
  inMemorySimulations.set(simulationId, { request: entry.request, outcome });
  return outcome;
}

export async function getSimulation(
  simulationId: string
): Promise<SimulationOutcome | null> {
  const entry = inMemorySimulations.get(simulationId);
  if (entry?.outcome) return entry.outcome;
  // If not yet executed, execute on demand
  if (entry?.request) {
    return executeSimulation(simulationId);
  }
  return executeSimulation(simulationId);
}

export async function runCounterfactualAnalysis(params: {
  decisionId: string;
  alternativeStrategyId: string;
}): Promise<CounterfactualResult> {
  return evaluateCounterfactualDecision(params.decisionId, params.alternativeStrategyId);
}

export async function getSimulationCertification(
  simulationId: string,
  options?: { enforceSafetyBreach?: boolean }
): Promise<SimulationCertificationResult> {
  const outcome = await getSimulation(simulationId);
  if (!outcome) {
    throw new Error(`Simulation ${simulationId} not found.`);
  }
  return certifySimulationOutcome(outcome, options);
}
