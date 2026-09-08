/**
 * Phase 31-M14: Institutional Simulation & Futures Intelligence Contracts
 *
 * Implements:
 * - Multi-Path Scenario Generation & Horizon Forecasting (INV-OI70, INV-OI71)
 * - 100% Outcome Attribution & Driver Explainability (INV-OI72)
 * - Counterfactual Decision & Alternative Path Analysis (INV-OI73)
 * - Simulation Safety Boundaries & Policy Constraints (INV-OI74)
 * - Certified Simulation Gating for Recommendations & Actions (INV-OI75)
 * - M14 Gate Traceability Matrix (M14-Gate-01 through M14-Gate-10)
 */

export type SimulationType =
  | 'STRATEGIC'
  | 'RISK'
  | 'LEARNING'
  | 'GOVERNANCE'
  | 'PORTFOLIO';

export type SimulationHorizon =
  | '30D'
  | '90D'
  | '180D'
  | '365D';

export type AssumptionCategory =
  | 'MARKET'
  | 'RISK'
  | 'LEARNING'
  | 'GOVERNANCE'
  | 'RESILIENCE';

export interface SimulationAssumption {
  assumptionId: string;
  category: AssumptionCategory;
  parameter: string;
  value: number | string;
  confidenceScore: number; // 0.0 - 1.0
  impactWeight?: number;
}

export interface SimulationRequest {
  simulationId: string;
  committeeId: string;
  createdAtUtc: string;
  simulationType: SimulationType;
  horizon: SimulationHorizon;
  assumptions: SimulationAssumption[];
  candidateStrategies: string[];
}

export type ScenarioType =
  | 'BASELINE'
  | 'OPTIMISTIC'
  | 'ADVERSE'
  | 'STRESS';

export interface SimulationScenario {
  scenarioId: string;
  scenarioType: ScenarioType;
  probability: number;
  projectedOHI: number;
  projectedODEI: number;
  projectedRiskScore: number;
  projectedGroupthinkScore: number;
  projectedLearningVelocity: number;
  drivers?: Array<{ name: string; weight: number; impact: number }>;
}

export interface SimulationOutcome {
  simulationId: string;
  certified: boolean;
  confidenceScore: number;
  scenarios: SimulationScenario[];
  recommendedStrategyId: string;
  outcomeHash: string;
  generatedAtUtc: string;
  attributionCoverage: number; // Must be 1.0 for INV-OI72
  safetyViolations?: string[];
}

export interface CounterfactualResult {
  decisionId: string;
  alternativeDecisionId: string;
  actualOutcome: number;
  simulatedOutcome: number;
  delta: number;
  explanation: string;
  causalDrivers: Array<{ factor: string; deltaContribution: number }>;
  lineageType: 'DECISION' | 'LEARNING' | 'RISK' | 'RECOMMENDATION' | 'POLICY';
}

export interface StrategyRanking {
  strategyId: string;
  name: string;
  bestReturnScore: number;
  bestGovernanceScore: number;
  bestLearningScore: number;
  bestResilienceScore: number;
  overallScore: number;
  rank: number;
}

export interface SimulationCertificationResult {
  simulationId: string;
  certified: boolean;
  invariantsPassed: string[];
  failedInvariants: string[];
  auditHash: string;
  timestampUtc: string;
}

// -------------------------------------------------------------
// CANONICAL FIXTURES
// -------------------------------------------------------------

export const CANONICAL_ASSUMPTIONS_FIXTURE: SimulationAssumption[] = [
  { assumptionId: 'ASM-MKT-01', category: 'MARKET', parameter: 'Macro Market Dispersion', value: -12.5, confidenceScore: 0.92, impactWeight: 0.35 },
  { assumptionId: 'ASM-RSK-01', category: 'RISK', parameter: 'Counterparty VaR Compression', value: 18.0, confidenceScore: 0.88, impactWeight: 0.25 },
  { assumptionId: 'ASM-GOV-01', category: 'GOVERNANCE', parameter: 'Committee Member Turnover', value: 10.0, confidenceScore: 0.95, impactWeight: 0.20 },
  { assumptionId: 'ASM-LRN-01', category: 'LEARNING', parameter: 'Decision Feedback Pacing', value: 14.0, confidenceScore: 0.90, impactWeight: 0.20 },
];

export const CANONICAL_CANDIDATE_STRATEGIES = [
  { strategyId: 'STRAT-A', name: 'Balanced Institutional Diversification (Status Quo)' },
  { strategyId: 'STRAT-B', name: 'Aggressive Capital Deployment & High Velocity' },
  { strategyId: 'STRAT-C', name: 'Conservative Macro Hedge & VaR Buffering' },
];

export const CANONICAL_HISTORICAL_DECISIONS = [
  { decisionId: 'DEC-001', title: 'Flow Regime Allocation', actualOHI: 84.2, committeeId: 'COM-001' },
  { decisionId: 'DEC-002', title: 'Tech Liquidity Tranche', actualOHI: 86.4, committeeId: 'COM-001' },
  { decisionId: 'DEC-003', title: 'Sovereign Debt Buffer Shift', actualOHI: 82.0, committeeId: 'COM-002' },
];

export const M14_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'M14-Gate-01', name: 'Scenario Coverage Certification', requirement: 'Mandatory 4 scenario regimes generated (BASELINE, OPTIMISTIC, ADVERSE, STRESS) (INV-OI71)' },
  { gateId: 'M14-Gate-02', name: 'Simulation Reproducibility', requirement: '100 identical simulation replays yield 1 unique SHA-256 hash (INV-OI70)' },
  { gateId: 'M14-Gate-03', name: 'Outcome Explainability', requirement: '100% driver attribution coverage; driver weights sum to exactly 1.0 (INV-OI72)' },
  { gateId: 'M14-Gate-04', name: 'Counterfactual Integrity', requirement: '100% bidirectional lineage to Decision, Learning, Risk, or Policy (INV-OI73)' },
  { gateId: 'M14-Gate-05', name: 'Future State Forecast Certification', requirement: 'Complete multi-horizon projection across OHI, ODEI, Risk, and Velocity' },
  { gateId: 'M14-Gate-06', name: 'Simulation Safety Certification', requirement: 'Governance limits and fail-close controls strictly enforced (INV-OI74)' },
  { gateId: 'M14-Gate-07', name: 'Strategy Ranking Integrity', requirement: 'Deterministic multi-criteria ordering across 5 dimensions' },
  { gateId: 'M14-Gate-08', name: 'Simulation UI Certification', requirement: 'Interactive Scenario Builder, Counterfactual Explorer, and Certification Panel operational' },
  { gateId: 'M14-Gate-09', name: 'Simulation API Certification', requirement: 'All 5 simulation endpoints validated with typed contracts' },
  { gateId: 'M14-Gate-10', name: 'Institutional Futures Master Certification', requirement: 'All M14 gates certified, First Load JS <= 100.0 kB, 0 regressions' },
];
