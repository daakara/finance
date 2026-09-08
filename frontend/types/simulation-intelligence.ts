/**
 * Phase 31-M12: Strategic Simulation & Decision Laboratory (Digital Decision Twin) Contracts
 *
 * Implements:
 * - Simulation Request, Assumption, Scenario Definition & Result contracts
 * - Digital Twin State, Committee Twin, Policy Twin & Recommendation Twin models
 * - Intervention Comparison, Strategy Candidate & Multi-Regime Survivability models
 * - Deterministic Replay (100 Replays = 1 Hash, INV-OI64)
 * - Strict State Isolation & Baseline Preservation (INV-OI66)
 * - Recommendation Approval Gating (INV-OI69)
 * - Typed Simulation Error Contracts (SIM-ERR-001 through SIM-ERR-005)
 * - M12 Gate Traceability Matrix (M12-Gate-01 through M12-Gate-10)
 */

export type SimulationType =
  | 'RISK_FORECAST'
  | 'GROUPTHINK'
  | 'LEARNING'
  | 'COACHING'
  | 'RESILIENCE'
  | 'SURVIVABILITY'
  | 'AUTONOMOUS_ACTION'
  | 'STRATEGY_DECISION';

export type ForecastPeriod = '30D' | '90D' | '180D' | '365D';

export type ScenarioType = 'BASE' | 'OPTIMISTIC' | 'ADVERSE' | 'STRESS';

export type AssumptionCategory =
  | 'MARKET'
  | 'GOVERNANCE'
  | 'LEARNING'
  | 'RESOURCE'
  | 'RISK'
  | 'BEHAVIORAL';

export interface SimulationAssumption {
  assumptionId: string;
  name: string;
  category: AssumptionCategory;
  currentValue: number;
  projectedValue: number;
  rationale: string;
}

export interface ScenarioDefinition {
  scenarioId: string;
  scenarioName: string;
  scenarioType: ScenarioType;
  probability: number;
  ohiFloor: number;
  description: string;
  certified: boolean;
}

export interface ForecastDriver {
  driverId: string;
  name: string;
  weightPct: number;
  deltaImpact: number;
  attributionCategory: string;
}

export interface GovernanceForecast {
  projectedOHI: number;
  projectedODEI: number;
  projectedRiskScore: number;
  projectedSurvivability: number;
  stressProbability: number;
  confidencePct: number;
}

export interface ScenarioResult {
  scenarioId: string;
  scenarioType: ScenarioType;
  probability: number;
  projectedOHI: number;
  projectedODEI: number;
  projectedRisk: number;
  projectedGroupthinkScore: number;
  survivabilityScore: number;
  certificationStatus: 'PASS' | 'FAIL';
  drivers: ForecastDriver[];
}

export interface SimulationRequest {
  simulationId: string;
  initiatedBy: string;
  createdAtUtc: string;
  simulationType: SimulationType;
  forecastPeriod: ForecastPeriod;
  committeeIds: string[];
  scenarioIds: string[];
  assumptions: SimulationAssumption[];
  deterministicReplay: boolean;
}

export interface SimulationResult {
  simulationId: string;
  completedAtUtc: string;
  status: 'COMPLETED' | 'FAILED' | 'PARTIAL';
  scenarioResults: ScenarioResult[];
  overallForecast: GovernanceForecast;
  replayHash: string;
  deterministic: boolean;
  productionMutated: false; // Invariant INV-OI66: Baseline Preservation
}

export interface CommitteeTwin {
  committeeId: string;
  committeeName: string;
  memberCount: number;
  dissentFriction: number;
  consensusThreshold: number;
  projectedVoteDistribution: {
    approve: number;
    reject: number;
    abstain: number;
  };
  groupthinkVulnerability: number;
}

export interface DigitalTwinState {
  twinId: string;
  timestampUtc: string;
  committees: CommitteeTwin[];
  simulatedOHI: number;
  simulatedODEI: number;
  isolated: true; // Invariant INV-OI66: Always isolated
}

export interface StrategyCandidate {
  candidateId: string;
  name: string;
  description: string;
  baselineId: string;
  deltaOHI: number;
  deltaODEI: number;
  deltaRisk: number;
  survivabilityScore: number;
  rank: number;
  recommendationApproved: boolean; // Gated by INV-OI69
  simulationCertified: boolean;
}

export interface InterventionComparison {
  comparisonId: string;
  baselineId: string;
  baselineOHI: number;
  candidates: StrategyCandidate[];
  recommendedCandidateId: string;
  evaluatedAtUtc: string;
}

export interface SurvivabilityAssessment {
  assessmentId: string;
  strategyId: string;
  evaluatedAtUtc: string;
  survivabilityScore: number;
  robustnessScore: number;
  failureProbabilityPct: number;
  certificationStatus: 'CERTIFIED' | 'NON_CERTIFIED';
  scenarioCoveragePct: number;
}

export interface ReplayVerificationResult {
  scenarioId: string;
  replayHash: string;
  deterministic: boolean;
  replayCount: number;
  driftCount: number;
  status: 'PASS' | 'FAIL';
}

// -------------------------------------------------------------
// TYPED SIMULATION ERROR CONTRACTS
// -------------------------------------------------------------

export interface SimulationError {
  errorCode: string;
  errorType: string;
  simulationId?: string;
  message: string;
  correlationId: string;
  timestampUtc: string;
}

export interface ScenarioValidationError extends SimulationError {
  errorType: 'SCENARIO_VALIDATION_ERROR';
  scenarioId: string;
}

export interface ForecastFailureError extends SimulationError {
  errorType: 'FORECAST_FAILURE';
  failedDriver: string;
}

export interface ReplayDriftError extends SimulationError {
  errorType: 'REPLAY_DRIFT';
  expectedHash: string;
  actualHash: string;
}

export interface StateIsolationViolationError extends SimulationError {
  errorType: 'STATE_ISOLATION_VIOLATION';
  attemptedMutationTarget: string;
}

export interface UnsimulatedRecommendationError extends SimulationError {
  errorType: 'UNSIMULATED_RECOMMENDATION_ERROR';
  recommendationId: string;
}

// -------------------------------------------------------------
// CANONICAL SCENARIOS FIXTURE
// -------------------------------------------------------------

export const CANONICAL_SCENARIOS: ScenarioDefinition[] = [
  {
    scenarioId: 'SCN-BASE-01',
    scenarioName: 'Nominal Baseline Operating Regime',
    scenarioType: 'BASE',
    probability: 0.50,
    ohiFloor: 80.0,
    description: 'Current nominal economic and governance parameters with expected baseline flow.',
    certified: true,
  },
  {
    scenarioId: 'SCN-OPT-01',
    scenarioName: 'Accelerated Learning & Productivity Expansion',
    scenarioType: 'OPTIMISTIC',
    probability: 0.20,
    ohiFloor: 85.0,
    description: 'Favorable capital allocation flow, low dissent friction, and elevated cross-committee transfer velocity.',
    certified: true,
  },
  {
    scenarioId: 'SCN-ADV-01',
    scenarioName: 'Macro Volatility & Committee Turnover Shock',
    scenarioType: 'ADVERSE',
    probability: 0.20,
    ohiFloor: 72.0,
    description: 'Elevated market dispersion, 20% key member turnover, and moderate liquidity contraction.',
    certified: true,
  },
  {
    scenarioId: 'SCN-STR-01',
    scenarioName: 'Correlated Liquidity Freeze & Governance Gridlock',
    scenarioType: 'STRESS',
    probability: 0.10,
    ohiFloor: 65.0,
    description: 'Severe stress testing: -25% market shock, communication breakdown, and maximum groupthink pressure.',
    certified: true,
  },
];

export const M12_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'M12-Gate-01', invariant: 'INV-OI64', name: 'Simulation Determinism', requirement: '100 identical runs = 1 identical hash' },
  { gateId: 'M12-Gate-02', invariant: 'INV-OI65', name: 'Scenario Traceability', requirement: '100% explainability & driver lineage coverage' },
  { gateId: 'M12-Gate-03', invariant: 'INV-OI66', name: 'Baseline Isolation', requirement: '0 production state mutations from simulation' },
  { gateId: 'M12-Gate-04', invariant: 'INV-OI67', name: 'Intervention Comparability', requirement: 'All candidate strategies share identical baseline' },
  { gateId: 'M12-Gate-05', invariant: 'INV-OI68', name: 'Scenario Robustness', requirement: 'BASE, OPTIMISTIC, ADVERSE, STRESS all executed' },
  { gateId: 'M12-Gate-06', invariant: 'INV-OI69', name: 'Recommendation Validation', requirement: 'No unsimulated recommendation marked APPROVED' },
  { gateId: 'M12-Gate-07', invariant: 'INV-OI66', name: 'Digital Twin Integrity', requirement: 'Twin state complete, isolated and non-mutating' },
  { gateId: 'M12-Gate-08', invariant: 'INV-OI65', name: 'Forecast Explainability', requirement: 'Sum of driver impacts = 100% attribution' },
  { gateId: 'M12-Gate-09', invariant: 'INV-OI64', name: 'Simulation Auditability', requirement: 'Replayable, hashable, verifiable without drift' },
  { gateId: 'M12-Gate-10', invariant: 'INV-OI64..69', name: 'Master Simulation Certification', requirement: 'All gates PASS, build clean, 0 regressions' },
];
