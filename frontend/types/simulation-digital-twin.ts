/**
 * Horizon 2: Executive Simulation & Digital Twin Contracts
 *
 * Implements:
 * - Directed Dependency Graph Model (Nodes, Edges, Propagation Weights)
 * - Organizational Snapshot Baseline Schema
 * - Seeded Monte Carlo Simulation Engine Contracts (XorShift32)
 * - Shock Testing Scenarios & Extreme Condition Invariants
 * - Multi-Level Rollback Strategies (L1-L4)
 * - Traceability Lineage & Explainability Invariants (INV-OI53..INV-OI60)
 */

export type DependencyNodeType =
  | 'COMMITTEE'
  | 'DECISION'
  | 'LEARNING'
  | 'RISK'
  | 'RECOMMENDATION'
  | 'SCENARIO'
  | 'BUDGET';

export type DependencyRelationship =
  | 'INFLUENCES'
  | 'DEPENDS_ON'
  | 'ADOPTS'
  | 'MITIGATES'
  | 'AMPLIFIES';

export interface DependencyNode {
  id: string;
  name: string;
  type: DependencyNodeType;
  state: Record<string, number>;
}

export interface DependencyEdge {
  sourceId: string;
  targetId: string;
  dependencyWeight: number; // [-1.0, 1.0]
  relationship: DependencyRelationship;
  confidencePct: number;
}

export interface DependencyGraph {
  nodes: DependencyNode[];
  edges: DependencyEdge[];
  lastCalibratedUtc: string;
}

export interface OrganizationalSnapshot {
  snapshotId: string;
  generatedAtUtc: string;
  ohi: number;
  odei: number;
  riskScore: number;
  learningVelocity: number;
  transferRatePct: number;
  resilienceRtoMinutes: number;
  activeCommitteesCount: number;
  pendingDecisionsCount: number;
  stateHash: string;
}

export interface ScenarioChange {
  targetMetric: string;
  previousValue: number;
  newValue: number;
  changeType: 'PERCENT_DELTA' | 'ABSOLUTE_DELTA' | 'STRUCTURAL_REPLACE';
}

export interface SimulatedState {
  simulationId: string;
  scenarioId: string;
  baselineSnapshotId: string;
  projectedOhi: number;
  projectedRisk: number;
  projectedTransferRate: number;
  projectedLearningVelocity: number;
  confidencePct: number;
  simulatedDays: number;
  timestampUtc: string;
}

export interface TraceRecord {
  traceId: string;
  simulationId: string;
  sourceMetric: string;
  targetMetric: string;
  contributionPct: number;
  valueBefore: number;
  valueAfter: number;
  weightUsed: number;
  createdAtUtc?: string;
}

export interface MonteCarloConfig {
  iterations: number;
  seed: number;
  confidenceLevel: number;
  timeHorizonDays: number;
}

export interface MonteCarloResult {
  simulationId: string;
  iterationsRun: number;
  seed: number;
  meanOhi: number;
  medianOhi: number;
  percentile5: number;
  percentile95: number;
  standardDeviation: number;
  confidenceInterval: [number, number];
  executionDurationMs: number;
  replayHash: string;
}

export type RollbackLevel = 'L1_CONFIG' | 'L2_ORGANIZATIONAL' | 'L3_OPTIMIZATION' | 'L4_FULL_RECOVERY';

export interface RollbackStrategy {
  rollbackId: string;
  strategyName: string;
  level: RollbackLevel;
  triggerConditions: string[];
  rollbackActions: string[];
  estimatedRecoveryHours: number;
  targetRecoveryStateId?: string;
}

export interface ShockScenario {
  scenarioId: string;
  name: string;
  category: 'FINANCIAL' | 'GOVERNANCE' | 'RISK' | 'LEARNING' | 'OPERATIONAL' | 'MULTI_SYSTEM';
  severity: 'MINOR' | 'MODERATE' | 'SEVERE' | 'EXTREME';
  shocks: Array<{
    targetMetric: string;
    shockType: 'PERCENT_DECREASE' | 'PERCENT_INCREASE' | 'FAILURE' | 'REMOVAL';
    magnitude: number;
  }>;
}

/**
 * Invariant Rule Registry (INV-OI53..INV-OI60)
 */
export const SIMULATION_INVARIANTS = {
  INV_OI53: 'Simulation Explainability: Inputs + Assumptions + Transformations + Outputs must be visible',
  INV_OI54: 'Simulation Determinism: 100 Replays = 1 Hash using seeded PRNG (XorShift32)',
  INV_OI55: 'Rollback Availability: Every simulation must include Primary, Fallback, and Rollback Strategy',
  INV_OI56: 'Sandboxed Isolation: Simulations never mutate production state; operate on immutable copies',
  INV_OI57: 'Attribution Integrity: Driver contributions must sum strictly to 100.0%',
  INV_OI58: 'Trace Completeness: Every projected metric must have verified upstream lineage',
  INV_OI59: 'Shock Recoverability: Extreme scenarios must map to certified recovery-states',
  INV_OI60: 'Replay Drift Free: Replay drift tolerance is strictly 0.0000%',
  INV_OI61: 'Portfolio Completeness: Every candidate strategy must be evaluated across all required scenarios',
  INV_OI62: 'Strategy Comparability: All candidate strategies must evaluate against the exact same twin snapshot',
  INV_OI63: 'Portfolio Explainability: Every ranking decision must provide explicit deterministic ranking rationale',
  INV_OI64: 'Edge Confidence Coverage: 100% of trace edges must contain calibrated confidencePct',
  INV_OI65: 'Confidence Calibration: confidencePct must be strictly bounded within [0, 100]',
  INV_OI66: 'Sensitivity Coverage: Every top-level executive metric must provide sensitivity analysis',
} as const;


export interface DependencyDefinition {
  sourceMetric: string;
  targetMetric: string;
  impactWeight: number; // [-1.0, 1.0]
  relationship?: DependencyRelationship;
  confidencePct?: number;
  description?: string;
}

export interface ImpactPath {
  path: string[];
  totalWeight: number;
  steps: Array<{
    source: string;
    target: string;
    weight: number;
  }>;
}

export interface TraceNode {
  nodeId: string;
  metricId: string;
  metricName: string;
  beforeValue: number;
  afterValue: number;
  delta: number;
  simulationId: string;
}

export interface TraceEdge {
  edgeId: string;
  sourceNodeId: string;
  targetNodeId: string;
  contributionPct: number;
  confidencePct?: number;
  sensitivityScore?: number;
}

export interface TraceLedger {
  ledgerId: string;
  simulationId: string;
  records: TraceRecord[];
  summary: {
    totalDrivers: number;
    primaryDriver: string;
    confidencePct: number;
    fullyTraceable: boolean;
  };
}

export interface TraceGraph {
  nodes: TraceNode[];
  edges: TraceEdge[];
}

export interface LineageStep {
  metric: string;
  value: number;
  delta: number;
  upstreamFrom?: string;
  contributionPct?: number;
}

export interface LineageChain {
  targetMetric: string;
  steps: LineageStep[];
  complete: boolean;
  rootMetric: string;
}

export interface TwinState {
  currentSnapshot: OrganizationalSnapshot;
  baselineSnapshot: OrganizationalSnapshot;
  dependencyGraph: DependencyGraph;
  activeScenarios: string[];
  calibrationUtc: string;
  integrityStatus: 'CERTIFIED' | 'DEGRADED' | 'INVALID';
}

export interface TwinMetadata {
  twinId: string;
  version: string;
  calibratedAtUtc: string;
  rootMetricsCount: number;
  dependentMetricsCount: number;
  hash: string;
}

export interface ScenarioDefinition {
  scenarioId: string;
  title: string;
  category: 'INVESTMENT' | 'RISK_SHOCK' | 'GOVERNANCE' | 'CUSTOM';
  description: string;
  parameterChanges: Array<{
    metric: string;
    changePct: number;
    absoluteDelta?: number;
  }>;
  shockType?: 'NONE' | 'MODERATE' | 'SEVERE' | 'EXTREME';
}

export interface SimulationResult {
  simulationId: string;
  scenarioId: string;
  baselineSnapshot: OrganizationalSnapshot;
  projectedState: SimulatedState;
  projectedMetrics: Record<string, {
    baseline: number;
    projected: number;
    delta: number;
    changePct: number;
  }>;
  monteCarlo?: MonteCarloResult;
  waterfallAttribution: Array<{
    driver: string;
    driverName: string;
    contributionPoints: number;
    contributionPct: number;
  }>;
  traceLedger: TraceLedger;
  traceGraph: TraceGraph;
  rollbackStrategy: RollbackStrategy;
  explainabilityCertified: boolean;
  executionDurationMs: number;
  replayHash: string;
}

export const M14_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'M14-Gate-01', name: 'Digital Twin Integrity', requirement: 'Twin state hash and calibration integrity verified fail-closed' },
  { gateId: 'M14-Gate-02', name: 'Snapshot Certification', requirement: 'Immutable organizational baseline snapshots with full state validation' },
  { gateId: 'M14-Gate-03', name: 'Dependency Graph Certification', requirement: 'Acyclic causal dependency graph with DFS cycle detection' },
  { gateId: 'M14-Gate-04', name: 'Traceability Completeness', requirement: 'INV-OI58 satisfied: 100% trace coverage, 0 orphan nodes, fail-closed roots' },
  { gateId: 'M14-Gate-05', name: 'Simulation Explainability', requirement: 'Deterministic backward lineage reconstruction and waterfall attribution' },
  { gateId: 'M14-Gate-06', name: 'Monte Carlo Determinism', requirement: 'INV-OI54/60: 100 replays yield identical hash with 0.0000% drift' },
  { gateId: 'M14-Gate-07', name: 'Shock Test Certification', requirement: 'Stress scenario resilience bounds and degradation limits verified' },
  { gateId: 'M14-Gate-08', name: 'Rollback Plan Coverage', requirement: 'Multi-level rollback strategies (L1-L4) with certified state reversal' },
  { gateId: 'M14-Gate-09', name: 'Executive Sandbox UX', requirement: 'Executive, Analyst, and Audit views conform to Horizon Design System' },
  { gateId: 'M14-Gate-10', name: 'Simulation Platform Certified', requirement: '100% compliance across Invariants INV-OI53..INV-OI60 and static export' },
] as const;


export interface StrategyIntervention {
  targetMetric: string;
  interventionType: 'BUDGET_INCREASE' | 'GOVERNANCE_RULE' | 'RISK_DAMPENING' | 'CAPITAL_ALLOCATION';
  parameterDeltaPct: number;
  costUSD: number;
  implementationWeeks: number;
}

export interface Strategy {
  strategyId: string;
  name: string;
  description: string;
  interventions: StrategyIntervention[];
  assumptions: string[];
  constraints: string[];
}

export interface ScenarioOutcome {
  scenarioType: 'BASELINE' | 'OPTIMISTIC' | 'ADVERSE' | 'STRESS';
  projectedOhi: number;
  projectedRisk: number;
  projectedVelocity: number;
}

export interface StrategyEvaluation {
  strategyId: string;
  strategyName: string;
  projectedOhi: number;
  projectedRisk: number;
  implementationCost: number;
  confidencePct: number;
  robustnessScore: number;
  survivabilityScore: number;
  expectedRoi: number;
  weightedScore: number;
  scenarioOutcomes: ScenarioOutcome[];
  overallRank: number;
  rankingRationale: string;
  rollbackCoveragePct: number;
  recoveryHours: number;
  failureProbabilityPct: number;
}

export interface StrategyPortfolioResult {
  portfolioId: string;
  snapshotId: string;
  evaluations: StrategyEvaluation[];
  topRecommendedStrategyId: string;
  recommendedRationale: string;
  evaluatedAtUtc: string;
  deterministicReplayHash: string;
}

export const M15_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'M15-Gate-01', name: 'Portfolio Completeness', requirement: 'INV-OI61: Every strategy evaluated across Baseline, Optimistic, Adverse, Stress' },
  { gateId: 'M15-Gate-02', name: 'Strategy Comparability', requirement: 'INV-OI62: All strategies evaluated against identical baseline snapshot' },
  { gateId: 'M15-Gate-03', name: 'Portfolio Explainability', requirement: 'INV-OI63: Deterministic decision rationale generated for all rankings' },
  { gateId: 'M15-Gate-04', name: 'Edge Confidence Coverage', requirement: 'INV-OI64: 100% of trace edges contain calibrated confidencePct' },
  { gateId: 'M15-Gate-05', name: 'Confidence Calibration', requirement: 'INV-OI65: confidencePct strictly bounded within [0, 100]' },
  { gateId: 'M15-Gate-06', name: 'Sensitivity Analysis', requirement: 'INV-OI66: Sensitivity leverage score computed for all causal edges' },
  { gateId: 'M15-Gate-07', name: 'Robustness Metric Validation', requirement: 'Robustness score strictly satisfies Mean Outcome / StdDev formula' },
  { gateId: 'M15-Gate-08', name: 'Survivability Scoring', requirement: 'Survivability synthesizes rollback coverage, recovery SLA, and failure risk' },
  { gateId: 'M15-Gate-09', name: 'Strategy Laboratory UX', requirement: 'Horizon Design System compliance across 4 interactive perspective tabs' },
  { gateId: 'M15-Gate-10', name: 'Platform Performance & Bundle Budget', requirement: 'Static export compiles cleanly across 141+ routes with sub-100 kB shared JS' },
] as const;

// -------------------------------------------------------------
// HORIZON 4: ADAPTIVE STRATEGY ORCHESTRATOR & DRIFT CONTRACTS (M17)
// -------------------------------------------------------------

export type SignalCategory = 'ECONOMIC' | 'REGULATORY' | 'MARKET' | 'WORKFORCE';

export interface ExternalSignal {
  signalId: string;
  source: string;
  category: SignalCategory;
  rawValue: number | string;
  normalizedImpact: number; // -100 to +100
  confidencePct: number;
  observedAtUtc: string;
  description: string;
}

export interface NormalizedSignal {
  signalId: string;
  category: SignalCategory;
  normalizedImpact: number; // -100 to +100
  effectiveImpact: number;  // normalizedImpact * (confidencePct / 100)
  confidencePct: number;
  timestampUtc: string;
  source: string;
  affectsMetricId: string;
}

export interface DriftObservation {
  metricId: string;
  metricName: string;
  expectedValue: number;
  actualValue: number;
  driftPct: number; // |Actual - Expected| / Expected * 100
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  thresholdPct: number;
  exceedsThreshold: boolean;
  persistenceDays: number;
  observedAtUtc: string;
}

export interface ImpactWindow {
  metricId: string;
  expectedDaysToImpact: number;
}

export interface DriftRule {
  metricId: string;
  thresholdPct: number;
  minimumDurationDays: number;
}

export interface DriftDecision {
  driftDetected: boolean;
  reoptimizationRequired: boolean;
  maxDriftPct: number;
  criticalMetricCount: number;
  observations: DriftObservation[];
  rootCauses: {
    driver: string;
    impactDelta: number;
    signalId?: string;
    category: 'INTERNAL_METRIC' | 'EXTERNAL_SIGNAL';
    confidencePct: number;
  }[];
  explanation: string[];
  evaluatedAtUtc: string;
}

export interface CalibrationResult {
  metricId: string;
  previousWeight: number;
  calibratedWeight: number;
  weightDelta: number;
  predictionError: number;
  mae: number;
  rmse: number;
  predictionBias: number;
  confidencePct: number;
  sampleSize: number;
  calibratedAtUtc: string;
}

export interface BacktestObservation {
  decisionId: string;
  interventionType: string;
  predictedDelta: number;
  actualDelta: number;
  error: number;
  observedAtUtc: string;
}

export interface QuarterlyStrategyPlan {
  quarter: 'Q1' | 'Q2' | 'Q3' | 'Q4';
  horizonMonths: number;
  primaryStrategyId: string;
  primaryStrategyName: string;
  fallbackStrategyId: string;
  recoveryStrategyId: string;
  projectedOhi: number;
  expectedRoi: number;
  rollbackCoveragePct: number;
  recoveryHours: number;
  status: 'ACTIVE' | 'RECOMMENDED' | 'PLANNED' | 'RESERVE';
}

export interface StrategySequence {
  sequenceId: string;
  name: string;
  timeframeHorizon: '12_MONTHS' | '24_MONTHS';
  quarters: QuarterlyStrategyPlan[];
  overallProjectedOhi: number;
  cumulativeRoi: number;
  averageRobustness: number;
  meanSurvivability: number;
  transitionIntegrityScore: number;
}

export interface OrchestratorState {
  orchestratorId: string;
  activeStrategyId: string;
  activeStrategyName: string;
  activeStatus: 'ON_TRACK' | 'WARNING' | 'DRIFT_DETECTED' | 'REOPTIMIZATION_REQUIRED';
  currentConfidencePct: number;
  expectedOhi: number;
  actualOhi: number;
  driftPct: number;
  portfolioRank: string;
  lastReoptimizedUtc: string;
  deterministicReplayHash: string;
  driftDecision: DriftDecision;
  activeSequence: StrategySequence;
  survivabilityScore: number;
  recoveryHours: number;
  rollbackCoveragePct: number;
  failureProbabilityPct: number;
}

// -------------------------------------------------------------
// HORIZON 4 INVARIANTS (INV-OI67 through INV-OI74)
// -------------------------------------------------------------

export interface SimulationInvariant {
  invariantId: string;
  name: string;
  formalRule: string;
  severity: 'HIGH' | 'CRITICAL';
  certificationThreshold: number;
  enforcedBy: string;
}

export const INV_OI67: SimulationInvariant = {
  invariantId: 'INV-OI67',
  name: 'Strategy Transition Integrity',
  formalRule: 'StrategyTransition(A -> B) => TraceabilityPreserved && AuditLogged && RollbackCoverage >= 90.0%',
  severity: 'CRITICAL',
  certificationThreshold: 90.0,
  enforcedBy: 'adaptiveStrategyOrchestrator.ts',
};

export const INV_OI68: SimulationInvariant = {
  invariantId: 'INV-OI68',
  name: 'Portfolio Evolution Coverage',
  formalRule: 'forall q in {Q1, Q2, Q3, Q4}: Primary(q) != null && Fallback(q) != null && Recovery(q) != null',
  severity: 'CRITICAL',
  certificationThreshold: 100.0,
  enforcedBy: 'adaptiveStrategyOrchestrator.ts',
};

export const INV_OI69: SimulationInvariant = {
  invariantId: 'INV-OI69',
  name: 'Adaptive Re-Optimization Trigger',
  formalRule: 'MaterialShockDetected || DriftRequiresReopt => ReEvaluatePortfolio() && ZeroSilentSuppression',
  severity: 'CRITICAL',
  certificationThreshold: 100.0,
  enforcedBy: 'adaptiveStrategyOrchestrator.ts',
};

export const INV_OI70: SimulationInvariant = {
  invariantId: 'INV-OI70',
  name: 'Strategy Drift Detection',
  formalRule: 'DriftPct > Threshold && DurationDays >= MinDuration => SetFlag(REOPTIMIZATION_REQUIRED)',
  severity: 'CRITICAL',
  certificationThreshold: 100.0,
  enforcedBy: 'strategyDriftEngine.ts',
};

export const INV_OI71: SimulationInvariant = {
  invariantId: 'INV-OI71',
  name: 'Model Calibration Accuracy',
  formalRule: 'Backtest(Models) => MAE_calibrated <= MAE_initial && PredictionBiasBounded',
  severity: 'HIGH',
  certificationThreshold: 85.0,
  enforcedBy: 'modelCalibrationEngine.ts',
};

export const INV_OI72: SimulationInvariant = {
  invariantId: 'INV-OI72',
  name: 'External Signal Integrity',
  formalRule: 'forall s in Signals: Timestamped(s) && Sourced(s) && NormalizedRange[-100, 100](s) && Confidence(s) in [0, 100]',
  severity: 'CRITICAL',
  certificationThreshold: 100.0,
  enforcedBy: 'externalSignalEngine.ts',
};

export const INV_OI73: SimulationInvariant = {
  invariantId: 'INV-OI73',
  name: 'Re-Optimization Explainability',
  formalRule: 'StrategyReplaced(A -> B) => RootCausesCount >= 1 && RationaleLength > 0',
  severity: 'CRITICAL',
  certificationThreshold: 100.0,
  enforcedBy: 'strategyDriftEngine.ts',
};

export const INV_OI74: SimulationInvariant = {
  invariantId: 'INV-OI74',
  name: 'Signal-to-Outcome Traceability',
  formalRule: 'SignalAffectsRecommendation(s) => CausalTraceContainsNode(s.signalId)',
  severity: 'CRITICAL',
  certificationThreshold: 100.0,
  enforcedBy: 'externalSignalEngine.ts',
};

export const M17_INVARIANTS = [
  INV_OI67,
  INV_OI68,
  INV_OI69,
  INV_OI70,
  INV_OI71,
  INV_OI72,
  INV_OI73,
  INV_OI74,
] as const;

export const M17_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'M17-Gate-01', name: 'Strategy Transition Integrity', requirement: 'INV-OI67: Strategy changes preserve traceability, auditability, and >= 90% rollback coverage' },
  { gateId: 'M17-Gate-02', name: 'Portfolio Evolution Coverage', requirement: 'INV-OI68: Every quarter (Q1-Q4) defines Primary, Fallback, and Recovery plans' },
  { gateId: 'M17-Gate-03', name: 'Adaptive Re-Optimization Trigger', requirement: 'INV-OI69: Material shocks & persistent drift trigger automated re-evaluation' },
  { gateId: 'M17-Gate-04', name: 'Strategy Drift Detection', requirement: 'INV-OI70: Multi-metric drift calculated with metric-specific thresholds and persistence' },
  { gateId: 'M17-Gate-05', name: 'Model Calibration Accuracy', requirement: 'INV-OI71: Historical backtesting calibrates edge weights and confidence with reduced MAE' },
  { gateId: 'M17-Gate-06', name: 'External Signal Integrity', requirement: 'INV-OI72: External signals sourced, normalized [-100, 100], and confidence-scored' },
  { gateId: 'M17-Gate-07', name: 'Re-Optimization Explainability', requirement: 'INV-OI73: Strategy updates provide explicit root-cause attribution breakdown' },
  { gateId: 'M17-Gate-08', name: 'Signal-to-Outcome Traceability', requirement: 'INV-OI74: External signals injected as upstream root nodes in causal trace graph' },
  { gateId: 'M17-Gate-09', name: 'Strategy Orchestrator UX', requirement: 'Horizon Design System compliance across 6 cockpit zones and 3 perspective views' },
  { gateId: 'M17-Gate-10', name: 'Platform Performance & Bundle Budget', requirement: 'Static export compiles cleanly across 142 routes with sub-100 kB shared JS' },
] as const;
