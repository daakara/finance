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
