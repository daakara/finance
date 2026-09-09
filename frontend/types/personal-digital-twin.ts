/**
 * Personal Digital Twin & Life Operating System Contracts (Horizon 5)
 *
 * Core models for:
 * - Personal Capacity (Time, Money, Energy, Attention) & INV-OI75-P
 * - Weekly Time Budget (168-hour ceiling, sleep floors)
 * - Energy & Attention States
 * - Personal Drift Tracking (Career, Finance, Health, Learning)
 * - Future Recovery Projections & INV-OI83-P
 */

export interface PersonalCapacity {
  weeklyHours: number;           // Free discretionary + development hours per week
  monthlyBudget: number;         // Discretionary budget per month ($ or €)
  energyCapacity: number;        // 0 to 100 physiological/psychological energy score
  attentionCapacity: number;     // 0 to 100 cognitive attention units
}

export interface WeeklyTimeBudget {
  totalHours: 168;               // Immutable physical constant
  sleepHours: number;            // Must be >= 49 (7h/night floor for INV-OI75-P)
  workHours: number;             // Employment or baseline commitments
  commuteHours: number;          // Transit time
  familyHours: number;           // Family, caretaking & core obligations
  exerciseHours: number;         // Physical fitness maintenance
  adminHours: number;            // Chores, food prep, hygiene
  discretionaryHours: number;    // Calculated: 168 - sum(all above)
}

export interface EnergyState {
  energyScore: number;           // 0 to 100 overall vitality
  sleepQuality: number;          // 0 to 100 restorative quality
  recoveryScore: number;         // 0 to 100 autonomic balance (HRV proxy)
  stressLoad: number;            // 0 to 100 allostatic load
}

export interface AttentionState {
  focusCapacity: number;         // 0 to 100 uninterrupted deep work capacity
  contextSwitches: number;       // Average daily interruptions/meetings
  cognitiveLoad: number;         // 0 to 100 ongoing mental queue load
}

export interface FinancialCapacity {
  monthlyDisposableIncome: number; // Discretionary cash flow
  emergencyFundMonths: number;     // Months of survival runway
  investmentCapacity: number;      // Monthly capital available for growth
}

export interface PersonalDemand {
  weeklyHours: number;
  monthlyBudget: number;
  energyDemand: number;
  attentionDemand: number;
}

export type CapacityViolationType =
  | "TIME_CAPACITY_EXCEEDED"
  | "SLEEP_FLOOR_VIOLATION"
  | "MONEY_CAPACITY_EXCEEDED"
  | "ENERGY_CAPACITY_EXCEEDED"
  | "ATTENTION_CAPACITY_EXCEEDED";

export interface CapacityCheckResult {
  isFeasible: boolean;
  violations: CapacityViolationType[];
  utilization: {
    timePct: number;
    moneyPct: number;
    energyPct: number;
    attentionPct: number;
  };
  rebalanceSuggestions: Array<{
    domain: string;
    suggestedReductionHours: number;
    reason: string;
  }>;
}

export interface AllocationDomain {
  id: string;
  name: string;
  hours: number;
  impactScore: number;
  color: string;
  description: string;
}

export interface PersonalAllocationPlan {
  domains: Record<string, number>; // domainId -> hours
  projectedLhiCurrent: number;
  projectedLhi6m: number;
  projectedLhi12m: number;
  confidencePct: number;
}

export interface PersonalDriftCard {
  domain: "CAREER" | "FINANCE" | "HEALTH" | "LEARNING";
  metric: string;
  expected: number;
  actual: number;
  driftPct: number;
  severity: "LOW" | "MEDIUM" | "HIGH";
  unit: string;
  recommendation: string;
  waterfallCauses: Array<{
    cause: string;
    impact: number; // negative number for drag
  }>;
}

export interface RecoveryStrategy {
  strategyId: string;
  name: string;
  type: "TIME_REALLOCATION" | "ACCELERATOR" | "COACH_MENTOR" | "COMBINED";
  description: string;
  weeklyHoursRequired: number;
  monthlyCost: number;
  energyDemand: number;
  attentionDemand: number;
  projectedLhi: number;
  recoveryTimeMonths: number;
  recoveryVelocity: number; // Gap points closed per month
  confidencePct: number;
  isFeasible: boolean; // INV-OI83-P checked against capacity
  violationReason?: string;
  traceabilityLineage: Array<{
    step: string;
    delta: string;
  }>;
}

export interface RecoveryProjection {
  projectionId: string;
  gapName: string;
  currentLhi: number;
  baselineProjectedLhi: number;
  driftLevel: "MINOR" | "MODERATE" | "MAJOR" | "CRITICAL";
  strategies: RecoveryStrategy[];
  recommendedStrategyId: string;
  monteCarloSimulations: {
    runs: number;
    p10Months: number;
    p50Months: number;
    p90Months: number;
    confidencePct: number;
  };
  waterfallImpact: Array<{
    lever: string;
    lhiContribution: number;
  }>;
}

export interface LifeCommandSummary {
  lifeHealthIndex: number;
  lhiDelta30d: number;
  trajectoryStatus: "ON_TRACK" | "DRIFT_DETECTED" | "CRITICAL_DRIFT";
  topGoals: Array<{
    id: string;
    title: string;
    progressPct: number;
    targetDate: string;
    status: "ON_TRACK" | "AT_RISK";
  }>;
  currentRisks: Array<{
    id: string;
    title: string;
    severity: "LOW" | "MEDIUM" | "HIGH";
    impactArea: string;
  }>;
  dailyActions: Array<{
    id: string;
    title: string;
    timeCommitmentHours: number;
    focusDomain: string;
    expectedImpact: string;
    confidencePct: number;
  }>;
  futureSelfProjection: {
    currentRole: string;
    months6: { role: string; probability: number };
    months12: { role: string; probability: number };
    months24: { role: string; probability: number };
  };
}

// =========================================================================
// HORIZON 6: PERSONAL DATA INTEGRATION & REAL-WORLD SIGNAL LAYER
// =========================================================================

export type SignalCategory = "TIME" | "HEALTH" | "FINANCE" | "CAREER" | "LEARNING";

export interface SignalMetadata {
  source: string;              // e.g. "GOOGLE_CALENDAR", "APPLE_HEALTH", "PLAID"
  observedAtUtc: string;       // ISO 8601 timestamp
  confidencePct: number;       // 0 to 100
  freshnessHours: number;      // Calculated: (now - observedAtUtc) in hours
  maxAllowedAgeHours: number;  // Invariant ceiling for category
}

export interface PersonalSignal {
  signalId: string;
  category: SignalCategory;
  metricId: string;            // e.g. "DEEP_WORK_HOURS", "SLEEP_DURATION", "SAVINGS_RATE"
  value: number;
  unit: string;
  metadata: SignalMetadata;
}

export type SignalErrorCode =
  | "SIGNAL_STALE"
  | "SIGNAL_SOURCE_UNKNOWN"
  | "SIGNAL_TIMESTAMP_MISSING";

export interface SignalFreshnessResult {
  isFresh: boolean;
  freshnessScore: number;      // 0 to 100 calculated from decay formula
  ageHours: number;
  maxAgeHours: number;
  status: "FRESH" | "AGING" | "STALE";
  errorCode?: SignalErrorCode;
}

export interface SignalReliability {
  source: string;
  confidencePct: number;
  priority: number;            // Lower number = higher authoritative rank (1 is highest)
}

export type ConflictResolutionMethod = "PRIORITY" | "WEIGHTED" | "USER_OVERRIDE";

export interface SignalConflict {
  conflictId: string;
  metricId: string;
  sourceA: string;
  valueA: number;
  confidenceA: number;
  sourceB: string;
  valueB: number;
  confidenceB: number;
  resolutionMethod: ConflictResolutionMethod;
  resolvedValue: number;
  auditReason: string;
  resolvedAtUtc: string;
}

export interface DecisionOutcome {
  decisionId: string;
  recommendationTitle: string;
  category: SignalCategory;
  expectedMetricGain: number;
  actualMetricGain: number;
  calibrationDeltaPct: number;  // (actual - expected) / expected * 100
  decidedAtUtc: string;
  outcomeObservedAtUtc: string;
  brierScoreContribution: number;
}

export interface SignalQualityComposite {
  freshnessScore: number;      // Average freshness across all active signals (0-100)
  coverageScore: number;       // Breadth across all 5 life categories (0-100)
  confidenceScore: number;     // Average source confidence (0-100)
  overallQualityScore: number; // (Freshness + Coverage + Confidence) / 3
  status: "EXCELLENT" | "GOOD" | "DEGRADED" | "CRITICAL";
}

export interface ConnectedSource {
  id: string;
  name: string;
  category: SignalCategory;
  provider: string;
  status: "SYNCED" | "SYNCING" | "ERROR" | "DISCONNECTED";
  lastSyncUtc: string;
  freshnessScore: number;
  confidencePct: number;
  totalSignalsTracked: number;
}


/**
 * Horizon 7: Integrated Life Twin & Unified Life Causal Graph Contracts
 */
export type LifeDomainType =
  | "HEALTH"
  | "CAREER"
  | "LEARNING"
  | "FINANCE"
  | "RELATIONSHIPS"
  | "TIME";

export interface LifeDomainNode {
  id: string;                  // e.g. "SLEEP_HOURS", "RECOVERY_SCORE", "SAVINGS_RATE"
  name: string;
  domain: LifeDomainType;
  baselineValue: number;
  unit: string;
  currentValue: number;
  minSafeValue: number;
  maxSafeValue: number;
  description: string;
}

export interface LifeEdge {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  sensitivity: number;         // Rate of change multiplier (+ or -)
  latencyWeeks: number;        // Time lag for propagation
  confidencePct: number;
  mechanism: string;           // Biological, financial, or cognitive explanation
}

export interface UnifiedLifeGraph {
  nodes: Record<string, LifeDomainNode>;
  edges: LifeEdge[];
  topologicalOrder: string[];  // Computed cycle-free topological order
  isAcyclic: boolean;
}

export interface CrossDomainScenario {
  id: string;
  title: string;
  description: string;
  leverChanges: Record<string, number>; // nodeId -> delta
  horizonWeeks: number;
  simulatedNodeDeltas: Record<string, number>;
  lhiDelta: number;
  domainContributions: Record<LifeDomainType, number>;
  unintendedConsequences: string[];
  isPlausible: boolean;
}

export interface CrossDomainTraceNode {
  step: number;
  nodeId: string;
  nodeName: string;
  domain: LifeDomainType;
  priorValue: number;
  newValue: number;
  delta: number;
  causedByEdgeId?: string;
  mechanism?: string;
  latencyWeeksCumulative: number;
}

export interface CrossDomainSimulationResult {
  scenarioId: string;
  scenarioTitle: string;
  baselineLhi: number;
  projectedLhi: number;
  lhiDelta: number;
  domainScores: Record<LifeDomainType, { baseline: number; projected: number; delta: number }>;
  traceLineage: CrossDomainTraceNode[];
  isTraceable: boolean;        // INV-OI88-P
  isConsistent: boolean;       // INV-OI89-P
  consistencyViolations: string[];
  monteCarloDistribution: {
    p10: number;
    p50: number;
    p90: number;
    iterations: number;
  };
}


/**
 * Horizon 8: Relational Twins & Household Intelligence Contracts
 */
export type RelationshipRole =
  | "PARTNER"
  | "CHILD"
  | "PARENT"
  | "FRIEND"
  | "COFOUNDER";

export interface RelationshipNode {
  relationshipId: string;
  name: string;
  role: RelationshipRole;
  relevanceWeight: number;        // 0.0 to 1.0 (e.g. Partner: 0.35, Child: 0.15)
  impactSensitivity: number;      // Collateral multiplier (0.5 to 2.0)
  sharedResources: string[];      // e.g. ["EVENING_HOURS", "WEEKEND_BLOCKS", "HOUSEHOLD_BUDGET"]
  baselineWellbeing: number;      // 0 to 100
  currentWellbeing: number;       // 0 to 100
  sentimentAlignment: number;     // 0 to 100 agreement on strategic life trajectory
  description: string;
}

export type SharedResourceType =
  | "TIME_BLOCK"
  | "FINANCIAL_CAPITAL"
  | "CARETAKING_DUTY";

export interface CommitmentAllocation {
  commitmentId: string;
  allocatedTo: string;          // e.g. "MY_WORK", "PARTNER_STUDY", "FAMILY_DINNER"
  amount: number;
  timeWindow?: string;          // e.g. "FRI_1900_2100" (for calendar collision detection)
  priority: number;             // 1 = High, 3 = Flexible
}

export interface SharedResourceAllocation {
  resourceId: string;
  name: string;
  type: SharedResourceType;
  capacityUnits: number;        // e.g. 10 hours/week or $2000/month
  unit: string;
  allocatedCommitments: CommitmentAllocation[];
  totalAllocated: number;
  isDoubleBooked: boolean;      // Triggered by INV-OI90-P
  conflictDetails: string[];
}

export interface HouseholdHealthIndex {
  compositeHhi: number;         // 0 to 100
  personalLhi: number;          // 0 to 100
  relationalWeightedScore: number;
  householdFinancialRunway: number; // months
  sharedTimeFrictionPenalty: number;
  delta: number;
  trajectory: "STABLE" | "EXPANDING" | "STRAINED" | "CRITICAL";
}

export interface RelationalMemberImpact {
  relationshipId: string;
  name: string;
  role: RelationshipRole;
  wellbeingPrior: number;
  wellbeingNew: number;
  delta: number;
  keyConcerns: string[];
}

export interface RelationalScenarioResult {
  scenarioId: string;
  scenarioTitle: string;
  personalLhiDelta: number;
  householdHhiDelta: number;
  memberImpacts: RelationalMemberImpact[];
  isSharedResourceIntegral: boolean;  // INV-OI90-P
  sharedResourceViolations: string[];
  isRelationalImpactVisible: boolean; // INV-OI91-P
  visibilityViolations: string[];
  relationalVisibilityCard: {
    personalImpactSummary: string;
    financialImpactSummary: string;
    relationalImpactSummary: string;
    unvarnishedTradeOffs: string[];
  };
}


/**
 * Horizon 9: Multi-Year Household Trajectory Sequencing, Compounding & Copilot Contracts
 */
export interface TrajectoryStep {
  stepId: string;
  year: number;                 // 1, 2, 3...
  interventionId: string;
  title: string;
  personalLhiImpact: number;
  householdHhiImpact: number;
  annualCapitalDelta: number;
  discretionaryHoursDelta: number;
  stateAfterStep: Record<string, number>;
}

export interface HouseholdTrajectory {
  trajectoryId: string;
  name: string;
  description: string;
  steps: TrajectoryStep[];
  projectedLhi: number;
  projectedHhi: number;
  confidencePct: number;
  totalCost: number;
  riskScore: number;            // 0 to 100 (lower is better)
  optionalityScore: number;     // 0 to 100 (future feasible opportunity sets)
  recoveryScore: number;        // 0 to 100
  multiCriteriaScore: number;   // 0 to 100 composite ranking score
  stabilityScore: number;       // expectedOutcome / variance (INV-OI93-P)
  isParetoOptimal: boolean;
  unintendedConsequences: string[];
}

export interface ExplainableRecommendation {
  recommendationId: string;
  title: string;
  whyNow: string;
  triggerSignals: string[];
  tracePath: string[];
  expectedLhiImpact: number;
  expectedHhiImpact: number;
  confidencePct: number;
  p10: number;
  p50: number;
  p90: number;
  stabilityScore: number;
  timestampUtc: string;
}

export interface RippleMapNode {
  id: string;
  label: string;
  domain: string;
  magnitude: number;           // Sizing
  delta: number;
  unit: string;
}

export interface RippleMapEdge {
  id: string;
  from: string;
  to: string;
  impact: number;              // Width
  isPositive: boolean;         // Polarity
  latencyWeeks: number;
}

export interface RippleMapData {
  rootIntervention: string;
  nodes: RippleMapNode[];
  edges: RippleMapEdge[];
  netHhiDelta: number;
  netLhiDelta: number;
}
