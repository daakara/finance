/**
 * Phase 30: Capability Intelligence Type Definitions
 *
 * Epics:
 * - CI-100: Capability Graph & Dependency Engine
 * - CI-200: Capability Attribution Economics
 * - CI-300: Self-Optimizing Platform & Retirement Model
 * - CI-400: Executive Intelligence for Product Strategy
 * - CI-500: Autonomous Intelligence Governance
 *
 * Invariants:
 * - INV-CI1 (Value Attribution Integrity): Sum(Attributed Value) <= Actual Realized Value
 * - INV-CI2 (Capability Dependency Completeness): All prerequisites verified
 */

export type CapabilityLifecycleStatus =
  | 'EXPERIMENTAL'
  | 'PILOT'
  | 'PROTECTED'
  | 'CORE'
  | 'RETIREMENT_REVIEW'
  | 'SUNSET';

export type InvestmentAction =
  | 'INVEST_MORE'
  | 'MAINTAIN'
  | 'OPTIMIZE'
  | 'REDESIGN'
  | 'RETIRE';

export interface CapabilityDependencyEdge {
  sourceCapabilityId: string;
  targetCapabilityId: string;
  dependencyType: 'PREREQUISITE' | 'ENHANCER' | 'FEEDBACK_LOOP';
  criticality: 'CRITICAL' | 'OPTIONAL';
}

export type CapabilityPortfolioCluster =
  | 'PORTFOLIO_A_CORE'
  | 'PORTFOLIO_B_GROWTH'
  | 'PORTFOLIO_C_GOVERNANCE'
  | 'PORTFOLIO_D_RETIREMENT';

export interface CapabilityImpactMetrics {
  behaviorImpactScore: number; // 0 - 100
  outcomeImpactScore: number;  // 0 - 100
  valueImpactScore: number;    // 0 - 100
  adoptionImpactScore: number; // 0 - 100
  cii: number;                 // Capability Impact Index (0 - 100)
  cae: number;                 // Capability Adoption Efficiency = Behavior Impact / Adoption Rate
  sms: number;                 // Capability Strategic Moat Score (0 - 100)
}

export interface CapabilityEconomics {
  valueGeneratedDollars: number;
  operationalCostDollars: number;
  cie: number;                 // Capability Impact Efficiency = Value / Cost (e.g. 4.25x)
  cvd: number;                 // Capability Value Density = Value / Active Users (e.g. $201.52/user)
  capitalPreservedDollars: number;
  excessReturnContributionPct: number;
  confidencePct: number;
  sampleSize: number;
  valueHistory: number[];      // Historical value tracking across review periods
}

export interface CapabilityNode {
  id: string;
  name: string;
  category: 'ANALYTICS' | 'EXECUTION' | 'GOVERNANCE' | 'COACHING' | 'SIMULATION';
  portfolio: CapabilityPortfolioCluster;
  status: CapabilityLifecycleStatus;
  monitoringEnabled: boolean;
  metrics: CapabilityImpactMetrics;
  economics: CapabilityEconomics;
  dependencies: string[]; // List of required capability IDs
  recommendedAction: InvestmentAction;
  actionRationale: string;
}

export interface CapabilityDependencyGraph {
  nodes: CapabilityNode[];
  edges: CapabilityDependencyEdge[];
  totalCapabilities: number;
  highValueCount: number; // Top 20%
  underperformingCount: number;
  retirementCandidatesCount: number;
}

export interface ValueAttributionIntegrityResult {
  isSatisfied: boolean;
  totalAttributedValueDollars: number;
  actualRealizedValueDollars: number;
  discrepancyDollars: number;
  inflationRatio: number; // Must be <= 1.00
  details: string;
}

// ---------------------------------------------------------------------------
// INV-OI12: Capability Value Decay Detection & ODEI Confidence Model
// ---------------------------------------------------------------------------

export interface CapabilityValueDecayResult {
  capabilityId: string;
  capabilityName: string;
  isDecayViolated: boolean;
  decayVariancePct: number;
  consecutiveDecliningPeriods: number;
  requiresRetirementReview: boolean;
  alertLevel: 'HEALTHY' | 'WARNING' | 'CRITICAL';
  recommendedRemediation: string;
}

export interface ODEIConfidenceResult {
  odeiScore: number;
  confidenceScore: number;
  confidenceBand: { lower: number; upper: number };
  sampleSize: number;
  observationWindowDays: number;
  benchmarkPopulation: number;
  status: 'CONFIRMED' | 'INSUFFICIENT_DATA' | 'DEGRADED';
}

// ---------------------------------------------------------------------------
// Phase 30 Certification Gates (CI-Gate-01 through CI-Gate-09)
// ---------------------------------------------------------------------------

export interface Phase30CertificationGate {
  gateId: string; // CI-Gate-01 to CI-Gate-09
  gateName: string;
  target: string;
  actual: string;
  status: 'PASS' | 'FAIL';
  details: string;
}

export interface Phase30CertificationResult {
  status: 'CERTIFIED' | 'RELEASE_CANDIDATE' | 'NOT_READY';
  overallScore: number;
  gates: Phase30CertificationGate[];
  certifiedAt: string;
  releaseTrain: 'PHASE_30';
}

