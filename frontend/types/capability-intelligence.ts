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

export interface CapabilityImpactMetrics {
  behaviorImpactScore: number; // 0 - 100
  outcomeImpactScore: number;  // 0 - 100
  valueImpactScore: number;    // 0 - 100
  adoptionImpactScore: number; // 0 - 100
  cii: number;                 // Capability Impact Index (0 - 100)
}

export interface CapabilityEconomics {
  valueGeneratedDollars: number;
  operationalCostDollars: number;
  cie: number;                 // Capability Impact Efficiency = Value / Cost (e.g. 4.25x)
  capitalPreservedDollars: number;
  excessReturnContributionPct: number;
  confidencePct: number;
  sampleSize: number;
}

export interface CapabilityNode {
  id: string;
  name: string;
  category: 'ANALYTICS' | 'EXECUTION' | 'GOVERNANCE' | 'COACHING' | 'SIMULATION';
  status: CapabilityLifecycleStatus;
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
