/**
 * Phase 31-M7: Optimization Intelligence Contracts
 *
 * Implements:
 * - Optimization Objectives, Resource Pools & Constraints
 * - Intervention Candidates & Monte Carlo Simulation Results
 * - Resource Allocation & Knapsack Recommendation Contracts
 * - Optimization Run Lifecycle & Explainability Lineage
 * - Typed Error Envelopes & Fail-Close Error Codes (OPT-FAIL-01..08)
 * - Automated Recovery & Idempotent CSC Workflows (OPT-REC-01..05)
 * - Invariants INV-OI39 through INV-OI44
 */

import { ApiErrorResponse } from './coaching-intelligence';

export type OptimizationDirection = 'MAXIMIZE' | 'MINIMIZE';

export interface OptimizationObjective {
  objectiveId: string;
  name: string;
  description: string;
  weight: number;
  direction: OptimizationDirection;
  targetMetric: string;
}

export type ResourceType = 'BUDGET' | 'HEADCOUNT' | 'TIME' | 'TRAINING' | 'GOVERNANCE';

export interface ResourcePool {
  resourceId: string;
  resourceType: ResourceType;
  availableUnits: number;
  allocatedUnits: number;
  unitLabel: string;
}

export type ConstraintType = 'BUDGET' | 'CAPACITY' | 'RISK' | 'CERTIFICATION' | 'GOVERNANCE';

export interface OptimizationConstraint {
  constraintId: string;
  name: string;
  type: ConstraintType;
  value: number;
  mandatory: boolean;
  operator?: '<=' | '>=' | '==' | '<' | '>';
  currentValue?: number;
  satisfied?: boolean;
}

export interface InterventionCandidate {
  interventionId: string;
  title: string;
  category: 'GOVERNANCE' | 'LEARNING' | 'RISK' | 'COACHING' | 'OPERATIONAL';
  targetCommitteeId: string;
  cost: number;
  effortHours: number;
  headcountRequired: number;
  expectedRiskReduction: number;
  expectedOHIImprovement: number;
  expectedLearningVelocityGain: number;
  expectedTransferRateGain: number;
  probabilityOfSuccess: number;
  ownerId?: string;
  timelineWeeks?: number;
}

export interface InterventionSimulationResult {
  simulationId: string;
  interventionId: string;
  iterations: number;
  expectedOHI: number;
  expectedRiskScore: number;
  expectedLearningVelocity: number;
  confidenceLowerBound: number;
  confidenceUpperBound: number;
  recommendationRank: number;
  stabilityIndex: 'HIGH' | 'MEDIUM' | 'LOW';
  variance: number;
}

export type TargetArea = 'LEARNING' | 'RISK' | 'GOVERNANCE' | 'COACHING' | 'NETWORK';

export interface ResourceAllocationRecommendation {
  allocationId: string;
  committeeId: string;
  resourceId: string;
  targetArea: TargetArea;
  allocatedUnits: number;
  allocationPct: number;
  expectedBenefit: string;
  marginalBenefit: number;
  priorityScore: number;
  confidencePct: number;
  rationale: string;
}

export type OptimizationRunStatus = 'PENDING' | 'RUNNING' | 'FAILED' | 'COMPLETED' | 'CERTIFIED';

export type OptimizationType =
  | 'ORGANIZATIONAL_HEALTH'
  | 'RISK_REDUCTION'
  | 'RESOURCE_ALLOCATION'
  | 'INTERVENTION_PRIORITIZATION'
  | 'CAPACITY_PLANNING';

export type OptimizationErrorCode =
  | 'NO_FEASIBLE_SOLUTION'
  | 'CONSTRAINT_CONTRADICTION'
  | 'MISSING_OPTIMIZATION_INPUT'
  | 'INVALID_OPTIMIZATION_DRIVER'
  | 'NON_FINITE_OBJECTIVE_FUNCTION'
  | 'INCOMPLETE_OPTIMIZATION_ATTRIBUTION'
  | 'OPTIMIZATION_REPLAY_DRIFT'
  | 'CAPACITY_EXCEEDED'
  | 'FAIRNESS_VIOLATION'
  | 'ALLOCATION_SUM_MISMATCH';

export interface APIError {
  errorCode: OptimizationErrorCode | string;
  message: string;
  severity: 'INFO' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  field?: string;
  remediation?: string;
}

export interface DriverContribution {
  driverId: string;
  driverName: string;
  contributionPct: number;
  sourceMetric: string;
  evidenceIds: string[];
}

export interface OptimizationRun {
  runId: string;
  status: OptimizationRunStatus;
  optimizationType: OptimizationType;
  startedAtUtc: string;
  completedAtUtc?: string;
  objectiveFunction: string;
  objectiveScore?: number;
  replayHash?: string;
  iterationCount: number;
  constraintCount: number;
  warnings: string[];
  errors: APIError[];
  allocationResultId?: string;
  objectives?: OptimizationObjective[];
  constraints?: OptimizationConstraint[];
  allocations?: ResourceAllocationRecommendation[];
  projectedOHI?: number;
  projectedRiskReduction?: number;
  certificationStatus?: 'PASS' | 'FAIL';
}

export interface AllocationResult {
  allocationResultId: string;
  runId: string;
  certified: boolean;
  totalAllocationPct: number;
  generatedAtUtc: string;
  overallHealthGainPct: number;
  projectedRiskReductionPct: number;
  projectedLearningGainPct: number;
  allocations: ResourceAllocationRecommendation[];
  driverContributions: DriverContribution[];
  certificationStatus: 'PENDING' | 'PASSED' | 'FAILED';
  allocatedBudget: number;
  totalAvailableBudget: number;
  allocatedHeadcount: number;
  totalAvailableHeadcount: number;
}

export interface OptimizationPlanRequest {
  committeeId: string;
  objective: 'IMPROVE_ODEI' | 'REDUCE_GROUPTHINK' | 'REDUCE_FRICTION' | 'IMPROVE_TRANSFER' | 'INCREASE_GOVERNANCE_HEALTH';
  forecastPeriodDays: number;
}

export interface OptimizationPlanResponse {
  optimizationId: string;
  planId: string;
  expectedImprovement: number;
  estimatedRiskReduction: number;
  recommendations: string[];
  status: 'GENERATED';
}

export interface OptimizationExecutionResponse {
  executionId: string;
  planId: string;
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  startedAtUtc: string;
}

export interface OptimizationValidationError extends ApiErrorResponse {
  errorType: 'OPTIMIZATION_VALIDATION_ERROR';
  optimizationId?: string;
  fieldErrors: {
    field: string;
    constraint: string;
  }[];
}

export interface RecoveryConflictError extends ApiErrorResponse {
  errorType: 'RECOVERY_CONFLICT';
  recoveryId: string;
  existingStatus: 'QUEUED' | 'IN_PROGRESS';
}

export interface IdempotencyViolationError extends ApiErrorResponse {
  errorType: 'IDEMPOTENCY_VIOLATION';
  idempotencyKey: string;
  originalRequestId: string;
}

export interface SensitivityScenario {
  scenarioId: string;
  name: string;
  perturbationPct: number;
  resultingOHI: number;
  resultingRisk: number;
  isStable: boolean;
}

export interface TradeoffPoint {
  candidateId: string;
  name: string;
  cost: number;
  ohiGain: number;
  riskReduction: number;
  efficiencyRatio: number;
  isParetoOptimal: boolean;
}
