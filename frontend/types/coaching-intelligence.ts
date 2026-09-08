/**
 * Phase 31-M5: Collective Intelligence Coach & Prescriptive Intelligence Data Contracts
 *
 * Implements:
 * - Coaching recommendations, evidence attribution, and remediation actions
 * - Cognitive bias alerts and fairness monitors
 * - Intervention plans and sequential workflows
 * - Outcome measurement and Coach Impact Ratio metrics
 * - Non-coercive human override records
 * - Typed API errors and JSON Schema validation models
 * - Invariants INV-OI23 through INV-OI32
 */

export type CoachingCategory =
  | 'GROUPTHINK'
  | 'LEARNING'
  | 'GOVERNANCE'
  | 'KNOWLEDGE_TRANSFER'
  | 'RISK'
  | 'OPERATIONAL';

export type CoachingPriority = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export type RecommendationStatus =
  | 'PROPOSED'
  | 'ACTIVE'
  | 'ACCEPTED'
  | 'IN_PROGRESS'
  | 'COMPLETED'
  | 'REJECTED'
  | 'EXPIRED';

export type CoachingRecommendationType =
  | 'BIAS_WARNING'
  | 'DISSENT_PROMPT'
  | 'ALTERNATIVE_OPTION'
  | 'LEARNING_NUDGE'
  | 'RISK_REVIEW';

export type MetricSourceType =
  | 'ODEI'
  | 'CDQI'
  | 'GROUPTHINK'
  | 'FRICTION'
  | 'TRANSFER_RATE'
  | 'DIRATIO';

export interface RecommendationEvidence {
  evidenceId: string;
  sourceMetric: MetricSourceType;
  observedValue: number;
  thresholdValue: number;
  contributionPct: number;
  explanation: string;
}

export type ActionLifecycleStatus = 'OPEN' | 'IN_PROGRESS' | 'CLOSED';

export interface RemediationAction {
  actionId: string;
  title: string;
  ownerId: string;
  dueDateUtc: string;
  status: ActionLifecycleStatus;
  expectedBenefit: string;
}

export interface ExpectedImpact {
  projectedODEIDelta: number;
  projectedRiskReduction: number;
  confidencePct: number;
  timeframeHorizon: '30D' | '90D' | '180D';
}

export interface AlternativeRecommendation {
  alternativeId: string;
  title: string;
  approach: string;
  tradeOffSummary: string;
  confidenceScore: number;
}

export interface CoachingRecommendation {
  recommendationId: string;
  committeeId: string;
  decisionId?: string;
  type?: CoachingRecommendationType;
  title: string;
  description: string;
  category: CoachingCategory;
  priority: CoachingPriority;
  confidenceScore: number; // 0-100
  confidencePct?: number; // alias
  status: RecommendationStatus;
  ownerCommitteeId: string;
  rationale: string;
  supportingEvidence: RecommendationEvidence[];
  evidenceIds?: string[];
  historicalCaseIds?: string[];
  actions: RemediationAction[];
  expectedImpact: ExpectedImpact;
  expectedBenefit?: string;
  alternatives?: AlternativeRecommendation[];
  createdAtUtc: string;
  generatedAtUtc?: string;
}

export type BiasType =
  | 'CONFIRMATION'
  | 'AUTHORITY'
  | 'RECENCY'
  | 'GROUPTHINK'
  | 'ANCHORING'
  | 'MONOCULTURE'
  | 'FAVORITISM'
  | 'OWNER_CONCENTRATION';

export type BiasSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface BiasAlert {
  alertId: string;
  committeeId: string;
  biasType: BiasType;
  riskScore: number; // 0-100
  severity: BiasSeverity;
  evidenceIds: string[];
  explanation: string;
  detectedAtUtc: string;
}

export interface HumanOverride {
  overrideId: string;
  recommendationId: string;
  userId: string;
  reason: string;
  overriddenAtUtc: string;
}

export type OutcomeStatus = 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';

export interface RecommendationOutcome {
  recommendationId: string;
  measuredAtUtc: string;
  baselineODEI: number;
  currentODEI: number;
  baselineGroupthinkScore: number;
  currentGroupthinkScore: number;
  improvementPct: number;
  outcomeStatus: OutcomeStatus;
  attributionConfidence: number; // 0-1
}

export interface InterventionPlan {
  planId: string;
  committeeId: string;
  generatedAtUtc: string;
  title: string;
  recommendations: CoachingRecommendation[];
  totalRiskReduction: number;
  estimatedCompletionDays: number;
  expectedOutcomeScore: number;
  primaryOwnerId: string;
}

export interface RecommendationEffectiveness {
  recommendationId: string;
  accepted: boolean;
  implemented: boolean;
  measurableImprovement: boolean;
  improvementScore: number;
  attributionConfidence: number;
}

export interface CoachingEffectiveness {
  recommendationFamily: string;
  issuedCount: number;
  acceptedCount: number;
  improvedOutcomeCount: number;
  degradedOutcomeCount: number;
  impactRatio: number;
}

export interface RecommendationEvaluation {
  recommendationId: string;
  isValid: boolean;
  explainabilityScore: number;
  actionabilityScore: number;
  nonCoercive: boolean;
  violations: string[];
}

export interface RecommendationExplanation {
  recommendationId: string;
  title: string;
  rationale: string;
  supportingEvidence: RecommendationEvidence[];
  expectedImpact: ExpectedImpact;
  historicalCases: { caseId: string; outcomeSummary: string; relevanceScore: number }[];
  attributionCompletenessPct: number;
}

export interface CertificationResult {
  certified: boolean;
  gateScores: Record<string, boolean>;
  invariantViolations: string[];
  timestampUtc: string;
}

// Typed API Error contracts
export interface ApiErrorResponse {
  errorCode: string;
  errorType: string;
  message: string;
  details?: Record<string, unknown>;
  correlationId: string;
  timestampUtc: string;
}

export interface ValidationErrorResponse extends ApiErrorResponse {
  errorType: 'VALIDATION_ERROR';
  fieldErrors: {
    field: string;
    reason: string;
    value?: unknown;
  }[];
}

export interface BiasViolationErrorResponse extends ApiErrorResponse {
  errorType: 'BIAS_VIOLATION';
  invariant: 'INV-OI28' | 'INV-OI29' | 'INV-OI30' | 'INV-OI31' | 'INV-OI32';
  severity: BiasSeverity;
  affectedEntities: string[];
}

export interface CertificationFailureResponse extends ApiErrorResponse {
  errorType: 'CERTIFICATION_FAILURE';
  failedGate: string;
  invariantViolations: string[];
}
