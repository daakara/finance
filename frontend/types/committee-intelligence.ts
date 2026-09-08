/**
 * ARX Terminal vNext - Committee Intelligence & Governance Contracts
 * Source: docs/architecture/COMMITTEE_INTELLIGENCE_ARCHITECTURE.md
 */

export type CommitteeRole = "VIEWER" | "ANALYST" | "PORTFOLIO_MANAGER" | "CIO" | "ADMIN";
export const CommitteeRole = {
  VIEWER: "VIEWER" as const,
  ANALYST: "ANALYST" as const,
  PORTFOLIO_MANAGER: "PORTFOLIO_MANAGER" as const,
  CIO: "CIO" as const,
  ADMIN: "ADMIN" as const,
};

export type BaselineScope = "PERSONAL" | "TEAM" | "COMMITTEE";
export const BaselineScope = {
  PERSONAL: "PERSONAL" as const,
  TEAM: "TEAM" as const,
  COMMITTEE: "COMMITTEE" as const,
};

export interface SharedBaseline {
  baselineId: string;
  scope: BaselineScope;
  ticker: string;
  createdAt: string;
  createdBy: string;
  setupScore: number;
  executionState: string;
  marketRegime: string;
  flowZScore: number;
  validationTier: string;
  snapshotHash: string;
  version: number;
}

export interface CommitteeBaseline {
  baselineId: string;
  ticker: string;
  acknowledgedBy: string;
  approvedBy?: string;
  committeeId: string;
  snapshotHash: string;
  acknowledgedAt: string;
  status: "PENDING" | "ACTIVE" | "SUPERSEDED";
}

export type AcknowledgementStatus = "PENDING" | "ACKNOWLEDGED" | "DISAGREED" | "ESCALATED";
export const AcknowledgementStatus = {
  PENDING: "PENDING" as const,
  ACKNOWLEDGED: "ACKNOWLEDGED" as const,
  DISAGREED: "DISAGREED" as const,
  ESCALATED: "ESCALATED" as const,
};

export interface CommitteeAcknowledgement {
  acknowledgementId: string;
  ticker: string;
  deltaId: string;
  userId: string;
  status: AcknowledgementStatus;
  rationale?: string;
  acknowledgedAt: string;
}

export interface CommitteeConsensus {
  ticker: string;
  totalReviewers: number;
  acknowledged: number;
  disagreed: number;
  pending: number;
  escalated: number;
  consensusPercent: number;
}

export type ConflictSeverity = "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
export const ConflictSeverity = {
  LOW: "LOW" as const,
  MODERATE: "MODERATE" as const,
  HIGH: "HIGH" as const,
  CRITICAL: "CRITICAL" as const,
};

export type ConflictCategory = "THESIS" | "EXECUTION" | "REGIME" | "RISK" | "GOVERNANCE";
export const ConflictCategory = {
  THESIS: "THESIS" as const,
  EXECUTION: "EXECUTION" as const,
  REGIME: "REGIME" as const,
  RISK: "RISK" as const,
  GOVERNANCE: "GOVERNANCE" as const,
};

export interface ConflictPosition {
  userId: string;
  recommendation: "BUY" | "HOLD" | "AVOID";
  rationale: string;
  submittedAt: string;
}

export interface ConflictRecord {
  conflictId: string;
  ticker: string;
  category: ConflictCategory;
  severity: ConflictSeverity;
  createdAt: string;
  participants: string[];
  positions: ConflictPosition[];
  status: "OPEN" | "RESOLVED" | "OVERRIDDEN";
}

export interface BaselineConflict {
  conflictId: string;
  ticker: string;
  committeeId: string;
  baselineA: string;
  baselineB: string;
  resolvedBy?: string;
  resolution: "LATEST_ACCEPTED" | "REVIEW_REQUIRED";
  conflictCreated: boolean;
  timestamp: string;
}

export type AuditAction =
  | "BASELINE_CREATED"
  | "BASELINE_UPDATED"
  | "BASELINE_ACKNOWLEDGED"
  | "BASELINE_APPROVED"
  | "BASELINE_SUPERSEDED"
  | "BASELINE_CONFLICT_RESOLVED"
  | "REVIEW_STARTED"
  | "REVIEW_CLOSED"
  | "NOTE_ADDED";

export interface AuditEvent {
  eventId: string;
  committeeId: string;
  ticker: string;
  actorId: string;
  actorRole: CommitteeRole;
  action: AuditAction;
  timestamp: string;
  previousHash: string;
  eventHash: string;
  metadata: Record<string, unknown>;
}

export interface CommitteeFeedItem {
  id: string;
  ticker: string;
  severity: "CRITICAL" | "MAJOR" | "MINOR";
  headline: string;
  category: ConflictCategory;
  requiresAction: boolean;
  createdAt: string;
}

/**
 * ============================================================================
 * Phase 31-M1: Committee Intelligence Foundations Contracts (Epic AI-001)
 * ============================================================================
 */

export type CommitteeDecisionStatus =
  | 'PROPOSED'
  | 'UNDER_REVIEW'
  | 'APPROVED'
  | 'REJECTED'
  | 'SUPERSEDED';

export type DissentSeverity =
  | 'LOW'
  | 'MEDIUM'
  | 'HIGH'
  | 'MATERIAL';

export interface CommitteeParticipant {
  userId: string;
  role: string;
  votingEligible: boolean;
}

export interface CommitteeDissent {
  dissentId: string;
  decisionId: string;
  authorId: string;
  severity: DissentSeverity;
  alternativeRecommendation: string;
  riskAssessment: string;
  evidenceIds: string[];
  acceptedForReview: boolean;
  timestampUtc: string;
}

export interface CommitteeDecision {
  committeeId: string;
  decisionId: string;
  proposalId: string;
  title: string;
  participants: CommitteeParticipant[];
  evidenceIds: string[];
  dissents: CommitteeDissent[];
  finalDecision: string;
  status: CommitteeDecisionStatus;
  materialDecision: boolean;
  evidenceLinked: boolean;
  participantsRecorded: boolean;
  outcomeLinked: boolean;
  attributionLinked: boolean;
  dissentRecorded: boolean;
  alternativeViewPresent?: boolean;
  riskAssessmentPresent?: boolean;
  dissentEvidenceLinked?: boolean;
  outcomeId?: string;
  decisionQuality?: number;
  timestampUtc: string;
}

export interface CommitteeHealth {
  committeeId: string;
  committeeName: string;
  cdqi: number;
  committeeODEI: number;
  committeeDIRatio: number;
  dissentCoveragePct: number;
  learningVelocityPct: number;
  governanceCompliancePct: number;
  transparencyCoveragePct: number;
}

export interface CommitteeDecisionTrace {
  decisionId: string;
  proposalId: string;
  evidenceLinked: boolean;
  participantsRecorded: boolean;
  outcomeLinked: boolean;
  attributionLinked: boolean;
  traceabilityScore: number;
}

export interface CommitteeNetworkNode {
  committeeId: string;
  committeeName: string;
  decisionCount: number;
  qualityScore: number;
}

export interface CommitteeNetworkEdge {
  sourceCommitteeId: string;
  targetCommitteeId: string;
  sharedDecisionCount: number;
  influenceScore: number;
}

export interface CommitteeIntelligenceDashboard {
  committeeODEI: number;
  committeeDIRatio: number;
  dissentUtilizationRate: number;
  committeeCount: number;
  governanceCompliancePct: number;
}

export type CIIGateId =
  | 'CII-Gate-01'
  | 'CII-Gate-02'
  | 'CII-Gate-03'
  | 'CII-Gate-04'
  | 'CII-Gate-05'
  | 'CII-Gate-06'
  | 'CII-Gate-07'
  | 'CII-Gate-08'
  | 'CII-Gate-09'
  | 'CII-Gate-10';

export interface CIIGateEvaluation {
  gateId: CIIGateId;
  name: string;
  status: 'PASS' | 'FAIL';
  actualValue: string | number;
  targetValue: string | number;
  rationale: string;
}

export interface CommitteeCertificationResult {
  certified: boolean;
  gates: CIIGateEvaluation[];
  totalAssertions: number;
  passedAssertions: number;
  failedAssertions: number;
  oi13Violations: number;
  oi14Violations: number;
  certificationStatus: 'PASS' | 'FAIL';
}

/**
 * ============================================================================
 * Phase 31-M1.1: Adversarial Hardening, Replay & Audit Reconstruction Contracts
 * ============================================================================
 */

export interface CommitteeProposal {
  proposalId: string;
  title: string;
  createdBy: string;
  createdAtUtc: string;
  businessObjective: string;
}

export interface CommitteeEvidence {
  evidenceId: string;
  sourceType: string;
  sourceReference: string;
  confidencePct: number;
}

export interface CommitteeOutcome {
  outcomeId: string;
  realizedValueDollars: number;
  outcomeQualityScore: number;
  measuredAtUtc: string;
}

export interface CommitteeAttribution {
  attributionId: string;
  decisionId: string;
  capabilityIds: string[];
  learningIds: string[];
  individualContributionPct: number;
  teamContributionPct: number;
  committeeContributionPct: number;
  systemContributionPct: number;
  totalContributionPct: number;
}

export interface ReconstructionCoverage {
  proposalRecovered: boolean;
  evidenceRecovered: boolean;
  participantsRecovered: boolean;
  dissentsRecovered: boolean;
  outcomeRecovered: boolean;
  attributionRecovered: boolean;
  completenessPct: number;
}

export interface CommitteeAuditSnapshot {
  snapshotId: string;
  committeeId: string;
  decisionId: string;
  capturedAtUtc: string;
  hash: string;
  proposalHash: string;
  evidenceHashes: string[];
  participantHashes: string[];
  dissentHashes: string[];
  outcomeHash?: string;
  attributionHash?: string;
  reconstructionVersion: string;
}

export interface AuditReconstructionRequest {
  decisionId: string;
  requestedBy: string;
  requestedAtUtc: string;
  includeDissents: boolean;
  includeEvidence: boolean;
  includeAttribution: boolean;
}

export interface AuditReconstructionResult {
  decisionId: string;
  success: boolean;
  coverage: ReconstructionCoverage;
  proposal?: CommitteeProposal;
  evidence?: CommitteeEvidence[];
  participants?: CommitteeParticipant[];
  dissents?: CommitteeDissent[];
  outcome?: CommitteeOutcome;
  attribution?: CommitteeAttribution;
  missingArtifacts: string[];
  elapsedMs: number;
}

export interface ReplayExpectedResults {
  transparencyCoveragePct: number;
  dissentCoveragePct: number;
  committeeODEI: number;
  committeeDIRatio: number;
  oi13Violations: number;
  oi14Violations: number;
  certificationStatus: 'PASS' | 'FAIL';
}

export interface ReplayFixture {
  fixtureId: string;
  fixtureVersion: string;
  createdAtUtc: string;
  committees: CommitteeHealth[];
  decisions: CommitteeDecision[];
  expectedResults: ReplayExpectedResults;
  sha256?: string;
}

export interface ReplayComparisonResult {
  matchesExpected: boolean;
  deterministic: boolean;
  expectedHash: string;
  actualHash: string;
  mismatchedFields: string[];
  mismatchDetails?: DeepComparisonMismatch[];
}

export interface ReplayDeterminismResult {
  deterministic: boolean;
  iterations: number;
  uniqueHashes: number;
  canonicalHash: string;
  failures: string[];
}

export interface DeepComparisonMismatch {
  path: string;
  expected: unknown;
  actual: unknown;
  reason:
    | 'VALUE_MISMATCH'
    | 'TYPE_MISMATCH'
    | 'MISSING_PROPERTY'
    | 'EXTRA_PROPERTY'
    | 'ARRAY_LENGTH_MISMATCH'
    | 'NAN_DETECTED'
    | 'CYCLE_MISMATCH';
}

export interface DeepComparisonResult {
  equal: boolean;
  mismatches: DeepComparisonMismatch[];
}

export type CorruptionType =
  | 'MISSING_PROPOSAL'
  | 'MISSING_EVIDENCE'
  | 'MISSING_PARTICIPANTS'
  | 'MISSING_OUTCOME'
  | 'MISSING_ATTRIBUTION'
  | 'MISSING_DISSENT'
  | 'MISSING_ALTERNATIVE_VIEW'
  | 'MISSING_RISK_ASSESSMENT'
  | 'MISSING_DISSENT_EVIDENCE'
  | 'DUPLICATE_MEMBER'
  | 'NO_VOTING_MEMBER'
  | 'NO_CHAIR'
  | 'INVALID_NETWORK_EDGE'
  | 'SELF_REFERENTIAL_EDGE';

export interface CorruptionFixture {
  fixtureId: string;
  corruptionType: CorruptionType;
  expectedInvariantViolation: 'INV-OI13' | 'INV-OI14' | 'MEMBERSHIP' | 'NETWORK';
  expectedDetection: boolean;
  payload: Record<string, unknown>;
}

export interface CorruptionDetectionResult {
  fixtureId: string;
  detected: boolean;
  violatedInvariant: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  remediation: string;
}

export interface StressProfile {
  profileId: string;
  committeeCount: number;
  decisionCount: number;
  participantCount: number;
  evidenceCount: number;
  dissentCount: number;
  targetDurationMs: number;
}

export interface StressTestResult {
  profileId: string;
  completed: boolean;
  elapsedMs: number;
  memoryMb: number;
  certificationDriftDetected: boolean;
  invariantViolations: number;
  numericalFailures: number;
}

export interface StressAggregationResult {
  fixturesExecuted: number;
  passed: number;
  failed: number;
  averageExecutionMs: number;
  maxExecutionMs: number;
  replayFailures: number;
  numericalFailures: number;
  reliabilityScore: number;
  stabilityScore: number;
  certificationStatus: 'PASS' | 'FAIL';
}
