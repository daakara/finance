/**
 * Phase 31-M3: Organizational Learning Intelligence Contracts
 *
 * Implements data models for:
 * - Learning Record & Publication Repository (Epic M3-101)
 * - Team Learning Velocity & INV-OI17 Telemetry (Epic M3-102)
 * - Cross-Committee Knowledge Transfer Network & INV-OI18 (Epic M3-103)
 * - Learning Friction Engine Diagnostics (Epic M3-104)
 * - Multi-Signal Incident Correlation & Fatigue Controls (CORR-01..05, FAT-01..06)
 */

export type LearningCategory =
  | 'PROCESS'
  | 'RISK'
  | 'ALLOCATION'
  | 'GOVERNANCE'
  | 'STRATEGY';

export type LearningStatus =
  | 'DRAFT'
  | 'PUBLISHED'
  | 'RETIRED';

export type AdoptionStatus =
  | 'PENDING'
  | 'ADOPTED'
  | 'REJECTED'
  | 'EXPIRED';

export type VelocityStatus =
  | 'POSITIVE'
  | 'STAGNANT'
  | 'DEGRADING';

export type LearningFrictionCategory =
  | 'IGNORED'
  | 'REJECTED'
  | 'EXPIRED'
  | 'UNKNOWN'
  | 'OWNERSHIP_GAP'
  | 'GOVERNANCE_GAP';

export type CorrelatedIncidentType =
  | 'ORGANIZATIONAL_LEARNING_BREAKDOWN'
  | 'AUDIT_INTEGRITY_INCIDENT'
  | 'DETERMINISM_FAILURE'
  | 'NETWORK_GOVERNANCE_INCIDENT';

export type IncidentSeverity =
  | 'LOW'
  | 'MEDIUM'
  | 'HIGH'
  | 'CRITICAL';

export type IncidentLifecycleStatus =
  | 'OPEN'
  | 'INVESTIGATING'
  | 'MITIGATING'
  | 'RESOLVED'
  | 'CLOSED';

export interface LearningRecord {
  learningId: string;
  title: string;
  description: string;
  sourceCommitteeId: string;
  sourceDecisionId: string;
  sourceOutcomeId?: string;
  category: LearningCategory;
  publishedAtUtc: string;
  status: LearningStatus;
  authorId: string;
  tags?: string[];
  evidenceReference?: string;
  expectedOdeiImpact?: number;
}

export interface LearningAdoption {
  adoptionId: string;
  learningId: string;
  sourceCommitteeId: string;
  targetCommitteeId: string;
  adoptionStatus: AdoptionStatus;
  adoptedAtUtc?: string;
  justification?: string;
  reviewingUserId?: string;
  targetDecisionId?: string;
}

export interface LearningVelocityResult {
  committeeId: string;
  baselineODEI: number;
  currentODEI: number;
  elapsedQuarters: number;
  velocity: number;
  status: VelocityStatus;
  annualizedVelocity: number;
  forecastNextQuarter: number;
  invariantSatisfied: boolean;
  attributableLearnings: string[];
  historicalQuarterlyVelocities: {
    quarter: string;
    velocity: number;
    odei: number;
  }[];
}

export interface KnowledgeTransferEdge {
  sourceCommitteeId: string;
  targetCommitteeId: string;
  publishedLearnings: number;
  adoptedLearnings: number;
  transferRatePct: number;
  velocityImpact: number;
  status: 'COMPLIANT' | 'BREACH';
  learningIds: string[];
}

export interface LearningFrictionItem {
  learningId: string;
  targetCommitteeId: string;
  category: LearningFrictionCategory;
  explanation: string;
  daysPending: number;
}

export interface LearningFrictionResult {
  committeeId: string;
  totalEvaluated: number;
  ignoredCount: number;
  rejectedCount: number;
  expiredCount: number;
  unknownCount: number;
  ownershipGapCount: number;
  governanceGapCount: number;
  frictionScore: number;
  topFrictionCategory: LearningFrictionCategory;
  items: LearningFrictionItem[];
}

export interface CorrelatedIncident {
  incidentId: string;
  incidentType: CorrelatedIncidentType;
  severity: IncidentSeverity;
  sourceAlerts: string[];
  occurrenceCount: number;
  status: IncidentLifecycleStatus;
  firstSeenAtUtc: string;
  lastSeenAtUtc: string;
  rootCauseHypothesis: string;
  affectedCommitteeIds: string[];
  slaDeadlineUtc: string;
  resolutionNotes?: string;
  resolvedAtUtc?: string;
}

export interface IncidentFilterCriteria {
  type?: CorrelatedIncidentType | 'ALL';
  severity?: IncidentSeverity | 'ALL';
  status?: IncidentLifecycleStatus | 'ALL';
}
