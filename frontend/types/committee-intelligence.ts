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
