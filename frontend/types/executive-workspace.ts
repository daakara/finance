/**
 * Phase 31-M13: Executive Productivity & Decision Acceleration (ARX Horizon Executive OS) Contracts
 *
 * Implements:
 * - Executive Workspace Profile, Assigned Committees & Personal Task Queue
 * - Unified Decision Inbox, Multi-Source Deduplication & Concurrency Guard
 * - One-Click Executive Briefing Package & Telemetry Lineage
 * - Edge-Case Error Contracts (WS-EC-01..07, DI-EC-01..08, BRF-EC-01..08)
 * - M13 Gate Traceability Matrix (M13-Gate-01 through M13-Gate-10)
 */

export type ExecutiveUserRole =
  | 'CHIEF_INVESTMENT_OFFICER'
  | 'CHIEF_RISK_OFFICER'
  | 'BOARD_DIRECTOR'
  | 'COMMITTEE_CHAIR'
  | 'AUDIT_PARTNER';

export interface AssignedCommittee {
  committeeId: string;
  name: string;
  role: string;
  activeDecisions: number;
  healthScore: number;
  lastActivityUtc: string;
}

export interface ExecutiveTask {
  taskId: string;
  title: string;
  category: 'APPROVAL' | 'ESCALATION' | 'REVIEW' | 'RUNBOOK';
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  committeeId: string;
  slaRemainingHours: number;
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED';
}

export interface WorkspaceProfile {
  userId: string;
  name: string;
  role: ExecutiveUserRole;
  assignedCommittees: AssignedCommittee[];
  ownedTasks: ExecutiveTask[];
  activeRisksCount: number;
  pendingApprovalsCount: number;
  lastSnapshotTimestampUtc: string;
  isDegraded: boolean;
  telemetryFreshnessSlaMinutes: number;
}

export type DecisionCategory =
  | 'APPROVAL'
  | 'ESCALATION'
  | 'RECOMMENDATION'
  | 'RUNBOOK'
  | 'OPTIMIZATION';

export interface DecisionInboxItem {
  itemId: string;
  title: string;
  description: string;
  category: DecisionCategory;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';
  sourceCenter: string;
  entityId: string;
  owner: string;
  slaTargetMinutes: number;
  createdAtUtc: string;
  duplicateSources?: string[];
  status: 'PENDING' | 'EXECUTING' | 'RESOLVED' | 'QUARANTINED';
  lockedBy?: string;
}

export type BriefingAudience =
  | 'EXECUTIVE'
  | 'BOARD'
  | 'COMMITTEE'
  | 'INCIDENT';

export interface BriefingFinding {
  findingId: string;
  text: string;
  category: string;
  telemetrySource: string;
  metricValue: number;
  benchmarkFloor: number;
  supported: boolean; // Must be backed by evidence (BRF-EC-03)
}

export interface BriefingPackage {
  briefingId: string;
  audience: BriefingAudience;
  title: string;
  generatedAtUtc: string;
  status: 'COMPLETE' | 'PARTIAL' | 'FALLBACK';
  headline: string;
  executiveSummary: string;
  findings: BriefingFinding[];
  recommendations: string[];
  replayHash: string;
  lineaged: boolean;
}

// -------------------------------------------------------------
// EDGE-CASE ERROR CONTRACTS
// -------------------------------------------------------------

export interface ProductivityError {
  errorCode: string;
  errorType: string;
  message: string;
  entityId?: string;
  correlationId: string;
  timestampUtc: string;
}

export interface UnauthorizedCommitteeAccessError extends ProductivityError {
  errorType: 'UNAUTHORIZED_COMMITTEE_ACCESS';
  committeeId: string;
}

export interface WorkspaceConsistencyError extends ProductivityError {
  errorType: 'WORKSPACE_CONSISTENCY_DRIFT';
  workspaceMetric: number;
  canonicalSourceMetric: number;
}

export interface AuditUnavailableError extends ProductivityError {
  errorType: 'AUDIT_UNAVAILABLE_FAIL_CLOSE';
  attemptedActionId: string;
}

export interface UnsupportedFindingError extends ProductivityError {
  errorType: 'UNSUPPORTED_FINDING_EXCLUSION';
  findingText: string;
}

// -------------------------------------------------------------
// CANONICAL FIXTURES
// -------------------------------------------------------------

export const CANONICAL_WORKSPACE_PROFILES: Record<ExecutiveUserRole, WorkspaceProfile> = {
  CHIEF_INVESTMENT_OFFICER: {
    userId: 'USR-CIO-001',
    name: 'Alexandra Vance (CIO)',
    role: 'CHIEF_INVESTMENT_OFFICER',
    assignedCommittees: [
      { committeeId: 'COM-001', name: 'Investment Committee', role: 'Chair', activeDecisions: 3, healthScore: 86.4, lastActivityUtc: '2026-09-08T18:00:00Z' },
      { committeeId: 'COM-004', name: 'Strategic Capital Committee', role: 'Executive Member', activeDecisions: 2, healthScore: 84.1, lastActivityUtc: '2026-09-08T16:30:00Z' },
    ],
    ownedTasks: [
      { taskId: 'TSK-01', title: 'Ratify Q3 Tech Allocation Tranche', category: 'APPROVAL', priority: 'CRITICAL', committeeId: 'COM-001', slaRemainingHours: 2, status: 'PENDING' },
      { taskId: 'TSK-02', title: 'Review Autonomous Portfolio Rebalance', category: 'RUNBOOK', priority: 'HIGH', committeeId: 'COM-004', slaRemainingHours: 6, status: 'PENDING' },
      { taskId: 'TSK-03', title: 'Minority Dissent Review on Liquid Buffer', category: 'REVIEW', priority: 'MEDIUM', committeeId: 'COM-001', slaRemainingHours: 18, status: 'PENDING' },
    ],
    activeRisksCount: 2,
    pendingApprovalsCount: 3,
    lastSnapshotTimestampUtc: '2026-09-08T20:00:00Z',
    isDegraded: false,
    telemetryFreshnessSlaMinutes: 30,
  },
  CHIEF_RISK_OFFICER: {
    userId: 'USR-CRO-002',
    name: 'Marcus Sterling (CRO)',
    role: 'CHIEF_RISK_OFFICER',
    assignedCommittees: [
      { committeeId: 'COM-002', name: 'Risk Oversight Board', role: 'Chair', activeDecisions: 4, healthScore: 88.0, lastActivityUtc: '2026-09-08T19:15:00Z' },
      { committeeId: 'COM-001', name: 'Investment Committee', role: 'Risk Delegate', activeDecisions: 1, healthScore: 86.4, lastActivityUtc: '2026-09-08T18:00:00Z' },
    ],
    ownedTasks: [
      { taskId: 'TSK-04', title: 'Macro Shock VaR Threshold Investigation', category: 'ESCALATION', priority: 'CRITICAL', committeeId: 'COM-002', slaRemainingHours: 1, status: 'PENDING' },
      { taskId: 'TSK-05', title: 'Audit Committee Counterparty Exposure', category: 'REVIEW', priority: 'HIGH', committeeId: 'COM-002', slaRemainingHours: 8, status: 'PENDING' },
    ],
    activeRisksCount: 4,
    pendingApprovalsCount: 2,
    lastSnapshotTimestampUtc: '2026-09-08T20:00:00Z',
    isDegraded: false,
    telemetryFreshnessSlaMinutes: 30,
  },
  BOARD_DIRECTOR: {
    userId: 'USR-BOD-003',
    name: 'Helena Thorne (Lead Independent Director)',
    role: 'BOARD_DIRECTOR',
    assignedCommittees: [
      { committeeId: 'COM-003', name: 'Audit & Governance Committee', role: 'Audit Lead', activeDecisions: 2, healthScore: 92.5, lastActivityUtc: '2026-09-08T17:45:00Z' },
    ],
    ownedTasks: [
      { taskId: 'TSK-06', title: 'Annual Model Governance Attestation', category: 'APPROVAL', priority: 'HIGH', committeeId: 'COM-003', slaRemainingHours: 24, status: 'PENDING' },
    ],
    activeRisksCount: 1,
    pendingApprovalsCount: 1,
    lastSnapshotTimestampUtc: '2026-09-08T20:00:00Z',
    isDegraded: false,
    telemetryFreshnessSlaMinutes: 60,
  },
  COMMITTEE_CHAIR: {
    userId: 'USR-CHR-004',
    name: 'David Chen (Committee Chair)',
    role: 'COMMITTEE_CHAIR',
    assignedCommittees: [
      { committeeId: 'COM-004', name: 'Strategic Capital Committee', role: 'Chair', activeDecisions: 3, healthScore: 84.1, lastActivityUtc: '2026-09-08T16:30:00Z' },
    ],
    ownedTasks: [
      { taskId: 'TSK-07', title: 'Finalize Tranche Authorization Minutes', category: 'REVIEW', priority: 'MEDIUM', committeeId: 'COM-004', slaRemainingHours: 12, status: 'PENDING' },
    ],
    activeRisksCount: 2,
    pendingApprovalsCount: 2,
    lastSnapshotTimestampUtc: '2026-09-08T20:00:00Z',
    isDegraded: false,
    telemetryFreshnessSlaMinutes: 30,
  },
  AUDIT_PARTNER: {
    userId: 'USR-AUD-005',
    name: 'Sarah Jenkins (Audit Lead)',
    role: 'AUDIT_PARTNER',
    assignedCommittees: [
      { committeeId: 'COM-003', name: 'Audit & Governance Committee', role: 'Senior Auditor', activeDecisions: 1, healthScore: 92.5, lastActivityUtc: '2026-09-08T17:45:00Z' },
    ],
    ownedTasks: [
      { taskId: 'TSK-08', title: 'Deterministic Replay Verification Sign-Off', category: 'REVIEW', priority: 'HIGH', committeeId: 'COM-003', slaRemainingHours: 4, status: 'PENDING' },
    ],
    activeRisksCount: 0,
    pendingApprovalsCount: 1,
    lastSnapshotTimestampUtc: '2026-09-08T20:00:00Z',
    isDegraded: false,
    telemetryFreshnessSlaMinutes: 30,
  },
};

export const M13_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'M13-Gate-01', name: 'Workspace Personalization', requirement: 'Correct attribution of committees, risks & tasks (WS-01, WS-02)' },
  { gateId: 'M13-Gate-02', name: 'Workspace Outage Resilience', requirement: 'Graceful degradation to certified snapshots upon telemetry loss (WS-03, WS-EC-03)' },
  { gateId: 'M13-Gate-03', name: 'Decision Inbox Aggregation', requirement: 'Multi-source decision aggregation into one unified triage queue (DI-01)' },
  { gateId: 'M13-Gate-04', name: 'Deterministic Priority & SLA', requirement: 'Severity-first and SLA expiration ranking with oscillation damping (DI-02, DI-EC-02)' },
  { gateId: 'M13-Gate-05', name: 'Concurrency & Audit Fail-Close', requirement: 'Deduplication, simultaneous execution lock & audit failure blocking (DI-EC-01..06)' },
  { gateId: 'M13-Gate-06', name: 'One-Click Briefing Synthesis', requirement: 'Instant generation across 4 audiences (Executive, Board, Committee, Incident)' },
  { gateId: 'M13-Gate-07', name: 'Briefing Lineage & Attribution', requirement: '100% evidence lineage; unsupported finding prevention (BRF-02, BRF-EC-03)' },
  { gateId: 'M13-Gate-08', name: 'Narrative Replay Determinism', requirement: '100 Replays of canonical briefing = 1 unique SHA-256 hash (BRF-03, BRF-EC-04)' },
  { gateId: 'M13-Gate-09', name: 'End-to-End Executive Journey', requirement: 'Workspace -> Inbox -> Execution -> Briefing seamless lifecycle (E2E-UX-01)' },
  { gateId: 'M13-Gate-10', name: 'Master Performance & Bundle', requirement: 'First Load JS Shared <= 100.0 kB, build clean, 0 regressions' },
];
