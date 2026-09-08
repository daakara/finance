/**
 * Phase 31-M13: Executive Workspace Engine (ARX Horizon Executive OS)
 *
 * Implements:
 * - Role-based executive workspace profiles (CIO, CRO, BOD, Chair, Auditor)
 * - Telemetry outage fallback & certified snapshot recovery (WS-03, WS-EC-03)
 * - Unauthorized committee access verification (WS-EC-05)
 * - Workspace consistency verification against canonical sources (WS-EC-06)
 * - Workload pagination & state hash generation (WS-EC-02, WS-EC-07)
 */

import {
  ExecutiveUserRole,
  WorkspaceProfile,
  ExecutiveTask,
  CANONICAL_WORKSPACE_PROFILES,
  UnauthorizedCommitteeAccessError,
  WorkspaceConsistencyError,
} from '@/types/executive-workspace';
import { sha256Hex } from '@/lib/governance/sha256';

export interface WorkspaceEngineOptions {
  simulateTelemetryOutage?: boolean;
  snapshotAgeMinutes?: number;
}

export function getWorkspaceProfile(
  role: ExecutiveUserRole = 'CHIEF_INVESTMENT_OFFICER',
  options?: WorkspaceEngineOptions
): WorkspaceProfile {
  const baseProfile = CANONICAL_WORKSPACE_PROFILES[role] ?? CANONICAL_WORKSPACE_PROFILES.CHIEF_INVESTMENT_OFFICER;

  // Deep clone to prevent mutation
  const profile: WorkspaceProfile = JSON.parse(JSON.stringify(baseProfile));

  if (options?.simulateTelemetryOutage) {
    profile.isDegraded = true;
    profile.lastSnapshotTimestampUtc = '2026-09-08T18:00:00Z (CERTIFIED_SNAPSHOT_FALLBACK)';
  } else if (options?.snapshotAgeMinutes && options.snapshotAgeMinutes > profile.telemetryFreshnessSlaMinutes) {
    profile.isDegraded = true;
  }

  return profile;
}

export function resolveWorkspaceProfile(
  userId: string,
  options?: WorkspaceEngineOptions
): WorkspaceProfile {
  const matched = Object.values(CANONICAL_WORKSPACE_PROFILES).find((p) => p.userId === userId);
  const role = matched ? matched.role : 'CHIEF_INVESTMENT_OFFICER';
  return getWorkspaceProfile(role, options);
}

/**
 * Invariant WS-EC-05: Unauthorized Committee Access Check
 * Blocks access to committee intelligence if user is not an assigned member.
 */
export function checkCommitteeAccess(
  profile: WorkspaceProfile,
  committeeId: string
): { granted: boolean; error?: UnauthorizedCommitteeAccessError } {
  const isAssigned = profile.assignedCommittees.some((c) => c.committeeId === committeeId);
  if (!isAssigned) {
    const error: UnauthorizedCommitteeAccessError = {
      errorCode: 'WS-ERR-001',
      errorType: 'UNAUTHORIZED_COMMITTEE_ACCESS',
      committeeId,
      message: `User ${profile.userId} (${profile.name}) is not authorized to access committee ${committeeId}. Access blocked fail-closed.`,
      correlationId: `CORR-AUTH-${Date.now()}`,
      timestampUtc: new Date().toISOString(),
    };
    return { granted: false, error };
  }
  return { granted: true };
}

/**
 * Invariant WS-EC-06: Workspace Consistency Verification
 * Detects drift between workspace metric aggregations and canonical telemetry.
 */
export function verifyWorkspaceConsistency(
  profile: WorkspaceProfile,
  canonicalSourceMetric: number
): { consistent: boolean; error?: WorkspaceConsistencyError } {
  if (profile.pendingApprovalsCount !== canonicalSourceMetric) {
    const error: WorkspaceConsistencyError = {
      errorCode: 'WS-ERR-002',
      errorType: 'WORKSPACE_CONSISTENCY_DRIFT',
      workspaceMetric: profile.pendingApprovalsCount,
      canonicalSourceMetric,
      message: `Workspace pending approvals (${profile.pendingApprovalsCount}) drifts from canonical source (${canonicalSourceMetric}). Reconciliation required.`,
      correlationId: `CORR-DRIFT-${Date.now()}`,
      timestampUtc: new Date().toISOString(),
    };
    return { consistent: false, error };
  }
  return { consistent: true };
}

/**
 * Invariant WS-EC-02: Workload Pagination
 * Deterministically paginates executive tasks for high-load profiles.
 */
export function paginateTasks(
  tasks: ExecutiveTask[],
  page: number = 1,
  pageSize: number = 10
): { items: ExecutiveTask[]; totalPages: number; currentPage: number; totalItems: number } {
  const totalItems = tasks.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const safePage = Math.min(Math.max(1, page), totalPages);
  const startIdx = (safePage - 1) * pageSize;
  const items = tasks.slice(startIdx, startIdx + pageSize);

  return {
    items,
    totalPages,
    currentPage: safePage,
    totalItems,
  };
}

/**
 * Invariant WS-EC-07: Workspace State Hash
 * Generates deterministic SHA-256 fingerprint of current workspace state.
 */
export function computeWorkspaceStateHash(profile: WorkspaceProfile): string {
  const payload = [
    profile.userId,
    profile.role,
    profile.isDegraded ? 'DEGRADED' : 'HEALTHY',
    profile.pendingApprovalsCount,
    profile.activeRisksCount,
    profile.assignedCommittees.map((c) => `${c.committeeId}:${c.healthScore}`).sort().join('|'),
    profile.ownedTasks.map((t) => `${t.taskId}:${t.status}`).sort().join('|'),
  ].join(':::');

  return sha256Hex(payload);
}
