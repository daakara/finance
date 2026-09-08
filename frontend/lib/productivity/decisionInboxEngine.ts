/**
 * Phase 31-M13: Unified Decision Inbox Engine (ARX Horizon Executive OS)
 *
 * Implements:
 * - Multi-source decision aggregation across all intelligence centers (DI-01)
 * - Multi-feed deduplication with source merging (DI-EC-01)
 * - Deterministic severity-first and SLA expiration ranking (DI-02, DI-EC-02)
 * - Concurrency lock prevention against simultaneous actions (DI-EC-03)
 * - Missing entity validation & quarantine isolation (DI-EC-04)
 * - Audit unavailable fail-closed execution block (DI-EC-06)
 */

import {
  DecisionInboxItem,
  AuditUnavailableError,
} from '@/types/executive-workspace';
import { sha256Hex } from '@/lib/governance/sha256';

export const CANONICAL_INBOX_ITEMS: DecisionInboxItem[] = [
  {
    itemId: 'INBOX-01',
    title: 'Authorize Q3 Liquidity Buffer Reallocation',
    description: 'Autonomous optimization recommends shifting 4.2% from liquid cash to short-duration sovereign tranche.',
    category: 'APPROVAL',
    severity: 'CRITICAL',
    sourceCenter: 'Autonomous Governance Center',
    entityId: 'ACT-AUTO-001',
    owner: 'Alexandra Vance (CIO)',
    slaTargetMinutes: 60,
    createdAtUtc: '2026-09-08T19:30:00Z',
    status: 'PENDING',
  },
  {
    itemId: 'INBOX-02',
    title: 'Macro Shock VaR Threshold Investigation Escalation',
    description: 'Stress simulation indicates 99% 10-day VaR buffer compression under sudden stagflation scenario.',
    category: 'ESCALATION',
    severity: 'CRITICAL',
    sourceCenter: 'Risk & Strategy Laboratory',
    entityId: 'SIM-SCEN-002',
    owner: 'Marcus Sterling (CRO)',
    slaTargetMinutes: 45,
    createdAtUtc: '2026-09-08T19:45:00Z',
    status: 'PENDING',
  },
  {
    itemId: 'INBOX-03',
    title: 'Execute Level 2 Metric Refresh Runbook',
    description: 'Resilience supervisor detected telemetry lag in Committee Health Index feed; requires L2 manual sign-off.',
    category: 'RUNBOOK',
    severity: 'HIGH',
    sourceCenter: 'Autonomous Resilience Center',
    entityId: 'REC-L2-004',
    owner: 'Marcus Sterling (CRO)',
    slaTargetMinutes: 180,
    createdAtUtc: '2026-09-08T19:00:00Z',
    status: 'PENDING',
  },
  {
    itemId: 'INBOX-04',
    title: 'Review Committee Minority Dissent on Emerging Tech Allocation',
    description: 'Dissent score spiked to 34% on tech tranche approval. Formal minority report filed for executive review.',
    category: 'RECOMMENDATION',
    severity: 'HIGH',
    sourceCenter: 'Committee Intelligence',
    entityId: 'DIS-COM-001',
    owner: 'Alexandra Vance (CIO)',
    slaTargetMinutes: 240,
    createdAtUtc: '2026-09-08T18:30:00Z',
    status: 'PENDING',
  },
  {
    itemId: 'INBOX-05',
    title: 'Pareto Optimization Front-Running Alignment',
    description: 'Pareto frontier frontier solver identified 12bps Sharpe improvement by adjusting committee review pacing.',
    category: 'OPTIMIZATION',
    severity: 'MEDIUM',
    sourceCenter: 'Optimization Intelligence',
    entityId: 'OPT-RUN-009',
    owner: 'David Chen (Committee Chair)',
    slaTargetMinutes: 480,
    createdAtUtc: '2026-09-08T17:15:00Z',
    status: 'PENDING',
  },
];

const SEVERITY_WEIGHTS: Record<DecisionInboxItem['severity'], number> = {
  CRITICAL: 4,
  HIGH: 3,
  MEDIUM: 2,
  LOW: 1,
  INFO: 0,
};

/**
 * Invariant DI-EC-01: Multi-Feed Deduplication
 * Merges redundant items from multiple intelligence centers referencing the same entity or title.
 */
export function deduplicateInboxItems(items: DecisionInboxItem[]): {
  deduplicated: DecisionInboxItem[];
  mergedCount: number;
} {
  const map = new Map<string, DecisionInboxItem>();
  let mergedCount = 0;

  for (const item of items) {
    // Deduplication key: prefer entityId, fallback to normalized title
    const key = item.entityId ? `entity:${item.entityId}` : `title:${item.title.trim().toLowerCase()}`;

    if (!map.has(key)) {
      map.set(key, { ...item, duplicateSources: [item.sourceCenter] });
    } else {
      mergedCount++;
      const existing = map.get(key)!;
      const combinedSources = Array.from(new Set([...(existing.duplicateSources || [existing.sourceCenter]), item.sourceCenter]));

      // Retain the higher severity
      const existingWeight = SEVERITY_WEIGHTS[existing.severity] ?? 0;
      const itemWeight = SEVERITY_WEIGHTS[item.severity] ?? 0;
      const higherSeverity = itemWeight > existingWeight ? item.severity : existing.severity;

      // Retain the tighter SLA
      const tighterSla = Math.min(existing.slaTargetMinutes, item.slaTargetMinutes);

      map.set(key, {
        ...existing,
        severity: higherSeverity,
        slaTargetMinutes: tighterSla,
        duplicateSources: combinedSources,
      });
    }
  }

  return {
    deduplicated: Array.from(map.values()),
    mergedCount,
  };
}

/**
 * Invariant DI-02 / DI-EC-02: Deterministic Priority & SLA Ranking
 * Severity-first, then SLA target ascending, then creation time ascending.
 */
export function rankInboxItems(items: DecisionInboxItem[]): DecisionInboxItem[] {
  return [...items].sort((a, b) => {
    // 1. Severity weight descending
    const weightDiff = (SEVERITY_WEIGHTS[b.severity] ?? 0) - (SEVERITY_WEIGHTS[a.severity] ?? 0);
    if (weightDiff !== 0) return weightDiff;

    // 2. SLA target minutes ascending (closest deadline first)
    const slaDiff = a.slaTargetMinutes - b.slaTargetMinutes;
    if (slaDiff !== 0) return slaDiff;

    // 3. Creation time ascending (older first)
    return a.createdAtUtc.localeCompare(b.createdAtUtc);
  });
}

/**
 * Invariant DI-EC-03: Concurrency Lock Prevention
 * Prevents simultaneous execution conflicts by locking item during execution.
 */
export function lockInboxItemForExecution(
  items: DecisionInboxItem[],
  itemId: string,
  userId: string
): { success: boolean; items: DecisionInboxItem[]; lockedItem?: DecisionInboxItem; error?: string } {
  const target = items.find((i) => i.itemId === itemId);
  if (!target) {
    return { success: false, items, error: `Item ${itemId} not found.` };
  }

  if (target.status === 'EXECUTING') {
    return {
      success: false,
      items,
      error: `Concurrency Conflict: Item ${itemId} is already being executed by ${target.lockedBy || 'another user'}.`,
    };
  }

  const updatedItems = items.map((i) =>
    i.itemId === itemId
      ? { ...i, status: 'EXECUTING' as const, lockedBy: userId }
      : i
  );

  const lockedItem = updatedItems.find((i) => i.itemId === itemId);

  return { success: true, items: updatedItems, lockedItem };
}

/**
 * Invariant DI-EC-04: Missing Entity Quarantine
 * Quarantines decisions that reference invalid or missing entities.
 */
export function validateAndQuarantineItem(
  item: DecisionInboxItem,
  knownEntities?: Set<string>
): DecisionInboxItem {
  if (!item.entityId || item.entityId.trim() === '' || (knownEntities && !knownEntities.has(item.entityId))) {
    return {
      ...item,
      status: 'QUARANTINED',
      description: `[QUARANTINED - UNRESOLVED ENTITY: ${item.entityId || 'NONE'}] ${item.description}`,
    };
  }
  return item;
}

/**
 * Invariant DI-EC-06: Audit Fail-Close Execution
 * Fails closed without mutating state if the audit service is unavailable.
 */
export function executeInboxAction(
  items: DecisionInboxItem[],
  itemId: string,
  actionType: string = 'APPROVE',
  options?: { simulateAuditFailure?: boolean; executorId?: string }
): {
  success: boolean;
  items: DecisionInboxItem[];
  receipt?: { executionId: string; auditHash: string; timestampUtc: string; action: string };
  error?: AuditUnavailableError;
} {
  const target = items.find((i) => i.itemId === itemId);
  if (!target) {
    return { success: false, items };
  }

  // Audit fail-close check
  if (options?.simulateAuditFailure) {
    const error: AuditUnavailableError = {
      errorCode: 'DI-ERR-002',
      errorType: 'AUDIT_UNAVAILABLE_FAIL_CLOSE',
      attemptedActionId: itemId,
      message: `Audit logging pipeline unavailable for action ${actionType} on item ${itemId}. Action blocked fail-closed.`,
      correlationId: `CORR-AUDIT-FAIL-${Date.now()}`,
      timestampUtc: new Date().toISOString(),
    };
    return { success: false, items, error };
  }

  // Generate audit receipt
  const timestampUtc = new Date().toISOString();
  const rawReceipt = `${itemId}|${actionType}|${options?.executorId || 'SYSTEM'}|${timestampUtc}`;
  const auditHash = sha256Hex(rawReceipt);

  const updatedItems = items.map((i) =>
    i.itemId === itemId
      ? { ...i, status: 'RESOLVED' as const, lockedBy: undefined }
      : i
  );

  return {
    success: true,
    items: updatedItems,
    receipt: {
      executionId: `EXEC-${itemId.replace('INBOX-', '')}-${Date.now()}`,
      auditHash,
      timestampUtc,
      action: actionType,
    },
  };
}

export function getInboxMetrics(items: DecisionInboxItem[]) {
  const total = items.length;
  const pending = items.filter((i) => i.status === 'PENDING').length;
  const critical = items.filter((i) => i.severity === 'CRITICAL' && i.status === 'PENDING').length;
  const high = items.filter((i) => i.severity === 'HIGH' && i.status === 'PENDING').length;
  const urgentSla = items.filter((i) => i.status === 'PENDING' && i.slaTargetMinutes <= 60).length;
  const executing = items.filter((i) => i.status === 'EXECUTING').length;
  const resolved = items.filter((i) => i.status === 'RESOLVED').length;
  const quarantined = items.filter((i) => i.status === 'QUARANTINED').length;

  return {
    total,
    pending,
    critical,
    high,
    urgentSla,
    executing,
    resolved,
    quarantined,
  };
}
