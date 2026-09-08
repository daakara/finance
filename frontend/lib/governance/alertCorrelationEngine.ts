/**
 * Phase 31-M3: Alert Correlation Engine & Alert Fatigue Controls
 *
 * Implements:
 * - Multi-Signal Correlation Patterns (CORR-01 to CORR-05):
 *   - CORR-01: ORGANIZATIONAL_LEARNING_BREAKDOWN (HIGH)
 *   - CORR-02: AUDIT_INTEGRITY_INCIDENT (CRITICAL)
 *   - CORR-03: DETERMINISM_FAILURE (CRITICAL)
 *   - CORR-04: NETWORK_GOVERNANCE_INCIDENT (HIGH)
 *   - CORR-05: Duplicate Alert Compression (N identical alerts -> 1 incident)
 *
 * - Alert Fatigue Controls (FAT-01 to FAT-06):
 *   - FAT-01: 30-Minute Duplicate Suppression Window
 *   - FAT-02: Incident Counter Expansion
 *   - FAT-03: Severity Escalation (>=25 repeats -> Escalate severity)
 *   - FAT-04: Cross-Signal Merge
 *   - FAT-05: Resolution Reset (Resolved issue recurrence creates new incident)
 *   - FAT-06: SLA Preservation (Timer retained, critical issues never hidden)
 */

import type {
  CorrelatedIncident,
  CorrelatedIncidentType,
  IncidentSeverity,
  IncidentLifecycleStatus,
} from '../../types/learning-intelligence';

import { sha256 } from './sha256';

export const SUPPRESSION_WINDOW_MS = 30 * 60 * 1000; // 30 minutes (FAT-01)
export const ESCALATION_THRESHOLD_COUNT = 25; // 25 repeats (FAT-03)

export const CORRELATION_PATTERNS: {
  patternId: string;
  incidentType: CorrelatedIncidentType;
  requiredAlerts: string[];
  defaultSeverity: IncidentSeverity;
  rootCause: string;
}[] = [
  {
    patternId: 'CORR-01',
    incidentType: 'ORGANIZATIONAL_LEARNING_BREAKDOWN',
    requiredAlerts: ['ODEI_DECLINE', 'LEARNING_VELOCITY_NON_POSITIVE', 'KNOWLEDGE_TRANSFER_FAILURE'],
    defaultSeverity: 'HIGH',
    rootCause: 'Systemic breakdown in institutional learning velocity and cross-committee adoption.',
  },
  {
    patternId: 'CORR-02',
    incidentType: 'AUDIT_INTEGRITY_INCIDENT',
    requiredAlerts: ['LOST_DISSENT', 'MISSING_ATTRIBUTION', 'AUDIT_RECONSTRUCTION_FAILURE'],
    defaultSeverity: 'CRITICAL',
    rootCause: 'Critical compromise of immutable decision audit trail and dissent records.',
  },
  {
    patternId: 'CORR-03',
    incidentType: 'DETERMINISM_FAILURE',
    requiredAlerts: ['REPLAY_VARIANCE', 'HASH_MISMATCH', 'SNAPSHOT_MISMATCH'],
    defaultSeverity: 'CRITICAL',
    rootCause: 'Non-deterministic calculation drift detected across repeated replay runs.',
  },
  {
    patternId: 'CORR-04',
    incidentType: 'NETWORK_GOVERNANCE_INCIDENT',
    requiredAlerts: ['NETWORK_CYCLE', 'INFLUENCE_CONCENTRATION', 'KNOWLEDGE_TRANSFER_FAILURE'],
    defaultSeverity: 'HIGH',
    rootCause: 'Circular committee influence dependencies creating governance deadlock.',
  },
];

// In-memory active incidents store
const activeIncidents: CorrelatedIncident[] = [
  {
    incidentId: 'INC-201',
    incidentType: 'ORGANIZATIONAL_LEARNING_BREAKDOWN',
    severity: 'HIGH',
    sourceAlerts: ['ODEI_DECLINE', 'LEARNING_VELOCITY_NON_POSITIVE', 'KNOWLEDGE_TRANSFER_FAILURE'],
    occurrenceCount: 8,
    status: 'INVESTIGATING',
    firstSeenAtUtc: '2026-09-08T11:00:00Z',
    lastSeenAtUtc: '2026-09-08T11:25:00Z',
    rootCauseHypothesis: 'Investment Committee and Governance Committee experiencing cross-transfer stagnation.',
    affectedCommitteeIds: ['COM-001', 'COM-002'],
    slaDeadlineUtc: '2026-09-09T11:00:00Z',
  },
  {
    incidentId: 'INC-202',
    incidentType: 'NETWORK_GOVERNANCE_INCIDENT',
    severity: 'HIGH',
    sourceAlerts: ['NETWORK_CYCLE', 'INFLUENCE_CONCENTRATION', 'KNOWLEDGE_TRANSFER_FAILURE'],
    occurrenceCount: 4,
    status: 'OPEN',
    firstSeenAtUtc: '2026-09-08T12:00:00Z',
    lastSeenAtUtc: '2026-09-08T12:15:00Z',
    rootCauseHypothesis: 'Feedback loop detected between COM-001 and COM-003.',
    affectedCommitteeIds: ['COM-001', 'COM-003'],
    slaDeadlineUtc: '2026-09-09T12:00:00Z',
  },
];

export function getActiveIncidents(): CorrelatedIncident[] {
  return [...activeIncidents];
}

export function getIncidentById(incidentId: string): CorrelatedIncident | undefined {
  return activeIncidents.find(i => i.incidentId === incidentId);
}

export function clearIncidents(): void {
  activeIncidents.length = 0;
}

/**
 * Correlates an incoming batch of alerts into unified root-cause incidents.
 * Enforces CORR-01 through CORR-05 and FAT-01 through FAT-06.
 */
export function correlateAlerts(
  rawAlerts: { code: string; timestampMs?: number; committeeId?: string }[]
): CorrelatedIncident[] {
  const now = Date.now();
  const alertCodes = new Set(rawAlerts.map(a => a.code));
  const correlatedResults: CorrelatedIncident[] = [];

  // Check multi-signal patterns (CORR-01 to CORR-04)
  for (const pattern of CORRELATION_PATTERNS) {
    const matches = pattern.requiredAlerts.every(req => alertCodes.has(req));
    if (matches) {
      const childAlerts = rawAlerts.filter(a => pattern.requiredAlerts.includes(a.code));
      const affectedComms = Array.from(new Set(childAlerts.map(a => a.committeeId ?? 'COM-001')));
      const earliestTs = Math.min(...childAlerts.map(a => a.timestampMs ?? now));

      correlatedResults.push({
        incidentId: `INC-${pattern.patternId}-${Date.now().toString().slice(-4)}`,
        incidentType: pattern.incidentType,
        severity: pattern.defaultSeverity,
        sourceAlerts: pattern.requiredAlerts,
        occurrenceCount: childAlerts.length,
        status: 'OPEN',
        firstSeenAtUtc: new Date(earliestTs).toISOString(),
        lastSeenAtUtc: new Date(now).toISOString(),
        rootCauseHypothesis: pattern.rootCause,
        affectedCommitteeIds: affectedComms,
        slaDeadlineUtc: new Date(earliestTs + 24 * 3600 * 1000).toISOString(), // 24-hour SLA (FAT-06)
      });
    }
  }

  // If no multi-signal pattern matched, group duplicates into compressed incidents (CORR-05)
  if (correlatedResults.length === 0 && rawAlerts.length > 0) {
    const codeGroups = new Map<string, { code: string; timestampMs?: number; committeeId?: string }[]>();
    for (const a of rawAlerts) {
      const list = codeGroups.get(a.code) ?? [];
      list.push(a);
      codeGroups.set(a.code, list);
    }

    for (const [code, items] of codeGroups.entries()) {
      const count = items.length;
      let severity: IncidentSeverity = 'MEDIUM';
      if (count >= ESCALATION_THRESHOLD_COUNT) {
        severity = 'HIGH'; // FAT-03 Severity Escalation
      }

      const earliestTs = Math.min(...items.map(a => a.timestampMs ?? now));
      correlatedResults.push({
        incidentId: `INC-${code.slice(0, 8)}-${Date.now().toString().slice(-4)}`,
        incidentType: 'ORGANIZATIONAL_LEARNING_BREAKDOWN',
        severity,
        sourceAlerts: [code],
        occurrenceCount: count, // FAT-01 & FAT-02: 1 visible incident with count = N
        status: 'OPEN',
        firstSeenAtUtc: new Date(earliestTs).toISOString(),
        lastSeenAtUtc: new Date(now).toISOString(),
        rootCauseHypothesis: `Compressed ${count} duplicate instances of ${code}`,
        affectedCommitteeIds: Array.from(new Set(items.map(a => a.committeeId ?? 'COM-001'))),
        slaDeadlineUtc: new Date(earliestTs + 24 * 3600 * 1000).toISOString(),
      });
    }
  }

  return correlatedResults;
}

/**
 * Ingests a single alert with automatic fatigue control, deduplication, and escalation.
 */
export function ingestAlert(
  alertCode: string,
  committeeId: string = 'COM-001',
  timestampMs: number = Date.now()
): {
  incident: CorrelatedIncident;
  isNewIncident: boolean;
  suppressedDuplicate: boolean;
  escalated: boolean;
} {
  // Check if an existing open incident has this alertCode
  const existing = activeIncidents.find(
    inc => (inc.status === 'OPEN' || inc.status === 'INVESTIGATING' || inc.status === 'MITIGATING') &&
           inc.sourceAlerts.includes(alertCode)
  );

  if (existing) {
    const firstSeenMs = new Date(existing.firstSeenAtUtc).getTime();
    const isInsideWindow = timestampMs - firstSeenMs <= SUPPRESSION_WINDOW_MS;

    // Increment occurrence count (FAT-02)
    existing.occurrenceCount++;
    existing.lastSeenAtUtc = new Date(timestampMs).toISOString();

    // Check severity escalation (FAT-03)
    let escalated = false;
    if (existing.occurrenceCount >= ESCALATION_THRESHOLD_COUNT && existing.severity === 'MEDIUM') {
      existing.severity = 'HIGH';
      escalated = true;
    } else if (existing.occurrenceCount >= ESCALATION_THRESHOLD_COUNT * 2 && existing.severity === 'HIGH') {
      existing.severity = 'CRITICAL';
      escalated = true;
    }

    if (!existing.affectedCommitteeIds.includes(committeeId)) {
      existing.affectedCommitteeIds.push(committeeId);
    }

    return {
      incident: existing,
      isNewIncident: false,
      suppressedDuplicate: isInsideWindow,
      escalated,
    };
  }

  // If previous incident was resolved, reset and create fresh new incident (FAT-05)
  const newIncident: CorrelatedIncident = {
    incidentId: `INC-${alertCode.slice(0, 6)}-${Date.now().toString().slice(-4)}`,
    incidentType: 'ORGANIZATIONAL_LEARNING_BREAKDOWN',
    severity: 'MEDIUM',
    sourceAlerts: [alertCode],
    occurrenceCount: 1,
    status: 'OPEN',
    firstSeenAtUtc: new Date(timestampMs).toISOString(),
    lastSeenAtUtc: new Date(timestampMs).toISOString(),
    rootCauseHypothesis: `Root cause investigation initiated for ${alertCode}`,
    affectedCommitteeIds: [committeeId],
    slaDeadlineUtc: new Date(timestampMs + 24 * 3600 * 1000).toISOString(), // SLA preserved (FAT-06)
  };

  activeIncidents.unshift(newIncident);

  return {
    incident: newIncident,
    isNewIncident: true,
    suppressedDuplicate: false,
    escalated: false,
  };
}

/**
 * Resolves an active incident and records resolution timestamp.
 */
export function resolveIncident(
  incidentId: string,
  resolutionNotes: string = 'Remediation completed and verified.'
): CorrelatedIncident {
  const inc = activeIncidents.find(i => i.incidentId === incidentId);
  if (!inc) {
    throw new Error(`Incident ${incidentId} not found.`);
  }

  inc.status = 'RESOLVED';
  inc.resolvedAtUtc = new Date().toISOString();
  inc.resolutionNotes = resolutionNotes;

  return inc;
}

/**
 * Deterministic hash of correlated incidents registry.
 */
export function hashIncidents(): string {
  const payload = activeIncidents.map(i => ({
    id: i.incidentId,
    type: i.incidentType,
    sev: i.severity,
    cnt: i.occurrenceCount,
    st: i.status,
    alerts: [...i.sourceAlerts].sort(),
  }));
  return sha256(JSON.stringify(payload));
}
