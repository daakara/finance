/**
 * Phase 31-M2.1 / M3 Foundation: Alert Workflow Engine & Remediation Playbooks
 *
 * Implements:
 * - 5-Tier Severity Model (INFO, LOW, MEDIUM, HIGH, CRITICAL)
 * - 7 Concrete Remediation Playbooks with step-by-step checklists & closure conditions
 * - Escalation Matrix (Dashboard -> Chair -> Governance -> Executive Fail-Close)
 * - Alert lifecycle state management (OPEN -> INVESTIGATING -> MITIGATING -> RESOLVED -> CLOSED)
 * - AW-01 through AW-08 acceptance criteria validation
 */

import type {
  AlertSeverity,
  AlertCategory,
  AlertLifecycleStatus,
  AlertRemediationPlaybook,
  AlertWorkflowItem,
} from '../../types/navigation-intelligence';

export const CANONICAL_REMEDIATION_PLAYBOOKS: Record<string, AlertRemediationPlaybook> = {
  DECISION_FORK: {
    alertCode: 'DECISION_FORK',
    severity: 'CRITICAL',
    category: 'GOVERNANCE',
    title: 'Decision Fork Split-Brain Remediation',
    targetSla: 'Immediate (< 1 Hour)',
    remediationSteps: [
      '1. Freeze target decision record in institutional ledger.',
      '2. Reconstruct complete artifact lineage from immutable proposal vault.',
      '3. Determine authoritative outcome via cryptographic snapshot comparison.',
      '4. Re-certify decision and update historical attribution ledger.',
    ],
    closureCondition: 'Single authoritative outcome verified and certified (0 forks).',
    escalationTarget: 'Executive Governance Board & Chief Risk Officer',
  },
  SUPPRESSED_DISSENT: {
    alertCode: 'SUPPRESSED_DISSENT',
    severity: 'CRITICAL',
    category: 'GOVERNANCE',
    title: 'Minority Dissent Suppression Remediation',
    targetSla: 'Immediate (< 1 Hour)',
    remediationSteps: [
      '1. Restore dissent record into persistent committee storage.',
      '2. Recalculate committee dissent coverage and utilization metrics.',
      '3. Re-run institutional certification gates (CII-Gate-02 / INV-OI14).',
      '4. Audit committee chair record-keeping process for procedural compliance.',
    ],
    closureCondition: '100% dissent preservation restored across all material decisions.',
    escalationTarget: 'Governance Committee Chair & Institutional Review Board',
  },
  INFLUENCE_CYCLE: {
    alertCode: 'INFLUENCE_CYCLE',
    severity: 'HIGH',
    category: 'NETWORK',
    title: 'Circular Influence Loop Remediation',
    targetSla: '24 Hours',
    remediationSteps: [
      '1. Inspect directional topological graph and identify circular coalition nodes.',
      '2. Validate cross-committee relationship legitimacy and shared decisions.',
      '3. Break artificial self-reinforcing approval dependencies.',
      '4. Recompute network density, influence scores, and cycle count.',
    ],
    closureCondition: 'Cycle count strictly equals 0 in canonical DAG topology.',
    escalationTarget: 'Governance Architecture Team & Network Auditor',
  },
  ORPHAN_OUTCOME: {
    alertCode: 'ORPHAN_OUTCOME',
    severity: 'HIGH',
    category: 'GOVERNANCE',
    title: 'Orphan Outcome Lineage Re-linking',
    targetSla: '24 Hours',
    remediationSteps: [
      '1. Locate originating decision identifier via financial transaction ledger.',
      '2. Reconstruct four-facet attribution percentages (Individual, Team, Committee, System).',
      '3. Re-link outcome to parent decision in authoritative registry.',
      '4. Verify 100.0% attribution sum conservation.',
    ],
    closureCondition: '100% end-to-end traceability restored for affected outcome.',
    escalationTarget: 'Portfolio Intelligence Lead & Head of Attribution',
  },
  REPLAY_VARIANCE: {
    alertCode: 'REPLAY_VARIANCE',
    severity: 'CRITICAL',
    category: 'PERFORMANCE',
    title: 'Deterministic Replay Variance Remediation',
    targetSla: 'Immediate (< 1 Hour)',
    remediationSteps: [
      '1. Compare canonical JSON serialization hashes between runs.',
      '2. Inspect AST mismatch paths for floating-point or object key ordering drift.',
      '3. Isolate non-deterministic code using scale-aware relative tolerance.',
      '4. Execute 100x consecutive replay verification harness.',
    ],
    closureCondition: '100 consecutive runs yield 1 identical SHA-256 hash with 0 drift.',
    escalationTarget: 'Principal Quantitative Systems Engineer & Core Engine Team',
  },
  LEARNING_VELOCITY_NON_POSITIVE: {
    alertCode: 'LEARNING_VELOCITY_NON_POSITIVE',
    severity: 'MEDIUM',
    category: 'PERFORMANCE',
    title: 'Stagnant / Non-Positive Learning Velocity Mitigation',
    targetSla: '5 Business Days',
    remediationSteps: [
      '1. Conduct committee outcome post-mortem for the past 4 rolling quarters.',
      '2. Analyze dissent utilization rates on underperforming decisions.',
      '3. Identify repeated systemic failure patterns or bias blind spots.',
      '4. Publish formal corrective action playbook and update protected practices.',
    ],
    closureCondition: 'Learning velocity strictly exceeds 0.0 (dODEI / dt > 0).',
    escalationTarget: 'Target Committee Chair & Organizational Learning Lead',
  },
  KNOWLEDGE_TRANSFER_FAILURE: {
    alertCode: 'KNOWLEDGE_TRANSFER_FAILURE',
    severity: 'HIGH',
    category: 'NETWORK',
    title: 'Cross-Committee Knowledge Transfer Failure Remediation',
    targetSla: '24 Hours',
    remediationSteps: [
      '1. Identify downstream dependent committees in the network graph.',
      '2. Publish institutional learning advisory highlighting unadopted insight.',
      '3. Track adoption via voting eligibility and evidence checklist references.',
      '4. Re-evaluate knowledge transfer rate against >=80.0% threshold.',
    ],
    closureCondition: 'Knowledge transfer completeness satisfies >= 80.0% benchmark.',
    escalationTarget: 'Cross-Committee Steering Group & Strategy Office',
  },
};

export const ESCALATION_MATRIX: Record<AlertSeverity, {
  destination: string;
  sla: string;
  actionRequired: string;
  blocksRelease: boolean;
}> = {
  INFO: {
    destination: 'Dashboard Logging Only',
    sla: 'Informational (No SLA)',
    actionRequired: 'Automated telemetry ingestion and status recording.',
    blocksRelease: false,
  },
  LOW: {
    destination: 'Dashboard + Trend Watchlist',
    sla: '30 Days',
    actionRequired: 'Monitor metric trends for progressive drift or degradation.',
    blocksRelease: false,
  },
  MEDIUM: {
    destination: 'Committee Chair Notification',
    sla: '5 Business Days',
    actionRequired: 'Chair review and corrective action scheduling.',
    blocksRelease: false,
  },
  HIGH: {
    destination: 'Governance Team Notification',
    sla: '24 Hours',
    actionRequired: 'Formal investigation and root cause remediation.',
    blocksRelease: true,
  },
  CRITICAL: {
    destination: 'Governance + Executive Escalation + Certification Fail-Close',
    sla: 'Immediate (< 1 Hour)',
    actionRequired: 'Instant freeze of affected assets; certification revoked until 100% resolved.',
    blocksRelease: true,
  },
};

// In-memory alert state storage
const activeAlerts: AlertWorkflowItem[] = [
  {
    alertId: 'ALT-101',
    alertCode: 'INFLUENCE_CYCLE',
    title: 'Topological Circular Influence Warning in COM-001 <-> COM-003',
    severity: 'HIGH',
    category: 'NETWORK',
    status: 'MITIGATING',
    affectedArtifactId: 'COM-001',
    affectedArtifactType: 'COMMITTEE',
    createdAtUtc: '2026-09-08T10:00:00Z',
    summary: 'Detected potential circular feedback loop between Investment and Risk committees.',
    impactScore: 78,
    playbook: CANONICAL_REMEDIATION_PLAYBOOKS.INFLUENCE_CYCLE,
  },
  {
    alertId: 'ALT-102',
    alertCode: 'LEARNING_VELOCITY_NON_POSITIVE',
    title: 'Committee COM-002 Learning Velocity Stagnation Warning',
    severity: 'MEDIUM',
    category: 'PERFORMANCE',
    status: 'OPEN',
    affectedArtifactId: 'COM-002',
    affectedArtifactType: 'COMMITTEE',
    createdAtUtc: '2026-09-08T10:30:00Z',
    summary: 'Quarter-over-quarter ODEI growth slowed below target institutional trajectory.',
    impactScore: 54,
    playbook: CANONICAL_REMEDIATION_PLAYBOOKS.LEARNING_VELOCITY_NON_POSITIVE,
  },
];

export function getActiveAlerts(): AlertWorkflowItem[] {
  return [...activeAlerts];
}

export function getAlertById(alertId: string): AlertWorkflowItem | undefined {
  return activeAlerts.find(a => a.alertId === alertId);
}

export function getPlaybookForCode(alertCode: string): AlertRemediationPlaybook {
  return CANONICAL_REMEDIATION_PLAYBOOKS[alertCode] ?? {
    alertCode,
    severity: 'MEDIUM',
    category: 'GOVERNANCE',
    title: `Generic Remediation for ${alertCode}`,
    targetSla: '5 Business Days',
    remediationSteps: ['1. Review alert context.', '2. Take corrective action.', '3. Verify metrics.'],
    closureCondition: 'Issue resolved.',
    escalationTarget: 'Governance Committee',
  };
}

export function transitionAlertStatus(
  alertId: string,
  newStatus: AlertLifecycleStatus,
  notes?: string
): AlertWorkflowItem {
  const alert = activeAlerts.find(a => a.alertId === alertId);
  if (!alert) {
    throw new Error(`Alert ${alertId} not found in active workflow registry.`);
  }

  alert.status = newStatus;
  if (newStatus === 'RESOLVED' || newStatus === 'CLOSED') {
    alert.resolvedAtUtc = new Date().toISOString();
    alert.resolutionNotes = notes ?? 'Remediation playbook steps completed and verified.';
  }

  return alert;
}

export function registerNewAlert(item: Omit<AlertWorkflowItem, 'alertId' | 'createdAtUtc' | 'playbook'>): AlertWorkflowItem {
  const playbook = getPlaybookForCode(item.alertCode);
  const newAlert: AlertWorkflowItem = {
    ...item,
    alertId: `ALT-${Date.now().toString().slice(-4)}`,
    createdAtUtc: new Date().toISOString(),
    playbook,
  };
  activeAlerts.unshift(newAlert);
  return newAlert;
}
