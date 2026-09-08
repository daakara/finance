/**
 * Phase 31-M6: Unified Telemetry Hub
 *
 * Implements:
 * - Telemetry aggregation across Committee, Network, Learning, Risk, and Prescriptive engines
 * - Unified Telemetry Snapshot & System Health scoring
 * - Enterprise Alert aggregation
 * - Pure TypeScript SHA-256 State Hashing
 */

import type {
  TelemetrySnapshot,
  SystemHealth,
  EnterpriseAlert,
  OrganizationalState,
} from '../../types/oos-intelligence';

import { sha256Hex } from '../governance/sha256';
import { CANONICAL_ORGANIZATIONAL_HEALTH_INDEX } from './organizationalHealthEngine';

export const CANONICAL_ENTERPRISE_ALERTS: EnterpriseAlert[] = [
  {
    alertId: 'ALT-ENT-001',
    sourceSystem: 'RISK',
    severity: 'CRITICAL',
    title: 'Decision Unanimity Drift in Risk Committee',
    description: 'Unanimity reached 90% with zero dissenting votes across recent macro-tranche votes.',
    timestampUtc: '2026-09-08T17:30:00Z',
    actionRequired: true,
    remediationPlaybookId: 'PB-RSK-002',
  },
  {
    alertId: 'ALT-ENT-002',
    sourceSystem: 'COACHING',
    severity: 'HIGH',
    title: 'Intervention Plan PLAN-001 Phase 2 Pending Review',
    description: 'Contrarian review protocol active for Investment Committee sector outlier theses.',
    timestampUtc: '2026-09-08T16:45:00Z',
    actionRequired: true,
    remediationPlaybookId: 'PB-REC-001',
  },
  {
    alertId: 'ALT-ENT-003',
    sourceSystem: 'LEARNING',
    severity: 'MEDIUM',
    title: 'Knowledge Transfer Stagnation on Strategy Category',
    description: 'Cross-committee adoption latency reached 4.2 days on LRN-004 execution.',
    timestampUtc: '2026-09-08T15:00:00Z',
    actionRequired: false,
  },
  {
    alertId: 'ALT-ENT-004',
    sourceSystem: 'CONSISTENCY',
    severity: 'LOW',
    title: 'Cross-System Snapshot Sync Verified',
    description: 'Zero variance observed across Dashboard, API, and Audit Reconstruction mirrors.',
    timestampUtc: '2026-09-08T18:00:00Z',
    actionRequired: false,
  },
];

export const CANONICAL_SYSTEM_HEALTH: SystemHealth = {
  status: 'OPTIMAL',
  overallScore: 94.5,
  activeSubsystems: 5,
  healthySubsystems: 5,
  subsystemStatuses: {
    COMMITTEE_INTELLIGENCE: {
      status: 'ONLINE',
      score: 96.0,
      lastPingUtc: '2026-09-08T18:00:00Z',
    },
    LEARNING_INTELLIGENCE: {
      status: 'ONLINE',
      score: 93.5,
      lastPingUtc: '2026-09-08T18:00:00Z',
    },
    RISK_INTELLIGENCE: {
      status: 'ONLINE',
      score: 91.0,
      lastPingUtc: '2026-09-08T18:00:00Z',
    },
    PRESCRIPTIVE_COACH: {
      status: 'ONLINE',
      score: 97.0,
      lastPingUtc: '2026-09-08T18:00:00Z',
    },
    CONSISTENCY_MIRROR: {
      status: 'ONLINE',
      score: 95.0,
      lastPingUtc: '2026-09-08T18:00:00Z',
    },
  },
};

export const CANONICAL_TELEMETRY_SNAPSHOT: TelemetrySnapshot = {
  snapshotId: 'SNAP-OOS-001',
  timestampUtc: '2026-09-08T18:00:00Z',
  committeeCount: 3,
  averageODEI: 85.0,
  averageCDQI: 83.0,
  averageDIRatio: 25.0,
  learningVelocity: 12.0,
  knowledgeTransferRate: 88.0,
  learningFrictionScore: 18.5,
  groupthinkMaxScore: 8.0,
  openCriticalRisks: 4,
  activeRecommendations: 12,
  coachImpactRatio: 2.14,
  enterpriseAlertCount: CANONICAL_ENTERPRISE_ALERTS.length,
  stateHash: '',
};

export function hashTelemetryState(snapshot: TelemetrySnapshot): string {
  const payload = {
    id: snapshot.snapshotId,
    ts: snapshot.timestampUtc,
    odei: snapshot.averageODEI,
    cdqi: snapshot.averageCDQI,
    dir: snapshot.averageDIRatio,
    lv: snapshot.learningVelocity,
    kt: snapshot.knowledgeTransferRate,
    lf: snapshot.learningFrictionScore,
    gt: snapshot.groupthinkMaxScore,
    risks: snapshot.openCriticalRisks,
    recs: snapshot.activeRecommendations,
    cir: snapshot.coachImpactRatio,
  };
  return sha256Hex(JSON.stringify(payload));
}

CANONICAL_TELEMETRY_SNAPSHOT.stateHash = hashTelemetryState(CANONICAL_TELEMETRY_SNAPSHOT);

export function getUnifiedTelemetrySnapshot(): TelemetrySnapshot {
  return CANONICAL_TELEMETRY_SNAPSHOT;
}

export function getSystemHealth(): SystemHealth {
  return CANONICAL_SYSTEM_HEALTH;
}

export function getEnterpriseAlerts(): EnterpriseAlert[] {
  return CANONICAL_ENTERPRISE_ALERTS;
}
