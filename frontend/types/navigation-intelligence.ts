/**
 * Phase 31-M2.1 / M3 Foundation: Navigation & Governance Intelligence Contracts
 *
 * Implements:
 * - Global Search & Entity Resolver Contracts
 * - Universal Cross-Linking & Related Artifacts
 * - Historical Trends (30d, 90d, 180d, 365d) for ODEI, CDQI, DIRatio, Dissent, Learning Velocity
 * - Alert Workflow & Remediation Playbooks (5 Severity Tiers, Escalation Matrix)
 * - M2-Gate-01 through M2-Gate-05 Exit Criteria
 */

export type NavigationEntityType =
  | 'DECISION'
  | 'OUTCOME'
  | 'DISSENT'
  | 'COMMITTEE'
  | 'PROPOSAL'
  | 'EVIDENCE'
  | 'SNAPSHOT'
  | 'LEARNING'
  | 'INCIDENT'
  | 'RISK'
  | 'GROUPTHINK'
  | 'RECOMMENDATION'
  | 'INTERVENTION_PLAN'
  | 'BIAS_ALERT'
  | 'OOS_REPORT'
  | 'OHI_METRIC'
  | 'CSC_RECOVERY'
  | 'OPTIMIZATION_RUN'
  | 'ALLOCATION_RESULT'
  | 'INTERVENTION_SIMULATION'
  | 'RECOVERY_STATE'
  | 'FAILOVER_EVENT'
  | 'STRATEGY_SURVIVABILITY'
  | 'SCENARIO_DEFINITION'
  | 'AUTONOMOUS_ACTION'
  | 'GOVERNANCE_POLICY'
  | 'HUMAN_OVERRIDE'
  | 'POLICY_EVALUATION'
  | 'FAIL_CLOSE_ERROR'
  | 'OPERATIONAL_RUNBOOK';

export interface EntityResolution {
  input: string;
  entityType?: NavigationEntityType;
  entityId?: string;
  title?: string;
  canonicalRoute?: string;
  found: boolean;
  suggestions: string[];
  error?: string;
  targetParams?: Record<string, string>;
}

export interface SearchTelemetry {
  query: string;
  entityType?: NavigationEntityType;
  latencyMs: number;
  resultFound: boolean;
  targetRoute?: string;
  timestampUtc: string;
}

export interface RelatedArtifactItem {
  entityId: string;
  entityType: NavigationEntityType;
  title: string;
  subtitle?: string;
  canonicalRoute: string;
  relationship:
    | 'PARENT_COMMITTEE'
    | 'SOURCE_DECISION'
    | 'REALIZED_OUTCOME'
    | 'PRESERVED_DISSENT'
    | 'ORIGINAL_PROPOSAL'
    | 'VERIFIED_EVIDENCE'
    | 'CRYPTOGRAPHIC_SNAPSHOT'
    | 'INFLUENCE_DEPENDENCY'
    | 'ATTRIBUTED_LEARNING'
    | 'CORRELATED_INCIDENT'
    | 'PREDICTED_RISK'
    | 'GROUPTHINK_SIGNAL'
    | 'COACHING_RECOMMENDATION'
    | 'INTERVENTION_ACTION'
    | 'BIAS_WARNING'
    | 'SYSTEM_CONSISTENCY'
    | 'OPTIMIZATION_CONSTRAINT'
    | 'RESILIENCE_FALLBACK'
    | 'AUTONOMOUS_APPROVAL'
    | 'POLICY_CONSTRAINT'
    | 'HUMAN_SUPERSEDENCE'
    | 'FAIL_CLOSE_TRIGGER'
    | 'RUNBOOK_EXECUTION';
  statusBadge?: string;
}

export interface RelatedArtifactsSummary {
  primaryEntityId: string;
  primaryEntityType: NavigationEntityType;
  items: RelatedArtifactItem[];
  totalConnectedArtifacts: number;
  auditReconstructible: boolean;
}

export type TrendTimeframe = '30D' | '90D' | '180D' | '365D';
export type TrendMetricType =
  | 'ODEI'
  | 'CDQI'
  | 'DIRATIO'
  | 'DISSENT_UTIL'
  | 'LEARNING_VELOCITY'
  | 'KNOWLEDGE_TRANSFER';

export interface HistoricalTrendPoint {
  timestampUtc: string;
  dayIndex: number;
  value: number;
  baselineFloor?: number;
  isFloorBreach?: boolean;
  underlyingArtifactIds?: string[];
  note?: string;
}

export interface HistoricalTrendSeries {
  metric: TrendMetricType;
  metricLabel: string;
  committeeId: string;
  committeeName: string;
  timeframe: TrendTimeframe;
  points: HistoricalTrendPoint[];
  currentValue: number;
  startValue: number;
  trendDirection: 'UP' | 'DOWN' | 'FLAT';
  deltaAbsolute: number;
  deltaPct: number;
  hasDeteriorationWarning: boolean;
  floorThreshold?: number;
  rollingAverage90d?: number;
}

export type AlertSeverity = 'INFO' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type AlertCategory = 'GOVERNANCE' | 'NETWORK' | 'PERFORMANCE';
export type AlertLifecycleStatus = 'OPEN' | 'INVESTIGATING' | 'MITIGATING' | 'RESOLVED' | 'CLOSED';

export interface AlertRemediationPlaybook {
  alertCode: string;
  severity: AlertSeverity;
  category: AlertCategory;
  title: string;
  targetSla: string;
  remediationSteps: string[];
  closureCondition: string;
  escalationTarget: string;
}

export interface AlertWorkflowItem {
  alertId: string;
  alertCode: string;
  title: string;
  severity: AlertSeverity;
  category: AlertCategory;
  status: AlertLifecycleStatus;
  affectedArtifactId: string;
  affectedArtifactType: NavigationEntityType;
  createdAtUtc: string;
  summary: string;
  impactScore: number;
  playbook: AlertRemediationPlaybook;
  resolvedAtUtc?: string;
  resolutionNotes?: string;
}
