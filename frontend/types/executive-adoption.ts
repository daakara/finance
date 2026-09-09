/**
 * Horizon 1: Executive Adoption & Usage Instrumentation Contracts
 *
 * Implements:
 * - Executive Usage Events across M11-M16
 * - 6 Core Adoption KPIs (DAE, TTD, Actions, Briefings, Search, Feature Adoption)
 * - Workflow Completion Cohorts & Stage Bottleneck Telemetry
 * - Productivity Gain & Economic ROI Summary
 * - EAD Traceability Matrix (EAD-Gate-01 through EAD-Gate-10)
 */

export type ExecutiveUsageEventType =
  | 'WORKSPACE_VISIT'
  | 'INBOX_TRIAGE'
  | 'BRIEFING_GENERATED'
  | 'ACTION_EXECUTED'
  | 'GRAPH_EXPLORED'
  | 'RELEASE_VIEWED'
  | 'SIMULATION_TESTED';

export type ExecutiveRoleType =
  | 'CEO'
  | 'CIO'
  | 'CRO'
  | 'COO'
  | 'COMMITTEE_CHAIR'
  | 'AUDIT_CHAIR';

export interface ExecutiveUsageEvent {
  eventId: string;
  eventType: ExecutiveUsageEventType;
  userId: string;
  userRole: ExecutiveRoleType;
  entityId?: string;
  durationMs?: number;
  metadata?: Record<string, string | number | boolean>;
  timestampUtc: string;
}

export interface ExecutiveAdoptionMetrics {
  dailyActiveExecutives: number;
  monthlyActiveExecutives: number;
  medianTimeToDecisionMinutes: number;
  baselineTimeToDecisionMinutes: number;
  timeReductionPct: number;
  totalActionsExecuted: number;
  actionSlaAdherencePct: number;
  totalBriefingsGenerated: number;
  briefingReplayDeterminismPct: number;
  searchSuccessRatePct: number;
  averageSearchLatencyMs: number;
  overallFeatureAdoptionPct: number;
}

export interface MilestoneAdoptionBreakdown {
  milestoneId: string;
  milestoneName: string;
  activeUsers: number;
  usageCount: number;
  adoptionPct: number;
  trend: 'RISING' | 'STABLE' | 'DECLINING';
}

export interface LifecycleStageMetrics {
  stageNumber: number;
  stageName: string;
  medianMinutes: number;
  completionRatePct: number;
  dropOffRatePct: number;
  isBottleneck: boolean;
}

export interface WorkflowCompletionCohort {
  cohortId: string;
  totalWorkflowsInitiated: number;
  totalWorkflowsCompleted: number;
  completionRatePct: number;
  medianCycleTimeMinutes: number;
  stages: LifecycleStageMetrics[];
}

export interface ProductivityGainSummary {
  hoursSavedPerExecutiveMonthly: number;
  totalHoursSavedMonthly: number;
  effectiveCostSavingsUSD: number;
  decisionVelocityMultiplier: number;
  riskAvoidanceEvents: number;
}

export interface ExecutiveAdoptionSnapshot {
  snapshotId: string;
  generatedAtUtc: string;
  metrics: ExecutiveAdoptionMetrics;
  milestoneBreakdown: MilestoneAdoptionBreakdown[];
  workflowCohort: WorkflowCompletionCohort;
  productivity: ProductivityGainSummary;
  replayHash: string;
}

export const EAD_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'EAD-Gate-01', name: 'KPI Completeness & Rendering', requirement: 'All 6 institutional adoption KPIs render with bounded calibration' },
  { gateId: 'EAD-Gate-02', name: 'Time to Decision Calibration', requirement: 'Demonstrates statistically significant latency drop vs 4.2h baseline' },
  { gateId: 'EAD-Gate-03', name: 'Workflow Lifecycle Completion Tracking', requirement: 'All 8 lifecycle stages instrumented with completion & drop-off metrics' },
  { gateId: 'EAD-Gate-04', name: 'Feature Adoption Matrix across M11-M16', requirement: 'Every Phase 31 capability tracked with discrete user cohorts' },
  { gateId: 'EAD-Gate-05', name: 'Productivity Gain Mathematical Invariant', requirement: 'Hours Saved = Volume * Delta-TTD invariant strictly satisfied' },
  { gateId: 'EAD-Gate-06', name: 'Accessibility & WCAG 2.2 AA Conformance', requirement: 'Zero axe violations, visible focus rings, ARIA landmarks' },
  { gateId: 'EAD-Gate-07', name: 'Keyboard Navigation & Responsive Layout', requirement: 'Full keyboard navigation and responsive viewport scaling (XS..2XL)' },
  { gateId: 'EAD-Gate-08', name: 'Telemetry Replay Determinism', requirement: '100 replay runs yield strictly identical snapshot hash' },
  { gateId: 'EAD-Gate-09', name: 'Navigation & Entity Resolver Integration', requirement: 'ADP prefix registered with canonical routing to /adoption-center' },
  { gateId: 'EAD-Gate-10', name: 'Production Build & Performance Budget', requirement: 'Static export clean, Shared JS strictly <= 100.0 kB' },
] as const;
