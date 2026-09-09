/**
 * Horizon 1: Executive Adoption Baseline Fixtures
 */

import {
  ExecutiveAdoptionSnapshot,
  ExecutiveUsageEvent,
} from '../../../types/executive-adoption';

export const CANONICAL_ADOPTION_BASELINE: ExecutiveAdoptionSnapshot = {
  snapshotId: 'ADP-SNAP-2026.09',
  generatedAtUtc: '2026-09-09T08:00:00Z',
  metrics: {
    dailyActiveExecutives: 34,
    monthlyActiveExecutives: 48,
    medianTimeToDecisionMinutes: 18.4,
    baselineTimeToDecisionMinutes: 252.0, // 4.2 hours legacy
    timeReductionPct: 92.7,
    totalActionsExecuted: 142,
    actionSlaAdherencePct: 94.6,
    totalBriefingsGenerated: 86,
    briefingReplayDeterminismPct: 100.0,
    searchSuccessRatePct: 96.8,
    averageSearchLatencyMs: 42,
    overallFeatureAdoptionPct: 88.5,
  },
  milestoneBreakdown: [
    { milestoneId: 'M11', milestoneName: 'Unified UX & Narrative Cockpit', activeUsers: 42, usageCount: 680, adoptionPct: 95.5, trend: 'RISING' },
    { milestoneId: 'M12', milestoneName: 'Strategic Simulation Foundation', activeUsers: 31, usageCount: 420, adoptionPct: 86.1, trend: 'RISING' },
    { milestoneId: 'M13', milestoneName: 'Unified Executive Home & Inbox', activeUsers: 46, usageCount: 950, adoptionPct: 97.9, trend: 'RISING' },
    { milestoneId: 'M14', milestoneName: 'Institutional Futures & Scenarios', activeUsers: 28, usageCount: 310, adoptionPct: 78.4, trend: 'RISING' },
    { milestoneId: 'M15', milestoneName: 'Overview & Universal Graph Explorer', activeUsers: 39, usageCount: 740, adoptionPct: 91.2, trend: 'RISING' },
    { milestoneId: 'M16', milestoneName: 'Executive Decision Workspace OS', activeUsers: 35, usageCount: 610, adoptionPct: 88.5, trend: 'RISING' },
  ],
  workflowCohort: {
    cohortId: 'COHORT-2026-Q3',
    totalWorkflowsInitiated: 156,
    totalWorkflowsCompleted: 142,
    completionRatePct: 91.0,
    medianCycleTimeMinutes: 18.4,
    stages: [
      { stageNumber: 1, stageName: 'Signal Detection', medianMinutes: 1.2, completionRatePct: 100.0, dropOffRatePct: 0.0, isBottleneck: false },
      { stageNumber: 2, stageName: 'Decision Packaging', medianMinutes: 2.8, completionRatePct: 98.1, dropOffRatePct: 1.9, isBottleneck: false },
      { stageNumber: 3, stageName: 'Option Tradeoff Analysis', medianMinutes: 4.5, completionRatePct: 96.2, dropOffRatePct: 1.9, isBottleneck: false },
      { stageNumber: 4, stageName: 'Governance Validation', medianMinutes: 2.1, completionRatePct: 95.5, dropOffRatePct: 0.7, isBottleneck: false },
      { stageNumber: 5, stageName: 'Executive Digital Sign-off', medianMinutes: 1.8, completionRatePct: 94.6, dropOffRatePct: 0.9, isBottleneck: false },
      { stageNumber: 6, stageName: 'Autonomous Execution Tranche', medianMinutes: 1.0, completionRatePct: 93.6, dropOffRatePct: 1.0, isBottleneck: false },
      { stageNumber: 7, stageName: 'Outcome Trajectory Tracking', medianMinutes: 3.2, completionRatePct: 92.3, dropOffRatePct: 1.3, isBottleneck: false },
      { stageNumber: 8, stageName: 'Organizational Learning Closure', medianMinutes: 1.8, completionRatePct: 91.0, dropOffRatePct: 1.3, isBottleneck: false },
    ],
  },
  productivity: {
    hoursSavedPerExecutiveMonthly: 23.4,
    totalHoursSavedMonthly: 795.6,
    effectiveCostSavingsUSD: 198900,
    decisionVelocityMultiplier: 13.7,
    riskAvoidanceEvents: 14,
  },
  replayHash: 'ADP-HASH-0x7a3f9b2c8e1d5a41',
};

export const SAMPLE_USAGE_EVENTS: ExecutiveUsageEvent[] = [
  {
    eventId: 'EVT-001',
    eventType: 'WORKSPACE_VISIT',
    userId: 'USR-EXEC-01',
    userRole: 'CIO',
    entityId: 'WS-CIO-001',
    durationMs: 420000,
    metadata: { route: '/executive-workspace', tab: 'decisions' },
    timestampUtc: '2026-09-09T07:10:00Z',
  },
  {
    eventId: 'EVT-002',
    eventType: 'INBOX_TRIAGE',
    userId: 'USR-EXEC-02',
    userRole: 'CEO',
    entityId: 'INBOX-01',
    durationMs: 180000,
    metadata: { itemsTriaged: 4, urgentApproved: 2 },
    timestampUtc: '2026-09-09T07:25:00Z',
  },
  {
    eventId: 'EVT-003',
    eventType: 'BRIEFING_GENERATED',
    userId: 'USR-EXEC-01',
    userRole: 'CIO',
    entityId: 'BRF-EXEC-01',
    durationMs: 45000,
    metadata: { type: 'MONTHLY_BOARD_BRIEF', replayDeterministic: true },
    timestampUtc: '2026-09-09T07:35:00Z',
  },
  {
    eventId: 'EVT-004',
    eventType: 'ACTION_EXECUTED',
    userId: 'USR-EXEC-03',
    userRole: 'CRO',
    entityId: 'ACT-2026-001',
    durationMs: 90000,
    metadata: { actionType: 'CIRCUIT_BREAKER_UPDATE', status: 'COMPLETED' },
    timestampUtc: '2026-09-09T07:42:00Z',
  },
  {
    eventId: 'EVT-005',
    eventType: 'GRAPH_EXPLORED',
    userId: 'USR-EXEC-04',
    userRole: 'COO',
    entityId: 'GRP-001',
    durationMs: 310000,
    metadata: { nodeDepth: 3, rootEntity: 'COM-001' },
    timestampUtc: '2026-09-09T07:50:00Z',
  },
  {
    eventId: 'EVT-006',
    eventType: 'RELEASE_VIEWED',
    userId: 'USR-EXEC-05',
    userRole: 'AUDIT_CHAIR',
    entityId: 'REL-2026.09-PROD',
    durationMs: 120000,
    metadata: { decisionVerified: 'APPROVED', readinessChecked: 98 },
    timestampUtc: '2026-09-09T07:58:00Z',
  },
];
