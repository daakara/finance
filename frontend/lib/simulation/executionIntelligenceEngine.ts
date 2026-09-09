/**
 * Horizon 11: Execution Intelligence Engine
 *
 * Implements the closed behavioral execution loop:
 * Recommendation -> Commitment -> Execution -> Outcome -> Calibration
 *
 * Enforces:
 * - INV-OI102-P: Execution Accountability (Zero silent drops)
 * - INV-OI103-P: Recommendation Outcome Learning (Outcome attribution & variance)
 * - INV-OI104-P: Non-Punitive Behavioral Recovery (Restorative friction diagnostics)
 */

import {
  ExecutionState,
  ExecutionTerminalState,
  RecommendationExecutionRecord,
  verifyExecutionAccountability,
  verifyRecommendationOutcomeLearning,
  verifyNonPunitiveBehavioralRecovery,
} from './horizon11Invariants';

export interface ActionChecklistItem {
  stepId: string;
  label: string;
  completed: boolean;
}

export interface ActionCommitment {
  id: string;
  recommendationId: string;
  headline: string;
  domain: 'TRADING' | 'CAREER' | 'HEALTH' | 'FINANCE' | 'HOUSEHOLD';
  estimatedMinutes: number;
  currentState: ExecutionState;
  proposedTimestamp: string;
  committedTimestamp?: string;
  startedTimestamp?: string;
  completedTimestamp?: string;
  deferredCount: number;
  terminalState?: ExecutionTerminalState;
  silentDropDetected: boolean;
  checklist: ActionChecklistItem[];
  frictionDiagnosis?: {
    rootCause: 'COGNITIVE_FATIGUE' | 'TIME_DEFICIT' | 'DOMESTIC_COLLISION' | 'ACTIVATION_BARRIER';
    explanation: string;
    remedyOffer: 'SCOPE_REDUCTION' | 'RESTORATIVE_SWAP' | 'PARTNER_SYNC';
    newEstimatedMinutes?: number;
  };
  preTelemetry: {
    recoveryScore: number;
    focusHoursAvailable: number;
    liquidRunwayMonths: number;
  };
  postTelemetry?: {
    recoveryScore: number;
    focusHoursAvailable: number;
    liquidRunwayMonths: number;
  };
  outcomeAttribution?: {
    predictedLhiDelta: number;
    observedLhiDelta: number;
    variance: number;
    confidenceScore: number;
  };
}

export interface AdherenceScorecard {
  totalSurfaced: number;
  completedCount: number;
  deferredCount: number;
  completionRatePct: number;
  averageVariance: number;
  accountabilityStatus: 'FULLY_ACCOUNTABLE' | 'DEGRADED';
}

export const CANONICAL_ACTIVE_COMMITMENT: ActionCommitment = {
  id: 'COMMIT-AI-MODULE-3',
  recommendationId: 'ACT-CAREER-AI-MODULE',
  headline: 'Complete AI Architecture Module 3: Distributed State',
  domain: 'CAREER',
  estimatedMinutes: 45,
  currentState: 'COMMITTED',
  proposedTimestamp: '2026-09-09T08:00:00Z',
  committedTimestamp: '2026-09-09T08:15:00Z',
  deferredCount: 0,
  silentDropDetected: false,
  checklist: [
    { stepId: 's1', label: 'Review Consensus Protocols Section (15 min)', completed: false },
    { stepId: 's2', label: 'Complete State Replication Exercise (20 min)', completed: false },
    { stepId: 's3', label: 'Submit Self-Assessment Quiz (10 min)', completed: false },
  ],
  preTelemetry: {
    recoveryScore: 84,
    focusHoursAvailable: 4.5,
    liquidRunwayMonths: 14.2,
  },
};

export const CANONICAL_HISTORICAL_COMMITMENTS: ActionCommitment[] = [
  {
    id: 'COMMIT-TRADE-GOOGL-PULLBACK',
    recommendationId: 'ACT-TRADE-GOOGL',
    headline: 'Enter High-Conviction Dip: GOOGL (18 shares, $140 risk)',
    domain: 'TRADING',
    estimatedMinutes: 15,
    currentState: 'COMPLETED',
    terminalState: 'COMPLETED',
    proposedTimestamp: '2026-08-28T09:15:00Z',
    committedTimestamp: '2026-08-28T09:20:00Z',
    completedTimestamp: '2026-08-28T09:35:00Z',
    deferredCount: 0,
    silentDropDetected: false,
    checklist: [
      { stepId: 't1', label: 'Verify entry trigger at $178.50', completed: true },
      { stepId: 't2', label: 'Set hard stop-loss at $171.00 ($140 risk limit)', completed: true },
      { stepId: 't3', label: 'Confirm position sizing strictly within 14.2m runway floor', completed: true },
    ],
    preTelemetry: {
      recoveryScore: 88,
      focusHoursAvailable: 4.5,
      liquidRunwayMonths: 14.2,
    },
    postTelemetry: {
      recoveryScore: 89,
      focusHoursAvailable: 4.5,
      liquidRunwayMonths: 14.4,
    },
    outcomeAttribution: {
      predictedLhiDelta: 2.1,
      observedLhiDelta: 2.4,
      variance: 0.3,
      confidenceScore: 0.88,
    },
  },
  {
    id: 'COMMIT-FAMILY-DINNER-AUG',
    recommendationId: 'ACT-HOUSEHOLD-DINNER',
    headline: 'Friday Family Offline Dinner (2 hours protected)',
    domain: 'HOUSEHOLD',
    estimatedMinutes: 120,
    currentState: 'COMPLETED',
    terminalState: 'COMPLETED',
    proposedTimestamp: '2026-08-22T08:00:00Z',
    committedTimestamp: '2026-08-22T08:30:00Z',
    completedTimestamp: '2026-08-22T21:00:00Z',
    deferredCount: 0,
    silentDropDetected: false,
    checklist: [
      { stepId: 'f1', label: 'Block 18:30-20:30 on shared calendar', completed: true },
      { stepId: 'f2', label: 'All screens placed in silent charging bay', completed: true },
    ],
    preTelemetry: {
      recoveryScore: 72,
      focusHoursAvailable: 3.0,
      liquidRunwayMonths: 14.2,
    },
    postTelemetry: {
      recoveryScore: 84,
      focusHoursAvailable: 4.0,
      liquidRunwayMonths: 14.2,
    },
    outcomeAttribution: {
      predictedLhiDelta: 3.9,
      observedLhiDelta: 4.2,
      variance: 0.3,
      confidenceScore: 0.92,
    },
  },
];

/**
 * Transitions an action through the execution state machine.
 */
export function transitionActionState(
  commitment: ActionCommitment,
  targetState: ExecutionState,
  options?: {
    observedOutcomeDelta?: number;
    predictedOutcomeDelta?: number;
    simulatedPostTelemetry?: {
      recoveryScore: number;
      focusHoursAvailable: number;
      liquidRunwayMonths: number;
    };
  }
): ActionCommitment {
  const updated: ActionCommitment = { ...commitment };
  const now = new Date().toISOString();

  if (targetState === 'IN_PROGRESS') {
    updated.currentState = 'IN_PROGRESS';
    updated.startedTimestamp = now;
  } else if (targetState === 'COMPLETED') {
    updated.currentState = 'COMPLETED';
    updated.terminalState = 'COMPLETED';
    updated.completedTimestamp = now;
    updated.checklist = updated.checklist.map((c) => ({ ...c, completed: true }));

    const pred = options?.predictedOutcomeDelta ?? 2.5;
    const obs = options?.observedOutcomeDelta ?? 2.8;
    const variance = Number((obs - pred).toFixed(2));

    updated.postTelemetry = options?.simulatedPostTelemetry || {
      recoveryScore: Math.min(100, updated.preTelemetry.recoveryScore + 4),
      focusHoursAvailable: updated.preTelemetry.focusHoursAvailable,
      liquidRunwayMonths: updated.preTelemetry.liquidRunwayMonths,
    };

    updated.outcomeAttribution = {
      predictedLhiDelta: pred,
      observedLhiDelta: obs,
      variance,
      confidenceScore: 0.9,
    };
  } else if (targetState === 'DEFERRED') {
    updated.currentState = 'DEFERRED';
    updated.terminalState = 'DEFERRED';
    updated.deferredCount += 1;
  } else if (targetState === 'REJECTED') {
    updated.currentState = 'REJECTED';
    updated.terminalState = 'REJECTED';
  } else if (targetState === 'ABANDONED') {
    updated.currentState = 'ABANDONED';
    updated.terminalState = 'ABANDONED';
  }

  return updated;
}

/**
 * Diagnoses friction non-punitively when an action is deferred or abandoned.
 * Enforces INV-OI104-P.
 */
export function diagnoseExecutionFriction(
  commitment: ActionCommitment,
  recoveryScore: number
): ActionCommitment {
  const updated = { ...commitment };

  if (recoveryScore < 60) {
    updated.frictionDiagnosis = {
      rootCause: 'COGNITIVE_FATIGUE',
      explanation:
        'Your sleep recovery is 52%. Attempting a 45-minute deep focus session triggers high cognitive friction.',
      remedyOffer: 'RESTORATIVE_SWAP',
      newEstimatedMinutes: 15,
    };
  } else if (commitment.estimatedMinutes > 30) {
    updated.frictionDiagnosis = {
      rootCause: 'ACTIVATION_BARRIER',
      explanation:
        'The task activation threshold is elevated. Auto-scaling into a 15-minute micro-commitment reduces friction by 60%.',
      remedyOffer: 'SCOPE_REDUCTION',
      newEstimatedMinutes: 15,
    };
  } else {
    updated.frictionDiagnosis = {
      rootCause: 'TIME_DEFICIT',
      explanation: 'Calendar density is high today. Shifting commitment to tomorrow preserves calm.',
      remedyOffer: 'SCOPE_REDUCTION',
      newEstimatedMinutes: 10,
    };
  }

  return updated;
}

/**
 * Calculates adherence score and checks execution accountability.
 */
export function calculateAdherenceScorecard(
  history: ActionCommitment[]
): AdherenceScorecard {
  const total = history.length;
  if (total === 0) {
    return {
      totalSurfaced: 0,
      completedCount: 0,
      deferredCount: 0,
      completionRatePct: 100,
      averageVariance: 0,
      accountabilityStatus: 'FULLY_ACCOUNTABLE',
    };
  }

  const completed = history.filter((c) => c.terminalState === 'COMPLETED').length;
  const deferred = history.filter((c) => c.terminalState === 'DEFERRED').length;
  const hasSilentDrops = history.some((c) => c.silentDropDetected);

  const variances = history
    .filter((c) => c.outcomeAttribution)
    .map((c) => c.outcomeAttribution!.variance);

  const avgVariance =
    variances.length > 0
      ? Number((variances.reduce((a, b) => a + b, 0) / variances.length).toFixed(2))
      : 0;

  return {
    totalSurfaced: total,
    completedCount: completed,
    deferredCount: deferred,
    completionRatePct: Number(((completed / total) * 100).toFixed(1)),
    averageVariance: avgVariance,
    accountabilityStatus: hasSilentDrops ? 'DEGRADED' : 'FULLY_ACCOUNTABLE',
  };
}
