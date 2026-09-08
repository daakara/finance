/**
 * Phase 31-M8: Failover Orchestrator
 *
 * Implements:
 * - Failover routing for 8 Canonical Failure Classes
 * - Automatic Recovery Level resolution (L1, L2, L3, L4)
 * - RTO verification (< 60s for automatic failover)
 * - Degraded mode publication guard
 */

import {
  FailureClass,
  FailoverSeverity,
  FailoverEvent,
  FailoverResult,
  RecoveryLevel,
} from '../../types/resilience-intelligence';
import { activateRecoveryState, CANONICAL_RECOVERY_STATES } from './recoveryStateEngine';
import { sha256Hex } from '../governance/sha256';

export const FAILURE_CLASS_ROUTING: Record<
  FailureClass,
  { defaultLevel: RecoveryLevel; severity: FailoverSeverity; recoveryStateId: string }
> = {
  OPTIMIZATION_FAILURE: { defaultLevel: 'L3', severity: 'HIGH', recoveryStateId: 'RECSTATE-L3-FAILOVER' },
  FORECAST_FAILURE: { defaultLevel: 'L2', severity: 'MEDIUM', recoveryStateId: 'RECSTATE-L2-SNAPSHOT' },
  CONSISTENCY_FAILURE: { defaultLevel: 'L2', severity: 'HIGH', recoveryStateId: 'RECSTATE-L2-SNAPSHOT' },
  DATA_INTEGRITY_FAILURE: { defaultLevel: 'L2', severity: 'CRITICAL', recoveryStateId: 'RECSTATE-L2-SNAPSHOT' },
  TELEMETRY_OUTAGE: { defaultLevel: 'L1', severity: 'MEDIUM', recoveryStateId: 'RECSTATE-L1-REFRESH' },
  REPLAY_DRIFT: { defaultLevel: 'L4', severity: 'CRITICAL', recoveryStateId: 'RECSTATE-L4-SAFE' },
  RESOURCE_EXHAUSTION: { defaultLevel: 'L3', severity: 'HIGH', recoveryStateId: 'RECSTATE-L3-FAILOVER' },
  GOVERNANCE_VIOLATION: { defaultLevel: 'L4', severity: 'CRITICAL', recoveryStateId: 'RECSTATE-L4-SAFE' },
};

export function executeFailover(
  failureClass: FailureClass,
  customSeverity?: FailoverSeverity,
  impactedSystems: string[] = ['OHI_ENGINE', 'EXECUTIVE_DASHBOARD']
): { event: FailoverEvent; result: FailoverResult } {
  const route = FAILURE_CLASS_ROUTING[failureClass];
  const severity = customSeverity || route.severity;
  const failoverId = `FAIL-${Date.now().toString().slice(-4)}`;

  const event: FailoverEvent = {
    failoverId,
    failureClass,
    severity,
    impactedSystems,
    detectedAtUtc: new Date().toISOString(),
    autoRecoveryAttempted: true,
    recoveryStateId: route.recoveryStateId,
  };

  // Trigger Recovery
  const activation = activateRecoveryState(route.recoveryStateId);

  const result: FailoverResult = {
    failoverId,
    success: activation.success,
    recoveryStateActivated: true,
    activationDurationSeconds: activation.durationSeconds,
    resultingOHI: activation.recoveryState.projectedOHI,
    remainingRiskScore: activation.recoveryState.projectedRiskScore,
    replayHash: sha256Hex(`FAILOVER:${failoverId}:${failureClass}:${activation.durationSeconds}`),
  };

  return { event, result };
}

export function checkFailoverRTO(activationDurationSeconds: number): { pass: boolean; targetRTO: number } {
  const targetRTO = 60; // AC-M8-003: Recovery < 60 seconds
  return {
    pass: activationDurationSeconds < targetRTO,
    targetRTO,
  };
}
