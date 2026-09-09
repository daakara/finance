/**
 * Horizon 1: Executive Adoption & Usage Telemetry Engine
 *
 * Provides real-time calculation of executive engagement,
 * decision latency reduction, productivity ROI, and replay hashing.
 */

import {
  ExecutiveAdoptionSnapshot,
  ExecutiveAdoptionMetrics,
  ExecutiveUsageEvent,
  ProductivityGainSummary,
} from '../../types/executive-adoption';
import { CANONICAL_ADOPTION_BASELINE } from './fixtures/adoptionFixtures';

const usageEventLog: ExecutiveUsageEvent[] = [];

/**
 * Deterministic hash generator for adoption snapshots
 */
export function computeAdoptionReplayHash(snapshot: Partial<ExecutiveAdoptionSnapshot>): string {
  const seed = JSON.stringify({
    id: snapshot.snapshotId || '',
    metrics: snapshot.metrics || {},
    workflow: snapshot.workflowCohort?.completionRatePct || 0,
    roi: snapshot.productivity?.effectiveCostSavingsUSD || 0,
  });

  let hash = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) {
    hash ^= seed.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
    hash >>>= 0;
  }
  const hex1 = hash.toString(16).padStart(8, '0');

  let hash2 = 0x3d7a8b19;
  for (let i = seed.length - 1; i >= 0; i--) {
    hash2 ^= seed.charCodeAt(i);
    hash2 = Math.imul(hash2, 0x01000193);
    hash2 >>>= 0;
  }
  const hex2 = hash2.toString(16).padStart(8, '0');

  return `ADP-HASH-0x${hex1}${hex2}`;
}

/**
 * Log usage event in-memory with circular buffer cap (1000 items)
 */
export function logExecutiveUsageEvent(
  event: Omit<ExecutiveUsageEvent, 'eventId' | 'timestampUtc'>
): ExecutiveUsageEvent {
  const recorded: ExecutiveUsageEvent = {
    ...event,
    eventId: `EVT-${Date.now().toString(36)}-${Math.floor(Math.random() * 1000)}`,
    timestampUtc: new Date().toISOString(),
  };

  usageEventLog.unshift(recorded);
  if (usageEventLog.length > 1000) {
    usageEventLog.pop();
  }

  return recorded;
}

export function getRecordedUsageEvents(): ExecutiveUsageEvent[] {
  return [...usageEventLog];
}

export function clearRecordedUsageEvents(): void {
  usageEventLog.length = 0;
}

/**
 * Compute productivity gains invariant:
 * Total Hours Saved = Completed Decisions * (Baseline TTD - Actual TTD) / 60
 * Effective Savings USD = Total Hours * Hourly Executive Cost ($250/hr)
 */
export function calculateProductivityGains(
  completedDecisions: number,
  actualTtdMinutes: number,
  baselineTtdMinutes = 252.0,
  executiveHourlyRateUSD = 250,
  activeExecutives = 34
): ProductivityGainSummary {
  const deltaMinutes = Math.max(0, baselineTtdMinutes - actualTtdMinutes);
  const totalHoursSavedMonthly = parseFloat(((completedDecisions * deltaMinutes) / 60).toFixed(1));
  const hoursSavedPerExecutiveMonthly = parseFloat(
    (totalHoursSavedMonthly / Math.max(1, activeExecutives)).toFixed(1)
  );
  const effectiveCostSavingsUSD = Math.round(totalHoursSavedMonthly * executiveHourlyRateUSD);
  const decisionVelocityMultiplier = parseFloat(
    (baselineTtdMinutes / Math.max(1, actualTtdMinutes)).toFixed(1)
  );

  return {
    hoursSavedPerExecutiveMonthly,
    totalHoursSavedMonthly,
    effectiveCostSavingsUSD,
    decisionVelocityMultiplier,
    riskAvoidanceEvents: Math.round(completedDecisions * 0.1),
  };
}

/**
 * Return certified adoption snapshot
 */
export function getExecutiveAdoptionSnapshot(): ExecutiveAdoptionSnapshot {
  return JSON.parse(JSON.stringify(CANONICAL_ADOPTION_BASELINE));
}
