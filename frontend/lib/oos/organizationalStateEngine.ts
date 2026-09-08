/**
 * Phase 31-M6: Organizational State Engine
 *
 * Implements:
 * - Multi-Horizon State Transitions (CURRENT, FORECAST_30D, 90D, 180D, 365D)
 * - Invariant INV-OI37: Organizational Forecast Integrity (100 replays -> 1 hash)
 * - Invariant INV-OI38: Executive Readiness (Simultaneous Current, Projected, Risk, Learning states)
 * - Pure TypeScript SHA-256 State Hashing
 */

import type {
  OperatingMode,
  OrganizationalState,
  StateTransitionForecast,
} from '../../types/oos-intelligence';

import { sha256Hex } from '../governance/sha256';
import { CANONICAL_ORGANIZATIONAL_HEALTH_INDEX } from './organizationalHealthEngine';
import {
  CANONICAL_SYSTEM_HEALTH,
  CANONICAL_ENTERPRISE_ALERTS,
  CANONICAL_TELEMETRY_SNAPSHOT,
} from './organizationalTelemetryHub';

export function getOrganizationalStateByMode(mode: OperatingMode = 'CURRENT'): OrganizationalState {
  const baselineOHI = CANONICAL_ORGANIZATIONAL_HEALTH_INDEX;
  let ohiScore = baselineOHI.score;
  let activeRisks = 10;
  let criticalRisks = 4;
  let learningVelocity = 12.0;

  if (mode === 'FORECAST_30D') {
    ohiScore = 85.1;
    criticalRisks = 3;
    learningVelocity = 12.5;
  } else if (mode === 'FORECAST_90D') {
    ohiScore = 86.8;
    criticalRisks = 2;
    learningVelocity = 13.2;
  } else if (mode === 'FORECAST_180D') {
    ohiScore = 88.5;
    criticalRisks = 1;
    learningVelocity = 14.0;
  } else if (mode === 'FORECAST_365D') {
    ohiScore = 90.2;
    criticalRisks = 0;
    learningVelocity = 15.0;
  }

  const ohiInstance = {
    ...baselineOHI,
    score: ohiScore,
  };

  const state: OrganizationalState = {
    stateId: `ST-${mode}`,
    mode,
    asOfUtc: '2026-09-08T18:00:00Z',
    ohi: ohiInstance,
    systemHealth: CANONICAL_SYSTEM_HEALTH,
    activeRisksCount: activeRisks,
    criticalRisksCount: criticalRisks,
    learningVelocity,
    alerts: CANONICAL_ENTERPRISE_ALERTS,
    stateHash: '',
  };

  const payload = {
    id: state.stateId,
    mode: state.mode,
    score: state.ohi.score,
    risks: state.criticalRisksCount,
    velocity: state.learningVelocity,
  };
  state.stateHash = sha256Hex(JSON.stringify(payload));
  return state;
}

export function getStateTransitionForecast(
  fromStateId: string = 'ST-CURRENT',
  targetHorizon: '30D' | '90D' | '180D' | '365D' = '90D'
): StateTransitionForecast {
  const baseline = getOrganizationalStateByMode('CURRENT').ohi.score;
  const targetState = getOrganizationalStateByMode(`FORECAST_${targetHorizon}` as OperatingMode);
  const projectedOHI = targetState.ohi.score;
  const projectedDelta = Math.round((projectedOHI - baseline) * 10) / 10;

  let transitionProbabilityPct = 85.0;
  let lower = projectedOHI - 1.8;
  let upper = projectedOHI + 1.8;

  if (targetHorizon === '30D') {
    transitionProbabilityPct = 92.0;
    lower = projectedOHI - 0.8;
    upper = projectedOHI + 0.8;
  } else if (targetHorizon === '90D') {
    transitionProbabilityPct = 88.0;
    lower = projectedOHI - 1.5;
    upper = projectedOHI + 1.5;
  } else if (targetHorizon === '180D') {
    transitionProbabilityPct = 80.0;
    lower = projectedOHI - 2.4;
    upper = projectedOHI + 2.4;
  } else if (targetHorizon === '365D') {
    transitionProbabilityPct = 72.0;
    lower = projectedOHI - 3.8;
    upper = projectedOHI + 3.8;
  }

  const forecast: StateTransitionForecast = {
    fromStateId,
    targetHorizon,
    baselineOHI: baseline,
    projectedOHI,
    projectedDelta,
    transitionProbabilityPct,
    confidenceInterval: {
      lower: Math.round(lower * 10) / 10,
      upper: Math.round(upper * 10) / 10,
    },
    stateHash: '',
  };

  const payload = {
    from: forecast.fromStateId,
    horizon: forecast.targetHorizon,
    base: forecast.baselineOHI,
    proj: forecast.projectedOHI,
    prob: forecast.transitionProbabilityPct,
  };
  forecast.stateHash = sha256Hex(JSON.stringify(payload));
  return forecast;
}

// Invariant INV-OI37: Forecast Determinism & Replay Integrity
export function verifyINV_OI37(
  targetHorizon: '30D' | '90D' | '180D' | '365D' = '90D',
  replayRuns: number = 100
): {
  pass: boolean;
  uniqueHashCount: number;
  drift: number;
  hash: string;
} {
  const hashes = new Set<string>();

  for (let i = 0; i < replayRuns; i++) {
    const f = getStateTransitionForecast('ST-CURRENT', targetHorizon);
    hashes.add(f.stateHash);
  }

  const pass = hashes.size === 1;
  const hash = Array.from(hashes)[0];
  const drift = hashes.size - 1;

  return {
    pass,
    uniqueHashCount: hashes.size,
    drift,
    hash,
  };
}

// Invariant INV-OI38: Executive Readiness (Simultaneous State Availability)
export function verifyINV_OI38(): {
  pass: boolean;
  coverageRatio: number;
  activeStates: string[];
  violations: string[];
} {
  const violations: string[] = [];
  const requiredModes: OperatingMode[] = [
    'CURRENT',
    'FORECAST_30D',
    'FORECAST_90D',
    'FORECAST_180D',
    'FORECAST_365D',
  ];

  const states = requiredModes.map(m => getOrganizationalStateByMode(m));
  const validStates = states.filter(s => s.ohi && s.ohi.score > 0 && s.systemHealth);

  if (validStates.length !== requiredModes.length) {
    violations.push(`INV-OI38 Violation: Only ${validStates.length}/${requiredModes.length} operating states available`);
  }

  const coverageRatio = validStates.length / requiredModes.length;

  return {
    pass: violations.length === 0 && coverageRatio === 1.0,
    coverageRatio,
    activeStates: validStates.map(s => s.mode),
    violations,
  };
}
