/**
 * Phase 31-M8: Autonomous Resilience, Scenario Robustness & Survivability Intelligence Contracts
 *
 * Implements:
 * - Recovery States (L1 Metric Refresh, L2 Snapshot Recovery, L3 Failover Calculation, L4 Executive Safe Mode)
 * - Recovery Plans & Recovery Steps with sequential orchestration and rollback
 * - Failover Events & Results across 8 canonical failure classes
 * - Strategy Survivability & Multi-Scenario Robustness modeling
 * - OHI Chaos Fixtures (OHI-FIX-001..005) & Recovery State definitions (RECSTATE-OHI-L1..L4)
 * - Machine-readable Gate-to-Test Traceability Matrix (M8-Gate-01..13)
 * - Executable Gherkin Step Definition contracts (Phases A-H)
 */

import type { ApiErrorResponse } from './coaching-intelligence';

export type RecoveryLevel = 'L1' | 'L2' | 'L3' | 'L4';

export type RecoveryStatus = 'STANDBY' | 'READY' | 'ACTIVATED' | 'FAILED' | 'RETIRED';

export interface RecoveryState {
  recoveryStateId: string;
  scenarioId: string;
  status: RecoveryStatus;
  recoveryLevel: RecoveryLevel;
  triggerCondition: string;
  projectedOHI: number;
  projectedRiskScore: number;
  confidencePct: number;
  fallbackStrategyId: string;
  activatedAtUtc?: string;
  createdAtUtc: string;
}

export interface RecoveryStep {
  stepId: string;
  sequence: number;
  description: string;
  ownerId: string;
  expectedDurationMinutes: number;
  mandatory: boolean;
}

export interface RecoveryPlan {
  planId: string;
  recoveryStateId: string;
  title: string;
  objective: string;
  estimatedRecoveryHours: number;
  steps: RecoveryStep[];
  successCriteria: string[];
  rollbackPlanId?: string;
}

export type FailureClass =
  | 'OPTIMIZATION_FAILURE'
  | 'FORECAST_FAILURE'
  | 'CONSISTENCY_FAILURE'
  | 'DATA_INTEGRITY_FAILURE'
  | 'TELEMETRY_OUTAGE'
  | 'REPLAY_DRIFT'
  | 'RESOURCE_EXHAUSTION'
  | 'GOVERNANCE_VIOLATION';

export type FailoverSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface FailoverEvent {
  failoverId: string;
  failureClass: FailureClass;
  severity: FailoverSeverity;
  impactedSystems: string[];
  detectedAtUtc: string;
  autoRecoveryAttempted: boolean;
  recoveryStateId?: string;
}

export interface FailoverResult {
  failoverId: string;
  success: boolean;
  recoveryStateActivated: boolean;
  activationDurationSeconds: number;
  resultingOHI: number;
  remainingRiskScore: number;
  replayHash: string;
}

export type SurvivabilityRating = 'LOW' | 'MEDIUM' | 'HIGH' | 'CERTIFIED';

export interface StrategySurvivability {
  strategyId: string;
  primaryScenarioScore: number;
  optimisticScenarioScore: number;
  adverseScenarioScore: number;
  stressScenarioScore: number;
  robustnessScore: number;
  failureProbabilityPct: number;
  survivabilityRating: SurvivabilityRating;
}

export type ScenarioType = 'BASE' | 'OPTIMISTIC' | 'ADVERSE' | 'STRESS';

export interface ScenarioDefinition {
  scenarioId: string;
  name: string;
  type: ScenarioType;
  perturbationFactor: number;
  description: string;
  expectedOHI: number;
  expectedRiskScore: number;
  probabilityPct: number;
}

export interface OHIFixtureHealthy {
  fixtureId: 'OHI-FIX-001';
  odei: number;
  cdqi: number;
  diRatio: number;
  learningVelocity: number;
  transferRate: number;
  groupthinkResistance: number;
  riskHealth: number;
  expectedOHI: number;
}

export interface OHIFixtureMissing {
  fixtureId: 'OHI-FIX-002';
  odei: null;
  expectedError: 'MISSING_OHI_DRIVER';
}

export interface OHIFixtureNaN {
  fixtureId: 'OHI-FIX-003';
  odei: 'NaN';
  expectedError: 'NAN_DRIVER_VALUE';
}

export interface OHIFixtureWeight {
  fixtureId: 'OHI-FIX-004';
  weightTotal: number;
  expectedError: 'INVALID_WEIGHT_CONFIGURATION';
}

export interface OHIFixtureDrift {
  fixtureId: 'OHI-FIX-005';
  hashCount: number;
  expectedError: 'OHI_REPLAY_DRIFT';
}

export type OHIFixture =
  | OHIFixtureHealthy
  | OHIFixtureMissing
  | OHIFixtureNaN
  | OHIFixtureWeight
  | OHIFixtureDrift;

export interface RecoveryAuditLog {
  auditId: string;
  failoverId: string;
  recoveryStateId: string;
  level: RecoveryLevel;
  initiatedAtUtc: string;
  completedAtUtc: string;
  durationSeconds: number;
  actorId: string;
  priorStateHash: string;
  recoveredStateHash: string;
  status: 'SUCCESS' | 'FAILED' | 'ROLLED_BACK';
}

// Canonical OHI Chaos Fixtures
export const CANONICAL_OHI_FIXTURES: {
  healthy: OHIFixtureHealthy;
  missingDriver: OHIFixtureMissing;
  nanCorruption: OHIFixtureNaN;
  weightCorruption: OHIFixtureWeight;
  replayDrift: OHIFixtureDrift;
} = {
  healthy: {
    fixtureId: 'OHI-FIX-001',
    odei: 84.0,
    cdqi: 82.0,
    diRatio: 78.0,
    learningVelocity: 86.0,
    transferRate: 88.0,
    groupthinkResistance: 92.0,
    riskHealth: 85.0,
    expectedOHI: 84.2,
  },
  missingDriver: {
    fixtureId: 'OHI-FIX-002',
    odei: null,
    expectedError: 'MISSING_OHI_DRIVER',
  },
  nanCorruption: {
    fixtureId: 'OHI-FIX-003',
    odei: 'NaN',
    expectedError: 'NAN_DRIVER_VALUE',
  },
  weightCorruption: {
    fixtureId: 'OHI-FIX-004',
    weightTotal: 122.4,
    expectedError: 'INVALID_WEIGHT_CONFIGURATION',
  },
  replayDrift: {
    fixtureId: 'OHI-FIX-005',
    hashCount: 2,
    expectedError: 'OHI_REPLAY_DRIFT',
  },
};

// Canonical OHI Recovery States
export const CANONICAL_OHI_RECOVERY_STATES = [
  {
    recoveryStateId: 'RECSTATE-OHI-L1',
    level: 'L1' as RecoveryLevel,
    strategy: 'REFRESH_DRIVER',
    description: 'Metric Refresh - re-queries live telemetry source',
  },
  {
    recoveryStateId: 'RECSTATE-OHI-L2',
    level: 'L2' as RecoveryLevel,
    strategy: 'CERTIFIED_SNAPSHOT_RESTORE',
    description: 'Snapshot Recovery - restores prior certified immutable snapshot',
  },
  {
    recoveryStateId: 'RECSTATE-OHI-L3',
    level: 'L3' as RecoveryLevel,
    strategy: 'SECONDARY_OHI_ENGINE',
    description: 'Failover Calculation - activates secondary OHI compute node',
  },
  {
    recoveryStateId: 'RECSTATE-OHI-L4',
    level: 'L4' as RecoveryLevel,
    strategy: 'PUBLISH_LAST_CERTIFIED_OHI',
    description: 'Executive Safe Mode - freezes last known certified OHI publication',
  },
];

// Machine-Readable Gate-to-Test Traceability Matrix
export const M8_GATE_TRACEABILITY_MATRIX: Record<
  string,
  { name: string; tests: string[] }
> = {
  'M8-Gate-01': {
    name: 'Scenario Coverage',
    tests: ['SCN-001', 'SCN-002', 'SCN-003', 'SCN-004'],
  },
  'M8-Gate-02': {
    name: 'Recovery State Certification',
    tests: ['CHAOS-OHI-01', 'CHAOS-OHI-02', 'CHAOS-OHI-03', 'REC-001', 'REC-002'],
  },
  'M8-Gate-03': {
    name: 'Failover Certification',
    tests: ['FAIL-001', 'FAIL-002', 'FAIL-003', 'CHAOS-CSC-01'],
  },
  'M8-Gate-04': {
    name: 'Strategy Survivability',
    tests: ['SURV-001', 'SURV-002', 'SURV-003', 'SURV-004'],
  },
  'M8-Gate-05': {
    name: 'Optimization Chaos Resistance',
    tests: [
      'OPT-FAIL-01',
      'OPT-FAIL-02',
      'OPT-FAIL-03',
      'OPT-FAIL-04',
      'OPT-FAIL-05',
      'OPT-REC-01',
      'OPT-REC-02',
    ],
  },
  'M8-Gate-06': {
    name: 'Forecast Chaos Resistance',
    tests: ['CHAOS-FOR-01', 'CHAOS-FOR-02', 'CHAOS-OHI-06'],
  },
  'M8-Gate-07': {
    name: 'Cross-System Resilience',
    tests: [
      'CSC-01',
      'CSC-02',
      'CSC-03',
      'CSC-FAIL-01',
      'CSC-FAIL-02',
      'CSC-REC-01',
      'CHAOS-OHI-05',
    ],
  },
  'M8-Gate-08': {
    name: 'Telemetry Recovery',
    tests: ['CHAOS-TEL-01', 'CHAOS-TEL-02'],
  },
  'M8-Gate-09': {
    name: 'Replay Determinism',
    tests: ['CHAOS-REP-01', 'CHAOS-REP-02', 'CHAOS-OHI-04'],
  },
  'M8-Gate-10': {
    name: 'Resource Exhaustion Recovery',
    tests: ['CHAOS-RES-01', 'CHAOS-RES-02'],
  },
  'M8-Gate-11': {
    name: 'Governance Protection',
    tests: ['CHAOS-GOV-01', 'CHAOS-GOV-02'],
  },
  'M8-Gate-12': {
    name: 'Executive Readiness',
    tests: ['READY-001', 'READY-002', 'READY-003', 'READY-004'],
  },
  'M8-Gate-13': {
    name: 'Organizational Resilience Certified',
    tests: ['E2E-001', 'E2E-002', 'E2E-003', 'E2E-004', 'E2E-005'],
  },
};
