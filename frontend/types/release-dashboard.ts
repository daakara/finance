/**
 * Phase 31: Executive Release-Gate Dashboard Contracts
 *
 * Implements:
 * - Release Readiness Response & Summary Schema
 * - 6 Institutional KPI Dimensions (Quality, Governance, Accessibility, Resilience, Performance, Executive Readiness)
 * - Release Gate Contract for M1-M16 Gates
 * - 4 Verification Pillars (Verification, Accessibility, Performance, Security)
 * - Release Decision Logic ('APPROVED' | 'CONDITIONAL' | 'BLOCKED')
 * - KPI Event Instrumentation Schema
 * - RGD Traceability Matrix (RGD-Gate-01 through RGD-Gate-15)
 */

export type ReleaseDecision = 'APPROVED' | 'CONDITIONAL' | 'BLOCKED';

export type GatePhase =
  | 'M1'
  | 'M2'
  | 'M3'
  | 'M4'
  | 'M5'
  | 'M6'
  | 'M7'
  | 'M8'
  | 'M9'
  | 'M10'
  | 'M11'
  | 'M12'
  | 'M13'
  | 'M14'
  | 'M15'
  | 'M16';

export type GateStatus = 'PASS' | 'FAIL' | 'WARNING' | 'BLOCKED';

export interface ReleaseGate {
  gateId: string;
  gateName: string;
  phase: GatePhase;
  owner: string;
  status: GateStatus;
  assertionCount: number;
  passedAssertions: number;
  executionDurationMs: number;
  lastVerifiedUtc: string;
  failureReason?: string;
}

export interface ReleaseSummary {
  certificationGatesPassed: number;
  certificationGatesTotal: number;
  totalAssertionsPassed: number;
  totalAssertionsExecuted: number;
  regressionSuitesPassed: number;
  regressionSuitesTotal: number;
}

export interface ReleaseKpiBundle {
  qualityScore: number;
  governanceScore: number;
  accessibilityScore: number;
  resilienceScore: number;
  performanceScore: number;
  executiveReadinessScore: number;
}

export interface VerificationSummary {
  totalAssertions: number;
  passedAssertions: number;
  failedAssertions: number;
  flakyTests: number;
  coveragePct: number;
}

export interface AccessibilitySummary {
  wcagLevel: 'AA' | 'AAA';
  axeViolations: number;
  keyboardNavigationPassed: boolean;
  focusManagementPassed: boolean;
  colorContrastPassed: boolean;
}

export interface PerformanceSummary {
  sharedJsKb: number;
  jsBudgetKb: number;
  staticRoutes: number;
  buildPassed: boolean;
  pageLoadSeconds: number;
}

export interface SecuritySummary {
  criticalVulnerabilities: number;
  replayDriftIncidents: number;
  consistencyViolations: number;
  governanceViolations: number;
}

export interface ReleaseReadinessResponse {
  releaseId: string;
  releaseVersion: string;
  generatedAtUtc: string;
  overallReadinessPct: number;
  releaseDecision: ReleaseDecision;
  summary: ReleaseSummary;
  kpis: ReleaseKpiBundle;
  gates: ReleaseGate[];
  verification: VerificationSummary;
  accessibility: AccessibilitySummary;
  performance: PerformanceSummary;
  security: SecuritySummary;
  replayHash: string;
}

export interface KpiEvent {
  eventId: string;
  eventType:
    | 'OHI_UPDATED'
    | 'RISK_UPDATED'
    | 'LEARNING_UPDATED'
    | 'ALLOCATION_COMPLETED'
    | 'BRIEF_GENERATED'
    | 'FAILOVER_EXECUTED';
  sourceSystem: string;
  metricName: string;
  metricValue: number;
  previousValue?: number;
  timestampUtc: string;
}

export const RGD_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'RGD-Gate-01', name: 'KPI Rendering & Completeness', requirement: 'All 6 institutional KPI metrics render with valid bounds' },
  { gateId: 'RGD-Gate-02', name: 'KPI Accuracy & Value Calibration', requirement: 'Metrics accurately calibrate against underlying test logs' },
  { gateId: 'RGD-Gate-03', name: 'Gate Visibility across M1-M16', requirement: '100% of milestone certification gates discoverable in grid' },
  { gateId: 'RGD-Gate-04', name: 'Gate Drilldown & Assertion Details', requirement: 'Interactive inspection of assertion counts, timing, and owners' },
  { gateId: 'RGD-Gate-05', name: 'Release Decision Logic', requirement: 'Strict fail-closed evaluation for APPROVED, CONDITIONAL, BLOCKED' },
  { gateId: 'RGD-Gate-06', name: 'Accessibility & WCAG 2.2 AA', requirement: 'Zero axe violations, visible focus rings, aria landmarks' },
  { gateId: 'RGD-Gate-07', name: 'Keyboard Navigation & Focus', requirement: 'Full keyboard tab accessibility across cards and filters' },
  { gateId: 'RGD-Gate-08', name: 'Responsive Layout across Viewports', requirement: 'Grid adapts across mobile (1 col), tablet (2 col), desktop (3-4 col)' },
  { gateId: 'RGD-Gate-09', name: 'API Contract Validation', requirement: 'Release readiness response strictly validates against schema' },
  { gateId: 'RGD-Gate-10', name: 'Fixture Determinism & Replay', requirement: '100 replays produce identical SHA-256 release hash' },
  { gateId: 'RGD-Gate-11', name: 'Pass State Rendering', requirement: 'Approved release displays green status with full attestation' },
  { gateId: 'RGD-Gate-12', name: 'Warning State Rendering', requirement: 'Conditional release displays amber warning banner and itemized risks' },
  { gateId: 'RGD-Gate-13', name: 'Fail State Rendering', requirement: 'Failed gates display prominent red status and root-cause reasons' },
  { gateId: 'RGD-Gate-14', name: 'Blocked Release Handling', requirement: 'Blocked release enforces zero-mutation lock and escalation guide' },
  { gateId: 'RGD-Gate-15', name: 'Master Dashboard Certification', requirement: 'Next.js build clean, First Load JS shared <= 100.0 kB, 0 regressions' },
];
