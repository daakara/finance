/**
 * Phase 31-M6: Organizational Operating System (OOS) Data Contracts
 *
 * Implements:
 * - Organizational Health Index (OHI) and Driver Envelopes (INV-OI33)
 * - Fail-Close Validation Categories OHI-VAL-001 through OHI-VAL-010
 * - Certification Self-Correction (CSC) Recovery Workflow Models
 * - Unified Telemetry Hub & Snapshot Models
 * - Executive Reporting Models & Invariant INV-OI36 Contract
 * - Cross-System Consistency Verification Models (INV-OI35)
 * - Multi-Horizon Organizational State Models (INV-OI37 & INV-OI38)
 */

export type OHIDriverCode = 'LV' | 'KT' | 'LF' | 'GT' | 'DQ' | 'GH';

export type OHIStatus = 'OPTIMAL' | 'STABLE' | 'WARNING' | 'ELEVATED_RISK' | 'CRITICAL';

export type DriverTrend = 'IMPROVING' | 'STABLE' | 'DEGRADING';

export interface OHIDriverMeasurement {
  driverId: OHIDriverCode;
  value: number;
  measuredAtUtc: string;
  sourceSnapshotId: string;
  certified: boolean;
}

export interface HealthDriver {
  driverId: OHIDriverCode;
  name: string;
  weight: number;
  rawValue: number;
  normalizedValue: number;
  weightedScore: number;
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL';
  trend: DriverTrend;
  targetFloor: number;
  explanation: string;
}

export interface HealthContribution {
  driverId: OHIDriverCode;
  weightPct: number;
  contributionPoints: number;
  percentageOfTotal: number;
}

export interface HealthForecast {
  horizon: '30D' | '90D' | '180D' | '365D';
  projectedOHI: number;
  confidenceInterval: {
    lower: number;
    upper: number;
  };
  confidencePct: number;
  primaryDrivers: string[];
  assumptions: string[];
}

export interface OrganizationalHealthIndex {
  ohiId: string;
  score: number;
  measuredAtUtc: string;
  status: OHIStatus;
  drivers: HealthDriver[];
  driverContributions: Record<OHIDriverCode, HealthContribution>;
  certificationPassed: boolean;
  violations: string[];
  stateHash: string;
}

// ── Fail-Close Validation Categories ─────────────────────────────────

export interface OHIMissingDriverError {
  errorCode: 'OHI-VAL-001';
  errorType: 'MISSING_DRIVER' | 'MULTIPLE_MISSING_OHI_DRIVERS';
  driverId?: string;
  missingDrivers?: string[];
  missingDriverCount?: number;
  message: string;
  correlationId: string;
  timestampUtc: string;
}

export interface OHIDuplicateDriverError {
  errorCode: 'OHI-VAL-002';
  errorType: 'DUPLICATE_DRIVER';
  driverId: string;
  timestampUtc: string;
  duplicates: number;
}

export interface OHINaNError {
  errorCode: 'OHI-VAL-003';
  errorType: 'INVALID_NUMERIC_VALUE';
  driverId: string;
  receivedValue: string;
}

export interface OHIInfiniteError {
  errorCode: 'OHI-VAL-004';
  errorType: 'NON_FINITE_DRIVER_VALUE';
  driverId: string;
  receivedValue: string;
}

export interface OHIFutureDatedError {
  errorCode: 'OHI-VAL-005';
  errorType: 'FUTURE_DATED_MEASUREMENT';
  driverId: string;
  measuredAtUtc: string;
}

export interface OHIStaleError {
  errorCode: 'OHI-VAL-006';
  errorType: 'STALE_MEASUREMENT';
  driverId: string;
  ageDays: number;
}

export interface OHIOutOfRangeError {
  errorCode: 'OHI-VAL-007';
  errorType: 'INVALID_DRIVER_RANGE';
  driverId: string;
  field: string;
  receivedValue: number;
  allowedRange: [number, number];
}

export interface OHICertificationMissingError {
  errorCode: 'OHI-VAL-008';
  errorType: 'CERTIFICATION_MISSING';
  driverId: string;
}

export interface OHITimestampDisorderError {
  errorCode: 'OHI-VAL-009';
  errorType: 'TIMESTAMP_DISORDER';
  message: string;
}

export interface OHICoverageFailureError {
  errorCode: 'OHI-VAL-010';
  errorType: 'COVERAGE_FAILURE';
  coveredDrivers: number;
  requiredDrivers: number;
  coverageRatio: number;
}

export type OHIValidationError =
  | OHIMissingDriverError
  | OHIDuplicateDriverError
  | OHINaNError
  | OHIInfiniteError
  | OHIFutureDatedError
  | OHIStaleError
  | OHIOutOfRangeError
  | OHICertificationMissingError
  | OHITimestampDisorderError
  | OHICoverageFailureError;

// ── CSC (Certification Self-Correction) Recovery Workflow Contracts ─

export type CSCRecoveryMode = 'AUTO_REPAIR' | 'MANUAL_REVIEW' | 'RECONSTRUCTION' | 'ROLLBACK';

export type CSCRecoveryStatus = 'QUEUED' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED';

export interface CSCRecoveryRequest {
  recoveryId: string;
  validationErrorCode: string;
  affectedDrivers: string[];
  initiatedAtUtc: string;
  recoveryMode: CSCRecoveryMode;
  actorId: string;
}

export interface CSCRecoveryResponse {
  recoveryId: string;
  status: CSCRecoveryStatus;
  startedAtUtc: string;
  completedAtUtc?: string;
  correctedArtifacts: string[];
  certificationRestored: boolean;
}

export interface CSCRecoveryAuditRecord {
  recoveryId: string;
  artifactId: string;
  beforeHash: string;
  afterHash: string;
  repairedAtUtc: string;
  repairedBy: string;
  validationCode: string;
}

export interface CSCRecoveryOutcome {
  recoveryId: string;
  success: boolean;
  restoredDrivers: string[];
  unresolvedDrivers: string[];
  certificationStatus: 'PASS' | 'FAIL';
}

// ── Unified Telemetry Hub & System Health ─────────────────────────────

export interface EnterpriseAlert {
  alertId: string;
  sourceSystem: 'COMMITTEE' | 'LEARNING' | 'RISK' | 'COACHING' | 'CONSISTENCY';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  title: string;
  description: string;
  timestampUtc: string;
  actionRequired: boolean;
  remediationPlaybookId?: string;
}

export interface TelemetrySnapshot {
  snapshotId: string;
  timestampUtc: string;
  committeeCount: number;
  averageODEI: number;
  averageCDQI: number;
  averageDIRatio: number;
  learningVelocity: number;
  knowledgeTransferRate: number;
  learningFrictionScore: number;
  groupthinkMaxScore: number;
  openCriticalRisks: number;
  activeRecommendations: number;
  coachImpactRatio: number;
  enterpriseAlertCount: number;
  stateHash: string;
}

export interface SystemHealth {
  status: 'OPTIMAL' | 'DEGRADED' | 'CRITICAL';
  overallScore: number;
  activeSubsystems: number;
  healthySubsystems: number;
  subsystemStatuses: Record<string, {
    status: 'ONLINE' | 'DEGRADED' | 'OFFLINE';
    score: number;
    lastPingUtc: string;
  }>;
}

// ── Executive Reporting Models (INV-OI36) ─────────────────────────────

export interface ReportFinding {
  findingId: string;
  finding: string;
  evidence: string[];
  trend: DriverTrend;
  risk: string;
  recommendation: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export interface ReportSection {
  sectionId: string;
  title: string;
  summary: string;
  metrics: Record<string, number | string>;
  findings: ReportFinding[];
}

export interface ExecutiveReport {
  reportId: string;
  title: string;
  reportType: 'EXECUTIVE_SUMMARY' | 'BOARD_REPORT' | 'QUARTERLY_REVIEW' | 'RISK_DIGEST';
  generatedAtUtc: string;
  reportingPeriod: string;
  sourceSnapshotId: string;
  ohiScore: number;
  executiveSummary: string;
  sections: ReportSection[];
  explainabilityCertified: boolean;
  stateHash: string;
}

// ── Cross-System Consistency (INV-OI35) ──────────────────────────────

export type CrossSystemSource = 'DASHBOARD' | 'API' | 'REPORT' | 'AUDIT' | 'FORECAST';

export interface CrossSystemComparison {
  metricName: string;
  values: Record<CrossSystemSource, number>;
  variance: number;
  isConsistent: boolean;
  status: 'PASS' | 'FAIL';
}

export interface ConsistencyVerificationResult {
  verifiedAtUtc: string;
  allSourcesEqual: boolean;
  overallVariance: number;
  comparisons: CrossSystemComparison[];
  violations: string[];
  certified: boolean;
}

// ── Multi-Horizon Organizational State Models (INV-OI37 & INV-OI38) ──

export type OperatingMode = 'CURRENT' | 'FORECAST_30D' | 'FORECAST_90D' | 'FORECAST_180D' | 'FORECAST_365D';

export interface OrganizationalState {
  stateId: string;
  mode: OperatingMode;
  asOfUtc: string;
  ohi: OrganizationalHealthIndex;
  systemHealth: SystemHealth;
  activeRisksCount: number;
  criticalRisksCount: number;
  learningVelocity: number;
  alerts: EnterpriseAlert[];
  stateHash: string;
}

export interface StateTransitionForecast {
  fromStateId: string;
  targetHorizon: '30D' | '90D' | '180D' | '365D';
  baselineOHI: number;
  projectedOHI: number;
  projectedDelta: number;
  transitionProbabilityPct: number;
  confidenceInterval: {
    lower: number;
    upper: number;
  };
  stateHash: string;
}
