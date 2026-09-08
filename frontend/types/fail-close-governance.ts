/**
 * Phase 31-M10: Autonomous Governance Safety & Fail-Close Error Architecture
 *
 * Implements:
 * - Base FailCloseErrorResponse & FailCloseStateResponse
 * - Override Violation Errors (GOV-OVR-001, GOV-OVR-002, GOV-OVR-003)
 * - Escalation Violation Errors (GOV-ESC-001, GOV-ESC-002, GOV-ESC-003, GOV-ESC-004)
 * - Policy Boundary Violation Errors (GOV-POL-001..GOV-POL-005)
 * - Operational Runbooks (M9-RB-01..M9-RB-07)
 * - Traceability Matrix (M10-Gate-01..M10-Gate-10)
 */

export type FailCloseErrorType =
  | 'OVERRIDE_VIOLATION'
  | 'ESCALATION_VIOLATION'
  | 'POLICY_BOUNDARY_VIOLATION';

export type FailCloseSeverity = 'HIGH' | 'CRITICAL';

export type CertificationImpact = 'DEGRADED' | 'FAILED';

export type RecoveryMode =
  | 'AUTO_REPAIR'
  | 'ROLLBACK'
  | 'SAFE_MODE'
  | 'MANUAL_REVIEW';

export interface FailCloseErrorResponse {
  errorCode: string;
  errorType: FailCloseErrorType;
  severity: FailCloseSeverity;
  certificationImpact: CertificationImpact;
  message: string;
  invariantId: string;
  affectedArtifactId: string;
  correlationId: string;
  detectedAtUtc: string;
  failCloseActivated: boolean;
  recoveryRequired: boolean;
  recommendedRecoveryMode: RecoveryMode;
}

// -------------------------------------------------------------
// OVERRIDE VIOLATION ERRORS
// -------------------------------------------------------------

export interface UnauthorizedOverrideError extends FailCloseErrorResponse {
  errorCode: 'GOV-OVR-001';
  errorType: 'OVERRIDE_VIOLATION';
  attemptedAction: string;
  actorId: string;
  requiredApprovalLevel: string;
}

export interface CertificationBypassError extends FailCloseErrorResponse {
  errorCode: 'GOV-OVR-002';
  errorType: 'OVERRIDE_VIOLATION';
  certificationGate: string;
  artifactStatus: string;
}

export interface HumanApprovalBypassError extends FailCloseErrorResponse {
  errorCode: 'GOV-OVR-003';
  errorType: 'OVERRIDE_VIOLATION';
  approvalWorkflowId: string;
  requiredApprovers: string[];
  actualApprovers: string[];
}

export type OverrideError =
  | UnauthorizedOverrideError
  | CertificationBypassError
  | HumanApprovalBypassError;

// -------------------------------------------------------------
// ESCALATION VIOLATION ERRORS
// -------------------------------------------------------------

export interface EscalationSuppressionError extends FailCloseErrorResponse {
  errorCode: 'GOV-ESC-001';
  errorType: 'ESCALATION_VIOLATION';
  alertId: string;
  severityLevel: string;
  escalationTarget: string;
}

export interface EscalationSlaViolationError extends FailCloseErrorResponse {
  errorCode: 'GOV-ESC-002';
  errorType: 'ESCALATION_VIOLATION';
  alertId: string;
  targetSlaHours: number;
  actualElapsedHours: number;
}

export interface EscalationRoutingError extends FailCloseErrorResponse {
  errorCode: 'GOV-ESC-003';
  errorType: 'ESCALATION_VIOLATION';
  escalationRole: string;
  alertId: string;
  routingAttempts: number;
}

export interface SeverityDowngradeViolationError extends FailCloseErrorResponse {
  errorCode: 'GOV-ESC-004';
  errorType: 'ESCALATION_VIOLATION';
  originalSeverity: string;
  modifiedSeverity: string;
  justificationPresent: boolean;
}

export type EscalationError =
  | EscalationSuppressionError
  | EscalationSlaViolationError
  | EscalationRoutingError
  | SeverityDowngradeViolationError;

// -------------------------------------------------------------
// POLICY BOUNDARY VIOLATION ERRORS
// -------------------------------------------------------------

export interface PolicyBoundaryViolationError extends FailCloseErrorResponse {
  errorCode: 'GOV-POL-001';
  errorType: 'POLICY_BOUNDARY_VIOLATION';
  policyId: string;
  attemptedAction: string;
  allowedActions: string[];
}

export interface AutonomousBoundaryViolationError extends FailCloseErrorResponse {
  errorCode: 'GOV-POL-002';
  errorType: 'POLICY_BOUNDARY_VIOLATION';
  autonomousAgentId: string;
  actionCategory: string;
  policyBoundaryId: string;
}

export interface RiskToleranceViolationError extends FailCloseErrorResponse {
  errorCode: 'GOV-POL-003';
  errorType: 'POLICY_BOUNDARY_VIOLATION';
  approvedRiskLimit: number;
  projectedRisk: number;
  excessRisk: number;
}

export interface DissentProtectionViolationError extends FailCloseErrorResponse {
  errorCode: 'GOV-POL-004';
  errorType: 'POLICY_BOUNDARY_VIOLATION';
  dissentIds: string[];
  suppressionAttemptDetected: boolean;
}

export interface ExplainabilityBoundaryViolationError extends FailCloseErrorResponse {
  errorCode: 'GOV-POL-005';
  errorType: 'POLICY_BOUNDARY_VIOLATION';
  artifactId: string;
  missingEvidenceCount: number;
  missingRationale: boolean;
}

export type PolicyViolationErrorM10 =
  | PolicyBoundaryViolationError
  | AutonomousBoundaryViolationError
  | RiskToleranceViolationError
  | DissentProtectionViolationError
  | ExplainabilityBoundaryViolationError;

export type AnyFailCloseError =
  | OverrideError
  | EscalationError
  | PolicyViolationErrorM10;

// -------------------------------------------------------------
// FAIL-CLOSE STATE RESPONSES
// -------------------------------------------------------------

export type FailCloseSystemState = 'CERTIFIED' | 'DEGRADED' | 'FAIL_CLOSED';

export interface FailCloseStateResponse {
  state: FailCloseSystemState;
  triggeringErrorCode: string;
  certificationRestored: boolean;
  activeRecoveryWorkflowId?: string;
  safeModeEnabled: boolean;
  blockedCapabilities: string[];
  timestampUtc: string;
}

// -------------------------------------------------------------
// OPERATIONAL RUNBOOKS (M9-RB-01 to M9-RB-07)
// -------------------------------------------------------------

export interface OperationalRunbook {
  runbookId: string;
  name: string;
  triggerDescription: string;
  alertCode: string;
  severity: FailCloseSeverity;
  automatedActions: string[];
  recoverySteps: string[];
  exitCriteria: string;
}

export const CANONICAL_OPERATIONAL_RUNBOOKS: OperationalRunbook[] = [
  {
    runbookId: 'M9-RB-01',
    name: 'Autonomous Governance Health Degradation',
    triggerDescription: 'OHI drops > 10% OR Governance Health < 75 OR Learning Velocity <= 0',
    alertCode: 'GOVERNANCE_HEALTH_DEGRADATION',
    severity: 'HIGH',
    automatedActions: [
      'Freeze new autonomous actions',
      'Increase monitoring frequency to 10s intervals',
      'Create governance incident INC-GOV-DEG',
    ],
    recoverySteps: [
      'Validate ODEI trend',
      'Audit transfer rate and institutional friction',
      'Verify replay integrity across all subsystems',
    ],
    exitCriteria: 'OHI > 80 AND Governance Health > 80 across 2 consecutive healthy evaluations',
  },
  {
    runbookId: 'M9-RB-02',
    name: 'Replay Drift Emergency Lock',
    triggerDescription: 'Replay hash mismatch OR Expected hashes > 1 across 100 replays',
    alertCode: 'REPLAY_DRIFT',
    severity: 'CRITICAL',
    automatedActions: [
      'Block all autonomous execution immediately',
      'Enable L4 Safe Mode',
      'Lock certification state to DEGRADED',
    ],
    recoverySteps: [
      'Reconstruct telemetry and decision snapshots',
      'Execute 100-replay verification certification drill',
      'Reissue authority root SHA-256 hash',
    ],
    exitCriteria: '100/100 identical hashes with Drift = 0',
  },
  {
    runbookId: 'M9-RB-03',
    name: 'Autonomous Recommendation Degradation',
    triggerDescription: 'Recommendation realization success rate < 60%',
    alertCode: 'RECOMMENDATION_DEGRADATION',
    severity: 'HIGH',
    automatedActions: [
      'Reduce recommendation confidence score cap to 65.0%',
      'Enable mandatory dual-human review gate',
      'Initiate recommendation model calibration loop',
    ],
    recoverySteps: [
      'Audit feature weights against realized outcome telemetry',
      'Run cross-committee bias validation',
      'Recalibrate Bayesian attribution model',
    ],
    exitCriteria: 'Recommendation success rate > 80% across 50 simulated interventions',
  },
  {
    runbookId: 'M9-RB-04',
    name: 'Scenario Survivability Failure',
    triggerDescription: 'Stress Score < 70 OR Failure Probability > 10%',
    alertCode: 'SURVIVABILITY_FAILURE',
    severity: 'HIGH',
    automatedActions: [
      'Suspend portfolio optimization executions',
      'Generate survivability incident INC-SURV-FAIL',
      'Trigger multi-factor stress recomputation',
    ],
    recoverySteps: [
      'Evaluate allocation defensive bounds',
      'Shift optimization focus to downside variance dampening',
      'Re-test against 4 canonical scenario regimes',
    ],
    exitCriteria: 'Stress Score >= 70 AND Failure Probability <= 5%',
  },
  {
    runbookId: 'M9-RB-05',
    name: 'Certified Snapshot Rollback',
    triggerDescription: 'Data corruption, NaN propagation, or certification failure detected',
    alertCode: 'DATA_CORRUPTION_ROLLBACK',
    severity: 'CRITICAL',
    automatedActions: [
      'Freeze state writes',
      'Identify latest certified immutable snapshot',
      'Initiate L2 Snapshot Recovery workflow',
    ],
    recoverySteps: [
      'Verify SHA-256 checksum of golden snapshot',
      'Restore snapshot into in-memory and durable storage',
      'Execute replay validation across affected engines',
    ],
    exitCriteria: 'PASS certification restored with 0 corruption evidence',
  },
  {
    runbookId: 'M9-RB-06',
    name: 'Autonomous Action Rollback',
    triggerDescription: 'Policy violation, override conflict, or unsafe recommendation mutation',
    alertCode: 'ACTION_ROLLBACK',
    severity: 'HIGH',
    automatedActions: [
      'Cancel pending action immediately',
      'Execute state inversion on mutated entities',
      'Write immutable rollback audit log',
    ],
    recoverySteps: [
      'Verify pre-action snapshot equivalence',
      'Escalate to governance committee review',
      'Flag action pattern for policy refinement',
    ],
    exitCriteria: 'System verified bit-for-bit identical to pre-action state',
  },
  {
    runbookId: 'M9-RB-07',
    name: 'Emergency Safe Mode Activation',
    triggerDescription: 'Unknown failure, critical governance incident, or policy engine outage',
    alertCode: 'SAFE_MODE_TRIGGER',
    severity: 'CRITICAL',
    automatedActions: [
      'Transition system into L4 Safe Mode',
      'Suspend all autonomous execution threads',
      'Convert recommendations to read-only advisory mode',
    ],
    recoverySteps: [
      'Mandate explicit human executive sign-off for resumption',
      'Conduct holistic post-incident audit reconstruction',
      'Re-certify all 10 M10 gates',
    ],
    exitCriteria: 'Executive governance charter sign-off and 10/10 gates green',
  },
];

// -------------------------------------------------------------
// M10 CERTIFICATION GATE MATRIX
// -------------------------------------------------------------

export interface M10CertificationGate {
  gateId: string;
  name: string;
  targetInvariant: string;
  targetRequirement: string;
  status: 'PASS' | 'FAIL';
}

export const M10_GATE_TRACEABILITY_MATRIX: M10CertificationGate[] = [
  { gateId: 'M10-Gate-01', name: 'Policy Enforcement', targetInvariant: 'INV-OI58 (INV-OI50)', targetRequirement: 'Zero execution without policy clearance', status: 'PASS' },
  { gateId: 'M10-Gate-02', name: 'Override Integrity', targetInvariant: 'INV-OI59 (INV-OI51)', targetRequirement: 'Zero bypass of human overrides (0s latency)', status: 'PASS' },
  { gateId: 'M10-Gate-03', name: 'Escalation Integrity', targetInvariant: 'INV-OI61 (INV-OI53)', targetRequirement: '100% Critical alert escalation, SLA enforcement', status: 'PASS' },
  { gateId: 'M10-Gate-04', name: 'Safe Termination', targetInvariant: 'INV-OI63 (INV-OI55)', targetRequirement: 'Termination before state mutation occurs', status: 'PASS' },
  { gateId: 'M10-Gate-05', name: 'Action Explainability', targetInvariant: 'INV-OI62 (INV-OI54)', targetRequirement: '100% Decision trace & rationale visibility', status: 'PASS' },
  { gateId: 'M10-Gate-06', name: 'Boundary Compliance', targetInvariant: 'INV-OI60 (INV-OI52)', targetRequirement: 'Zero unauthorized actions across limits', status: 'PASS' },
  { gateId: 'M10-Gate-07', name: 'Rollback Safety', targetInvariant: 'INV-OI55', targetRequirement: '100% Rollback coverage & state inversion', status: 'PASS' },
  { gateId: 'M10-Gate-08', name: 'Autonomous Auditability', targetInvariant: 'INV-OI54', targetRequirement: '100 Replays -> 1 Hash (0 drift)', status: 'PASS' },
  { gateId: 'M10-Gate-09', name: 'Human Governance Protection', targetInvariant: 'INV-OI59 (INV-OI51)', targetRequirement: 'Human supremacy guaranteed over automation', status: 'PASS' },
  { gateId: 'M10-Gate-10', name: 'Autonomous Governance Certification', targetInvariant: 'ALL_M10_INVARIANTS', targetRequirement: 'Full regression & fail-close preservation', status: 'PASS' },
];
