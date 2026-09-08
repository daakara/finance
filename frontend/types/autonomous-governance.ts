/**
 * Phase 31-M9: Autonomous Governance & Policy Intelligence Contracts
 *
 * Implements:
 * - AutonomousAction, AutonomousDecision, AutonomousExecutionRecord
 * - GovernancePolicy, PolicyRule, PolicyEvaluationResult, AutonomousActionRequest
 * - OverrideRequest, OverrideAuditRecord
 * - Typed Error Contracts: PolicyViolationError, AutonomousActionBlockedError, OverrideAuthorizationError, PolicyReplayFailureError
 * - Traceability Matrix (M9-Gate-01..M9-Gate-10)
 */

export type AutonomousActionCategory =
  | 'GOVERNANCE'
  | 'RISK'
  | 'LEARNING'
  | 'OPTIMIZATION'
  | 'RECOVERY';

export type AutonomousActionStatus =
  | 'PROPOSED'
  | 'APPROVED'
  | 'DENIED'
  | 'EXECUTING'
  | 'EXECUTED'
  | 'FAILED'
  | 'ROLLED_BACK'
  | 'PAUSED';

export interface AutonomousAction {
  actionId: string;
  category: AutonomousActionCategory;
  title: string;
  proposedAtUtc: string;
  approved: boolean;
  executed: boolean;
  policyApprovalId: string;
  expectedOutcome: string;
  rollbackAvailable: boolean;
  targetCommitteeId?: string;
  confidenceScore: number;
  status: AutonomousActionStatus;
  evidenceIds?: string[];
  parameters?: Record<string, unknown>;
}

export type ExecutionVerdict = 'APPROVED' | 'DENIED' | 'ESCALATED';

export interface AutonomousDecision {
  decisionId: string;
  actionId: string;
  evidenceIds: string[];
  policyRulesApplied: string[];
  confidenceScore: number;
  executionVerdict: ExecutionVerdict;
  rationale: string;
  decidedAtUtc: string;
}

export type ExecutionStatus = 'SUCCESS' | 'FAILED' | 'ROLLED_BACK';

export interface AutonomousExecutionRecord {
  executionId: string;
  actionId: string;
  executedAtUtc: string;
  executionStatus: ExecutionStatus;
  replayHash: string;
  executionDurationMs: number;
  attributionMetricDelta: number;
}

export type PolicyStatus = 'DRAFT' | 'ACTIVE' | 'RETIRED';
export type CertificationStatus = 'PASS' | 'FAIL';

export interface GovernancePolicy {
  policyId: string;
  policyName: string;
  version: string;
  status: PolicyStatus;
  effectiveFromUtc: string;
  effectiveUntilUtc?: string;
  rules: PolicyRule[];
  ownerId: string;
  certificationStatus: CertificationStatus;
}

export type PolicyRuleCategory =
  | 'RISK'
  | 'GOVERNANCE'
  | 'FINANCIAL'
  | 'COMPLIANCE'
  | 'AUTONOMY';

export type PolicyRuleAction = 'ALLOW' | 'DENY' | 'REQUIRE_APPROVAL';
export type PolicySeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface PolicyRule {
  ruleId: string;
  category: PolicyRuleCategory;
  action: PolicyRuleAction;
  condition: string;
  severity: PolicySeverity;
  thresholdValue?: number;
}

export interface AutonomousActionRequest {
  requestId: string;
  recommendationId: string;
  committeeId: string;
  initiatedAtUtc: string;
  proposedAction: string;
  rationale: string;
  policyEvaluationId: string;
  targetRiskScore?: number;
  budgetRequestedDollars?: number;
}

export type PolicyEvaluationDecision = 'APPROVED' | 'REJECTED' | 'REQUIRES_REVIEW';

export interface PolicyEvaluationResult {
  evaluationId: string;
  policyId: string;
  actionAllowed: boolean;
  blockingRules: string[];
  requiredApprovals: string[];
  decision: PolicyEvaluationDecision;
  riskScore: number;
  evaluatedAtUtc: string;
}

export type OverrideActionType = 'PAUSE' | 'CANCEL' | 'ROLLBACK' | 'FORCE_APPROVE';

export interface OverrideRequest {
  overrideId: string;
  policyId: string;
  actionId: string;
  justification: string;
  requestedBy: string;
  requestedAtUtc: string;
  overrideAction: OverrideActionType;
}

export interface OverrideAuditRecord {
  overrideId: string;
  approvedBy: string;
  approvedAtUtc: string;
  rationale: string;
  beforePolicyHash: string;
  afterPolicyHash: string;
  status: 'APPLIED' | 'REJECTED';
}

export interface SafetyMonitorAlert {
  alertId: string;
  actionId?: string;
  ruleId?: string;
  severity: PolicySeverity;
  alertType: 'POLICY_VIOLATION' | 'EXECUTION_DRIFT' | 'RISK_AMPLIFICATION' | 'CONTROL_FAILURE';
  description: string;
  detectedAtUtc: string;
  escalatedToHuman: boolean;
}

export interface PolicyErrorResponse {
  errorCode: string;
  errorType: string;
  message: string;
  policyId?: string;
  correlationId: string;
  timestampUtc: string;
}

export interface PolicyViolationError extends PolicyErrorResponse {
  errorType: 'POLICY_VIOLATION';
  violatedRuleId: string;
  severity: PolicySeverity;
}

export interface AutonomousActionBlockedError extends PolicyErrorResponse {
  errorType: 'AUTONOMOUS_ACTION_BLOCKED';
  blockingRules: string[];
  actionId: string;
}

export interface OverrideAuthorizationError extends PolicyErrorResponse {
  errorType: 'OVERRIDE_AUTHORIZATION_FAILURE';
  actorId: string;
  requiredRole: string;
}

export interface PolicyReplayFailureError extends PolicyErrorResponse {
  errorType: 'POLICY_REPLAY_FAILURE';
  expectedHash: string;
  actualHash: string;
}

export const CANONICAL_POLICIES: GovernancePolicy[] = [
  {
    policyId: 'POL-RISK-001',
    policyName: 'Capital At Risk Boundary Policy',
    version: '1.2.0',
    status: 'ACTIVE',
    effectiveFromUtc: '2026-01-01T00:00:00.000Z',
    ownerId: 'RISK-COMMITTEE',
    certificationStatus: 'PASS',
    rules: [
      {
        ruleId: 'RULE-RSK-01',
        category: 'RISK',
        action: 'DENY',
        condition: 'VaR 99% drawdown must not exceed 15.0%',
        severity: 'CRITICAL',
        thresholdValue: 15.0,
      },
      {
        ruleId: 'RULE-RSK-02',
        category: 'RISK',
        action: 'REQUIRE_APPROVAL',
        condition: 'Portfolio reallocation > $100,000 requires dual committee sign-off',
        severity: 'HIGH',
        thresholdValue: 100000,
      },
    ],
  },
  {
    policyId: 'POL-GOV-001',
    policyName: 'Institutional Quorum & Dissent Preservation',
    version: '2.0.0',
    status: 'ACTIVE',
    effectiveFromUtc: '2026-01-01T00:00:00.000Z',
    ownerId: 'BOARD-GOVERNANCE',
    certificationStatus: 'PASS',
    rules: [
      {
        ruleId: 'RULE-GOV-01',
        category: 'GOVERNANCE',
        action: 'DENY',
        condition: 'Decisions without dissent records are blocked if DIRatio < 0.70',
        severity: 'HIGH',
        thresholdValue: 0.70,
      },
      {
        ruleId: 'RULE-GOV-02',
        category: 'GOVERNANCE',
        action: 'REQUIRE_APPROVAL',
        condition: 'Emergency charter modifications mandate board confirmation',
        severity: 'CRITICAL',
      },
    ],
  },
  {
    policyId: 'POL-AUTONOMY-001',
    policyName: 'Autonomous Remediation & Self-Correction Limits',
    version: '1.0.0',
    status: 'ACTIVE',
    effectiveFromUtc: '2026-06-01T00:00:00.000Z',
    ownerId: 'CHIEF-OPERATING-OFFICER',
    certificationStatus: 'PASS',
    rules: [
      {
        ruleId: 'RULE-AUTO-01',
        category: 'AUTONOMY',
        action: 'ALLOW',
        condition: 'Autonomous cache invalidation and L1 telemetry refresh permitted if latency < 5s',
        severity: 'LOW',
      },
      {
        ruleId: 'RULE-AUTO-02',
        category: 'AUTONOMY',
        action: 'DENY',
        condition: 'Autonomous execution without rollback pathways is strictly prohibited',
        severity: 'CRITICAL',
      },
      {
        ruleId: 'RULE-AUTO-03',
        category: 'AUTONOMY',
        action: 'REQUIRE_APPROVAL',
        condition: 'Autonomous resource rebalancing > 10% requires human escalation',
        severity: 'HIGH',
        thresholdValue: 10.0,
      },
    ],
  },
];

export const CANONICAL_ACTIONS: AutonomousAction[] = [
  {
    actionId: 'ACT-2026-001',
    category: 'OPTIMIZATION',
    title: 'Autonomous Portfolio Variance Dampening',
    proposedAtUtc: '2026-09-08T10:00:00.000Z',
    approved: true,
    executed: true,
    policyApprovalId: 'EVAL-2026-001',
    expectedOutcome: 'Reduce 90-day volatility by 3.4% and elevate OHI +1.8 pts',
    rollbackAvailable: true,
    targetCommitteeId: 'COM-001',
    confidenceScore: 96.5,
    status: 'EXECUTED',
    evidenceIds: ['EVD-VAR-101', 'EVD-OHI-842'],
  },
  {
    actionId: 'ACT-2026-002',
    category: 'RECOVERY',
    title: 'L1 In-Memory Metric Cache Hot Reload',
    proposedAtUtc: '2026-09-08T11:15:00.000Z',
    approved: true,
    executed: true,
    policyApprovalId: 'EVAL-2026-002',
    expectedOutcome: 'Resolve telemetry desync within 4.2s (RTO < 5s)',
    rollbackAvailable: true,
    targetCommitteeId: 'COM-002',
    confidenceScore: 99.0,
    status: 'EXECUTED',
    evidenceIds: ['EVD-TEL-202'],
  },
  {
    actionId: 'ACT-2026-003',
    category: 'RISK',
    title: 'Counter-Groupthink Contrarian Mandate',
    proposedAtUtc: '2026-09-08T12:30:00.000Z',
    approved: true,
    executed: false,
    policyApprovalId: 'EVAL-2026-003',
    expectedOutcome: 'Inject independent contrarian reviewer for upcoming macro allocation',
    rollbackAvailable: true,
    targetCommitteeId: 'COM-001',
    confidenceScore: 92.0,
    status: 'APPROVED',
    evidenceIds: ['EVD-GT-303'],
  },
  {
    actionId: 'ACT-2026-004',
    category: 'GOVERNANCE',
    title: 'High-Impact Budget Overrun Override Attempt',
    proposedAtUtc: '2026-09-08T14:00:00.000Z',
    approved: false,
    executed: false,
    policyApprovalId: 'EVAL-2026-004',
    expectedOutcome: 'Blocked by Rule RULE-RSK-02: Exceeds $100k unreviewed threshold',
    rollbackAvailable: false,
    targetCommitteeId: 'COM-003',
    confidenceScore: 45.0,
    status: 'DENIED',
    evidenceIds: ['EVD-BUDGET-OVER'],
  },
];

export const M9_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'M9-Gate-01', name: 'Autonomous Safety Certification', invariant: 'INV-OI50', target: '100% Policy Pass' },
  { gateId: 'M9-Gate-02', name: 'Explainability Certification', invariant: 'INV-OI51', target: '100% Rationale Visibility' },
  { gateId: 'M9-Gate-03', name: 'Human Override Certification', invariant: 'INV-OI52', target: 'Instant Supersession (0s)' },
  { gateId: 'M9-Gate-04', name: 'Policy Boundary Certification', invariant: 'INV-OI53', target: '0 Unauthorized Actions' },
  { gateId: 'M9-Gate-05', name: 'Replay Determinism Certification', invariant: 'INV-OI54', target: '100 Replays -> 1 Hash' },
  { gateId: 'M9-Gate-06', name: 'Rollback Certification', invariant: 'INV-OI55', target: '100% Rollback Coverage' },
  { gateId: 'M9-Gate-07', name: 'Outcome Accountability Certification', invariant: 'INV-OI56', target: '100% Attribution Coverage' },
  { gateId: 'M9-Gate-08', name: 'Escalation Certification', invariant: 'INV-OI57', target: '100% Escalation Coverage' },
  { gateId: 'M9-Gate-09', name: 'Autonomous Governance Resilience', invariant: 'INV-OI47', target: 'All M8 Guards Preserved' },
  { gateId: 'M9-Gate-10', name: 'Master Autonomous Governance Certified', invariant: 'ALL_INVARIANTS', target: 'Full Regression Pass' },
];
