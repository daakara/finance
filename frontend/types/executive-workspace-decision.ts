/**
 * Phase 31-M16: Executive Decision Workspace (ARX Horizon Executive OS) Contracts
 *
 * Implements:
 * - Full 8-Stage Unified Decision Lifecycle (Signal -> Package -> Options -> Governance -> Approval -> Execution -> Outcome -> Learning)
 * - 5 Executive Personalization Roles (RP-001..005)
 * - Personalization Guardrails (GP-001..006): Zero Fact Drift, Invariant Truth & Non-Hideable Critical Risks
 * - Multi-Center Intelligence Synthesis (OHI, ODEI, Risk, Resilience, Optimization, Simulation)
 * - Multi-Option Analysis (>=3 options with deltas and tradeoff ranking)
 * - Fail-Closed Governance Checklists & Cryptographic Sign-Off Receipts
 * - 100% Causal Driver Attribution (Positive/Negative summing strictly to 100.0%)
 * - Outcome Monitoring & Trajectory Divergence Tracking
 * - Learning Capture with Cryptographic Decision Provenance
 * - Board Briefings (Monthly, Quarterly, Decision Pack) with Deterministic Replay
 * - Master Certification Traceability (M16-Gate-01 through M16-Gate-16)
 */

export type ExecutiveDecisionRole =
  | 'Executive'
  | 'CommitteeChair'
  | 'Analyst'
  | 'Auditor'
  | 'GovernanceOfficer';

export type DecisionLifecycleStage =
  | 'SIGNAL'
  | 'PACKAGE'
  | 'OPTION_ANALYSIS'
  | 'GOVERNANCE_VALIDATION'
  | 'EXECUTIVE_APPROVAL'
  | 'EXECUTION'
  | 'OUTCOME_MONITORING'
  | 'LEARNING_CAPTURE';

export type PackageStatus =
  | 'DRAFT'
  | 'READY_FOR_APPROVAL'
  | 'APPROVED'
  | 'EXECUTING'
  | 'COMPLETED'
  | 'REJECTED'
  | 'FAILED';

export type DriverPolarity = 'POSITIVE' | 'NEGATIVE';

export interface DriverItem {
  id: string;
  name: string;
  category: string;
  percentage: number; // e.g. 35.0 (sum strictly 100.0)
  polarity: DriverPolarity;
  description: string;
  telemetrySource: string;
}

export interface IntelligenceCenterSynthesis {
  ohi: {
    baseline: number;
    projected: number;
    delta: number;
    trend: 'IMPROVING' | 'STABLE' | 'DEGRADING';
  };
  risk: {
    baselineScore: number;
    projectedScore: number;
    delta: number;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    topThreat: string;
  };
  resilience: {
    score: number;
    rtoMinutes: number;
    failoverStatus: 'ACTIVE' | 'STANDBY' | 'DEGRADED';
  };
  optimization: {
    capitalAllocationUSD: number;
    efficiencyGainPct: number;
    paybackMonths: number;
  };
  simulation: {
    stressScore: number;
    worstCaseVaRUSD: number;
    monteCarloConfidencePct: number;
  };
}

export interface DecisionOption {
  optionId: string;
  title: string;
  description: string;
  ohiDelta: number;
  odeiDelta: number;
  riskDelta: number;
  implementationCostUSD: number;
  confidencePct: number;
  tradeoffScore: number; // 0-100
  recommendationRank: number; // 1 = highest
  isRecommended: boolean;
  rationale: string;
  governanceCompliance: boolean;
}

export interface GovernanceRuleCheck {
  ruleId: string;
  name: string;
  category: 'SAFETY_BOUNDARY' | 'RISK_CEILING' | 'BUDGET_LIMIT' | 'QUORUM' | 'AUDIT_READINESS';
  threshold: string;
  observedValue: string;
  status: 'PASS' | 'FAIL' | 'WARN';
  failClosed: boolean;
}

export interface GovernanceValidationResult {
  passed: boolean;
  failClosed: boolean;
  ruleChecks: GovernanceRuleCheck[];
  failureReasons: string[];
  certifiedAtUtc: string;
  auditorSignoffRequired: boolean;
  stateHash: string;
}

export interface ApprovalReceipt {
  receiptId: string;
  packageId: string;
  approverRole: ExecutiveDecisionRole;
  approverName: string;
  timestampUtc: string;
  signatureHash: string;
  policyChecklistVerified: boolean;
  auditReceiptHash: string;
  status: 'APPROVED' | 'REJECTED' | 'BLOCKED_FAIL_CLOSE';
  lockId: string;
}

export interface OutcomeTrajectoryPoint {
  timestampUtc: string;
  expected: number;
  actual: number;
  metricName: string;
}

export interface OutcomeRecord {
  outcomeId: string;
  packageId: string;
  metricName: string;
  baselineValue: number;
  targetValue: number;
  actualValue: number;
  status: 'ON_TRACK' | 'AT_RISK' | 'DIVERGENT' | 'COMPLETED';
  divergencePct: number;
  trajectory: OutcomeTrajectoryPoint[];
  drivers: DriverItem[]; // Must sum to strictly 100.0%
  lastUpdatedUtc: string;
}

export interface LearningRecord {
  learningId: string;
  sourceDecisionId: string;
  sourceOutcomeId: string;
  title: string;
  lessonCategory: 'GOVERNANCE' | 'RISK_MANAGEMENT' | 'EXECUTION' | 'CAPITAL_ALLOCATION' | 'MODEL_CALIBRATION';
  insightText: string;
  provenanceHash: string;
  targetAdoptionRate: number; // e.g. 85.0%
  currentAdoptionRate: number;
  capturedAtUtc: string;
  status: 'RECORDED' | 'IN_TRIAL' | 'ADOPTED_INSTITUTIONAL';
}

export interface DecisionPackage {
  packageId: string;
  title: string;
  originatingCommittee: string;
  committeeId: string;
  urgency: 'CRITICAL' | 'HIGH' | 'MEDIUM';
  status: PackageStatus;
  currentStage: DecisionLifecycleStage;
  targetMetric: string;
  baselineMetricValue: number;
  projectedMetricValue: number;
  intelligenceSynthesis: IntelligenceCenterSynthesis;
  driverAttribution: DriverItem[]; // Strictly 100.0% sum
  options: DecisionOption[];
  governanceValidation: GovernanceValidationResult;
  approvalReceipt?: ApprovalReceipt;
  outcome?: OutcomeRecord;
  learning?: LearningRecord;
  stateHash: string;
  createdAtUtc: string;
  updatedAtUtc: string;
  assignedRoles: ExecutiveDecisionRole[];
  criticalRisks: string[]; // Can NEVER be hidden or filtered (GP-005)
}

export interface RoleLayoutConfig {
  role: ExecutiveDecisionRole;
  label: string;
  subtitle: string;
  badgeColor: string;
  focusOrdering: string[];
  defaultDensity: 'COMPACT' | 'COMFORTABLE' | 'DETAILED';
  priorityWidgets: string[];
  authorizedActions: string[];
  description: string;
}

export interface GuardrailComplianceResult {
  compliant: boolean;
  violations: string[];
  zeroFactDriftVerified: boolean;
  criticalRisksVisible: boolean;
  hashConsistencyVerified: boolean;
  governanceInvariantPreserved: boolean;
}

export interface BoardBriefingPack {
  briefingId: string;
  type: 'MONTHLY_BRIEF' | 'QUARTERLY_REPORT' | 'DECISION_PACK';
  generatedAtUtc: string;
  title: string;
  headline: string;
  executiveSummary: string;
  ohiTrendSummary: string;
  riskPostureSummary: string;
  completedDecisionsCount: number;
  activePackagesCount: number;
  openEscalationsCount: number;
  keyRecommendations: string[];
  replayHash: string;
}

export const M16_GATE_TRACEABILITY_MATRIX = [
  { gateId: 'M16-Gate-01', name: 'End-to-End Decision Lifecycle', requirement: 'Full 8-stage lifecycle execution on a single workspace page' },
  { gateId: 'M16-Gate-02', name: 'Multi-Center Intelligence Synthesis', requirement: 'Unified aggregation across OHI, Risk, Optimization, Resilience, Simulation' },
  { gateId: 'M16-Gate-03', name: 'Decision Package Integrity', requirement: 'Schema conformity, deterministic SHA-256 state hash, replay validation' },
  { gateId: 'M16-Gate-04', name: 'Option Comparison Matrix', requirement: 'Evaluation of >= 3 alternative options with delta projections and tradeoff ranks' },
  { gateId: 'M16-Gate-05', name: 'Role-Based Personalization', requirement: '5 distinct roles (Executive, CommitteeChair, Analyst, Auditor, GovernanceOfficer)' },
  { gateId: 'M16-Gate-06', name: 'Personalization Guardrails Compliance', requirement: 'Strict GP-001..006 zero fact drift; unhideable critical risks' },
  { gateId: 'M16-Gate-07', name: 'Fail-Closed Governance & Approval Gates', requirement: 'FAIL status strictly blocks execution; digital signatures recorded' },
  { gateId: 'M16-Gate-08', name: 'Immutable Audit Trail & Provenance', requirement: 'Cryptographic SHA-256 chain, tamper-evident action receipts' },
  { gateId: 'M16-Gate-09', name: 'Outcome Monitoring & Trajectory Tracking', requirement: 'Real-time baseline/target tracking, trajectory divergence detection' },
  { gateId: 'M16-Gate-10', name: '100% Driver Attribution', requirement: 'Causal drivers sum strictly to 100.0% with zero unallocated residuals' },
  { gateId: 'M16-Gate-11', name: 'Learning Capture & Provenance Linkage', requirement: 'Outcome-to-learning capture linked directly to decision ID and memory' },
  { gateId: 'M16-Gate-12', name: 'Narrative Intelligence & Board Briefings', requirement: 'Monthly, Quarterly, and Decision Pack briefings with replay hashes' },
  { gateId: 'M16-Gate-13', name: 'Single-Page Workflow Invariant', requirement: '100% of governance cycle completable without navigating away' },
  { gateId: 'M16-Gate-14', name: 'ARX Horizon Design System Compliance', requirement: 'Horizon tokens, WCAG 2.2 AA accessibility, responsive layout' },
  { gateId: 'M16-Gate-15', name: 'Cross-Role Consistency & Replay Determinism', requirement: '100 replays = 1 hash; identical metrics across all roles' },
  { gateId: 'M16-Gate-16', name: 'Platform Performance & Zero Regression', requirement: 'Clean Next.js build, First Load JS shared <= 100.0 kB, 0 test regressions' },
];
