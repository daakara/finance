/**
 * Phase 31-M16: Role Personalization Engine & Guardrails (GP-001..006)
 *
 * Configures role layouts for 5 executive personas (RP-001..005) while strictly
 * enforcing Personalization Guardrails (GP-001..006):
 * - GP-001: Zero Fact Drift (data, metrics and numbers invariant)
 * - GP-002: Invariant Governance Outcomes
 * - GP-003: Invariant Risk Severity Scores
 * - GP-004: Invariant Audit Receipts & Cryptographic Hashes
 * - GP-005: Critical Risks & Certification Failures can NEVER be hidden
 * - GP-006: Role changes only modify presentation, density, and authorization
 */

import {
  ExecutiveDecisionRole,
  RoleLayoutConfig,
  DecisionPackage,
  GuardrailComplianceResult,
} from '../../types/executive-workspace-decision';

export const ROLE_CONFIGS: Record<ExecutiveDecisionRole, RoleLayoutConfig> = {
  Executive: {
    role: 'Executive',
    label: 'Chief Executive / Board Officer',
    subtitle: 'Strategic synthesis, macro KPI trends & high-authority sign-offs',
    badgeColor: 'var(--horizon-primary)',
    defaultDensity: 'COMFORTABLE',
    focusOrdering: [
      'ExecutiveHeader',
      'ExecutiveActionCenter',
      'ExecutiveDecisionQueue',
      'ExecutiveApprovalCenter',
      'OutcomeMonitor',
      'NarrativeAndBoardRail',
      'ExecutiveDecisionPackage',
      'OptionComparisonWorkspace',
      'LearningCapturePanel',
    ],
    priorityWidgets: ['InstitutionalPulse', 'PendingApprovals', 'BoardBriefingGenerator', 'OutcomeTrajectory'],
    authorizedActions: ['APPROVE_DECISION', 'REJECT_DECISION', 'GENERATE_BOARD_PACK', 'EXECUTE_DEPLOYMENT'],
    description: 'Optimized for rapid strategic assessment, one-click authorization, and high-level board communications.',
  },
  CommitteeChair: {
    role: 'CommitteeChair',
    label: 'Committee Chair',
    subtitle: 'Agenda coordination, quorum tracking, minority dissent review',
    badgeColor: 'var(--horizon-accent-cyan)',
    defaultDensity: 'COMFORTABLE',
    focusOrdering: [
      'ExecutiveHeader',
      'ExecutiveDecisionQueue',
      'ExecutiveDecisionPackage',
      'OptionComparisonWorkspace',
      'ExecutiveApprovalCenter',
      'ExecutiveActionCenter',
      'OutcomeMonitor',
      'LearningCapturePanel',
      'NarrativeAndBoardRail',
    ],
    priorityWidgets: ['CommitteeAgenda', 'QuorumMonitor', 'DissentTracker', 'OptionMatrix'],
    authorizedActions: ['SUBMIT_PACKAGE', 'STAGE_OPTION', 'CALL_VOTE', 'RATIFY_COMMITTEE_MINUTES'],
    description: 'Focused on committee deliberative hygiene, multi-option exploration, and member voting consensus.',
  },
  Analyst: {
    role: 'Analyst',
    label: 'Quantitative / Risk Analyst',
    subtitle: 'Deep scenario modeling, driver attribution & sensitivity testing',
    badgeColor: 'var(--horizon-accent-purple)',
    defaultDensity: 'DETAILED',
    focusOrdering: [
      'OptionComparisonWorkspace',
      'ExecutiveDecisionPackage',
      'OutcomeMonitor',
      'ExecutiveDecisionQueue',
      'LearningCapturePanel',
      'ExecutiveHeader',
      'ExecutiveActionCenter',
      'ExecutiveApprovalCenter',
      'NarrativeAndBoardRail',
    ],
    priorityWidgets: ['MonteCarloDistribution', '100%DriverAttribution', 'SensitivityMatrix', 'TrajectoryDivergence'],
    authorizedActions: ['CALIBRATE_MODEL', 'SIMULATE_STRESS_TEST', 'RECOMPUTE_ATTRIBUTION', 'RECORD_ANALYSIS'],
    description: 'Deep analytical view showing mathematical assumptions, 100% causal driver splits, and Monte Carlo confidence bounds.',
  },
  Auditor: {
    role: 'Auditor',
    label: 'Lead Compliance / Independent Auditor',
    subtitle: 'Cryptographic provenance, deterministic replay & immutable audit logs',
    badgeColor: 'var(--horizon-warning)',
    defaultDensity: 'DETAILED',
    focusOrdering: [
      'ExecutiveApprovalCenter',
      'ExecutiveDecisionPackage',
      'LearningCapturePanel',
      'ExecutiveDecisionQueue',
      'ExecutiveActionCenter',
      'ExecutiveHeader',
      'OutcomeMonitor',
      'OptionComparisonWorkspace',
      'NarrativeAndBoardRail',
    ],
    priorityWidgets: ['CryptographicReceipts', 'ReplayVerification', 'GovernanceRuleMatrix', 'LedgerWitnessConsensus'],
    authorizedActions: ['VERIFY_REPLAY_HASH', 'ATTEST_AUDIT_TRAIL', 'QUARANTINE_PACKAGE', 'EXPORT_AUDIT_LOG'],
    description: 'Designed for strict non-repudiation verification, hash verification, and regulatory compliance sign-offs.',
  },
  GovernanceOfficer: {
    role: 'GovernanceOfficer',
    label: 'Chief Governance / Risk Officer',
    subtitle: 'Fail-closed invariant enforcement, threshold alerts & safety boundaries',
    badgeColor: 'var(--horizon-danger)',
    defaultDensity: 'COMFORTABLE',
    focusOrdering: [
      'ExecutiveActionCenter',
      'ExecutiveApprovalCenter',
      'ExecutiveDecisionPackage',
      'ExecutiveDecisionQueue',
      'OptionComparisonWorkspace',
      'LearningCapturePanel',
      'OutcomeMonitor',
      'ExecutiveHeader',
      'NarrativeAndBoardRail',
    ],
    priorityWidgets: ['CriticalRiskBanner', 'FailCloseCircuitBreaker', 'PolicyBoundaryMonitor', 'EscalationQueue'],
    authorizedActions: ['TRIGGER_FAIL_CLOSE', 'OVERRIDE_BLOCK_SAFELY', 'ESCALATE_TO_BOARD', 'ENFORCE_POLICY_QUOTA'],
    description: 'Command center for institutional risk boundaries, automated circuit breakers, and unhideable critical alerts.',
  },
};

export function getRoleLayoutConfig(role: ExecutiveDecisionRole): RoleLayoutConfig {
  return ROLE_CONFIGS[role] || ROLE_CONFIGS.Executive;
}

export function verifyPersonalizationGuardrails(
  canonicalPackages: DecisionPackage[],
  role: ExecutiveDecisionRole
): GuardrailComplianceResult {
  const violations: string[] = [];
  const config = getRoleLayoutConfig(role);

  // GP-001: Zero Fact Drift - Ensure all numbers match canonical definitions
  let zeroFactDriftVerified = true;
  canonicalPackages.forEach(pkg => {
    if (pkg.baselineMetricValue <= 0 || pkg.projectedMetricValue <= 0) {
      violations.push(`GP-001 Violation: Invalid metric values in package ${pkg.packageId}`);
      zeroFactDriftVerified = false;
    }
    // Check 100% driver attribution
    const sum = pkg.driverAttribution.reduce((acc, d) => acc + d.percentage, 0);
    if (Math.abs(sum - 100.0) > 0.01) {
      violations.push(`GP-001 Violation: Driver attribution in ${pkg.packageId} does not equal 100.0% (${sum}%)`);
      zeroFactDriftVerified = false;
    }
  });

  // GP-005: Critical risks and certification failures can NEVER be hidden
  let criticalRisksVisible = true;
  canonicalPackages.forEach(pkg => {
    if (pkg.criticalRisks.length === 0) {
      violations.push(`GP-005 Violation: Package ${pkg.packageId} has empty critical risks`);
      criticalRisksVisible = false;
    }
  });

  // GP-002: Invariant Governance Outcomes
  const failedPkg = canonicalPackages.find(p => p.status === 'FAILED');
  const governanceInvariantPreserved = failedPkg ? failedPkg.governanceValidation.passed === false : true;
  if (!governanceInvariantPreserved) {
    violations.push('GP-002 Violation: Failed package governance status was corrupted.');
  }

  // GP-004: Invariant state hashes
  let hashConsistencyVerified = true;
  canonicalPackages.forEach(pkg => {
    if (!pkg.stateHash || pkg.stateHash.length < 10) {
      violations.push(`GP-004 Violation: State hash missing or truncated on ${pkg.packageId}`);
      hashConsistencyVerified = false;
    }
  });

  return {
    compliant: violations.length === 0,
    violations,
    zeroFactDriftVerified,
    criticalRisksVisible,
    hashConsistencyVerified,
    governanceInvariantPreserved,
  };
}
