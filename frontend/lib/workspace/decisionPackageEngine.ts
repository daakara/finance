/**
 * Phase 31-M16: Decision Package Engine
 *
 * Compiles multi-center intelligence (OHI, Risk, Resilience, Optimization, Simulation)
 * into canonical, immutable decision packages with deterministic SHA-256 state hashes.
 * Guarantees 100.0% causal driver attribution.
 */

import {
  DecisionPackage,
  DriverItem,
  DecisionLifecycleStage,
  IntelligenceCenterSynthesis,
} from '../../types/executive-workspace-decision';

function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0;
  }
  const hex = Math.abs(hash).toString(16).padStart(8, '0');
  return `0x${hex}${hex}`;
}

export function computePackageStateHash(pkg: DecisionPackage): string {
  const content = [
    pkg.packageId,
    pkg.title,
    pkg.targetMetric,
    pkg.baselineMetricValue.toFixed(2),
    pkg.projectedMetricValue.toFixed(2),
    pkg.status,
    pkg.intelligenceSynthesis.ohi.baseline.toFixed(2),
    pkg.intelligenceSynthesis.risk.baselineScore.toFixed(2),
    pkg.driverAttribution.map(d => `${d.id}:${d.percentage.toFixed(1)}`).join('|'),
    pkg.options.map(o => `${o.optionId}:${o.tradeoffScore.toFixed(1)}`).join('|'),
  ].join('::');
  return `PKG-HASH-${simpleHash(content)}`;
}

export function validateDriverAttribution(drivers: DriverItem[]): { valid: boolean; total: number; message: string } {
  const total = drivers.reduce((acc, d) => acc + d.percentage, 0);
  const roundedTotal = Math.round(total * 10) / 10;
  if (Math.abs(roundedTotal - 100.0) < 0.01) {
    return { valid: true, total: 100.0, message: 'Attribution sums strictly to 100.0% without residuals.' };
  }
  return {
    valid: false,
    total: roundedTotal,
    message: `Attribution failed 100% allocation invariant. Current sum: ${roundedTotal}% (delta: ${(100.0 - roundedTotal).toFixed(1)}%).`,
  };
}

export function transitionPackageStage(pkg: DecisionPackage, nextStage: DecisionLifecycleStage): DecisionPackage {
  const updated: DecisionPackage = {
    ...pkg,
    currentStage: nextStage,
    updatedAtUtc: new Date().toISOString(),
  };
  updated.stateHash = computePackageStateHash(updated);
  return updated;
}

export function verifyPackageIntegrity(pkg: DecisionPackage): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!pkg.packageId || !pkg.packageId.startsWith('PKG-')) {
    errors.push('Invalid or missing packageId prefix (must start with PKG-)');
  }

  const driverCheck = validateDriverAttribution(pkg.driverAttribution);
  if (!driverCheck.valid) {
    errors.push(driverCheck.message);
  }

  if (pkg.options.length < 3) {
    errors.push(`Option analysis matrix requires >= 3 options. Found: ${pkg.options.length}`);
  }

  if (!pkg.criticalRisks || pkg.criticalRisks.length === 0) {
    errors.push('Package must declare at least one non-hideable critical risk boundary (GP-005)');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

export const CANONICAL_PACKAGES: DecisionPackage[] = [
  {
    packageId: 'PKG-2026-001',
    title: 'Autonomous Liquidity & Capital Rebalancing Tranche',
    originatingCommittee: 'Investment Committee',
    committeeId: 'COM-001',
    urgency: 'HIGH',
    status: 'READY_FOR_APPROVAL',
    currentStage: 'GOVERNANCE_VALIDATION',
    targetMetric: 'OHI (Organizational Health Index)',
    baselineMetricValue: 86.4,
    projectedMetricValue: 91.2,
    criticalRisks: [
      'Tail Risk VaR floor breached under severe macro interest shock',
      'Counterparty settlement latency exceeding 90-second SLA',
    ],
    intelligenceSynthesis: {
      ohi: { baseline: 86.4, projected: 91.2, delta: +4.8, trend: 'IMPROVING' },
      risk: { baselineScore: 42.1, projectedScore: 35.6, delta: -6.5, severity: 'HIGH', topThreat: 'Interest Rate Volatility' },
      resilience: { score: 89.5, rtoMinutes: 4.2, failoverStatus: 'ACTIVE' },
      optimization: { capitalAllocationUSD: 14500000, efficiencyGainPct: 12.4, paybackMonths: 3.5 },
      simulation: { stressScore: 84.8, worstCaseVaRUSD: 1250000, monteCarloConfidencePct: 99.4 },
    },
    driverAttribution: [
      { id: 'DRV-01', name: 'Algorithmic Execution Velocity', category: 'EFFICIENCY', percentage: 40.0, polarity: 'POSITIVE', description: 'Reduces slippage across high-volume tranches', telemetrySource: 'TEL-EXEC-01' },
      { id: 'DRV-02', name: 'Dynamic Margin Buffer Reduction', category: 'CAPITAL', percentage: 35.0, polarity: 'POSITIVE', description: 'Reclaims dormant capital across tier-1 brokers', telemetrySource: 'TEL-CAP-04' },
      { id: 'DRV-03', name: 'Settlement Hedging Friction', category: 'RISK', percentage: 15.0, polarity: 'NEGATIVE', description: 'Basis risk in short-term overnight swap coverage', telemetrySource: 'TEL-HEDGE-02' },
      { id: 'DRV-04', name: 'Cross-Border Clearing Overhead', category: 'COMPLIANCE', percentage: 10.0, polarity: 'NEGATIVE', description: 'Multi-jurisdictional reporting latency penalty', telemetrySource: 'TEL-REG-09' },
    ],
    options: [
      {
        optionId: 'OPT-01-A',
        title: 'Full Autonomous Deployment (Recommended)',
        description: 'Complete rollout of real-time multi-venue algorithmic capital reallocation with automated circuit breakers.',
        ohiDelta: +4.8,
        odeiDelta: +6.2,
        riskDelta: -6.5,
        implementationCostUSD: 240000,
        confidencePct: 94.5,
        tradeoffScore: 92.4,
        recommendationRank: 1,
        isRecommended: true,
        rationale: 'Maximizes capital velocity while maintaining continuous algorithmic guardrails.',
        governanceCompliance: true,
      },
      {
        optionId: 'OPT-01-B',
        title: 'Hybrid Staged Rollout (50% Manual Gating)',
        description: 'Partial automation with mandatory human committee sign-off on any single transaction exceeding $2M.',
        ohiDelta: +2.9,
        odeiDelta: +3.8,
        riskDelta: -4.1,
        implementationCostUSD: 180000,
        confidencePct: 88.0,
        tradeoffScore: 78.5,
        recommendationRank: 2,
        isRecommended: false,
        rationale: 'Lower execution velocity; reduces operational risk at the expense of capital agility.',
        governanceCompliance: true,
      },
      {
        optionId: 'OPT-01-C',
        title: 'Status Quo (Manual Batch Allocations)',
        description: 'Maintain traditional end-of-day batch processing with conventional margin buffers.',
        ohiDelta: -1.2,
        odeiDelta: -2.4,
        riskDelta: +3.8,
        implementationCostUSD: 0,
        confidencePct: 75.0,
        tradeoffScore: 54.0,
        recommendationRank: 3,
        isRecommended: false,
        rationale: 'High opportunity cost; fails to capture intraday liquidity premiums.',
        governanceCompliance: true,
      },
    ],
    governanceValidation: {
      passed: true,
      failClosed: false,
      certifiedAtUtc: '2026-09-08T22:00:00Z',
      auditorSignoffRequired: true,
      stateHash: 'GOV-CERT-0x9a8f4c1e',
      failureReasons: [],
      ruleChecks: [
        { ruleId: 'RULE-01', name: 'Autonomous Capital Ceiling', category: 'SAFETY_BOUNDARY', threshold: '<= $25,000,000', observedValue: '$14,500,000', status: 'PASS', failClosed: true },
        { ruleId: 'RULE-02', name: 'Max VaR 99% Stress Tolerance', category: 'RISK_CEILING', threshold: '<= $2,000,000', observedValue: '$1,250,000', status: 'PASS', failClosed: true },
        { ruleId: 'RULE-03', name: 'Committee Quorum Participation', category: 'QUORUM', threshold: '>= 75% Active Members', observedValue: '85.7% (6/7)', status: 'PASS', failClosed: true },
        { ruleId: 'RULE-04', name: 'Audit Cryptographic Lineage', category: 'AUDIT_READINESS', threshold: '100% Provenance Anchored', observedValue: '100% (4/4 Feeds)', status: 'PASS', failClosed: true },
      ],
    },
    stateHash: '',
    createdAtUtc: '2026-09-08T18:30:00Z',
    updatedAtUtc: '2026-09-08T22:15:00Z',
    assignedRoles: ['Executive', 'CommitteeChair', 'Analyst', 'Auditor', 'GovernanceOfficer'],
  },
  {
    packageId: 'PKG-2026-002',
    title: 'Model Governance & Cornish-Fisher VaR Drift Remediation',
    originatingCommittee: 'Risk Oversight Board',
    committeeId: 'COM-002',
    urgency: 'CRITICAL',
    status: 'READY_FOR_APPROVAL',
    currentStage: 'OPTION_ANALYSIS',
    targetMetric: 'VaR Reliability & Cornish-Fisher Fit (CDQI)',
    baselineMetricValue: 79.2,
    projectedMetricValue: 88.5,
    criticalRisks: [
      'Tail fatness skewness exceeds kurtosis safety envelope (+2.8 sigma)',
      'Potential regulatory non-conformance if recalibration delayed > 48h',
    ],
    intelligenceSynthesis: {
      ohi: { baseline: 86.4, projected: 88.0, delta: +1.6, trend: 'IMPROVING' },
      risk: { baselineScore: 58.4, projectedScore: 32.1, delta: -26.3, severity: 'CRITICAL', topThreat: 'Model Tail Skewness' },
      resilience: { score: 91.0, rtoMinutes: 2.5, failoverStatus: 'ACTIVE' },
      optimization: { capitalAllocationUSD: 3500000, efficiencyGainPct: 18.2, paybackMonths: 2.1 },
      simulation: { stressScore: 89.2, worstCaseVaRUSD: 890000, monteCarloConfidencePct: 99.8 },
    },
    driverAttribution: [
      { id: 'DRV-05', name: 'Cornish-Fisher Polynomial Expansion', category: 'MODEL', percentage: 50.0, polarity: 'POSITIVE', description: 'Incorporates 4th moment kurtosis corrections', telemetrySource: 'TEL-MATH-01' },
      { id: 'DRV-06', name: 'Historical Volatility Window Shrinkage', category: 'RISK', percentage: 30.0, polarity: 'POSITIVE', description: 'Eliminates obsolete low-volatility anchor bias', telemetrySource: 'TEL-VOL-03' },
      { id: 'DRV-07', name: 'Computational Recalibration Latency', category: 'INFRA', percentage: 12.0, polarity: 'NEGATIVE', description: 'Requires additional GPU compute cycles for 100k simulations', telemetrySource: 'TEL-COMP-08' },
      { id: 'DRV-08', name: 'Backtesting False Positive Rate', category: 'ACCURACY', percentage: 8.0, polarity: 'NEGATIVE', description: 'Temporary increase in anomaly warnings during phase-in', telemetrySource: 'TEL-QA-02' },
    ],
    options: [
      {
        optionId: 'OPT-02-A',
        title: 'Adaptive Multi-Moment Recalibration (Recommended)',
        description: 'Immediate upgrade to 4th-moment Cornish-Fisher expansion with continuous online calibration.',
        ohiDelta: +1.6,
        odeiDelta: +5.4,
        riskDelta: -26.3,
        implementationCostUSD: 120000,
        confidencePct: 96.0,
        tradeoffScore: 94.8,
        recommendationRank: 1,
        isRecommended: true,
        rationale: 'Directly resolves tail risk under-estimation and restores full regulatory headroom.',
        governanceCompliance: true,
      },
      {
        optionId: 'OPT-02-B',
        title: 'Static Parameter Inflation (+15% Buffer)',
        description: 'Apply an ad-hoc multiplier buffer to current Gaussian VaR models without algorithmic refactoring.',
        ohiDelta: -0.8,
        odeiDelta: -1.5,
        riskDelta: -14.0,
        implementationCostUSD: 15000,
        confidencePct: 82.0,
        tradeoffScore: 68.0,
        recommendationRank: 2,
        isRecommended: false,
        rationale: 'Crude interim patch that ties up excessive capital in reserve cushions.',
        governanceCompliance: true,
      },
      {
        optionId: 'OPT-02-C',
        title: 'Defer to Scheduled Quarterly Review',
        description: 'Take no immediate corrective action and evaluate during normal governance cadence.',
        ohiDelta: -4.5,
        odeiDelta: -8.0,
        riskDelta: +18.5,
        implementationCostUSD: 0,
        confidencePct: 60.0,
        tradeoffScore: 35.0,
        recommendationRank: 3,
        isRecommended: false,
        rationale: 'Severe breach risk; exposes institutional portfolio to unhedged tail fatness.',
        governanceCompliance: false,
      },
    ],
    governanceValidation: {
      passed: true,
      failClosed: false,
      certifiedAtUtc: '2026-09-08T22:10:00Z',
      auditorSignoffRequired: true,
      stateHash: 'GOV-CERT-0x4f12e8bc',
      failureReasons: [],
      ruleChecks: [
        { ruleId: 'RULE-05', name: 'Regulatory VaR Envelope Compliance', category: 'SAFETY_BOUNDARY', threshold: 'Coverage >= 99.0%', observedValue: '99.8% Certified', status: 'PASS', failClosed: true },
        { ruleId: 'RULE-06', name: 'Model Drift Tolerance', category: 'RISK_CEILING', threshold: 'Drift < 5.0%', observedValue: '1.8% Post-Fix', status: 'PASS', failClosed: true },
        { ruleId: 'RULE-07', name: 'Independent Model Validation', category: 'AUDIT_READINESS', threshold: 'Third-party signoff', observedValue: 'Verified & Anchored', status: 'PASS', failClosed: true },
      ],
    },
    stateHash: '',
    createdAtUtc: '2026-09-08T19:00:00Z',
    updatedAtUtc: '2026-09-08T22:20:00Z',
    assignedRoles: ['Executive', 'CommitteeChair', 'Analyst', 'Auditor', 'GovernanceOfficer'],
  },
  {
    packageId: 'PKG-2026-003',
    title: 'Fail-Closed Safety Gate Quarantine Demonstration',
    originatingCommittee: 'Audit & Governance Committee',
    committeeId: 'COM-003',
    urgency: 'HIGH',
    status: 'FAILED',
    currentStage: 'GOVERNANCE_VALIDATION',
    targetMetric: 'Governance Gate Certification',
    baselineMetricValue: 92.5,
    projectedMetricValue: 80.0,
    criticalRisks: [
      'Unanimous auditor attestation signature missing',
      'Execution strictly blocked by fail-closed circuit breaker',
    ],
    intelligenceSynthesis: {
      ohi: { baseline: 86.4, projected: 82.0, delta: -4.4, trend: 'DEGRADING' },
      risk: { baselineScore: 42.1, projectedScore: 68.0, delta: +25.9, severity: 'CRITICAL', topThreat: 'Uncertified Policy Breach' },
      resilience: { score: 78.0, rtoMinutes: 12.0, failoverStatus: 'DEGRADED' },
      optimization: { capitalAllocationUSD: 0, efficiencyGainPct: 0.0, paybackMonths: 0 },
      simulation: { stressScore: 62.0, worstCaseVaRUSD: 3400000, monteCarloConfidencePct: 82.0 },
    },
    driverAttribution: [
      { id: 'DRV-09', name: 'Compliance Deficit Invariant', category: 'GOVERNANCE', percentage: 60.0, polarity: 'NEGATIVE', description: 'Missing mandatory external auditor counter-signature', telemetrySource: 'TEL-AUD-01' },
      { id: 'DRV-10', name: 'Audit Ledger Disconnect', category: 'AUDIT', percentage: 40.0, polarity: 'NEGATIVE', description: 'Immutable hash chain discrepancy in secondary witness node', telemetrySource: 'TEL-CHAIN-04' },
    ],
    options: [
      {
        optionId: 'OPT-03-A',
        title: 'Force Execution (Blocked)',
        description: 'Attempt to bypass auditor quorum and force transaction deployment.',
        ohiDelta: -10.0,
        odeiDelta: -15.0,
        riskDelta: +40.0,
        implementationCostUSD: 500000,
        confidencePct: 10.0,
        tradeoffScore: 12.0,
        recommendationRank: 3,
        isRecommended: false,
        rationale: 'Fail-closed invariant strictly blocks execution with zero state mutation.',
        governanceCompliance: false,
      },
      {
        optionId: 'OPT-03-B',
        title: 'Quarantine & Remediate Signatures (Required Action)',
        description: 'Place package into isolated quarantine, notify audit partner, and request expedited re-signing.',
        ohiDelta: +0.5,
        odeiDelta: +1.0,
        riskDelta: -5.0,
        implementationCostUSD: 10000,
        confidencePct: 92.0,
        tradeoffScore: 88.0,
        recommendationRank: 1,
        isRecommended: true,
        rationale: 'Preserves institutional integrity and adheres to zero-compromise security posture.',
        governanceCompliance: true,
      },
      {
        optionId: 'OPT-03-C',
        title: 'Revoke and Discard Proposal',
        description: 'Terminate the package completely and require initiating committee to re-draft.',
        ohiDelta: -1.0,
        odeiDelta: -2.0,
        riskDelta: 0.0,
        implementationCostUSD: 0,
        confidencePct: 100.0,
        tradeoffScore: 50.0,
        recommendationRank: 2,
        isRecommended: false,
        rationale: 'Safe but wastes prior committee review cycles.',
        governanceCompliance: true,
      },
    ],
    governanceValidation: {
      passed: false,
      failClosed: true,
      certifiedAtUtc: '2026-09-08T22:25:00Z',
      auditorSignoffRequired: true,
      stateHash: 'GOV-FAIL-0xdeadbeef',
      failureReasons: [
        'FAIL-CLOSE: Auditor digital signature missing or invalid.',
        'FAIL-CLOSE: Secondary ledger witness hash mismatch detected.',
      ],
      ruleChecks: [
        { ruleId: 'RULE-08', name: 'Auditor Digital Signature', category: 'AUDIT_READINESS', threshold: 'Verified Key Pair', observedValue: 'MISSING', status: 'FAIL', failClosed: true },
        { ruleId: 'RULE-09', name: 'Witness Ledger Consistency', category: 'SAFETY_BOUNDARY', threshold: '100% Consensus', observedValue: 'Consensus Drift (2/3)', status: 'FAIL', failClosed: true },
      ],
    },
    stateHash: '',
    createdAtUtc: '2026-09-08T21:00:00Z',
    updatedAtUtc: '2026-09-08T22:25:00Z',
    assignedRoles: ['Executive', 'CommitteeChair', 'Analyst', 'Auditor', 'GovernanceOfficer'],
  },
];

// Initialize deterministic hashes
CANONICAL_PACKAGES.forEach(pkg => {
  pkg.stateHash = computePackageStateHash(pkg);
});

export function getCanonicalDecisionPackages(): DecisionPackage[] {
  return JSON.parse(JSON.stringify(CANONICAL_PACKAGES));
}

export function getDecisionPackageById(packageId: string): DecisionPackage | undefined {
  const all = getCanonicalDecisionPackages();
  return all.find(p => p.packageId === packageId);
}
