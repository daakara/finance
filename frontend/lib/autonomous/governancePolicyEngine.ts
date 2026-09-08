/**
 * Phase 31-M9: Governance Policy Engine
 *
 * Implements:
 * - Deterministic Policy Rule Evaluation
 * - Policy Boundary Enforcement (INV-OI53: 0 unauthorized actions outside certified limits)
 * - Deterministic SHA-256 Policy State Hashing (INV-OI54: 100 replays -> 1 hash)
 * - Safe Policy Mutation & Validation
 */

import {
  GovernancePolicy,
  PolicyRule,
  PolicyEvaluationResult,
  AutonomousActionRequest,
  CANONICAL_POLICIES,
} from '@/types/autonomous-governance';
import { sha256Hex } from '@/lib/governance/sha256';

export function getCanonicalPolicies(): GovernancePolicy[] {
  return JSON.parse(JSON.stringify(CANONICAL_POLICIES));
}

/**
 * Evaluates a single governance policy against an action request.
 */
export function evaluatePolicy(
  policy: GovernancePolicy,
  request: AutonomousActionRequest
): PolicyEvaluationResult {
  const blockingRules: string[] = [];
  const requiredApprovals: string[] = [];
  let computedRisk = request.targetRiskScore ?? 50.0;

  for (const rule of policy.rules) {
    if (rule.category === 'RISK') {
      if (rule.thresholdValue !== undefined && rule.ruleId === 'RULE-RSK-01') {
        // VaR drawdown limit (15.0%)
        if (computedRisk > 85.0) {
          blockingRules.push(rule.ruleId);
        }
      }
      if (rule.thresholdValue !== undefined && rule.ruleId === 'RULE-RSK-02') {
        // Budget reallocation threshold ($100k)
        if ((request.budgetRequestedDollars ?? 0) > rule.thresholdValue) {
          requiredApprovals.push(rule.ruleId);
        }
      }
    }

    if (rule.category === 'GOVERNANCE') {
      if (rule.ruleId === 'RULE-GOV-01') {
        // Dissent ratio check
        if (computedRisk > 75.0 && request.committeeId === 'COM-DEFAULT') {
          blockingRules.push(rule.ruleId);
        }
      }
      if (rule.ruleId === 'RULE-GOV-02' && request.proposedAction.toLowerCase().includes('charter')) {
        requiredApprovals.push(rule.ruleId);
      }
    }

    if (rule.category === 'AUTONOMY') {
      if (rule.ruleId === 'RULE-AUTO-02') {
        // Rollback pathway required
        if (request.proposedAction.toLowerCase().includes('no_rollback')) {
          blockingRules.push(rule.ruleId);
        }
      }
      if (rule.ruleId === 'RULE-AUTO-03' && rule.thresholdValue !== undefined) {
        // Resource rebalancing > 10%
        if ((request.budgetRequestedDollars ?? 0) > 50000) {
          requiredApprovals.push(rule.ruleId);
        }
      }
    }
  }

  const actionAllowed = blockingRules.length === 0 && requiredApprovals.length === 0;
  const decision = blockingRules.length > 0
    ? 'REJECTED'
    : requiredApprovals.length > 0
    ? 'REQUIRES_REVIEW'
    : 'APPROVED';

  return {
    evaluationId: `EVAL-${policy.policyId}-${Date.now().toString(36)}`,
    policyId: policy.policyId,
    actionAllowed,
    blockingRules,
    requiredApprovals,
    decision,
    riskScore: computedRisk,
    evaluatedAtUtc: new Date().toISOString(),
  };
}

/**
 * Evaluates all active policies against an action request.
 */
export function evaluateAllPolicies(
  policies: GovernancePolicy[],
  request: AutonomousActionRequest
): PolicyEvaluationResult[] {
  return policies
    .filter((p) => p.status === 'ACTIVE')
    .map((policy) => evaluatePolicy(policy, request));
}

/**
 * Invariant INV-OI53: Policy Boundary Enforcement.
 * Returns true if and only if zero blocking rules triggered across all certified policies.
 */
export function checkPolicyBoundary(
  policies: GovernancePolicy[],
  request: AutonomousActionRequest
): { allowed: boolean; blockingRules: string[]; evaluations: PolicyEvaluationResult[] } {
  const evaluations = evaluateAllPolicies(policies, request);
  const blockingRules = evaluations.flatMap((e) => e.blockingRules);
  return {
    allowed: blockingRules.length === 0,
    blockingRules,
    evaluations,
  };
}

/**
 * Invariant INV-OI54: Deterministic SHA-256 Policy State Hashing.
 */
export function computePolicyHash(policies: GovernancePolicy[]): string {
  const sorted = [...policies].sort((a, b) => a.policyId.localeCompare(b.policyId));
  const payload = JSON.stringify(
    sorted.map((p) => ({
      policyId: p.policyId,
      version: p.version,
      status: p.status,
      cert: p.certificationStatus,
      rules: [...p.rules].sort((r1, r2) => r1.ruleId.localeCompare(r2.ruleId)),
    }))
  );
  return sha256Hex(payload);
}

/**
 * Verifies that 100 replays yield 1 identical SHA-256 hash (0 drift).
 */
export function verifyPolicyReplay(
  policies: GovernancePolicy[],
  iterations = 100
): { pass: boolean; uniqueHashes: number; hash: string } {
  const hashes = new Set<string>();
  for (let i = 0; i < iterations; i++) {
    hashes.add(computePolicyHash(policies));
  }
  return {
    pass: hashes.size === 1,
    uniqueHashes: hashes.size,
    hash: Array.from(hashes)[0],
  };
}
