/**
 * Horizon 13 Invariants: Identity Intelligence Layer
 *
 * Implements three fail-closed behavioral safety invariants:
 * - INV-OI109-P: Identity Consistency (Recommendations must support declared identity trajectory)
 * - INV-OI110-P: Identity Drift Detection (Contradictory behavior patterns must be detected)
 * - INV-OI111-P: Identity Traceability (Every identity progression must be causally explainable)
 */

export interface IdentityConsistencyRecord {
  actionId: string;
  actionTitle: string;
  domain: string;
  priorityRank: number; // 1 = top priority
  identityContributionScore: number; // 0 to 100
  isEmergencyOverride: boolean;
  hasCausalIdentityJustification: boolean;
}

export interface InvariantResult {
  compliant: boolean;
  invariantId: string;
  violations: string[];
  metadata?: Record<string, unknown>;
}

/**
 * INV-OI109-P: Identity Consistency
 * Recommendations must actively support the declared identity trajectory.
 * Low-value, unrelated tasks must not supersede declared identity building actions
 * without explicit explanation or urgent necessity.
 */
export function verifyIdentityConsistency(
  candidateActions: IdentityConsistencyRecord[]
): InvariantResult {
  const violations: string[] = [];

  // Sort candidate actions by priority rank
  const sorted = [...candidateActions].sort((a, b) => a.priorityRank - b.priorityRank);

  // If top action has very low identity contribution (< 30) and is not an emergency,
  // and there exists a high-identity action (>= 75) that was deprioritized without justification:
  const topAction = sorted[0];
  const highIdentityAction = sorted.find((a) => a.identityContributionScore >= 75);

  if (topAction && highIdentityAction && topAction.actionId !== highIdentityAction.actionId) {
    if (
      topAction.identityContributionScore < 30 &&
      !topAction.isEmergencyOverride &&
      !topAction.hasCausalIdentityJustification
    ) {
      violations.push(
        `INV-OI109-P VIOLATION: Top recommendation "${topAction.actionTitle}" (score: ${topAction.identityContributionScore}) supersedes high-leverage identity action "${highIdentityAction.actionTitle}" (score: ${highIdentityAction.identityContributionScore}) without emergency override or justification.`
      );
    }
  }

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI109-P',
    violations,
    metadata: {
      actionsEvaluated: candidateActions.length,
      topActionContribution: topAction?.identityContributionScore ?? 0,
    },
  };
}

export interface IdentityDriftRecord {
  targetRole: string;
  domain: string;
  daysSinceLastActivity: number;
  thresholdDays: number;
  alertEmitted: boolean;
  driftSeverity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

/**
 * INV-OI110-P: Identity Drift Detection
 * Repeated behavior patterns contradicting target identity must be detected
 * and surfaced as fail-closed alerts when inactivity threshold is exceeded.
 */
export function verifyIdentityDriftDetection(
  driftRecords: IdentityDriftRecord[]
): InvariantResult {
  const violations: string[] = [];
  let unalertedDrifts = 0;

  driftRecords.forEach((record) => {
    const isDrifting = record.daysSinceLastActivity >= record.thresholdDays;
    if (isDrifting && !record.alertEmitted) {
      unalertedDrifts++;
      violations.push(
        `INV-OI110-P VIOLATION: Target identity domain "${record.domain}" for role "${record.targetRole}" has been inactive for ${record.daysSinceLastActivity} days (threshold: ${record.thresholdDays}) without an emitted Identity Drift Alert.`
      );
    }
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI110-P',
    violations,
    metadata: {
      recordsEvaluated: driftRecords.length,
      unalertedDrifts,
    },
  };
}

export interface IdentityTraceabilityNode {
  traitName: string;
  previousLevel: number;
  newLevel: number;
  delta: number;
  evidenceChain: string[];
}

/**
 * INV-OI111-P: Identity Traceability
 * Every identity trait progression or momentum change must be causally explainable
 * via an unbroken evidentiary lineage chain.
 */
export function verifyIdentityTraceability(
  nodes: IdentityTraceabilityNode[]
): InvariantResult {
  const violations: string[] = [];
  let untracedDeltas = 0;

  nodes.forEach((node) => {
    if (node.delta > 0 && (!node.evidenceChain || node.evidenceChain.length === 0)) {
      untracedDeltas++;
      violations.push(
        `INV-OI111-P VIOLATION: Trait "${node.traitName}" gained +${node.delta} points (from ${node.previousLevel} to ${node.newLevel}) with zero evidentiary proof points in its lineage chain.`
      );
    }
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI111-P',
    violations,
    metadata: {
      nodesAudited: nodes.length,
      untracedDeltas,
    },
  };
}

/**
 * Master Audit for Horizon 13 Identity Intelligence Invariants
 */
export function auditHorizon13Master(payload: {
  candidateActions: IdentityConsistencyRecord[];
  driftRecords: IdentityDriftRecord[];
  traceabilityNodes: IdentityTraceabilityNode[];
}): {
  certified: boolean;
  results: Record<string, InvariantResult>;
  totalViolations: number;
} {
  const consistency = verifyIdentityConsistency(payload.candidateActions);
  const drift = verifyIdentityDriftDetection(payload.driftRecords);
  const traceability = verifyIdentityTraceability(payload.traceabilityNodes);

  const totalViolations =
    consistency.violations.length + drift.violations.length + traceability.violations.length;

  return {
    certified: totalViolations === 0,
    results: {
      'INV-OI109-P': consistency,
      'INV-OI110-P': drift,
      'INV-OI111-P': traceability,
    },
    totalViolations,
  };
}
