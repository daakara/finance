/**
 * Horizon 12: Personal Adaptation & Behavioral Learning Invariants
 *
 * Enforces the 3 Foundational Personalization Contracts:
 * - INV-OI106-P: Behavioral Personalization Invariant (Adapts to execution history; no repeated unadapted failures)
 * - INV-OI107-P: Friction Learning Invariant (Repeated friction triggers programmatic parameter updates)
 * - INV-OI108-P: Attention Respect Invariant (No repeated suggestions of ignored actions without micro-scoping)
 */

export interface InvariantAuditResult {
  valid: boolean;
  invariantId: string;
  violations: string[];
  metrics?: Record<string, unknown>;
}

export interface CandidateActionAdaptationRecord {
  candidateId: string;
  actionTitle: string;
  domain: 'TRADING' | 'CAREER' | 'HEALTH' | 'FINANCE' | 'HOUSEHOLD';
  scheduledHour: number; // 0-23
  estimatedMinutes: number;
  historicalFailuresCount: number; // times this specific pattern was deferred or abandoned
  hasStructuralAdaptation: boolean; // whether timing, scope, or modality was modified
  adaptationNotes?: string;
}

export interface FrictionPatternRecord {
  patternId: string;
  patternType: 'CHRONOTYPE_MISMATCH' | 'DURATION_TOO_LARGE' | 'DOMAIN_RESISTANCE';
  observedFailureCount: number;
  activeAdaptiveRuleAssigned: boolean;
  ruleDescription?: string;
}

export interface DomainAdherenceRecord {
  domain: 'TRADING' | 'CAREER' | 'HEALTH' | 'FINANCE' | 'HOUSEHOLD';
  historicalCompletionRatePct: number;
  proposedEstimatedMinutes: number;
  isMicroAction: boolean; // <= 15 minutes
}

/**
 * INV-OI106-P: Behavioral Personalization Invariant
 * Gating: Recommendations MUST adapt to observed execution history.
 * Prohibits proposing the exact same recommendation pattern if it has failed >= 3 consecutive times with zero adaptation.
 */
export function verifyBehavioralPersonalization(
  record: CandidateActionAdaptationRecord
): {
  compliant: boolean;
  adaptationEnforced: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  if (record.historicalFailuresCount >= 3 && !record.hasStructuralAdaptation) {
    violations.push(
      `INV-OI106-P VIOLATION: Recommendation "${record.actionTitle}" has failed ${record.historicalFailuresCount} consecutive times with ZERO structural adaptation. System must adapt timing, duration, or modality.`
    );
  }

  return {
    compliant: violations.length === 0,
    adaptationEnforced: record.hasStructuralAdaptation,
    violations,
  };
}

/**
 * INV-OI107-P: Friction Learning Invariant
 * Gating: Repeated friction patterns (failures >= 3) must directly trigger active adaptive rules
 * (e.g. chronotype shifts, duration clamps).
 */
export function verifyFrictionLearning(
  frictionPatterns: FrictionPatternRecord[]
): {
  compliant: boolean;
  unaddressedPatternsCount: number;
  violations: string[];
} {
  const violations: string[] = [];
  let unaddressed = 0;

  frictionPatterns.forEach((fp) => {
    if (fp.observedFailureCount >= 3 && !fp.activeAdaptiveRuleAssigned) {
      unaddressed++;
      violations.push(
        `INV-OI107-P VIOLATION: Friction pattern "${fp.patternId}" (${fp.patternType}) has ${fp.observedFailureCount} failures but no active adaptive rule assigned.`
      );
    }
  });

  return {
    compliant: violations.length === 0,
    unaddressedPatternsCount: unaddressed,
    violations,
  };
}

/**
 * INV-OI108-P: Attention Respect Invariant
 * Gating: When suggesting actions in a historically low-conversion domain (< 40%),
 * the task MUST be scaled down into a micro-action (<= 15 minutes).
 */
export function verifyAttentionRespect(
  record: DomainAdherenceRecord
): {
  compliant: boolean;
  microActionRequired: boolean;
  violations: string[];
} {
  const violations: string[] = [];
  const isLowConversion = record.historicalCompletionRatePct < 40;

  if (isLowConversion) {
    if (!record.isMicroAction || record.proposedEstimatedMinutes > 15) {
      violations.push(
        `INV-OI108-P VIOLATION: Domain "${record.domain}" has a low completion rate (${record.historicalCompletionRatePct}%). Action of ${record.proposedEstimatedMinutes} minutes violates attention respect; must be clamped to <= 15 minutes.`
      );
    }
  }

  return {
    compliant: violations.length === 0,
    microActionRequired: isLowConversion,
    violations,
  };
}

/**
 * Master Personal Adaptation Safety Auditor for Horizon 12
 */
export function auditHorizon12Master(payload: {
  candidateRecord: CandidateActionAdaptationRecord;
  frictionPatterns: FrictionPatternRecord[];
  domainRecord: DomainAdherenceRecord;
}): {
  valid: boolean;
  auditResults: Record<string, InvariantAuditResult>;
  allViolations: string[];
} {
  const r106 = verifyBehavioralPersonalization(payload.candidateRecord);
  const r107 = verifyFrictionLearning(payload.frictionPatterns);
  const r108 = verifyAttentionRespect(payload.domainRecord);

  const allViolations = [
    ...r106.violations,
    ...r107.violations,
    ...r108.violations,
  ];

  return {
    valid: allViolations.length === 0,
    auditResults: {
      'INV-OI106-P': { valid: r106.compliant, invariantId: 'INV-OI106-P', violations: r106.violations },
      'INV-OI107-P': { valid: r107.compliant, invariantId: 'INV-OI107-P', violations: r107.violations },
      'INV-OI108-P': { valid: r108.compliant, invariantId: 'INV-OI108-P', violations: r108.violations },
    },
    allViolations,
  };
}
