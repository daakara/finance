/**
 * Horizon 11: Behavioral Execution & Decision Calibration Invariants
 *
 * Enforces the 4 Behavioral Execution Safety & Learning Contracts:
 * - INV-OI102-P: Execution Accountability Invariant (No silent drops)
 * - INV-OI103-P: Recommendation Outcome Learning Invariant (Observed delta attribution)
 * - INV-OI104-P: Non-Punitive Behavioral Recovery Invariant (Zero shame, restorative friction diagnostics)
 * - INV-OI105-P: Decision Outcome Calibration Invariant (Brier score & calibration ledger)
 */

export type ExecutionTerminalState = 'COMPLETED' | 'DEFERRED' | 'REJECTED' | 'ABANDONED';
export type ExecutionActiveState = 'PROPOSED' | 'COMMITTED' | 'IN_PROGRESS';
export type ExecutionState = ExecutionActiveState | ExecutionTerminalState;

export interface InvariantAuditResult {
  valid: boolean;
  invariantId: string;
  violations: string[];
  metrics?: Record<string, unknown>;
}

export interface RecommendationExecutionRecord {
  recommendationId: string;
  actionTitle: string;
  domain: 'TRADING' | 'CAREER' | 'HEALTH' | 'FINANCE' | 'HOUSEHOLD';
  currentState: ExecutionState;
  proposedTimestamp: string;
  committedTimestamp?: string;
  completedTimestamp?: string;
  deferredCount: number;
  terminalState?: ExecutionTerminalState;
  silentDropDetected: boolean;
  preTelemetry?: {
    recoveryScore: number;
    focusHoursAvailable: number;
    liquidRunwayMonths: number;
  };
  postTelemetry?: {
    recoveryScore: number;
    focusHoursAvailable: number;
    liquidRunwayMonths: number;
  };
  outcomeAttribution?: {
    predictedLhiDelta: number;
    observedLhiDelta: number;
    variance: number;
    confidenceScore: number;
  };
}

/**
 * INV-OI102-P: Execution Accountability Invariant
 * Gating: Every primary recommendation must reach an explicit terminal state.
 * Silent disappearances or unaddressed expirations trigger a fail-closed violation.
 */
export function verifyExecutionAccountability(
  record: RecommendationExecutionRecord
): {
  compliant: boolean;
  terminalStateAssigned: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  if (record.silentDropDetected) {
    violations.push(
      `INV-OI102-P VIOLATION: Recommendation "${record.actionTitle}" (${record.recommendationId}) disappeared silently without explicit human resolution.`
    );
  }

  const validTerminalStates: ExecutionTerminalState[] = [
    'COMPLETED',
    'DEFERRED',
    'REJECTED',
    'ABANDONED',
  ];

  const hasTerminal =
    record.terminalState !== undefined && validTerminalStates.includes(record.terminalState);

  // If action is resolved or past expiration, it must have a valid terminal state
  if (['COMPLETED', 'DEFERRED', 'REJECTED', 'ABANDONED'].includes(record.currentState)) {
    if (!hasTerminal) {
      violations.push(
        `INV-OI102-P VIOLATION: Current state is ${record.currentState} but terminal state is missing or invalid.`
      );
    }
  }

  return {
    compliant: violations.length === 0,
    terminalStateAssigned: hasTerminal,
    violations,
  };
}

/**
 * INV-OI103-P: Recommendation Outcome Learning Invariant
 * Gating: Completed recommendations must be linked to measurable outcome telemetry and variance attribution.
 */
export function verifyRecommendationOutcomeLearning(
  record: RecommendationExecutionRecord
): {
  compliant: boolean;
  hasMeasuredOutcome: boolean;
  varianceCalculated: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  if (record.currentState === 'COMPLETED' || record.terminalState === 'COMPLETED') {
    if (!record.outcomeAttribution) {
      violations.push(
        `INV-OI103-P VIOLATION: Completed recommendation "${record.actionTitle}" lacks outcome attribution and variance tracking.`
      );
    } else {
      const oa = record.outcomeAttribution;
      if (typeof oa.observedLhiDelta !== 'number' || typeof oa.predictedLhiDelta !== 'number') {
        violations.push(
          `INV-OI103-P VIOLATION: Incomplete numerical metrics in outcome attribution.`
        );
      }
      const expectedVariance = Number((oa.observedLhiDelta - oa.predictedLhiDelta).toFixed(2));
      if (Math.abs(oa.variance - expectedVariance) > 0.05) {
        violations.push(
          `INV-OI103-P VIOLATION: Variance mathematical mismatch. Expected ${expectedVariance}, recorded ${oa.variance}.`
        );
      }
    }

    if (!record.postTelemetry || !record.preTelemetry) {
      violations.push(
        `INV-OI103-P VIOLATION: Completed action missing pre/post telemetry snapshot for causal attribution.`
      );
    }
  }

  return {
    compliant: violations.length === 0,
    hasMeasuredOutcome: Boolean(record.outcomeAttribution),
    varianceCalculated:
      record.outcomeAttribution !== undefined && typeof record.outcomeAttribution.variance === 'number',
    violations,
  };
}

/**
 * INV-OI104-P: Non-Punitive Behavioral Recovery Invariant
 * Gating: When actions are deferred >= 2 times or abandoned, system must NOT shame or moralize,
 * and must provide a restorative friction diagnosis or scope reduction.
 */
export function verifyNonPunitiveBehavioralRecovery(
  record: RecommendationExecutionRecord,
  systemFeedbackCopy: string,
  hasFrictionDiagnosis: boolean,
  hasScopeReductionOffer: boolean
): {
  compliant: boolean;
  shameDetected: boolean;
  recoveryRemedyProvided: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  // Shame / Guilt detection patterns
  const punitiveTerms = [
    'failed',
    'missed your goal',
    'falling behind',
    'lacking discipline',
    'slacking',
    'disappointing',
    'broken streak',
    'guilty',
  ];

  const lowerCopy = systemFeedbackCopy.toLowerCase();
  const shameFound = punitiveTerms.some((term) => lowerCopy.includes(term));

  if (shameFound) {
    violations.push(
      `INV-OI104-P VIOLATION: System feedback contains punitive, moralizing, or guilt-inducing copy: "${systemFeedbackCopy}".`
    );
  }

  const isStruggling = record.deferredCount >= 2 || record.terminalState === 'ABANDONED';

  if (isStruggling) {
    if (!hasFrictionDiagnosis) {
      violations.push(
        `INV-OI104-P VIOLATION: Multiple deferrals (${record.deferredCount}) or abandonment requires automated friction diagnosis.`
      );
    }
    if (!hasScopeReductionOffer) {
      violations.push(
        `INV-OI104-P VIOLATION: High friction requires non-punitive scope reduction or restorative recovery option.`
      );
    }
  }

  return {
    compliant: violations.length === 0,
    shameDetected: shameFound,
    recoveryRemedyProvided: hasFrictionDiagnosis && hasScopeReductionOffer,
    violations,
  };
}

/**
 * INV-OI105-P: Decision Outcome Calibration Invariant
 * Gating: Calculates Brier score across historical decisions: (1/N) * sum((prob - outcome)^2).
 * Model calibration must remain honest; excessive variance triggers wider confidence intervals.
 */
export function verifyDecisionOutcomeCalibration(
  decisions: Array<{
    decisionId: string;
    predictedSuccessProbability: number; // 0.0 - 1.0
    actualSuccessBinary: number; // 1 for success, 0 for failure
    brierScoreContribution?: number;
  }>
): {
  compliant: boolean;
  brierScore: number;
  sampleSize: number;
  calibrationStatus: 'WELL_CALIBRATED' | 'MODERATE_DRIFT' | 'OVERCONFIDENT';
  violations: string[];
} {
  const violations: string[] = [];
  const n = decisions.length;

  if (n === 0) {
    return {
      compliant: true,
      brierScore: 0,
      sampleSize: 0,
      calibrationStatus: 'WELL_CALIBRATED',
      violations: [],
    };
  }

  let totalSquaredError = 0;
  decisions.forEach((d) => {
    if (d.predictedSuccessProbability < 0 || d.predictedSuccessProbability > 1) {
      violations.push(
        `INV-OI105-P VIOLATION: Decision ${d.decisionId} has invalid probability ${d.predictedSuccessProbability} (must be in [0, 1]).`
      );
    }
    if (d.actualSuccessBinary !== 0 && d.actualSuccessBinary !== 1) {
      violations.push(
        `INV-OI105-P VIOLATION: Decision ${d.decisionId} actual outcome must be binary 0 or 1.`
      );
    }
    const err = Math.pow(d.predictedSuccessProbability - d.actualSuccessBinary, 2);
    totalSquaredError += err;
  });

  const brierScore = Number((totalSquaredError / n).toFixed(4));

  let status: 'WELL_CALIBRATED' | 'MODERATE_DRIFT' | 'OVERCONFIDENT' = 'WELL_CALIBRATED';
  if (brierScore > 0.35) {
    status = 'OVERCONFIDENT';
    violations.push(
      `INV-OI105-P VIOLATION: Brier score ${brierScore} exceeds 0.35 threshold. Model demonstrates severe overconfidence drift.`
    );
  } else if (brierScore > 0.22) {
    status = 'MODERATE_DRIFT';
  }

  return {
    compliant: violations.length === 0,
    brierScore,
    sampleSize: n,
    calibrationStatus: status,
    violations,
  };
}

/**
 * Master Execution Safety Auditor for Horizon 11
 */
export function auditHorizon11Master(payload: {
  record: RecommendationExecutionRecord;
  systemFeedbackCopy: string;
  hasFrictionDiagnosis: boolean;
  hasScopeReductionOffer: boolean;
  historicalDecisions: Array<{
    decisionId: string;
    predictedSuccessProbability: number;
    actualSuccessBinary: number;
  }>;
}): {
  valid: boolean;
  auditResults: Record<string, InvariantAuditResult>;
  allViolations: string[];
} {
  const r102 = verifyExecutionAccountability(payload.record);
  const r103 = verifyRecommendationOutcomeLearning(payload.record);
  const r104 = verifyNonPunitiveBehavioralRecovery(
    payload.record,
    payload.systemFeedbackCopy,
    payload.hasFrictionDiagnosis,
    payload.hasScopeReductionOffer
  );
  const r105 = verifyDecisionOutcomeCalibration(payload.historicalDecisions);

  const allViolations = [
    ...r102.violations,
    ...r103.violations,
    ...r104.violations,
    ...r105.violations,
  ];

  return {
    valid: allViolations.length === 0,
    auditResults: {
      'INV-OI102-P': { valid: r102.compliant, invariantId: 'INV-OI102-P', violations: r102.violations },
      'INV-OI103-P': { valid: r103.compliant, invariantId: 'INV-OI103-P', violations: r103.violations },
      'INV-OI104-P': { valid: r104.compliant, invariantId: 'INV-OI104-P', violations: r104.violations },
      'INV-OI105-P': { valid: r105.compliant, invariantId: 'INV-OI105-P', violations: r105.violations },
    },
    allViolations,
  };
}
