/**
 * Horizon 10: Behavioral Safety & Decision Reduction Invariants
 *
 * Enforces the 5 Foundational Consumer Safety Contracts:
 * - INV-OI97-P: Cognitive Trading Discipline Invariant
 * - INV-OI98-P: Household Capital Protection Invariant
 * - INV-OI99-P: Recommendation Overload Prevention Invariant
 * - INV-OI100-P: Human Agency Preservation Invariant
 * - INV-OI101-P: Decision Simplicity Invariant (Primary = 1, Secondary <= 2)
 */

export interface InvariantAuditResult {
  valid: boolean;
  invariantId: string;
  violations: string[];
  metrics?: Record<string, unknown>;
}

/**
 * INV-OI97-P: Cognitive Trading Discipline Invariant
 * Gating: Lockout if recoveryScore < 55, paper-only if loss streak >= 2, lockout if dailyDrawdownPct > 0.03 (3%).
 */
export function verifyCognitiveTradingDiscipline(
  recoveryScore: number,
  recentLossStreak: number,
  dailyDrawdownPct: number
): {
  compliant: boolean;
  safeToTrade: boolean;
  enforcementAction: "PROCEED" | "PAPER_ONLY" | "LOCKOUT";
  reason: string;
  violations: string[];
} {
  const violations: string[] = [];
  let enforcementAction: "PROCEED" | "PAPER_ONLY" | "LOCKOUT" = "PROCEED";
  let reason = "Cognitive and risk parameters nominal.";

  if (recoveryScore < 55) {
    violations.push(`INV-OI97-P VIOLATION: Recovery score ${recoveryScore}% below cognitive safety floor (55%). High tilt probability.`);
    enforcementAction = "LOCKOUT";
    reason = "Recovery score degraded. Impulse trading shield engaged.";
  } else if (dailyDrawdownPct > 0.03) {
    violations.push(`INV-OI97-P VIOLATION: Daily portfolio drawdown ${(dailyDrawdownPct * 100).toFixed(1)}% exceeds 3.0% safety threshold.`);
    enforcementAction = "LOCKOUT";
    reason = "Intra-day drawdown limit breached. Capital preservation lockout active.";
  } else if (recentLossStreak >= 2) {
    violations.push(`INV-OI97-P VIOLATION: Consecutive loss streak of ${recentLossStreak} detected (>= 2). Revenge trading circuit breaker tripped.`);
    enforcementAction = "PAPER_ONLY";
    reason = "Consecutive losses detected. Trading restricted to paper mode.";
  }

  const compliant = violations.length === 0;
  return {
    compliant,
    safeToTrade: enforcementAction === "PROCEED",
    enforcementAction,
    reason,
    violations
  };
}

/**
 * INV-OI98-P: Household Capital Protection Invariant
 * Gating: Liquid cash reserve must remain >= 6.0 months essential burn after any capital allocation.
 */
export function verifyHouseholdCapitalProtection(
  currentLiquidCash: number,
  monthlyEssentialBurn: number,
  proposedCapitalDeployment: number
): {
  compliant: boolean;
  currentRunwayMonths: number;
  postDeploymentRunwayMonths: number;
  violations: string[];
} {
  const violations: string[] = [];
  const safeBurn = Math.max(1, monthlyEssentialBurn);
  const currentRunway = currentLiquidCash / safeBurn;
  const remainingCash = currentLiquidCash - proposedCapitalDeployment;
  const postDeploymentRunway = remainingCash / safeBurn;

  if (postDeploymentRunway < 6.0) {
    violations.push(
      `INV-OI98-P VIOLATION: Proposed capital deployment of $${proposedCapitalDeployment.toLocaleString()} reduces household runway to ${postDeploymentRunway.toFixed(1)} months (strict minimum floor is 6.0 months).`
    );
  }

  if (proposedCapitalDeployment > currentLiquidCash) {
    violations.push(
      `INV-OI98-P VIOLATION: Proposed capital deployment ($${proposedCapitalDeployment}) exceeds available liquid cash ($${currentLiquidCash}).`
    );
  }

  return {
    compliant: violations.length === 0,
    currentRunwayMonths: Number(currentRunway.toFixed(2)),
    postDeploymentRunwayMonths: Number(postDeploymentRunway.toFixed(2)),
    violations
  };
}

/**
 * INV-OI99-P: Recommendation Overload Prevention Invariant
 * Gating: Total visible recommendations must be <= 3.
 */
export function verifyRecommendationOverloadPrevention(
  actions: Array<{ id: string }>
): {
  compliant: boolean;
  visibleCount: number;
  violations: string[];
} {
  const violations: string[] = [];
  const visibleCount = actions.length;

  if (visibleCount > 3) {
    violations.push(
      `INV-OI99-P VIOLATION: Surfaced ${visibleCount} recommendations simultaneously (maximum allowable is 3 to prevent choice paralysis).`
    );
  }

  return {
    compliant: violations.length === 0,
    visibleCount,
    violations
  };
}

/**
 * INV-OI100-P: Human Agency Preservation Invariant
 * Gating: Prohibits autonomous execution of irreversible financial/career mutations; requires explicit user consent.
 */
export function verifyHumanAgencyPreservation(
  actionType: "SUGGESTION" | "AUTONOMOUS_EXECUTION",
  hasExplicitUserConsent: boolean
): {
  compliant: boolean;
  violationRisk: string;
  violations: string[];
} {
  const violations: string[] = [];

  if (actionType === "AUTONOMOUS_EXECUTION") {
    violations.push(
      "INV-OI100-P FATAL VIOLATION: Autonomous financial or career mutation attempted. Platform strictly forbids unconfirmed mutations."
    );
  } else if (!hasExplicitUserConsent) {
    violations.push(
      "INV-OI100-P VIOLATION: Pending explicit two-factor human confirmation before execution."
    );
  }

  return {
    compliant: violations.length === 0,
    violationRisk: violations.length > 0 ? violations[0] : "NONE",
    violations
  };
}

/**
 * INV-OI101-P: Decision Simplicity Invariant
 * Gating: The system must surface exactly ONE primary action. Additional actions must be secondary (<= 2).
 */
export function verifyDecisionSimplicity(
  actions: Array<{ id: string; isPrimary?: boolean; isSecondary?: boolean }>
): {
  compliant: boolean;
  primaryCount: number;
  secondaryCount: number;
  violations: string[];
} {
  const violations: string[] = [];
  const primaryCount = actions.filter(a => a.isPrimary).length;
  const secondaryCount = actions.filter(a => !a.isPrimary).length;

  if (primaryCount !== 1) {
    violations.push(
      `INV-OI101-P VIOLATION: Found ${primaryCount} primary actions (exactly 1 is required for decision clarity).`
    );
  }

  if (secondaryCount > 2) {
    violations.push(
      `INV-OI101-P VIOLATION: Found ${secondaryCount} secondary actions (maximum allowable is 2).`
    );
  }

  return {
    compliant: violations.length === 0,
    primaryCount,
    secondaryCount,
    violations
  };
}

/**
 * Master Auditor for Horizon 10 Behavioral Safety
 */
export function auditHorizon10Master(payload: {
  recoveryScore: number;
  recentLossStreak: number;
  dailyDrawdownPct: number;
  currentLiquidCash: number;
  monthlyEssentialBurn: number;
  proposedCapitalDeployment: number;
  actions: Array<{ id: string; isPrimary?: boolean; isSecondary?: boolean }>;
  actionType: "SUGGESTION" | "AUTONOMOUS_EXECUTION";
  hasExplicitUserConsent: boolean;
}): {
  valid: boolean;
  auditResults: Record<string, InvariantAuditResult>;
  allViolations: string[];
} {
  const r97 = verifyCognitiveTradingDiscipline(
    payload.recoveryScore,
    payload.recentLossStreak,
    payload.dailyDrawdownPct
  );
  const r98 = verifyHouseholdCapitalProtection(
    payload.currentLiquidCash,
    payload.monthlyEssentialBurn,
    payload.proposedCapitalDeployment
  );
  const r99 = verifyRecommendationOverloadPrevention(payload.actions);
  const r100 = verifyHumanAgencyPreservation(payload.actionType, payload.hasExplicitUserConsent);
  const r101 = verifyDecisionSimplicity(payload.actions);

  const allViolations = [
    ...r97.violations,
    ...r98.violations,
    ...r99.violations,
    ...r100.violations,
    ...r101.violations
  ];

  return {
    valid: allViolations.length === 0,
    auditResults: {
      "INV-OI97-P": { valid: r97.compliant, invariantId: "INV-OI97-P", violations: r97.violations, metrics: { enforcementAction: r97.enforcementAction } },
      "INV-OI98-P": { valid: r98.compliant, invariantId: "INV-OI98-P", violations: r98.violations, metrics: { postRunway: r98.postDeploymentRunwayMonths } },
      "INV-OI99-P": { valid: r99.compliant, invariantId: "INV-OI99-P", violations: r99.violations, metrics: { visibleCount: r99.visibleCount } },
      "INV-OI100-P": { valid: r100.compliant, invariantId: "INV-OI100-P", violations: r100.violations, metrics: { risk: r100.violationRisk } },
      "INV-OI101-P": { valid: r101.compliant, invariantId: "INV-OI101-P", violations: r101.violations, metrics: { primaryCount: r101.primaryCount, secondaryCount: r101.secondaryCount } }
    },
    allViolations
  };
}
