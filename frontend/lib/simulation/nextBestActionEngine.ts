/**
 * Horizon 10: Next Best Action (NBA) Engine
 *
 * Primary Orchestration Layer for Intelligence Reduction.
 * Evaluates candidate actions across Trading, Career, Health, Finance, and Household,
 * filtering them through fail-closed invariants and ranking them so the user is presented
 * with EXACTLY ONE primary move for today and at most two secondary focus items.
 *
 * Pipeline:
 * Signals -> Personal Twin -> Household Twin -> Trajectory Engine -> Wisdom Engine -> NBA Engine -> 30-Second Cockpit
 */

import {
  verifyCognitiveTradingDiscipline,
  verifyHouseholdCapitalProtection,
  verifyRecommendationOverloadPrevention,
  verifyDecisionSimplicity,
} from './horizon10Invariants';

export type ActionDomain = 'TRADING' | 'CAREER' | 'HEALTH' | 'FINANCE' | 'HOUSEHOLD';

export interface CandidateAction {
  id: string;
  domain: ActionDomain;
  headline: string;
  explanation: string;
  utilityScore: number; // 0-100 base score
  expectedImpact: {
    lhiDelta: number;
    financialDeltaDollars?: number;
    stressReductionPct?: number;
  };
  frictionRating: 'EFFORTLESS' | 'LOW_FRICTION' | 'FOCUSED_EFFORT';
  urgency: 'CRITICAL_TODAY' | 'STRATEGIC_LEVER' | 'OPPORTUNITY';
  isPrimary?: boolean;
  isSecondary?: boolean;
  actionPayload?: {
    route: string;
    ctaLabel: string;
  };
  guardrailStatus: 'CLEARED' | 'BLOCKED_BY_INVARIANT';
  invariantsViolated?: string[];
  proofDetails?: {
    setupType?: string;
    riskRewardRatio?: number;
    maxDollarRisk?: number;
    dagTraceNode?: string;
    confidenceInterval?: [number, number];
  };
}

export interface UserOperationalContext {
  recoveryScore: number;              // 0-100 (from wearables)
  recentLossStreak: number;          // realized losses in last 24h
  dailyDrawdownPct: number;          // intra-day portfolio drawdown
  liquidCash: number;                // current emergency liquid reserve
  monthlyEssentialBurn: number;      // household monthly expenses
  householdStrainIndex: number;      // 0-100
  focusHoursAvailable: number;       // deep work hours available today
}

export interface CockpitActionQueue {
  primaryAction: CandidateAction;
  secondaryActions: CandidateAction[];
  suppressedActionsCount: number;
  intelligenceReductionRatio: number; // e.g., 8 candidates -> 3 visible = 62.5% reduction
  auditCompliance: {
    invOi97Cleared: boolean;
    invOi98Cleared: boolean;
    invOi99Compliant: boolean;
    invOi101Compliant: boolean;
  };
}

export const CANONICAL_CANDIDATE_POOL: CandidateAction[] = [
  {
    id: 'ACT-TRADE-GOOGL',
    domain: 'TRADING',
    headline: 'Enter High-Conviction Dip: GOOGL',
    explanation:
      'Alphabet tested its 20-day moving average on institutional volume. Your safe dollar risk budget today is $140.',
    utilityScore: 89,
    expectedImpact: {
      lhiDelta: 2.1,
      financialDeltaDollars: 850,
      stressReductionPct: 0,
    },
    frictionRating: 'LOW_FRICTION',
    urgency: 'OPPORTUNITY',
    guardrailStatus: 'CLEARED',
    actionPayload: {
      route: '/screener',
      ctaLabel: 'Review Execution Plan',
    },
    proofDetails: {
      setupType: 'Minervini VCP Pullback',
      riskRewardRatio: 3.4,
      maxDollarRisk: 140,
      dagTraceNode: 'Focus -> Trade Execution -> Capital Growth',
      confidenceInterval: [0.74, 0.86],
    },
  },
  {
    id: 'ACT-HEALTH-RECOVERY',
    domain: 'HEALTH',
    headline: 'Protect 60-Minute Afternoon Recharge',
    explanation:
      'Wearable telemetry indicates acute sleep fragmentation. An early screen shutdown tonight will prevent tomorrow cognitive fatigue.',
    utilityScore: 86,
    expectedImpact: {
      lhiDelta: 4.8,
      stressReductionPct: 32,
    },
    frictionRating: 'EFFORTLESS',
    urgency: 'CRITICAL_TODAY',
    guardrailStatus: 'CLEARED',
    actionPayload: {
      route: '/me/signals',
      ctaLabel: 'View Biometric Signals',
    },
  },
  {
    id: 'ACT-CAREER-AI-MODULE',
    domain: 'CAREER',
    headline: 'Complete AI Architecture Module 3',
    explanation:
      'Only 45 minutes required to finish Section 3. Completing this unlocks next quarter promotion eligibility with zero household friction.',
    utilityScore: 84,
    expectedImpact: {
      lhiDelta: 3.2,
      financialDeltaDollars: 2400,
      stressReductionPct: 10,
    },
    frictionRating: 'LOW_FRICTION',
    urgency: 'STRATEGIC_LEVER',
    guardrailStatus: 'CLEARED',
    actionPayload: {
      route: '/me/trajectories',
      ctaLabel: 'Track Trajectory',
    },
  },
  {
    id: 'ACT-HOUSEHOLD-DINNER',
    domain: 'HOUSEHOLD',
    headline: 'Confirm Friday Family Offline Dinner',
    explanation:
      'Reserving 2 hours of protected partner time restores household alignment and reduces weekly domestic strain by 25%.',
    utilityScore: 82,
    expectedImpact: {
      lhiDelta: 3.9,
      stressReductionPct: 25,
    },
    frictionRating: 'EFFORTLESS',
    urgency: 'STRATEGIC_LEVER',
    guardrailStatus: 'CLEARED',
    actionPayload: {
      route: '/me/household',
      ctaLabel: 'Open Household OS',
    },
  },
  {
    id: 'ACT-FINANCE-TAX-OPTIMIZE',
    domain: 'FINANCE',
    headline: 'Automate Q3 Tax Reserve Deposit',
    explanation:
      'Transfer $450 into the high-yield tax vault to protect household cash runway against unexpected year-end obligations.',
    utilityScore: 78,
    expectedImpact: {
      lhiDelta: 1.5,
      financialDeltaDollars: 450,
      stressReductionPct: 15,
    },
    frictionRating: 'LOW_FRICTION',
    urgency: 'OPPORTUNITY',
    guardrailStatus: 'CLEARED',
    actionPayload: {
      route: '/portfolio',
      ctaLabel: 'View Portfolio',
    },
  },
];

/**
 * Evaluates physiological dampening factor based on sleep & recovery
 */
export function calculateBiometricDampening(recoveryScore: number): number {
  if (recoveryScore >= 80) return 1.0;
  if (recoveryScore >= 65) return 0.9;
  if (recoveryScore >= 50) return 0.75;
  return 0.5;
}

/**
 * Filters and orchestrates candidate actions into the calm 30-Second Cockpit Action Queue.
 * Enforces INV-OI97-P, INV-OI98-P, INV-OI99-P, and INV-OI101-P.
 */
export function orchestrateNextBestActions(
  candidates: CandidateAction[],
  context: UserOperationalContext
): CockpitActionQueue {
  const biometricFactor = calculateBiometricDampening(context.recoveryScore);

  // 1. Audit each candidate against fail-closed domain invariants
  const evaluatedCandidates: CandidateAction[] = candidates.map((cand) => {
    const violations: string[] = [];

    // Check INV-OI97-P if candidate is a speculative trade action
    if (cand.domain === 'TRADING' && cand.id.startsWith('ACT-TRADE')) {
      const audit97 = verifyCognitiveTradingDiscipline(
        context.recoveryScore,
        context.recentLossStreak,
        context.dailyDrawdownPct
      );
      if (!audit97.compliant) {
        violations.push(...audit97.violations);
      }
    }

    // Check INV-OI98-P if candidate proposes a capital deployment
    const capitalRequired = cand.expectedImpact.financialDeltaDollars || 0;
    if (cand.domain === 'TRADING' || cand.domain === 'FINANCE') {
      const audit98 = verifyHouseholdCapitalProtection(
        context.liquidCash,
        context.monthlyEssentialBurn,
        capitalRequired > 0 && cand.id.includes('DEPOSIT') ? 0 : Math.min(capitalRequired, 1000)
      );
      if (!audit98.compliant) {
        violations.push(...audit98.violations);
      }
    }

    const isCleared = violations.length === 0;
    // Compute effective contextual utility:
    // High friction is penalized more heavily when biometric recovery is low
    const frictionPenalty =
      cand.frictionRating === 'FOCUSED_EFFORT' ? 12 * (1 - biometricFactor) : 0;
    const effectiveUtility = Number(
      (cand.utilityScore * biometricFactor - frictionPenalty).toFixed(1)
    );

    return {
      ...cand,
      utilityScore: effectiveUtility,
      guardrailStatus: isCleared ? 'CLEARED' : 'BLOCKED_BY_INVARIANT',
      invariantsViolated: violations,
    };
  });

  // 2. Filter to CLEARED actions and sort by effective utility descending
  const clearedCandidates = evaluatedCandidates
    .filter((c) => c.guardrailStatus === 'CLEARED')
    .sort((a, b) => b.utilityScore - a.utilityScore);

  // Fallback safe action if everything was blocked by invariants
  let primaryCandidate: CandidateAction;
  let runnerUps: CandidateAction[] = [];

  if (clearedCandidates.length === 0) {
    primaryCandidate = {
      id: 'ACT-SAFE-RECHARGE',
      domain: 'HEALTH',
      headline: 'Rest & Recover Today',
      explanation:
        'Cognitive telemetry indicates fatigue. Your capital and commitments are safely shielded—take today to recharge.',
      utilityScore: 100,
      expectedImpact: { lhiDelta: 5.0, stressReductionPct: 40 },
      frictionRating: 'EFFORTLESS',
      urgency: 'CRITICAL_TODAY',
      guardrailStatus: 'CLEARED',
    };
  } else {
    primaryCandidate = { ...clearedCandidates[0], isPrimary: true, isSecondary: false };
    runnerUps = clearedCandidates.slice(1, 3).map((c) => ({
      ...c,
      isPrimary: false,
      isSecondary: true,
    }));
  }

  const allVisibleActions = [primaryCandidate, ...runnerUps];

  // 3. Verify INV-OI99-P & INV-OI101-P
  const audit99 = verifyRecommendationOverloadPrevention(allVisibleActions);
  const audit101 = verifyDecisionSimplicity(allVisibleActions);

  const totalInputCandidates = candidates.length;
  const suppressedCount = Math.max(0, totalInputCandidates - allVisibleActions.length);
  const reductionRatio = Number(
    ((suppressedCount / Math.max(1, totalInputCandidates)) * 100).toFixed(1)
  );

  return {
    primaryAction: primaryCandidate,
    secondaryActions: runnerUps,
    suppressedActionsCount: suppressedCount,
    intelligenceReductionRatio: reductionRatio,
    auditCompliance: {
      invOi97Cleared: primaryCandidate.domain !== 'TRADING' || context.recoveryScore >= 55,
      invOi98Cleared: (context.liquidCash / Math.max(1, context.monthlyEssentialBurn)) >= 6.0,
      invOi99Compliant: audit99.compliant,
      invOi101Compliant: audit101.compliant,
    },
  };
}
