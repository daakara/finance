/**
 * Horizon 10: Cognitive Trading & Life Companion Engine
 *
 * Connects quantitative market setups (/screener, /portfolio, /smart-money)
 * with personal biology and household runway (/me, /me/signals, /me/household).
 *
 * Enforces:
 * - INV-OI97-P: Cognitive Trading Discipline
 * - INV-OI98-P: Household Capital Protection
 *
 * Emits standardized CandidateAction for the Next Best Action (NBA) Engine.
 */

import { CandidateAction, UserOperationalContext } from './nextBestActionEngine';
import {
  verifyCognitiveTradingDiscipline,
  verifyHouseholdCapitalProtection,
} from './horizon10Invariants';

export interface MarketEnvironment {
  regime: 'TRENDING_BULL' | 'CHOPPY_SIDEWAYS' | 'HIGH_VOLATILITY_BEAR';
  vixLevel: number;
  marketTrendConfidence: number; // 0-100
}

export interface TradingOpportunity {
  ticker: string;
  companyName: string;
  setupPattern: string; // e.g. "Minervini VCP", "Magic Formula ROC"
  entryPrice: number;
  stopLoss: number;
  targetPrice: number;
  confluenceScore: number; // 0-100
  thesis: string;
}

export interface CognitiveTradingAssessment {
  candidateAction: CandidateAction;
  calculatedMaxDollarRisk: number;
  recommendedShares: number;
  postTradeRunwayMonths: number;
  mentalClarityRating: 'PRIME' | 'MODERATE' | 'IMPAIRED';
  disciplineShieldActive: boolean;
  shieldReason?: string;
}

export const CANONICAL_MARKET_OPPORTUNITIES: TradingOpportunity[] = [
  {
    ticker: 'GOOGL',
    companyName: 'Alphabet Inc.',
    setupPattern: 'Minervini VCP 3-Stage Contraction',
    entryPrice: 178.5,
    stopLoss: 171.0, // Risk per share = $7.50 (4.2%)
    targetPrice: 198.0, // Gain per share = $19.50 (10.9%)
    confluenceScore: 92,
    thesis:
      'Institutional accumulation at 20-day moving average with high ROIC and PEG < 1.4.',
  },
  {
    ticker: 'NVDA',
    companyName: 'NVIDIA Corp.',
    setupPattern: 'High Tight Flag Breakout',
    entryPrice: 122.0,
    stopLoss: 116.0,
    targetPrice: 138.0,
    confluenceScore: 88,
    thesis: 'Consolidation breakout after earnings digest with surging volume.',
  },
];

/**
 * Computes safe dollar risk per trade dynamically based on biometric recovery and runway.
 */
export function calculateDynamicMaxDollarRisk(
  baselineRiskDollars: number,
  liquidRunwayMonths: number,
  recoveryScore: number,
  recentLossStreak: number
): number {
  if (recoveryScore < 55 || recentLossStreak >= 2) {
    return 0; // Shielded to Paper Mode
  }

  // Biometric damping: quadratic scaling below 100%
  const recoveryFactor = Math.pow(Math.min(1.0, recoveryScore / 100), 2);

  // Runway damping: scales down linearly if runway is below 6 months
  const runwayFactor = Math.max(0, Math.min(1.0, (liquidRunwayMonths - 3.0) / 3.0));

  const safeRisk = baselineRiskDollars * recoveryFactor * runwayFactor;
  return Number(Math.max(0, safeRisk).toFixed(0));
}

/**
 * Evaluates market opportunity through cognitive and household invariants,
 * producing a candidate action suitable for the NBA engine.
 */
export function evaluateCognitiveTradingOpportunity(
  opportunity: TradingOpportunity,
  context: UserOperationalContext,
  market: MarketEnvironment
): CognitiveTradingAssessment {
  const disciplineAudit = verifyCognitiveTradingDiscipline(
    context.recoveryScore,
    context.recentLossStreak,
    context.dailyDrawdownPct
  );

  const baselineRiskBudget = 200; // $200 baseline risk for a standard portfolio
  const liquidRunwayMonths = context.liquidCash / Math.max(1, context.monthlyEssentialBurn);

  const safeDollarRisk = calculateDynamicMaxDollarRisk(
    baselineRiskBudget,
    liquidRunwayMonths,
    context.recoveryScore,
    context.recentLossStreak
  );

  const riskPerShare = Math.max(0.5, opportunity.entryPrice - opportunity.stopLoss);
  const recommendedShares =
    safeDollarRisk > 0 ? Math.max(1, Math.floor(safeDollarRisk / riskPerShare)) : 0;
  const totalCapitalRequired = recommendedShares * opportunity.entryPrice;

  const capitalAudit = verifyHouseholdCapitalProtection(
    context.liquidCash,
    context.monthlyEssentialBurn,
    totalCapitalRequired
  );

  let mentalClarity: 'PRIME' | 'MODERATE' | 'IMPAIRED' = 'PRIME';
  if (context.recoveryScore < 55) mentalClarity = 'IMPAIRED';
  else if (context.recoveryScore < 75) mentalClarity = 'MODERATE';

  // Determine CandidateAction based on safety shield
  let candidateAction: CandidateAction;

  if (!disciplineAudit.safeToTrade || safeDollarRisk === 0) {
    candidateAction = {
      id: `ACT-TRADE-SHIELD-${opportunity.ticker}`,
      domain: 'TRADING',
      headline: 'Trading Shield Active: Paper Mode Only',
      explanation: `${disciplineAudit.reason} Active trade entries are disabled to protect capital and mental poise.`,
      utilityScore: 40, // Low utility so health/household actions take precedence
      expectedImpact: {
        lhiDelta: 3.5,
        stressReductionPct: 30,
      },
      frictionRating: 'EFFORTLESS',
      urgency: 'CRITICAL_TODAY',
      guardrailStatus: 'CLEARED', // The shield recommendation itself is safe
      proofDetails: {
        setupType: 'Behavioral Circuit Breaker',
        maxDollarRisk: 0,
        dagTraceNode: 'Fatigue -> Amygdala Hijack Prevention -> Capital Preservation',
      },
      actionPayload: {
        route: '/screener',
        ctaLabel: 'View Paper Trading Setup',
      },
    };

    return {
      candidateAction,
      calculatedMaxDollarRisk: 0,
      recommendedShares: 0,
      postTradeRunwayMonths: Number(liquidRunwayMonths.toFixed(1)),
      mentalClarityRating: mentalClarity,
      disciplineShieldActive: true,
      shieldReason: disciplineAudit.reason,
    };
  }

  // Market Chop Protection: If market is sideways/bearish and setup score is moderate
  if (market.regime === 'CHOPPY_SIDEWAYS' && opportunity.confluenceScore < 85) {
    candidateAction = {
      id: 'ACT-TRADE-HOLD-CASH',
      domain: 'TRADING',
      headline: 'Action: Hold Cash Today',
      explanation:
        'Market chop is elevated and risk/reward is unfavorable. Preserving cash runway maintains high future optionality.',
      utilityScore: 78,
      expectedImpact: {
        lhiDelta: 1.8,
        stressReductionPct: 20,
      },
      frictionRating: 'EFFORTLESS',
      urgency: 'OPPORTUNITY',
      guardrailStatus: 'CLEARED',
      proofDetails: {
        setupType: 'Cash Preservation Posture',
        maxDollarRisk: 0,
      },
      actionPayload: {
        route: '/portfolio',
        ctaLabel: 'Review Cash Vault',
      },
    };

    return {
      candidateAction,
      calculatedMaxDollarRisk: 0,
      recommendedShares: 0,
      postTradeRunwayMonths: Number(liquidRunwayMonths.toFixed(1)),
      mentalClarityRating: mentalClarity,
      disciplineShieldActive: false,
    };
  }

  // High-Conviction Opportunity Setup
  const rewardRiskRatio = Number(
    ((opportunity.targetPrice - opportunity.entryPrice) / riskPerShare).toFixed(1)
  );

  candidateAction = {
    id: `ACT-TRADE-${opportunity.ticker}`,
    domain: 'TRADING',
    headline: `High-Conviction Setup: ${opportunity.ticker}`,
    explanation: `${opportunity.companyName} shows institutional accumulation. Buy ${recommendedShares} shares at $${opportunity.entryPrice.toFixed(2)}, risking $${safeDollarRisk} to stop at $${opportunity.stopLoss.toFixed(2)}.`,
    utilityScore: 91, // High utility when nominal
    expectedImpact: {
      lhiDelta: 2.4,
      financialDeltaDollars: Math.round(
        recommendedShares * (opportunity.targetPrice - opportunity.entryPrice)
      ),
      stressReductionPct: 0,
    },
    frictionRating: 'LOW_FRICTION',
    urgency: 'OPPORTUNITY',
    guardrailStatus: capitalAudit.compliant ? 'CLEARED' : 'BLOCKED_BY_INVARIANT',
    invariantsViolated: capitalAudit.violations,
    proofDetails: {
      setupType: opportunity.setupPattern,
      riskRewardRatio: rewardRiskRatio,
      maxDollarRisk: safeDollarRisk,
      dagTraceNode: 'Biometrics Prime -> Confluence Conviction -> Exact Position Sizing',
      confidenceInterval: [0.76, 0.88],
    },
    actionPayload: {
      route: '/screener',
      ctaLabel: 'Review Execution Plan',
    },
  };

  return {
    candidateAction,
    calculatedMaxDollarRisk: safeDollarRisk,
    recommendedShares,
    postTradeRunwayMonths: capitalAudit.postDeploymentRunwayMonths,
    mentalClarityRating: mentalClarity,
    disciplineShieldActive: false,
  };
}
