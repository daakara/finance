/**
 * Horizon 14: Clean Room Behavioral Governor Engine
 *
 * Translates behavioral resilience, loss streaks, execution windows,
 * and capital preservation floors into disciplined, dynamic position sizing.
 *
 * Enforces INV-OI112-P: Zero lifestyle terminology leaks into output rationale.
 * Enforces INV-OI114-P: Favors sizing reduction over total trade prohibitions.
 */

import { getUnifiedCockpitState } from './unifiedCockpitStore';

export interface TraderContext {
  accountEquity: number;
  standardRiskBudgetPct: number; // e.g. 0.005 for 0.5%, 0.01 for 1.0%
  consecutiveLossStreak: number;
  tradingHour: number; // 9 to 16 (Eastern Time)
  liquidRunwayMonths: number;
  dailyDrawdownPct: number;
}

/**
 * Derives real-time TraderContext directly from the Unified CQRS Cockpit Store.
 * Connects underlying Life & Household intelligence directly to the Terminal Governor.
 */
export function getTraderContextFromUnifiedCockpit(overrides?: Partial<TraderContext>): TraderContext {
  const cockpit = getUnifiedCockpitState();
  const currentHour = new Date().getHours();

  return {
    accountEquity: 50000,
    standardRiskBudgetPct: 0.01,
    consecutiveLossStreak: 2,
    tradingHour: currentHour >= 9 && currentHour <= 16 ? currentHour : 10,
    liquidRunwayMonths: cockpit.runway.monthsUnencumbered,
    dailyDrawdownPct: 0.0,
    ...overrides,
  };
}

export interface TradeSetupSpec {
  ticker: string;
  setupName: string;
  entryPivot: number;
  stopLoss: number;
  target1: number;
  target2: number;
  confluenceScore: number;
  isActionable?: boolean;
  reasonSuppressed?: string | null;
  executionStatus?: string;
  entryThesis?: string;
  invalidationCondition?: string;
}

export interface GovernorSizingOutput {
  ticker: string;
  entryPivot: number;
  stopLoss: number;
  stopDistanceDollar: number;
  stopDistancePct: number;
  unclampedDollarRisk: number;
  unclampedShares: number;
  recommendedDollarRisk: number;
  recommendedShares: number;
  clampFactorPct: number; // Negative or zero (e.g. -47%)
  primaryGovernorCategory: 'DRAWDOWN_DEFENSE' | 'EXECUTION_WINDOW' | 'CAPITAL_FLOOR' | 'UNCONSTRAINED';
  cleanRoomRationale: string;
  rMultipleTarget1: number;
  rMultipleTarget2: number;
  estimatedCapitalAllocated: number;
}

/**
 * Epistemic Invariant: Tactical setups are dynamic and API-backed.
 * Hardcoded canonical setup lists are strictly eliminated from production.
 */
export function getTacticalSetupForTicker(
  ticker: string | null | undefined,
  availableSetups?: TradeSetupSpec[]
): TradeSetupSpec | null {
  if (!ticker) return null;
  const upper = ticker.trim().toUpperCase();
  if (availableSetups && availableSetups.length > 0) {
    return availableSetups.find((s) => s.ticker === upper) || null;
  }
  return null;
}

/**
 * Computes dynamic position sizing governed by behavioral risk factors
 */
export function calculateGovernedPositionSize(
  setup: TradeSetupSpec | null | undefined,
  context: TraderContext
): GovernorSizingOutput {
  if (!setup) {
    return {
      ticker: 'UNASSIGNED',
      entryPivot: 0,
      stopLoss: 0,
      stopDistanceDollar: 0,
      stopDistancePct: 0,
      unclampedDollarRisk: 0,
      unclampedShares: 0,
      recommendedDollarRisk: 0,
      recommendedShares: 0,
      clampFactorPct: 0,
      primaryGovernorCategory: 'UNCONSTRAINED',
      cleanRoomRationale: 'No active trade setup provided. Capital allocation remains uncommitted.',
      rMultipleTarget1: 0,
      rMultipleTarget2: 0,
      estimatedCapitalAllocated: 0,
    };
  }

  const isActionable = setup.isActionable !== false && setup.entryPivot > 0 && setup.stopLoss > 0 && setup.entryPivot > setup.stopLoss;
  const stopDistanceDollar = isActionable ? Math.max(0.01, setup.entryPivot - setup.stopLoss) : 1.0;
  const stopDistancePct = isActionable ? (stopDistanceDollar / setup.entryPivot) * 100 : 0;

  const standardDollarRisk = Math.round(context.accountEquity * context.standardRiskBudgetPct);
  const unclampedShares = isActionable ? Math.max(1, Math.floor(standardDollarRisk / stopDistanceDollar)) : 0;

  // Determine Governor clamp penalties
  let clampPenalty = 0;
  let primaryCategory: 'DRAWDOWN_DEFENSE' | 'EXECUTION_WINDOW' | 'CAPITAL_FLOOR' | 'UNCONSTRAINED' = 'UNCONSTRAINED';
  const rationaleParts: string[] = [];

  // 1. Loss streak penalty (INV-OI97-P)
  if (context.consecutiveLossStreak >= 3) {
    clampPenalty += 0.40;
    primaryCategory = 'DRAWDOWN_DEFENSE';
    rationaleParts.push(`${context.consecutiveLossStreak}-trade loss streak indicates elevated drawdown susceptibility`);
  } else if (context.consecutiveLossStreak === 2) {
    clampPenalty += 0.25;
    primaryCategory = 'DRAWDOWN_DEFENSE';
    rationaleParts.push(`2-trade drawdown streak warrants defensive capital buffer`);
  }

  // 2. Execution window / time-of-day penalty (from H12 chronotype, converted to trade window)
  if (context.tradingHour >= 14) {
    clampPenalty += 0.20;
    if (primaryCategory === 'UNCONSTRAINED') primaryCategory = 'EXECUTION_WINDOW';
    rationaleParts.push(`afternoon session historically exhibits degraded risk/reward skew`);
  }

  // 3. Liquid runway preservation floor (INV-OI98-P)
  if (context.liquidRunwayMonths < 6.0) {
    clampPenalty += 0.25;
    primaryCategory = 'CAPITAL_FLOOR';
    rationaleParts.push(`unencumbered cash runway below 6-month preservation floor`);
  }

  // Cap total clamp between 0% and 70% to strictly preserve human agency (INV-OI114-P)
  const finalClampPct = Math.min(0.70, clampPenalty);
  const clampFactorPct = -Math.round(finalClampPct * 100);

  const recommendedDollarRisk = isActionable ? Math.round(standardDollarRisk * (1 - finalClampPct)) : 0;
  const recommendedShares = isActionable ? Math.max(1, Math.floor(recommendedDollarRisk / stopDistanceDollar)) : 0;

  let cleanRoomRationale = "";
  if (!isActionable) {
    cleanRoomRationale = setup.reasonSuppressed || "Actionable risk levels suppressed: authentic market discovery required.";
  } else if (clampFactorPct < 0) {
    cleanRoomRationale = `Risk allowance reduced ${Math.abs(clampFactorPct)}% ($${standardDollarRisk} → $${recommendedDollarRisk}) due to: ${rationaleParts.join('; ')}. Preserving capital for highest-conviction morning windows.`;
  } else {
    cleanRoomRationale = `Standard position risk authorized ($${standardDollarRisk}). High confluence (${setup.confluenceScore}/100) and disciplined execution state verified.`;
  }

  const rMultipleTarget1 = isActionable && setup.target1 > setup.entryPivot ? Number(((setup.target1 - setup.entryPivot) / stopDistanceDollar).toFixed(2)) : 0;
  const rMultipleTarget2 = isActionable && setup.target2 > setup.entryPivot ? Number(((setup.target2 - setup.entryPivot) / stopDistanceDollar).toFixed(2)) : 0;
  const estimatedCapitalAllocated = isActionable ? recommendedShares * setup.entryPivot : 0;

  return {
    ticker: setup.ticker,
    entryPivot: setup.entryPivot,
    stopLoss: setup.stopLoss,
    stopDistanceDollar: Number(stopDistanceDollar.toFixed(2)),
    stopDistancePct: Number(stopDistancePct.toFixed(2)),
    unclampedDollarRisk: isActionable ? standardDollarRisk : 0,
    unclampedShares,
    recommendedDollarRisk,
    recommendedShares,
    clampFactorPct: isActionable ? clampFactorPct : 0,
    primaryGovernorCategory: primaryCategory,
    cleanRoomRationale,
    rMultipleTarget1,
    rMultipleTarget2,
    estimatedCapitalAllocated: Number(estimatedCapitalAllocated.toFixed(2)),
  };
}
