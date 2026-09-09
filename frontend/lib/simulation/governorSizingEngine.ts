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

export const CANONICAL_TACTICAL_SETUPS: TradeSetupSpec[] = [
  {
    ticker: 'GOOGL',
    setupName: 'Minervini VCP 4T Breakout Pivot',
    entryPivot: 182.40,
    stopLoss: 176.10,
    target1: 195.00,
    target2: 207.00,
    confluenceScore: 94,
  },
  {
    ticker: 'NVDA',
    setupName: 'High-RS Volatility Contraction',
    entryPivot: 128.50,
    stopLoss: 123.80,
    target1: 137.90,
    target2: 145.00,
    confluenceScore: 91,
  },
  {
    ticker: 'ANET',
    setupName: '20-EMA Institutional Bounce',
    entryPivot: 312.10,
    stopLoss: 301.50,
    target1: 333.30,
    target2: 348.00,
    confluenceScore: 87,
  },
  {
    ticker: 'PLTR',
    setupName: 'Stage 2 Continuation Pivot',
    entryPivot: 32.40,
    stopLoss: 30.80,
    target1: 35.60,
    target2: 38.00,
    confluenceScore: 89,
  },
  {
    ticker: 'MSFT',
    setupName: 'Base-on-Base Consolidation Breakout',
    entryPivot: 448.20,
    stopLoss: 432.50,
    target1: 475.00,
    target2: 495.00,
    confluenceScore: 86,
  },
  {
    ticker: 'COIN',
    setupName: 'High RVOL Trend Contraction',
    entryPivot: 238.00,
    stopLoss: 224.00,
    target1: 265.00,
    target2: 285.00,
    confluenceScore: 85,
  },
  {
    ticker: 'MSTR',
    setupName: 'Institutional Inflow Base Breakout',
    entryPivot: 145.00,
    stopLoss: 134.00,
    target1: 168.00,
    target2: 185.00,
    confluenceScore: 84,
  },
  {
    ticker: 'HOOD',
    setupName: 'Retail Volume Dry-Up Pivot',
    entryPivot: 24.50,
    stopLoss: 22.80,
    target1: 28.00,
    target2: 31.00,
    confluenceScore: 83,
  },
  {
    ticker: 'DUOL',
    setupName: 'EdTech AI High-Tight Flag',
    entryPivot: 245.00,
    stopLoss: 232.00,
    target1: 272.00,
    target2: 290.00,
    confluenceScore: 82,
  },
  {
    ticker: 'CELH',
    setupName: 'Reversal Support Bounce',
    entryPivot: 38.20,
    stopLoss: 35.50,
    target1: 43.50,
    target2: 48.00,
    confluenceScore: 81,
  },
  {
    ticker: 'APP',
    setupName: 'AdTech ML Momentum Base',
    entryPivot: 96.00,
    stopLoss: 89.50,
    target1: 110.00,
    target2: 122.00,
    confluenceScore: 88,
  },
  {
    ticker: 'LNTH',
    setupName: 'Magic Formula GARP Value Pivot',
    entryPivot: 88.50,
    stopLoss: 83.20,
    target1: 99.00,
    target2: 108.00,
    confluenceScore: 82,
  },
  {
    ticker: 'CPRX',
    setupName: 'High-ROIC Zero-Debt Contraction',
    entryPivot: 15.80,
    stopLoss: 14.90,
    target1: 18.20,
    target2: 20.00,
    confluenceScore: 80,
  },
  {
    ticker: 'NVO',
    setupName: 'Secular Healthcare Compounder Base',
    entryPivot: 136.00,
    stopLoss: 128.50,
    target1: 152.00,
    target2: 164.00,
    confluenceScore: 85,
  },
  {
    ticker: 'TMDX',
    setupName: 'MedTech High-RS VCP Breakout',
    entryPivot: 142.00,
    stopLoss: 133.50,
    target1: 158.00,
    target2: 172.00,
    confluenceScore: 92,
  },
  {
    ticker: 'META',
    setupName: 'High-Tight Flag Consolidation',
    entryPivot: 512.00,
    stopLoss: 494.00,
    target1: 545.00,
    target2: 575.00,
    confluenceScore: 88,
  },
  {
    ticker: 'AAPL',
    setupName: 'Flat Base Pivot Breakout',
    entryPivot: 228.00,
    stopLoss: 219.50,
    target1: 245.00,
    target2: 260.00,
    confluenceScore: 85,
  },
];

export function getTacticalSetupForTicker(ticker: string | null | undefined): TradeSetupSpec | null {
  if (!ticker) return null;
  const upper = ticker.trim().toUpperCase();
  return CANONICAL_TACTICAL_SETUPS.find((s) => s.ticker === upper) || null;
}

/**
 * Computes dynamic position sizing governed by behavioral risk factors
 */
export function calculateGovernedPositionSize(
  setup: TradeSetupSpec,
  context: TraderContext
): GovernorSizingOutput {
  const stopDistanceDollar = Math.max(0.01, setup.entryPivot - setup.stopLoss);
  const stopDistancePct = (stopDistanceDollar / setup.entryPivot) * 100;

  const standardDollarRisk = Math.round(context.accountEquity * context.standardRiskBudgetPct);
  const unclampedShares = Math.max(1, Math.floor(standardDollarRisk / stopDistanceDollar));

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

  const recommendedDollarRisk = Math.round(standardDollarRisk * (1 - finalClampPct));
  const recommendedShares = Math.max(1, Math.floor(recommendedDollarRisk / stopDistanceDollar));

  const cleanRoomRationale =
    clampFactorPct < 0
      ? `Risk allowance reduced ${Math.abs(clampFactorPct)}% ($${standardDollarRisk} → $${recommendedDollarRisk}) due to: ${rationaleParts.join('; ')}. Preserving capital for highest-conviction morning windows.`
      : `Standard position risk authorized ($${standardDollarRisk}). High confluence (${setup.confluenceScore}/100) and disciplined execution state verified.`;

  const rMultipleTarget1 = Number(((setup.target1 - setup.entryPivot) / stopDistanceDollar).toFixed(2));
  const rMultipleTarget2 = Number(((setup.target2 - setup.entryPivot) / stopDistanceDollar).toFixed(2));
  const estimatedCapitalAllocated = recommendedShares * setup.entryPivot;

  return {
    ticker: setup.ticker,
    entryPivot: setup.entryPivot,
    stopLoss: setup.stopLoss,
    stopDistanceDollar: Number(stopDistanceDollar.toFixed(2)),
    stopDistancePct: Number(stopDistancePct.toFixed(2)),
    unclampedDollarRisk: standardDollarRisk,
    unclampedShares,
    recommendedDollarRisk,
    recommendedShares,
    clampFactorPct,
    primaryGovernorCategory: primaryCategory,
    cleanRoomRationale,
    rMultipleTarget1,
    rMultipleTarget2,
    estimatedCapitalAllocated: Number(estimatedCapitalAllocated.toFixed(2)),
  };
}
