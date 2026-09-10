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
import { loadPortfolioPositions, calculatePortfolioSummary } from '../portfolio';

export interface TraderContext {
  accountEquity: number | null;
  standardRiskBudgetPct: number; // e.g. 0.005 for 0.5%, 0.01 for 1.0%
  consecutiveLossStreak: number | null;
  tradingHour: number | null; // 0 to 23 in Eastern Time (America/New_York), or null if unestablished
  liquidRunwayMonths: number | null;
  dailyDrawdownPct: number | null;
  isAvailable: boolean;
  unavailableReason?: string;
  missingInputs?: string[];
}

/**
 * Returns current trading hour in America/New_York (Eastern Time: 0 to 23),
 * or null if market time cannot be established authoritatively.
 *
 * Zero fabricated fallback to hour 10. Zero hardcoded UTC-4 guessing.
 * Uses authoritative America/New_York IANA timezone rules across both
 * Daylight Saving Time (EDT: UTC-4) and Standard Time (EST: UTC-5).
 */
export function getEasternTradingHour(dateInput?: Date | string | number | null): number | null {
  try {
    const d = dateInput ? (dateInput instanceof Date ? dateInput : new Date(dateInput)) : new Date();
    if (isNaN(d.getTime())) {
      return null;
    }

    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      hour: "numeric",
      hourCycle: "h23", // Guarantees 00-23 in all environments; midnight is 0
    });

    const parts = formatter.formatToParts(d);
    const hourPart = parts.find((p) => p.type === "hour");
    if (!hourPart) {
      return null;
    }

    const parsed = parseInt(hourPart.value, 10);
    if (isNaN(parsed) || parsed < 0 || parsed > 23) {
      return null;
    }

    return parsed;
  } catch {
    // If Intl or America/New_York is unsupported, return explicit null (unavailable).
    // NEVER substitute hour 10 or guess with fixed UTC-4 offset.
    return null;
  }
}

/**
 * Derives real-time TraderContext directly from persistent storage, API state,
 * and authentic Eastern Time. Zero manufactured $50,000 equity or fake loss streaks.
 * Sizing is strictly unavailable if account equity or risk history is unconfigured.
 */
export function getTraderContextFromUnifiedCockpit(overrides?: Partial<TraderContext>): TraderContext {
  const cockpit = getUnifiedCockpitState();
  const tradingHour = getEasternTradingHour();

  let accountEquity: number | null = null;

  // 1. Authoritative API-backed data path: /api/v1/cockpit/state portfolio summary takes precedence
  if (cockpit.portfolio?.isComplete && cockpit.portfolio?.totalMarketValue !== null && cockpit.portfolio.totalMarketValue > 0) {
    accountEquity = cockpit.portfolio.totalMarketValue;
  } else if (typeof window !== "undefined") {
    // 2. Fallback to locally loaded portfolio from /portfolio API only when cockpit portfolio summary is unpopulated
    const positions = loadPortfolioPositions();
    if (positions.length > 0) {
      const summary = calculatePortfolioSummary(positions);
      // Only use live priced totalEquity when isComplete is true and totalEquity > 0.
      // Never substitute cost basis (summary.totalCost) for unpriced equity.
      if (summary.isComplete && summary.totalEquity !== null && summary.totalEquity > 0) {
        accountEquity = summary.totalEquity;
      }
    }
    // 3. Fallback to localStorage ONLY if no API portfolio data exists.
    // Editing browser storage cannot override authoritative API risk inputs.
    if (!accountEquity) {
      const savedSize = localStorage.getItem("FINANCE_USER_ACCOUNT_SIZE");
      if (savedSize && !isNaN(Number(savedSize)) && Number(savedSize) > 0) {
        accountEquity = Number(savedSize);
      }
    }
  }

  // Never substitute household liquid reserves (cockpit.runway.liquidReserves) for trading account equity.

  let consecutiveLossStreak: number | null = null;
  let dailyDrawdownPct: number | null = null;

  if (typeof window !== "undefined") {
    const savedLoss = localStorage.getItem("FINANCE_JOURNAL_LOSS_STREAK");
    if (savedLoss !== null && savedLoss.trim() !== "" && !isNaN(Number(savedLoss))) {
      consecutiveLossStreak = Math.max(0, parseInt(savedLoss, 10));
    } else {
      try {
        const rawLogs = localStorage.getItem("FINANCE_JOURNAL_LOGS");
        if (rawLogs) {
          const parsed = JSON.parse(rawLogs);
          if (Array.isArray(parsed) && parsed.length > 0) {
            let streak = 0;
            for (let i = parsed.length - 1; i >= 0; i--) {
              if (parsed[i].rAchieved < 0) {
                streak++;
              } else {
                break;
              }
            }
            consecutiveLossStreak = streak;
          }
        }
      } catch {}
    }

    const savedDd = localStorage.getItem("FINANCE_DAILY_DRAWDOWN_PCT");
    if (savedDd !== null && savedDd.trim() !== "" && !isNaN(Number(savedDd))) {
      dailyDrawdownPct = Math.max(0, parseFloat(savedDd));
    }
  }

  const missingInputs: string[] = [];
  if (!accountEquity || accountEquity <= 0) {
    missingInputs.push("Account Equity (unrecorded in API portfolio holdings)");
  }
  if (consecutiveLossStreak === null || consecutiveLossStreak === undefined) {
    missingInputs.push("Loss Streak History (unresolved data dependency: no backend API trade journal endpoint exists)");
  }
  if (dailyDrawdownPct === null || dailyDrawdownPct === undefined) {
    missingInputs.push("Daily Drawdown History (unresolved data dependency: no automated intraday drawdown telemetry exists)");
  }
  if (tradingHour === null || tradingHour === undefined) {
    missingInputs.push("Market Trading Time (unable to establish Eastern session time)");
  }

  const isAvailable = missingInputs.length === 0;
  const unavailableReason = !isAvailable
    ? `Governor sizing unavailable: Missing required risk input(s): ${missingInputs.join(", ")}. Sizing calculation and ticket copying are disabled until risk parameters are configured.`
    : undefined;

  const result: TraderContext = {
    accountEquity,
    standardRiskBudgetPct: 0.01,
    consecutiveLossStreak,
    tradingHour,
    liquidRunwayMonths: cockpit.runway?.monthsUnencumbered ?? null,
    dailyDrawdownPct,
    isAvailable,
    unavailableReason,
    missingInputs,
    ...overrides,
  };

  if (overrides) {
    const hasEq = result.accountEquity !== null && result.accountEquity > 0;
    const hasStreak = result.consecutiveLossStreak !== null && result.consecutiveLossStreak !== undefined;
    const hasDd = result.dailyDrawdownPct !== null && result.dailyDrawdownPct !== undefined;
    const hasHour = result.tradingHour !== null && result.tradingHour !== undefined;
    if (overrides.isAvailable === undefined) {
      result.isAvailable = hasEq && hasStreak && hasDd && hasHour;
      if (!result.isAvailable && !result.unavailableReason) {
        const missing: string[] = [];
        if (!hasEq) missing.push("Account Equity");
        if (!hasStreak) missing.push("Loss Streak History");
        if (!hasDd) missing.push("Daily Drawdown History");
        if (!hasHour) missing.push("Market Trading Time");
        result.unavailableReason = `Governor sizing unavailable: Missing required risk input(s): ${missing.join(", ")}.`;
      }
    }
  }

  return result;
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
  stagePhase?: string;
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
  isAvailable?: boolean;
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
      isAvailable: false,
    };
  }

  const entryPivotNum = (setup && typeof setup.entryPivot === 'number' && !isNaN(setup.entryPivot)) ? setup.entryPivot : 0;
  const stopLossNum = (setup && typeof setup.stopLoss === 'number' && !isNaN(setup.stopLoss)) ? setup.stopLoss : 0;
  const target1Num = (setup && typeof setup.target1 === 'number' && !isNaN(setup.target1)) ? setup.target1 : 0;
  const target2Num = (setup && typeof setup.target2 === 'number' && !isNaN(setup.target2)) ? setup.target2 : 0;

  const isActionable = Boolean(
    setup.isActionable !== false &&
    entryPivotNum > 0 &&
    stopLossNum > 0 &&
    entryPivotNum > stopLossNum
  );
  const stopDistanceDollar = isActionable ? Math.max(0.01, entryPivotNum - stopLossNum) : 0;
  const stopDistancePct = isActionable && entryPivotNum > 0 ? (stopDistanceDollar / entryPivotNum) * 100 : 0;

  // If context is unavailable, do NOT invent fake numbers
  if (
    !context.isAvailable ||
    context.accountEquity === null ||
    context.accountEquity <= 0 ||
    context.consecutiveLossStreak === null ||
    context.consecutiveLossStreak === undefined ||
    context.dailyDrawdownPct === null ||
    context.dailyDrawdownPct === undefined ||
    context.tradingHour === null ||
    context.tradingHour === undefined
  ) {
    const missing: string[] = [];
    if (context.accountEquity === null || context.accountEquity <= 0) missing.push("Account Equity");
    if (context.consecutiveLossStreak === null || context.consecutiveLossStreak === undefined) missing.push("Loss Streak History");
    if (context.dailyDrawdownPct === null || context.dailyDrawdownPct === undefined) missing.push("Daily Drawdown History");
    if (context.tradingHour === null || context.tradingHour === undefined) missing.push("Market Trading Time");
    const reason = context.unavailableReason || `Governor sizing unavailable: Missing required risk input(s): ${missing.join(", ")}. Sizing calculation and ticket copying are disabled until risk parameters are configured.`;

    return {
      ticker: setup.ticker || 'UNASSIGNED',
      entryPivot: entryPivotNum,
      stopLoss: stopLossNum,
      stopDistanceDollar: Number(stopDistanceDollar.toFixed(2)),
      stopDistancePct: Number(stopDistancePct.toFixed(2)),
      unclampedDollarRisk: 0,
      unclampedShares: 0,
      recommendedDollarRisk: 0,
      recommendedShares: 0,
      clampFactorPct: 0,
      primaryGovernorCategory: 'CAPITAL_FLOOR',
      cleanRoomRationale: reason,
      rMultipleTarget1: 0,
      rMultipleTarget2: 0,
      estimatedCapitalAllocated: 0,
      isAvailable: false,
    };
  }

  const standardDollarRisk = Math.round(context.accountEquity * context.standardRiskBudgetPct);
  const unclampedShares = isActionable && stopDistanceDollar > 0 ? Math.max(1, Math.floor(standardDollarRisk / stopDistanceDollar)) : 0;

  // Determine Governor clamp penalties
  let clampPenalty = 0;
  let primaryCategory: 'DRAWDOWN_DEFENSE' | 'EXECUTION_WINDOW' | 'CAPITAL_FLOOR' | 'UNCONSTRAINED' = 'UNCONSTRAINED';
  const rationaleParts: string[] = [];

  // 1. Loss streak penalty (INV-OI97-P)
  if (context.consecutiveLossStreak !== null && context.consecutiveLossStreak >= 3) {
    clampPenalty += 0.40;
    primaryCategory = 'DRAWDOWN_DEFENSE';
    rationaleParts.push(`${context.consecutiveLossStreak}-trade loss streak indicates elevated drawdown susceptibility`);
  } else if (context.consecutiveLossStreak !== null && context.consecutiveLossStreak === 2) {
    clampPenalty += 0.25;
    primaryCategory = 'DRAWDOWN_DEFENSE';
    rationaleParts.push(`2-trade drawdown streak warrants defensive capital buffer`);
  }

  // 2. Daily session drawdown penalty
  if (context.dailyDrawdownPct !== null && context.dailyDrawdownPct >= 2.0) {
    clampPenalty += 0.30;
    primaryCategory = 'DRAWDOWN_DEFENSE';
    rationaleParts.push(`daily session drawdown of ${context.dailyDrawdownPct.toFixed(1)}% exceeds defensive threshold`);
  } else if (context.dailyDrawdownPct !== null && context.dailyDrawdownPct >= 1.0) {
    clampPenalty += 0.15;
    if (primaryCategory === 'UNCONSTRAINED') primaryCategory = 'DRAWDOWN_DEFENSE';
    rationaleParts.push(`daily session drawdown of ${context.dailyDrawdownPct.toFixed(1)}% active`);
  }

  // 3. Execution window / time-of-day penalty in America/New_York
  if (context.tradingHour !== null && context.tradingHour >= 14) {
    clampPenalty += 0.20;
    if (primaryCategory === 'UNCONSTRAINED') primaryCategory = 'EXECUTION_WINDOW';
    rationaleParts.push(`afternoon session historically exhibits degraded risk/reward skew`);
  }

  // 3. Liquid runway preservation floor (INV-OI98-P)
  if (context.liquidRunwayMonths !== null && context.liquidRunwayMonths < 6.0) {
    clampPenalty += 0.25;
    primaryCategory = 'CAPITAL_FLOOR';
    rationaleParts.push(`unencumbered cash runway below 6-month preservation floor`);
  }

  // Cap total clamp between 0% and 70% to strictly preserve human agency (INV-OI114-P)
  const finalClampPct = Math.min(0.70, clampPenalty);
  const clampFactorPct = -Math.round(finalClampPct * 100);

  const recommendedDollarRisk = isActionable ? Math.round(standardDollarRisk * (1 - finalClampPct)) : 0;
  const recommendedShares = isActionable && stopDistanceDollar > 0 ? Math.max(1, Math.floor(recommendedDollarRisk / stopDistanceDollar)) : 0;

  let cleanRoomRationale = "";
  if (!isActionable) {
    cleanRoomRationale = setup.reasonSuppressed || "Actionable risk levels suppressed: authentic market discovery required.";
  } else if (clampFactorPct < 0) {
    cleanRoomRationale = `Risk allowance reduced ${Math.abs(clampFactorPct)}% ($${standardDollarRisk} → $${recommendedDollarRisk}) due to: ${rationaleParts.join('; ')}. Preserving capital for highest-conviction morning windows.`;
  } else {
    cleanRoomRationale = `Standard position risk authorized ($${standardDollarRisk}). High confluence (${setup.confluenceScore || 0}/100) and disciplined execution state verified.`;
  }

  const rMultipleTarget1 = isActionable && target1Num > entryPivotNum && stopDistanceDollar > 0 ? Number(((target1Num - entryPivotNum) / stopDistanceDollar).toFixed(2)) : 0;
  const rMultipleTarget2 = isActionable && target2Num > entryPivotNum && stopDistanceDollar > 0 ? Number(((target2Num - entryPivotNum) / stopDistanceDollar).toFixed(2)) : 0;
  const estimatedCapitalAllocated = isActionable ? recommendedShares * entryPivotNum : 0;

  return {
    ticker: setup.ticker || 'UNASSIGNED',
    entryPivot: entryPivotNum,
    stopLoss: stopLossNum,
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
    isAvailable: true,
  };
}
