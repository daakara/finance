/**
 * Performance Metrics & Eligibility Engine (Option A Canonical)
 *
 * Implements strict trade eligibility and genuine empirical metric calculations:
 * 1. Strict CLOSED status enforcement (OPEN records with or without exit price are strictly excluded).
 * 2. Requires sufficient valid realized outcome data (finite entry price, exit price, shares, and P&L).
 * 3. Preserves valid explicitly recorded zero outcomes (pnl === 0 as scratch trades).
 * 4. Zero silent inference of net P&L from price differences.
 * 5. Cumulative realized P&L ordered strictly by verified closing timestamps (exitDate / closedAt).
 *    If any eligible trade lacks a closing timestamp, trajectory is reported as unavailable.
 * 6. Zero fallback symbols (no "ASSET") or dates (no "Recent").
 * 7. Discloses sample counts and dataset coverage limits.
 */

import { JournalTradeRecord } from "./api";

export interface EligibleLiveTrade {
  id: string;
  ticker: string | null;
  setupName: string | null;
  entryPrice: number;
  exitPrice: number;
  shares: number;
  pnl: number;
  rAchieved: number | null;
  followedRules: boolean | null;
  entryDate: string | null;
  exitDate: string | null;
  outcome: 'WIN' | 'LOSS' | 'SCRATCH';
}

export interface RealizedPerformanceMetrics {
  totalCompletedTrades: number;
  totalRealizedPnL: number;
  wins: number;
  losses: number;
  scratches: number;
  winRatePct: string | null;
  profitFactor: string | null;
  avgR: string | null;
  eligibleRTradesCount: number;
  coverageNotice: string;
}

export interface ChronologicalTrajectoryPoint {
  tradeIndex: number;
  id: string;
  ticker: string | null;
  realizedPnL: number;
  cumulativePnL: number;
  exitDate: string;
}

export interface ChronologicalTrajectoryResult {
  isAvailable: boolean;
  unavailableReason?: string;
  points: ChronologicalTrajectoryPoint[];
  totalCumulativePnL?: number;
}

export interface SetupPatternGroup {
  name: string;
  count: number;
  wins: number;
  winRatePct: string;
  totalPnL: number;
}

/**
 * Filter trades that meet the strict completed-trade contract.
 *
 * Requirements:
 * - status === 'CLOSED' strictly (OPEN with or without exit price is excluded).
 * - entryPrice > 0, exitPrice > 0, shares > 0 (all finite).
 * - Explicit realized outcome: pnlRaw or pnl string must be provided and finite.
 *   Does NOT infer net P&L from exit - entry price difference.
 * - Explicit zero P&L (pnl === 0) remains eligible as SCRATCH.
 */
export function filterEligibleLiveTrades(trades: JournalTradeRecord[]): EligibleLiveTrade[] {
  if (!Array.isArray(trades)) return [];

  return trades
    .filter((t): t is JournalTradeRecord => {
      // 1. Strict CLOSED lifecycle status
      if (!t || typeof t.status !== 'string' || t.status.trim().toUpperCase() !== 'CLOSED') {
        return false;
      }
      // 2. Valid execution levels
      if (typeof t.entryPrice !== 'number' || !Number.isFinite(t.entryPrice) || t.entryPrice <= 0) {
        return false;
      }
      if (typeof t.exitPrice !== 'number' || !Number.isFinite(t.exitPrice) || t.exitPrice <= 0) {
        return false;
      }
      if (typeof t.shares !== 'number' || !Number.isFinite(t.shares) || t.shares <= 0) {
        return false;
      }
      return true;
    })
    .map((t, idx) => {
      // 3. Explicit realized outcome (no silent inference from price delta)
      let pnl: number | null = null;
      if (t.pnlRaw !== undefined && t.pnlRaw !== null && typeof t.pnlRaw === 'number' && Number.isFinite(t.pnlRaw)) {
        pnl = t.pnlRaw;
      } else if (typeof t.pnl === 'number' && Number.isFinite(t.pnl)) {
        pnl = t.pnl;
      } else if (typeof t.pnl === 'string' && t.pnl.trim() !== '') {
        const clean = t.pnl.replace(/[^0-9.-]/g, '');
        const parsed = parseFloat(clean);
        if (Number.isFinite(parsed)) pnl = parsed;
      }

      // If P&L is missing or non-finite, this record cannot contribute to realized outcome metrics
      if (pnl === null || !Number.isFinite(pnl)) {
        return null;
      }

      // 4. Preserve missing R and compliance evidence
      const rRaw = t.rAchieved !== undefined && t.rAchieved !== null ? Number(t.rAchieved) : null;
      const rAchieved = rRaw !== null && Number.isFinite(rRaw) ? rRaw : null;

      const followedRules = typeof t.followedRules === 'boolean' ? t.followedRules : null;

      // 5. Categorize outcome cleanly (valid zero outcome is preserved as SCRATCH)
      const outcome: 'WIN' | 'LOSS' | 'SCRATCH' = pnl > 0 ? 'WIN' : pnl < 0 ? 'LOSS' : 'SCRATCH';

      // 6. Setup name preservation (no keyword guessing)
      const rawSetup = (t.setupName || t.setup || '').trim();
      const setupName = rawSetup.length > 0 ? rawSetup : null;

      // 7. Identity preservation (no "ASSET" or "Recent" fallbacks)
      const rawSym = (t.ticker || t.symbol || '').trim().toUpperCase();
      const ticker = rawSym.length > 0 ? rawSym : null;

      const rawEntryDate = (t.entryDate || t.date || '').trim();
      const entryDate = rawEntryDate.length > 0 ? rawEntryDate : null;

      const rawExitDate = (t.exitDate || t.closedAt || '').trim();
      const exitDate = rawExitDate.length > 0 ? rawExitDate : null;

      return {
        id: String(t.id || idx + 1),
        ticker,
        setupName,
        entryPrice: t.entryPrice,
        exitPrice: t.exitPrice!,
        shares: t.shares,
        pnl,
        rAchieved,
        followedRules,
        entryDate,
        exitDate,
        outcome,
      };
    })
    .filter((t): t is EligibleLiveTrade => t !== null);
}

/**
 * Compute genuine empirical realized metrics from eligible trades.
 */
export function computeRealizedMetrics(
  eligibleTrades: EligibleLiveTrade[],
  totalFetched: number
): RealizedPerformanceMetrics {
  const total = eligibleTrades.length;
  const totalRealizedPnL = eligibleTrades.reduce((acc, t) => acc + t.pnl, 0);

  const wins = eligibleTrades.filter((t) => t.outcome === 'WIN').length;
  const losses = eligibleTrades.filter((t) => t.outcome === 'LOSS').length;
  const scratches = eligibleTrades.filter((t) => t.outcome === 'SCRATCH').length;

  const winRatePct = total > 0 ? ((wins / total) * 100).toFixed(1) : null;

  const grossWins = eligibleTrades.filter((t) => t.pnl > 0).reduce((acc, t) => acc + t.pnl, 0);
  const grossLosses = Math.abs(eligibleTrades.filter((t) => t.pnl < 0).reduce((acc, t) => acc + t.pnl, 0));
  const profitFactor = grossLosses > 0 ? (grossWins / grossLosses).toFixed(2) : (grossWins > 0 ? "∞" : null);

  // Exclude missing R from the average-R denominator
  const validRTrades = eligibleTrades.filter((t) => t.rAchieved !== null);
  const avgR = validRTrades.length > 0
    ? (validRTrades.reduce((acc, t) => acc + t.rAchieved!, 0) / validRTrades.length).toFixed(2)
    : null;

  const coverageNotice = totalFetched >= 200
    ? `Query limit reached (${totalFetched} records fetched; ${total} eligible closed trades). Dataset coverage may be partial.`
    : `Showing all ${totalFetched} fetched record(s) (${total} eligible closed trades).`;

  return {
    totalCompletedTrades: total,
    totalRealizedPnL,
    wins,
    losses,
    scratches,
    winRatePct,
    profitFactor,
    avgR,
    eligibleRTradesCount: validRTrades.length,
    coverageNotice,
  };
}

/**
 * Build chronological cumulative realized P&L trajectory in ascending realized-event order.
 *
 * Invariants:
 * - Ordered strictly by closing timestamp (exitDate).
 * - Deterministic tie-breaker: id ascending.
 * - Does NOT substitute entryDate or createdAt for closing time.
 * - If any eligible trade lacks a valid closing timestamp, trajectory is returned as unavailable.
 */
export function computeChronologicalTrajectory(
  eligibleTrades: EligibleLiveTrade[]
): ChronologicalTrajectoryResult {
  if (eligibleTrades.length === 0) {
    return {
      isAvailable: false,
      unavailableReason: "No eligible completed trades recorded.",
      points: [],
    };
  }

  // Check if every trade has a valid closing timestamp
  const missingExitDateCount = eligibleTrades.filter((t) => !t.exitDate || isNaN(Date.parse(t.exitDate))).length;
  if (missingExitDateCount > 0) {
    return {
      isAvailable: false,
      unavailableReason: `${missingExitDateCount} of ${eligibleTrades.length} trade(s) lack a verified closing timestamp (exitDate). Chronological trajectory requires closing timestamps to order realized events accurately without substituting entry or creation times.`,
      points: [],
    };
  }

  // Sort ascending by closing timestamp, with id tie-breaker
  const sorted = [...eligibleTrades].sort((a, b) => {
    const timeA = new Date(a.exitDate!).getTime();
    const timeB = new Date(b.exitDate!).getTime();
    if (timeA !== timeB) return timeA - timeB;
    return a.id.localeCompare(b.id, undefined, { numeric: true });
  });

  let running = 0;
  const points: ChronologicalTrajectoryPoint[] = sorted.map((t, idx) => {
    running += t.pnl;
    return {
      tradeIndex: idx + 1,
      id: t.id,
      ticker: t.ticker,
      realizedPnL: t.pnl,
      cumulativePnL: running,
      exitDate: t.exitDate!,
    };
  });

  return {
    isAvailable: true,
    points,
    totalCumulativePnL: running,
  };
}

/**
 * Group eligible trades by their raw recorded setup pattern name.
 */
export function groupSetupsByPattern(eligibleTrades: EligibleLiveTrade[]): SetupPatternGroup[] {
  const setupMap = new Map<string, { count: number; wins: number; totalPnL: number }>();

  eligibleTrades.forEach((t) => {
    const name = t.setupName || 'Unspecified Setup';
    const cur = setupMap.get(name) || { count: 0, wins: 0, totalPnL: 0 };
    cur.count += 1;
    if (t.outcome === 'WIN') cur.wins += 1;
    cur.totalPnL += t.pnl;
    setupMap.set(name, cur);
  });

  return Array.from(setupMap.entries()).map(([name, data]) => ({
    name,
    count: data.count,
    wins: data.wins,
    winRatePct: ((data.wins / data.count) * 100).toFixed(1),
    totalPnL: data.totalPnL,
  }));
}
