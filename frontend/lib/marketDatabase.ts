/**
 * Persistent Client-Side Market Database Store
 * Eliminates synthetic/hardcoded fallback generators in favor of persisted snapshots
 * and authentic historical market data records.
 */

import { CandleData, AnalyticsResponse } from "./api";
import { SHARED_FACTOR_SCORES } from "./constants";

const DB_STORAGE_PREFIX = "finance_market_db_v1_";
const DB_INDEX_KEY = "finance_market_db_index";
export const SNAPSHOT_TTL_MS = 15 * 60 * 1000; // 15-minute strict live data freshness window

export interface PersistedMarketRecord {
  symbol: string;
  observedAt?: number; // Authentic market observation timestamp from provider
  fetchedAt?: number;  // Timestamp when network response was received
  storedAt: number;    // Timestamp when record was written to persistent storage
  lastUpdated: number; // Set strictly to observedAt (or 0 if missing); NEVER storedAt/Date.now()
  currentPrice: number;
  priceChangePct24h: number;
  dailyCandles: CandleData[];
  technicals?: any;
  factorScores: any;
  catalyst: any;
  smartMoney: any;
  isStale?: boolean;
}

/**
 * Check if a symbol's persisted snapshot is within the 15-minute freshness window.
 */
export function isSnapshotFresh(symbol: string): boolean {
  if (typeof window === "undefined") return false;
  const upper = symbol.toUpperCase().replace("-USD", "");
  try {
    const raw = localStorage.getItem(`${DB_STORAGE_PREFIX}${upper}`);
    if (!raw) return false;
    const record = JSON.parse(raw) as PersistedMarketRecord;
    const refTime = typeof record.storedAt === "number" && Number.isFinite(record.storedAt) && record.storedAt > 0
      ? record.storedAt
      : record.lastUpdated;
    return typeof refTime === "number" && (Date.now() - refTime) < SNAPSHOT_TTL_MS;
  } catch {
    return false;
  }
}

/**
 * Save a verified live market response to browser persistent storage.
 * Preserves authentic provider observation timestamps and separates observedAt from fetchedAt/storedAt.
 */
export function persistMarketSnapshot(symbol: string, data: AnalyticsResponse): void {
  if (typeof window === "undefined" || !data || !data.candles || data.candles.length === 0) {
    return;
  }
  const upper = symbol.toUpperCase().replace("-USD", "");
  const prior = getPersistedMarketSnapshot(upper, true);

  const isDaily = data.interval === "1d" || data.interval === "1y_hist" || !data.interval;
  const now = Date.now();
  const observedAt = (typeof data.observedAt === "number" && Number.isFinite(data.observedAt) && data.observedAt > 0)
    ? data.observedAt
    : (typeof data.freshness?.observedAt === "number" && Number.isFinite(data.freshness.observedAt) && data.freshness.observedAt > 0
      ? data.freshness.observedAt
      : undefined);

  const fetchedAt = (typeof data.fetchedAt === "number" && Number.isFinite(data.fetchedAt) && data.fetchedAt > 0)
    ? data.fetchedAt
    : (typeof data.freshness?.fetchedAt === "number" && Number.isFinite(data.freshness.fetchedAt) && data.freshness.fetchedAt > 0
      ? data.freshness.fetchedAt
      : now);

  const storedAt = now;

  const record: PersistedMarketRecord = {
    symbol: upper,
    observedAt,
    fetchedAt,
    storedAt,
    // Invariant: Never reset observation age to Date.now() on storage; strictly preserve observedAt
    lastUpdated: observedAt || 0,
    currentPrice: data.currentPrice,
    priceChangePct24h: data.priceChangePct24h,
    dailyCandles: isDaily ? data.candles : (prior?.dailyCandles || []),
    technicals: data.technicals || prior?.technicals,
    factorScores: data.factorScores || prior?.factorScores,
    catalyst: data.catalystForecast || prior?.catalyst,
    smartMoney: data.smartMoney || prior?.smartMoney,
  };

  try {
    localStorage.setItem(`${DB_STORAGE_PREFIX}${upper}`, JSON.stringify(record));
    const indexStr = localStorage.getItem(DB_INDEX_KEY);
    const index: string[] = indexStr ? JSON.parse(indexStr) : [];
    if (!index.includes(upper)) {
      index.push(upper);
      localStorage.setItem(DB_INDEX_KEY, JSON.stringify(index));
    }
  } catch (err) {
    console.warn("Storage quota exceeded when persisting market snapshot:", err);
  }
}

/**
 * Retrieve all persisted market snapshots across tracked assets.
 */
export function getAllPersistedMarketSnapshots(allowStale: boolean = false): Record<string, PersistedMarketRecord> {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const indexStr = localStorage.getItem(DB_INDEX_KEY);
    const index: string[] = indexStr ? JSON.parse(indexStr) : [];
    const results: Record<string, PersistedMarketRecord> = {};
    for (const sym of index) {
      const snap = getPersistedMarketSnapshot(sym, allowStale);
      if (snap) results[sym] = snap;
    }
    return results;
  } catch {
    return {};
  }
}

/**
 * Retrieve a persisted market snapshot from local storage with strict TTL enforcement.
 */
export function getPersistedMarketSnapshot(symbol: string, allowStale: boolean = false): PersistedMarketRecord | null {
  if (typeof window === "undefined") {
    return null;
  }
  const upper = symbol.toUpperCase().replace("-USD", "");
  try {
    const raw = localStorage.getItem(`${DB_STORAGE_PREFIX}${upper}`);
    if (!raw) return null;
    const record = JSON.parse(raw) as PersistedMarketRecord;
    
    // Self-healing check 1: Discard poisoned legacy fallback snapshots where un-cataloged stocks got stuck at Apple's $319.64
    if (upper !== "AAPL" && Math.abs(record.currentPrice - 319.64) < 0.01) {
      localStorage.removeItem(`${DB_STORAGE_PREFIX}${upper}`);
      return null;
    }

    // Self-healing check 2: Strict 15-minute storage TTL invalidation
    const storedTime = typeof record.storedAt === "number" && Number.isFinite(record.storedAt) && record.storedAt > 0
      ? record.storedAt
      : record.lastUpdated;
    const ageMs = Date.now() - storedTime;
    if (ageMs > SNAPSHOT_TTL_MS) {
      if (!allowStale) {
        return null; // Force live re-fetch
      }
      record.isStale = true;
    }

    // Self-healing check 3: Discard flatlined/corrupt datasets (e.g. 5 identical flat bars from upstream Yahoo glitch)
    if (record.dailyCandles && record.dailyCandles.length > 0) {
      let minL = Infinity;
      let maxH = -Infinity;
      for (const c of record.dailyCandles) {
        if (c.low < minL) minL = c.low;
        if (c.high > maxH) maxH = c.high;
      }
      if (record.dailyCandles.length < 15 || (maxH - minL) < 0.01) {
        localStorage.removeItem(`${DB_STORAGE_PREFIX}${upper}`);
        return null;
      }
    }

    return record;
  } catch {
    return null;
  }
}

/**
 * Generate authentic horizon candle slices from persisted daily history.
 */
export function slicePersistedCandles(
  candles: CandleData[],
  period: string,
  interval: string,
  basePrice: number
): CandleData[] {
  if (!candles || candles.length === 0) {
    return [];
  }

  const isIntraday = interval === "1m" || interval === "5m" || interval === "15m" || interval === "1h";
  if (isIntraday) {
    // Epistemic Invariant: Synthetic random-walk candles cannot be fabricated from daily historical stores
    return [];
  }

  // Daily / Macro Horizon Slicing from real historical bars
  const requiredPoints = period === "1mo" ? 22 : period === "6mo" ? 130 : period === "1y" ? 252 : period === "3y" ? 156 : 60;
  if (candles.length <= requiredPoints) {
    return candles;
  }
  return candles.slice(-requiredPoints);
}
