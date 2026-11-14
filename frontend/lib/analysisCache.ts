/**
 * ARX Terminal — Analysis Cache Authority & Freshness Module
 * Tracker: Issue #198 (TD-01)
 *
 * Epistemic & Authority Invariants:
 * 1. Analysis responses stored in client memory must have a deterministic TTL.
 * 2. Absolute Maximum Memory TTL: 15 minutes (MAX_ANALYSIS_CACHE_AGE_MS). After 15 minutes,
 *    any cached response is evicted, even for historical intervals, preventing stale macro,
 *    corporate earnings, and technical ratings from leaking into decision views.
 * 3. Live Observation Max TTL: 5 minutes (MAX_LIVE_OBSERVATION_AGE_MS). If a response is
 *    marked LIVE or REALTIME, it is evicted if its source observation age exceeds 5 minutes.
 * 4. Stale / Expired / Unavailable Feeds: A response with liveFreshness === "STALE", "EXPIRED",
 *    or "UNAVAILABLE" must never be served as a valid cache hit and must be evicted immediately.
 * 5. Observation Time Transparency: Observation times reflect actual source timestamps.
 *    If unrecorded, the UI displays "Observation time unrecorded" rather than deceptive current
 *    wall-clock "(Rendered)".
 * 6. Live Freshness Extraction: Correctly extracts plain string or object-wrapped status
 *    to prevent bypassing Pre-Flight price staleness checks.
 */

import type { AnalyticsResponse } from "./api";

export const MAX_ANALYSIS_CACHE_AGE_MS = 15 * 60 * 1000; // 15 minutes
export const MAX_LIVE_OBSERVATION_AGE_MS = 5 * 60 * 1000; // 5 minutes

export interface CachedAnalyticsEntry {
  data: AnalyticsResponse;
  cachedAt: number; // Date.now() when stored
}

/**
 * Authoritatively resolves the live freshness status string from an AnalyticsResponse.
 * Handles both plain string values ("LIVE", "STALE") and object structures ({ status: "LIVE" }).
 */
export function resolvePreflightLiveFreshness(
  data: AnalyticsResponse | null | undefined
): string | undefined {
  if (!data) return undefined;

  // Direct string property on data
  if (typeof data.liveFreshness === "string" && data.liveFreshness.trim()) {
    return data.liveFreshness.trim().toUpperCase();
  }

  // Nested in marketPriceState as string
  const mps = data.marketPriceState as any;
  if (typeof mps?.liveFreshness === "string" && mps.liveFreshness.trim()) {
    return mps.liveFreshness.trim().toUpperCase();
  }

  // Object wrapper with .status on data.liveFreshness
  if (typeof (data.liveFreshness as any)?.status === "string" && (data.liveFreshness as any).status.trim()) {
    return (data.liveFreshness as any).status.trim().toUpperCase();
  }

  // Object wrapper with .status on marketPriceState.liveFreshness
  if (typeof mps?.liveFreshness?.status === "string" && mps.liveFreshness.status.trim()) {
    return mps.liveFreshness.status.trim().toUpperCase();
  }

  return undefined;
}

/**
 * Validates whether a cached analytics entry remains authoritative and fresh.
 */
export function isAnalysisCacheEntryFresh(
  entry: CachedAnalyticsEntry | null | undefined,
  nowMs: number = Date.now()
): boolean {
  if (!entry || !entry.data || typeof entry.cachedAt !== "number") {
    return false;
  }

  // 1. Enforce absolute upper bound on in-memory cache age
  const cacheAgeMs = nowMs - entry.cachedAt;
  if (cacheAgeMs < 0 || cacheAgeMs > MAX_ANALYSIS_CACHE_AGE_MS) {
    return false;
  }

  const { data } = entry;
  const freshnessStr = resolvePreflightLiveFreshness(data);

  // 2. Reject explicit stale, expired, or unavailable feeds from being served as fresh cache hits
  if (freshnessStr === "STALE" || freshnessStr === "EXPIRED" || freshnessStr === "UNAVAILABLE") {
    return false;
  }

  // 3. Enforce 5-minute live observation freshness for live/realtime feeds
  const obsTimeStr = data.liveObservedAt || (data.marketPriceState as any)?.asOfTimestamp;
  if (obsTimeStr) {
    const obsTime = new Date(obsTimeStr).getTime();
    if (!isNaN(obsTime)) {
      const obsAgeMs = nowMs - obsTime;
      const isLiveFeed = freshnessStr === "LIVE" || freshnessStr === "REALTIME" || data._dataSource === "live";
      if (isLiveFeed && obsAgeMs > MAX_LIVE_OBSERVATION_AGE_MS) {
        return false;
      }
    }
  }

  return true;
}

/**
 * Formats the observation timestamp string truthfully without misleading rendered wall-clock fallbacks.
 */
export function formatAnalysisObservationTime(
  data: AnalyticsResponse | null | undefined
): string {
  if (!data) return "";

  const obsTimeStr = data.liveObservedAt || (data.marketPriceState as any)?.asOfTimestamp;
  if (obsTimeStr) {
    const d = new Date(obsTimeStr);
    if (!isNaN(d.getTime())) {
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) + " (Observed)";
    }
  }

  if (data.analysisReferenceDate) {
    return `${data.analysisReferenceDate} (Close)`;
  }

  if (data.quoteStatus === "COMPLETED_SESSION" || data._dataSource === "historical") {
    return "Prior Session Close";
  }

  return "Observation time unrecorded";
}
