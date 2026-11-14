/**
 * ARX-EVIDENCE-INTEGRITY-032 Formal Acceptance Suite
 * Tracker: Issue #198 (TD-01 Analysis Cache Authority & Freshness)
 *
 * Verifies:
 * 1. String & Object Live Freshness Resolution: Plain string data.liveFreshness ("STALE", "LIVE")
 *    is authoritatively resolved and never dropped as undefined.
 * 2. Pre-Flight Price Staleness Enforcement: STALE liveFreshnessStatus strictly triggers fail-closed
 *    isCleared === false and STALE PRICE OBSERVATION blocker.
 * 3. In-Memory Cache TTL Bounds:
 *    - Global 15-minute upper bound (MAX_ANALYSIS_CACHE_AGE_MS) evicts stale memory cache entries.
 *    - 5-minute observation freshness (MAX_LIVE_OBSERVATION_AGE_MS) evicts stale live/realtime feeds.
 *    - Explicitly STALE, EXPIRED, or UNAVAILABLE feeds are never served as fresh cache hits.
 * 4. Transparent Observation Presentation:
 *    - Observation times reflect true source timestamps (Observed / Close / Prior Session Close).
 *    - Unrecorded timestamps explicitly display "Observation time unrecorded" rather than deceptive
 *      current wall-clock "(Rendered)".
 */

import assert from "node:assert";
import {
  isAnalysisCacheEntryFresh,
  formatAnalysisObservationTime,
  resolvePreflightLiveFreshness,
  MAX_ANALYSIS_CACHE_AGE_MS,
  MAX_LIVE_OBSERVATION_AGE_MS,
  CachedAnalyticsEntry,
} from "../lib/analysisCache";
import { evaluatePreFlightChecks, PreFlightChecklistInputs } from "../lib/preflightEvaluator";
import type { AnalyticsResponse } from "../lib/api";

console.log("=== ARX-EVIDENCE-INTEGRITY-032 Acceptance Suite (Issue #198 / TD-01) ===\n");

// -----------------------------------------------------------------------------
// Test 1: Live Freshness Status Extraction (Plain String vs Object)
// -----------------------------------------------------------------------------
console.log("1. Verifying resolvePreflightLiveFreshness handles plain string & object formats...");
{
  // Case 1a: Plain string on data.liveFreshness
  const dataStringStale: Partial<AnalyticsResponse> = {
    symbol: "NVDA",
    liveFreshness: "STALE",
  };
  assert.strictEqual(
    resolvePreflightLiveFreshness(dataStringStale as AnalyticsResponse),
    "STALE",
    "Plain string 'STALE' on data.liveFreshness must be resolved"
  );

  // Case 1b: Plain string on data.liveFreshness (lowercase normalized)
  const dataStringLive: Partial<AnalyticsResponse> = {
    symbol: "NVDA",
    liveFreshness: "live",
  };
  assert.strictEqual(
    resolvePreflightLiveFreshness(dataStringLive as AnalyticsResponse),
    "LIVE",
    "Plain string 'live' must normalize to uppercase 'LIVE'"
  );

  // Case 1c: Object wrapper on data.liveFreshness
  const dataObjectWrapped: Partial<AnalyticsResponse> = {
    symbol: "NVDA",
    liveFreshness: { status: "EXPIRED" } as any,
  };
  assert.strictEqual(
    resolvePreflightLiveFreshness(dataObjectWrapped as AnalyticsResponse),
    "EXPIRED",
    "Object { status: 'EXPIRED' } on data.liveFreshness must be resolved"
  );

  // Case 1d: Nested in marketPriceState as string
  const dataMpsString: Partial<AnalyticsResponse> = {
    symbol: "NVDA",
    marketPriceState: { liveFreshness: "REALTIME" } as any,
  };
  assert.strictEqual(
    resolvePreflightLiveFreshness(dataMpsString as AnalyticsResponse),
    "REALTIME",
    "marketPriceState.liveFreshness string must be resolved"
  );

  // Case 1e: Nested in marketPriceState as object
  const dataMpsObject: Partial<AnalyticsResponse> = {
    symbol: "NVDA",
    marketPriceState: { liveFreshness: { status: "STALE" } } as any,
  };
  assert.strictEqual(
    resolvePreflightLiveFreshness(dataMpsObject as AnalyticsResponse),
    "STALE",
    "marketPriceState.liveFreshness object must be resolved"
  );

  // Case 1f: Missing or null
  assert.strictEqual(resolvePreflightLiveFreshness(null), undefined);
  assert.strictEqual(resolvePreflightLiveFreshness({} as any), undefined);

  console.log("   ✓ Confirmed: Plain string and object liveFreshness variants are robustly extracted");
}

// -----------------------------------------------------------------------------
// Test 2: Pre-Flight Fail-Closed Authority Integration
// -----------------------------------------------------------------------------
console.log("2. Verifying resolved STALE freshness strictly fails closed in preflightEvaluator...");
{
  const mockAnalyticsData: Partial<AnalyticsResponse> = {
    symbol: "MSFT",
    currentPrice: 410.0,
    liveFreshness: "STALE", // Plain string from backend API
    liveObservedAt: Date.now() - 3600000, // 1 hour ago
    _dataSource: "live",
    optimalExecution: {
      stop_loss: 395.0,
      take_profit_1: 445.0,
      risk_reward_ratio: 2.33,
      setup_pattern: "Minervini VCP",
      optimal_entry_min: 405.0,
      optimal_entry_max: 412.0,
      breakout_pivot: 410.0,
    } as any,
  };

  const resolvedFreshness = resolvePreflightLiveFreshness(mockAnalyticsData as AnalyticsResponse);
  assert.strictEqual(resolvedFreshness, "STALE");

  const preflightInputs: PreFlightChecklistInputs = {
    symbol: "MSFT",
    currentPrice: 410.0,
    stopLoss: 395.0,
    takeProfit1: 445.0,
    riskRewardRatio: 2.33,
    setupPattern: "Minervini VCP",
    isActionable: true,
    canSizeTrade: true,
    optimalEntryMin: 405.0,
    optimalEntryMax: 412.0,
    breakoutPivot: 410.0,
    isStage4: false,
    catalystEvidence: {
      state: "PASS",
      daysToEarnings: 20.0,
      eventDate: "2026-12-01",
      confirmationStatus: "CONFIRMED",
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE",
      riskTier: "NORMAL_RUNWAY",
    },
    macroEvidence: {
      vix: 14.2,
      observationTime: new Date().toISOString(),
      marketSession: "REGULAR",
      freshnessStatus: "LIVE",
    },
    dataSource: "live",
    liveSource: "ALPACA_IEX",
    liveFreshnessStatus: resolvedFreshness, // Passed directly from resolved helper
  };

  const result = evaluatePreFlightChecks(preflightInputs);
  assert.strictEqual(result.isCleared, false, "Stale price feed MUST fail closed");
  assert.ok(
    result.allBlockers.includes("STALE PRICE OBSERVATION"),
    "Blockers list must contain STALE PRICE OBSERVATION"
  );
  console.log("   ✓ Confirmed: Resolved STALE feed strictly fails closed in preflightEvaluator");
}

// -----------------------------------------------------------------------------
// Test 3: In-Memory Cache TTL Upper Bound & Invalidation
// -----------------------------------------------------------------------------
console.log("3. Verifying in-memory cache TTL bounds & invalidation rules...");
{
  const now = 1700000000000;

  // 3a. Fresh live entry within bounds (cached 2 min ago, observed 2 min ago)
  const freshEntry: CachedAnalyticsEntry = {
    cachedAt: now - 2 * 60 * 1000,
    data: {
      symbol: "AAPL",
      liveFreshness: "LIVE",
      liveObservedAt: now - 2 * 60 * 1000,
      _dataSource: "live",
    } as any,
  };
  assert.strictEqual(isAnalysisCacheEntryFresh(freshEntry, now), true, "Fresh live entry should be valid");

  // 3b. Cache entry older than 15 minutes (MAX_ANALYSIS_CACHE_AGE_MS)
  const expiredMemoryEntry: CachedAnalyticsEntry = {
    cachedAt: now - (MAX_ANALYSIS_CACHE_AGE_MS + 1000),
    data: {
      symbol: "AAPL",
      quoteStatus: "HISTORICAL_EOD",
      _dataSource: "historical",
    } as any,
  };
  assert.strictEqual(
    isAnalysisCacheEntryFresh(expiredMemoryEntry, now),
    false,
    "Cache entry exceeding 15 minutes must be evicted as stale"
  );

  // 3c. Live feed observed more than 5 minutes ago (MAX_LIVE_OBSERVATION_AGE_MS)
  const staleObservedEntry: CachedAnalyticsEntry = {
    cachedAt: now - 1 * 60 * 1000, // Cached 1 min ago
    data: {
      symbol: "AAPL",
      liveFreshness: "LIVE",
      liveObservedAt: now - (MAX_LIVE_OBSERVATION_AGE_MS + 5000), // Observed 5m05s ago
      _dataSource: "live",
    } as any,
  };
  assert.strictEqual(
    isAnalysisCacheEntryFresh(staleObservedEntry, now),
    false,
    "Live quote older than 5 minutes observation age must be evicted"
  );

  // 3d. Explicitly STALE feed cached
  const staleFeedEntry: CachedAnalyticsEntry = {
    cachedAt: now - 30 * 1000,
    data: {
      symbol: "AAPL",
      liveFreshness: "STALE",
      _dataSource: "live",
    } as any,
  };
  assert.strictEqual(
    isAnalysisCacheEntryFresh(staleFeedEntry, now),
    false,
    "Explicitly STALE feed must never be served as fresh cache hit"
  );

  // 3e. Explicitly EXPIRED feed cached
  const expiredFeedEntry: CachedAnalyticsEntry = {
    cachedAt: now - 30 * 1000,
    data: {
      symbol: "AAPL",
      liveFreshness: "EXPIRED",
      _dataSource: "live",
    } as any,
  };
  assert.strictEqual(
    isAnalysisCacheEntryFresh(expiredFeedEntry, now),
    false,
    "Explicitly EXPIRED feed must never be served as fresh cache hit"
  );

  // 3f. Null / undefined handling
  assert.strictEqual(isAnalysisCacheEntryFresh(null, now), false);
  assert.strictEqual(isAnalysisCacheEntryFresh(undefined, now), false);

  console.log("   ✓ Confirmed: 15-min global TTL, 5-min live TTL, and explicit stale feed eviction verified");
}

// -----------------------------------------------------------------------------
// Test 4: Transparent Observation Timestamp Formatting
// -----------------------------------------------------------------------------
console.log("4. Verifying observation timestamp formatting avoids deceptive current rendered clock...");
{
  // 4a. Genuine observation timestamp formatted
  const liveObsTime = "2026-11-14T14:30:00Z";
  const liveData: Partial<AnalyticsResponse> = {
    symbol: "GOOGL",
    liveObservedAt: new Date(liveObsTime).getTime(),
  };
  const liveFormatted = formatAnalysisObservationTime(liveData as AnalyticsResponse);
  assert.ok(liveFormatted.endsWith("(Observed)"), `Must end with (Observed): ${liveFormatted}`);

  // 4b. Analysis reference date (e.g. historical EOD)
  const eodData: Partial<AnalyticsResponse> = {
    symbol: "GOOGL",
    analysisReferenceDate: "2026-11-13",
  };
  const eodFormatted = formatAnalysisObservationTime(eodData as AnalyticsResponse);
  assert.strictEqual(eodFormatted, "2026-11-13 (Close)");

  // 4c. Completed session fallback
  const completedData: Partial<AnalyticsResponse> = {
    symbol: "GOOGL",
    quoteStatus: "COMPLETED_SESSION",
  };
  const completedFormatted = formatAnalysisObservationTime(completedData as AnalyticsResponse);
  assert.strictEqual(completedFormatted, "Prior Session Close");

  // 4d. Unrecorded observation time (NEVER current wall-clock system time + Rendered)
  const unrecordedData: Partial<AnalyticsResponse> = {
    symbol: "GOOGL",
  };
  const unrecordedFormatted = formatAnalysisObservationTime(unrecordedData as AnalyticsResponse);
  assert.strictEqual(unrecordedFormatted, "Observation time unrecorded");
  assert.ok(!unrecordedFormatted.includes("(Rendered)"), "Must never output deceptive '(Rendered)' clock time");

  console.log("   ✓ Confirmed: Timestamp display communicates truth of source observation");
}

console.log("\n=== ALL ARX-EVIDENCE-INTEGRITY-032 (TD-01) ACCEPTANCE TESTS PASSED ===");
