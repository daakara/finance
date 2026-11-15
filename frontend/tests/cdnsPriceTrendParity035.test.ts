/**
 * ARX-QA-AUTHORITY-035: CDNS Price Trend Moving-Average Parity & Explanation Suite
 *
 * Verifies:
 * 1. Pullback State (Price above SMA50, below EMA20):
 *    - CDNS fixture: Price $323.01, EMA20 $328.26, SMA50 $320.83
 *    - Status MUST be MIXED (not FAVORABLE).
 *    - Observation MUST accurately state that price is pulling back below the 20 EMA
 *      while holding above the 50-day average.
 *    - Observation MUST NEVER claim "holding firmly above the 20 EMA" when price < EMA20!
 *    - Evidence list MUST include both Price vs 50D SMA (POSITIVE) and Price vs 20D EMA (NEGATIVE).
 * 2. Pristine Uptrend State (Price above both EMA20 and SMA50):
 *    - Status is FAVORABLE (+25 points).
 *    - Observation states holding firmly above both moving averages.
 * 3. Downtrend / Stage 4 State (Price below SMA50):
 *    - Status is UNFAVORABLE (-25 points).
 *    - Observation states price is below the 50-day average.
 */

import assert from "node:assert";
import { generateQuantitativeInsight } from "../lib/insightGenerator";

console.log("=== ARX-QA-AUTHORITY-035 CDNS Price Trend Parity Test Suite ===\n");

function createCandles(sma50Target: number, ema20Target: number, lastClose: number) {
  const candles: Array<{ time: string; open: number; high: number; low: number; close: number; volume: number }> = [];
  for (let i = 0; i < 30; i++) {
    candles.push({ time: `2025-10-${String(i + 1).padStart(2, "0")}`, open: 310, high: 320, low: 305, close: 311.2, volume: 100000 });
  }
  for (let i = 0; i < 19; i++) {
    candles.push({ time: `2025-11-${String(i + 1).padStart(2, "0")}`, open: 330, high: 340, low: 325, close: 334.8, volume: 100000 });
  }
  candles.push({ time: `2025-11-20`, open: 325, high: 328, low: 320, close: lastClose, volume: 100000 });
  return candles;
}

// ── Test 1: CDNS Pullback Reproduction (Price $323.01 < EMA20 $330.49, > SMA50 $320.40) ────
console.log("Test 1: CDNS Pullback State (Price above SMA50, below EMA20)...");
{
  const candles = createCandles(320.83, 328.26, 323.01);
  const insight = generateQuantitativeInsight(
    "CDNS",
    "Cadence Design Systems",
    323.01,
    0.5,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    candles
  );

  const trend = insight.terminalState?.domains?.find((d) => d.domainId === "trend");
  assert.ok(trend, "Trend domain must exist");
  assert.strictEqual(trend.status, "MIXED", `Expected status MIXED for pullback below 20 EMA, got ${trend.status}`);
  assert.strictEqual(trend.pointImpact, 10, `Expected pointImpact 10 for moderated pullback, got ${trend.pointImpact}`);
  assert.ok(
    trend.observation.includes("pulling back below the 20 EMA"),
    `Observation must state pulling back below 20 EMA. Got: "${trend.observation}"`
  );
  assert.ok(
    trend.observation.includes("holding above the 50-day average"),
    `Observation must state holding above 50-day average. Got: "${trend.observation}"`
  );
  assert.ok(
    !trend.observation.includes("holding firmly above the 20 EMA"),
    "Observation must NEVER claim holding firmly above 20 EMA when price is below it!"
  );

  // Check dual evidence items
  const smaEv = trend.evidence?.find((e) => e.metricName === "Price vs 50-Day SMA");
  const emaEv = trend.evidence?.find((e) => e.metricName === "Price vs 20-Day EMA");
  assert.ok(smaEv, "Price vs 50-Day SMA evidence must exist");
  assert.strictEqual(smaEv?.status, "POSITIVE", "Price vs 50-Day SMA must be POSITIVE");
  assert.ok(emaEv, "Price vs 20-Day EMA evidence must exist");
  assert.strictEqual(emaEv?.status, "NEGATIVE", "Price vs 20-Day EMA must be NEGATIVE when price < EMA20");
  console.log("  [PASS] CDNS pullback correctly classified as MIXED (+10) with truthful observation and dual evidence.");
}

// ── Test 2: Pristine Uptrend (Price above both EMA20 and SMA50) ──────────────
console.log("\nTest 2: Pristine Uptrend (Price above both MAs)...");
{
  const candles = Array.from({ length: 50 }, (_, i) => ({
    time: `2025-10-${String(i + 1).padStart(2, "0")}`,
    open: 300,
    high: 310,
    low: 295,
    close: 300 + i, // rising trend
    volume: 100000,
  }));
  const lastPrice = candles[49].close; // 349

  const insight = generateQuantitativeInsight(
    "AAPL",
    "Apple Inc",
    lastPrice,
    1.2,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    candles
  );

  const trend = insight.terminalState?.domains?.find((d) => d.domainId === "trend");
  assert.ok(trend, "Trend domain must exist");
  assert.strictEqual(trend.status, "FAVORABLE");
  assert.strictEqual(trend.pointImpact, 25);
  assert.ok(trend.observation.includes("holding firmly above"), `Got: "${trend.observation}"`);
  console.log("  [PASS] Pristine uptrend correctly classified as FAVORABLE (+25).");
}

// ── Test 3: Downtrend / Stage 4 (Price below SMA50) ──────────────────────────
console.log("\nTest 3: Downtrend / Stage 4 (Price below 50-day SMA)...");
{
  const candles = Array.from({ length: 50 }, (_, i) => ({
    time: `2025-10-${String(i + 1).padStart(2, "0")}`,
    open: 350,
    high: 355,
    low: 345,
    close: 350,
    volume: 100000,
  }));
  candles[49].close = 310; // breakdown well below 350 SMA50

  const insight = generateQuantitativeInsight(
    "BREAK",
    "Breakdown Corp",
    310.0,
    -3.5,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    candles
  );

  const trend = insight.terminalState?.domains?.find((d) => d.domainId === "trend");
  assert.ok(trend, "Trend domain must exist");
  assert.strictEqual(trend.status, "UNFAVORABLE");
  assert.strictEqual(trend.pointImpact, -25);
  assert.ok(trend.observation.includes("below the 50-day average"), `Got: "${trend.observation}"`);
  console.log("  [PASS] Downtrend correctly classified as UNFAVORABLE (-25).");
}

console.log("\n=== ALL CDNS PRICE TREND PARITY TESTS PASSED ===");
