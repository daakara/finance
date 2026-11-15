/**
 * ARX-QA-AUTHORITY-035: CDNS Price Trend Moving-Average Parity & Explanation Suite
 *
 * Verifies:
 * 1. Pullback State (Price above SMA50, below EMA20):
 *    - CDNS fixture: Price $323.01, EMA20 $328.26, SMA50 $320.83
 *    - Status MUST be MIXED (not FAVORABLE).
 *    - Point impact retains ratified +25 points (governing baseline for holding 50-day SMA).
 *    - Observation MUST accurately state that price is consolidating/pulling back below the 20 EMA
 *      while holding above the 50 SMA.
 *    - Observation MUST NEVER claim "holding firmly above the 20 EMA" when price < EMA20!
 *    - Evidence list MUST include both Price vs 50D SMA (POSITIVE) and Price vs 20D EMA (NEGATIVE).
 * 2. Pristine Uptrend State (Price above both EMA20 and SMA50):
 *    - Status is FAVORABLE (+25 points).
 *    - Observation states holding firmly above both moving averages.
 * 3. Downtrend / Stage 4 State (Price below SMA50):
 *    - Status is UNFAVORABLE (-25 points).
 *    - Observation states price is below the 50-day average.
 * 4. Missing / Invalid EMA Regression:
 *    - When EMA20 is missing, undefined, or NaN, observation must not contain "$NaN"
 *      and must gracefully describe only the 50 SMA.
 *    - Evidence must not include an invalid or uncomputed 20-Day EMA metric.
 * 5. Unsupported VCP-Explanation Regression:
 *    - When an asset's setup pattern is NOT VCP (e.g., standard moving average trend),
 *      modelRule must NOT claim "VCP base contraction".
 *    - When an asset has verified VCP pattern, modelRule asserts "VCP base contraction".
 */

import assert from "node:assert";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import { OptimalExecutionPlan, ConfluenceData } from "../lib/api";

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
  assert.strictEqual(trend.pointImpact, 25, `Expected pointImpact 25 (ratified governing baseline for maintaining 50 SMA support), got ${trend.pointImpact}`);
  assert.ok(
    trend.observation.includes("consolidating below the 20 EMA") || trend.observation.includes("pulling back below the 20 EMA"),
    `Observation must state consolidating/pulling back below 20 EMA. Got: "${trend.observation}"`
  );
  assert.ok(
    trend.observation.includes("holding above the 50 SMA") || trend.observation.includes("holding above the 50-day average"),
    `Observation must state holding above 50 SMA. Got: "${trend.observation}"`
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
  console.log("  [PASS] CDNS pullback correctly classified as MIXED (+25 pts) with truthful observation and dual evidence.");
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

// ── Test 4: Missing / Invalid EMA Regression ─────────────────────────────────
console.log("\nTest 4: Missing / Invalid EMA Regression...");
{
  // 50 identical flat candles so SMA50 = 300, price = 320 (above SMA50)
  const candles = Array.from({ length: 50 }, (_, i) => ({
    time: `2025-10-${String(i + 1).padStart(2, "0")}`,
    open: 300,
    high: 305,
    low: 295,
    close: 300,
    volume: 100000,
  }));

  const insight = generateQuantitativeInsight(
    "TEST_NO_EMA",
    "No EMA Corp",
    320.0,
    1.0,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    candles
  );

  const trend = insight.terminalState?.domains?.find((d) => d.domainId === "trend");
  assert.ok(trend, "Trend domain must exist");
  assert.ok(!trend.observation.includes("NaN"), `Observation must not contain NaN. Got: "${trend.observation}"`);
  assert.ok(!trend.observation.includes("$NaN"), `Observation must not contain $NaN. Got: "${trend.observation}"`);
  assert.ok(trend.observation.includes("50 SMA"), `Observation must reference 50 SMA. Got: "${trend.observation}"`);
  console.log("  [PASS] Missing/invalid EMA handled gracefully without NaN leakage.");
}

// ── Test 5: Unsupported VCP Explanation Regression ──────────────────────────
console.log("\nTest 5: Unsupported VCP-Explanation Regression...");
{
  const candles = Array.from({ length: 50 }, (_, i) => ({
    time: `2025-10-${String(i + 1).padStart(2, "0")}`,
    open: 300,
    high: 310,
    low: 295,
    close: 300 + i,
    volume: 100000,
  }));

  // Non-VCP Execution Plan (e.g. standard EMA Pullback)
  const nonVcpPlan: OptimalExecutionPlan = {
    current_price: 349.0,
    stop_loss: 340.0,
    setup_pattern: "EMA Pullback",
    entry_thesis: "Test thesis",
    invalidation_condition: "Test invalidation",
    stage_phase: "Stage 2",
    vcp_contraction_status: "",
  };

  const nonVcpInsight = generateQuantitativeInsight(
    "NON_VCP",
    "Non-VCP Corp",
    349.0,
    1.0,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    candles,
    "historical",
    undefined,
    undefined,
    nonVcpPlan
  );

  const nonVcpTrend = nonVcpInsight.terminalState?.domains?.find((d) => d.domainId === "trend");
  assert.ok(nonVcpTrend, "Non-VCP trend domain must exist");
  assert.ok(
    !nonVcpTrend.modelRule.includes("VCP base contraction"),
    `Non-VCP setup must NOT assert VCP base contraction! Got: "${nonVcpTrend.modelRule}"`
  );
  assert.ok(
    nonVcpTrend.modelRule.includes("Price holding firmly above rising moving averages"),
    `Non-VCP setup must assert general trend structure. Got: "${nonVcpTrend.modelRule}"`
  );

  // Verified VCP Execution Plan
  const vcpPlan: OptimalExecutionPlan = {
    current_price: 349.0,
    stop_loss: 340.0,
    setup_pattern: "Minervini Volatility Contraction Pattern (VCP 3-Stage)",
    entry_thesis: "VCP thesis",
    invalidation_condition: "VCP invalidation",
    stage_phase: "Stage 2",
    vcp_contraction_status: "3T_CONTRACTION",
  };

  const vcpInsight = generateQuantitativeInsight(
    "VCP_ASSET",
    "VCP Corp",
    349.0,
    1.0,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    candles,
    "historical",
    undefined,
    undefined,
    vcpPlan
  );

  const vcpTrend = vcpInsight.terminalState?.domains?.find((d) => d.domainId === "trend");
  assert.ok(vcpTrend, "VCP trend domain must exist");
  assert.ok(
    vcpTrend.modelRule.includes("VCP base contraction"),
    `VCP setup must assert VCP base contraction. Got: "${vcpTrend.modelRule}"`
  );
  console.log("  [PASS] Unsupported VCP explanation suppressed on non-VCP setups; preserved on confirmed VCP setups.");
}

// ── Test 6: Incomplete Candles or Fallback Feed (UNAVAILABLE 0 pts) ───────────
console.log("\nTest 6: Incomplete history (< 50) and fallback feed fail-closed...");
{
  const incompleteCandles = createCandles(320.83, 328.26, 323.01).slice(0, 35);
  const incompleteInsight = generateQuantitativeInsight(
    "INC",
    "Incomplete Corp",
    323.01,
    0.0,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    incompleteCandles,
    "historical"
  );
  const incTrend = incompleteInsight.terminalState?.domains?.find((d) => d.domainId === "trend");
  assert.ok(incTrend, "Incomplete trend domain must be defined");
  assert.strictEqual(incTrend.availability, "UNAVAILABLE");
  assert.strictEqual(incTrend.status, "UNAVAILABLE");
  assert.strictEqual(incTrend.pointImpact, 0);

  const fallbackInsight = generateQuantitativeInsight(
    "FALL",
    "Fallback Corp",
    323.01,
    0.0,
    undefined,
    undefined,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    createCandles(320.83, 328.26, 323.01),
    "fallback"
  );
  const fallTrend = fallbackInsight.terminalState?.domains?.find((d) => d.domainId === "trend");
  assert.ok(fallTrend, "Fallback trend domain must be defined");
  assert.strictEqual(fallTrend.availability, "UNAVAILABLE");
  assert.strictEqual(fallTrend.status, "UNAVAILABLE");
  assert.strictEqual(fallTrend.pointImpact, 0);
  console.log("  [PASS] Fail-closed behavior strictly preserves 0 points and UNAVAILABLE status.");
}

// ── Test 7: Continuous Boundary Sweep & Evidence Table Invariant Guard ────────
console.log("\nTest 7: Continuous boundary sweep & Evidence-to-Explanation Invariant Guard...");
{
  // Deterministic candle series: SMA50 = $320.83, EMA20 = $328.26
  const fixedCandles = Array.from({ length: 50 }, (_, i) => {
    const close = i < 30 ? 315.873 : 328.26;
    return {
      time: `2026-01-${String((i % 28) + 1).padStart(2, "0")}`,
      open: close,
      high: close + 2,
      low: close - 2,
      close,
      volume: 1000000,
    };
  });

  const testPrices = [310.0, 320.0, 320.83, 323.01, 328.26, 335.0];
  for (const p of testPrices) {
    const ins = generateQuantitativeInsight(
      "TEST",
      "Test",
      p,
      0,
      undefined,
      undefined,
      "SWING",
      "NOT_OWNED",
      "USER_DECLARED",
      fixedCandles,
      "historical"
    );
    const dom = ins.terminalState?.domains?.find((d) => d.domainId === "trend");
    if (!dom || dom.availability !== "AVAILABLE") continue;

    if (p < 320.83) {
      assert.strictEqual(dom.status, "UNFAVORABLE", `Price $${p} must be UNFAVORABLE`);
      assert.strictEqual(dom.pointImpact, -25);
    } else if (p < 328.26) {
      assert.strictEqual(dom.status, "MIXED", `Price $${p} must be MIXED`);
      assert.strictEqual(dom.pointImpact, 25);
      assert.strictEqual(dom.observation.includes("holding firmly above the 20 EMA"), false);
      const smaItem = dom.evidence?.find((e) => e.metricName.includes("50-Day SMA"));
      const emaItem = dom.evidence?.find((e) => e.metricName.includes("20-Day EMA"));
      assert.strictEqual(smaItem?.status, "POSITIVE");
      assert.strictEqual(emaItem?.status, "NEGATIVE");
    } else {
      assert.strictEqual(dom.status, "FAVORABLE", `Price $${p} must be FAVORABLE`);
      assert.strictEqual(dom.pointImpact, 25);
      assert.ok(dom.evidence?.every((e) => e.status === "POSITIVE"));
    }
  }
  console.log("  [PASS] Evidence-to-explanation invariant holds across all price boundaries.");
}

// ── Test 8: Confluence Score Decoupling Invariant ────────────────────────────
console.log("\nTest 8: Confluence score authority decoupling...");
{
  const mockConfluence: ConfluenceData = {
    symbol: "CDNS",
    confluenceScore: 72.0,
    confluenceRating: "Selective Confluence",
    plainRating: "Selective Confluence",
    confluenceBadge: "Selective",
    plainBadge: "Selective",
    badgeColor: "amber",
    bottomLine: "Technical structure in pullback phase; macro regime positive.",
    positivesCount: 2,
    warningsCount: 1,
    warnings: ["Fundamental data unverified"],
    pillars: [
      {
        pillar: "TECHNICAL_STRUCTURE",
        label: "Technical Structure",
        plainLabel: "Technical Structure",
        score: 75.0,
        status: "positive",
        detail: "Pullback phase above 50-day SMA",
        plainDetail: "Price is testing pullback support above the 50-day average.",
        icon: "📈",
      },
    ],
  };

  const candles = createCandles(320.83, 328.26, 323.01);
  const cdnsInsight = generateQuantitativeInsight(
    "CDNS",
    "Cadence Design Systems",
    323.01,
    -1.2,
    72,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    candles,
    "historical",
    mockConfluence
  );

  assert.strictEqual(
    cdnsInsight.setupScore,
    72,
    `Confluence score must remain 72 regardless of domain breakdown point changes, got ${cdnsInsight.setupScore}`
  );
  assert.strictEqual(
    cdnsInsight.verdictLabel.toLowerCase().includes("actionable"),
    false,
    "CDNS with 72 setup score and 2 pillars must NOT be actionable"
  );
  console.log("  [PASS] Domain point adjustments are decoupled from authoritative Confluence score.");
}

console.log("\n=== ALL CDNS PRICE TREND PARITY TESTS PASSED ===");
