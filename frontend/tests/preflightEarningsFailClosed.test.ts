import assert from "node:assert";
import { evaluatePreFlightChecks, PreFlightChecklistInputs } from "../lib/preflightEvaluator";

console.log("=====================================================================");
console.log("ARX TERMINAL — PREFLIGHT CHECK #4 FAIL-CLOSED & ESTIMATE GATE SUITE");
console.log("Directive: DEF-POL-CHK4-001");
console.log("=====================================================================\n");

const BASE_PASS_INPUTS: PreFlightChecklistInputs = {
  symbol: "NVDA",
  currentPrice: 145.0,
  stopLoss: 140.0,
  takeProfit1: 155.0,
  riskRewardRatio: 2.0,
  optimalEntryMin: 142.0,
  optimalEntryMax: 146.0,
  breakoutPivot: 144.0,
  isActionable: true,
  canSizeTrade: true,
  vix: 18.5,
  vixObservationTime: new Date(Date.now() - 60000).toISOString(),
  quoteStatus: "LIVE_OBSERVATION",
  dataSource: "live",
  liveSource: "ALPACA_IEX",
  liveFreshnessStatus: "LIVE",
  observationTime: new Date(Date.now() - 5000).toISOString(),
  assetClass: "EQUITY",
  securityType: "COMMON_STOCK",
  setupPattern: "Cup and Handle",
};

// ── Test 1: Confirmed Events Beyond 7-Day Window Pass ───────────────────────
console.log("Test 1: Confirmed earnings beyond 7-day window evaluates to PASS...");
{
  const res = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: 14.0,
      confirmationStatus: "CONFIRMED",
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE",
      riskTier: "NORMAL_RUNWAY",
      sourceName: "NVIDIA Investor Relations",
      eventDate: "2026-12-12",
    },
  });

  const chk4 = res.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk4.status, "PASS");
  assert.strictEqual(chk4.outcomeCategory, "PASS");
  assert.strictEqual(chk4.passed, true);
  assert.ok(chk4.summary.includes("Runway > 7d"));
  assert.ok(chk4.detail.includes("Clean runway verified"));
  assert.strictEqual(res.isCleared, true, "All 5 checks must pass when catalyst is confirmed");
  assert.strictEqual(res.mandatoryChecksVerified, true);
  assert.strictEqual(res.passedCount, 5);
  console.log("  ✓ Test 1 Passed: Confirmed event with clean runway produces Check #4 PASS.\n");
}

// ── Test 2: Unconfirmed Distant Estimates Fail Closed to UNAVAILABLE ────────
console.log("Test 2: Unconfirmed distant estimates (>7d) fail closed to UNAVAILABLE (DEF-POL-CHK4-001)...");
{
  // AAPL live scenario: 61.27d estimate
  const resAapl = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    symbol: "AAPL",
    catalystEvidence: {
      daysToEarnings: 61.27,
      confirmationStatus: "ESTIMATED",
      coverageStatus: "ESTIMATED_UPCOMING",
      freshnessStatus: "LIVE",
      riskTier: "NORMAL_RUNWAY",
      sourceName: "Yahoo Finance (Aggregator)",
      eventDate: "2027-01-28",
    },
  });

  const chk4Aapl = resAapl.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk4Aapl.status, "UNAVAILABLE");
  assert.strictEqual(chk4Aapl.outcomeCategory, "UNAVAILABLE");
  assert.strictEqual(chk4Aapl.passed, false, "Estimated distant date CANNOT pass Check #4");
  assert.strictEqual(chk4Aapl.summary, "Unconfirmed Earnings Estimate");
  assert.ok(chk4Aapl.detail.includes("unverified by primary source"));
  assert.ok(chk4Aapl.detail.includes("Cannot certify forward runway"));
  assert.strictEqual(resAapl.isCleared, false, "Execution clearance MUST be blocked when catalyst is UNAVAILABLE");
  assert.strictEqual(resAapl.mandatoryChecksVerified, false);
  assert.strictEqual(resAapl.passedCount, 4);
  assert.strictEqual(resAapl.totalCount, 5, "Denominator must remain fixed at 5");
  assert.strictEqual(resAapl.unavailableCount, 1);
  assert.strictEqual(resAapl.activeBlocker, "UNCONFIRMED EARNINGS ESTIMATE");

  // Generic 14d estimate
  const res14d = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: 14.0,
      confirmationStatus: "ESTIMATED",
      coverageStatus: "ESTIMATED_UPCOMING",
      freshnessStatus: "LIVE",
    },
  });
  const chk4_14d = res14d.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk4_14d.status, "UNAVAILABLE");
  assert.strictEqual(chk4_14d.passed, false);
  assert.strictEqual(res14d.isCleared, false);
  console.log("  ✓ Test 2 Passed: Unconfirmed distant estimates return UNAVAILABLE and block clearance.\n");
}

// ── Test 3: Estimated Imminent Events Retain Hazard Warning (WARN) ──────────
console.log("Test 3: Estimated imminent events (<=7d) retain non-clearing hazard warning (WARN)...");
{
  const res = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: 3.5,
      confirmationStatus: "ESTIMATED",
      coverageStatus: "ESTIMATED_UPCOMING",
      freshnessStatus: "LIVE",
      eventDate: "2026-12-01",
    },
  });

  const chk4 = res.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk4.status, "HAZARD");
  assert.strictEqual(chk4.outcomeCategory, "FAIL");
  assert.strictEqual(chk4.passed, false);
  assert.strictEqual(chk4.summary, "Imminent Earnings Event");
  assert.ok(chk4.detail.includes("within 7-day risk window"));
  assert.strictEqual(res.isCleared, false);
  assert.strictEqual(res.activeBlocker, "IMMINENT EARNINGS HAZARD (<= 7 DAYS)");
  console.log("  ✓ Test 3 Passed: Imminent estimates retain non-clearing hazard warning.\n");
}

// ── Test 4: Missing Earnings Coverage Returns UNAVAILABLE ────────────────────
console.log("Test 4: Missing earnings coverage fails closed to UNAVAILABLE...");
{
  const res = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: null,
  });

  const chk4 = res.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk4.status, "UNAVAILABLE");
  assert.strictEqual(chk4.outcomeCategory, "UNAVAILABLE");
  assert.strictEqual(chk4.passed, false);
  assert.strictEqual(res.isCleared, false);
  assert.strictEqual(res.activeBlocker, "CATALYST COVERAGE UNAVAILABLE");
  console.log("  ✓ Test 4 Passed: Missing evidence fails closed to UNAVAILABLE.\n");
}

// ── Test 5: Stale Earnings Coverage Returns UNAVAILABLE ──────────────────────
console.log("Test 5: Stale earnings coverage fails closed to UNAVAILABLE...");
{
  const res = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: 14.0,
      confirmationStatus: "CONFIRMED",
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "STALE",
    },
  });

  const chk4 = res.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk4.status, "UNAVAILABLE");
  assert.strictEqual(chk4.outcomeCategory, "UNAVAILABLE");
  assert.strictEqual(chk4.passed, false);
  assert.strictEqual(res.isCleared, false);
  assert.strictEqual(res.activeBlocker, "STALE CATALYST COVERAGE");
  console.log("  ✓ Test 5 Passed: Stale coverage returns UNAVAILABLE.\n");
}

// ── Test 6: Elapsed Historical Events Return UNAVAILABLE ────────────────────
console.log("Test 6: Elapsed historical events (<0d) return UNAVAILABLE without permanent hazard...");
{
  const res = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: -3.5,
      coverageStatus: "ELAPSED_RECENT",
      freshnessStatus: "RECENT",
      eventDate: "2026-11-24",
    },
  });

  const chk4 = res.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk4.status, "UNAVAILABLE");
  assert.strictEqual(chk4.outcomeCategory, "UNAVAILABLE");
  assert.strictEqual(chk4.passed, false);
  assert.ok(chk4.detail.includes("Historical Event Elapsed"));
  assert.ok(chk4.detail.includes("Forward 7-day calendar unverified"));
  assert.strictEqual(res.isCleared, false);
  console.log("  ✓ Test 6 Passed: Elapsed historical events return UNAVAILABLE.\n");
}

// ── Test 7: Exact Seven-Day Boundary (7.00 vs 7.01) ─────────────────────────
console.log("Test 7: Exact seven-day boundary differentiation (7.00d vs 7.01d)...");
{
  // Boundary 7.00d confirmed -> HAZARD (WARN)
  const res7_00 = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: 7.0,
      confirmationStatus: "CONFIRMED",
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE",
    },
  });
  const chk7_00 = res7_00.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk7_00.status, "HAZARD");
  assert.strictEqual(chk7_00.passed, false);

  // Boundary 7.01d confirmed -> PASS
  const res7_01_conf = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: 7.01,
      confirmationStatus: "CONFIRMED",
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE",
    },
  });
  const chk7_01_conf = res7_01_conf.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk7_01_conf.status, "PASS");
  assert.strictEqual(chk7_01_conf.passed, true);

  // Boundary 7.01d estimated -> UNAVAILABLE
  const res7_01_est = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: 7.01,
      confirmationStatus: "ESTIMATED",
      coverageStatus: "ESTIMATED_UPCOMING",
      freshnessStatus: "LIVE",
    },
  });
  const chk7_01_est = res7_01_est.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk7_01_est.status, "UNAVAILABLE");
  assert.strictEqual(chk7_01_est.passed, false);

  console.log("  ✓ Test 7 Passed: Exact 7-day boundary differentiates 7.00 vs 7.01 confirmed and estimated.\n");
}

// ── Test 8: Documented ETF Exemptions Preserve PASS ─────────────────────────
console.log("Test 8: Documented ETF exemptions (SPY) preserve PASS...");
{
  const resSpy = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    symbol: "SPY",
    assetClass: "ETF",
    securityType: "ETF",
    catalystEvidence: {
      coverageStatus: "NOT_APPLICABLE",
      confirmationStatus: "NOT_APPLICABLE",
      daysToEarnings: null,
    },
  });

  const chk4Spy = resSpy.checks.find((c) => c.id === "CHK-4")!;
  assert.strictEqual(chk4Spy.status, "PASS");
  assert.strictEqual(chk4Spy.outcomeCategory, "PASS");
  assert.strictEqual(chk4Spy.passed, true);
  assert.strictEqual(chk4Spy.summary, "ETF: Earnings N/A");
  assert.ok(chk4Spy.detail.includes("Exchange-Traded Fund (ETF)"));
  assert.strictEqual(resSpy.isCleared, true);
  assert.strictEqual(resSpy.passedCount, 5);
  console.log("  ✓ Test 8 Passed: Documented ETF exemptions cleanly pass Check #4.\n");
}

// ── Test 9: Five-Check Accounting & No Clearance on Unavailable Evidence ───
console.log("Test 9: Five-check accounting integrity under UNAVAILABLE catalyst...");
{
  const res = evaluatePreFlightChecks({
    ...BASE_PASS_INPUTS,
    catalystEvidence: {
      daysToEarnings: 10.0,
      confirmationStatus: "ESTIMATED",
      coverageStatus: "ESTIMATED_UPCOMING",
      freshnessStatus: "LIVE",
    },
  });

  assert.strictEqual(res.checks.length, 5, "Checklist must contain exactly 5 checks");
  assert.strictEqual(res.totalCount, 5, "Total count must remain fixed at 5");
  assert.strictEqual(res.passedCount, 4, "Passed count must be 4 when Check 4 is unavailable");
  assert.strictEqual(res.convictionPct, 80, "Conviction must be 80% (4/5)");
  assert.strictEqual(res.isCleared, false, "Execution clearance must be BLOCKED");
  assert.strictEqual(res.mandatoryChecksVerified, false, "mandatoryChecksVerified must be false");
  console.log("  ✓ Test 9 Passed: Fixed 5-check accounting verified; clearance strictly blocked.\n");
}

console.log("=====================================================================");
console.log("ALL 9 DEF-POL-CHK4-001 PREFLIGHT TESTS PASSED SUCCESSFULLY!");
console.log("=====================================================================");
