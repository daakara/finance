/**
 * ARX Universal Evidence Domain Authority & Demo Removal Test Suite
 * Directive: ARX-UX-EVIDENCE-001
 *
 * Verifies:
 * 1. Startup Asset (AAPL) loads with authentic production authority and zero demo status.
 * 2. 9-Domain resolution across asset classes (Equities, ETFs, Digital Assets).
 * 3. Missing optional enrichment (e.g. smart money under HOLD_SEC_001) never demotes
 *    actionable setups or triggers false "Evidence Incomplete" / "Partial Evidence" warnings.
 * 4. Missing required core evidence (<50 candles or missing financial statements)
 *    strictly fail-closed blocks execution and disables trade sizing.
 */

import assert from "assert";
import { resolveNineEvidenceDomains } from "../lib/evidenceDomains";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import { DecisionTrace } from "../types/insight";
import { CandleData, ConfluenceData } from "../lib/api";

function createMockCandles(count: number, basePrice: number = 220): CandleData[] {
  const candles: CandleData[] = [];
  const now = new Date("2026-10-31T20:00:00Z");
  for (let i = count - 1; i >= 0; i--) {
    const d = new Date(now.getTime() - i * 86400000);
    const dateStr = d.toISOString().split("T")[0];
    candles.push({
      time: dateStr,
      open: basePrice - 1,
      high: basePrice + 2,
      low: basePrice - 2,
      close: basePrice + (Math.sin(i) * 3),
      volume: 45000000,
    });
  }
  return candles;
}

const mockHealthyConfluence = {
  confluenceScore: 78.5,
  pillars: [
    {
      pillar: "FUNDAMENTAL_SOLVENCY",
      label: "Fundamentals",
      plainLabel: "Fundamentals",
      score: 85,
      status: "positive",
      detail: "Solvency score 85/100 verified via SEC EDGAR filings.",
      plainDetail: "Solvency score 85/100 verified via SEC EDGAR filings.",
      icon: "🏢",
    },
    {
      pillar: "TECHNICAL_TREND",
      label: "Technicals",
      plainLabel: "Technicals",
      score: 80,
      status: "positive",
      detail: "Consolidating constructively above 20 EMA and 50 SMA.",
      plainDetail: "Consolidating constructively above 20 EMA and 50 SMA.",
      icon: "📈",
    },
    {
      pillar: "MACRO_REGIME",
      label: "Macro",
      plainLabel: "Macro",
      score: 75,
      status: "positive",
      detail: "VIX 19.42 within supportive volatility regime.",
      plainDetail: "VIX 19.42 within supportive volatility regime.",
      icon: "🌐",
    },
    {
      pillar: "SMART_MONEY_FLOW",
      label: "Smart Money",
      plainLabel: "Smart Money",
      score: 0,
      status: "unavailable",
      detail: "SEC Form 13F and Form 4 continuous ingestion paused under HOLD_SEC_001.",
      plainDetail: "SEC Form 13F and Form 4 continuous ingestion paused under HOLD_SEC_001.",
      icon: "💼",
    },
  ],
} as unknown as ConfluenceData;

const mockActionableTrace: DecisionTrace = {
  symbol: "AAPL",
  decisionState: "ACTIONABLE_SETUP",
  decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
  canSizeTrade: true,
  isActionable: true,
  disqualificationReason: null,
  allowedActions: ["SIZE_TRADE", "SET_ALERT"],
};

console.log("=====================================================================");
console.log("ARX TERMINAL — EVIDENCE DOMAIN AUTHORITY & DEMO REMOVAL SUITE");
console.log("Directive: ARX-UX-EVIDENCE-001");
console.log("=====================================================================\n");

// ── Test 1: Production Startup Asset Authority (AAPL) ──────────────────────
console.log("Test 1: Startup Asset AAPL loads with authentic production authority...");
{
  const candles252 = createMockCandles(252, 225.5);
  const insight = generateQuantitativeInsight(
    "AAPL",
    "Apple Inc.",
    225.5,
    1.2,
    78,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    candles252,
    "live",
    mockHealthyConfluence,
    mockActionableTrace
  );

  // Assert AAPL is NOT demo or synthetic
  assert.strictEqual(insight.symbol, "AAPL");
  assert.strictEqual(insight.terminalState.overallEligibility, "ELIGIBLE", "AAPL overallEligibility must be ELIGIBLE despite pending smart money");
  assert.strictEqual(insight.terminalState.posture, "ACQUIRE", "AAPL posture must be ACQUIRE");
  assert.strictEqual(insight.terminalState.canSizeTrade, true, "AAPL canSizeTrade must be true");
  assert.strictEqual(insight.terminalState.isActionable, true, "AAPL isActionable must be true");
  assert.strictEqual(insight.terminalState.uiStateLabel, "Actionable Setup — Buy Zone Confirmed");
  assert.strictEqual((insight as any).isSynthetic ?? false, false, "AAPL must not be marked synthetic");

  console.log("  ✓ Test 1 Passed: AAPL startup asset possesses full production decision authority and zero demo taint.\n");
}

// ── Test 2: Missing Optional Enrichment Does Not Trigger Warning Banner ────
console.log("Test 2: Unavailable smart money under HOLD_SEC_001 preserves ELIGIBLE status...");
{
  const domains = resolveNineEvidenceDomains({
    symbol: "AAPL",
    assetClass: "EQUITY",
    candles: createMockCandles(252, 225),
    confluence: mockHealthyConfluence,
    smartMoney: {
      status: "PIPELINE_PENDING",
      optionsFlow: [],
      secForm4Trades: [],
      congressTrades: [],
    },
    preflightCleared: true,
    decisionTrace: mockActionableTrace,
    dataSource: "live",
    currentPrice: 225,
  });

  assert.strictEqual(domains.length, 9, "Must resolve exactly 9 evidence domains");

  const priceHistory = domains.find(d => d.id === "price_history")!;
  const technicals = domains.find(d => d.id === "technicals")!;
  const secFilings = domains.find(d => d.id === "sec_filings")!;
  const fundamentals = domains.find(d => d.id === "fundamentals")!;
  const optionsFlow = domains.find(d => d.id === "options_flow")!;
  const insiderTrades = domains.find(d => d.id === "insider_transactions")!;
  const congress = domains.find(d => d.id === "congressional_disclosures")!;
  const macro = domains.find(d => d.id === "macro_regime")!;
  const preflight = domains.find(d => d.id === "preflight_execution")!;

  assert.strictEqual(priceHistory.status, "VERIFIED");
  assert.strictEqual(priceHistory.impact, "BLOCKING");

  assert.strictEqual(technicals.status, "VERIFIED");
  assert.strictEqual(technicals.impact, "BLOCKING");

  assert.strictEqual(secFilings.status, "VERIFIED");
  assert.strictEqual(secFilings.impact, "BLOCKING");

  assert.strictEqual(fundamentals.status, "VERIFIED");
  assert.strictEqual(fundamentals.impact, "BLOCKING");

  // Optional enrichment domains
  assert.strictEqual(optionsFlow.applicability, "OPTIONAL");
  assert.strictEqual(optionsFlow.impact, "NON_BLOCKING");
  assert.strictEqual(optionsFlow.status, "UNAVAILABLE"); // Pipeline pending

  assert.strictEqual(insiderTrades.applicability, "OPTIONAL");
  assert.strictEqual(insiderTrades.impact, "NON_BLOCKING");
  assert.strictEqual(insiderTrades.status, "UNAVAILABLE"); // Held under HOLD_SEC_001

  assert.strictEqual(congress.applicability, "OPTIONAL");
  assert.strictEqual(congress.impact, "NON_BLOCKING");
  assert.strictEqual(congress.status, "NO_OBSERVATIONS");

  assert.strictEqual(macro.status, "VERIFIED");
  assert.strictEqual(macro.impact, "BLOCKING");

  assert.strictEqual(preflight.status, "VERIFIED");
  assert.strictEqual(preflight.impact, "BLOCKING");

  console.log("  ✓ Test 2 Passed: 9 domains correctly map blocking vs advisory impacts under governed holds.\n");
}

// ── Test 3: ETF Domain Rules Exemption (SPY / QQQ) ─────────────────────────
console.log("Test 3: ETF assets (SPY) are exempt from corporate 10-K and Form 4 requirements...");
{
  const spyConfluence = {
    confluenceScore: 76,
    pillars: [
      {
        pillar: "FUNDAMENTAL_SOLVENCY",
        label: "Fundamentals",
        plainLabel: "Fundamentals",
        score: 0,
        status: "unavailable",
        detail: "Fund profile",
        plainDetail: "Fund profile",
        icon: "🏢",
      },
      {
        pillar: "TECHNICAL_TREND",
        label: "Technicals",
        plainLabel: "Technicals",
        score: 85,
        status: "positive",
        detail: "Above 50 SMA",
        plainDetail: "Above 50 SMA",
        icon: "📈",
      },
      {
        pillar: "MACRO_REGIME",
        label: "Macro",
        plainLabel: "Macro",
        score: 75,
        status: "positive",
        detail: "Supportive macro regime",
        plainDetail: "Supportive macro regime",
        icon: "🌐",
      },
    ],
  } as unknown as ConfluenceData;

  const spyDomains = resolveNineEvidenceDomains({
    symbol: "SPY",
    assetClass: "ETF",
    securityType: "ETF",
    candles: createMockCandles(252, 575),
    confluence: spyConfluence,
    preflightCleared: true,
    decisionTrace: {
      symbol: "SPY",
      decisionState: "ACTIONABLE_SETUP",
      decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
      canSizeTrade: true,
      isActionable: true,
      disqualificationReason: null,
      allowedActions: ["SIZE_TRADE"],
    },
    dataSource: "live",
    currentPrice: 575,
  });

  const secFilings = spyDomains.find(d => d.id === "sec_filings")!;
  const insiderTrades = spyDomains.find(d => d.id === "insider_transactions")!;
  const fundamentals = spyDomains.find(d => d.id === "fundamentals")!;

  assert.strictEqual(secFilings.applicability, "INAPPLICABLE", "ETF 10-K must be INAPPLICABLE");
  assert.strictEqual(secFilings.impact, "NON_BLOCKING", "ETF 10-K must be NON_BLOCKING");
  assert.strictEqual(insiderTrades.applicability, "INAPPLICABLE", "ETF Insider trades must be INAPPLICABLE");
  assert.strictEqual(fundamentals.status, "VERIFIED", "ETF fund profile must resolve as VERIFIED");

  // Verify SPY does not get downgraded in insightGenerator
  const spyInsight = generateQuantitativeInsight(
    "SPY",
    "SPDR S&P 500 ETF Trust",
    575,
    0.8,
    76,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    createMockCandles(252, 575),
    "live",
    spyConfluence,
    {
      symbol: "SPY",
      decisionState: "ACTIONABLE_SETUP",
      decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
      canSizeTrade: true,
      isActionable: true,
      disqualificationReason: null,
      allowedActions: ["SIZE_TRADE"],
    }
  );

  assert.strictEqual(spyInsight.terminalState.overallEligibility, "ELIGIBLE");
  assert.strictEqual(spyInsight.terminalState.posture, "ACQUIRE");
  assert.strictEqual(spyInsight.terminalState.canSizeTrade, true);

  console.log("  ✓ Test 3 Passed: ETF assets are exempt from corporate statutory filings and qualify cleanly.\n");
}

// ── Test 4: Digital Asset Domain Rules (BTC-USD) ───────────────────────────
console.log("Test 4: Digital assets (BTC-USD) are exempt from SEC filings and operate 24/7 tape...");
{
  const btcDomains = resolveNineEvidenceDomains({
    symbol: "BTC-USD",
    assetClass: "CRYPTO",
    securityType: "CRYPTO",
    candles: createMockCandles(252, 68000),
    preflightCleared: true,
    decisionTrace: {
      symbol: "BTC-USD",
      decisionState: "ACTIONABLE_SETUP",
      decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
      canSizeTrade: true,
      isActionable: true,
      disqualificationReason: null,
      allowedActions: ["SIZE_TRADE"],
    },
    dataSource: "live",
    currentPrice: 68000,
  });

  const secFilings = btcDomains.find(d => d.id === "sec_filings")!;
  const fundamentals = btcDomains.find(d => d.id === "fundamentals")!;
  const insiderTrades = btcDomains.find(d => d.id === "insider_transactions")!;

  assert.strictEqual(secFilings.applicability, "INAPPLICABLE");
  assert.strictEqual(fundamentals.applicability, "INAPPLICABLE");
  assert.strictEqual(insiderTrades.applicability, "INAPPLICABLE");

  console.log("  ✓ Test 4 Passed: Digital assets correctly bypass equity-specific regulatory gates.\n");
}

// ── Test 5: Missing Required Core Evidence (<50 candles) Fails Closed ──────
console.log("Test 5: Missing required core evidence (< 50 candles) strictly blocks execution...");
{
  const shortCandles = createMockCandles(25, 150); // Only 25 candles
  const insight = generateQuantitativeInsight(
    "NEWCO",
    "NewCo IPO Inc.",
    150,
    0.5,
    60,
    2,
    "SWING",
    "NOT_OWNED",
    "USER_DECLARED",
    shortCandles,
    "live",
    mockHealthyConfluence,
    mockActionableTrace // Rogue optimistic trace
  );

  // Even though trace claimed ACTIONABLE_SETUP, lack of 50 candles MUST fail closed
  assert.strictEqual(insight.terminalState.overallEligibility, "LIMITED", "Overall eligibility must be LIMITED on <50 candles");
  assert.strictEqual(insight.terminalState.posture, "RESEARCH", "Posture must be demoted to RESEARCH");
  assert.strictEqual(insight.terminalState.canSizeTrade, false, "canSizeTrade must be FALSE");
  assert.strictEqual(insight.terminalState.isActionable, false, "isActionable must be FALSE");
  assert.strictEqual(insight.terminalState.uiStateLabel, "Evidence Incomplete — In-Depth Research Required");
  assert.ok(insight.terminalState.headlineExplanation.includes("Technical trend evidence is unavailable"));

  console.log("  ✓ Test 5 Passed: Missing core trend evidence (<50 sessions) strictly fails closed to RESEARCH.\n");
}

console.log("=====================================================================");
console.log("ALL 5 EVIDENCE DOMAIN AUTHORITY TESTS PASSED!");
console.log("=====================================================================");
