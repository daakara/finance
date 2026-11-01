/**
 * ARX Smart Money Bounded Remediation Test Suite
 * Directive: ARX_SMART_MONEY_BOUNDED_REMEDIATION_001
 *
 * Verifies:
 * 1. Independent feed states for Options Flow, SEC Form 4, and Congressional Disclosures.
 * 2. Mandatory analytical evaluation for applicable equities (AAPL, NVDA).
 * 3. Exempt asset classification for funds (SPY), crypto (BTC-USD), and Foreign Private Issuers (ASML).
 * 4. Truthful coverage representation: missing/delayed feeds never masquerade as verified zero activity.
 * 5. Pre-flight execution independence: advisory smart money status never blocks trade sizing or execution clearance.
 */

import assert from "assert";
import { resolveNineEvidenceDomains } from "../lib/evidenceDomains";
import { CandleData, ConfluenceData } from "../lib/api";
import { DecisionTrace } from "../types/insight";

function createMockCandles(count: number, basePrice: number = 225): CandleData[] {
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
      close: basePrice + Math.sin(i) * 3,
      volume: 45000000,
    });
  }
  return candles;
}

const mockHealthyConfluence = {
  confluenceScore: 82.0,
  pillars: [
    { pillar: "TECHNICAL_STRUCTURE", label: "Technicals", score: 85, status: "positive", detail: "Bullish structure" },
    { pillar: "FUNDAMENTAL_SOLVENCY", label: "Fundamentals", score: 80, status: "positive", detail: "Strong balance sheet" },
    { pillar: "SMART_MONEY_FLOW", label: "Smart Money", score: 75, status: "neutral", detail: "Form 4 evaluated" },
    { pillar: "MACRO_SAFETY_FLOOR", label: "Macro", score: 78, status: "positive", detail: "VIX 19.4" },
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
console.log("ARX SMART MONEY BOUNDED REMEDIATION TEST SUITE");
console.log("Directive: ARX_SMART_MONEY_BOUNDED_REMEDIATION_001");
console.log("=====================================================================\n");

// ── Test 1: AAPL Independent Feeds with Verified SEC Form 4 ──────────────────
console.log("Test 1: AAPL independent feed resolution (Verified SEC Form 4, Pending Options/Congress)...");
{
  const domains = resolveNineEvidenceDomains({
    symbol: "AAPL",
    assetClass: "EQUITY",
    securityType: "COMMON_STOCK",
    candles: createMockCandles(252, 225),
    confluence: mockHealthyConfluence,
    smartMoney: {
      status: "AVAILABLE",
      overall_direction: "BEARISH",
      overall_confidence: "HIGH",
      window_days: 90,
      feeds: {
        options_flow: {
          status: "PIPELINE_PENDING",
          status_label: "Pipeline Pending",
          coverage: "SUPPORTED_PUBLIC_EQUITY",
          observations: [],
          observation_count: 0,
          direction: "UNVERIFIED",
          confidence: "NONE",
          explanation: "Live OPRA options sweep feed disconnected; order flow monitoring pending.",
          provenance: "OPRA Tape",
        },
        sec_form4: {
          status: "VERIFIED",
          status_label: "Verified",
          coverage: "SUPPORTED_PUBLIC_EQUITY",
          cik: "0000320193",
          observations: [
            { insider_name: "Tim Cook", acquired_disposed: "D", total_value_usd: 15000000 },
          ],
          observation_count: 24,
          direction: "BEARISH",
          confidence: "HIGH",
          explanation: "24 Form 4 insider transactions evaluated (net insider selling across executive officers).",
          provenance: "SEC EDGAR Form 4",
        },
        congressional_disclosures: {
          status: "PIPELINE_PENDING",
          status_label: "Pipeline Pending",
          coverage: "SUPPORTED_PUBLIC_EQUITY",
          observations: [],
          observation_count: 0,
          direction: "UNVERIFIED",
          confidence: "NONE",
          explanation: "Live Congressional disclosure sync feed disconnected; real-time legislative updates pending.",
          provenance: "STOCK Act Disclosures",
        },
      },
      secForm4Trades: [{ insider_name: "Tim Cook" } as any],
      optionsFlow: [],
      congressTrades: [],
    },
    preflightCleared: true,
    decisionTrace: mockActionableTrace,
    dataSource: "live",
    currentPrice: 225,
  });

  const optionsFlow = domains.find((d) => d.id === "options_flow")!;
  const insiderTrades = domains.find((d) => d.id === "insider_transactions")!;
  const congress = domains.find((d) => d.id === "congressional_disclosures")!;

  // 1. Verify Options Flow is Pipeline Pending (never false zero observations)
  assert.strictEqual(optionsFlow.status, "UNAVAILABLE");
  assert.strictEqual(optionsFlow.statusLabel, "Pipeline Pending");
  assert.strictEqual(optionsFlow.impact, "NON_BLOCKING");
  assert.strictEqual(optionsFlow.applicability, "OPTIONAL");

  // 2. Verify Insider Transactions is Verified with authentic count & explanation
  assert.strictEqual(insiderTrades.status, "VERIFIED");
  assert.strictEqual(insiderTrades.statusLabel, "Verified");
  assert.strictEqual(insiderTrades.impact, "NON_BLOCKING");
  assert.ok(insiderTrades.explanation.includes("24 Form 4"));

  // 3. Verify Congressional Disclosures is Pipeline Pending (never false zero observations)
  assert.strictEqual(congress.status, "UNAVAILABLE");
  assert.strictEqual(congress.statusLabel, "Pipeline Pending");
  assert.strictEqual(congress.impact, "NON_BLOCKING");
  assert.ok(congress.explanation.includes("disconnected"));

  console.log("  ✓ Test 1 Passed: Independent feed states correctly separated without false zero activity.\n");
}

// ── Test 2: SPY (ETF) Inapplicable Classification ───────────────────────────
console.log("Test 2: SPY (ETF) exempt asset classification...");
{
  const domains = resolveNineEvidenceDomains({
    symbol: "SPY",
    assetClass: "ETF",
    securityType: "ETF",
    candles: createMockCandles(252, 580),
    confluence: mockHealthyConfluence,
    smartMoney: {
      status: "INAPPLICABLE",
      overall_direction: "UNVERIFIED",
      overall_confidence: "NONE",
      window_days: 90,
      feeds: {
        options_flow: { status: "PIPELINE_PENDING", status_label: "Pipeline Pending" },
        sec_form4: {
          status: "INAPPLICABLE",
          status_label: "Exempt (Fund / Digital Asset)",
          coverage: "EXEMPT_ASSET",
          explanation: "Corporate insider transactions inapplicable to funds and digital assets.",
        },
        congressional_disclosures: { status: "PIPELINE_PENDING", status_label: "Pipeline Pending" },
      },
    },
    preflightCleared: true,
    currentPrice: 580,
  });

  const insiderTrades = domains.find((d) => d.id === "insider_transactions")!;
  assert.strictEqual(insiderTrades.status, "INAPPLICABLE");
  assert.strictEqual(insiderTrades.applicability, "INAPPLICABLE");
  assert.strictEqual(insiderTrades.impact, "NON_BLOCKING");

  console.log("  ✓ Test 2 Passed: ETF assets cleanly exempt from Form 4 insider transactions.\n");
}

// ── Test 3: BTC-USD (Crypto) Full Exemption ─────────────────────────────────
console.log("Test 3: BTC-USD (Crypto) complete smart money exemption...");
{
  const domains = resolveNineEvidenceDomains({
    symbol: "BTC-USD",
    assetClass: "CRYPTO",
    securityType: "CRYPTO",
    candles: createMockCandles(252, 68000),
    confluence: mockHealthyConfluence,
    smartMoney: {
      status: "INAPPLICABLE",
      overall_direction: "UNVERIFIED",
      overall_confidence: "NONE",
      window_days: 90,
      feeds: {
        options_flow: {
          status: "INAPPLICABLE",
          status_label: "Exempt (Digital Asset)",
          explanation: "Equity options flow sweeps inapplicable to digital assets.",
        },
        sec_form4: {
          status: "INAPPLICABLE",
          status_label: "Exempt (Fund / Digital Asset)",
          explanation: "Corporate insider transactions inapplicable to funds and digital assets.",
        },
        congressional_disclosures: {
          status: "INAPPLICABLE",
          status_label: "Exempt (Digital Asset)",
          explanation: "Congressional STOCK Act single-asset disclosures inapplicable to crypto tokens.",
        },
      },
    },
    preflightCleared: true,
    currentPrice: 68000,
  });

  const optionsFlow = domains.find((d) => d.id === "options_flow")!;
  const insiderTrades = domains.find((d) => d.id === "insider_transactions")!;
  const congress = domains.find((d) => d.id === "congressional_disclosures")!;

  assert.strictEqual(optionsFlow.status, "INAPPLICABLE");
  assert.strictEqual(insiderTrades.status, "INAPPLICABLE");
  assert.strictEqual(congress.status, "INAPPLICABLE");

  console.log("  ✓ Test 3 Passed: Digital assets cleanly exempt across all 3 smart money feeds.\n");
}

// ── Test 4: ASML (Foreign Private Issuer / ADR) Exemption ───────────────────
console.log("Test 4: ASML (Foreign ADR / FPI) regulatory exemption...");
{
  const domains = resolveNineEvidenceDomains({
    symbol: "ASML",
    assetClass: "EQUITY",
    securityType: "COMMON_STOCK",
    candles: createMockCandles(252, 750),
    confluence: mockHealthyConfluence,
    smartMoney: {
      status: "PIPELINE_PENDING",
      overall_direction: "UNVERIFIED",
      overall_confidence: "NONE",
      window_days: 90,
      feeds: {
        options_flow: { status: "PIPELINE_PENDING", status_label: "Pipeline Pending" },
        sec_form4: {
          status: "INAPPLICABLE",
          status_label: "Exempt (Foreign Issuer)",
          coverage: "FOREIGN_FPI_EXEMPT",
          explanation: "Foreign Private Issuer (FPI) exempt from US SEC Form 4 filings; local regulatory disclosures apply.",
          provenance: "Home Country Regulatory Authorities",
        },
        congressional_disclosures: { status: "PIPELINE_PENDING", status_label: "Pipeline Pending" },
      },
    },
    preflightCleared: true,
    currentPrice: 750,
  });

  const insiderTrades = domains.find((d) => d.id === "insider_transactions")!;
  assert.strictEqual(insiderTrades.status, "INAPPLICABLE");
  assert.strictEqual(insiderTrades.applicability, "INAPPLICABLE");
  assert.ok(insiderTrades.explanation.includes("Foreign Private Issuer"));

  console.log("  ✓ Test 4 Passed: Foreign Private Issuer exemption correctly recognized.\n");
}

// ── Test 5: Pre-Flight Execution Independence ───────────────────────────────
console.log("Test 5: Pre-Flight execution independence under pending smart money...");
{
  const domains = resolveNineEvidenceDomains({
    symbol: "AAPL",
    assetClass: "EQUITY",
    candles: createMockCandles(252, 225),
    confluence: mockHealthyConfluence,
    smartMoney: {
      status: "PIPELINE_PENDING",
      feeds: {
        options_flow: { status: "PIPELINE_PENDING" },
        sec_form4: { status: "PIPELINE_PENDING" },
        congressional_disclosures: { status: "PIPELINE_PENDING" },
      },
    },
    preflightCleared: true,
    decisionTrace: mockActionableTrace,
    dataSource: "live",
    currentPrice: 225,
  });

  // Verify non-blocking status of all 3 smart money domains
  for (const domainId of ["options_flow", "insider_transactions", "congressional_disclosures"]) {
    const domain = domains.find((d) => d.id === domainId)!;
    assert.strictEqual(domain.impact, "NON_BLOCKING", `${domainId} must be NON_BLOCKING`);
    assert.strictEqual(domain.applicability, "OPTIONAL", `${domainId} must be OPTIONAL`);
  }

  // Verify pre-flight clearance is strictly determined by core gates (preflightCleared=true -> VERIFIED)
  const preflight = domains.find((d) => d.id === "preflight_execution")!;
  assert.strictEqual(preflight.status, "VERIFIED");
  assert.strictEqual(preflight.impact, "BLOCKING");

  console.log("  ✓ Test 5 Passed: Advisory Smart Money never gates or compromises Pre-Flight execution clearance.\n");
}

console.log("=====================================================================");
console.log("ALL 5 SMART MONEY BOUNDED REMEDIATION TESTS PASSED CLEANLY!");
console.log("=====================================================================");
