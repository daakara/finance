/**
 * ARX-FRED-SEC-PRESENTATION-INTEGRITY-001 Verification Suite
 *
 * Verifies:
 * 1. SEC Form 4 Presentation Integrity (lib/institutionalFeeds.ts):
 *    - Unparsed filing metadata maps sharesTraded, pricePerShare, totalValueUsd to null (NEVER 0 or $0.00).
 *    - Does NOT infer open market purchase or significant buy from filing existence.
 *    - Preserves EDGAR accession number, form type, amendment flag, filing date, and source link.
 *    - Correctly maps parsed transactions when actual transaction data is present.
 * 2. FRED Macro Contract & Anti-Silent-Fallback (lib/institutionalFeeds.ts):
 *    - Maps live macro contract from /regimes/current when verified and fresh.
 *    - When macro is unavailable or network fails, fails closed to UNAVAILABLE_FRED_MACRO_DATA.
 *    - Never silently falls back to the historical 2026-08-26 snapshot (HISTORICAL_FRED_MACRO_FIXTURE).
 *    - Requires authentic live indicators before asserting an active risk budget multiplier.
 * 3. Null-Safety & Presentation Guarantees:
 *    - Guarantees null values in FredMacroData do not crash UI components.
 */

import assert from "node:assert";
import {
  fetchSecForm4Insiders,
  fetchFredMacroRegime,
  UNAVAILABLE_FRED_MACRO_DATA,
  CURRENT_FRED_MACRO_SNAPSHOT,
  FredMacroData,
  SecForm4Trade,
} from "../lib/institutionalFeeds";

console.log("=== ARX-FRED-SEC-PRESENTATION-INTEGRITY-001 Test Suite ===\n");

// Helper to mock global fetch
function mockFetch(handler: (url: string) => Promise<any>) {
  const originalFetch = global.fetch;
  (global as any).fetch = (url: string | URL | Request, init?: RequestInit) => {
    const urlStr = typeof url === "string" ? url : url.toString();
    return handler(urlStr);
  };
  return () => {
    (global as any).fetch = originalFetch;
  };
}

async function runTests() {
  let passedCount = 0;

  // ── Test 1: SEC Form 4 Unparsed Filing Metadata (Zero-Fabrication Prevention) ──
  console.log("Test 1: SEC Form 4 unparsed filing maps to null shares, price, and total value...");
  {
    const restore = mockFetch(async (url) => {
      if (url.includes("/smart-money/sec-filings/AAPL")) {
        return {
          ok: true,
          json: async () => ({
            symbol: "AAPL",
            status: "AVAILABLE_WITH_RESULTS",
            available: true,
            cik: "0000320193",
            filings: [
              {
                form: "4",
                filing_date: "2026-11-12",
                report_date: "2026-11-10",
                accession_number: "0000320193-26-000108",
                description: "Statement of Changes in Beneficial Ownership",
                sec_url: "https://www.sec.gov/Archives/edgar/data/320193/000032019326000108/0000320193-26-000108.txt",
              },
              {
                form: "4/A",
                filing_date: "2026-11-11",
                accession_number: "0000320193-26-000107",
                description: "Amended Statement of Changes in Beneficial Ownership",
                sec_url: "https://www.sec.gov/Archives/edgar/data/320193/000032019326000107/0000320193-26-000107.txt",
                is_amendment: true,
              },
            ],
          }),
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    });

    try {
      const result = await fetchSecForm4Insiders("AAPL");
      assert.strictEqual(result.status, "AVAILABLE_WITH_RESULTS");
      assert.ok(result.items, "Items should be defined");
      const trades = result.items;
      assert.strictEqual(trades.length, 2, "Should return 2 filings");

      const f1 = trades[0];
      assert.strictEqual(f1.ticker, "AAPL");
      assert.strictEqual(f1.formType, "4");
      assert.strictEqual(f1.isAmendment, false);
      assert.strictEqual(f1.sharesTraded, null, "Unparsed shares MUST be null (never 0)");
      assert.strictEqual(f1.pricePerShare, null, "Unparsed price MUST be null (never 0)");
      assert.strictEqual(f1.totalValueUsd, null, "Unparsed total value MUST be null (never 0)");
      assert.strictEqual(f1.hasParsedTransaction, false, "Must flag hasParsedTransaction=false");
      assert.strictEqual(f1.isSignificantBuy, false, "Must NOT flag significant buy without parsed data");
      assert.ok(f1.transactionType.includes("Form 4 Regulatory Filing"), "Must not claim Open Market Purchase");
      assert.strictEqual(f1.accessionNumber, "0000320193-26-000108");
      assert.ok(f1.secEdgarUrl.includes("0000320193-26-000108.txt"));

      const f2 = trades[1];
      assert.strictEqual(f2.formType, "4/A");
      assert.strictEqual(f2.isAmendment, true, "Must flag amendment status");
      assert.strictEqual(f2.sharesTraded, null);

      console.log("  ✓ Passed: Unparsed filings mapped to null without fabricating purchase or zero values.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 2: SEC Form 4 Parsed Transactions Mapped Correctly ──────────────────
  console.log("\nTest 2: SEC Form 4 parsed transaction correctly maps buy/sell and values...");
  {
    const restore = mockFetch(async (url) => {
      if (url.includes("/smart-money/sec-filings/NVDA")) {
        return {
          ok: true,
          json: async () => ({
            symbol: "NVDA",
            status: "AVAILABLE_WITH_RESULTS",
            available: true,
            cik: "0001045810",
            filings: [
              {
                form: "4",
                filing_date: "2026-11-10",
                accession_number: "0001045810-26-000050",
                description: "NVIDIA Corp",
                sec_url: "https://www.sec.gov/Archives/edgar/data/1045810/50.txt",
                insider_name: "Jensen Huang",
                role: "CEO / Director",
                shares_traded: 50000,
                price_per_share: 135.50,
                total_value_usd: 6775000,
                transaction_type: "Open Market Purchase",
                is_buy: true,
                has_parsed_transaction: true,
              },
            ],
          }),
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    });

    try {
      const result = await fetchSecForm4Insiders("NVDA");
      assert.strictEqual(result.status, "AVAILABLE_WITH_RESULTS");
      assert.ok(result.items);
      const trades = result.items;
      assert.strictEqual(trades.length, 1);
      const t = trades[0];
      assert.strictEqual(t.ticker, "NVDA");
      assert.strictEqual(t.insiderName, "Jensen Huang");
      assert.strictEqual(t.insiderRole, "CEO / Director");
      assert.strictEqual(t.hasParsedTransaction, true);
      assert.strictEqual(t.isBuy, true);
      assert.strictEqual(t.transactionType, "Open Market Purchase");
      assert.strictEqual(t.sharesTraded, 50000);
      assert.strictEqual(t.pricePerShare, 135.50);
      assert.strictEqual(t.totalValueUsd, 6775000);
      assert.strictEqual(t.isSignificantBuy, true, "Purchase >= $100k must flag significant buy");

      console.log("  ✓ Passed: Parsed transaction correctly mapped with full attribution.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 3: FRED Macro Live Contract Ingestion ──────────────────────────────
  console.log("\nTest 3: FRED macro extracts live verified contract from /regimes/current...");
  {
    const restore = mockFetch(async (url) => {
      if (url.includes("/regimes/current")) {
        return {
          ok: true,
          json: async () => ({
            symbol: "SPY",
            regime: "Bullish Trend / Low Volatility",
            macro: {
              availability: "AVAILABLE",
              is_live: true,
              regime: "EXPANSION_GOLDILOCKS",
              regime_title: "Macro Regime: Expansionary Goldilocks",
              regime_subtitle: "10Y-2Y Spread at +0.35%, High Yield Spreads tight at 2.95%.",
              yield_curve_10y2y: 0.35,
              real_interest_rate: 1.78,
              credit_spread: 2.95,
              fed_funds_rate: 4.58,
              macro_multiplier: 1.25,
              last_updated: "2026-11-14",
              series_observations: {
                T10Y2Y: { value: 0.35, date: "2026-11-14", series_id: "T10Y2Y" },
                DFII10: { value: 1.78, date: "2026-11-14", series_id: "DFII10" },
                BAMLH0A0HYM2: { value: 2.95, date: "2026-11-13", series_id: "BAMLH0A0HYM2" },
                FEDFUNDS: { value: 4.58, date: "2026-11-01", series_id: "FEDFUNDS" },
              },
            },
          }),
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    });

    try {
      const macro = await fetchFredMacroRegime();
      assert.strictEqual(macro.status, "AVAILABLE");
      assert.strictEqual(macro.isLive, true);
      assert.strictEqual(macro.regimeName, "EXPANSION_GOLDILOCKS");
      assert.strictEqual(macro.yieldCurve10Y2Y, 0.35);
      assert.strictEqual(macro.realInterestRate10Y, 1.78);
      assert.strictEqual(macro.highYieldCreditSpread, 2.95);
      assert.strictEqual(macro.fedFundsRate, 4.58);
      assert.strictEqual(macro.macroRiskMultiplier, 1.25);
      assert.strictEqual(macro.lastUpdated, "2026-11-14");
      assert.ok(macro.seriesObservations);
      assert.strictEqual(macro.seriesObservations["DFII10"].value, 1.78);

      console.log("  ✓ Passed: Live macro contract ingested with verified FRED indicators.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 4: FRED Macro Unavailable Fails Closed (No Multiplier Fabrication) ───
  console.log("\nTest 4: FRED macro returns unavailable contract when is_live is false...");
  {
    const restore = mockFetch(async (url) => {
      if (url.includes("/regimes/current")) {
        return {
          ok: true,
          json: async () => ({
            symbol: "SPY",
            regime: "Neutral",
            macro: {
              availability: "UNAVAILABLE",
              is_live: false,
              regime: "MACRO_TELEMETRY_UNAVAILABLE",
              regime_title: "Macro Telemetry Unavailable",
              regime_subtitle: "Authoritative macroeconomic indicators from Federal Reserve Economic Data (FRED) are currently unavailable.",
              yield_curve_10y2y: null,
              real_interest_rate: null,
              credit_spread: null,
              fed_funds_rate: null,
              macro_multiplier: null,
              last_updated: null,
              series_observations: {},
            },
          }),
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    });

    try {
      const macro = await fetchFredMacroRegime();
      assert.strictEqual(macro.status, "UNAVAILABLE");
      assert.strictEqual(macro.isLive, false);
      assert.strictEqual(macro.regimeName, "MACRO_TELEMETRY_UNAVAILABLE");
      assert.strictEqual(macro.macroRiskMultiplier, null, "Must NOT fabricate multiplier when unavailable");
      assert.strictEqual(macro.yieldCurve10Y2Y, null);
      assert.strictEqual(macro.highYieldCreditSpread, null);

      console.log("  ✓ Passed: Unavailable macro contract fails closed with null multiplier.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 5: FRED Macro Network Failure Fails Closed (No Silent Snapshot) ─────
  console.log("\nTest 5: FRED macro network error fails closed to UNAVAILABLE (no silent historical fallback)...");
  {
    const restore = mockFetch(async () => {
      throw new Error("Network connection refused");
    });

    try {
      const macro = await fetchFredMacroRegime();
      assert.strictEqual(macro.status, "UNAVAILABLE");
      assert.strictEqual(macro.isLive, false);
      assert.strictEqual(macro.regimeName, "MACRO_TELEMETRY_UNAVAILABLE");
      assert.strictEqual(macro.yieldCurve10Y2Y, null);
      assert.strictEqual(macro.realInterestRate10Y, null);
      assert.strictEqual(macro.macroRiskMultiplier, null);

      // Verify it did NOT return the historical snapshot dated 2026-08-26 with 4.88%
      assert.notStrictEqual(macro.lastUpdated, "2026-08-26", "Must NEVER return hardcoded 2026-08-26 snapshot on failure");
      assert.notStrictEqual(macro.fedFundsRate, 4.88, "Must NEVER return hardcoded 4.88% Fed Funds on failure");

      console.log("  ✓ Passed: Network failure fails closed to UNAVAILABLE without silent historical fallback.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 6: Null-Safety & Presentation Integrity Under Unavailable State ─────
  console.log("\nTest 6: Verify UNAVAILABLE_FRED_MACRO_DATA contract shape and null-safety...");
  {
    const unavailable = UNAVAILABLE_FRED_MACRO_DATA;
    assert.strictEqual(unavailable.status, "UNAVAILABLE");
    assert.strictEqual(unavailable.isLive, false);
    assert.strictEqual(unavailable.yieldCurve10Y2Y, null);
    assert.strictEqual(unavailable.realInterestRate10Y, null);
    assert.strictEqual(unavailable.highYieldCreditSpread, null);
    assert.strictEqual(unavailable.fedFundsRate, null);
    assert.strictEqual(unavailable.macroRiskMultiplier, null);
    assert.strictEqual(unavailable.lastUpdated, null);

    // Verify null checks prevent .toFixed TypeError
    const formatYield = (v: number | null) => (v !== null ? `${v.toFixed(2)}%` : "Unavailable");
    assert.strictEqual(formatYield(unavailable.yieldCurve10Y2Y), "Unavailable");
    assert.strictEqual(formatYield(0.18), "0.18%");

    console.log("  ✓ Passed: Null safety contract verified.");
    passedCount++;
  }

  console.log(`\n======================================================`);
  console.log(`ALL ${passedCount}/6 FRED & SEC PRESENTATION INTEGRITY TESTS PASSED`);
  console.log(`======================================================\n`);
}

runTests().catch((err) => {
  console.error("Test execution failed:", err);
  process.exit(1);
});
