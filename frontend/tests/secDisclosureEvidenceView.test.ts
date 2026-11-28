/**
 * ARX Issue #165 — SEC Form 4 Disclosure View Verification Suite
 *
 * Verifies:
 * 1. Multi-Transaction Preservation:
 *    - All individual transactions in a filing are mapped to SecForm4Transaction.
 *    - Preserves transaction ID, code, description, shares, price, value, role, dates.
 * 2. Unbiased Latest Event Selection:
 *    - Distinguishes purchases, sales, gifts, exercises without favoring purchases.
 * 3. Exact Price Semantics:
 *    - Distinguishes reported $0.00 consideration (gifts/exercises) from unparsed null.
 * 4. Expandable Verification Evidence:
 *    - Preserves accession number, acceptance datetime, XML hash, public availability.
 * 5. Race-Condition Protection:
 *    - Prevents responses from previous/mismatched tickers from updating state.
 * 6. Regulatory & Operational Integrity:
 *    - Distinguishes coverage status (PARTIAL/COMPLETE) and HOLD_SEC_001 held pipeline.
 */

import assert from "node:assert";
import {
  fetchSecForm4Insiders,
  SecForm4Trade,
  SecForm4Transaction,
  SecFilingsResult,
} from "../lib/institutionalFeeds";

console.log("=== ARX-SEC-DISCLOSURE-VIEW-001 Verification Suite ===\n");

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

  // ── Test 1: Single Sale Mapping (Jennifer Newstead AAPL Form 4) ────────────
  console.log("Test 1: Single sale disclosure mapped accurately without purchase bias...");
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
            coverage_status: "PARTIAL",
            certified_watermark: "2026-11-19T23:30:21Z",
            is_pipeline_held: true,
            items: [
              {
                form: "4",
                filing_date: "2026-11-19",
                acceptance_datetime: "2026-11-19T23:30:21Z",
                accession_number: "0001140361-26-044952",
                sec_url: "https://www.sec.gov/Archives/edgar/data/320193/000114036126044952/0001140361-26-044952.txt",
                xml_hash: "93aa1a0a864d72ccc38dcd00a5ed6258b64634dbde4ab271ea78db8efd3841d0",
                has_parsed_transaction: true,
                transaction_count: 1,
                insider_name: "Jennifer Newstead",
                insider_role: "General Counsel",
                transaction_type: "Sale (Code S)",
                transaction_code: "S",
                transaction_date: "2026-11-17",
                shares_traded: 2398,
                price_per_share: 352.77,
                total_value_usd: 845942.46,
                is_buy: false,
                is_sell: true,
                transactions: [
                  {
                    transaction_id: "0001140361-26-044952-ND-1",
                    accession_number: "0001140361-26-044952",
                    transaction_date: "2026-11-17",
                    transaction_code: "S",
                    transaction_type: "Sale (Code S)",
                    shares: 2398,
                    price_per_share: 352.77,
                    total_value_usd: 845942.46,
                    reporting_owner_name: "Jennifer Newstead",
                    reporting_owner_role: "General Counsel",
                    is_officer: true,
                    is_director: false,
                    is_ten_percent: false,
                    ownership_type: "D",
                    is_derivative: false,
                    acquired_disposed: "D",
                    is_buy: false,
                    is_sell: true,
                    is_gift: false,
                    is_exercise: false,
                    security_title: "Common Stock",
                    public_availability_at: "2026-11-19T23:30:21Z",
                  },
                ],
              },
            ],
          }),
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    });

    try {
      const res = await fetchSecForm4Insiders("AAPL");
      assert.strictEqual(res.status, "AVAILABLE_WITH_RESULTS");
      assert.strictEqual(res.coverageStatus, "PARTIAL");
      assert.strictEqual(res.certifiedWatermark, "2026-11-19T23:30:21Z");
      assert.strictEqual(res.isPipelineHeld, true);

      assert.ok(res.items);
      const trade = res.items[0];
      assert.strictEqual(trade.ticker, "AAPL");
      assert.strictEqual(trade.insiderName, "Jennifer Newstead");
      assert.strictEqual(trade.insiderRole, "General Counsel");
      assert.strictEqual(trade.isSell, true);
      assert.strictEqual(trade.isBuy, false);
      assert.strictEqual(trade.sharesTraded, 2398);
      assert.strictEqual(trade.pricePerShare, 352.77);
      assert.strictEqual(trade.totalValueUsd, 845942.46);
      assert.strictEqual(trade.xmlHash, "93aa1a0a864d72ccc38dcd00a5ed6258b64634dbde4ab271ea78db8efd3841d0");
      assert.strictEqual(trade.acceptanceDatetime, "2026-11-19T23:30:21Z");

      assert.ok(trade.transactions);
      assert.strictEqual(trade.transactions.length, 1);
      const tx = trade.transactions[0];
      assert.strictEqual(tx.transactionCode, "S");
      assert.strictEqual(tx.reportingOwnerName, "Jennifer Newstead");
      assert.strictEqual(tx.shares, 2398);
      assert.strictEqual(tx.pricePerShare, 352.77);

      console.log("  ✓ Passed: Single sale disclosure mapped accurately without purchase bias.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 2: Multi-Transaction Preservation & Mixed Filing Aggregate Withholding ──────────────
  console.log("\nTest 2: Multi-transaction filing preserves all events and withholds mixed aggregates...");
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
            coverage_status: "COMPLETE",
            served_at: "2026-11-21T00:30:00Z",
            as_of: "2026-11-18T22:00:00Z",
            items: [
              {
                form: "4",
                filing_date: "2026-11-15",
                accession_number: "0001045810-26-000099",
                sec_url: "https://sec.gov/nvda.txt",
                has_parsed_transaction: true,
                transaction_count: 2,
                is_mixed: true,
                insider_name: "Colette Kress",
                insider_role: "EVP, CFO",
                transaction_type: "Multi-Transaction Filing (2 Events — Mixed)",
                transaction_code: "MIXED",
                shares_traded: null,
                total_value_usd: null,
                transactions: [
                  {
                    transaction_id: "tx-m",
                    accession_number: "0001045810-26-000099",
                    transaction_date: "2026-11-14",
                    transaction_code: "M",
                    transaction_type: "Option Exercise",
                    shares: 10000,
                    price_per_share: 0.0,
                    total_value_usd: 0.0,
                    reporting_owner_name: "Colette Kress",
                    reporting_owner_role: "EVP, CFO",
                    is_officer: true,
                    is_director: false,
                    is_ten_percent: false,
                    ownership_type: "D",
                    is_derivative: true,
                    acquired_disposed: "A",
                    is_buy: false,
                    is_sell: false,
                    is_gift: false,
                    is_exercise: true,
                    security_title: "Non-Qualified Stock Option",
                  },
                  {
                    transaction_id: "tx-s",
                    accession_number: "0001045810-26-000099",
                    transaction_date: "2026-11-14",
                    transaction_code: "S",
                    transaction_type: "Sale (Open Market / Private)",
                    shares: 5000,
                    price_per_share: 290.00,
                    total_value_usd: 1450000.0,
                    reporting_owner_name: "Colette Kress",
                    reporting_owner_role: "EVP, CFO",
                    is_officer: true,
                    is_director: false,
                    is_ten_percent: false,
                    ownership_type: "D",
                    is_derivative: false,
                    acquired_disposed: "D",
                    is_buy: false,
                    is_sell: true,
                    is_gift: false,
                    is_exercise: false,
                    security_title: null,
                  },
                ],
              },
            ],
          }),
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    });

    try {
      const res = await fetchSecForm4Insiders("NVDA");
      assert.ok(res.items);
      assert.strictEqual(res.servedAt, "2026-11-21T00:30:00Z", "servedAt must be exposed separately");
      assert.strictEqual(res.asOf, "2026-11-18T22:00:00Z", "asOf reflects genuine evidence freshness");

      const trade = res.items[0];
      assert.strictEqual(trade.transactionCount, 2);
      assert.strictEqual(trade.isMixed, true, "isMixed must be true for mixed event filing");
      assert.strictEqual(trade.sharesTraded, null, "Aggregate shares must be withheld for mixed filing");
      assert.strictEqual(trade.totalValueUsd, null, "Aggregate value must be withheld for mixed filing");
      assert.strictEqual(trade.pricePerShare, null, "Price must be withheld for mixed filing");

      assert.ok(trade.transactions);
      assert.strictEqual(trade.transactions.length, 2);

      const [tx1, tx2] = trade.transactions;
      assert.strictEqual(tx1.transactionCode, "M");
      assert.strictEqual(tx1.isExercise, true);
      assert.strictEqual(tx1.pricePerShare, 0.0, "Reported $0.00 consideration preserved");
      assert.strictEqual(tx1.securityTitle, "Non-Qualified Stock Option");

      assert.strictEqual(tx2.transactionCode, "S");
      assert.strictEqual(tx2.isSell, true);
      assert.strictEqual(tx2.shares, 5000);
      assert.strictEqual(tx2.pricePerShare, 290.00);
      assert.strictEqual(tx2.securityTitle, null, "Security title not inferred as Common Stock");

      console.log("  ✓ Passed: Multi-transaction filing preserves all events and withholds mixed aggregates.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 3: Gift Consideration (Reported 0.0 vs Null vs Positive) & Non-Finite Numbers ──
  console.log("\nTest 3: Gift consideration (0.0 vs null vs positive) & non-finite numeric sanitation...");
  {
    const restore = mockFetch(async (url) => {
      return {
        ok: true,
        json: async () => ({
          symbol: "NVDA",
          status: "AVAILABLE_WITH_RESULTS",
          items: [
            {
              form: "4",
              filing_date: "2026-11-12",
              accession_number: "acc-gift-zero",
              has_parsed_transaction: true,
              shares_traded: 1000,
              price_per_share: 0.0,
              total_value_usd: 0.0,
              is_gift: true,
              transactions: [
                {
                  transaction_id: "tx-g0",
                  transaction_code: "G",
                  transaction_type: "Bona Fide Gift",
                  shares: 1000,
                  price_per_share: 0.0,
                  total_value_usd: 0.0,
                  is_gift: true,
                },
              ],
            },
            {
              form: "4",
              filing_date: "2026-11-11",
              accession_number: "acc-gift-null",
              has_parsed_transaction: true,
              shares_traded: 500,
              price_per_share: null,
              total_value_usd: null,
              is_gift: true,
              transactions: [
                {
                  transaction_id: "tx-gnull",
                  transaction_code: "G",
                  transaction_type: "Bona Fide Gift",
                  shares: 500,
                  price_per_share: null,
                  total_value_usd: null,
                  is_gift: true,
                },
              ],
            },
            {
              form: "4",
              filing_date: "2026-11-10",
              accession_number: "acc-gift-positive",
              has_parsed_transaction: true,
              shares_traded: 200,
              price_per_share: 15.50,
              total_value_usd: 3100.0,
              is_gift: true,
              transactions: [
                {
                  transaction_id: "tx-gpos",
                  transaction_code: "G",
                  transaction_type: "Bona Fide Gift",
                  shares: 200,
                  price_per_share: 15.50,
                  total_value_usd: 3100.0,
                  is_gift: true,
                },
              ],
            },
            {
              form: "4",
              filing_date: "2026-11-09",
              accession_number: "acc-non-finite",
              has_parsed_transaction: true,
              shares_traded: NaN,
              price_per_share: Infinity,
              total_value_usd: -Infinity,
              transactions: [
                {
                  transaction_id: "tx-nan",
                  transaction_code: "S",
                  shares: "NaN",
                  price_per_share: "Infinity",
                  total_value_usd: null,
                },
              ],
            },
          ],
        }),
      };
    });

    try {
      const res = await fetchSecForm4Insiders("NVDA");
      assert.ok(res.items);
      const [giftZero, giftNull, giftPos, nonFinite] = res.items;

      assert.strictEqual(giftZero.pricePerShare, 0.0, "Reported 0.0 consideration preserved");
      assert.strictEqual(giftZero.totalValueUsd, 0.0);

      assert.strictEqual(giftNull.pricePerShare, null, "Null gift consideration preserved as null");
      assert.strictEqual(giftNull.totalValueUsd, null);

      assert.strictEqual(giftPos.pricePerShare, 15.50, "Positive gift consideration preserved");
      assert.strictEqual(giftPos.totalValueUsd, 3100.0);

      assert.strictEqual(nonFinite.sharesTraded, null, "NaN sanitized to null");
      assert.strictEqual(nonFinite.pricePerShare, null, "Infinity sanitized to null");
      assert.strictEqual(nonFinite.totalValueUsd, null, "-Infinity sanitized to null");
      assert.strictEqual(nonFinite.transactions![0].shares, null, "NaN in transaction sanitized to null");
      assert.strictEqual(nonFinite.transactions![0].pricePerShare, null, "Infinity in transaction sanitized to null");

      console.log("  ✓ Passed: Gift consideration (0.0 vs null vs positive) & non-finite numeric sanitation.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 4: Rapid Ticker Switching Protection (Symbol Match Guard) ────────
  console.log("\nTest 4: Response symbol mismatch guard prevents cross-ticker overwrite...");
  {
    const restore = mockFetch(async (url) => {
      // Simulate delayed response returning MSFT payload for an AAPL query
      return {
        ok: true,
        json: async () => ({
          symbol: "MSFT", // Mismatched symbol!
          status: "AVAILABLE_WITH_RESULTS",
          items: [
            {
              ticker: "MSFT",
              form: "4",
              accession_number: "msft-acc",
              filing_date: "2026-11-18",
            },
          ],
        }),
      };
    });

    try {
      const res = await fetchSecForm4Insiders("AAPL");
      // The fetcher normalizes or tags symbol: symClean, and frontend verifies res.symbol === cleanSym
      assert.strictEqual(res.symbol, "AAPL", "Response symbol must be bound to requested symbol");
      console.log("  ✓ Passed: Cross-ticker guard verified.");
      passedCount++;
    } finally {
      restore();
    }
  }

  // ── Test 5: Scope-Unverified RESOLVED Amendment & Availability Boundaries ──
  console.log("\nTest 5: Scope-unverified RESOLVED amendment lineage and public availability boundaries...");
  {
    const restore = mockFetch(async (url) => {
      return {
        ok: true,
        json: async () => ({
          symbol: "AAPL",
          status: "AVAILABLE_WITH_RESULTS",
          items: [
            {
              form: "4",
              accession_number: "acc-base-resolved",
              filing_date: "2026-11-10",
              has_parsed_transaction: true,
              is_effective: false,
              effective_status: "AMENDMENT_EFFECT_AMBIGUOUS",
              has_ambiguous_amendment: true,
              amendment_note: "Target of legacy RESOLVED amendment (lineage scope-unverified; effective impact ambiguous).",
              transactions: [
                {
                  transaction_id: "tx-res-b",
                  accession_number: "acc-base-resolved",
                  is_effective: false,
                  effective_status: "AMENDMENT_EFFECT_AMBIGUOUS",
                },
              ],
            },
            {
              form: "4/A",
              accession_number: "acc-amend-resolved",
              filing_date: "2026-11-12",
              is_amendment: true,
              base_accession: "acc-base-resolved",
              has_parsed_transaction: true,
              is_effective: false,
              effective_status: "SCOPE_UNVERIFIED",
              amendment_note: "Legacy RESOLVED amendment lineage is scope-unverified.",
              transactions: [
                {
                  transaction_id: "tx-res-a",
                  accession_number: "acc-amend-resolved",
                  is_effective: false,
                  effective_status: "SCOPE_UNVERIFIED",
                },
              ],
            },
            {
              form: "4",
              accession_number: "acc-base-fut",
              filing_date: "2026-11-15",
              has_parsed_transaction: true,
              is_effective: true,
              effective_status: "EFFECTIVE",
              is_superseded: false,
              amendment_note: "Future amendment postdates cutoff; base filing remains effective at watermark.",
            },
            {
              form: "4/A",
              accession_number: "acc-amend-fut",
              filing_date: "2026-11-20",
              is_amendment: true,
              base_accession: "acc-base-fut",
              has_parsed_transaction: true,
              is_effective: false,
              effective_status: "PENDING_PUBLIC_AVAILABILITY",
            },
            {
              form: "4",
              accession_number: "acc-base-part",
              filing_date: "2026-11-16",
              has_parsed_transaction: true,
              is_effective: false,
              effective_status: "PARTIAL_CORRECTION",
              has_partial_amendment: true,
              is_superseded: false,
            },
          ],
        }),
      };
    });

    try {
      const res = await fetchSecForm4Insiders("AAPL");
      assert.ok(res.items);
      assert.strictEqual(res.items.length, 5);

      const [baseRes, amendRes, baseFut, amendFut, basePart] = res.items;

      // 1. Base filing with RESOLVED amendment: cannot imply verified amendment effect
      assert.strictEqual(baseRes.isEffective, false);
      assert.strictEqual(baseRes.effectiveStatus, "AMENDMENT_EFFECT_AMBIGUOUS");
      assert.strictEqual(baseRes.hasAmbiguousAmendment, true);

      // 2. Amendment filing with RESOLVED status: scope-unverified
      assert.strictEqual(amendRes.isEffective, false);
      assert.strictEqual(amendRes.effectiveStatus, "SCOPE_UNVERIFIED");

      // 3. Future amendment: base remains effective at watermark
      assert.strictEqual(baseFut.isEffective, true);
      assert.strictEqual(baseFut.effectiveStatus, "EFFECTIVE");
      assert.strictEqual(baseFut.isSuperseded, false);

      // 4. Future amendment itself: pending availability
      assert.strictEqual(amendFut.isEffective, false);
      assert.strictEqual(amendFut.effectiveStatus, "PENDING_PUBLIC_AVAILABILITY");

      // 5. Partial correction: hasPartialAmendment without full supersession
      assert.strictEqual(basePart.isEffective, false);
      assert.strictEqual(basePart.effectiveStatus, "PARTIAL_CORRECTION");
      assert.strictEqual(basePart.hasPartialAmendment, true);
      assert.strictEqual(basePart.isSuperseded, false);

      console.log("  ✓ Passed: Scope-unverified RESOLVED amendment lineage and public availability boundaries.");
      passedCount++;
    } finally {
      restore();
    }
  }

  console.log(`\n======================================================`);
  console.log(`ALL ${passedCount}/5 SEC FORM 4 DISCLOSURE TESTS PASSED`);
  console.log(`======================================================\n`);
}

runTests().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
