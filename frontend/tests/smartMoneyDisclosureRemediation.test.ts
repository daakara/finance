/**
 * frontend/tests/smartMoneyDisclosureRemediation.test.ts
 *
 * ARX Terminal — Smart Money Statutory Disclosure & Evidence Integrity Test Suite.
 * Enforces permanent end-to-end regression invariants for PR #72 / Issue #67 (Directive ARX_PR72_DISCLOSURE_RECONCILIATION_027):
 * 1. DEF-72-01: Congressional PTR filing deadline follows 30d notice / 45d maximum transaction rule (STOCK Act, 5 U.S.C. § 13105(l)).
 * 2. DEF-72-02: SEC Form 4 timing explicitly specifies "Generally 2 business days" with statutory exception awareness; eliminates ambiguous shorthand.
 * 3. DEF-72-03: Unsupported promotional claims ("whales", "before it hits the headlines", conflation of distinct populations, predictions) are strictly absent.
 * 4. DEF-72-04: Arbitrary unverified $112.8M options benchmark is eliminated; options volume benchmark displays explicit "Unavailable / Historical Reference".
 * 5. DEF-72-05: Inactive Smart Money scanner truthfully reports PIPELINE_PENDING; historical disclosures, curated examples, observations, and offline states remain distinct.
 * 6. DEF-72-05b: Feed errors and connection failures are never represented as confirmed zero verified transactions.
 * 7. Non-actionability: Disclosures explicitly state they are historical references, not investment advice or trading recommendations.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

console.log("\n===============================================================================");
console.log("   ARX TERMINAL — SMART MONEY DISCLOSURE INTEGRITY SUITE (PR #72 / ISSUE #67)  ");
console.log("===============================================================================\n");

const pagePath = path.join(__dirname, "../app/smart-money/page.tsx");
const pageSource = fs.readFileSync(pagePath, "utf8");

// ---------------------------------------------------------------------------
// 1. DEF-72-01: Correct STOCK Act Filing Deadline
// ---------------------------------------------------------------------------
console.log("1. Verifying Congressional STOCK Act PTR statutory disclosure accuracy...");

// A. Elimination of inaccurate "15 to 45 calendar days"
const has15To45 = pageSource.includes("15 to 45 calendar days") || pageSource.includes("15–45 Days") || pageSource.includes("15-45 Days");
assert.strictEqual(
  has15To45,
  false,
  "DEF-72-01 Violation: Inaccurate '15 to 45 calendar days' must be completely eliminated"
);
console.log("   [OK] Inaccurate '15 to 45 calendar days' phrasing completely removed");

// B. Accurate statutory copy: 30 days after notification / 45 days after transaction
assert.ok(
  pageSource.includes("30 days after notification") && pageSource.includes("45 days after the transaction"),
  "DEF-72-01 Violation: Copy must state PTRs generally must be filed within 30 days after notification, and no later than 45 days after transaction"
);
assert.ok(
  pageSource.includes("5 U.S.C. § 13105(l)") || pageSource.includes("STOCK Act"),
  "DEF-72-01 Violation: Must reference statutory basis (STOCK Act / 5 U.S.C. § 13105(l))"
);
console.log("   [OK] Accurate statutory filing window (30d notification / 45d transaction) verified");

// C. Short-form badge text
assert.ok(
  pageSource.includes("Congressional PTR: 30d from notice / 45d maximum"),
  "DEF-72-01 Violation: Short-form badge must be 'Congressional PTR: 30d from notice / 45d maximum'"
);
console.log("   [OK] Short-form badge 'Congressional PTR: 30d from notice / 45d maximum' verified");

// ---------------------------------------------------------------------------
// 2. DEF-72-02: Clarify SEC Form 4 Timing
// ---------------------------------------------------------------------------
console.log("\n2. Verifying SEC Form 4 business-day disclosure and shorthand elimination...");

// A. Elimination of ambiguous shorthand "SEC Form 4: ≤2 Days"
const hasLe2Days = pageSource.includes("SEC Form 4: ≤2 Days") || pageSource.includes("SEC Form 4: <=2 Days");
assert.strictEqual(
  hasLe2Days,
  false,
  "DEF-72-02 Violation: Ambiguous shorthand 'SEC Form 4: ≤2 Days' must be eliminated"
);
console.log("   [OK] Ambiguous shorthand 'SEC Form 4: ≤2 Days' eliminated");

// B. Precise business day badge
assert.ok(
  pageSource.includes("SEC Form 4: Generally 2 business days"),
  "DEF-72-02 Violation: Badge must explicitly state 'SEC Form 4: Generally 2 business days'"
);
console.log("   [OK] Precise badge 'SEC Form 4: Generally 2 business days' verified");

// C. Text explicitly specifies business days and exceptions
assert.ok(
  pageSource.includes("2 business days") && pageSource.includes("statutory exceptions"),
  "DEF-72-02 Violation: Text must specify 2 business days subject to specific statutory exceptions"
);
console.log("   [OK] Statutory 2 business days text with exception awareness verified");

// ---------------------------------------------------------------------------
// 3. DEF-72-03: Remove Unsupported Institutional-Flow Claims & Promotional Copy
// ---------------------------------------------------------------------------
console.log("\n3. Verifying removal of unsupported promotional claims and source conflation...");

// A. No promotional copy
assert.strictEqual(
  pageSource.includes("whales"),
  false,
  "DEF-72-03 Violation: Promotional term 'whales' must be 0"
);
assert.strictEqual(
  pageSource.includes("before it hits the headlines"),
  false,
  "DEF-72-03 Violation: Promotional claim 'before it hits the headlines' must be 0"
);
console.log("   [OK] Promotional copy ('whales', 'before it hits the headlines') completely removed");

// B. Distinct populations not treated as interchangeable "institutional capital flows"
assert.strictEqual(
  pageSource.includes("Track disclosed institutional and congressional capital flows"),
  false,
  "DEF-72-03 Violation: Conflation of insider/congressional data as 'institutional capital flows' must be eliminated"
);
assert.ok(
  pageSource.includes("Distinguishes corporate insider disclosures (SEC Form 4), congressional reporting (STOCK Act PTRs), and historical options flow samples"),
  "DEF-72-03 Violation: Introduction must explicitly distinguish the three distinct populations"
);
console.log("   [OK] Three distinct populations explicitly differentiated");

// C. Explicit disclaimer against live accumulation and future prediction
assert.ok(
  pageSource.includes("not live institutional accumulation or forward-looking predictions"),
  "DEF-72-03 Violation: Must state disclosures do not reflect live institutional accumulation or forward-looking predictions"
);
console.log("   [OK] Live accumulation and prediction disclaimers verified");

// ---------------------------------------------------------------------------
// 4. DEF-72-04: Verify and Eliminate Unbacked $112.8M Options Benchmark
// ---------------------------------------------------------------------------
console.log("\n4. Verifying elimination of unbacked $112.8M options benchmark...");

const has1128M = pageSource.includes("$112.8M") || pageSource.includes("112.8M");
assert.strictEqual(
  has1128M,
  false,
  "DEF-72-04 Violation: Unverified $112.8M benchmark must not appear in Smart Money UI"
);
console.log("   [OK] Unverified $112.8M benchmark completely eliminated from Smart Money page");

assert.ok(
  pageSource.includes("Options Benchmark") && pageSource.includes("Unavailable") && pageSource.includes("Historical Reference"),
  "DEF-72-04 Violation: Card must display Options Benchmark as Unavailable with Historical Reference indicator"
);
console.log("   [OK] Options benchmark displayed as 'Unavailable' with 'Historical Reference' badge");

// ---------------------------------------------------------------------------
// 5. DEF-72-05: Truthful Scanner Availability & Distinguishable States
// ---------------------------------------------------------------------------
console.log("\n5. Verifying truthful scanner PIPELINE_PENDING and distinct operational states...");

// A. PIPELINE_PENDING badge and copy
assert.ok(
  pageSource.includes("Scanner: PIPELINE_PENDING"),
  "DEF-72-05 Violation: Badge must show 'Scanner: PIPELINE_PENDING'"
);
assert.ok(
  pageSource.includes("Continuous automated scanner pipelines remain in a held <strong>PIPELINE_PENDING</strong> state"),
  "DEF-72-05 Violation: Paragraph must state scanner pipelines remain held in PIPELINE_PENDING"
);
console.log("   [OK] Inactive scanner held state 'Scanner: PIPELINE_PENDING' verified");

// B. Feed Error / Offline handling prevents false zero representation
assert.ok(
  pageSource.includes("feedError") && pageSource.includes("Feed Offline / Connection Failure"),
  "DEF-72-05 Violation: Must support feedError state and display explicit 'Feed Offline / Connection Failure'"
);
assert.ok(
  pageSource.includes("Failed or unavailable data feeds are not represented as zero verified transactions"),
  "DEF-72-05 Violation: Must explicitly inform users that failed feeds are not zero transactions"
);
console.log("   [OK] Feed error state verified: failed feeds never masquerade as zero verified transactions");

// ---------------------------------------------------------------------------
// 6. Non-Actionability & Investment Advice Disclaimer
// ---------------------------------------------------------------------------
console.log("\n6. Verifying non-actionability and investment advice disclaimers...");

assert.ok(
  pageSource.includes("Historical Reference — Not Investment Advice"),
  "Non-actionability Violation: Must include 'Historical Reference — Not Investment Advice' disclaimer"
);
console.log("   [OK] 'Historical Reference — Not Investment Advice' disclaimer verified");

// ---------------------------------------------------------------------------
// 7. DEF-72-05 Runtime Network & Feed-Integrity Regression Tests (8 Mocked Scenarios)
// ---------------------------------------------------------------------------
console.log("\n7. Executing deterministic runtime network and feed-integrity regression tests...");

import { fetchSmartMoneyOverview, fetchSmartMoneyOverviewStrict } from "../lib/api";

interface SimulatedPageState {
  data: any;
  feedError: boolean;
  dataSource: "live" | "curated" | "delayed" | "fallback" | "unavailable";
  displayedDisclosuresCount: string | number;
  displayedSecCount: string | number;
  sourceBadgeLabel: string;
  errorBannerVisible: boolean;
  emptyStateVisible: boolean;
  fallbackWarningBannerVisible: boolean;
}

async function simulatePageLoadData(fetcher: () => Promise<any>): Promise<SimulatedPageState> {
  let data: any = null;
  let feedError = false;
  let dataSource: "live" | "curated" | "delayed" | "fallback" | "unavailable" = "curated";

  try {
    const res = await fetcher();
    if (res._dataSource === "fallback") {
      data = res;
      feedError = true;
      dataSource = "fallback";
    } else {
      data = res;
      feedError = false;
      dataSource = res._dataSource === "live" ? "delayed" : res._dataSource === "curated" ? "curated" : "curated";
    }
  } catch (err) {
    data = null;
    feedError = true;
    dataSource = "unavailable";
  }

  const congressTrades = data?.congress_trades || [];
  const secInsiders = data?.sec_insider_trades || [];

  const displayedDisclosuresCount =
    feedError || dataSource === "fallback" || dataSource === "unavailable"
      ? "Unavailable"
      : congressTrades.length;

  const displayedSecCount =
    feedError || dataSource === "fallback" || dataSource === "unavailable"
      ? "Unavailable"
      : secInsiders.length;

  const sourceBadgeLabel =
    dataSource === "curated"
      ? "📚 Curated Research (Aug 2026)"
      : dataSource === "delayed"
      ? "⏱️ Regulatory Filings (Delayed)"
      : dataSource === "unavailable"
      ? "⚠️ Feed Offline"
      : "⚠️ Unverified Demo Fixtures";

  const errorBannerVisible = feedError && dataSource === "unavailable";
  const emptyStateVisible = !feedError && congressTrades.length === 0;
  const fallbackWarningBannerVisible = dataSource === "fallback";

  return {
    data,
    feedError,
    dataSource,
    displayedDisclosuresCount,
    displayedSecCount,
    sourceBadgeLabel,
    errorBannerVisible,
    emptyStateVisible,
    fallbackWarningBannerVisible,
  };
}

async function runRuntimeNetworkTests() {
  const originalFetch = global.fetch;

  try {
    // 7.1 Successful Verified Response
    console.log("   7.1 Testing Successful Verified Response (status=LIVE, 2 records)...");
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: [{ id: "t1", ticker: "AAPL" }, { id: "t2", ticker: "MSFT" }],
        sec_insider_trades: [{ id: "s1", ticker: "NVDA" }],
      }),
    } as any);

    const s1 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s1.feedError, false, "7.1: feedError must be false on verified success");
    assert.strictEqual(s1.dataSource, "delayed", "7.1: dataSource must be 'delayed' on live regulatory filings");
    assert.strictEqual(s1.displayedDisclosuresCount, 2, "7.1: disclosures count must equal verified records (2)");
    assert.strictEqual(s1.displayedSecCount, 1, "7.1: SEC count must equal verified records (1)");
    assert.strictEqual(s1.sourceBadgeLabel, "⏱️ Regulatory Filings (Delayed)");
    assert.strictEqual(s1.errorBannerVisible, false);
    assert.strictEqual(s1.emptyStateVisible, false);
    console.log("       [OK] Verified live response correctly propagates with authentic counts and delayed badge");

    // 7.2 Explicit Curated Response
    console.log("   7.2 Testing Explicit Curated Response (status=CURATED, 3 records)...");
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        status: "CURATED",
        disclosure: "Curated August 2026 Research",
        congress_trades: [{ id: "c1", ticker: "PLTR" }, { id: "c2", ticker: "GOOG" }, { id: "c3", ticker: "AMZN" }],
        sec_insider_trades: [],
      }),
    } as any);

    const s2 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s2.feedError, false);
    assert.strictEqual(s2.dataSource, "curated", "7.2: dataSource must be 'curated'");
    assert.strictEqual(s2.displayedDisclosuresCount, 3);
    assert.strictEqual(s2.sourceBadgeLabel, "📚 Curated Research (Aug 2026)");
    assert.strictEqual(s2.errorBannerVisible, false);
    console.log("       [OK] Curated response preserved with curated badge and authentic counts");

    // 7.3 HTTP 500 Internal Server Error
    console.log("   7.3 Testing HTTP 500 Outage...");
    global.fetch = async () => ({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => ({ error: "Internal Server Error" }),
    } as any);

    let threw500 = false;
    try {
      await fetchSmartMoneyOverviewStrict();
    } catch {
      threw500 = true;
    }
    assert.strictEqual(threw500, true, "7.3: fetchSmartMoneyOverviewStrict must throw on HTTP 500");

    const s3 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s3.feedError, true, "7.3: feedError must be true on HTTP 500");
    assert.strictEqual(s3.dataSource, "unavailable", "7.3: dataSource must be 'unavailable'");
    assert.strictEqual(s3.displayedDisclosuresCount, "Unavailable", "7.3: count must be 'Unavailable' (not 0, not fabricated)");
    assert.strictEqual(s3.displayedSecCount, "Unavailable");
    assert.strictEqual(s3.sourceBadgeLabel, "⚠️ Feed Offline");
    assert.strictEqual(s3.errorBannerVisible, true, "7.3: error banner must be visible");
    console.log("       [OK] HTTP 500 propagates feedError=true, displays Unavailable count and Offline banner");

    // 7.4 Network Rejection (ECONNREFUSED)
    console.log("   7.4 Testing Network Rejection (ECONNREFUSED)...");
    global.fetch = async () => {
      throw new Error("Network request failed: ECONNREFUSED");
    };

    let threwNet = false;
    try {
      await fetchSmartMoneyOverview({ throwOnError: true });
    } catch {
      threwNet = true;
    }
    assert.strictEqual(threwNet, true, "7.4: Must throw on network rejection with throwOnError: true");

    const s4 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s4.feedError, true);
    assert.strictEqual(s4.dataSource, "unavailable");
    assert.strictEqual(s4.displayedDisclosuresCount, "Unavailable");
    assert.strictEqual(s4.sourceBadgeLabel, "⚠️ Feed Offline");
    assert.strictEqual(s4.errorBannerVisible, true);
    console.log("       [OK] Network rejection triggers catch block, feedError=true, and Unavailable stats");

    // 7.5 Timeout (AbortError)
    console.log("   7.5 Testing Timeout (AbortError)...");
    global.fetch = async () => {
      const err = new Error("The operation was aborted due to timeout");
      err.name = "AbortError";
      throw err;
    };

    let threwTimeout = false;
    try {
      await fetchSmartMoneyOverview({ throwOnError: true });
    } catch {
      threwTimeout = true;
    }
    assert.strictEqual(threwTimeout, true, "7.5: Must throw on timeout");

    const s5 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s5.feedError, true);
    assert.strictEqual(s5.dataSource, "unavailable");
    assert.strictEqual(s5.displayedDisclosuresCount, "Unavailable");
    assert.strictEqual(s5.sourceBadgeLabel, "⚠️ Feed Offline");
    assert.strictEqual(s5.errorBannerVisible, true);
    console.log("       [OK] Timeout correctly triggers feedError=true and Unavailable stats");

    // 7.6 Invalid JSON / Malformed Response
    console.log("   7.6 Testing Invalid JSON / Malformed Response...");
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => {
        throw new SyntaxError("Unexpected token < in JSON at position 0");
      },
    } as any);

    let threwJson = false;
    try {
      await fetchSmartMoneyOverview({ throwOnError: true });
    } catch {
      threwJson = true;
    }
    assert.strictEqual(threwJson, true, "7.6: Must throw on malformed JSON");

    const s6 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s6.feedError, true);
    assert.strictEqual(s6.dataSource, "unavailable");
    assert.strictEqual(s6.displayedDisclosuresCount, "Unavailable");
    assert.strictEqual(s6.sourceBadgeLabel, "⚠️ Feed Offline");
    assert.strictEqual(s6.errorBannerVisible, true);
    console.log("       [OK] Malformed response triggers catch block and Unavailable stats");

    // 7.7 Hardcoded Fallback Dataset
    console.log("   7.7 Testing Fallback Dataset Handling & Segregation...");
    global.fetch = async () => {
      throw new Error("Network unavailable");
    };

    // Default call without throwOnError returns fallback object for backwards compatibility
    const fallbackData = await fetchSmartMoneyOverview();
    assert.strictEqual(fallbackData._dataSource, "fallback", "7.7: Default fallback call returns _dataSource='fallback'");

    const s7 = await simulatePageLoadData(async () => fallbackData);
    assert.strictEqual(s7.feedError, true, "7.7: feedError must be true for fallback dataset (missing live source evidence)");
    assert.strictEqual(s7.dataSource, "fallback", "7.7: dataSource must be 'fallback'");
    assert.strictEqual(s7.displayedDisclosuresCount, "Unavailable", "7.7: Fallback must NEVER fabricate 14 as verified disclosures count");
    assert.strictEqual(s7.displayedSecCount, "Unavailable", "7.7: Fallback must NEVER fabricate 5 as verified SEC count");
    assert.strictEqual(s7.sourceBadgeLabel, "⚠️ Unverified Demo Fixtures");
    assert.strictEqual(s7.fallbackWarningBannerVisible, true, "7.7: Fallback warning banner must be active");
    console.log("       [OK] Fallback dataset is segregated, displays Unavailable verified counts, and bears unverified demo badge");

    // 7.8 Empty but Authoritative Successful Response
    console.log("   7.8 Testing Empty Authoritative Response (status=LIVE, 0 records)...");
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: [],
        sec_insider_trades: [],
      }),
    } as any);

    const s8 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s8.feedError, false, "7.8: feedError must be false on authoritative empty response");
    assert.strictEqual(s8.dataSource, "delayed", "7.8: dataSource must be 'delayed'");
    assert.strictEqual(s8.displayedDisclosuresCount, 0, "7.8: Genuine empty count must be 0 (NOT Unavailable)");
    assert.strictEqual(s8.displayedSecCount, 0, "7.8: Genuine empty count must be 0");
    assert.strictEqual(s8.sourceBadgeLabel, "⏱️ Regulatory Filings (Delayed)");
    assert.strictEqual(s8.errorBannerVisible, false, "7.8: Feed error banner must NOT be visible on genuine zero");
    assert.strictEqual(s8.emptyStateVisible, true, "7.8: Empty state banner must be visible on genuine zero");

    // CRITICAL EPISTEMIC INVARIANT: Outage vs Genuine Zero Distinguishability
    assert.notStrictEqual(
      s8.displayedDisclosuresCount,
      s3.displayedDisclosuresCount,
      "EPISTEMIC INVARIANT: Genuine zero disclosures (0) must be strictly distinguishable from feed outage (Unavailable)"
    );
    assert.notStrictEqual(
      s8.errorBannerVisible,
      s3.errorBannerVisible,
      "EPISTEMIC INVARIANT: Outage banner must only render during feed outage, never during genuine zero response"
    );
    console.log("       [OK] Authoritative zero response (0) is strictly distinguishable from API outage (Unavailable)");

  } finally {
    global.fetch = originalFetch;
  }
}

// ---------------------------------------------------------------------------
// 8. Static Source Code Assertions for Page-Level Feed-Integrity Wiring
// ---------------------------------------------------------------------------
console.log("\n8. Verifying static source wiring in frontend/app/smart-money/page.tsx...");

assert.ok(
  pageSource.includes("fetchSmartMoneyOverview({ throwOnError: true })"),
  "Wiring Violation: page.tsx must opt into throwOnError: true"
);
console.log("   [OK] page.tsx invokes fetchSmartMoneyOverview with { throwOnError: true }");

assert.ok(
  pageSource.includes('labelFallback="⚠️ Unverified Demo Fixtures"'),
  "Wiring Violation: DataSourceBadge in page.tsx must supply explicit fallback label"
);
console.log("   [OK] DataSourceBadge explicitly labels fallback as '⚠️ Unverified Demo Fixtures'");

assert.ok(
  pageSource.includes('feedError || dataSource === "fallback" || dataSource === "unavailable"'),
  "Wiring Violation: Verified counts must display 'Unavailable' when source evidence is missing (feedError, fallback, or unavailable)"
);
console.log("   [OK] Verified stats evaluate missing source evidence and display Unavailable");

assert.ok(
  pageSource.includes("Static Fallback Demonstration Fixtures"),
  "Wiring Violation: Static fallback demonstration banner must be defined"
);
console.log("   [OK] Static fallback demonstration fixture warning banner verified");

// Run runtime async suite
runRuntimeNetworkTests().then(() => {
  console.log("\n===============================================================================");
  console.log("   ALL SMART MONEY DISCLOSURE & FEED-INTEGRITY TESTS PASSED (15/15 GATES)    ");
  console.log("===============================================================================\n");
}).catch((err) => {
  console.error("FATAL RUNTIME TEST ERROR:", err);
  process.exit(1);
});
