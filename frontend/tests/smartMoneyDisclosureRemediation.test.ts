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

import { fetchSmartMoneyOverview, fetchSmartMoneyOverviewStrict, isValidSmartMoneyPayload, classifySmartMoneyProvenance } from "../lib/api";

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
    } else if (res._dataSource === "unavailable") {
      data = res;
      feedError = true;
      dataSource = "unavailable";
    } else if (res._dataSource === "live") {
      data = res;
      feedError = false;
      dataSource = "delayed";
    } else {
      data = res;
      feedError = false;
      dataSource = "curated";
    }
  } catch (err) {
    data = null;
    feedError = true;
    dataSource = "unavailable";
  }

  const isCongressAvailable = !feedError && Boolean(data && Array.isArray(data.congress_trades)) && dataSource !== "fallback" && dataSource !== "unavailable";
  const isSecAvailable = !feedError && Boolean(data && Array.isArray(data.sec_insider_trades)) && dataSource !== "fallback" && dataSource !== "unavailable";
  const isOptionsAvailable = !feedError && Boolean(data && Array.isArray(data.options_flow)) && dataSource !== "fallback" && dataSource !== "unavailable";

  const congressTrades = isCongressAvailable ? (data?.congress_trades || []) : [];
  const secInsiders = isSecAvailable ? (data?.sec_insider_trades || []) : [];

  const displayedDisclosuresCount =
    feedError || dataSource === "fallback" || dataSource === "unavailable" || !isCongressAvailable
      ? "Unavailable"
      : congressTrades.length;

  const displayedSecCount =
    feedError || dataSource === "fallback" || dataSource === "unavailable" || !isSecAvailable
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
  const emptyStateVisible = !feedError && isCongressAvailable && congressTrades.length === 0;
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

    // 7.9 Missing options_flow in Successful Response
    console.log("   7.9 Testing Missing options_flow in Successful Response...");
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: [{ id: "t1", ticker: "AAPL" }],
        sec_insider_trades: [{ id: "s1", ticker: "NVDA" }],
      }),
    } as any);

    const s9 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s9.feedError, false, "7.9: Missing options_flow should not fail entire overview fetch");
    assert.strictEqual(s9.displayedDisclosuresCount, 1, "7.9: Congress disclosures should remain available (1)");
    assert.strictEqual(s9.displayedSecCount, 1, "7.9: SEC insiders should remain available (1)");
    console.log("       [OK] Response missing options_flow preserves other feeds without error");

    // 7.10 Missing optional sec_insider_trades
    console.log("   7.10 Testing Missing Optional sec_insider_trades...");
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: [{ id: "t1", ticker: "AAPL" }],
        options_flow: [{ ticker: "AAPL", type: "CALL", strike: "230", expiration: "2026-09", spot_price: 220, premium: "$100K", volume_oi_ratio: 2.5, sentiment: "Bullish" }],
      }),
    } as any);

    const s10 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s10.feedError, false, "7.10: Missing sec_insider_trades should not fail overview");
    assert.strictEqual(s10.displayedDisclosuresCount, 1, "7.10: Congress count is 1");
    assert.strictEqual(s10.displayedSecCount, "Unavailable", "7.10: SEC count must be 'Unavailable' (NOT 0)");
    console.log("       [OK] Response missing sec_insider_trades marks SEC as Unavailable while Congress remains 1");

    // 7.11 Non-array congress_trades
    console.log("   7.11 Testing Structurally Invalid congress_trades (Non-array string)...");
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: "invalid_string_not_array",
        sec_insider_trades: [],
      }),
    } as any);

    let threwNonArray = false;
    try {
      await fetchSmartMoneyOverview({ throwOnError: true });
    } catch (err: any) {
      threwNonArray = true;
      assert.ok(err.message.includes("failed runtime shape validation"));
    }
    assert.strictEqual(threwNonArray, true, "7.11: Non-array congress_trades must fail runtime shape validation");

    const s11 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s11.feedError, true, "7.11: feedError must be true on shape validation error");
    assert.strictEqual(s11.displayedDisclosuresCount, "Unavailable");
    console.log("       [OK] Non-array congress_trades fails shape validation and sets feedError=true");

    // 7.12 Explicit Partial Availability (Congress valid, other feeds null)
    console.log("   7.12 Testing Explicit Partial Availability (Congress valid, others null)...");
    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: [{ id: "t1", ticker: "PLTR" }, { id: "t2", ticker: "NVDA" }],
        sec_insider_trades: null,
        options_flow: null,
      }),
    } as any);

    const s12 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s12.feedError, false, "7.12: Partial availability is a valid overview payload");
    assert.strictEqual(s12.displayedDisclosuresCount, 2, "7.12: Congress count displays 2");
    assert.strictEqual(s12.displayedSecCount, "Unavailable", "7.12: SEC count displays Unavailable (not 0)");
    console.log("       [OK] Partial response correctly isolates valid Congress feed (2) and unavailable SEC (Unavailable)");

    // 7.13 isValidSmartMoneyPayload direct validation
    console.log("   7.13 Testing isValidSmartMoneyPayload Shape Validation Contract directly...");
    assert.strictEqual(isValidSmartMoneyPayload(null), false, "Rejects null");
    assert.strictEqual(isValidSmartMoneyPayload(undefined), false, "Rejects undefined");
    assert.strictEqual(isValidSmartMoneyPayload("string"), false, "Rejects string");
    assert.strictEqual(isValidSmartMoneyPayload(123), false, "Rejects number");
    assert.strictEqual(isValidSmartMoneyPayload([]), false, "Rejects array root");
    assert.strictEqual(isValidSmartMoneyPayload({}), false, "Rejects empty object with no recognized feeds or fields");
    assert.strictEqual(isValidSmartMoneyPayload({ congress_trades: "not-an-array" }), false, "Rejects non-array congress_trades");
    assert.strictEqual(isValidSmartMoneyPayload({ sec_insider_trades: 42 }), false, "Rejects non-array sec_insider_trades");
    assert.strictEqual(isValidSmartMoneyPayload({ options_flow: {} }), false, "Rejects non-array options_flow");
    assert.strictEqual(isValidSmartMoneyPayload({ congress_trades: [] }), true, "Accepts authoritative empty congress_trades");
    assert.strictEqual(isValidSmartMoneyPayload({ total_congress_filings_30d: 0 }), true, "Accepts overview property");
    assert.strictEqual(isValidSmartMoneyPayload({ status: "CURATED" }), true, "Accepts status property");
    assert.strictEqual(isValidSmartMoneyPayload({ congress_trades: [], sec_insider_trades: null }), true, "Accepts partial null feeds");
    console.log("       [OK] isValidSmartMoneyPayload adheres to runtime validation requirements");

    // 7.14 Authentic Backend Curated Contract Fixture Consumption (ARX_PR72_BACKEND_PROVENANCE_CONTRACT_037)
    console.log("   7.14 Testing Authentic Backend Curated Contract Fixture (backendSmartMoneyOverviewCurated.json)...");
    const curatedFixturePath = path.join(__dirname, "fixtures/backendSmartMoneyOverviewCurated.json");
    const curatedFixture = JSON.parse(fs.readFileSync(curatedFixturePath, "utf8"));

    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => curatedFixture,
    } as any);

    const curatedOverview = await fetchSmartMoneyOverview({ throwOnError: true });
    assert.strictEqual(curatedOverview._dataSource, "curated", "7.14: Actual backend curated response must be classified as 'curated'");
    assert.strictEqual(curatedOverview.status, "CURATED", "7.14: Backend contract status must be 'CURATED'");
    assert.strictEqual(curatedOverview.is_live, false, "7.14: is_live must be explicitly false");
    assert.strictEqual(curatedOverview.available, true, "7.14: available must be true");
    assert.strictEqual(curatedOverview.dataset_date, "2026-08-28", "7.14: dataset_date must match August 2026 release");
    assert.strictEqual(classifySmartMoneyProvenance(curatedFixture), "curated", "7.14: classifySmartMoneyProvenance must return 'curated'");

    const s14 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s14.feedError, false, "7.14: feedError must be false on valid curated archive");
    assert.strictEqual(s14.dataSource, "curated", "7.14: Page dataSource must be 'curated'");
    assert.strictEqual(s14.displayedDisclosuresCount, 40, "7.14: Curated disclosures count must match fixture (40)");
    assert.strictEqual(s14.displayedSecCount, 20, "7.14: Curated SEC count must match fixture (20)");
    assert.strictEqual(s14.sourceBadgeLabel, "📚 Curated Research (Aug 2026)", "7.14: Must display Curated Research badge");
    assert.notStrictEqual(s14.sourceBadgeLabel, "⏱️ Regulatory Filings (Delayed)", "7.14: Must NEVER assign Delayed/Live badge to curated archive");
    assert.strictEqual(s14.errorBannerVisible, false, "7.14: Error banner must not show");
    console.log("       [OK] Authentic backend curated fixture correctly classified as curated (40 disclosures, Curated badge, NEVER delayed)");

    // 7.15 Authentic Backend Unavailable Contract Fixture Consumption (include_curated=false)
    console.log("   7.15 Testing Authentic Backend Unavailable Contract Fixture (backendSmartMoneyOverviewUnavailable.json)...");
    const unavFixturePath = path.join(__dirname, "fixtures/backendSmartMoneyOverviewUnavailable.json");
    const unavFixture = JSON.parse(fs.readFileSync(unavFixturePath, "utf8"));

    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => unavFixture,
    } as any);

    const unavOverview = await fetchSmartMoneyOverview({ throwOnError: true });
    assert.strictEqual(unavOverview._dataSource, "unavailable", "7.15: Backend disconnected response must be classified as 'unavailable'");
    assert.strictEqual(unavOverview.status, "UNAVAILABLE", "7.15: Backend contract status must be 'UNAVAILABLE'");
    assert.strictEqual(unavOverview.is_live, false, "7.15: is_live must be explicitly false");
    assert.strictEqual(unavOverview.available, false, "7.15: available must be explicitly false");
    assert.strictEqual(classifySmartMoneyProvenance(unavFixture), "unavailable", "7.15: classifySmartMoneyProvenance must fail closed to 'unavailable'");

    const s15 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s15.feedError, true, "7.15: feedError must be true on disconnected live feed");
    assert.strictEqual(s15.dataSource, "unavailable", "7.15: Page dataSource must be 'unavailable'");
    assert.strictEqual(s15.displayedDisclosuresCount, "Unavailable", "7.15: Disclosures count must display Unavailable (not 0)");
    assert.strictEqual(s15.displayedSecCount, "Unavailable", "7.15: SEC count must display Unavailable (not 0)");
    assert.strictEqual(s15.sourceBadgeLabel, "⚠️ Feed Offline", "7.15: Must display Feed Offline badge");
    assert.strictEqual(s15.errorBannerVisible, true, "7.15: Feed error banner must be visible");
    console.log("       [OK] Authentic backend unavailable fixture fails closed to unavailable (Unavailable stats, Feed Offline badge)");

    // 7.16 Curated archive without legacy status fields (unannotated archive never promotes to live)
    console.log("   7.16 Testing Unannotated Legacy Archive (Never Promotes to Live)...");
    const legacyArchive = {
      congress_trades: [{ id: "leg-1", ticker: "AAPL" }, { id: "leg-2", ticker: "MSFT" }],
      sec_insider_trades: [{ id: "leg-sec", ticker: "NVDA" }],
    };
    assert.strictEqual(
      classifySmartMoneyProvenance(legacyArchive),
      "curated",
      "7.16: Unannotated observations must default to 'curated', NEVER 'live'"
    );

    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => legacyArchive,
    } as any);

    const s16 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s16.feedError, false);
    assert.strictEqual(s16.dataSource, "curated", "7.16: Must categorize as 'curated'");
    assert.strictEqual(s16.sourceBadgeLabel, "📚 Curated Research (Aug 2026)");
    assert.notStrictEqual(s16.sourceBadgeLabel, "⏱️ Regulatory Filings (Delayed)", "7.16: Unannotated archive must never bear live/delayed badge");
    console.log("       [OK] Unannotated legacy archive safely defaults to curated (never promoted to live)");

    // 7.17 Explicit verified-source response (status: LIVE, is_live: true)
    console.log("   7.17 Testing Explicit Verified Live Response (status=LIVE, is_live=true)...");
    const verifiedLiveResponse = {
      status: "LIVE",
      is_live: true,
      available: true,
      congress_trades: [{ id: "live-1", ticker: "GOOGL" }],
      sec_insider_trades: [{ id: "live-sec-1", ticker: "AMZN" }],
    };
    assert.strictEqual(classifySmartMoneyProvenance(verifiedLiveResponse), "live");

    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => verifiedLiveResponse,
    } as any);

    const s17 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s17.feedError, false);
    assert.strictEqual(s17.dataSource, "delayed");
    assert.strictEqual(s17.displayedDisclosuresCount, 1);
    assert.strictEqual(s17.displayedSecCount, 1);
    assert.strictEqual(s17.sourceBadgeLabel, "⏱️ Regulatory Filings (Delayed)");
    console.log("       [OK] Verified live response correctly receives delayed regulatory badge and authentic counts");

    // 7.18 Status-only response without usable observations (fails closed to unavailable)
    console.log("   7.18 Testing Status-Only / Metadata-Only Response (Fails Closed)...");
    const statusOnlyResponse = {
      status: "LIVE",
      is_live: true,
      regulatory_sources: {
        sec_edgar: "Official SEC Form 4 & 10-K Public API",
        finra_ats: "FINRA ATS Dark Pool Transparency Aggregation",
      },
      // Note: congress_trades, sec_insider_trades, options_flow are completely absent
    };
    assert.strictEqual(
      classifySmartMoneyProvenance(statusOnlyResponse),
      "unavailable",
      "7.18: Status-only response without observation arrays must fail closed to 'unavailable'"
    );

    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => statusOnlyResponse,
    } as any);

    const s18 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s18.feedError, true, "7.18: Missing observations must cause feedError=true");
    assert.strictEqual(s18.dataSource, "unavailable");
    assert.strictEqual(s18.displayedDisclosuresCount, "Unavailable");
    assert.strictEqual(s18.sourceBadgeLabel, "⚠️ Feed Offline");
    console.log("       [OK] Status-only response without observations fails closed to unavailable");

    // 7.19 Mixed feeds response (status: MIXED)
    console.log("   7.19 Testing Mixed Feeds Response (status=MIXED)...");
    const mixedResponse = {
      status: "MIXED",
      congress_trades: [{ id: "mix-1", ticker: "MSFT" }],
      sec_insider_trades: null,
      options_flow: null,
    };
    assert.strictEqual(classifySmartMoneyProvenance(mixedResponse), "mixed");

    global.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => mixedResponse,
    } as any);

    const s19 = await simulatePageLoadData(() => fetchSmartMoneyOverview({ throwOnError: true }));
    assert.strictEqual(s19.feedError, false);
    assert.strictEqual(s19.displayedDisclosuresCount, 1, "7.19: Available Congress feed displays count 1");
    assert.strictEqual(s19.displayedSecCount, "Unavailable", "7.19: Unavailable SEC feed displays Unavailable");
    console.log("       [OK] Mixed feed response correctly isolates available Congress feed (1) and unavailable SEC");

    // 7.20 Direct classifySmartMoneyProvenance Matrix Validation
    console.log("   7.20 Testing classifySmartMoneyProvenance Invariant Matrix...");
    assert.strictEqual(classifySmartMoneyProvenance(null), "unavailable", "Matrix: null -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance(undefined), "unavailable", "Matrix: undefined -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance("foo"), "unavailable", "Matrix: string -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance(42), "unavailable", "Matrix: number -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance([]), "unavailable", "Matrix: array -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance({}), "unavailable", "Matrix: empty object -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance({ available: false }), "unavailable", "Matrix: available:false -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance({ status: "DISCONNECTED" }), "unavailable", "Matrix: DISCONNECTED -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance({ status: "OFFLINE" }), "unavailable", "Matrix: OFFLINE -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance({ status: "PIPELINE_PENDING" }), "unavailable", "Matrix: PIPELINE_PENDING -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance({ status: "CURATED" }), "curated", "Matrix: CURATED -> curated");
    assert.strictEqual(classifySmartMoneyProvenance({ disclosure: "Curated August 2026" }), "curated", "Matrix: Curated disclosure -> curated");
    assert.strictEqual(classifySmartMoneyProvenance({ is_live: false, congress_trades: [] }), "curated", "Matrix: is_live:false -> curated");
    assert.strictEqual(classifySmartMoneyProvenance({ _dataSource: "fallback" }), "fallback", "Matrix: _dataSource:fallback -> fallback");
    assert.strictEqual(classifySmartMoneyProvenance({ _dataSource: "unavailable" }), "unavailable", "Matrix: _dataSource:unavailable -> unavailable");
    assert.strictEqual(classifySmartMoneyProvenance({ status: "LIVE", is_live: true, congress_trades: [] }), "live", "Matrix: live with empty array -> live");
    console.log("       [OK] classifySmartMoneyProvenance invariant matrix passes all 16 epistemic checks");

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
  console.log("   ALL SMART MONEY DISCLOSURE & FEED-INTEGRITY TESTS PASSED (22/22 GATES)    ");
  console.log("===============================================================================\n");
}).catch((err) => {
  console.error("FATAL RUNTIME TEST ERROR:", err);
  process.exit(1);
});
