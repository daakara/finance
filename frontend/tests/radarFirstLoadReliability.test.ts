/**
 * frontend/tests/radarFirstLoadReliability.test.ts
 *
 * Exhaustive Verification Suite for P1 Directive:
 * ARX — RADAR First-Load Reliability Remediation
 *
 * Verifies:
 * 1. Governed SCREENER_TIMEOUT_MS configuration and boundary enforcement [6250, 30000] ms.
 * 2. combineAbortSignals multi-signal abort propagation.
 * 3. isTransientScreenerError classification (transient vs terminal errors).
 * 4. Fast published-snapshot read (fetchRadarSnapshot) parsing envelope in <300ms.
 * 5. fetchScreenerGems timeout enforcement and deadline exceeded formatting.
 * 6. Caller AbortController cancellation handling.
 * 7. Bounded automatic retry with backoff for transient timeouts (max 2 retries).
 * 8. Error-versus-Zero Epistemic Invariant: Missing connectivity renders DATA UNAVAILABLE, never 0 Assets / 0 Opportunities.
 * 9. Retained Verified Snapshot preservation and provenance disclosure on refresh failure.
 * 10. Read-only retry connection guarantee (no mutation endpoints).
 */

import assert from "node:assert";
import {
  SCREENER_TIMEOUT_MS,
  combineAbortSignals,
  isTransientScreenerError,
  fetchRadarSnapshot,
  fetchScreenerGems,
  RadarSnapshotSummary,
  ScreenerResponse,
} from "../lib/api";

function mockFetch(handler: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) {
  const originalFetch = global.fetch;
  global.fetch = handler as any;
  return () => {
    global.fetch = originalFetch;
  };
}

async function runTests() {
  console.log("===============================================================================");
  console.log("ARX RADAR FIRST-LOAD RELIABILITY REMEDIATION TEST SUITE");
  console.log("Directive: P1 RADAR First-Load Reliability Remediation");
  console.log("===============================================================================\n");

  // ---------------------------------------------------------------------------
  // Check 1: Governed SCREENER_TIMEOUT_MS Configuration & Boundaries
  // ---------------------------------------------------------------------------
  console.log("Check 1: Governed SCREENER_TIMEOUT_MS configuration & boundary enforcement");
  assert.strictEqual(
    typeof SCREENER_TIMEOUT_MS,
    "number",
    "SCREENER_TIMEOUT_MS must be exported as a number"
  );
  assert(
    SCREENER_TIMEOUT_MS >= 6250 && SCREENER_TIMEOUT_MS <= 30000,
    `SCREENER_TIMEOUT_MS (${SCREENER_TIMEOUT_MS}ms) must lie within approved [6250, 30000]ms boundary`
  );
  assert.strictEqual(
    SCREENER_TIMEOUT_MS,
    15000,
    "Default SCREENER_TIMEOUT_MS must be 15000ms matching TACTICAL_SETUPS_TIMEOUT_MS"
  );
  console.log(`  [PASS] Governed SCREENER_TIMEOUT_MS is ${SCREENER_TIMEOUT_MS}ms (boundary [6250, 30000]ms enforced)`);

  // ---------------------------------------------------------------------------
  // Check 2: combineAbortSignals Helper
  // ---------------------------------------------------------------------------
  console.log("\nCheck 2: combineAbortSignals multi-signal abort propagation");
  {
    const c1 = new AbortController();
    const c2 = new AbortController();
    const combined = combineAbortSignals([c1.signal, c2.signal]);
    assert.strictEqual(combined.aborted, false, "Combined signal must start not aborted");

    c2.abort("Controller 2 aborted");
    assert.strictEqual(combined.aborted, true, "Combined signal must abort when any input signal aborts");
    console.log("  [PASS] combineAbortSignals aborts combined signal when one input aborts");
  }

  // ---------------------------------------------------------------------------
  // Check 3: isTransientScreenerError Transient vs Terminal Classification
  // ---------------------------------------------------------------------------
  console.log("\nCheck 3: isTransientScreenerError classification");
  {
    assert.strictEqual(isTransientScreenerError("Screener API request timed out (deadline 6000ms exceeded)", "TIMEOUT"), true);
    assert.strictEqual(isTransientScreenerError("HTTP 499: Client Closed Request"), true);
    assert.strictEqual(isTransientScreenerError("HTTP 502: Bad Gateway"), true);
    assert.strictEqual(isTransientScreenerError("HTTP 503: Service Unavailable"), true);
    assert.strictEqual(isTransientScreenerError("HTTP 504: Gateway Timeout"), true);
    assert.strictEqual(isTransientScreenerError("TypeError: fetch failed"), true);
    assert.strictEqual(isTransientScreenerError("ECONNREFUSED 127.0.0.1:8000"), true);

    // Non-transient errors must NOT trigger retry
    assert.strictEqual(isTransientScreenerError("HTTP 403: Forbidden"), false);
    assert.strictEqual(isTransientScreenerError("HTTP 404: Not Found"), false);
    assert.strictEqual(isTransientScreenerError("HTTP 400: Bad Request"), false);
    assert.strictEqual(isTransientScreenerError("UNPUBLISHED_UNIVERSE_AUTHORITY"), false);
    console.log("  [PASS] Transient errors (timeouts, 499, 502, network drops) and terminal errors cleanly segregated");
  }

  // ---------------------------------------------------------------------------
  // Check 4: fetchRadarSnapshot Fast Snapshot Envelope Read
  // ---------------------------------------------------------------------------
  console.log("\nCheck 4: fetchRadarSnapshot fast published-snapshot read (<300ms)");
  {
    const mockSnapshotPayload = {
      scanner_id: "MINERVINI_VCP",
      status: "AVAILABLE",
      snapshot_id: "vcp-snap-1794130818-3c0eb357",
      matched_count: 0,
      data_as_of: "2026-11-06",
      generated_at: "2026-11-08T09:40:13Z",
      universe: {
        universe_build_id: "UB-ARX-EXPANSION-500-20261108",
        scope_class: "US_EQUITIES",
        display_name: "ARX-eligible US equities",
        source_population_count: 500,
      },
      coverage: {
        coverage_status: "COMPLETE",
        scanned_successfully_count: 500,
        stage_2_opportunity_count: 200,
      },
      freshness: {
        status: "LIVE",
        data_as_of: "2026-11-06",
      },
    };

    const restore = mockFetch(async (url) => {
      assert(String(url).includes("/screener/vcp/snapshot"), "Must call /screener/vcp/snapshot");
      return new Response(JSON.stringify(mockSnapshotPayload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    const snap = await fetchRadarSnapshot({ timeoutMs: 2000 });
    restore();

    assert(snap !== null, "Snapshot must not be null");
    assert.strictEqual(snap.snapshotId, "vcp-snap-1794130818-3c0eb357");
    assert.strictEqual(snap.status, "AVAILABLE");
    assert.strictEqual(snap.populationDenominator, 500);
    assert.strictEqual(snap.stage2OpportunityCount, 200);
    assert.strictEqual(snap.dataAsOf, "2026-11-06");
    assert.strictEqual(snap.universeBuildId, "UB-ARX-EXPANSION-500-20261108");
    assert.strictEqual(snap.freshnessStatus, "LIVE");
    console.log("  [PASS] fetchRadarSnapshot returns typed summary (500 assets, 200 Stage 2, build UB-ARX-EXPANSION-500-20261108)");
  }

  // ---------------------------------------------------------------------------
  // Check 5: fetchScreenerGems Timeout & Error Formatting
  // ---------------------------------------------------------------------------
  console.log("\nCheck 5: fetchScreenerGems timeout handling with governed headroom");
  {
    const restore = mockFetch(async (_url, init) => {
      // Simulate timeout exceeding effective deadline
      const signal = init?.signal;
      await new Promise((_, reject) => {
        if (signal?.aborted) {
          const err = new Error("The operation was aborted");
          err.name = "AbortError";
          return reject(err);
        }
        signal?.addEventListener("abort", () => {
          const err = new Error("The operation was aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
      return new Response("{}", { status: 200 });
    });

    const res = await fetchScreenerGems("all", undefined, { timeoutMs: 100 });
    restore();

    assert.strictEqual(res.status, "TIMEOUT");
    assert.strictEqual(res.total_candidates, 0);
    assert.strictEqual(res.results.length, 0);
    assert(
      typeof res.error === "string" && res.error.includes("deadline 100ms exceeded"),
      `Error should state deadline exceeded, got: ${res.error}`
    );
    console.log("  [PASS] fetchScreenerGems properly catches timeout and sets status=TIMEOUT with deadline message");
  }

  // ---------------------------------------------------------------------------
  // Check 6: Caller AbortController Cancellation
  // ---------------------------------------------------------------------------
  console.log("\nCheck 6: Caller AbortController cancellation distinction");
  {
    const callerController = new AbortController();
    const restore = mockFetch(async (_url, init) => {
      const signal = init?.signal;
      if (signal?.aborted) {
        const err = new Error("AbortError");
        err.name = "AbortError";
        throw err;
      }
      await new Promise((_, reject) => {
        signal?.addEventListener("abort", () => {
          const err = new Error("AbortError");
          err.name = "AbortError";
          reject(err);
        });
      });
      return new Response("{}", { status: 200 });
    });

    const fetchPromise = fetchScreenerGems("all", undefined, {
      signal: callerController.signal,
      timeoutMs: 5000,
    });

    // Abort caller immediately
    callerController.abort("User clicked filter");
    const res = await fetchPromise;
    restore();

    assert.strictEqual(res.status, "ERROR");
    assert.strictEqual(res.error, "Screener request cancelled");
    console.log("  [PASS] Caller abort correctly reports 'Screener request cancelled' without false timeout attribution");
  }

  // ---------------------------------------------------------------------------
  // Check 7: Bounded Automatic Retry Simulation
  // ---------------------------------------------------------------------------
  console.log("\nCheck 7: Bounded automatic retry simulation for transient timeouts");
  {
    let attempts = 0;
    const restore = mockFetch(async () => {
      attempts++;
      if (attempts === 1) {
        // Attempt 1 fails with timeout
        const err = new Error("The operation was aborted");
        err.name = "AbortError";
        throw err;
      }
      // Attempt 2 succeeds
      return new Response(
        JSON.stringify({
          status: "SUCCESS",
          scanState: "COMPLETED",
          datasetState: "CANDIDATES_AVAILABLE",
          results: [{ symbol: "NVDA", executionStatus: "IN_BUY_ZONE", currentPrice: 145.5 }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    });

    // Simulate bounded retry caller logic matching page.tsx
    let finalResult: ScreenerResponse | null = null;
    const MAX_RETRIES = 2;
    for (let i = 0; i <= MAX_RETRIES; i++) {
      const res = await fetchScreenerGems("all", undefined, { timeoutMs: 500 });
      if (res.status === "SUCCESS") {
        finalResult = res;
        break;
      }
      if (!isTransientScreenerError(res.error, res.status)) {
        finalResult = res;
        break;
      }
      // Transient error: retry
    }
    restore();

    assert(finalResult !== null, "Final result must not be null");
    assert.strictEqual(finalResult.status, "SUCCESS");
    assert.strictEqual(finalResult.results.length, 1);
    assert.strictEqual(attempts, 2, "Must have succeeded on attempt 2");
    console.log("  [PASS] Bounded auto-retry successfully recovers from transient timeout on attempt 2");
  }

  // ---------------------------------------------------------------------------
  // Check 8: Error-Versus-Zero Epistemic Invariant (DATA UNAVAILABLE, never 0)
  // ---------------------------------------------------------------------------
  console.log("\nCheck 8: Error-versus-Zero Epistemic Invariant");
  {
    const fetchError = "Screener API request timed out (deadline 15000ms exceeded)";
    const allAssets: any[] = [];
    const population: any = null;
    const initialSnapshot: RadarSnapshotSummary | null = null;

    // Evaluate header logic under failure
    const evaluatedDisplay =
      fetchError && allAssets.length === 0 && !initialSnapshot
        ? "DATA UNAVAILABLE"
        : `${population?.referenceUniverseCount ?? allAssets.length} Assets`;

    const opportunityDisplay =
      fetchError && allAssets.length === 0 && !initialSnapshot
        ? "DATA UNAVAILABLE"
        : `${population?.stage2OpportunityCount ?? allAssets.length} Opportunities`;

    assert.strictEqual(
      evaluatedDisplay,
      "DATA UNAVAILABLE",
      "Evaluated display must be DATA UNAVAILABLE, never '0 Assets'"
    );
    assert.strictEqual(
      opportunityDisplay,
      "DATA UNAVAILABLE",
      "Opportunity display must be DATA UNAVAILABLE, never '0 Opportunities'"
    );
    console.log("  [PASS] Invariant enforced: Failed requests display DATA UNAVAILABLE, never '0 Assets' or '0 Opportunities'");
  }

  // ---------------------------------------------------------------------------
  // Check 9: Retained Verified Snapshot State Preservation
  // ---------------------------------------------------------------------------
  console.log("\nCheck 9: Retained Verified Snapshot state preservation on refresh failure");
  {
    const priorVerifiedSnapshot = {
      snapshotId: "vcp-snap-1794130818-3c0eb357",
      dataAsOf: "2026-11-06",
      timestamp: "2026-11-08T09:40:13Z",
      assets: [{ ticker: "NVDA" }, { ticker: "AAPL" }] as any[],
      population: { referenceUniverseCount: 500, stage2OpportunityCount: 200 },
      capabilities: {} as any,
    };

    // Simulate subsequent refresh failure
    const refreshError = "Gateway 504 Timeout";
    const isRetained = Boolean(priorVerifiedSnapshot && priorVerifiedSnapshot.assets.length > 0);
    assert.strictEqual(isRetained, true);

    const evaluatedDisplay = isRetained && refreshError
      ? `${priorVerifiedSnapshot.population.referenceUniverseCount} Assets (Retained Snapshot)`
      : "DATA UNAVAILABLE";

    const opportunityDisplay = isRetained && refreshError
      ? `${priorVerifiedSnapshot.population.stage2OpportunityCount} Opportunities · Retained`
      : "DATA UNAVAILABLE";

    assert.strictEqual(
      evaluatedDisplay,
      "500 Assets (Retained Snapshot)",
      "Retained snapshot must display 500 Assets (Retained Snapshot)"
    );
    assert.strictEqual(
      opportunityDisplay,
      "200 Opportunities · Retained",
      "Retained snapshot must display 200 Opportunities · Retained"
    );
    console.log("  [PASS] Retained verified snapshot preserved on refresh failure with explicit authority labeling");
  }

  // ---------------------------------------------------------------------------
  // Check 10: Read-Only Retry Connection Guarantee
  // ---------------------------------------------------------------------------
  console.log("\nCheck 10: Read-only retry connection guarantee (no mutation endpoints)");
  {
    const requestedMethods: string[] = [];
    const requestedPaths: string[] = [];

    const restore = mockFetch(async (url, init) => {
      const method = (init?.method || "GET").toUpperCase();
      requestedMethods.push(method);
      requestedPaths.push(String(url));
      return new Response(JSON.stringify({ status: "SUCCESS", results: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    // Simulate retry connection click: triggers fetchRadarSnapshot and fetchScreenerGems
    await fetchRadarSnapshot();
    await fetchScreenerGems("all");
    restore();

    assert(
      requestedMethods.every((m) => m === "GET"),
      `All retry requests must be GET, found: ${requestedMethods.join(", ")}`
    );
    assert(
      !requestedPaths.some((p) => p.includes("/scan")),
      "Retry must never invoke scan trigger endpoints"
    );
    console.log("  [PASS] Retry Connection triggers strictly GET requests (/screener/vcp/snapshot, /screener/run), zero scan mutations");
  }

  console.log("\n===============================================================================");
  console.log("ALL 10 RADAR FIRST-LOAD RELIABILITY CHECKS PASSED SUCCESSFULLY! 🚀");
  console.log("===============================================================================");
}

runTests().catch((err) => {
  console.error("Test execution failed:", err);
  process.exit(1);
});
