import assert from "node:assert";
import { TACTICAL_SETUPS_TIMEOUT_MS, fetchTacticalSetups } from "../lib/api";

console.log("Starting Tactical Setups Timeout & Client Reliability Test Suite...");

// ── 1. Timeout Value & Specification Invariants ──────────────────────────────
assert.strictEqual(typeof TACTICAL_SETUPS_TIMEOUT_MS, "number", "TACTICAL_SETUPS_TIMEOUT_MS must be an exported number");
assert.strictEqual(Number.isFinite(TACTICAL_SETUPS_TIMEOUT_MS), true, "TACTICAL_SETUPS_TIMEOUT_MS must be finite");
assert.strictEqual(TACTICAL_SETUPS_TIMEOUT_MS > 0, true, "TACTICAL_SETUPS_TIMEOUT_MS must be positive");

// Section 2.3 & 3.1: Minimum headroom (cold P95 * 1.25 = 5000 * 1.25 = 6250ms)
assert.strictEqual(
  TACTICAL_SETUPS_TIMEOUT_MS >= 6250,
  true,
  `Timeout (${TACTICAL_SETUPS_TIMEOUT_MS}ms) must exceed cold response P95 headroom requirement (>= 6250ms)`
);

// Section 2.3 & 3.1: Maximum allowed bound (30000ms)
assert.strictEqual(
  TACTICAL_SETUPS_TIMEOUT_MS <= 30000,
  true,
  `Timeout (${TACTICAL_SETUPS_TIMEOUT_MS}ms) must not exceed maximum approved threshold (<= 30000ms)`
);

assert.strictEqual(TACTICAL_SETUPS_TIMEOUT_MS, 15000, "TACTICAL_SETUPS_TIMEOUT_MS must equal approved 15000ms");
console.log("[OK] Test 1: Explicit finite timeout specification verified (15,000ms within [6250, 30000])");

// ── 2. Mock Fetch Simulation: Successful Response ────────────────────────────
async function testSuccessfulFetch() {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      assert.strictEqual(init?.signal !== undefined, true, "Fetch must include AbortSignal");
      return new Response(JSON.stringify({
        userRole: "LONG_TERM",
        totalSetups: 2,
        setups: [
          { ticker: "LNTH", setupName: "VCP" },
          { ticker: "MEDP", setupName: "Breakout" },
        ],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as any;

    const data = await fetchTacticalSetups(["LNTH", "MEDP"], "LONG_TERM");
    assert.strictEqual(Array.isArray(data), true, "Expected array of setups");
    assert.strictEqual(data.length, 2, "Expected 2 setups");
    assert.strictEqual(data[0].ticker, "LNTH");
    console.log("[OK] Test 2: Successful fetch parses setups payload correctly");
  } finally {
    globalThis.fetch = originalFetch;
  }
}

// ── 3. Mock Fetch Simulation: Timeout Abort Handling ─────────────────────────
async function testTimeoutAbort() {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      // Simulate AbortError when signal triggers
      return new Promise((_, reject) => {
        const error = new Error("Fetch is aborted");
        error.name = "AbortError";
        reject(error);
      });
    }) as any;

    let caughtError: any = null;
    try {
      await fetchTacticalSetups(["LNTH"], "LONG_TERM");
    } catch (err) {
      caughtError = err;
    }

    assert.notStrictEqual(caughtError, null, "Expected error on abort");
    assert.strictEqual(caughtError.name, "AbortError", "Expected AbortError name");
    assert.strictEqual(caughtError.message, "Fetch is aborted", "Expected abort message preserved");
    console.log("[OK] Test 3: AbortError correctly propagates to caller without being swallowed");
  } finally {
    globalThis.fetch = originalFetch;
  }
}

// ── 4. Mock Fetch Simulation: Server Error Handling ──────────────────────────
async function testServerError() {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = (async () => {
      return new Response("Internal Server Error", {
        status: 500,
        statusText: "Internal Server Error",
      });
    }) as any;

    let caughtError: any = null;
    try {
      await fetchTacticalSetups(["LNTH"], "LONG_TERM");
    } catch (err) {
      caughtError = err;
    }

    assert.notStrictEqual(caughtError, null, "Expected error on 500 response");
    assert.strictEqual(caughtError.message.includes("500"), true, "Expected 500 status in error message");
    console.log("[OK] Test 4: HTTP 500 error propagation verified");
  } finally {
    globalThis.fetch = originalFetch;
  }
}

(async () => {
  await testSuccessfulFetch();
  await testTimeoutAbort();
  await testServerError();
  console.log("ALL TACTICAL SETUPS TIMEOUT TESTS PASSED SUCCESSFULLY!");
})();
