import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { PortfolioPosition } from "../lib/portfolio";
import { extractSafeApiError } from "../lib/api";

console.log("Starting Manual Holding Exit Contract & UI Test Suite (MANUAL-UI-01 - MANUAL-UI-09)...");

const portfolioPagePath = path.resolve(__dirname, "../app/portfolio/page.tsx");
const portfolioSource = fs.readFileSync(portfolioPagePath, "utf-8");
const portfolioLibPath = path.resolve(__dirname, "../lib/portfolio.ts");
const portfolioLibSource = fs.readFileSync(portfolioLibPath, "utf-8");

// ============================================================================
// MANUAL-UI-01: Manual FULL exit calls /api/v1/portfolio/holdings/{id}/exit
// ============================================================================
assert.ok(
  portfolioLibSource.includes("export async function exitManualHoldingViaApi("),
  "MANUAL-UI-01 FAIL: portfolio.ts must export exitManualHoldingViaApi"
);
assert.ok(
  portfolioLibSource.includes("/portfolio/holdings/${encodeURIComponent(holdingId)}/exit"),
  "MANUAL-UI-01 FAIL: exitManualHoldingViaApi must call /portfolio/holdings/${encodeURIComponent(holdingId)}/exit"
);
assert.ok(
  portfolioSource.includes("await exitManualHoldingViaApi("),
  "MANUAL-UI-01 FAIL: Portfolio page must call exitManualHoldingViaApi for manual holdings"
);
assert.ok(
  portfolioSource.includes('exitType: isFull ? "FULL" : "PARTIAL"'),
  "MANUAL-UI-01 FAIL: Portfolio page must pass exitType FULL when full closing"
);
console.log("[OK] MANUAL-UI-01: Manual FULL exit calls /api/v1/portfolio/holdings/{id}/exit with exitType FULL");

// ============================================================================
// MANUAL-UI-02: Manual PARTIAL exit calls /api/v1/portfolio/holdings/{id}/exit
// ============================================================================
assert.ok(
  portfolioSource.includes("shares: isFull ? null : sharesNum"),
  "MANUAL-UI-02 FAIL: Portfolio page must pass shares parameter for partial manual exit"
);
console.log("[OK] MANUAL-UI-02: Manual PARTIAL exit calls new endpoint with exitType PARTIAL and shares quantity");

// ============================================================================
// MANUAL-UI-03: Journal FULL still calls /journal/close
// ============================================================================
assert.ok(
  portfolioSource.includes("if (isJournalBacked) {"),
  "MANUAL-UI-03 FAIL: Portfolio page must branch on isJournalBacked"
);
assert.ok(
  portfolioSource.includes("await recordTradeClose({"),
  "MANUAL-UI-03 FAIL: Journal-backed full close must invoke recordTradeClose (/journal/close)"
);
console.log("[OK] MANUAL-UI-03: Journal FULL still calls recordTradeClose (/journal/close)");

// ============================================================================
// MANUAL-UI-04: Journal PARTIAL still calls /journal/exit
// ============================================================================
assert.ok(
  portfolioSource.includes("await recordTradeExit({"),
  "MANUAL-UI-04 FAIL: Journal-backed partial exit must invoke recordTradeExit (/journal/exit)"
);
console.log("[OK] MANUAL-UI-04: Journal PARTIAL still calls recordTradeExit (/journal/exit)");

// ============================================================================
// MANUAL-UI-05: Mixed position label identifies manual scope
// ============================================================================
assert.ok(
  portfolioSource.includes('isTargetMixed ? "Close Manual Portion" : "Full Close (100%)"'),
  "MANUAL-UI-05 FAIL: Full close label must identify manual scope when mixed"
);
assert.ok(
  portfolioSource.includes('isTargetMixed ? "Reduce Manual Portion" : "Partial Scale-Out"'),
  "MANUAL-UI-05 FAIL: Partial exit label must identify manual scope when mixed"
);
assert.ok(
  portfolioSource.includes("Mixed Position Scope:"),
  "MANUAL-UI-05 FAIL: Mixed position banner/notice must be present in exit modal"
);
console.log("[OK] MANUAL-UI-05: Mixed position label explicitly scopes exit to manual portion");

// ============================================================================
// MANUAL-UI-06: Successful mixed FULL returns portfolioStatus OPEN
// ============================================================================
assert.ok(
  portfolioSource.includes('if (manualRes.data?.portfolioStatus === "CLOSED") {'),
  "MANUAL-UI-06 FAIL: Portfolio page must branch on portfolioStatus CLOSED vs OPEN"
);
assert.ok(
  portfolioSource.includes("shares: manualRes.data?.totalSharesRemaining ?? Number((exitTargetPosition.shares - sharesNum).toFixed(6)),"),
  "MANUAL-UI-06 FAIL: Mixed full close must update remaining total shares when position remains OPEN"
);
console.log("[OK] MANUAL-UI-06: Successful mixed FULL updates remaining shares when portfolioStatus is OPEN");

// ============================================================================
// MANUAL-UI-07: Manual-only FULL returns portfolioStatus CLOSED
// ============================================================================
assert.ok(
  portfolioSource.includes("if (manualRes.data?.portfolioStatus === \"CLOSED\") {\n            await removePortfolioPosition(exitTargetPosition.symbol);"),
  "MANUAL-UI-07 FAIL: When portfolioStatus is CLOSED, position must be removed"
);
console.log("[OK] MANUAL-UI-07: Manual-only FULL removes position when portfolioStatus is CLOSED");

// ============================================================================
// MANUAL-UI-08: Safe domain errors displayed
// ============================================================================
const domainError400 = extractSafeApiError(400, {
  detail: "Manual shares remaining (10.0) is insufficient for exit quantity (15.0).",
});
assert.strictEqual(
  domainError400,
  "Manual shares remaining (10.0) is insufficient for exit quantity (15.0).",
  "MANUAL-UI-08 FAIL: Safe domain 400 error must be displayed"
);
const domainError409 = extractSafeApiError(409, {
  detail: "Idempotency conflict: key already used with different parameters.",
});
assert.strictEqual(
  domainError409,
  "Idempotency conflict: key already used with different parameters.",
  "MANUAL-UI-08 FAIL: Safe domain 409 error must be displayed"
);
console.log("[OK] MANUAL-UI-08: Safe domain errors (400, 409) are preserved for display in UI");

// ============================================================================
// MANUAL-UI-09: Unknown 5xx remains generic
// ============================================================================
const unknown500 = extractSafeApiError(500, {
  detail: "OperationalError: table portfolio_holding_exit_events has no column named foo",
});
assert.strictEqual(
  unknown500,
  "Failed to record exit. Server rejected or returned an error.",
  "MANUAL-UI-09 FAIL: Internal 500 error must be redacted to generic safe message"
);
console.log("[OK] MANUAL-UI-09: Unknown 5xx error detail redacted behind generic fallback message");

console.log("ALL MANUAL HOLDING EXIT UI CONTRACT TESTS (MANUAL-UI-01 - MANUAL-UI-09) PASSED SUCCESSFULLY!");
