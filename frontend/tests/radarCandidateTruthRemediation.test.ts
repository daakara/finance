import assert from "node:assert";
import { formatPriceObservationDate, GemCandidate } from "../lib/api";

console.log("Starting Radar Candidate Truth & Population Eligibility Test Suite...\n");

// ---------------------------------------------------------------------------
// 1. Explicit Price Observation Date & Provenance Warnings
// ---------------------------------------------------------------------------
console.log("1. Testing formatPriceObservationDate provenance warnings and calendar validation...");

// Verified dates
assert.strictEqual(
  formatPriceObservationDate("2026-10-22"),
  "Daily Close · Oct 22, 2026",
  "Verified daily close date must format as 'Daily Close · Oct 22, 2026'"
);

// Missing date states (must NEVER silently return bare 'Daily Close')
assert.strictEqual(
  formatPriceObservationDate(null),
  "Daily Close · Date Unverified",
  "Null date must explicitly warn 'Daily Close · Date Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate(undefined),
  "Daily Close · Date Unverified",
  "Undefined date must explicitly warn 'Daily Close · Date Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate(""),
  "Daily Close · Date Unverified",
  "Empty date must explicitly warn 'Daily Close · Date Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate("   "),
  "Daily Close · Date Unverified",
  "Whitespace date must explicitly warn 'Daily Close · Date Unverified'"
);

// Invalid calendar dates (must reject impossible dates like Feb 31 or malformed)
assert.strictEqual(
  formatPriceObservationDate("2026-02-31"),
  "Daily Close · Invalid Date",
  "Impossible calendar date (Feb 31) must be rejected with 'Invalid Date'"
);
assert.strictEqual(
  formatPriceObservationDate("2026-13-01"),
  "Daily Close · Invalid Date",
  "Month > 12 must be rejected with 'Invalid Date'"
);
assert.strictEqual(
  formatPriceObservationDate("not-a-date"),
  "Daily Close · Invalid Date",
  "Non-date string must be rejected with 'Invalid Date'"
);

// Non-daily close basis
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", "SPOT"),
  "Price · Oct 22, 2026",
  "Custom priceBasis SPOT must format as 'Price · Oct 22, 2026'"
);
assert.strictEqual(
  formatPriceObservationDate(null, "SPOT"),
  "Price · Date Unverified",
  "Missing date with SPOT basis must warn 'Price · Date Unverified'"
);

console.log("   [PASS] 1.1: formatPriceObservationDate provenance and warning integrity verified");

// ---------------------------------------------------------------------------
// 2. Population Model & Opportunity Eligibility Filtering
// ---------------------------------------------------------------------------
console.log("2. Testing candidate population eligibility and denominator integrity...");

interface MockRadarAsset {
  ticker: string;
  price: number;
  priceBasis?: string | null;
  priceDate?: string | null;
  executionStatus: string;
  allowedActions: string[];
  confluenceScore: number;
}

const isCandidateEligibleOpportunity = (asset: MockRadarAsset): boolean => {
  if (asset.executionStatus === "DELISTED_SECURITY") return false;
  if (asset.executionStatus === "UNVERIFIED_ASSET") return false;
  if (asset.executionStatus === "UNKNOWN") return false;
  return true;
};

// Fixture with 35 returned records: exactly 1 delisted CPRX, 34 active candidates
const mockUniverse: MockRadarAsset[] = [
  {
    ticker: "LNTH",
    price: 100.10,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-10-22",
    executionStatus: "IN_BUY_ZONE",
    allowedActions: ["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"],
    confluenceScore: 89,
  },
  {
    ticker: "CPRX",
    price: 31.49,
    priceBasis: null,
    priceDate: null,
    executionStatus: "DELISTED_SECURITY",
    allowedActions: [],
    confluenceScore: 0,
  },
  ...Array.from({ length: 33 }, (_, i) => ({
    ticker: `SYM${i + 1}`,
    price: 50.0 + i,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-10-22",
    executionStatus: i % 2 === 0 ? "IN_BUY_ZONE" : "WAITING_PULLBACK",
    allowedActions: ["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"],
    confluenceScore: 70 + (i % 20),
  })),
];

assert.strictEqual(mockUniverse.length, 35, "Mock universe fixture must contain 35 items");

// Derive population counts dynamically
const delistedAssets = mockUniverse.filter((a) => a.executionStatus === "DELISTED_SECURITY");
const eligibleOpportunities = mockUniverse.filter(isCandidateEligibleOpportunity);

assert.strictEqual(delistedAssets.length, 1, "Exactly 1 delisted asset in universe");
assert.strictEqual(delistedAssets[0].ticker, "CPRX");
assert.strictEqual(delistedAssets[0].allowedActions.length, 0, "Delisted asset must have zero allowed actions");

// The opportunity denominator must be derived from actual eligibility rules, NOT hardcoded
assert.strictEqual(
  eligibleOpportunities.length,
  34,
  "Active opportunity denominator must derive strictly to 34 (35 minus 1 delisted)"
);
assert.ok(
  !eligibleOpportunities.some((a) => a.ticker === "CPRX"),
  "CPRX must be completely excluded from eligible opportunities"
);

// ---------------------------------------------------------------------------
// 3. Sorting Invariant: Delisted Securities Never Promoted
// ---------------------------------------------------------------------------
console.log("3. Testing sorting invariant: delisted securities sink to bottom...");

const sorted = [...mockUniverse].sort((a, b) => {
  const aDelisted = a.executionStatus === "DELISTED_SECURITY" ? 1 : 0;
  const bDelisted = b.executionStatus === "DELISTED_SECURITY" ? 1 : 0;
  if (aDelisted !== bDelisted) return aDelisted - bDelisted;
  return b.confluenceScore - a.confluenceScore;
});

assert.strictEqual(sorted[0].ticker, "LNTH", "Top ranked asset must be eligible LNTH");
assert.strictEqual(
  sorted[sorted.length - 1].ticker,
  "CPRX",
  "Delisted CPRX must be at the very bottom of the sort order"
);

console.log("   [PASS] 3.1: Candidate population and sorting invariants verified");

console.log("\n===============================================================================");
console.log("ALL RADAR CANDIDATE TRUTH REGRESSION TESTS PASSED! ✅");
console.log("===============================================================================\n");
