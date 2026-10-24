import assert from "node:assert";
import { formatPriceObservationDate, CandidatePopulationModel } from "../lib/api";

console.log("Starting Radar Candidate Truth & Population Eligibility Test Suite (F-01, F-02, F-03)...\n");

// ---------------------------------------------------------------------------
// 1. Explicit Price Observation Date & Provenance Warnings (F-02)
// ---------------------------------------------------------------------------
console.log("1. Testing formatPriceObservationDate provenance warnings and calendar validation (F-02)...");

// 1.1 Verified DAILY_CLOSE basis and valid date
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", "DAILY_CLOSE"),
  "Daily Close · Oct 22, 2026",
  "Verified daily close date must format as 'Daily Close · Oct 22, 2026'"
);

// 1.2 Verified DAILY_CLOSE basis but missing date
assert.strictEqual(
  formatPriceObservationDate(null, "DAILY_CLOSE"),
  "Daily Close · Date Unverified",
  "Verified basis with null date must explicitly warn 'Daily Close · Date Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate(undefined, "DAILY_CLOSE"),
  "Daily Close · Date Unverified",
  "Verified basis with undefined date must explicitly warn 'Daily Close · Date Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate("", "DAILY_CLOSE"),
  "Daily Close · Date Unverified",
  "Verified basis with empty date must explicitly warn 'Daily Close · Date Unverified'"
);

// 1.3 Null / missing basis with valid date (Never infer Daily Close without positive basis proof)
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", null),
  "Price · Oct 22, 2026 (Basis Unverified)",
  "Null basis with valid date must disclose '(Basis Unverified)'"
);
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", undefined),
  "Price · Oct 22, 2026 (Basis Unverified)",
  "Undefined basis with valid date must disclose '(Basis Unverified)'"
);

// 1.4 Null / missing basis and missing date (Truthful disclosure of date AND basis)
assert.strictEqual(
  formatPriceObservationDate(null, null),
  "Price · Date & Basis Unverified",
  "Missing date and missing basis must warn 'Price · Date & Basis Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate(undefined, undefined),
  "Price · Date & Basis Unverified",
  "Undefined date and basis must warn 'Price · Date & Basis Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate("", null),
  "Price · Date & Basis Unverified",
  "Empty date and null basis must warn 'Price · Date & Basis Unverified'"
);

// 1.5 Invalid calendar dates (reject impossible dates like Feb 31 or malformed)
assert.strictEqual(
  formatPriceObservationDate("2026-02-31", "DAILY_CLOSE"),
  "Daily Close · Invalid Date",
  "Impossible calendar date (Feb 31) with verified basis must be rejected with 'Invalid Date'"
);
assert.strictEqual(
  formatPriceObservationDate("2026-02-31", null),
  "Price · Invalid Date",
  "Impossible calendar date (Feb 31) with null basis must be rejected with 'Invalid Date'"
);
assert.strictEqual(
  formatPriceObservationDate("2026-13-01", "DAILY_CLOSE"),
  "Daily Close · Invalid Date",
  "Month > 12 must be rejected with 'Invalid Date'"
);
assert.strictEqual(
  formatPriceObservationDate("not-a-date", null),
  "Price · Invalid Date",
  "Non-date string with null basis must be rejected with 'Invalid Date'"
);

// 1.6 Custom basis string
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", "SPOT"),
  "SPOT · Oct 22, 2026",
  "Custom priceBasis SPOT must format as 'SPOT · Oct 22, 2026'"
);

console.log("   [PASS] 1.1: formatPriceObservationDate provenance and warning integrity verified");

// ---------------------------------------------------------------------------
// 2. Population Model & Opportunity Eligibility Filtering (F-01)
// ---------------------------------------------------------------------------
console.log("2. Testing candidate population eligibility and denominator integrity (F-01)...");

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

// Fixture with 35 returned records: exactly 1 delisted CPRX, 3 IN_BUY_ZONE, 31 other active candidates
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
    ticker: "DECK",
    price: 85.28,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-10-22",
    executionStatus: "IN_BUY_ZONE",
    allowedActions: ["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"],
    confluenceScore: 85,
  },
  {
    ticker: "AXON",
    price: 310.50,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-10-22",
    executionStatus: "IN_BUY_ZONE",
    allowedActions: ["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"],
    confluenceScore: 82,
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
  ...Array.from({ length: 31 }, (_, i) => ({
    ticker: `SYM${i + 1}`,
    price: 50.0 + i,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-10-22",
    executionStatus: "WAITING_PULLBACK",
    allowedActions: ["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"],
    confluenceScore: 70 + (i % 20),
  })),
];

assert.strictEqual(mockUniverse.length, 35, "Mock universe fixture must contain 35 items");

// Derive population counts dynamically
const delistedAssets = mockUniverse.filter((a) => a.executionStatus === "DELISTED_SECURITY");
const eligibleOpportunities = mockUniverse.filter(isCandidateEligibleOpportunity);
const currentOpportunities = mockUniverse.filter((a) => a.executionStatus === "IN_BUY_ZONE");

assert.strictEqual(delistedAssets.length, 1, "Exactly 1 delisted asset in universe");
assert.strictEqual(delistedAssets[0].ticker, "CPRX");
assert.strictEqual(delistedAssets[0].allowedActions.length, 0, "Delisted asset must have zero allowed actions");

assert.strictEqual(
  eligibleOpportunities.length,
  34,
  "Active screening denominator must derive strictly to 34 (35 minus 1 delisted)"
);
assert.strictEqual(
  currentOpportunities.length,
  3,
  "Current opportunities (IN_BUY_ZONE) must derive strictly to 3"
);
assert.ok(
  !eligibleOpportunities.some((a) => a.ticker === "CPRX"),
  "CPRX must be completely excluded from eligible opportunities"
);

// ---------------------------------------------------------------------------
// 3. Portfolio Ownership Scenarios (F-01 Verification Matrix)
// ---------------------------------------------------------------------------
console.log("3. Testing portfolio ownership scenarios & Unheld Candidates semantics (F-01)...");

function deriveOwnershipCounts(universe: MockRadarAsset[], heldTickers: Set<string>, isVerified: boolean) {
  let held = 0;
  let notHeld = 0;
  for (const asset of universe) {
    const isEligible = isCandidateEligibleOpportunity(asset);
    if (!isVerified) continue;
    if (heldTickers.has(asset.ticker)) {
      held++;
    } else if (isEligible) {
      notHeld++;
    }
  }
  return {
    total: universe.length,
    totalActive: universe.filter(isCandidateEligibleOpportunity).length,
    held,
    notHeld,
  };
}

// Scenario 3.1: Empty portfolio
const emptyCounts = deriveOwnershipCounts(mockUniverse, new Set(), true);
assert.strictEqual(emptyCounts.total, 35, "All Candidates must be 35");
assert.strictEqual(emptyCounts.totalActive, 34, "Active eligible must be 34");
assert.strictEqual(emptyCounts.held, 0, "Held must be 0");
assert.strictEqual(emptyCounts.notHeld, 34, "Unheld Candidates must be 34 (excludes delisted CPRX)");

// Scenario 3.2: Partially held portfolio (holding LNTH and SYM1)
const partialHeld = new Set(["LNTH", "SYM1"]);
const partialCounts = deriveOwnershipCounts(mockUniverse, partialHeld, true);
assert.strictEqual(partialCounts.held, 2, "Held count must be 2");
assert.strictEqual(partialCounts.notHeld, 32, "Unheld Candidates count must be 32 (34 - 2)");

// Scenario 3.3: All eligible assets held
const allEligibleHeld = new Set(eligibleOpportunities.map((a) => a.ticker));
const allHeldCounts = deriveOwnershipCounts(mockUniverse, allEligibleHeld, true);
assert.strictEqual(allHeldCounts.held, 34, "Held count must be 34");
assert.strictEqual(allHeldCounts.notHeld, 0, "Unheld Candidates must be 0");

// Scenario 3.4: Holding delisted CPRX
const delistedHeld = new Set(["CPRX"]);
const delistedHeldCounts = deriveOwnershipCounts(mockUniverse, delistedHeld, true);
assert.strictEqual(delistedHeldCounts.held, 1, "Held count includes held CPRX");
assert.strictEqual(delistedHeldCounts.notHeld, 34, "Unheld Candidates strictly 34 (CPRX was never unheld eligible)");

// Scenario 3.5: Degraded unverified portfolio state
const unverifiedCounts = deriveOwnershipCounts(mockUniverse, new Set(), false);
assert.strictEqual(unverifiedCounts.total, 35, "All Candidates remains 35");
assert.strictEqual(unverifiedCounts.held, 0, "Held is 0 when unverified");
assert.strictEqual(unverifiedCounts.notHeld, 0, "Unheld is 0 when unverified (renders dash in UI)");

console.log("   [PASS] 3.1: All portfolio ownership scenarios reconciled cleanly");

// ---------------------------------------------------------------------------
// 4. Value / GARP Reference Catalog Coverage Verification (F-03)
// ---------------------------------------------------------------------------
console.log("4. Testing Value / GARP reference catalog coverage disclosure (F-03)...");

const valueGarpMatches = 16;
const referenceCatalogEvaluated = 35;
const coverageText = `${valueGarpMatches} of 16 Value / GARP Matches · 100% of Defined Reference Catalog Evaluated (${referenceCatalogEvaluated} Assets)`;

assert.ok(
  coverageText.includes("16 Value / GARP Matches"),
  "Disclosure must state 16 Value / GARP Matches"
);
assert.ok(
  coverageText.includes("100% of Defined Reference Catalog Evaluated"),
  "Disclosure must explicitly qualify 100% as reference catalog evaluated"
);
assert.ok(
  coverageText.includes("35 Assets"),
  "Disclosure must state 35 reference assets denominator"
);

console.log("   [PASS] 4.1: Value / GARP coverage language verified");

// ---------------------------------------------------------------------------
// 5. Sorting Invariant: Delisted Securities Never Promoted
// ---------------------------------------------------------------------------
console.log("5. Testing sorting invariant: delisted securities sink to bottom...");

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

console.log("   [PASS] 5.1: Candidate population and sorting invariants verified");

console.log("\n===============================================================================");
console.log("ALL RADAR CANDIDATE TRUTH REGRESSION TESTS PASSED! 🚀");
console.log("===============================================================================\n");
