import assert from "node:assert";
import { formatPriceObservationDate, GemCandidate } from "../lib/api";

console.log("Starting Radar Price Provenance & Formatting Verification Suite (P1-002 / F-02)...\n");

// ---------------------------------------------------------------------------
// 1. formatPriceObservationDate helper unit tests
// ---------------------------------------------------------------------------
console.log("1. Testing formatPriceObservationDate transformation across trading day contexts & basis authority...");

// 1.1 Valid DAILY_CLOSE basis and valid dates
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", "DAILY_CLOSE"),
  "Daily Close · Oct 22, 2026",
  "Valid completed-session close date with verified DAILY_CLOSE must format as 'Daily Close · Oct 22, 2026'"
);

assert.strictEqual(
  formatPriceObservationDate("2026-10-23", "DAILY_CLOSE"),
  "Daily Close · Oct 23, 2026",
  "Current-day close date with verified DAILY_CLOSE must format as 'Daily Close · Oct 23, 2026'"
);

assert.strictEqual(
  formatPriceObservationDate("2026-10-21", "DAILY_CLOSE"),
  "Daily Close · Oct 21, 2026",
  "Previous trading-day close date must format as 'Daily Close · Oct 21, 2026'"
);

assert.strictEqual(
  formatPriceObservationDate("2026-10-16", "DAILY_CLOSE"),
  "Daily Close · Oct 16, 2026",
  "Weekend/holiday close must preserve historical Friday date without synthesizing dates"
);

assert.strictEqual(
  formatPriceObservationDate("2026-07-15", "DAILY_CLOSE"),
  "Daily Close · Jul 15, 2026",
  "Stale historical price must format faithfully with verified DAILY_CLOSE basis"
);

// 1.2 Valid DAILY_CLOSE basis but missing date
assert.strictEqual(
  formatPriceObservationDate(null, "DAILY_CLOSE"),
  "Daily Close · Date Unverified",
  "Verified basis with null date must display 'Daily Close · Date Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate(undefined, "DAILY_CLOSE"),
  "Daily Close · Date Unverified",
  "Verified basis with undefined date must display 'Daily Close · Date Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate("", "DAILY_CLOSE"),
  "Daily Close · Date Unverified",
  "Verified basis with empty date must display 'Daily Close · Date Unverified'"
);

// 1.3 Null / missing basis with valid date (F-02: Never silently assume Daily Close)
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", null),
  "Price · Oct 22, 2026 (Basis Unverified)",
  "Null basis with valid date must disclose unverified basis"
);
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", undefined),
  "Price · Oct 22, 2026 (Basis Unverified)",
  "Undefined basis with valid date must disclose unverified basis"
);

// 1.4 Null / missing basis and null / missing date (F-02: Full unverified disclosure)
assert.strictEqual(
  formatPriceObservationDate(null, null),
  "Price · Date & Basis Unverified",
  "Null basis and null date must display 'Price · Date & Basis Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate(undefined, undefined),
  "Price · Date & Basis Unverified",
  "Undefined basis and undefined date must display 'Price · Date & Basis Unverified'"
);
assert.strictEqual(
  formatPriceObservationDate("", null),
  "Price · Date & Basis Unverified",
  "Empty date and null basis must display 'Price · Date & Basis Unverified'"
);

// 1.5 Malformed / invalid calendar date strings
assert.strictEqual(
  formatPriceObservationDate("2026-99-99", "DAILY_CLOSE"),
  "Daily Close · Invalid Date",
  "Malformed date string with verified basis must be rejected with 'Daily Close · Invalid Date'"
);
assert.strictEqual(
  formatPriceObservationDate("2026-99-99", null),
  "Price · Invalid Date",
  "Malformed date string with null basis must be rejected with 'Price · Invalid Date'"
);
assert.strictEqual(
  formatPriceObservationDate("2026-02-31", "DAILY_CLOSE"),
  "Daily Close · Invalid Date",
  "Impossible calendar date (Feb 31) must be rejected with 'Daily Close · Invalid Date'"
);

// 1.6 Custom / Unknown basis enums
assert.strictEqual(
  formatPriceObservationDate("2026-10-22", "UNKNOWN"),
  "Price · Oct 22, 2026 (Basis Unverified)",
  "Explicit UNKNOWN basis enum must not be emitted as basis label"
);

console.log("   [PASS] 1.1: All formatPriceObservationDate trading day contexts and basis authority verified");

// ---------------------------------------------------------------------------
// 2. Candidate serialization and priceBasis invariants
// ---------------------------------------------------------------------------
console.log("2. Testing candidate serialization and multi-asset price provenance parity...");

const candidatesBatch: GemCandidate[] = [
  {
    ticker: "LNTH",
    composite_score: 89,
    categories: ["VALUE_GARP"],
    categoryEvidence: { VALUE_GARP: "CRITERIA_MATCHED" },
    peg_ratio: 1.2,
    risk_rating: "Medium",
    investment_thesis: "FDA fast track compounder",
    primary_catalyst: "Oncology diagnostic acceleration",
    expert_model: "Peter Lynch GARP Compounder",
    current_price: 100.10,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-10-22",
  },
  {
    ticker: "NVDA",
    composite_score: 95,
    categories: ["VALUE_GARP"],
    categoryEvidence: { VALUE_GARP: "CRITERIA_MATCHED" },
    peg_ratio: 1.4,
    risk_rating: "Medium",
    investment_thesis: "Datacenter AI GPU dominance",
    primary_catalyst: "Blackwell ultra-scale deployment",
    expert_model: "High-Beta Growth Leader",
    current_price: 212.21,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-10-22",
  },
  {
    ticker: "STALE_ASSET",
    composite_score: 65,
    categories: ["VALUE_GARP"],
    categoryEvidence: { VALUE_GARP: "CRITERIA_MATCHED" },
    peg_ratio: 0.9,
    risk_rating: "High",
    investment_thesis: "Niche industrial supplier",
    primary_catalyst: "Restructuring",
    expert_model: null,
    current_price: 14.50,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-10-15", // Different observation date
  },
  {
    ticker: "UNPRICED_ASSET",
    composite_score: null,
    categories: [],
    categoryEvidence: {},
    peg_ratio: 0,
    risk_rating: "Unverified",
    investment_thesis: "No tape data",
    primary_catalyst: "Awaiting tape",
    expert_model: null,
    current_price: undefined,
    priceBasis: null,
    priceDate: null,
  },
];

// Verify different observation dates are preserved across candidates
const dates = candidatesBatch.map((c) => c.priceDate).filter(Boolean);
assert.strictEqual(dates[0], "2026-10-22");
assert.strictEqual(dates[2], "2026-10-15");
assert.notStrictEqual(dates[0], dates[2], "Distinct observation dates must not be conflated");

// Verify unpriced asset fails closed with truthful basis disclosure
const unpriced = candidatesBatch.find((c) => c.ticker === "UNPRICED_ASSET");
assert.ok(unpriced);
assert.strictEqual(unpriced.current_price, undefined);
assert.strictEqual(unpriced.priceBasis, null);
assert.strictEqual(unpriced.priceDate, null);
assert.strictEqual(
  formatPriceObservationDate(unpriced.priceDate, unpriced.priceBasis),
  "Price · Date & Basis Unverified"
);

console.log("   [PASS] 2.1: Multi-asset price observation date distinction and unpriced fail-closed verified");

console.log("\n===============================================================================");
console.log("ALL RADAR PRICE PROVENANCE TESTS PASSED! 🚀");
console.log("===============================================================================\n");
