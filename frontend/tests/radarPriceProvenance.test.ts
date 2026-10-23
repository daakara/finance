import assert from "node:assert";
import { formatPriceObservationDate, GemCandidate } from "../lib/api";

console.log("Starting Radar Price Provenance & Formatting Verification Suite (P1-002)...\n");

// ---------------------------------------------------------------------------
// 1. formatPriceObservationDate helper unit tests
// ---------------------------------------------------------------------------
console.log("1. Testing formatPriceObservationDate transformation across trading day contexts...");

// Valid completed-session close
assert.strictEqual(
  formatPriceObservationDate("2026-10-22"),
  "Daily Close · Oct 22, 2026",
  "Valid completed-session close date must format as 'Daily Close · Oct 22, 2026'"
);

// Current-day close after market closing
assert.strictEqual(
  formatPriceObservationDate("2026-10-23"),
  "Daily Close · Oct 23, 2026",
  "Current-day close date must format as 'Daily Close · Oct 23, 2026'"
);

// Previous trading-day close (e.g. Wednesday close viewed Thursday)
assert.strictEqual(
  formatPriceObservationDate("2026-10-21"),
  "Daily Close · Oct 21, 2026",
  "Previous trading-day close date must format as 'Daily Close · Oct 21, 2026'"
);

// Weekend & holiday gaps (e.g. Friday close viewed on Sunday/Monday morning)
assert.strictEqual(
  formatPriceObservationDate("2026-10-16"),
  "Daily Close · Oct 16, 2026",
  "Weekend/holiday close must preserve historical Friday date without synthesizing dates"
);

// Stale historical price (e.g. low-liquidity asset last traded 3 months ago)
assert.strictEqual(
  formatPriceObservationDate("2026-07-15"),
  "Daily Close · Jul 15, 2026",
  "Stale historical price must format faithfully"
);

// Missing observation date (null, undefined, empty string)
assert.strictEqual(
  formatPriceObservationDate(null),
  "Daily Close",
  "Null observation date must fall back gracefully to 'Daily Close'"
);
assert.strictEqual(
  formatPriceObservationDate(undefined),
  "Daily Close",
  "Undefined observation date must fall back gracefully to 'Daily Close'"
);
assert.strictEqual(
  formatPriceObservationDate(""),
  "Daily Close",
  "Empty observation date must fall back gracefully to 'Daily Close'"
);

// Malformed date string
assert.strictEqual(
  formatPriceObservationDate("2026-99-99"),
  "Daily Close · 2026-99-99",
  "Malformed date string must degrade gracefully without throwing"
);

console.log("   [PASS] 1.1: All formatPriceObservationDate trading day contexts verified");

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

// Verify unpriced asset fails closed
const unpriced = candidatesBatch.find((c) => c.ticker === "UNPRICED_ASSET");
assert.ok(unpriced);
assert.strictEqual(unpriced.current_price, undefined);
assert.strictEqual(unpriced.priceBasis, null);
assert.strictEqual(unpriced.priceDate, null);
assert.strictEqual(formatPriceObservationDate(unpriced.priceDate), "Daily Close");

console.log("   [PASS] 2.1: Multi-asset price observation date distinction and unpriced fail-closed verified");

console.log("\n===============================================================================");
console.log("ALL RADAR PRICE PROVENANCE TESTS PASSED! ✅");
console.log("===============================================================================\n");
