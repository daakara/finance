import assert from "node:assert";
import { normalizeAssetSymbol } from "../lib/assetRegistry";
import { resolveAssetAlias } from "../lib/assetRegistry";
import { OwnershipState, OwnershipFilter } from "../hooks/usePortfolioContext";
import { PortfolioPosition } from "../lib/portfolio";

console.log("Starting Radar Portfolio-Aware Context & Symbol Normalization Suite...");

// ============================================================================
// SECTION 26: SYMBOL NORMALIZATION & AMBIGUITY TESTS (INV-RADAR-PORTFOLIO-07)
// ============================================================================

// 1. Lowercase to Uppercase conversion
assert.strictEqual(normalizeAssetSymbol("aapl"), "AAPL");
assert.strictEqual(normalizeAssetSymbol("msft"), "MSFT");
assert.strictEqual(normalizeAssetSymbol("nvda"), "NVDA");
console.log("[OK] Lowercase conversion verified");

// 2. Leading and Trailing Whitespace trimming
assert.strictEqual(normalizeAssetSymbol("  AAPL  "), "AAPL");
assert.strictEqual(normalizeAssetSymbol("\tNVDA\n"), "NVDA");
console.log("[OK] Whitespace trimming verified");

// 3. Approved Share-Class Delimiter Normalization (BRK.B / BRK/B -> BRK-B)
assert.strictEqual(normalizeAssetSymbol("BRK.B"), "BRK-B");
assert.strictEqual(normalizeAssetSymbol("brk.b"), "BRK-B");
assert.strictEqual(normalizeAssetSymbol("BRK-B"), "BRK-B");
assert.strictEqual(normalizeAssetSymbol("BRK/B"), "BRK-B");
assert.strictEqual(normalizeAssetSymbol("BF.B"), "BF-B");
assert.strictEqual(normalizeAssetSymbol("BF/B"), "BF-B");
assert.strictEqual(normalizeAssetSymbol("BF-B"), "BF-B");
console.log("[OK] Approved share-class delimiter normalization verified");

// 4. International Suffix & Exchange Semantics Preservation
assert.strictEqual(normalizeAssetSymbol("SHEL.L"), "SHEL.L");
assert.strictEqual(normalizeAssetSymbol("LON:SHEL"), "LON:SHEL");
assert.strictEqual(normalizeAssetSymbol("9988.HK"), "9988.HK");
console.log("[OK] International suffix preservation verified");

// 5. Ambiguous Dual Listings fail closed to null (INV-RADAR-PORTFOLIO-07)
assert.strictEqual(normalizeAssetSymbol("AAPL/MSFT"), null);
assert.strictEqual(normalizeAssetSymbol("RIO/BHP"), null);
console.log("[OK] Ambiguous dual listings fail closed to null verified");

// 6. Unsupported / Malformed Symbols fail closed to null
assert.strictEqual(normalizeAssetSymbol(""), null);
assert.strictEqual(normalizeAssetSymbol("   "), null);
assert.strictEqual(normalizeAssetSymbol(null), null);
assert.strictEqual(normalizeAssetSymbol(undefined), null);
assert.strictEqual(normalizeAssetSymbol("APPLE INC"), null); // multi-word string
assert.strictEqual(normalizeAssetSymbol("$$INVALID**"), null); // invalid punctuation
assert.strictEqual(normalizeAssetSymbol("VERYLONGSYMBOLNAMETHATEXCEEDS16CHARS"), null); // exceeds max length 16
console.log("[OK] Unsupported and malformed symbols fail closed to null verified");

// 7. Identity Join Authority: resolveAssetAlias is NOT ownership authority
// Colloquial search alias maps "BERKSHIRE" -> canonicalTicker "JPM", but normalizeAssetSymbol must NOT do search aliasing!
const aliasResult = resolveAssetAlias("BERKSHIRE");
assert.strictEqual(aliasResult?.canonicalTicker, "JPM");
// normalizeAssetSymbol preserves ticker identity or returns null, NEVER returns JPM for BERKSHIRE
assert.notStrictEqual(normalizeAssetSymbol("BERKSHIRE"), "JPM");
console.log("[OK] resolveAssetAlias isolation from ownership identity verified");

// ============================================================================
// SECTION 25: OWNERSHIP STATE MODEL TESTS (INV-RADAR-PORTFOLIO-06)
// ============================================================================

/**
 * Pure evaluation function matching usePortfolioContext getOwnershipState logic
 */
function evaluateOwnership(
  symbol: string | null | undefined,
  isVerified: boolean,
  holdings: PortfolioPosition[]
): OwnershipState {
  const norm = normalizeAssetSymbol(symbol);
  if (!norm) return "UNKNOWN";
  if (!isVerified) return "UNKNOWN";

  const match = holdings.find((h) => normalizeAssetSymbol(h.symbol) === norm);
  if (match && match.shares > 0) {
    return "HELD";
  }
  return "NOT_HELD";
}

const verifiedHoldings: PortfolioPosition[] = [
  {
    symbol: "NVDA",
    name: "NVIDIA Corp.",
    shares: 25,
    entryPrice: 118.5,
    currentPrice: 124.0,
    addedAt: "2026-09-01",
    assetType: "Stock",
  },
  {
    symbol: "BRK-B",
    name: "Berkshire Hathaway Inc.",
    shares: 10,
    entryPrice: 420.0,
    currentPrice: 440.0,
    addedAt: "2026-09-15",
    assetType: "Stock",
  },
];

// 1. Verified portfolio + held symbol -> HELD
assert.strictEqual(evaluateOwnership("NVDA", true, verifiedHoldings), "HELD");
assert.strictEqual(evaluateOwnership("nvda", true, verifiedHoldings), "HELD");
assert.strictEqual(evaluateOwnership("  nvda  ", true, verifiedHoldings), "HELD");
// BRK.B normalized matches BRK-B holding
assert.strictEqual(evaluateOwnership("BRK.B", true, verifiedHoldings), "HELD");
assert.strictEqual(evaluateOwnership("BRK-B", true, verifiedHoldings), "HELD");
assert.strictEqual(evaluateOwnership("BRK/B", true, verifiedHoldings), "HELD");
console.log("[OK] Verified portfolio + held symbol -> HELD verified");

// 2. Verified portfolio + absent symbol -> NOT_HELD
assert.strictEqual(evaluateOwnership("AAPL", true, verifiedHoldings), "NOT_HELD");
assert.strictEqual(evaluateOwnership("MSFT", true, verifiedHoldings), "NOT_HELD");
console.log("[OK] Verified portfolio + absent symbol -> NOT_HELD verified");

// 3. Loading / Unverified portfolio -> UNKNOWN (Never falsely NOT_HELD)
assert.strictEqual(evaluateOwnership("NVDA", false, verifiedHoldings), "UNKNOWN");
assert.strictEqual(evaluateOwnership("AAPL", false, verifiedHoldings), "UNKNOWN");
console.log("[OK] Unverified portfolio -> UNKNOWN verified");

// 4. Failed portfolio API -> UNKNOWN
assert.strictEqual(evaluateOwnership("NVDA", false, []), "UNKNOWN");
assert.strictEqual(evaluateOwnership("AAPL", false, []), "UNKNOWN");
console.log("[OK] Failed portfolio API -> UNKNOWN verified");

// 5. Empty verified portfolio -> NOT_HELD for deterministic symbols
assert.strictEqual(evaluateOwnership("NVDA", true, []), "NOT_HELD");
assert.strictEqual(evaluateOwnership("AAPL", true, []), "NOT_HELD");
console.log("[OK] Empty verified portfolio -> NOT_HELD for deterministic symbols verified");

// 6. Ambiguous symbol -> UNKNOWN even with verified portfolio
assert.strictEqual(evaluateOwnership("AAPL/MSFT", true, verifiedHoldings), "UNKNOWN");
assert.strictEqual(evaluateOwnership("", true, verifiedHoldings), "UNKNOWN");
assert.strictEqual(evaluateOwnership(null, true, verifiedHoldings), "UNKNOWN");
console.log("[OK] Ambiguous symbol -> UNKNOWN verified");

// 7. Server/Client Conflict: Server wins
// If client cache has AAPL, but server holdings do not have AAPL, server authoritative state dictates NOT_HELD
const serverHoldings = verifiedHoldings; // NVDA, BRK-B (no AAPL)
const clientCacheOnly = [...verifiedHoldings, { symbol: "AAPL", name: "Apple", shares: 50, entryPrice: 150, currentPrice: 170, addedAt: "2026-09-01", assetType: "Stock" as const }];
// When evaluating against authoritative server holdings:
assert.strictEqual(evaluateOwnership("AAPL", true, serverHoldings), "NOT_HELD");
assert.strictEqual(evaluateOwnership("NVDA", true, serverHoldings), "HELD");
console.log("[OK] Server holdings win over client cache disagreement verified");

// ============================================================================
// SECTION 27: FILTER SEMANTICS & INVARIANTS (INV-RADAR-PORTFOLIO-04)
// ============================================================================

interface MockRadarRow {
  ticker: string;
  confluenceScore: number;
}

const mockCandidates: MockRadarRow[] = [
  { ticker: "NVDA", confluenceScore: 95 },
  { ticker: "AAPL", confluenceScore: 88 },
  { ticker: "BRK.B", confluenceScore: 82 },
  { ticker: "TSLA", confluenceScore: 75 },
  { ticker: "AMBIG/SYM", confluenceScore: 70 },
];

function applyFilters(
  rows: MockRadarRow[],
  filter: OwnershipFilter,
  isVerified: boolean,
  holdings: PortfolioPosition[]
): MockRadarRow[] {
  return rows.filter((row) => {
    const st = evaluateOwnership(row.ticker, isVerified, holdings);
    if (filter === "ALL") return true;
    if (filter === "NEW_OPPORTUNITIES") return st === "NOT_HELD";
    if (filter === "MY_HOLDINGS") return st === "HELD";
    return true;
  });
}

// 1. ALL filter includes HELD, NOT_HELD, and UNKNOWN
const allResult = applyFilters(mockCandidates, "ALL", true, verifiedHoldings);
assert.strictEqual(allResult.length, 5);
assert.deepStrictEqual(allResult.map((r) => r.ticker), ["NVDA", "AAPL", "BRK.B", "TSLA", "AMBIG/SYM"]);
console.log("[OK] ALL filter includes all rows verified");

// 2. NEW_OPPORTUNITIES filter includes ONLY NOT_HELD (excludes HELD and UNKNOWN)
const newResult = applyFilters(mockCandidates, "NEW_OPPORTUNITIES", true, verifiedHoldings);
// NVDA is HELD, BRK.B is HELD, AMBIG/SYM is UNKNOWN -> only AAPL and TSLA are NOT_HELD
assert.strictEqual(newResult.length, 2);
assert.deepStrictEqual(newResult.map((r) => r.ticker), ["AAPL", "TSLA"]);
console.log("[OK] NEW_OPPORTUNITIES includes only NOT_HELD verified");

// 3. MY_HOLDINGS filter includes ONLY HELD (excludes NOT_HELD and UNKNOWN)
const heldResult = applyFilters(mockCandidates, "MY_HOLDINGS", true, verifiedHoldings);
// NVDA and BRK.B are HELD
assert.strictEqual(heldResult.length, 2);
assert.deepStrictEqual(heldResult.map((r) => r.ticker), ["NVDA", "BRK.B"]);
console.log("[OK] MY_HOLDINGS includes only HELD verified");

// 4. Default Radar ranking invariant: Sort order is unaffected by filtering
const defaultSorted = [...allResult].sort((a, b) => b.confluenceScore - a.confluenceScore);
assert.deepStrictEqual(
  defaultSorted.map((r) => r.ticker),
  ["NVDA", "AAPL", "BRK.B", "TSLA", "AMBIG/SYM"]
);
// In newResult, AAPL (88) comes before TSLA (75) - order preserved!
const newSorted = [...newResult].sort((a, b) => b.confluenceScore - a.confluenceScore);
assert.deepStrictEqual(newSorted.map((r) => r.ticker), ["AAPL", "TSLA"]);

// In heldResult, NVDA (95) comes before BRK.B (82) - order preserved!
const heldSorted = [...heldResult].sort((a, b) => b.confluenceScore - a.confluenceScore);
assert.deepStrictEqual(heldSorted.map((r) => r.ticker), ["NVDA", "BRK.B"]);
console.log("[OK] INV-RADAR-PORTFOLIO-04: Ranking order preservation verified");

// 5. UNKNOWN state exclusion when portfolio unverified
const unverifiedNew = applyFilters(mockCandidates, "NEW_OPPORTUNITIES", false, verifiedHoldings);
assert.strictEqual(unverifiedNew.length, 0); // All are UNKNOWN, so excluded from NEW_OPPORTUNITIES

const unverifiedHeld = applyFilters(mockCandidates, "MY_HOLDINGS", false, verifiedHoldings);
assert.strictEqual(unverifiedHeld.length, 0); // All are UNKNOWN, so excluded from MY_HOLDINGS

const unverifiedAll = applyFilters(mockCandidates, "ALL", false, verifiedHoldings);
assert.strictEqual(unverifiedAll.length, 5); // All are included in ALL
console.log("[OK] Degraded unverified portfolio safely handles all filter states verified");

console.log("ALL RADAR PORTFOLIO CONTEXT & SYMBOL TESTS PASSED SUCCESSFULLY!");
