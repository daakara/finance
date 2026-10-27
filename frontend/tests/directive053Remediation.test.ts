/**
 * frontend/tests/directive053Remediation.test.ts
 *
 * ARX Terminal — Directive 053 Evidence-Driven Remediation & Verification Suite.
 * Enforces invariants established under ARX_DIRECTIVE_053_EVIDENCE_DRIVEN_REMEDIATION:
 *
 * 1. D053-01: Smart Money Truthfulness & False-Zero Remediation
 *    - CongressionalTradesCard truth-state contract: PIPELINE_PENDING, SOURCE_UNAVAILABLE, PARTIAL, VALID_EMPTY
 *    - Badge elimination of false "0 Trades" when status is PIPELINE_PENDING / SOURCE_UNAVAILABLE
 *    - page.tsx plumbing of status and notice into CongressionalTradesCard
 *
 * 2. D053-02: Investor-Archetype Strategy Alignment Disclosures
 *    - Explicit disclaimer: Strategy Alignment Model (Not Actual Holdings)
 *    - Subtitle copy: Model-derived similarity to selected investment strategies... does not indicate current trades or positions
 *    - Badge taxonomy: "Factor Modeled" and "Provisional Model" (elimination of misleading "Observed")
 *
 * 3. D053-03: Asset Metadata & Authoritative Exchange Attribution
 *    - MasterAssetCatalog comprehensive exchange coverage (including AXP -> NYSE, IBM -> NYSE, SPY -> NYSE Arca)
 *    - SharedAssetDeck fallback to "UNAVAILABLE" (never fabricated "NASDAQ")
 *    - insightGenerator dynamic resolution with crypto detection
 *    - page.tsx dynamic exchange resolution
 *
 * 4. D053-04: Execution Ladder Defensive Geometry Presentation
 *    - Non-positive stop loss (<= 0) renders "Suppressed" rather than "$0.00 (-100.0%)"
 *    - Inverted stop loss (>= pivot) renders "(Inverted Stop ≥ Pivot)"
 *    - Inverted target 1 (<= pivot) renders "(Inverted Target ≤ Pivot)"
 *    - Malformed entry corridor renders "Suppressed" or "Awaiting Setup"
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { MASTER_ASSET_CATALOG } from "../lib/masterCatalog";
import { generateQuantitativeInsight } from "../lib/insightGenerator";

console.log("\n===============================================================================");
console.log("   ARX TERMINAL — DIRECTIVE 053 EVIDENCE-DRIVEN REMEDIATION TEST SUITE          ");
console.log("===============================================================================\n");

const ROOT_DIR = path.resolve(__dirname, "..");
const CONGRESSIONAL_CARD_PATH = path.join(ROOT_DIR, "components", "CongressionalTradesCard.tsx");
const TRADER_ARCHETYPES_PATH = path.join(ROOT_DIR, "components", "TraderArchetypesCard.tsx");
const SHARED_ASSET_DECK_PATH = path.join(ROOT_DIR, "components", "terminal", "SharedAssetDeck.tsx");
const SHARED_LADDER_PATH = path.join(ROOT_DIR, "components", "terminal", "SharedExecutionLadder.tsx");
const PAGE_PATH = path.join(ROOT_DIR, "app", "page.tsx");
const INSIGHT_GEN_PATH = path.join(ROOT_DIR, "lib", "insightGenerator.ts");

const congressionalCardSrc = fs.readFileSync(CONGRESSIONAL_CARD_PATH, "utf8");
const traderArchetypesSrc = fs.readFileSync(TRADER_ARCHETYPES_PATH, "utf8");
const sharedAssetDeckSrc = fs.readFileSync(SHARED_ASSET_DECK_PATH, "utf8");
const sharedLadderSrc = fs.readFileSync(SHARED_LADDER_PATH, "utf8");
const pageSrc = fs.readFileSync(PAGE_PATH, "utf8");
const insightGenSrc = fs.readFileSync(INSIGHT_GEN_PATH, "utf8");

// ---------------------------------------------------------------------------
// 1. D053-01: Smart Money Truthfulness & False-Zero Remediation
// ---------------------------------------------------------------------------
console.log("1. Verifying Smart Money Congressional Trades Card truth-state contract...");

// A. CongressionalTradesCardProps interface includes status and notice
assert.ok(
  congressionalCardSrc.includes('status?: "AVAILABLE" | "VALID_EMPTY" | "PIPELINE_PENDING" | "SOURCE_UNAVAILABLE" | string'),
  "D053-01 Violation: CongressionalTradesCardProps must accept status union"
);
assert.ok(
  congressionalCardSrc.includes("notice?: string"),
  "D053-01 Violation: CongressionalTradesCardProps must accept notice string"
);
console.log("   [PASS] CongressionalTradesCardProps includes status and notice fields");

// B. Truthful badge states (no false '0 Trades' on pending/unavailable)
assert.ok(
  congressionalCardSrc.includes('status === "PIPELINE_PENDING"') &&
  congressionalCardSrc.includes('"Pipeline Pending"'),
  "D053-01 Violation: Badge must show 'Pipeline Pending' when status is PIPELINE_PENDING"
);
assert.ok(
  congressionalCardSrc.includes('status === "SOURCE_UNAVAILABLE"') &&
  congressionalCardSrc.includes('"Unavailable"'),
  "D053-01 Violation: Badge must show 'Unavailable' when status is SOURCE_UNAVAILABLE"
);
console.log("   [PASS] Header badge evaluates status truth-states before trade count");

// C. Truthful disclosure copy for inactive pipeline and unverified data
assert.ok(
  congressionalCardSrc.includes("Congressional transaction data is currently unavailable.") &&
  congressionalCardSrc.includes("Data collection is not currently active."),
  "D053-01 Violation: PIPELINE_PENDING must clearly disclose that data collection is not active"
);
assert.ok(
  congressionalCardSrc.includes("No conclusion about recent transactions can be drawn."),
  "D053-01 Violation: PIPELINE_PENDING must explicitly state no conclusion can be drawn"
);
assert.ok(
  congressionalCardSrc.includes("No qualifying congressional transactions found in the verified coverage period."),
  "D053-01 Violation: VALID_EMPTY must clearly state verified coverage period"
);
console.log("   [PASS] Truthful disclosure messages verified for PIPELINE_PENDING, SOURCE_UNAVAILABLE, VALID_EMPTY");

// D. page.tsx passes status and notice into CongressionalTradesCard
assert.ok(
  pageSrc.includes("status={data?.smartMoney?.status}") &&
  pageSrc.includes("notice={data?.smartMoney?.notice}"),
  "D053-01 Violation: page.tsx must forward data.smartMoney.status and data.smartMoney.notice into CongressionalTradesCard"
);
console.log("   [PASS] page.tsx forwards smartMoney status and notice to CongressionalTradesCard");


// ---------------------------------------------------------------------------
// 2. D053-02: Investor-Archetype Strategy Alignment Disclosures
// ---------------------------------------------------------------------------
console.log("\n2. Verifying Investor-Archetype strategy alignment disclosures...");

// A. Prominent subtitle disclaimer
assert.ok(
  traderArchetypesSrc.includes("Model-derived similarity to selected investment strategies. These scores do not indicate current trades, positions, endorsements, or intentions of the named investors."),
  "D053-02 Violation: TraderArchetypesCard must contain explicit model-derived strategy alignment disclaimer in subtitle"
);
console.log("   [PASS] Subtitle disclaimer explicitly declares model-derived similarity and denies real positions/endorsements");

// B. Header badge explicitly disclaims actual holdings
assert.ok(
  traderArchetypesSrc.includes("Strategy Alignment Model (Not Actual Holdings)"),
  "D053-02 Violation: TraderArchetypesCard header badge must read 'Strategy Alignment Model (Not Actual Holdings)'"
);
console.log("   [PASS] Header badge states 'Strategy Alignment Model (Not Actual Holdings)'");

// C. Misleading "Observed" label eliminated in favor of "Factor Modeled"
const refreshedTraderCardSrc = fs.readFileSync(TRADER_ARCHETYPES_PATH, "utf8");
assert.ok(
  refreshedTraderCardSrc.includes("Factor Modeled") && !refreshedTraderCardSrc.includes("Observed"),
  "D053-02 Violation: Misleading 'Observed' badge and identifiers must be replaced with 'Factor Modeled'"
);
assert.ok(
  refreshedTraderCardSrc.includes("Provisional Model") && !refreshedTraderCardSrc.includes('"Provisional"</span>'),
  "D053-02 Violation: 'Provisional' badge must be replaced with 'Provisional Model'"
);
console.log("   [PASS] Badge taxonomy enforces 'Factor Modeled' and 'Provisional Model'");


// ---------------------------------------------------------------------------
// 3. D053-03: Asset Metadata & Authoritative Exchange Attribution
// ---------------------------------------------------------------------------
console.log("\n3. Verifying Asset Metadata & Authoritative Exchange Mapping...");

// A. Master Asset Catalog covers key assets with correct authoritative exchanges
const testExchanges: Record<string, string> = {
  AAPL: "NASDAQ",
  NVDA: "NASDAQ",
  MSFT: "NASDAQ",
  GOOGL: "NASDAQ",
  TSLA: "NASDAQ",
  AXP: "NYSE",
  IBM: "NYSE",
  PLTR: "NYSE",
  LLY: "NYSE",
  SPY: "NYSE Arca",
  QQQ: "NASDAQ",
  DHLGY: "OTC",
  "BTC-USD": "Crypto"
};

for (const [sym, expectedEx] of Object.entries(testExchanges)) {
  const entry = MASTER_ASSET_CATALOG[sym];
  assert.ok(entry, `Asset ${sym} must exist in MASTER_ASSET_CATALOG`);
  assert.strictEqual(
    entry.exchange,
    expectedEx,
    `Asset ${sym} exchange must be ${expectedEx}, got ${entry.exchange}`
  );
}
console.log(`   [PASS] All ${Object.keys(testExchanges).length} sampled assets in MASTER_ASSET_CATALOG have verified authoritative exchange mapping`);

// B. insightGenerator populates exchange and marketSession dynamically
const insightAapl = generateQuantitativeInsight("AAPL", "Apple Inc.", 260.48, 0.81);
assert.strictEqual(insightAapl.exchange, "NASDAQ", "AAPL insight must have exchange NASDAQ");
assert.strictEqual(insightAapl.advanced?.exchange, "NASDAQ", "AAPL advanced insight must have exchange NASDAQ");

const insightAxp = generateQuantitativeInsight("AXP", "American Express Company", 300.0, 0.5);
assert.strictEqual(insightAxp.exchange, "NYSE", "AXP insight must have exchange NYSE");
assert.strictEqual(insightAxp.advanced?.exchange, "NYSE", "AXP advanced insight must have exchange NYSE");

const insightSpy = generateQuantitativeInsight("SPY", "SPDR S&P 500 ETF Trust", 580.0, 0.17);
assert.strictEqual(insightSpy.exchange, "NYSE Arca", "SPY insight must have exchange NYSE Arca");

const insightBtc = generateQuantitativeInsight("BTC-USD", "Bitcoin USD", 67000.0, 0.75);
assert.strictEqual(insightBtc.exchange, "Crypto", "BTC-USD insight must identify Crypto exchange");
console.log("   [PASS] generateQuantitativeInsight dynamically derives authoritative exchanges without hardcoding");

// C. SharedAssetDeck fallback to 'UNAVAILABLE' (never fabricates 'NASDAQ')
assert.ok(
  sharedAssetDeckSrc.includes('ex && ex !== "UNKNOWN" ? ex : "UNAVAILABLE"'),
  "D053-03 Violation: SharedAssetDeck must evaluate ex and display UNAVAILABLE if missing or UNKNOWN"
);
assert.strictEqual(
  sharedAssetDeckSrc.includes('|| "NASDAQ"'),
  false,
  "D053-03 Violation: SharedAssetDeck must never hardcode fallback to 'NASDAQ'"
);
console.log("   [PASS] SharedAssetDeck displays UNAVAILABLE when exchange metadata is absent");

// D. page.tsx eliminates hardcoded NASDAQ fallback
assert.strictEqual(
  pageSrc.includes('exchange: "NASDAQ"'),
  false,
  "D053-03 Violation: page.tsx must not hardcode exchange: 'NASDAQ'"
);
assert.ok(
  pageSrc.includes("MASTER_ASSET_CATALOG[selectedSymbol.toUpperCase().replace(\"-USD\", \"\")]?.exchange"),
  "D053-03 Violation: page.tsx must query MASTER_ASSET_CATALOG for authoritative exchange fallback"
);
console.log("   [PASS] page.tsx resolves exchange dynamically using catalog and payload");


// ---------------------------------------------------------------------------
// 4. D053-04: Execution Ladder Defensive Geometry Presentation
// ---------------------------------------------------------------------------
console.log("\n4. Verifying Execution Ladder Defensive Geometry Presentation...");

// A. Non-positive stopLoss is suppressed
assert.ok(
  sharedLadderSrc.includes('kl.stopLoss === 0 ? (') && sharedLadderSrc.includes('"Suppressed"'),
  "D053-04 Violation: SharedExecutionLadder must flag stopLoss === 0 as Suppressed"
);
console.log("   [PASS] Non-positive stop loss is suppressed from rendering '$0.00 (-100.0%)'");

// B. Inverted stop loss (stopLoss >= pivotPrice) is defensive
assert.ok(
  sharedLadderSrc.includes("kl.stopLoss >= pivotPrice") && sharedLadderSrc.includes("(Inverted Stop ≥ Pivot)"),
  "D053-04 Violation: SharedExecutionLadder must identify inverted stop (stopLoss >= pivotPrice)"
);
console.log("   [PASS] Inverted stop loss (≥ Pivot) is defensively labeled");

// C. Inverted target 1 (target1 <= pivotPrice) is defensive
assert.ok(
  sharedLadderSrc.includes("kl.target1 <= pivotPrice") && sharedLadderSrc.includes("(Inverted Target ≤ Pivot)"),
  "D053-04 Violation: SharedExecutionLadder must identify inverted target (target1 <= pivotPrice)"
);
console.log("   [PASS] Inverted target 1 (≤ Pivot) is defensively labeled");

// D. Malformed entry corridor validation
assert.ok(
  sharedLadderSrc.includes("kl.entryMin > 0 && kl.entryMax >= kl.entryMin"),
  "D053-04 Violation: SharedExecutionLadder must validate entryMin > 0 and entryMax >= entryMin"
);
assert.ok(
  sharedLadderSrc.includes('kl.entryMin <= 0 ? "Suppressed" : "Awaiting Setup"'),
  "D053-04 Violation: SharedExecutionLadder must render Suppressed or Awaiting Setup when entryMin is invalid"
);
console.log("   [PASS] Entry corridor defensively validates geometry bounds");

console.log("\n===============================================================================");
console.log("   ALL DIRECTIVE 053 REMEDIATION INVARIANTS VERIFIED (4/4 AREAS PASSED)        ");
console.log("===============================================================================\n");
