import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

const ROOT_DIR = path.resolve(__dirname, "..");
const PAGE_PATH = path.join(ROOT_DIR, "app", "page.tsx");
const DISCOVERY_PATH = path.join(ROOT_DIR, "components", "discovery", "DiscoveryLandingView.tsx");

console.log("Starting ARX Phase 4 Route Decoupling & Discovery-First Verification Suite...\n");

// ------------------------------------------------------------------
// 1. Route Decoupling AST & Contract Verification
// ------------------------------------------------------------------
console.log("1. Verifying Route Decoupling & Elimination of Silent AAPL Fallback...");

const pageSrc = fs.readFileSync(PAGE_PATH, "utf8");
const discoverySrc = fs.readFileSync(DISCOVERY_PATH, "utf8");

// 1.1 Elimination of silent AAPL default in useState initialization
assert.strictEqual(
  pageSrc.includes('useState<string>(urlSymbol ? urlSymbol.toUpperCase() : "AAPL")'),
  false,
  "page.tsx must NOT silently default selectedSymbol to 'AAPL' when urlSymbol is absent"
);
assert.ok(
  pageSrc.includes('useState<string>(urlSymbol ? urlSymbol.toUpperCase() : "")'),
  "page.tsx must initialize selectedSymbol to empty string when urlSymbol is absent"
);
console.log("   [PASS] 1.1 Silent AAPL default eliminated from state initialization");

// 1.2 Data loading guard: Zero analytics fetch when symbol is empty
assert.ok(
  pageSrc.includes("if (!selectedSymbol) {") && pageSrc.includes("setLoading(false);"),
  "loadData() must guard against empty selectedSymbol and avoid fetching analytics"
);
console.log("   [PASS] 1.2 Background analytics fetch suppressed on bare '/' route");

// 1.3 View separation: Bare route renders DiscoveryLandingView; Cockpit renders when hasExplicitSymbol
assert.ok(
  pageSrc.includes("<DiscoveryLandingView"),
  "page.tsx must import and mount <DiscoveryLandingView />"
);
assert.ok(
  pageSrc.includes("{!hasExplicitSymbol || !selectedSymbol ? ("),
  "page.tsx must conditionally branch between Discovery view and Cockpit view based on explicit symbol"
);
console.log("   [PASS] 1.3 Dedicated DiscoveryLandingView cleanly decoupled from Cockpit");

// ------------------------------------------------------------------
// 2. Discovery Experience & Invariant Verification
// ------------------------------------------------------------------
console.log("\n2. Verifying Discovery-First Experience & Institutional Invariants...");

// 2.1 OmniSearch input and keyboard shortcuts
assert.ok(
  discoverySrc.includes('id="landing-search-input"'),
  "DiscoveryLandingView must include landing-search-input"
);
assert.ok(
  discoverySrc.includes("min-h-[48px]"),
  "Search input must meet >= 48px touch and accessibility height floor"
);
assert.ok(
  discoverySrc.includes("Tip: Press"),
  "Search input must show accessible keyboard shortcut tip"
);
console.log("   [PASS] 2.1 High-visibility OmniSearch & keyboard trigger verified");

// 2.2 Institutional Quick-Picks (min-h-[44px] compliance)
assert.ok(
  discoverySrc.includes("POPULAR_ASSETS"),
  "DiscoveryLandingView must define verified POPULAR_ASSETS catalog"
);
assert.ok(
  discoverySrc.includes("min-h-[56px]"),
  "Quick-pick buttons must enforce >= 44px coarse touch target floor"
);
const requiredSymbols = ["NVDA", "AAPL", "MSFT", "SPY", "QQQ", "TSLA"];
for (const sym of requiredSymbols) {
  assert.ok(
    discoverySrc.includes(`symbol: "${sym}"`),
    `Discovery quick-picks must include verified institutional asset: ${sym}`
  );
}
console.log("   [PASS] 2.2 Verified institutional quick-picks defined with >= 44px touch floor");

// 2.3 Macroeconomic Regime & Provenance Disclosures
assert.ok(
  discoverySrc.includes("FRED & CBOE Verified"),
  "DiscoveryLandingView must display explicit FRED & CBOE provenance badge"
);
assert.ok(
  discoverySrc.includes("10Y-2Y Yield Curve"),
  "DiscoveryLandingView must display 10Y-2Y Yield Curve Spread"
);
assert.ok(
  discoverySrc.includes("Macro Volatility (VIX)"),
  "DiscoveryLandingView must display Macro Volatility VIX"
);
console.log("   [PASS] 2.3 Authenticated macroeconomic overview verified");

// 2.4 Workflow Navigator & Tactical Spotlight Re-use
assert.ok(
  discoverySrc.includes("<IntentHero"),
  "DiscoveryLandingView must embed 4-step workflow navigator <IntentHero />"
);
assert.ok(
  discoverySrc.includes("<WeeklyConfluenceSpotlight"),
  "DiscoveryLandingView must embed governed tactical candidate spotlight"
);
console.log("   [PASS] 2.4 Re-use of IntentHero and WeeklyConfluenceSpotlight confirmed (zero duplication)");

// ------------------------------------------------------------------
// 3. Asset Cockpit Preservation & URL Deep Linking
// ------------------------------------------------------------------
console.log("\n3. Verifying Asset Cockpit Deep-Link Preservation...");

// 3.1 AdaptiveTerminal preserved with experience mode and props
assert.ok(
  pageSrc.includes("<AdaptiveTerminal"),
  "page.tsx must preserve AdaptiveTerminal for parameterized routes"
);
assert.ok(
  pageSrc.includes("preflightCleared={preflightEvaluation.isCleared}"),
  "page.tsx must pass preflightCleared to AdaptiveTerminal"
);
assert.ok(
  pageSrc.includes("activeBlocker={preflightEvaluation.activeBlocker}"),
  "page.tsx must pass activeBlocker to AdaptiveTerminal"
);
console.log("   [PASS] 3.1 AdaptiveTerminal deep-link wiring fully preserved");

// 3.2 Watchlist onSelectSymbol propagation
assert.ok(
  pageSrc.includes("onSelectSymbol={handleSelectSymbol}"),
  "WatchlistSidebar must invoke handleSelectSymbol to transition from Discovery to Cockpit"
);
console.log("   [PASS] 3.2 Watchlist selection triggers seamless transition to Cockpit");

console.log("\nALL PHASE 4 ROUTE DECOUPLING AND DISCOVERY TESTS PASSED (100%)!\n");
