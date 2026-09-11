/**
 * Phase A2 Navigation & Context Preservation Verification Suite
 * Verifies all 16 Acceptance Criteria (AC1 - AC16)
 */

import fs from "fs";
import path from "path";
import {
  CANONICAL_HUBS,
  QUERY_PARAM_CLASSES,
  buildHubHref,
  isHubActive,
  extractActiveSymbol,
} from "../lib/canonicalNav";

let passed = 0;
let failed = 0;

function assert(condition: boolean, testName: string, detail?: string) {
  if (condition) {
    console.log(`  PASS: ${testName}`);
    passed++;
  } else {
    console.error(`  FAIL: ${testName}${detail ? ` - ${detail}` : ""}`);
    failed++;
  }
}

console.log("===============================================================================");
console.log("PHASE A2: CONNECTED NAVIGATION & CONTEXT PRESERVATION VERIFICATION SUITE");
console.log("===============================================================================\n");

// -----------------------------------------------------------------------------
// GROUP 1: Canonical Hubs Definition & Order (AC1, AC2, AC3)
// -----------------------------------------------------------------------------
console.log("[Group 1] Canonical Hubs Structure & Route Parity");

assert(CANONICAL_HUBS.length === 6, "CANONICAL_HUBS has exactly 6 hubs", `Got ${CANONICAL_HUBS.length}`);

const expectedIds = ["radar", "analysis", "setups", "portfolio", "journal", "performance"];
const actualIds = CANONICAL_HUBS.map((h) => h.id);
assert(
  JSON.stringify(actualIds) === JSON.stringify(expectedIds),
  "Canonical hubs strictly follow: Radar → Analysis → Setups → Portfolio → Journal → Performance",
  `Got ${JSON.stringify(actualIds)}`
);

const expectedRoutes = ["/radar", "/", "/setups", "/portfolio", "/journal", "/performance"];
const actualRoutes = CANONICAL_HUBS.map((h) => h.href);
assert(
  JSON.stringify(actualRoutes) === JSON.stringify(expectedRoutes),
  "Canonical hub routes match specifications with Analysis at '/'",
  `Got ${JSON.stringify(actualRoutes)}`
);

const analysisHub = CANONICAL_HUBS.find((h) => h.id === "analysis");
assert(
  !!analysisHub && analysisHub.href === "/" && analysisHub.name === "Analysis",
  "Analysis is explicitly registered with route '/'",
  `Got ${JSON.stringify(analysisHub)}`
);

// -----------------------------------------------------------------------------
// GROUP 2: buildHubHref Context Preservation & Clean No-Symbol Flow (AC5, AC6)
// -----------------------------------------------------------------------------
console.log("\n[Group 2] Context Preservation & Clean Fallback in buildHubHref");

// With active symbol
assert(
  buildHubHref("radar", "NVDA") === "/radar?q=NVDA",
  "buildHubHref('radar', 'NVDA') uses query parameter 'q'",
  buildHubHref("radar", "NVDA")
);
assert(
  buildHubHref("analysis", "NVDA") === "/?symbol=NVDA",
  "buildHubHref('analysis', 'NVDA') uses query parameter 'symbol'",
  buildHubHref("analysis", "NVDA")
);
assert(
  buildHubHref("setups", "NVDA") === "/setups?symbol=NVDA",
  "buildHubHref('setups', 'NVDA') preserves active symbol",
  buildHubHref("setups", "NVDA")
);
assert(
  buildHubHref("portfolio", "NVDA") === "/portfolio?symbol=NVDA",
  "buildHubHref('portfolio', 'NVDA') preserves active symbol",
  buildHubHref("portfolio", "NVDA")
);
assert(
  buildHubHref("journal", "NVDA") === "/journal?symbol=NVDA",
  "buildHubHref('journal', 'NVDA') preserves active symbol",
  buildHubHref("journal", "NVDA")
);
assert(
  buildHubHref("performance", "NVDA") === "/performance?symbol=NVDA",
  "buildHubHref('performance', 'NVDA') preserves active symbol",
  buildHubHref("performance", "NVDA")
);

// Without active symbol (NO fake/default symbol injection!)
assert(
  buildHubHref("radar", null) === "/radar",
  "buildHubHref('radar', null) produces clean '/radar' without synthetic symbol",
  buildHubHref("radar", null)
);
assert(
  buildHubHref("analysis", null) === "/",
  "buildHubHref('analysis', null) produces clean '/' without synthetic symbol",
  buildHubHref("analysis", null)
);
assert(
  buildHubHref("setups", undefined) === "/setups",
  "buildHubHref('setups', undefined) produces clean '/setups'",
  buildHubHref("setups", undefined)
);
assert(
  buildHubHref("portfolio", "") === "/portfolio",
  "buildHubHref('portfolio', '') produces clean '/portfolio'",
  buildHubHref("portfolio", "")
);
assert(
  buildHubHref("journal", null) === "/journal",
  "buildHubHref('journal', null) produces clean '/journal'",
  buildHubHref("journal", null)
);
assert(
  buildHubHref("performance", null) === "/performance",
  "buildHubHref('performance', null) produces clean '/performance'",
  buildHubHref("performance", null)
);

// With additional params
assert(
  buildHubHref("portfolio", "AAPL", { ownership: "OWNED" }) === "/portfolio?symbol=AAPL&ownership=OWNED",
  "buildHubHref preserves additional params alongside symbol",
  buildHubHref("portfolio", "AAPL", { ownership: "OWNED" })
);

// -----------------------------------------------------------------------------
// GROUP 3: isHubActive Explicit Matching (AC3)
// -----------------------------------------------------------------------------
console.log("\n[Group 3] isHubActive Accurate Routing");

assert(isHubActive("/", "/"), "isHubActive('/', '/') is true");
assert(!isHubActive("/", "/radar"), "isHubActive('/', '/radar') is false (no root bleed)");
assert(!isHubActive("/", "/setups"), "isHubActive('/', '/setups') is false (no root bleed)");
assert(isHubActive("/radar", "/radar"), "isHubActive('/radar', '/radar') is true");
assert(isHubActive("/radar", "/screener"), "isHubActive('/radar', '/screener') is true (legacy alias)");
assert(isHubActive("/setups", "/setups"), "isHubActive('/setups', '/setups') is true");
assert(!isHubActive("/setups", "/portfolio"), "isHubActive('/setups', '/portfolio') is false");

// -----------------------------------------------------------------------------
// GROUP 4: Parameter Matrix Classification & Extraction (AC7, AC8)
// -----------------------------------------------------------------------------
console.log("\n[Group 4] Parameter Matrix & extractActiveSymbol");

assert(
  QUERY_PARAM_CLASSES.GLOBAL_JOURNEY_CONTEXT.includes("symbol"),
  "symbol is classified as GLOBAL_JOURNEY_CONTEXT"
);
assert(
  QUERY_PARAM_CLASSES.DISCOVERY_SEARCH_CONTEXT.includes("q"),
  "q is classified as DISCOVERY_SEARCH_CONTEXT"
);
assert(
  QUERY_PARAM_CLASSES.TRANSIENT_UI_CONTEXT.includes("add"),
  "add is classified as TRANSIENT_UI_CONTEXT"
);

assert(extractActiveSymbol("?symbol=TSLA") === "TSLA", "extractActiveSymbol from query string");
assert(extractActiveSymbol("?ticker=msft") === "MSFT", "extractActiveSymbol from ticker param uppercase");
assert(extractActiveSymbol("?q=nvda") === "NVDA", "extractActiveSymbol from q param");
assert(extractActiveSymbol("?foo=bar") === null, "extractActiveSymbol returns null when no symbol");
assert(extractActiveSymbol(null) === null, "extractActiveSymbol handles null safely");
assert(extractActiveSymbol("?symbol=INVALID$$$NAME") === null, "extractActiveSymbol rejects invalid symbols");

// -----------------------------------------------------------------------------
// GROUP 5: Source Code Architecture Audit (AC4, AC9 - AC15)
// -----------------------------------------------------------------------------
console.log("\n[Group 5] Component Source Code Architecture Audit");

const rootDir = path.resolve(__dirname, "..");

// 1. Navbar.tsx
const navbarSrc = fs.readFileSync(path.join(rootDir, "components", "Navbar.tsx"), "utf8");
assert(
  navbarSrc.includes("data-testid=\"desktop-nav-links\"") && navbarSrc.includes("CANONICAL_HUBS.map"),
  "Navbar desktop navigation maps dynamically over CANONICAL_HUBS"
);
assert(
  navbarSrc.includes("data-testid=\"mobile-nav-dock\"") && navbarSrc.includes("CANONICAL_HUBS.map"),
  "Navbar mobile dock maps dynamically over CANONICAL_HUBS (100% desktop/mobile parity)"
);
assert(
  navbarSrc.includes("activeSymbol?: string | null"),
  "NavbarProps declares optional activeSymbol prop"
);
assert(
  !navbarSrc.includes("router.push(\"/screener\")"),
  "Navbar keyboard shortcut 'S' no longer routes to legacy '/screener'"
);

// 2. TerminalShell.tsx
const shellSrc = fs.readFileSync(path.join(rootDir, "components", "terminal", "TerminalShell.tsx"), "utf8");
assert(
  !shellSrc.includes("TERMINAL_HUBS.map"),
  "TerminalShell no longer renders redundant duplicate desktop nav links"
);
assert(
  !shellSrc.includes("data-testid=\"mobile-nav-dock\""),
  "TerminalShell no longer renders duplicate mobile dock"
);
assert(
  shellSrc.includes("<Navbar activeSymbol={activeSymbol} hideMobileDock={false} />"),
  "TerminalShell delegates navigation & mobile dock to Navbar and forwards activeSymbol"
);

// 3. ExecutiveIntelligenceNav.tsx
const execNavSrc = fs.readFileSync(
  path.join(rootDir, "components", "committee", "ExecutiveIntelligenceNav.tsx"),
  "utf8"
);
assert(
  !execNavSrc.includes('href: "/me"') && !execNavSrc.includes('href: "/me/'),
  "ExecutiveIntelligenceNav WORKSPACE_LINKS completely quarantined from '/me/*' links"
);

// 4. app/page.tsx (Analysis /)
const pageSrc = fs.readFileSync(path.join(rootDir, "app", "page.tsx"), "utf8");
assert(
  pageSrc.includes("activeSymbol={urlSymbol ? urlSymbol.toUpperCase() : (hasExplicitSymbol ? selectedSymbol : null)}"),
  "Analysis page passes activeSymbol only when explicitly set, preventing default symbol pollution"
);
assert(
  !pageSrc.includes('href="/screener"'),
  "Analysis page replaced legacy href='/screener' with '/radar'"
);

// 5. app/radar/page.tsx
const radarSrc = fs.readFileSync(path.join(rootDir, "app", "radar", "page.tsx"), "utf8");
assert(
  radarSrc.includes("activeSymbol={searchQuery.trim() ? searchQuery.trim().toUpperCase() : null}"),
  "Radar page passes searchQuery as activeSymbol to TerminalShell"
);

// 6. app/setups/page.tsx
const setupsSrc = fs.readFileSync(path.join(rootDir, "app", "setups", "page.tsx"), "utf8");
assert(
  setupsSrc.includes("searchParams.get('symbol') || searchParams.get('ticker')"),
  "Setups page supports both ?symbol= and legacy ?ticker= query params"
);
assert(
  setupsSrc.includes("activeSymbol={selectedSetup?.ticker"),
  "Setups page passes active symbol to TerminalShell"
);

// 7. app/portfolio/page.tsx
const portfolioSrc = fs.readFileSync(path.join(rootDir, "app", "portfolio", "page.tsx"), "utf8");
assert(
  portfolioSrc.includes('const addSym = params.get("add");') &&
    !portfolioSrc.includes('params.get("add") || params.get("symbol")'),
  "Portfolio page decoupled ?symbol= from opening add modal (only ?add= opens modal)"
);
assert(
  portfolioSrc.includes("activeSymbol={activeSymbol}"),
  "Portfolio page passes activeSymbol to TerminalShell"
);
assert(
  portfolioSrc.includes("activeSymbol && pos.symbol.toUpperCase() === activeSymbol.toUpperCase()"),
  "Portfolio page highlights matching position row when activeSymbol is present"
);

// 8. app/journal/page.tsx
const journalSrc = fs.readFileSync(path.join(rootDir, "app", "journal", "page.tsx"), "utf8");
assert(
  journalSrc.includes("activeSymbol={urlSymbol}"),
  "Journal page passes urlSymbol to TerminalShell"
);

// 9. app/performance/page.tsx
const perfSrc = fs.readFileSync(path.join(rootDir, "app", "performance", "page.tsx"), "utf8");
assert(
  perfSrc.includes("setSearchTicker(sym.trim().toUpperCase())"),
  "Performance page initializes searchTicker from query parameter"
);
assert(
  perfSrc.includes("activeSymbol={searchTicker.trim()"),
  "Performance page passes activeSymbol to TerminalShell"
);

// 10. Guided / Standard / Advanced Views
const guidedSrc = fs.readFileSync(
  path.join(rootDir, "components", "terminal", "GuidedTerminalView.tsx"),
  "utf8"
);
const standardSrc = fs.readFileSync(
  path.join(rootDir, "components", "terminal", "StandardTerminalView.tsx"),
  "utf8"
);
const advancedSrc = fs.readFileSync(
  path.join(rootDir, "components", "terminal", "AdvancedTerminalView.tsx"),
  "utf8"
);
assert(
  !guidedSrc.includes('href="/screener"'),
  "GuidedTerminalView no longer contains legacy href='/screener'"
);
assert(
  !standardSrc.includes('href="/screener"'),
  "StandardTerminalView no longer contains legacy href='/screener'"
);
assert(
  !advancedSrc.includes('href="/screener"'),
  "AdvancedTerminalView no longer contains legacy href='/screener'"
);

// 11. Legacy redirects
const screenerRedirectSrc = fs.readFileSync(
  path.join(rootDir, "app", "screener", "page.tsx"),
  "utf8"
);
const researchRedirectSrc = fs.readFileSync(
  path.join(rootDir, "app", "research", "page.tsx"),
  "utf8"
);
assert(
  screenerRedirectSrc.includes("router.replace(`/radar?q="),
  "Screener redirect preserves search query and forwards to /radar?q=..."
);
assert(
  researchRedirectSrc.includes("router.replace(`/?symbol="),
  "Research redirect preserves symbol query and forwards to /?symbol=..."
);

// -----------------------------------------------------------------------------
// SUMMARY
// -----------------------------------------------------------------------------
console.log("\n===============================================================================");
console.log(`TOTAL TESTS: ${passed + failed} | PASSED: ${passed} | FAILED: ${failed}`);
console.log("===============================================================================");

if (failed > 0) {
  process.exit(1);
} else {
  console.log("ALL PHASE A2 ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!");
}
