import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import React from "react";
// @ts-ignore
import { renderToString } from "react-dom/server";
import MiniSparkline from "../components/MiniSparkline";
import { buildHubHref } from "../lib/canonicalNav";

console.log("Starting ARX UX Phase 1 Regression Test Suite...");

// ============================================================================
// 1. D04: MiniSparkline Truthful Presentation (Zero Synthetic Generation)
// ============================================================================
console.log("1. Testing D04 MiniSparkline truthfulness and unavailable representation...");

// 1.1 Null data produces unavailable indicator, never synthetic curve
const nullHtml = renderToString(
  React.createElement(MiniSparkline, {
    data: undefined,
    basePrice: 150,
    changePct: 2.5,
  })
);
assert.ok(nullHtml.includes("—"), "Missing data must render explicit '—' presentation");
assert.ok(
  nullHtml.includes('aria-label="No historical series available"'),
  "Missing data must include accessible aria-label"
);
assert.ok(
  !nullHtml.includes("<svg"),
  "Missing data must NOT render an SVG with synthetic or flat line curve"
);

// 1.2 Single data point (< 2) produces unavailable indicator
const singlePointHtml = renderToString(
  React.createElement(MiniSparkline, {
    data: [150],
    basePrice: 150,
    changePct: 0,
  })
);
assert.ok(singlePointHtml.includes("—"), "Data series length < 2 must render '—'");
assert.ok(!singlePointHtml.includes("<svg"), "Data series length < 2 must NOT render SVG curve");

// 1.3 Genuine data series renders valid SVG path
const genuineHtml = renderToString(
  React.createElement(MiniSparkline, {
    data: [150, 152, 151, 155, 158],
    basePrice: 150,
    changePct: 5.3,
  })
);
assert.ok(genuineHtml.includes("<svg"), "Genuine data series >= 2 points must render SVG");
assert.ok(genuineHtml.includes("<path"), "Genuine data series must render SVG path");

console.log("   [OK] D04: Zero synthetic sparklines and truthful '—' presentation verified");

// ============================================================================
// 2. D05: Keybinding Ownership (Navbar owns Cmd+K, OmniSearch owns /)
// ============================================================================
console.log("2. Testing D05 Keybinding Ownership across Navbar and UniversalOmniSearch...");

const navbarSrc = fs.readFileSync(
  path.join(__dirname, "../components/Navbar.tsx"),
  "utf-8"
);
const omniSrc = fs.readFileSync(
  path.join(__dirname, "../components/UniversalOmniSearch.tsx"),
  "utf-8"
);

// Navbar owns Cmd+K / Ctrl+K and strictly NOT '/'
assert.ok(
  navbarSrc.includes('e.key.toLowerCase() === "k"'),
  "Navbar must handle Cmd+K / Ctrl+K palette trigger"
);
assert.ok(
  !navbarSrc.includes('else if (e.key === "/"'),
  "Navbar must NOT intercept '/' key (collision with UniversalOmniSearch prevented)"
);

// UniversalOmniSearch owns '/' and strictly NOT Cmd+K
assert.ok(
  omniSrc.includes('e.key === "/"'),
  "UniversalOmniSearch must handle '/' symbol search trigger"
);
assert.ok(
  !omniSrc.includes('e.key === "/" || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k")'),
  "UniversalOmniSearch must NOT intercept Cmd+K (collision with Navbar palette prevented)"
);
assert.ok(
  omniSrc.includes("isContentEditable"),
  "UniversalOmniSearch must guard against input in contentEditable elements"
);
assert.ok(
  omniSrc.includes("SELECT"),
  "UniversalOmniSearch must guard against input in SELECT elements"
);

console.log("   [OK] D05: Keybinding ownership separation (Cmd+K vs /) verified");

// ============================================================================
// 3. D06 & D03: Accessibility Semantics & Landmarks
// ============================================================================
console.log("3. Testing D06 & D03 Accessibility Semantics & Single <main> Landmark...");

const watchlistSrc = fs.readFileSync(
  path.join(__dirname, "../components/WatchlistSidebar.tsx"),
  "utf-8"
);
const heroSrc = fs.readFileSync(
  path.join(__dirname, "../components/IntentHero.tsx"),
  "utf-8"
);
const portfolioSrc = fs.readFileSync(
  path.join(__dirname, "../app/portfolio/page.tsx"),
  "utf-8"
);

// WatchlistSidebar uses button type="button"
assert.ok(
  watchlistSrc.includes('type="button"') && watchlistSrc.includes("onSelectSymbol"),
  "WatchlistSidebar row selection must use native button element with type='button'"
);

// IntentHero Step 2 uses button type="button"
assert.ok(
  heroSrc.includes('type="button"') && heroSrc.includes("setIsSearchOpen(true)"),
  "IntentHero Step 2 tile must use native button element with type='button'"
);

// Portfolio page must NOT contain duplicate nested <main>
const mainMatches = portfolioSrc.match(/<main[\s>]/g);
assert.strictEqual(
  mainMatches,
  null,
  "Portfolio page must NOT contain nested <main> landmark (TerminalShell provides the primary <main>)"
);
assert.ok(
  portfolioSrc.includes('role="region" aria-label="Portfolio Risk Ledger"'),
  "Portfolio page must use semantic <div role='region' aria-label='Portfolio Risk Ledger'>"
);

console.log("   [OK] D06 & D03: Accessible interactive controls and single landmark verified");

// ============================================================================
// 4. D08_BOUNDED & D09_BOUNDED: Navigation Sync & No Duplicate Cards
// ============================================================================
console.log("4. Testing D08 & D09 Navigation Sync & Duplicate Card Prevention...");

// D08_BOUNDED: buildHubHref helper preserves symbol context without pollution
const hubRadar = buildHubHref("radar", "NVDA");
assert.strictEqual(hubRadar, "/radar?q=NVDA", "Radar href maps symbol to search query q");

const hubSetups = buildHubHref("setups", "NVDA");
assert.strictEqual(hubSetups, "/setups?symbol=NVDA", "Setups href maps symbol to symbol query");

const hubPortfolio = buildHubHref("portfolio", "NVDA");
assert.strictEqual(hubPortfolio, "/portfolio?symbol=NVDA", "Portfolio href maps symbol to symbol query");

const hubClean = buildHubHref("radar", null);
assert.strictEqual(hubClean, "/radar", "Clean hub href contains no query string when symbol is null");

// D09_BOUNDED: app/page.tsx mounts OptimalEntryExitCard exactly ONCE
const appPageSrc = fs.readFileSync(
  path.join(__dirname, "../app/page.tsx"),
  "utf-8"
);
const cardMatches = appPageSrc.match(/<OptimalEntryExitCard[\s\S]*?\/>/g);
assert.ok(cardMatches, "OptimalEntryExitCard must be present in app/page.tsx");
assert.strictEqual(
  cardMatches.length,
  1,
  "OptimalEntryExitCard must be mounted exactly ONCE (Mount B duplicate removed)"
);

console.log("   [OK] D08 & D09: Symbol navigation context and single OptimalEntryExitCard mount verified");

// ============================================================================
// 5. D12: Portfolio Truthfulness
// ============================================================================
console.log("5. Testing D12 Portfolio Truthfulness...");

assert.ok(
  portfolioSrc.includes("Tracked Portfolio Equity"),
  "Portfolio must display truthful 'Tracked Portfolio Equity' label"
);
assert.ok(
  !portfolioSrc.includes("Total Net Worth"),
  "Misleading 'Total Net Worth' label must NOT be present in portfolio"
);

console.log("   [OK] D12: Truthful portfolio equity label verified");

console.log("\n=======================================================");
console.log("ALL ARX UX PHASE 1 REGRESSION TESTS PASSED (5/5 SUITES)!");
console.log("=======================================================\n");
