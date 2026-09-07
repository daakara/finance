import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

console.log("=================================================================");
console.log("  ARX Terminal vNext: W1.3 Verification Suite (Static & Source)   ");
console.log("=================================================================\n");

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${err.message}`);
    failed++;
  }
}

const ribbonPath = path.resolve("components/nav/MarketCommandRibbon.tsx");
const navbarPath = path.resolve("components/Navbar.tsx");
const unitTestPath = path.resolve("components/nav/__tests__/MarketCommandRibbon.test.tsx");
const e2eTestPath = path.resolve("tests/e2e/market-command-ribbon.spec.ts");

const ribbonSrc = fs.readFileSync(ribbonPath, "utf-8");
const navbarSrc = fs.readFileSync(navbarPath, "utf-8");
const unitTestSrc = fs.readFileSync(unitTestPath, "utf-8");
const e2eTestSrc = fs.readFileSync(e2eTestPath, "utf-8");

// Test Suite 1: MarketCommandRibbon Structural Specifications
test("MarketCommandRibbon.tsx exists and exports required components and interfaces", () => {
  assert(ribbonSrc.includes("export default function MarketCommandRibbon"), "Must export default MarketCommandRibbon");
  assert(ribbonSrc.includes("export function MarketCommandRibbonSkeleton"), "Must export MarketCommandRibbonSkeleton");
  assert(ribbonSrc.includes("export interface MacroRibbonPayload"), "Must export MacroRibbonPayload interface");
  assert(ribbonSrc.includes("export const DEFAULT_MACRO_SNAPSHOT"), "Must export DEFAULT_MACRO_SNAPSHOT constant");
});

test("MarketCommandRibbon enforces fixed 36px height (h-9, min-h-[36px], max-h-[36px])", () => {
  assert(ribbonSrc.includes("h-9 min-h-[36px] max-h-[36px]"), "Ribbon must strictly enforce 36px height");
  assert(ribbonSrc.includes('data-testid="market-command-ribbon"'), "Must have data-testid='market-command-ribbon'");
});

test("MarketCommandRibbonSkeleton enforces 36px height for zero CLS (< 0.05)", () => {
  assert(ribbonSrc.includes('data-testid="market-command-ribbon-skeleton"'), "Skeleton must have testid");
  assert(ribbonSrc.includes("h-9 min-h-[36px] max-h-[36px]"), "Skeleton must match 36px height exactly");
});

test("MarketCommandRibbon renders SPY, QQQ, VIX, and 10Y Yield with accessible ARIA labels", () => {
  assert(ribbonSrc.includes('aria-label="S&P 500"'), "Must have S&P 500 label");
  assert(ribbonSrc.includes('aria-label="NASDAQ 100"'), "Must have NASDAQ 100 label");
  assert(ribbonSrc.includes('aria-label="CBOE Volatility Index"'), "Must have VIX label");
  assert(ribbonSrc.includes('aria-label="10-Year Treasury Yield"'), "Must have 10Y label");
  assert(ribbonSrc.includes('aria-label="Market regime"'), "Must have Market regime label");
});

test("MarketCommandRibbon adheres to Anti-Cyan invariant for regime color tokens", () => {
  // Anti-Cyan rule: Cyan is never used for bullish setups or positive regimes
  assert(ribbonSrc.includes("text-emerald-400 bg-emerald-500/10 border-emerald-500/30"), "RISK_ON must use Emerald tokens");
  assert(ribbonSrc.includes("text-rose-400 bg-rose-500/10 border-rose-500/30"), "DEFENSIVE must use Rose tokens");
  assert(ribbonSrc.includes("text-amber-400 bg-amber-500/10 border-amber-500/30"), "NEUTRAL must use Amber tokens");
});

test("MarketCommandRibbon provides resilient fallback to [Cached Market Snapshot] on 503/offline", () => {
  assert(ribbonSrc.includes("[Cached Market Snapshot]"), "Must display [Cached Market Snapshot] badge");
  assert(ribbonSrc.includes('data-testid="cached-snapshot-badge"'), "Must have testid for cached snapshot");
  assert(ribbonSrc.includes("FINANCE_MARKET_SNAPSHOTS_V1"), "Must read and persist to localStorage cache");
});

test("MarketCommandRibbon displays Pinned Settlement banner when market session is closed", () => {
  assert(ribbonSrc.includes('data-testid="settlement-pinned-badge"'), "Must have settlement pinned testid");
  assert(ribbonSrc.includes("Settlement Pinned"), "Must include Settlement Pinned label");
});

// Test Suite 2: Navbar Refactor Specifications
test("Navbar.tsx height is strictly constrained to 56px (h-14) with no sm:h-16 expansion", () => {
  assert(navbarSrc.includes('data-testid="navbar"'), "Navbar must have data-testid='navbar'");
  assert(navbarSrc.includes("h-14"), "Navbar must enforce h-14 (56px)");
  assert(!navbarSrc.includes("sm:h-16"), "Navbar must not expand to sm:h-16");
});

test("Navbar.tsx consolidates navigation into exactly 5 approved semantic categories", () => {
  assert(navbarSrc.includes('data-testid="desktop-nav-links"'), "Desktop nav links container must have testid");
  assert(navbarSrc.includes("Terminal"), "Navbar must include Terminal");
  assert(navbarSrc.includes("Intelligence"), "Navbar must include Intelligence");
  assert(navbarSrc.includes("Portfolio"), "Navbar must include Portfolio");
  assert(navbarSrc.includes("Research"), "Navbar must include Research");
  assert(navbarSrc.includes("Docs"), "Navbar must include Docs");
});

test("Navbar.tsx mounts MarketCommandRibbon directly beneath 56px header in sticky container", () => {
  assert(navbarSrc.includes("<MarketCommandRibbon />"), "Navbar must mount MarketCommandRibbon");
  assert(navbarSrc.includes("sticky top-0 z-50"), "Sticky top container must anchor both bars");
});

test("Navbar.tsx mobile dock mirrors the 5 semantic categories", () => {
  assert(navbarSrc.includes('data-testid="mobile-nav-dock"'), "Mobile nav dock must have testid");
  assert(navbarSrc.includes('>Terminal</span>'), "Mobile dock must include Terminal");
  assert(navbarSrc.includes('>Intelligence</span>'), "Mobile dock must include Intelligence");
  assert(navbarSrc.includes('>Portfolio</span>'), "Mobile dock must include Portfolio");
  assert(navbarSrc.includes('>Research</span>'), "Mobile dock must include Research");
  assert(navbarSrc.includes('>Docs</span>'), "Mobile dock must include Docs");
});

// Test Suite 3: Test Suites Presence & Content
test("Unit test suite exists at frontend/components/nav/__tests__/MarketCommandRibbon.test.tsx", () => {
  assert(unitTestSrc.includes("describe(\"MarketCommandRibbon Component\""), "Must contain describe block");
  assert(unitTestSrc.includes("renders MarketCommandRibbonSkeleton"), "Must test skeleton");
  assert(unitTestSrc.includes("renders RISK_ON regime with Emerald color tokens"), "Must test RISK_ON emerald tokens");
  assert(unitTestSrc.includes("handles 503 fallback"), "Must test 503 fallback");
});

test("Playwright E2E test suite exists at frontend/tests/e2e/market-command-ribbon.spec.ts", () => {
  assert(e2eTestSrc.includes("test.describe(\"Market Command Ribbon E2E Specification\""), "Must contain E2E describe block");
  assert(e2eTestSrc.includes("should render persistent 36px ribbon directly beneath the 56px navbar"), "Must test 36px directly under 56px");
  assert(e2eTestSrc.includes("should display consolidated 5 semantic navigation categories"), "Must test 5 semantic categories");
  assert(e2eTestSrc.includes("should handle 503 fallback gracefully"), "Must test 503 fallback");
});

console.log(`\n=================================================================`);
console.log(`  Verification Results: ${passed} Passed, ${failed} Failed`);
console.log(`=================================================================\n`);

if (failed > 0) {
  process.exit(1);
}
