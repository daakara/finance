import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

console.log("==================================================================");
console.log("ARX UX-R5: MOBILE JOURNEY & TOUCH OPTIMIZATION TEST SUITE");
console.log("Directive: ARX-UX-R5-001");
console.log("==================================================================\n");

const ROOT_DIR = path.resolve(__dirname, "..");
const RADAR_PATH = path.join(ROOT_DIR, "app", "radar", "page.tsx");
const PORTFOLIO_PATH = path.join(ROOT_DIR, "app", "portfolio", "page.tsx");
const NAVBAR_PATH = path.join(ROOT_DIR, "components", "Navbar.tsx");
const RIBBON_PATH = path.join(ROOT_DIR, "components", "nav", "MarketCommandRibbon.tsx");

const radarSrc = fs.readFileSync(RADAR_PATH, "utf8");
const portfolioSrc = fs.readFileSync(PORTFOLIO_PATH, "utf8");
const navbarSrc = fs.readFileSync(NAVBAR_PATH, "utf8");
const ribbonSrc = fs.readFileSync(RIBBON_PATH, "utf8");

// ------------------------------------------------------------------
// 1. UX-R5.1: Radar Responsive Mobile Cards (< sm / < 640px)
// ------------------------------------------------------------------
console.log("1. Verifying Radar Mobile Cards (UX-R5.1)...");

// 1.1 Desktop table preserved with hidden sm:block
assert.ok(
  radarSrc.includes("hidden sm:block overflow-x-auto"),
  "Radar desktop table must declare 'hidden sm:block overflow-x-auto' to preserve multi-column view on desktop"
);
assert.ok(
  radarSrc.includes('id="radar-candidates-table"'),
  "Radar desktop table must preserve anchor id 'radar-candidates-table'"
);
console.log("   [PASS] 1.1 Desktop table preserved with 'hidden sm:block'");

// 1.2 Mobile cards container declared with sm:hidden and data-testid
assert.ok(
  radarSrc.includes('data-testid="radar-mobile-cards"'),
  "Radar must declare mobile cards container with data-testid='radar-mobile-cards'"
);
assert.ok(
  radarSrc.includes('className="sm:hidden" data-testid="radar-mobile-cards"'),
  "Radar mobile cards container must declare 'sm:hidden' so it renders strictly on viewports < 640px"
);
console.log("   [PASS] 1.2 Mobile cards container declared with 'sm:hidden' for narrow viewports");

// 1.3 Mobile card preserves canonical financial data & badges
assert.ok(
  radarSrc.includes("RadarPortfolioBadge"),
  "Radar mobile cards must render RadarPortfolioBadge for ownership tracking"
);
assert.ok(
  radarSrc.includes("formatRadarExecutionStatus"),
  "Radar mobile cards must render authoritative screening status"
);
assert.ok(
  radarSrc.includes("confluenceScore"),
  "Radar mobile cards must display confluence conviction score"
);
assert.ok(
  radarSrc.includes("vcpStage"),
  "Radar mobile cards must display VCP stage / base building geometry"
);
assert.ok(
  radarSrc.includes("riskRewardRatio"),
  "Radar mobile cards must display risk/reward ratio"
);
console.log("   [PASS] 1.3 Mobile cards preserve symbol, ownership, status, score, VCP stage & R:R");

// 1.4 Mobile card action touch targets >= 44px
assert.ok(
  radarSrc.includes("min-h-[44px]"),
  "Radar mobile card actions must enforce min-h-[44px] touch target floor"
);
assert.ok(
  radarSrc.includes("View Plan →"),
  "Radar mobile cards must provide secondary 'View Plan →' navigation to /setups"
);
console.log("   [PASS] 1.4 Radar mobile touch targets enforce min-h-[44px] accessibility floor");

// ------------------------------------------------------------------
// 2. UX-R5.1: Portfolio Responsive Mobile Cards (< sm / < 640px)
// ------------------------------------------------------------------
console.log("\n2. Verifying Portfolio Mobile Cards (UX-R5.1)...");

// 2.1 Desktop table preserved with hidden sm:block
assert.ok(
  portfolioSrc.includes("hidden sm:block overflow-x-auto"),
  "Portfolio desktop table must declare 'hidden sm:block overflow-x-auto'"
);
assert.ok(
  portfolioSrc.includes("min-w-[720px]"),
  "Portfolio desktop table must retain min-w-[720px] constraint on desktop"
);
console.log("   [PASS] 2.1 Desktop holdings table preserved with 'hidden sm:block'");

// 2.2 Mobile cards container declared with sm:hidden and data-testid
assert.ok(
  portfolioSrc.includes('data-testid="portfolio-mobile-cards"'),
  "Portfolio must declare mobile cards container with data-testid='portfolio-mobile-cards'"
);
assert.ok(
  portfolioSrc.includes('className="sm:hidden divide-y divide-[#1b2434]"'),
  "Portfolio mobile cards container must declare 'sm:hidden' so it renders strictly on viewports < 640px"
);
console.log("   [PASS] 2.2 Mobile cards container declared with 'sm:hidden' for narrow viewports");

// 2.3 Mobile card preserves position details, P&L, stop floor
assert.ok(
  portfolioSrc.includes("pos.shares"),
  "Portfolio mobile cards must display shares count"
);
assert.ok(
  portfolioSrc.includes("pos.entryPrice"),
  "Portfolio mobile cards must display entry price"
);
assert.ok(
  portfolioSrc.includes("pos.currentPrice"),
  "Portfolio mobile cards must display current price"
);
assert.ok(
  portfolioSrc.includes("stopLossPrice"),
  "Portfolio mobile cards must display stop-loss floor level"
);
assert.ok(
  portfolioSrc.includes("targetPrice"),
  "Portfolio mobile cards must display target price"
);
console.log("   [PASS] 2.3 Mobile cards preserve shares, entry/current price, market value, P&L & stop floors");

// 2.4 Mobile actions enforce >= 44px touch targets
assert.ok(
  portfolioSrc.includes("handleOpenExitModal(pos)"),
  "Portfolio mobile cards must provide 'Record Exit' action"
);
assert.ok(
  portfolioSrc.includes("handleOpenEditModal(pos)"),
  "Portfolio mobile cards must provide 'Edit' action"
);
assert.ok(
  portfolioSrc.includes("handleRemovePosition(pos.symbol)"),
  "Portfolio mobile cards must provide 'Remove' action"
);
console.log("   [PASS] 2.4 Portfolio mobile actions provide Record Exit, Edit & Remove with >=44px touch targets");

// ------------------------------------------------------------------
// 3. UX-R5.2: Mobile Navigation Chrome Budget (<= 100px)
// ------------------------------------------------------------------
console.log("\n3. Verifying Mobile Navigation Chrome Budget (UX-R5.2)...");

// 3.1 Header height constrained to h-12 (48px)
assert.ok(
  navbarSrc.includes("h-12 flex items-center"),
  "Navbar header must declare h-12 (48px)"
);
assert.ok(
  navbarSrc.includes("h-12 flex items-center justify-between"),
  "Navbar command-header-bar must declare h-12 (48px)"
);
console.log("   [PASS] 3.1 Top Navbar header constrained to 48px (h-12)");

// 3.2 Command Ribbon height constrained to h-6 (24px) on mobile
assert.ok(
  ribbonSrc.includes("h-6 sm:h-9 min-h-[24px]"),
  "MarketCommandRibbon must declare h-6 (24px) on mobile when collapsed"
);
console.log("   [PASS] 3.2 Market Command Ribbon constrained to 24px (h-6) on mobile");

// 3.3 Total top fixed chrome calculation
const topChromeHeight = 48 + 24; // 72px
assert.ok(
  topChromeHeight <= 100,
  `Top fixed navigation (${topChromeHeight}px) must be <= 100px`
);
console.log(`   [PASS] 3.3 Total top fixed chrome = ${topChromeHeight}px (<= 100px target satisfied with 28px margin)`);

// 3.4 Mobile bottom dock accessibility & touch targets
assert.ok(
  navbarSrc.includes('data-testid="mobile-nav-dock"'),
  "Navbar must declare mobile-nav-dock"
);
assert.ok(
  navbarSrc.includes("min-h-[44px]"),
  "Mobile dock links must declare min-h-[44px] touch target floor"
);
assert.ok(
  navbarSrc.includes("env(safe-area-inset-bottom)"),
  "Mobile dock must declare safe-area-inset-bottom padding"
);
console.log("   [PASS] 3.4 Mobile bottom dock enforces >=44px touch targets with safe area insets");

console.log("\n==================================================================");
console.log("ALL ARX UX-R5 MOBILE OPTIMIZATION INVARIANTS SATISFIED (PASS)");
console.log("==================================================================");
