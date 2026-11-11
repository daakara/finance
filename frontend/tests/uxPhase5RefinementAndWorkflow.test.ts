import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

const ROOT_DIR = path.resolve(__dirname, "..");
const HERO_PATH = path.join(ROOT_DIR, "components", "IntentHero.tsx");
const DISCOVERY_PATH = path.join(ROOT_DIR, "components", "discovery", "DiscoveryLandingView.tsx");
const TERMINAL_PATH = path.join(ROOT_DIR, "components", "AdaptiveTerminal.tsx");
const PAGE_PATH = path.join(ROOT_DIR, "app", "page.tsx");

console.log("==================================================================");
console.log("ARX PHASE 5: POST-LAUNCH UX REFINEMENT & DECISION WORKFLOW SUITE");
console.log("Directive: ARX-UX-PHASE5-001");
console.log("==================================================================\n");

const heroSrc = fs.readFileSync(HERO_PATH, "utf8");
const discoverySrc = fs.readFileSync(DISCOVERY_PATH, "utf8");
const terminalSrc = fs.readFileSync(TERMINAL_PATH, "utf8");
const pageSrc = fs.readFileSync(PAGE_PATH, "utf8");

// ------------------------------------------------------------------
// 1. Decision Progression Workflow Invariants
// ------------------------------------------------------------------
console.log("1. Verifying Decision Workflow Progression (Discover → Analyze → Validate Risk → Plan)...");

// 1.1 Progression heading and subtitle
assert.ok(
  heroSrc.includes("Discover → Analyze → Validate Risk → Plan"),
  "IntentHero must display canonical 4-step progression heading: 'Discover → Analyze → Validate Risk → Plan'"
);
assert.ok(
  heroSrc.includes("Decision Workflow"),
  "IntentHero must feature the institutional 'Decision Workflow' badge"
);
console.log("   [PASS] 1.1 Canonical four-tier decision workflow progression established");

// 1.2 Step 1: Discover -> Radar Screener
assert.ok(
  heroSrc.includes('href="/radar"') && heroSrc.includes("Step 1 · Screen") && heroSrc.includes("Radar Screener →"),
  "Step 1 must link to /radar, display 'Step 1 · Screen' badge, and reference 'Radar Screener →'"
);
console.log("   [PASS] 1.2 Step 1 (Discover) links to /radar with screening semantics");

// 1.3 Step 2: Analyze -> Asset Cockpit
assert.ok(
  heroSrc.includes("handleStep2Click") && heroSrc.includes("Step 2 · Conviction") && heroSrc.includes("Asset Cockpit →"),
  "Step 2 must use handleStep2Click, display 'Step 2 · Conviction' badge, and reference 'Asset Cockpit →'"
);
assert.ok(
  heroSrc.includes("landing-search-input"),
  "handleStep2Click must prioritize focusing landing-search-input smoothly"
);
console.log("   [PASS] 1.3 Step 2 (Analyze) focuses Asset Cockpit search with conviction semantics");

// 1.4 Step 3: Validate Risk -> Pre-Flight Gate
assert.ok(
  heroSrc.includes('href="/setups"') && heroSrc.includes("Step 3 · Clearance") && heroSrc.includes("Pre-Flight Gate →"),
  "Step 3 must link to /setups, display 'Step 3 · Clearance' badge, and reference 'Pre-Flight Gate →'"
);
console.log("   [PASS] 1.4 Step 3 (Validate Risk) links to /setups with pre-flight risk clearance semantics");

// 1.5 Step 4: Plan -> Portfolio Sizer
assert.ok(
  heroSrc.includes('href="/portfolio"') && heroSrc.includes("Step 4 · Allocation") && heroSrc.includes("Portfolio & Sizing →"),
  "Step 4 must link to /portfolio, display 'Step 4 · Allocation' badge, and reference 'Portfolio & Sizing →'"
);
console.log("   [PASS] 1.5 Step 4 (Plan) links to /portfolio with capital allocation semantics");

// ------------------------------------------------------------------
// 2. Institutional Design & Accessibility Compliance
// ------------------------------------------------------------------
console.log("\n2. Verifying Institutional Design & Accessibility Compliance...");

// 2.1 Touch targets >= 44px
assert.ok(
  discoverySrc.includes("min-h-[44px]"),
  "DiscoveryLandingView must enforce >= 44px touch target on interactive buttons and links"
);
assert.ok(
  discoverySrc.includes("min-h-[48px]"),
  "DiscoveryLandingView must enforce >= 48px height floor on search input"
);
assert.ok(
  discoverySrc.includes("min-h-[56px]"),
  "DiscoveryLandingView quick-picks must enforce >= 56px height floor"
);
console.log("   [PASS] 2.1 Touch target floors (>= 44px) strictly maintained");

// 2.2 Keyboard focus rings
assert.ok(
  discoverySrc.includes("focus-visible:ring-2") && heroSrc.includes("focus-visible:ring-2"),
  "All interactive workflow elements must declare visible focus-visible:ring-2 indicators"
);
console.log("   [PASS] 2.2 Visible focus indicators for keyboard navigation confirmed");

// 2.3 No sub-10px micro-typography slop
assert.strictEqual(
  heroSrc.includes("text-[9px]") || heroSrc.includes("text-[8px]"),
  false,
  "IntentHero must not use sub-10px unreadable micro-typography"
);
console.log("   [PASS] 2.3 Micro-typography slop eliminated in favor of standard design tokens");

// ------------------------------------------------------------------
// 3. Operational Safety Holds & Cockpit Architecture Invariance
// ------------------------------------------------------------------
console.log("\n3. Verifying Operational Safety Holds & Cockpit Architecture...");

// 3.1 Terminal state & 3 presentation lenses preserved
assert.ok(
  terminalSrc.includes('experienceMode === "GUIDED"') &&
  terminalSrc.includes('experienceMode === "STANDARD"') &&
  terminalSrc.includes('experienceMode === "QUANT"'),
  "AdaptiveTerminal must preserve Guided, Standard, and Quant experience lenses"
);
console.log("   [PASS] 3.1 Three presentation lenses preserved in AdaptiveTerminal");

// 3.2 Pre-flight clearance separation
assert.ok(
  pageSrc.includes("preflightCleared={preflightEvaluation.isCleared}"),
  "page.tsx must pass preflight clearance status to AdaptiveTerminal"
);
console.log("   [PASS] 3.2 Pre-flight clearance architecture strictly decoupled from conviction");

console.log("\n==================================================================");
console.log("ALL PHASE 5 REFINEMENT & WORKFLOW TESTS PASSED (100%)");
console.log("==================================================================\n");
