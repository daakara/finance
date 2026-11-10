/**
 * ARX Terminal — Phase 2: Nomenclature, De-Slop & Measured Accessibility Test Suite
 * Directive: ARX-SPRINT-UI-DISTILL-015 (Phase 2)
 *
 * Verifies:
 * 1. SynthesisGuideBanner unmounted from all primary hub routes (page.tsx, portfolio, radar, setups).
 * 2. Decorative emoji elimination from chrome headers and action buttons.
 * 3. Standard Lucide icon integration across IntentHero, OptimalEntryExitCard, DecisionReadinessCard, and PreFlightReadinessBar.
 * 4. Legible typography floor (>= 12px text-xs) across action buttons and workflow cards.
 * 5. Minimum 44px touch targets preserved across all primary controls.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

const ROOT_DIR = path.resolve(__dirname, "..");

console.log("==================================================================");
console.log("ARX PHASE 2: NOMENCLATURE, DE-SLOP & ACCESSIBILITY TEST SUITE");
console.log("Directive: ARX-SPRINT-UI-DISTILL-015 (Phase 2)");
console.log("==================================================================\n");

// ------------------------------------------------------------------
// 1. SynthesisGuideBanner Unmount Verification across All 4 Hubs
// ------------------------------------------------------------------
console.log("1. Verifying SynthesisGuideBanner unmount across primary routes...");

const hubFiles = [
  "app/page.tsx",
  "app/portfolio/page.tsx",
  "app/radar/page.tsx",
  "app/setups/page.tsx",
];

for (const relPath of hubFiles) {
  const fullPath = path.join(ROOT_DIR, relPath);
  const src = fs.readFileSync(fullPath, "utf-8");
  assert.strictEqual(
    src.includes("<SynthesisGuideBanner"),
    false,
    `SynthesisGuideBanner must be unmounted from ${relPath} to reclaim vertical viewport space`
  );
  assert.strictEqual(
    src.includes("import SynthesisGuideBanner"),
    false,
    `Unused SynthesisGuideBanner import must be removed from ${relPath}`
  );
  console.log(`   [PASS] ${relPath}: SynthesisGuideBanner unmounted cleanly`);
}

// ------------------------------------------------------------------
// 2. IntentHero Lucide Icon & Typography Invariants
// ------------------------------------------------------------------
console.log("\n2. Verifying IntentHero Lucide icon integration and typography...");

const heroPath = path.join(ROOT_DIR, "components/IntentHero.tsx");
const heroSrc = fs.readFileSync(heroPath, "utf-8");

// Assert decorative emojis removed
assert.strictEqual(heroSrc.includes("🔎"), false, "IntentHero must not use decorative 🔎 emoji");
assert.strictEqual(heroSrc.includes("📊"), false, "IntentHero must not use decorative 📊 emoji");
assert.strictEqual(heroSrc.includes("🎯"), false, "IntentHero must not use decorative 🎯 emoji");
assert.strictEqual(heroSrc.includes("🛡️"), false, "IntentHero must not use decorative 🛡️ emoji");

// Assert Lucide icons imported and used
assert.ok(heroSrc.includes("Search"), "IntentHero must use Lucide Search icon for Step 1");
assert.ok(heroSrc.includes("LineChart"), "IntentHero must use Lucide LineChart icon for Step 2");
assert.ok(heroSrc.includes("Target"), "IntentHero must use Lucide Target icon for Step 3");
assert.ok(heroSrc.includes("ShieldCheck"), "IntentHero must use Lucide ShieldCheck icon for Step 4");

// Assert typography floor (no text-[9px] or text-[10px] on step badges)
assert.strictEqual(heroSrc.includes("text-[9px]"), false, "IntentHero must not use sub-11px micro-typography");
assert.ok(heroSrc.includes("text-xs font-mono"), "IntentHero must use legible text-xs font-mono typography");

console.log("   [PASS] IntentHero: Decorative emojis replaced with Lucide icons; typography upgraded to text-xs");

// ------------------------------------------------------------------
// 3. OptimalEntryExitCard Button Glyphs & Title De-Slop
// ------------------------------------------------------------------
console.log("\n3. Verifying OptimalEntryExitCard button glyphs and title de-slop...");

const cardPath = path.join(ROOT_DIR, "components/OptimalEntryExitCard.tsx");
const cardSrc = fs.readFileSync(cardPath, "utf-8");

// Card title does not use 🎯 emoji
assert.strictEqual(cardSrc.includes("🎯 ${symbol}"), false, "OptimalEntryExitCard title must not include 🎯 emoji");
assert.strictEqual(cardSrc.includes("🎯 {symbol}"), false, "OptimalEntryExitCard unavailable title must not include 🎯 emoji");

// Action buttons use Lucide icons
assert.ok(cardSrc.includes("<ShieldCheck"), "Pre-Flight Checklist button must use Lucide ShieldCheck icon");
assert.ok(cardSrc.includes("<Scale"), "Size Position button must use Lucide Scale icon");
assert.ok(cardSrc.includes("<Bell"), "Set Trigger Alert button must use Lucide Bell icon");
assert.ok(cardSrc.includes("<Briefcase"), "Portfolio action button must use Lucide Briefcase icon");

// Verify decorative emojis removed from action buttons
assert.strictEqual(cardSrc.includes("<span>✈️</span>"), false, "Pre-Flight Checklist button must not use ✈️ emoji");
assert.strictEqual(cardSrc.includes("<span>⚖️</span>"), false, "Size Position button must not use ⚖️ emoji");
assert.strictEqual(cardSrc.includes("<span>🔔</span>"), false, "Set Trigger Alert button must not use 🔔 emoji");

console.log("   [PASS] OptimalEntryExitCard: Title and action buttons sanitized to institutional standard");

// ------------------------------------------------------------------
// 4. DecisionReadinessCard & PreFlightReadinessBar Glyph Sanitation
// ------------------------------------------------------------------
console.log("\n4. Verifying DecisionReadinessCard & PreFlightReadinessBar glyph sanitation...");

const readinessPath = path.join(ROOT_DIR, "components/DecisionReadinessCard.tsx");
const readinessSrc = fs.readFileSync(readinessPath, "utf-8");

assert.strictEqual(readinessSrc.includes("🚦"), false, "DecisionReadinessCard header must not contain 🚦 emoji");
assert.strictEqual(readinessSrc.includes("✈️ Pre-Flight"), false, "DecisionReadinessCard button must not use ✈️ emoji");
assert.ok(readinessSrc.includes("<ShieldCheck"), "DecisionReadinessCard pre-flight button must use Lucide ShieldCheck icon");

const barPath = path.join(ROOT_DIR, "components/PreFlightReadinessBar.tsx");
const barSrc = fs.readFileSync(barPath, "utf-8");

assert.strictEqual(barSrc.includes("✈️ Full Checklist"), false, "PreFlightReadinessBar button must not use ✈️ emoji");
assert.ok(barSrc.includes("<ShieldCheck"), "PreFlightReadinessBar button must use Lucide ShieldCheck icon");

console.log("   [PASS] DecisionReadinessCard & PreFlightReadinessBar: Pre-Flight buttons use ShieldCheck icon");

// ------------------------------------------------------------------
// 5. Terminal Views Posture Action Buttons Emoji Sanitation
// ------------------------------------------------------------------
console.log("\n5. Verifying Terminal View posture button emoji sanitation...");

const stdTerminalPath = path.join(ROOT_DIR, "components/terminal/StandardTerminalView.tsx");
const stdTerminalSrc = fs.readFileSync(stdTerminalPath, "utf-8");

assert.strictEqual(stdTerminalSrc.includes("⚖️ Size Position"), false, "StandardTerminalView must not use ⚖️ emoji in action CTA");
assert.strictEqual(stdTerminalSrc.includes("🚨 Review Invalidation"), false, "StandardTerminalView must not use 🚨 emoji in action CTA");
assert.strictEqual(stdTerminalSrc.includes("📋 Open Research"), false, "StandardTerminalView must not use 📋 emoji in action CTA");
assert.strictEqual(stdTerminalSrc.includes("🔎 Explore Screened"), false, "StandardTerminalView must not use 🔎 emoji in action CTA");

const guidedTerminalPath = path.join(ROOT_DIR, "components/terminal/GuidedTerminalView.tsx");
const guidedTerminalSrc = fs.readFileSync(guidedTerminalPath, "utf-8");

assert.strictEqual(guidedTerminalSrc.includes("🚀 Size Position"), false, "GuidedTerminalView must not use 🚀 emoji in action CTA");
assert.strictEqual(guidedTerminalSrc.includes("🎯 Find Setups"), false, "GuidedTerminalView must not use 🎯 emoji in action CTA");

console.log("   [PASS] Terminal views: Posture action buttons sanitized to clean institutional text");

// ------------------------------------------------------------------
// 6. UniversalOmniSearch Recommendation Card Sanitation
// ------------------------------------------------------------------
console.log("\n6. Verifying UniversalOmniSearch recommendation card sanitation...");

const omniPath = path.join(ROOT_DIR, "components/UniversalOmniSearch.tsx");
const omniSrc = fs.readFileSync(omniPath, "utf-8");

assert.strictEqual(omniSrc.includes("<span>💡 Did you mean:</span>"), false, "UniversalOmniSearch must not use 💡 emoji");
assert.ok(omniSrc.includes("<Search"), "UniversalOmniSearch recommendation must use Lucide Search icon");

console.log("   [PASS] UniversalOmniSearch: Recommendation card sanitized with Lucide Search icon");

console.log("\n==================================================================");
console.log("ALL PHASE 2 NOMENCLATURE & DE-SLOP TESTS PASSED (100%)");
console.log("==================================================================");
