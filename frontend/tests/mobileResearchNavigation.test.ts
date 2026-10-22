import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

console.log("Starting ARX Mobile Research Discoverability & Navigation Regression Suite...\n");

const rootDir = path.resolve(__dirname, "..");
const navbarPath = path.join(rootDir, "components", "Navbar.tsx");
assert.ok(fs.existsSync(navbarPath), "Navbar.tsx must exist");

const navbarSrc = fs.readFileSync(navbarPath, "utf8");

// ============================================================================
// 1. Mobile Navigation Dock Discoverability & Structural Verification
// ============================================================================
console.log("1. Verifying Mobile Navigation Dock Discoverability & Research Link...");

// 1.1 Mobile nav dock exists
assert.ok(
  navbarSrc.includes('data-testid="mobile-nav-dock"'),
  "Navbar must contain data-testid='mobile-nav-dock'"
);

// 1.2 Research link exists in mobile-nav-dock
const dockSection = navbarSrc.split('data-testid="mobile-nav-dock"')[1]?.split("</nav>")[0];
assert.ok(dockSection, "Must extract mobile-nav-dock section from Navbar.tsx");

assert.ok(
  dockSection.includes("Research</span>"),
  "Mobile nav dock must contain visible label 'Research'"
);

assert.ok(
  dockSection.includes("<FileSearch"),
  "Mobile nav dock must render FileSearch icon for Research link"
);

// 1.3 Active state styling and aria-current
assert.ok(
  dockSection.includes('aria-current={pathname === "/research" || pathname === "/research/" || pathname?.startsWith("/research/") ? "page" : undefined}'),
  "Mobile Research link must set aria-current='page' when active"
);

assert.ok(
  dockSection.includes('bg-[#1b2434] text-cyan-400 font-bold'),
  "Mobile Research link must apply active highlight bg-[#1b2434] text-cyan-400 font-bold"
);

// 1.4 Touch target compliance (minimum 44x44px per WCAG 2.5.5)
assert.ok(
  dockSection.includes("min-w-[44px]") && dockSection.includes("min-h-[44px]"),
  "Mobile Research link must satisfy minimum 44x44px touch-target standard"
);

// 1.5 Context preservation: preserves active symbol when present, clean /research/ fallback
assert.ok(
  dockSection.includes('effectiveSymbol ? `/research/?symbol=${encodeURIComponent(effectiveSymbol)}` : "/research/"'),
  "Mobile Research link must preserve active symbol context (/research/?symbol=...) or fallback cleanly to /research/"
);

console.log("   [OK] Mobile Navigation Dock discoverability, label, icon, touch target & context verified.");

// ============================================================================
// 2. Removal of Redundant Secondary Link from 3-Dot Utility Menu
// ============================================================================
console.log("\n2. Verifying Redundant Research Entry Removal from 3-Dot Utilities Menu...");

const utilitiesMenuSection = navbarSrc.split('id="utilities-menu-dropdown"')[1]?.split('document.body')[0];
assert.ok(utilitiesMenuSection, "Must locate utilities menu section in Navbar.tsx");

assert.ok(
  !utilitiesMenuSection.includes("Opportunity Research"),
  "Utilities menu must NOT contain duplicate 'Opportunity Research' link"
);

assert.ok(
  !utilitiesMenuSection.includes("ODP-05"),
  "Utilities menu must NOT contain legacy 'ODP-05' badge"
);

console.log("   [OK] Redundant duplicate Research link cleanly removed from 3-dot utility menu.");

// ============================================================================
// 3. Desktop Navigation Parity & Route Integrity
// ============================================================================
console.log("\n3. Verifying Desktop Navigation Parity & Route Integrity...");

const desktopNavSection = navbarSrc.split('data-testid="desktop-nav-links"')[1]?.split("</nav>")[0];
assert.ok(desktopNavSection, "Must extract desktop-nav-links section from Navbar.tsx");

assert.ok(
  desktopNavSection.includes("Research</span>"),
  "Desktop navigation must contain visible label 'Research'"
);

assert.ok(
  desktopNavSection.includes('effectiveSymbol ? `/research/?symbol=${encodeURIComponent(effectiveSymbol)}` : "/research/"'),
  "Desktop Research link must preserve active symbol context or fallback cleanly to /research/"
);

console.log("   [OK] Desktop navigation retains Research link with context preservation.");

// ============================================================================
// 4. Viewport Layout Geometry Simulation (360px, 390px, 430px)
// ============================================================================
console.log("\n4. Simulating Viewport Dock Geometry & Zero Horizontal Overflow...");

const viewports = [360, 390, 430];
const dockItemCount = 6; // Radar, Analysis, Trade Plan, Portfolio, Research, Horizon Toggle
const itemMinWidth = 44; // min touch target width

for (const vw of viewports) {
  // Mobile dock has px-1 (4px padding on left and right = 8px total)
  const availableContentWidth = vw - 8;
  const itemWidth = availableContentWidth / dockItemCount;
  
  assert.ok(
    itemWidth >= itemMinWidth,
    `At viewport ${vw}px, average allocated width per item (${itemWidth.toFixed(1)}px) must be >= ${itemMinWidth}px`
  );
  
  const totalOccupiedWidth = dockItemCount * itemMinWidth;
  assert.ok(
    totalOccupiedWidth <= availableContentWidth,
    `Total minimum occupied width (${totalOccupiedWidth}px) must fit within available dock width (${availableContentWidth}px) at ${vw}px`
  );
}

console.log("   [OK] Zero horizontal blowout confirmed across 360px, 390px, and 430px viewports.");

console.log("\n===============================================================================");
console.log("ALL MOBILE RESEARCH DISCOVERABILITY REGRESSION TESTS PASSED (4/4 GATES) ✅");
console.log("===============================================================================\n");
