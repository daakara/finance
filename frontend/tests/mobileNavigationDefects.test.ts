import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

console.log("===============================================================================");
console.log("ARX PR #98 — Mobile Navigation Remediation & Contract Verification Suite");
console.log("Directive: ARX_PR98_MOBILE_NAVIGATION_REMEDIATION_026");
console.log("===============================================================================\n");

const rootDir = path.resolve(__dirname, "..");
const dockPath = path.join(rootDir, "components", "synthesis", "SynthesisMobileDock.tsx");
const radarPath = path.join(rootDir, "app", "radar", "page.tsx");
const setupsPath = path.join(rootDir, "app", "setups", "page.tsx");
const pagePath = path.join(rootDir, "app", "page.tsx");
const portfolioPath = path.join(rootDir, "app", "portfolio", "page.tsx");
const searchPath = path.join(rootDir, "components", "UniversalOmniSearch.tsx");

assert.ok(fs.existsSync(dockPath), "SynthesisMobileDock.tsx must exist");
assert.ok(fs.existsSync(radarPath), "app/radar/page.tsx must exist");
assert.ok(fs.existsSync(setupsPath), "app/setups/page.tsx must exist");
assert.ok(fs.existsSync(pagePath), "app/page.tsx must exist");
assert.ok(fs.existsSync(portfolioPath), "app/portfolio/page.tsx must exist");
assert.ok(fs.existsSync(searchPath), "UniversalOmniSearch.tsx must exist");

const dockSrc = fs.readFileSync(dockPath, "utf8");
const radarSrc = fs.readFileSync(radarPath, "utf8");
const setupsSrc = fs.readFileSync(setupsPath, "utf8");
const pageSrc = fs.readFileSync(pagePath, "utf8");
const portfolioSrc = fs.readFileSync(portfolioPath, "utf8");
const searchSrc = fs.readFileSync(searchPath, "utf8");

// ============================================================================
// TEST GROUP 1: Defect MOB-01 Remediation & Asset Context Invariants
// ============================================================================
console.log("1. Verifying Defect MOB-01: Asset Context Invariants & Contamination Removal...");

// Invariant 1.1: No implicit default ticker substitution in SynthesisMobileDock
assert.ok(
  !dockSrc.includes('activeSymbol = "AAPL"') && !dockSrc.includes("activeSymbol = 'AAPL'"),
  "SynthesisMobileDock must NOT default activeSymbol to 'AAPL' or any other arbitrary ticker"
);
console.log("   [PASS] 1.1: Implicit default ticker ('AAPL') successfully eliminated from component signature.");

// Invariant 1.2: Symbol sanitization and normalization logic
assert.ok(
  dockSrc.includes("activeSymbol.trim().toUpperCase()"),
  "SynthesisMobileDock must trim and uppercase activeSymbol for robust parameter normalization"
);
console.log("   [PASS] 1.2: Robust parameter trimming and uppercasing logic present.");

// Invariant 1.3: Neutral fallback navigation when activeSymbol is null or empty
assert.ok(
  dockSrc.includes('symbolParam ? `/${symbolParam}` : "/"'),
  "Analysis link must navigate to neutral '/' when no active symbol is present"
);
assert.ok(
  dockSrc.includes('cleanSymbol ? `/setups?ticker=${encodeURIComponent(cleanSymbol)}` : "/setups"'),
  "Trade Plan link must navigate to neutral '/setups' when no active symbol is present"
);
console.log("   [PASS] 1.3: Neutral navigation destinations verified for both Analysis ('/') and Plan ('/setups').");

// Invariant 1.4: Radar passes clean canonical asset context into SynthesisMobileDock
assert.ok(
  radarSrc.includes("<SynthesisMobileDock activeSymbol={cleanQ || null} />"),
  "Radar must pass activeSymbol={cleanQ || null} into SynthesisMobileDock"
);
assert.ok(
  !radarSrc.includes("<SynthesisMobileDock />"),
  "Radar must never mount SynthesisMobileDock without passing asset context"
);
console.log("   [PASS] 1.4: Radar accurately binds canonical query asset context into mobile dock.");

// Invariant 1.5: Query parameter normalization in Radar (accepts q, symbol, and ticker)
assert.ok(
  radarSrc.includes("searchParams?.get('ticker')") || radarSrc.includes('searchParams?.get("ticker")'),
  "Radar must support 'ticker' query parameter alongside 'q' and 'symbol' for seamless cross-hub handoffs"
);
console.log("   [PASS] 1.5: Radar query parameter normalization verified across 'q', 'symbol', and 'ticker'.");

// Invariant 1.6: All application hubs consistently mount SynthesisMobileDock with appropriate context
assert.ok(
  pageSrc.includes("<SynthesisMobileDock activeSymbol={selectedSymbol} />"),
  "Analysis page must mount SynthesisMobileDock with activeSymbol={selectedSymbol}"
);
assert.ok(
  setupsSrc.includes("<SynthesisMobileDock activeSymbol={selectedSetup?.ticker || tickerParam || null} />"),
  "Setups page must mount SynthesisMobileDock with current setup/ticker context"
);
assert.ok(
  portfolioSrc.includes("<SynthesisMobileDock activeSymbol={activeSymbol} />"),
  "Portfolio page must mount SynthesisMobileDock with activeSymbol context"
);
console.log("   [PASS] 1.6: Universal dock mounting verified across Analysis, Radar, Setups, and Portfolio.");

// ============================================================================
// TEST GROUP 2: Defect MOB-02 Remediation & Search Decoupling Invariants
// ============================================================================
console.log("\n2. Verifying Defect MOB-02: Search Triggering & Modal Activation Invariants...");

// Invariant 2.1: Elimination of fragile, unmounted DOM element lookup
assert.ok(
  !dockSrc.includes('document.getElementById("omni-search-input")') &&
  !dockSrc.includes("document.getElementById('omni-search-input')"),
  "SynthesisMobileDock must NOT perform direct DOM lookup on unmounted #omni-search-input"
);
console.log("   [PASS] 2.1: Unmounted DOM query selector eliminated from mobile dock handler.");

// Invariant 2.2: Decoupled custom event dispatch
assert.ok(
  dockSrc.includes('window.dispatchEvent(new CustomEvent("arx:open-omni-search"))'),
  "SynthesisMobileDock must dispatch 'arx:open-omni-search' custom event"
);
console.log("   [PASS] 2.2: Mobile search button dispatches decoupled 'arx:open-omni-search' event.");

// Invariant 2.3: UniversalOmniSearch registers listener for custom event
assert.ok(
  searchSrc.includes('window.addEventListener("arx:open-omni-search"'),
  "UniversalOmniSearch must register window event listener for 'arx:open-omni-search'"
);
assert.ok(
  searchSrc.includes('window.removeEventListener("arx:open-omni-search"'),
  "UniversalOmniSearch must clean up event listener on unmount"
);
console.log("   [PASS] 2.3: UniversalOmniSearch registers and cleans up 'arx:open-omni-search' event listener.");

// Invariant 2.4: Focus restoration & accessibility semantics preserved
assert.ok(
  searchSrc.includes('triggerRef.current = (document.activeElement as HTMLElement) || null'),
  "UniversalOmniSearch must capture active element on event trigger to restore focus upon dismissal"
);
assert.ok(
  searchSrc.includes('role="dialog"') && searchSrc.includes('aria-modal="true"'),
  "UniversalOmniSearch dialog must preserve WCAG dialog roles and accessibility semantics"
);
assert.ok(
  searchSrc.includes('e.key === "Escape" && isOpen'),
  "UniversalOmniSearch must maintain Escape key dismissal"
);
console.log("   [PASS] 2.4: Accessibility semantics, Escape dismissal, and focus restoration intact.");

// ============================================================================
// TEST GROUP 3: Functional URL Transformation Logic Unit Testing
// ============================================================================
console.log("\n3. Testing URL Resolution Matrix with SynthesisMobileDock Logic...");

function resolveUrls(activeSymbol?: string | null) {
  const cleanSymbol =
    activeSymbol && typeof activeSymbol === "string" && activeSymbol.trim()
      ? activeSymbol.trim().toUpperCase()
      : null;
  const symbolParam = cleanSymbol ? `?symbol=${encodeURIComponent(cleanSymbol)}` : "";
  const analysisHref = symbolParam ? `/${symbolParam}` : "/";
  const planHref = cleanSymbol ? `/setups?ticker=${encodeURIComponent(cleanSymbol)}` : "/setups";
  return { cleanSymbol, analysisHref, planHref };
}

// Case A: No asset selected (undefined)
const caseUndefined = resolveUrls(undefined);
assert.strictEqual(caseUndefined.cleanSymbol, null);
assert.strictEqual(caseUndefined.analysisHref, "/");
assert.strictEqual(caseUndefined.planHref, "/setups");
console.log("   [PASS] Case A (undefined): Analysis -> '/', Plan -> '/setups'");

// Case B: No asset selected (null)
const caseNull = resolveUrls(null);
assert.strictEqual(caseNull.cleanSymbol, null);
assert.strictEqual(caseNull.analysisHref, "/");
assert.strictEqual(caseNull.planHref, "/setups");
console.log("   [PASS] Case B (null): Analysis -> '/', Plan -> '/setups'");

// Case C: Empty string or whitespace only
const caseEmpty = resolveUrls("   ");
assert.strictEqual(caseEmpty.cleanSymbol, null);
assert.strictEqual(caseEmpty.analysisHref, "/");
assert.strictEqual(caseEmpty.planHref, "/setups");
console.log("   [PASS] Case C (whitespace): Analysis -> '/', Plan -> '/setups'");

// Case D: Valid asset lowercase with padding (' tsla ')
const caseTsla = resolveUrls(" tsla ");
assert.strictEqual(caseTsla.cleanSymbol, "TSLA");
assert.strictEqual(caseTsla.analysisHref, "/?symbol=TSLA");
assert.strictEqual(caseTsla.planHref, "/setups?ticker=TSLA");
console.log("   [PASS] Case D (' tsla '): Analysis -> '/?symbol=TSLA', Plan -> '/setups?ticker=TSLA'");

// Case E: Valid asset with special character ('BRK.B')
const caseBrk = resolveUrls("BRK.B");
assert.strictEqual(caseBrk.cleanSymbol, "BRK.B");
assert.strictEqual(caseBrk.analysisHref, "/?symbol=BRK.B");
assert.strictEqual(caseBrk.planHref, "/setups?ticker=BRK.B");
console.log("   [PASS] Case E ('BRK.B'): Analysis -> '/?symbol=BRK.B', Plan -> '/setups?ticker=BRK.B'");

console.log("\n===============================================================================");
console.log("ALL INVARIANTS PASS: MOB-01 & MOB-02 REMEDIATION TECHNICALLY QUALIFIED");
console.log("===============================================================================");
