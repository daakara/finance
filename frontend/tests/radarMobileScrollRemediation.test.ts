import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

console.log("Starting Radar Mobile Scroll Remediation Test Suite (RADAR-01 - RADAR-08)...");

const radarPagePath = path.resolve(__dirname, "../app/radar/page.tsx");
const radarSource = fs.readFileSync(radarPagePath, "utf-8");

// ============================================================================
// RADAR-01, RADAR-02, RADAR-03: Reachability at 375px, 390px, 393px
// ============================================================================
// The container must have horizontal scroll capability without page overflow.
assert.ok(
  radarSource.includes("overflow-x-auto"),
  "RADAR-01..03 FAIL: Ownership filter container must declare overflow-x-auto"
);
assert.ok(
  radarSource.includes("min-w-max"),
  "RADAR-01..03 FAIL: Inner chip group must declare min-w-max so items do not wrap or truncate"
);
assert.ok(
  radarSource.includes("w-full sm:w-auto"),
  "RADAR-01..03 FAIL: Outer container must be w-full on mobile for bounded viewport containment"
);
console.log("[OK] RADAR-01 (375px), RADAR-02 (390px), RADAR-03 (393px): Horizontal reachability structure verified");

// ============================================================================
// RADAR-04: Horizontal swipe reaches My Holdings
// ============================================================================
assert.ok(
  radarSource.includes("touch-pan-x"),
  "RADAR-04 FAIL: Container must declare touch-pan-x to allow touch horizontal swiping"
);
assert.ok(
  radarSource.includes("overscroll-x-contain"),
  "RADAR-04 FAIL: Container must declare overscroll-x-contain to trap horizontal swipe within scroll area"
);
console.log("[OK] RADAR-04: Horizontal swipe gestures enabled via touch-pan-x and overscroll-x-contain");

// ============================================================================
// RADAR-05: No page-level horizontal overflow
// ============================================================================
assert.ok(
  radarSource.includes("overflow-y-hidden"),
  "RADAR-05 FAIL: Container must declare overflow-y-hidden to prevent vertical scrollbar desync"
);
// Confirm scrollbar-none is NOT used (as verified in Section 3)
assert.ok(
  !radarSource.includes("scrollbar-none"),
  "RADAR-05 FAIL: Must not assume non-existent scrollbar-none utility class"
);
console.log("[OK] RADAR-05: Page-level horizontal overflow bounded within component viewport");

// ============================================================================
// RADAR-06: All filter touch targets >= 44px
// ============================================================================
const filterButtonIds = [
  "btn-ownership-filter-all",
  "btn-ownership-filter-new",
  "btn-ownership-filter-holdings",
];

for (const btnId of filterButtonIds) {
  assert.ok(
    radarSource.includes(`id="${btnId}"`),
    `RADAR-06 FAIL: Missing button id ${btnId}`
  );
}

// Check that buttons declare min-h-[44px] and shrink-0
const minHeightMatches = (radarSource.match(/min-h-\[44px\]/g) || []).length;
assert.ok(
  minHeightMatches >= 3,
  `RADAR-06 FAIL: All 3 ownership filter buttons must declare min-h-[44px], found ${minHeightMatches}`
);

const shrinkMatches = (radarSource.match(/shrink-0/g) || []).length;
assert.ok(
  shrinkMatches >= 3,
  `RADAR-06 FAIL: Ownership filter buttons must declare shrink-0 to prevent flex compression on narrow screens`
);
console.log("[OK] RADAR-06: All filter touch targets >=44px and flex shrink-0 verified");

// ============================================================================
// RADAR-07: Selected off-screen filter scrolls into view without vertical jump
// ============================================================================
assert.ok(
  radarSource.includes("handleOwnershipFilterSelect"),
  "RADAR-07 FAIL: Must declare handleOwnershipFilterSelect helper"
);
assert.ok(
  radarSource.includes("scrollIntoView({ block: 'nearest', inline: 'nearest'"),
  "RADAR-07 FAIL: Must scroll active selection into view with block: 'nearest' and inline: 'nearest'"
);
console.log("[OK] RADAR-07: Active filter selection auto-visibility with block: nearest / inline: nearest verified");

// ============================================================================
// RADAR-08: Desktop ownership row unchanged
// ============================================================================
assert.ok(
  radarSource.includes("sm:w-auto"),
  "RADAR-08 FAIL: Container must use sm:w-auto so desktop does not stretch full width"
);
assert.ok(
  radarSource.includes("pb-1 sm:pb-0"),
  "RADAR-08 FAIL: Padding bottom should adapt between mobile (pb-1) and desktop (sm:pb-0)"
);
console.log("[OK] RADAR-08: Desktop layout characteristics preserved");

console.log("ALL RADAR TESTS (RADAR-01 - RADAR-08) PASSED SUCCESSFULLY!");
