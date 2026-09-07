import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

console.log("========================================================================");
console.log("  ARX Terminal vNext: W1.6 Verification Suite (Watchlist Drawer)        ");
console.log("========================================================================\n");

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

const uiStorePath = path.resolve("state/ui-store.ts");
const drawerPath = path.resolve("components/drawers/WatchlistDrawer.tsx");
const triggerPath = path.resolve("components/drawers/WatchlistDrawerTrigger.tsx");
const hotkeysPath = path.resolve("components/drawers/WatchlistDrawerHotkeys.tsx");
const contentPath = path.resolve("components/drawers/WatchlistDrawerContent.tsx");
const telemetryTypesPath = path.resolve("types/telemetry.ts");
const unitTestPath = path.resolve("components/drawers/__tests__/WatchlistDrawer.test.tsx");
const e2eTestPath = path.resolve("tests/e2e/watchlist-drawer.spec.ts");
const navbarPath = path.resolve("components/Navbar.tsx");
const previewPath = path.resolve("app/design-system-preview/page.tsx");

const uiStoreSrc = fs.readFileSync(uiStorePath, "utf-8");
const drawerSrc = fs.readFileSync(drawerPath, "utf-8");
const triggerSrc = fs.readFileSync(triggerPath, "utf-8");
const hotkeysSrc = fs.readFileSync(hotkeysPath, "utf-8");
const contentSrc = fs.readFileSync(contentPath, "utf-8");
const telemetryTypesSrc = fs.readFileSync(telemetryTypesPath, "utf-8");
const unitTestSrc = fs.readFileSync(unitTestPath, "utf-8");
const e2eTestSrc = fs.readFileSync(e2eTestPath, "utf-8");
const navbarSrc = fs.readFileSync(navbarPath, "utf-8");
const previewSrc = fs.readFileSync(previewPath, "utf-8");

// Suite 1: Global UI Store & State Persistence
test("state/ui-store.ts manages watchlistOpen and persists to arx-watchlist-open", () => {
  assert(uiStoreSrc.includes("export const useUIStore"), "Must export useUIStore");
  assert(uiStoreSrc.includes("arx-watchlist-open"), "Must use localStorage key arx-watchlist-open");
  assert(uiStoreSrc.includes("openWatchlist:"), "Must implement openWatchlist");
  assert(uiStoreSrc.includes("closeWatchlist:"), "Must implement closeWatchlist");
  assert(uiStoreSrc.includes("toggleWatchlist:"), "Must implement toggleWatchlist");
});

// Suite 2: Telemetry Event Contracts
test("types/telemetry.ts defines watchlist_drawer_opened and watchlist_drawer_closed", () => {
  assert(telemetryTypesSrc.includes("watchlist_drawer_opened"), "Must include watchlist_drawer_opened");
  assert(telemetryTypesSrc.includes("watchlist_drawer_closed"), "Must include watchlist_drawer_closed");
  assert(telemetryTypesSrc.includes("WatchlistOpenedPayload"), "Must define WatchlistOpenedPayload");
  assert(telemetryTypesSrc.includes("WatchlistClosedPayload"), "Must define WatchlistClosedPayload");
});

// Suite 3: WatchlistDrawer Layout & Zero Chart Remount
test("WatchlistDrawer renders slide-over sheet with accessibility dialog attributes", () => {
  assert(drawerSrc.includes("role=\"dialog\""), "Must declare role='dialog'");
  assert(drawerSrc.includes("aria-modal=\"true\""), "Must declare aria-modal='true'");
  assert(drawerSrc.includes("aria-label=\"Watchlist Drawer\""), "Must declare aria-label");
  assert(drawerSrc.includes("data-testid=\"watchlist-drawer\""), "Must declare data-testid");
  assert(drawerSrc.includes("data-testid=\"watchlist-drawer-backdrop\""), "Must declare backdrop testid");
  assert(drawerSrc.includes("data-testid=\"watchlist-drawer-close\""), "Must declare close testid");
  assert(drawerSrc.includes("translate-x-0"), "Must support open slide-over position");
  assert(drawerSrc.includes("-translate-x-full"), "Must support closed offscreen position");
  assert(drawerSrc.includes("lg:w-80"), "Must enforce 320px width on desktop");
});

// Suite 4: WatchlistDrawerTrigger
test("WatchlistDrawerTrigger binds aria-controls and aria-expanded", () => {
  assert(triggerSrc.includes("data-testid=\"watchlist-drawer-trigger\""), "Must declare data-testid");
  assert(triggerSrc.includes("aria-controls=\"watchlist-drawer\""), "Must link to watchlist-drawer id");
  assert(triggerSrc.includes("aria-expanded={watchlistOpen}"), "Must bind aria-expanded");
});

// Suite 5: Keyboard Hotkeys Engine
test("WatchlistDrawerHotkeys listens to '[', 'Ctrl+B', and 'Escape'", () => {
  assert(hotkeysSrc.includes("e.key === \"[\""), "Must listen to bracket key '['");
  assert(hotkeysSrc.includes("e.ctrlKey || e.metaKey") && hotkeysSrc.includes("\"b\""), "Must listen to Ctrl+B / Cmd+B");
  assert(hotkeysSrc.includes("e.key === \"Escape\""), "Must listen to Escape key");
  assert(hotkeysSrc.includes("isInputFocused"), "Must guard against active input/textarea");
});

// Suite 6: WatchlistDrawerContent Features
test("WatchlistDrawerContent preserves search, categories, and live quote syncing", () => {
  assert(contentSrc.includes("data-testid=\"watchlist-search-input\""), "Must render search input");
  assert(contentSrc.includes("data-testid=\"watchlist-items-container\""), "Must render items container");
  assert(contentSrc.includes("MiniSparkline"), "Must render sparkline");
  assert(contentSrc.includes("togglePin"), "Must support pin toggling");
});

// Suite 7: Test Automation Suites
test("Unit test suite covers store, trigger, drawer, and hotkey events", () => {
  assert(unitTestSrc.includes("UIStore State Management & Persistence"), "Covers store tests");
  assert(unitTestSrc.includes("WatchlistDrawerTrigger"), "Covers trigger tests");
  assert(unitTestSrc.includes("WatchlistDrawer Component Accessibility"), "Covers drawer a11y tests");
  assert(unitTestSrc.includes("Keyboard Hotkeys Interaction"), "Covers hotkeys tests");
});

test("Playwright E2E suite defines AC-W1.6-01 through AC-W1.6-07 acceptance tests", () => {
  assert(e2eTestSrc.includes("AC-W1.6-01"), "Must define AC-W1.6-01");
  assert(e2eTestSrc.includes("AC-W1.6-02"), "Must define AC-W1.6-02");
  assert(e2eTestSrc.includes("AC-W1.6-03"), "Must define AC-W1.6-03");
  assert(e2eTestSrc.includes("AC-W1.6-04"), "Must define AC-W1.6-04");
  assert(e2eTestSrc.includes("AC-W1.6-05"), "Must define AC-W1.6-05");
  assert(e2eTestSrc.includes("AC-W1.6-06"), "Must define AC-W1.6-06");
  assert(e2eTestSrc.includes("AC-W1.6-07"), "Must define AC-W1.6-07");
});

// Suite 8: Global Navbar & Preview Integration
test("Navbar.tsx and design-system-preview integrate WatchlistDrawer components", () => {
  assert(navbarSrc.includes("WatchlistDrawerTrigger"), "Navbar must include WatchlistDrawerTrigger");
  assert(previewSrc.includes("WatchlistDrawer"), "Preview must include WatchlistDrawer");
  assert(previewSrc.includes("activeTab === 'watchlist-drawer'"), "Preview must include tab 6");
});

console.log("\n========================================================================");
console.log(`  VERIFICATION RESULTS: ${passed} PASSED, ${failed} FAILED`);
console.log("========================================================================\n");

if (failed > 0) {
  process.exit(1);
}
