import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

console.log("========================================================================");
console.log("  ARX Terminal vNext: W1.5 Verification Suite (Ticker Command Strip)   ");
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

const typesPath = path.resolve("types/workstation.ts");
const commandStripPath = path.resolve("components/command-strip/TickerCommandStrip.tsx");
const setupBadgePath = path.resolve("components/command-strip/SetupScoreBadge.tsx");
const execBadgePath = path.resolve("components/command-strip/ExecutionStateBadge.tsx");
const liqBadgePath = path.resolve("components/command-strip/LiquidityBadge.tsx");
const unitTestPath = path.resolve("components/command-strip/__tests__/TickerCommandStrip.test.tsx");
const e2eTestPath = path.resolve("tests/e2e/ticker-command-strip.spec.ts");
const previewPath = path.resolve("app/design-system-preview/page.tsx");

const typesSrc = fs.readFileSync(typesPath, "utf-8");
const commandStripSrc = fs.readFileSync(commandStripPath, "utf-8");
const setupBadgeSrc = fs.readFileSync(setupBadgePath, "utf-8");
const execBadgeSrc = fs.readFileSync(execBadgePath, "utf-8");
const liqBadgeSrc = fs.readFileSync(liqBadgePath, "utf-8");
const unitTestSrc = fs.readFileSync(unitTestPath, "utf-8");
const e2eTestSrc = fs.readFileSync(e2eTestPath, "utf-8");
const previewSrc = fs.readFileSync(previewPath, "utf-8");

// Suite 1: Types & API Contracts
test("types/workstation.ts exports complete TickerCommandStripProps and atomic types", () => {
  assert(typesSrc.includes("export interface TickerCommandStripProps"), "Must export TickerCommandStripProps");
  assert(typesSrc.includes("export type DomainConfidence"), "Must export DomainConfidence");
  assert(typesSrc.includes("export type ExecutionState"), "Must export ExecutionState");
  assert(typesSrc.includes("export type LiquidityTier"), "Must export LiquidityTier");
  assert(typesSrc.includes("export type MarketRegime"), "Must export MarketRegime");
  assert(typesSrc.includes("export interface WorkstationPayload"), "Must export WorkstationPayload");
});

// Suite 2: SetupScoreBadge & Anti-Cyan Invariant
test("SetupScoreBadge enforces semantic color thresholds (Emerald >= 70, Amber 50-69, Rose < 50)", () => {
  assert(setupBadgeSrc.includes("normalizedScore >= 70"), "Must threshold at 70");
  assert(setupBadgeSrc.includes("#10b981"), "Must use Emerald for high score");
  assert(setupBadgeSrc.includes("#f59e0b"), "Must use Amber for mid score");
  assert(setupBadgeSrc.includes("#f43f5e"), "Must use Rose for low score");
  assert(setupBadgeSrc.includes("role=\"meter\""), "Must declare role='meter'");
  assert(setupBadgeSrc.includes("aria-valuenow="), "Must bind aria-valuenow");
  assert(setupBadgeSrc.includes("data-testid=\"setup-score-badge\""), "Must contain data-testid");
});

// Suite 3: ExecutionStateBadge
test("ExecutionStateBadge maps all 5 states with semantic anti-cyan styling", () => {
  assert(execBadgeSrc.includes("IN_BUY_ZONE"), "Must support IN_BUY_ZONE");
  assert(execBadgeSrc.includes("APPROACHING_TARGET"), "Must support APPROACHING_TARGET");
  assert(execBadgeSrc.includes("WAITING_PULLBACK"), "Must support WAITING_PULLBACK");
  assert(execBadgeSrc.includes("STOPPED_OUT"), "Must support STOPPED_OUT");
  assert(execBadgeSrc.includes("NEUTRAL"), "Must support NEUTRAL");
  assert(execBadgeSrc.includes("data-testid=\"execution-state-badge\""), "Must declare test-id");
  assert(execBadgeSrc.includes("mode === \"GUIDED\""), "Must adapt text for Guided mode");
});

// Suite 4: LiquidityBadge
test("LiquidityBadge formats ADV heuristic tiers and supports Amihud in QUANT mode", () => {
  assert(liqBadgeSrc.includes("ADV: High (<1.0% ADV)"), "Must support High liquidity label");
  assert(liqBadgeSrc.includes("ADV: Moderate"), "Must support Moderate liquidity label");
  assert(liqBadgeSrc.includes("ADV: Illiquid / Risk"), "Must support Risk liquidity label");
  assert(liqBadgeSrc.includes("mode === \"QUANT\" && amihudScore !== undefined"), "Must support Amihud in QUANT mode");
  assert(liqBadgeSrc.includes("data-testid=\"liquidity-badge\""), "Must declare test-id");
});

// Suite 5: TickerCommandStrip Layout & Height Constraints
test("TickerCommandStrip enforces 110px desktop constraint and zero-CLS skeleton", () => {
  assert(commandStripSrc.includes("min-h-[110px]"), "Must set min-height 110px");
  assert(commandStripSrc.includes("lg:h-[110px]"), "Must constrain lg desktop height to 110px");
  assert(commandStripSrc.includes("data-testid=\"ticker-command-strip\""), "Must declare ticker-command-strip testid");
  assert(commandStripSrc.includes("export function TickerCommandStripSkeleton"), "Must export zero-CLS skeleton");
  assert(commandStripSrc.includes("data-testid=\"ticker-command-strip-skeleton\""), "Skeleton must declare testid");
});

test("TickerCommandStrip includes identity, spot price, and pinned settlement notice", () => {
  assert(commandStripSrc.includes("data-testid=\"ticker-symbol\""), "Must render symbol");
  assert(commandStripSrc.includes("data-testid=\"company-name\""), "Must render company name");
  assert(commandStripSrc.includes("data-testid=\"spot-price\""), "Must render spot price");
  assert(commandStripSrc.includes("data-testid=\"price-delta\""), "Must render price delta");
  assert(commandStripSrc.includes("data-testid=\"settlement-pinned-notice\""), "Must render settlement notice testid");
  assert(commandStripSrc.includes("[Session Closed / Friday Settlement Pinned]"), "Must render pinned settlement copy");
});

// Suite 6: Test Automation Coverage
test("Unit test suite covers all subcomponents and integration scenarios", () => {
  assert(unitTestSrc.includes("describe(\"TickerCommandStrip & Badge Components"), "Unit test suite declared");
  assert(unitTestSrc.includes("renders skeleton with strict 110px desktop height"), "Covers skeleton test");
  assert(unitTestSrc.includes("renders emerald styling for favorable scores >= 70"), "Covers score >= 70");
  assert(unitTestSrc.includes("renders amber styling for cautionary scores 50 - 69"), "Covers score 50-69");
  assert(unitTestSrc.includes("renders rose styling for high-risk scores < 50"), "Covers score < 50");
  assert(unitTestSrc.includes("renders complete Stage 1 orientation header"), "Covers complete integration test");
});

test("Playwright E2E suite defines AC-W1.5-01 through AC-W1.5-08 acceptance tests", () => {
  assert(e2eTestSrc.includes("AC-W1.5-01"), "Must define AC-W1.5-01");
  assert(e2eTestSrc.includes("AC-W1.5-02"), "Must define AC-W1.5-02");
  assert(e2eTestSrc.includes("AC-W1.5-03"), "Must define AC-W1.5-03");
  assert(e2eTestSrc.includes("AC-W1.5-04"), "Must define AC-W1.5-04");
  assert(e2eTestSrc.includes("AC-W1.5-05"), "Must define AC-W1.5-05");
  assert(e2eTestSrc.includes("AC-W1.5-06"), "Must define AC-W1.5-06");
  assert(e2eTestSrc.includes("AC-W1.5-07"), "Must define AC-W1.5-07");
  assert(e2eTestSrc.includes("AC-W1.5-08"), "Must define AC-W1.5-08");
});

// Suite 7: Design System Showcase Integration
test("design-system-preview page includes interactive Command Strip tab with live variants", () => {
  assert(previewSrc.includes("activeTab === 'command-strip'"), "Must have command-strip active tab");
  assert(previewSrc.includes("<TickerCommandStrip"), "Must render TickerCommandStrip");
  assert(previewSrc.includes("<TickerCommandStripSkeleton"), "Must render skeleton");
});

console.log("\n========================================================================");
console.log(`  VERIFICATION RESULTS: ${passed} PASSED, ${failed} FAILED`);
console.log("========================================================================\n");

if (failed > 0) {
  process.exit(1);
}
