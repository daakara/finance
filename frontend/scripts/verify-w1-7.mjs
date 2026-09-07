import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

console.log("========================================================================");
console.log("  ARX Terminal vNext: W1.7 & W1.8 Verification Suite                    ");
console.log("  (65/35 Decision Workspace & Telemetry Foundation)                     ");
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

const corridorPath = path.resolve("components/workstation/ExecutionCorridor.tsx");
const chartWsPath = path.resolve("components/workstation/PriceChartWorkspace.tsx");
const canvasPath = path.resolve("components/workstation/WorkstationCanvas.tsx");
const gridPath = path.resolve("components/layout/WorkstationGrid.tsx");
const telemetryTrackerPath = path.resolve("telemetry/tracker.ts");
const telemetryTypesPath = path.resolve("types/telemetry.ts");
const unitTestPath = path.resolve("components/workstation/__tests__/WorkstationCanvas.test.tsx");
const e2eTestPath = path.resolve("tests/e2e/workstation-viewport.spec.ts");

const corridorSrc = fs.readFileSync(corridorPath, "utf-8");
const chartWsSrc = fs.readFileSync(chartWsPath, "utf-8");
const canvasSrc = fs.readFileSync(canvasPath, "utf-8");
const gridSrc = fs.readFileSync(gridPath, "utf-8");
const telemetryTrackerSrc = fs.readFileSync(telemetryTrackerPath, "utf-8");
const telemetryTypesSrc = fs.readFileSync(telemetryTypesPath, "utf-8");
const unitTestSrc = fs.readFileSync(unitTestPath, "utf-8");
const e2eTestSrc = fs.readFileSync(e2eTestPath, "utf-8");

// Suite 1: 65/35 Workstation Layout Architecture
test("WorkstationGrid.tsx enforces 65% Chart (8 cols) and 35% Corridor (4 cols) with 620px min-height", () => {
  assert(gridSrc.includes("lg:col-span-8"), "Chart must occupy 8 of 12 columns (65%)");
  assert(gridSrc.includes("lg:col-span-4"), "Execution must occupy 4 of 12 columns (35%)");
  assert(gridSrc.includes("lg:min-h-[620px]"), "Desktop min-height 620px must be enforced for zero CLS");
  assert(gridSrc.includes("data-testid=\"price-chart-workspace\""), "Must declare chart testid");
  assert(gridSrc.includes("data-testid=\"execution-corridor\""), "Must declare corridor testid");
});

// Suite 2: ExecutionCorridor Component
test("ExecutionCorridor renders all decision levels with anti-cyan color tokens", () => {
  assert(corridorSrc.includes("data-testid=\"execution-corridor\""), "Must declare execution-corridor testid");
  assert(corridorSrc.includes("data-testid=\"corridor-target-2\""), "Must render target 2");
  assert(corridorSrc.includes("data-testid=\"corridor-target-1\""), "Must render target 1");
  assert(corridorSrc.includes("data-testid=\"corridor-entry\""), "Must render active entry corridor");
  assert(corridorSrc.includes("data-testid=\"corridor-stop\""), "Must render stop loss floor");
  assert(corridorSrc.includes("data-testid=\"corridor-sizer-cta\""), "Must render position sizer CTA");
  assert(corridorSrc.includes("bg-rose-500"), "Stop loss must use high contrast Rose token");
  assert(corridorSrc.includes("bg-emerald-500"), "Targets must use Emerald tokens");
});

// Suite 3: PriceChartWorkspace Component
test("PriceChartWorkspace wraps chart with toolbar, timeframe switcher, and min-height", () => {
  assert(chartWsSrc.includes("data-testid=\"price-chart-workspace\""), "Must declare price-chart-workspace testid");
  assert(chartWsSrc.includes("min-h-[420px]"), "Must set min-height 420px on mobile");
  assert(chartWsSrc.includes("lg:min-h-[540px]"), "Must set min-height 540px on desktop");
  assert(chartWsSrc.includes("role=\"tablist\""), "Must declare tablist for timeframes");
});

// Suite 4: WorkstationCanvas Integration
test("WorkstationCanvas integrates Stage 1 Command Strip and Stage 2 65/35 Grid", () => {
  assert(canvasSrc.includes("data-testid=\"workstation-canvas\""), "Must declare workstation-canvas testid");
  assert(canvasSrc.includes("<TickerCommandStrip"), "Must mount TickerCommandStrip");
  assert(canvasSrc.includes("<WorkstationGrid"), "Must mount WorkstationGrid");
  assert(canvasSrc.includes("<PriceChartWorkspace"), "Must mount PriceChartWorkspace");
  assert(canvasSrc.includes("<ExecutionCorridor"), "Must mount ExecutionCorridor");
});

// Suite 5: Telemetry Foundation (W1.8)
test("telemetry/tracker.ts and types/telemetry.ts capture high-res performance timers and core events", () => {
  assert(telemetryTrackerSrc.includes("performance.now()"), "Must use performance.now() monotonic timer");
  assert(telemetryTypesSrc.includes("workspace_loaded"), "Must define workspace_loaded event");
  assert(telemetryTypesSrc.includes("position_sizer_opened"), "Must define position_sizer_opened event");
  assert(telemetryTypesSrc.includes("ttc_started"), "Must define ttc_started event");
  assert(telemetryTypesSrc.includes("ttc_completed"), "Must define ttc_completed event");
  assert(telemetryTypesSrc.includes("ttfmi_completed"), "Must define ttfmi_completed event");
  assert(telemetryTypesSrc.includes("execution_corridor_viewed"), "Must define execution_corridor_viewed");
});

// Suite 6: Test Automation Suites
test("Unit test suite covers ExecutionCorridor, PriceChartWorkspace, and WorkstationCanvas", () => {
  assert(unitTestSrc.includes("ExecutionCorridor (35% Actionable Ladder)"), "Covers corridor unit tests");
  assert(unitTestSrc.includes("PriceChartWorkspace (65% Chart Canvas)"), "Covers chart workspace tests");
  assert(unitTestSrc.includes("WorkstationCanvas (Integrated 65/35 Above-the-Fold Assembly)"), "Covers canvas integration tests");
});

test("Playwright E2E suite defines UX-001 through UX-007 acceptance tests", () => {
  assert(e2eTestSrc.includes("UX-001: Chart Above Fold"), "Must define UX-001");
  assert(e2eTestSrc.includes("UX-002: Execution Corridor Visibility"), "Must define UX-002");
  assert(e2eTestSrc.includes("UX-003: Critical Data Discovery"), "Must define UX-003");
  assert(e2eTestSrc.includes("UX-004: Setup Score Discovery"), "Must define UX-004");
  assert(e2eTestSrc.includes("UX-005: Viewport Audit"), "Must define UX-005");
  assert(e2eTestSrc.includes("UX-006: Chart Dominance Control"), "Must define UX-006");
  assert(e2eTestSrc.includes("UX-007: Zero CLS Validation"), "Must define UX-007");
});

console.log("\n========================================================================");
console.log(`  VERIFICATION RESULTS: ${passed} PASSED, ${failed} FAILED`);
console.log("========================================================================\n");

if (failed > 0) {
  process.exit(1);
}
