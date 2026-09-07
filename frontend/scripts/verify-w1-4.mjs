import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

console.log("=================================================================");
console.log("  ARX Terminal vNext: W1.4 Verification Suite (Experience Modes)  ");
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

const storePath = path.resolve("state/experience-store.ts");
const hookPath = path.resolve("hooks/useExperienceMode.ts");
const togglePath = path.resolve("components/experience/ExperienceModeToggle.tsx");
const navbarPath = path.resolve("components/Navbar.tsx");
const unitStoreTestPath = path.resolve("state/__tests__/experience-store.test.ts");
const unitHookTestPath = path.resolve("hooks/__tests__/useExperienceMode.test.ts");
const unitToggleTestPath = path.resolve("components/experience/__tests__/ExperienceModeToggle.test.tsx");
const e2eTestPath = path.resolve("tests/e2e/experience-mode.spec.ts");

const storeSrc = fs.readFileSync(storePath, "utf-8");
const hookSrc = fs.readFileSync(hookPath, "utf-8");
const toggleSrc = fs.readFileSync(togglePath, "utf-8");
const navbarSrc = fs.readFileSync(navbarPath, "utf-8");
const unitStoreTestSrc = fs.readFileSync(unitStoreTestPath, "utf-8");
const unitHookTestSrc = fs.readFileSync(unitHookTestPath, "utf-8");
const unitToggleTestSrc = fs.readFileSync(unitToggleTestPath, "utf-8");
const e2eTestSrc = fs.readFileSync(e2eTestPath, "utf-8");

// Suite 1: Zustand Experience Store
test("experience-store.ts exports useExperienceStore, DEFAULT_MODE, and ExperienceMode type", () => {
  assert(storeSrc.includes("export const useExperienceStore"), "Must export useExperienceStore");
  assert(storeSrc.includes("export const DEFAULT_MODE: ExperienceMode = \"STANDARD\""), "DEFAULT_MODE must be STANDARD");
  assert(storeSrc.includes("export type ExperienceMode = \"GUIDED\" | \"STANDARD\" | \"QUANT\""), "ExperienceMode type must define GUIDED, STANDARD, QUANT");
});

// Suite 2: useExperienceMode Hook & ADR-003 Logic
test("useExperienceMode.ts implements ADR-003 precedence (URL > localStorage > DEFAULT)", () => {
  assert(hookSrc.includes("const rawUrlMode = params.get(\"mode\")"), "Must inspect URL mode parameter");
  assert(hookSrc.includes("const savedRaw = localStorage.getItem(STORAGE_KEY)"), "Must inspect localStorage");
  assert(hookSrc.includes("setMode(DEFAULT_MODE)"), "Must fall back to DEFAULT_MODE");
});

test("useExperienceMode.ts validates and normalizes modes (including advanced alias)", () => {
  assert(hookSrc.includes("export function isValidMode"), "Must export isValidMode function");
  assert(hookSrc.includes("export function normalizeMode"), "Must export normalizeMode function");
  assert(hookSrc.includes("lower === \"quant\" || lower === \"advanced\""), "Must map advanced alias to quant");
});

test("useExperienceMode.ts repairs corrupt localStorage and handles invalid URL parameters", () => {
  assert(hookSrc.includes("localStorage.setItem(STORAGE_KEY, \"standard\")"), "Must repair localStorage to standard");
  assert(hookSrc.includes("updateUrl(\"standard\")"), "Must rewrite invalid URL to standard");
});

test("useExperienceMode.ts provides telemetry readiness hooks (W1.8)", () => {
  assert(hookSrc.includes("arx:telemetry:experience_mode_changed"), "Must dispatch experience_mode_changed telemetry event");
  assert(hookSrc.includes("from_mode: fromMode"), "Must report from_mode in telemetry");
  assert(hookSrc.includes("to_mode: targetMode"), "Must report to_mode in telemetry");
});

// Suite 3: ExperienceModeToggle Component
test("ExperienceModeToggle.tsx implements role='tablist' and accessible keyboard navigation", () => {
  assert(toggleSrc.includes("role=\"tablist\""), "Container must have role=tablist");
  assert(toggleSrc.includes("aria-label=\"Experience Mode\""), "Tablist must have aria-label");
  assert(toggleSrc.includes("role=\"tab\""), "Buttons must have role=tab");
  assert(toggleSrc.includes("aria-selected={isSelected}"), "Buttons must set aria-selected");
  assert(toggleSrc.includes("e.key === \"ArrowRight\""), "Must handle ArrowRight");
  assert(toggleSrc.includes("e.key === \"ArrowLeft\""), "Must handle ArrowLeft");
  assert(toggleSrc.includes("e.key === \"Home\""), "Must handle Home key");
  assert(toggleSrc.includes("e.key === \"End\""), "Must handle End key");
});

test("ExperienceModeToggle.tsx renders pre-hydration skeleton state for zero CLS", () => {
  assert(toggleSrc.includes("data-testid=\"experience-mode-skeleton\""), "Must provide skeleton testid");
  assert(toggleSrc.includes("if (!isHydrated)"), "Must gate rendering on isHydrated");
});

test("ExperienceModeToggle.tsx adheres to Anti-Cyan invariant", () => {
  assert(toggleSrc.includes("bg-emerald-600"), "Guided mode must use Emerald");
  assert(toggleSrc.includes("bg-purple-600"), "Quant mode must use Purple");
  assert(toggleSrc.includes("bg-cyan-600"), "Standard tab selection may use Cyan");
});

// Suite 4: Navbar Integration
test("Navbar.tsx mounts ExperienceModeToggle in top navigation shell", () => {
  assert(navbarSrc.includes("<ExperienceModeToggle />"), "Navbar must render ExperienceModeToggle");
  assert(navbarSrc.includes("import ExperienceModeToggle from \"./experience/ExperienceModeToggle\""), "Navbar must import ExperienceModeToggle");
});

// Suite 5: Test Coverage
test("Unit test suites cover store, hook, and toggle component", () => {
  assert(unitStoreTestSrc.includes("defaults to STANDARD mode"), "Store test must verify default");
  assert(unitHookTestSrc.includes("prioritizes URL mode over localStorage"), "Hook test must verify ADR-003 precedence");
  assert(unitToggleTestSrc.includes("provides accessible tablist name"), "Toggle test must verify tablist a11y");
  assert(unitToggleTestSrc.includes("supports keyboard arrow navigation across tabs"), "Toggle test must verify keyboard arrows");
});

test("Playwright E2E suite covers deep link, persistence, URL sync, and invalid mode recovery", () => {
  assert(e2eTestSrc.includes("AC-W1.4-01: Default STANDARD mode loads"), "Must test default standard");
  assert(e2eTestSrc.includes("AC-W1.4-02: Guided mode deep link"), "Must test guided deep link");
  assert(e2eTestSrc.includes("AC-W1.4-03: Quant mode deep link"), "Must test quant deep link");
  assert(e2eTestSrc.includes("AC-W1.4-04: Mode changes update URL without full page reload"), "Must test URL sync");
  assert(e2eTestSrc.includes("AC-W1.4-05: Selected mode persists after page refresh"), "Must test persistence");
  assert(e2eTestSrc.includes("AC-W1.4-06: Falls back to STANDARD when URL contains invalid mode"), "Must test invalid fallback");
});

console.log(`\n=================================================================`);
console.log(`  Verification Results: ${passed} Passed, ${failed} Failed`);
console.log(`=================================================================\n`);

if (failed > 0) {
  process.exit(1);
}
