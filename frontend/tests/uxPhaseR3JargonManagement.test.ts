/**
 * ARX Terminal — Phase R3: Progressive Jargon Management & JargonTip Test Suite
 * Directive: ARX-UX-CONTINUATION-004 (Priority 2 / UX-R3)
 *
 * Verifies:
 * 1. JARGON_GLOSSARY canonical terminology dictionary integrity.
 * 2. getJargonDefinition lookup robustness across case, spaces, and acronym variants.
 * 3. Vernacular mode contract (PLAIN_ENGLISH default, event/storage synchronization).
 * 4. Mathematical invariance: Vernacular mode switching never alters numerical calculations.
 * 5. JargonTip WCAG 2.1 AA accessibility contracts (role="tooltip", aria-describedby, keyboard listeners).
 * 6. Hub components integration across CompositeConvictionCard, OptimalEntryExitCard, AssetFactorRadar, RiskMetricsCard, PreFlightReadinessBar.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { JARGON_GLOSSARY, getJargonDefinition } from "../lib/jargonGlossary";
import { VERNACULAR_STORAGE_KEY, VERNACULAR_EVENT_NAME } from "../lib/useVernacularMode";
import { evaluatePreFlightChecks, PreFlightChecklistInputs } from "../lib/preflightEvaluator";

const ROOT_DIR = path.resolve(__dirname, "..");

console.log("==================================================================");
console.log("ARX PHASE R3: PROGRESSIVE JARGON MANAGEMENT TEST SUITE");
console.log("Directive: ARX-UX-CONTINUATION-004 (UX-R3)");
console.log("==================================================================\n");

// ------------------------------------------------------------------
// 1. JARGON_GLOSSARY Canonical Dictionary Integrity
// ------------------------------------------------------------------
console.log("1. Verifying JARGON_GLOSSARY dictionary completeness and structure...");

const requiredTerms = [
  "VCP",
  "SMA50",
  "EMA20",
  "VWAP",
  "PIOTROSKI",
  "ALTMAN_Z",
  "CORNISH_FISHER_VAR",
  "R_MULTIPLE",
  "SORTINO",
  "AMIHUD",
  "RVOL",
  "PIVOT",
  "STOP_LOSS",
  "STAGE_2",
  "CONFLUENCE",
];

for (const termKey of requiredTerms) {
  const item = JARGON_GLOSSARY[termKey];
  assert.ok(item, `Term '${termKey}' must exist in JARGON_GLOSSARY`);
  assert.ok(item.term && item.term.length > 0, `'${termKey}' must have a valid term string`);
  assert.ok(item.shortName && item.shortName.length > 0, `'${termKey}' must have a valid shortName`);
  assert.ok(
    ["TECHNICAL", "FUNDAMENTAL", "RISK", "EXECUTION", "MACRO"].includes(item.category),
    `'${termKey}' must have a valid category, got ${item.category}`
  );
  assert.ok(
    item.plainEnglishSummary && item.plainEnglishSummary.length > 10,
    `'${termKey}' must have a meaningful plainEnglishSummary`
  );
  assert.ok(
    item.detailedDefinition && item.detailedDefinition.length > 15,
    `'${termKey}' must have a detailedDefinition`
  );
  console.log(`   [PASS] '${termKey}': ${item.term} (${item.category}) validated`);
}

// ------------------------------------------------------------------
// 2. getJargonDefinition Lookup Robustness
// ------------------------------------------------------------------
console.log("\n2. Verifying getJargonDefinition lookup with normalized variants...");

const lookupCases = [
  { input: "vcp", expectedShort: "VCP" },
  { input: "VCP", expectedShort: "VCP" },
  { input: "sma-50", expectedShort: "50 SMA" },
  { input: "50 SMA", expectedShort: "50 SMA" },
  { input: "vwap", expectedShort: "VWAP" },
  { input: "piotroski", expectedShort: "Piotroski F" },
  { input: "Piotroski F", expectedShort: "Piotroski F" },
  { input: "altman-z", expectedShort: "Altman Z" },
  { input: "cornish-fisher-var", expectedShort: "M-VaR" },
  { input: "R_MULTIPLE", expectedShort: "R-Multiple" },
  { input: "r-multiple", expectedShort: "R-Multiple" },
  { input: "sortino", expectedShort: "Sortino" },
  { input: "amihud", expectedShort: "Amihud Ratio" },
  { input: "stop_loss", expectedShort: "Stop Loss" },
  { input: "stage-2", expectedShort: "Stage 2" },
  { input: "confluence", expectedShort: "Confluence" },
];

for (const testCase of lookupCases) {
  const def = getJargonDefinition(testCase.input);
  assert.ok(def, `getJargonDefinition('${testCase.input}') must resolve to a definition`);
  assert.strictEqual(
    def.shortName,
    testCase.expectedShort,
    `getJargonDefinition('${testCase.input}') expected shortName '${testCase.expectedShort}', got '${def.shortName}'`
  );
  console.log(`   [PASS] Lookup '${testCase.input}' -> '${def.shortName}'`);
}

// Non-existent key returns undefined gracefully
assert.strictEqual(getJargonDefinition("NON_EXISTENT_METRIC_XYZ"), undefined);
console.log("   [PASS] Non-existent term safely returns undefined");

// ------------------------------------------------------------------
// 3. Vernacular Mode Storage & Event Constants
// ------------------------------------------------------------------
console.log("\n3. Verifying useVernacularMode contract and synchronization keys...");

assert.strictEqual(VERNACULAR_STORAGE_KEY, "ARX_VERNACULAR_MODE");
assert.strictEqual(VERNACULAR_EVENT_NAME, "finance:vernacular-change");

const hookSource = fs.readFileSync(path.join(ROOT_DIR, "lib/useVernacularMode.ts"), "utf-8");
assert.ok(hookSource.includes('"PLAIN_ENGLISH"'), "Default mode must be PLAIN_ENGLISH");
assert.ok(hookSource.includes("localStorage.getItem"), "Must read from localStorage");
assert.ok(hookSource.includes("window.dispatchEvent"), "Must dispatch window custom event");
assert.ok(hookSource.includes('window.addEventListener("storage"'), "Must synchronize across browser tabs");
console.log("   [PASS] useVernacularMode synchronization contracts confirmed");

// ------------------------------------------------------------------
// 4. Mathematical Invariance: Zero Impact on Financial Calculations
// ------------------------------------------------------------------
console.log("\n4. Verifying Mathematical Invariance across Vernacular Modes...");

const mockInputs: PreFlightChecklistInputs = {
  symbol: "NVDA",
  setupPattern: "Clear Stage 2 Minervini Setup with VCP Contraction",
  currentPrice: 135.5,
  optimalEntryMin: 134.0,
  optimalEntryMax: 136.0,
  stopLoss: 127.5,
  takeProfit1: 152.0,
  riskRewardRatio: 2.07,
  vix: 15.2,
};

const plainEval = evaluatePreFlightChecks(mockInputs, "PLAIN_ENGLISH");
const quantEval = evaluatePreFlightChecks(mockInputs, "PRO_QUANT");

// Mathematical attributes MUST BE 100% identical
assert.strictEqual(plainEval.passedCount, quantEval.passedCount, "Passed checks count must be identical");
assert.strictEqual(plainEval.convictionPct, quantEval.convictionPct, "Conviction percentage must be identical");
assert.strictEqual(plainEval.isActionableGranted, quantEval.isActionableGranted, "Actionability must be identical");
assert.strictEqual(plainEval.isCleared, quantEval.isCleared, "Clearance state must be identical");
assert.strictEqual(plainEval.checks.length, quantEval.checks.length, "Number of checks must be identical");

// Only labels and descriptions change for vernacular accessibility
for (let i = 0; i < plainEval.checks.length; i++) {
  assert.strictEqual(
    plainEval.checks[i].status,
    quantEval.checks[i].status,
    `Check ${i} status must be strictly identical across modes`
  );
  assert.strictEqual(
    plainEval.checks[i].id,
    quantEval.checks[i].id,
    `Check ${i} id must be strictly identical across modes`
  );
}
console.log("   [PASS] Mathematical invariance confirmed: Zero numerical divergence between modes");

// ------------------------------------------------------------------
// 5. JargonTip WCAG 2.1 AA Accessibility Contract
// ------------------------------------------------------------------
console.log("\n5. Verifying JargonTip component accessibility implementation...");

const jargonTipSource = fs.readFileSync(path.join(ROOT_DIR, "components/ui/JargonTip.tsx"), "utf-8");

assert.ok(jargonTipSource.includes('role="tooltip"'), "Must render tooltip with role='tooltip'");
assert.ok(jargonTipSource.includes("aria-describedby"), "Trigger button must have aria-describedby binding");
assert.ok(jargonTipSource.includes("aria-expanded"), "Trigger button must have aria-expanded attribute");
assert.ok(jargonTipSource.includes('e.key === "Escape"'), "Must dismiss on Escape key");
assert.ok(jargonTipSource.includes("handleClickOutside"), "Must dismiss on click-outside");
assert.ok(jargonTipSource.includes("border-dotted"), "Must include accessible dotted underline visual cue");
console.log("   [PASS] JargonTip WCAG 2.1 AA attributes and keyboard dismiss verified");

// ------------------------------------------------------------------
// 6. Hub Components Integration Verification
// ------------------------------------------------------------------
console.log("\n6. Verifying component integration across primary cards...");

const componentsToCheck = [
  {
    file: "components/CompositeConvictionCard.tsx",
    expectedUses: ["useVernacularMode", "JargonTip", 'term="CONFLUENCE"'],
  },
  {
    file: "components/OptimalEntryExitCard.tsx",
    expectedUses: ["useVernacularMode", "JargonTip", 'term="R_MULTIPLE"', 'term="STOP_LOSS"'],
  },
  {
    file: "components/AssetFactorRadar.tsx",
    expectedUses: ["useVernacularMode", "JargonTip", 'term="PIOTROSKI"'],
  },
  {
    file: "components/RiskMetricsCard.tsx",
    expectedUses: ["useVernacularMode", "JargonTip", 'term="CORNISH_FISHER_VAR"', 'term="SORTINO"'],
  },
  {
    file: "components/PreFlightReadinessBar.tsx",
    expectedUses: ["useVernacularMode"],
  },
];

for (const comp of componentsToCheck) {
  const fullPath = path.join(ROOT_DIR, comp.file);
  const src = fs.readFileSync(fullPath, "utf-8");
  for (const expected of comp.expectedUses) {
    assert.ok(
      src.includes(expected),
      `${comp.file} must contain '${expected}'`
    );
  }
  console.log(`   [PASS] ${comp.file}: Cleanly integrated with ${comp.expectedUses.join(", ")}`);
}

console.log("\n==================================================================");
console.log("ALL 6 ARX UX-R3 PROGRESSIVE JARGON TESTS PASSED (100% OK)");
console.log("==================================================================");
