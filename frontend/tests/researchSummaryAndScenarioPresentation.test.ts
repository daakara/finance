/**
 * frontend/tests/researchSummaryAndScenarioPresentation.test.ts
 *
 * Focused Behavioral & Arithmetic Test Suite for Research Summary & Scenario Presentation.
 * Verifies the implementation of ARX Closure Model v2 requirements:
 *
 * 1. Hardened calculateScenarioPriceDiffPct arithmetic & boundaries:
 *    - Finite numeric input enforcement (rejects Infinity, -Infinity, NaN, non-numbers)
 *    - Positive reference price enforcement (rejects <= 0 and non-finite)
 *    - Correct zero modeled equity value handling (returns -100.0, representing -100%)
 *    - Mathematically consistent finite negative modeled value handling (e.g. -20 vs 100 -> -120%)
 *    - Strict null return for invalid/nonfinite results
 * 2. Neutral scenario terminology & descriptive name filtering:
 *    - Standardized scenario labels (Bear case, Base case, Bull case)
 *    - isDescriptiveScenarioName cleanly distinguishes company-specific narratives from generic labels
 * 3. Component presentation & neutral shared wording:
 *    - Neutral interpretation block without unsupported generic company narratives
 *    - Explicit disclaimer that scenarios are conditional estimates, not probabilities or price targets
 *    - Central Case label used neutrally without "most likely" implication
 * 4. Architectural distinction between Valuation Suppression and Unpublished Fallback:
 *    - Unpublished symbols return null from accessor without synthetic data
 *    - Valuation suppression branch correctly suppresses forward scenario comparison
 * 5. Data integrity for all 6 published Quality Watch assets (FIX, DUOL, ULTA, MEDP, ISRG, DECK):
 *    - Order: Bear <= Base <= Bull
 *    - All retain WATCH disposition (zero BUY recommendations)
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import {
  calculateScenarioPriceDiffPct,
  getStandardizedScenarioLabel,
  isDescriptiveScenarioName,
  getOpportunityDossier,
  listOpportunityCohort,
  isCohortSymbol,
  OpportunityDossier,
} from "../lib/opportunityDossiers";

console.log("Starting Research Summary & Scenario Presentation Test Suite...");

// =========================================================================
// 1. Hardened Scenario Percentage Difference Calculation Tests
// =========================================================================
console.log("1. Testing hardened calculateScenarioPriceDiffPct()...");

// A. Standard finite positive difference: Modeled 110, Reference 100 -> +10%
const diffPos = calculateScenarioPriceDiffPct(110, 100);
assert.strictEqual(diffPos !== null, true);
assert.strictEqual(Number(diffPos!.toFixed(2)), 10.0);

// B. Standard finite negative difference: Modeled 80, Reference 100 -> -20%
const diffNeg = calculateScenarioPriceDiffPct(80, 100);
assert.strictEqual(diffNeg !== null, true);
assert.strictEqual(Number(diffNeg!.toFixed(2)), -20.0);

// C. Exact anchor parity: Modeled 100, Reference 100 -> 0.0%
const diffZero = calculateScenarioPriceDiffPct(100, 100);
assert.strictEqual(diffZero !== null, true);
assert.strictEqual(Number(diffZero!.toFixed(2)), 0.0);

// D. Zero modeled equity value: Modeled 0, Reference 100 -> -100.0%
const diffZeroModeled = calculateScenarioPriceDiffPct(0, 100);
assert.strictEqual(diffZeroModeled !== null, true);
assert.strictEqual(Number(diffZeroModeled!.toFixed(2)), -100.0);

// E. Negative modeled equity value: Modeled -20, Reference 100 -> -120.0%
const diffNegativeModeled = calculateScenarioPriceDiffPct(-20, 100);
assert.strictEqual(diffNegativeModeled !== null, true);
assert.strictEqual(Number(diffNegativeModeled!.toFixed(2)), -120.0);

// F. Input non-finiteness & boundary rejections:
assert.strictEqual(calculateScenarioPriceDiffPct(Infinity, 100), null, "modeledValue=Infinity must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(-Infinity, 100), null, "modeledValue=-Infinity must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(100, Infinity), null, "referencePrice=Infinity must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(100, -Infinity), null, "referencePrice=-Infinity must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(NaN, 100), null, "modeledValue=NaN must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(100, NaN), null, "referencePrice=NaN must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(undefined, 100), null, "modeledValue=undefined must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(100, undefined), null, "referencePrice=undefined must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(null, 100), null, "modeledValue=null must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(100, null), null, "referencePrice=null must return null");

// G. Strictly positive reference price requirement:
assert.strictEqual(calculateScenarioPriceDiffPct(100, 0), null, "referencePrice=0 must return null");
assert.strictEqual(calculateScenarioPriceDiffPct(100, -50), null, "referencePrice < 0 must return null");

// H. Type safety (non-numbers):
// @ts-expect-error - testing string input
assert.strictEqual(calculateScenarioPriceDiffPct("100", 100), null);
// @ts-expect-error - testing string input
assert.strictEqual(calculateScenarioPriceDiffPct(100, "100"), null);

console.log("   [OK] calculateScenarioPriceDiffPct adheres to hardened arithmetic boundaries");

// =========================================================================
// 2. Scenario Standardized Label Mapping & Descriptive Filter Tests
// =========================================================================
console.log("2. Testing scenario label standardization & descriptive narrative filter...");

// A. Standardized presentation labels
assert.strictEqual(getStandardizedScenarioLabel("conservative"), "Bear case");
assert.strictEqual(getStandardizedScenarioLabel("Conservative"), "Bear case");
assert.strictEqual(getStandardizedScenarioLabel("bear"), "Bear case");

assert.strictEqual(getStandardizedScenarioLabel("base"), "Base case");
assert.strictEqual(getStandardizedScenarioLabel("Base Case"), "Base case");

assert.strictEqual(getStandardizedScenarioLabel("optimistic"), "Bull case");
assert.strictEqual(getStandardizedScenarioLabel("Optimistic"), "Bull case");
assert.strictEqual(getStandardizedScenarioLabel("bull"), "Bull case");

// B. isDescriptiveScenarioName filter
assert.strictEqual(isDescriptiveScenarioName("Conservative"), false);
assert.strictEqual(isDescriptiveScenarioName("conservative"), false);
assert.strictEqual(isDescriptiveScenarioName("Base"), false);
assert.strictEqual(isDescriptiveScenarioName("base"), false);
assert.strictEqual(isDescriptiveScenarioName("Base Case"), false);
assert.strictEqual(isDescriptiveScenarioName("Optimistic"), false);
assert.strictEqual(isDescriptiveScenarioName("optimistic"), false);
assert.strictEqual(isDescriptiveScenarioName("Bear case"), false);
assert.strictEqual(isDescriptiveScenarioName("Bull case"), false);
assert.strictEqual(isDescriptiveScenarioName(""), false);
assert.strictEqual(isDescriptiveScenarioName(null), false);
assert.strictEqual(isDescriptiveScenarioName(undefined), false);

// Approved company-specific narratives must pass
assert.strictEqual(
  isDescriptiveScenarioName("Hyperscale capex moderation; operating margin normalizes toward historical 12.5%"),
  true
);
assert.strictEqual(
  isDescriptiveScenarioName("Paid subscriber conversion slows; operating margin flat at 20.0%"),
  true
);

console.log("   [OK] Scenario labeling and descriptive narrative filtering verified");

// =========================================================================
// 3. Dossier Component Source Code Invariants & Neutral Shared Copy
// =========================================================================
console.log("3. Testing OpportunityDossierCard component source invariants...");

const componentPath = path.resolve(__dirname, "../components/OpportunityDossierCard.tsx");
assert.strictEqual(fs.existsSync(componentPath), true, "OpportunityDossierCard.tsx must exist");
const componentSource = fs.readFileSync(componentPath, "utf-8");

// Neutral shared explanatory copy
assert.strictEqual(
  componentSource.includes(
    "Bear, base, and bull cases use the assumptions shown below. Their modeled values are conditional estimates, not assigned probabilities or future price targets."
  ),
  true,
  "Component must render neutral shared explanatory text"
);

// Prohibit unsupported generic narratives from shared interpretation block
assert.strictEqual(
  componentSource.includes("Downside stress case assuming customer capex moderation"),
  false,
  "Generic capex moderation narrative must not be applied universally across shared block"
);
assert.strictEqual(
  componentSource.includes("accelerated market adoption and sustained operating leverage"),
  false,
  "Generic accelerated market adoption narrative must not be applied universally across shared block"
);

// Verify use of isDescriptiveScenarioName in component
assert.strictEqual(
  componentSource.includes("isDescriptiveScenarioName(valuation.scenarios.conservative.name)"),
  true,
  "Component must use isDescriptiveScenarioName for conservative scenario"
);
assert.strictEqual(
  componentSource.includes("isDescriptiveScenarioName(valuation.scenarios.base.name)"),
  true,
  "Component must use isDescriptiveScenarioName for base scenario"
);
assert.strictEqual(
  componentSource.includes("isDescriptiveScenarioName(valuation.scenarios.optimistic.name)"),
  true,
  "Component must use isDescriptiveScenarioName for optimistic scenario"
);

// Verify Central Case label is used neutrally
assert.strictEqual(
  componentSource.includes("Central Case"),
  true,
  "Base case must be designated neutrally as Central Case"
);
assert.strictEqual(
  componentSource.includes("most likely"),
  false,
  "Component must not imply base case is the most likely outcome"
);

// Verify explicit probability and price target boundaries
assert.strictEqual(
  componentSource.includes("Scenarios reflect different assumptions, not assigned probabilities."),
  true
);
assert.strictEqual(
  componentSource.includes("not future price targets, guaranteed returns, or live market quotes"),
  true
);

console.log("   [OK] OpportunityDossierCard source satisfies presentation invariants and neutral copy");

// =========================================================================
// 4. Distinction Between Valuation Suppression and Unpublished Fallback
// =========================================================================
console.log("4. Testing distinction between valuation suppression and unpublished fallback...");

// A. Unpublished symbol: does not exist in published cohort, accessor returns null
const unpublishedSymbols = ["UNKNOWN", "IREN", "CORZ", "MU", "PLTR"];
for (const sym of unpublishedSymbols) {
  assert.strictEqual(isCohortSymbol(sym), false, `${sym} must not be in published cohort`);
  assert.strictEqual(getOpportunityDossier(sym), null, `${sym} must return null from public accessor`);
}

// B. Valuation Suppression: verify component contract for suppressed dossiers
// In OpportunityDossierCard, when valuation.valuation_status === "VALUATION_SUPPRESSED",
// the component renders the suppressed notice and excludes the conditional scenario comparison bracket.
assert.strictEqual(
  componentSource.includes('valuation.valuation_status === "VALUATION_SUPPRESSED"'),
  true,
  "Component must inspect VALUATION_SUPPRESSED status"
);
assert.strictEqual(
  componentSource.includes("{isSuppressed ? ("),
  true,
  "Component must branch on isSuppressed"
);
assert.strictEqual(
  componentSource.includes("VALUATION SUPPRESSED"),
  true,
  "Component must render fail-closed suppression notice when suppressed"
);

console.log("   [OK] Architectural distinction between unpublished fallback and valuation suppression verified");

// =========================================================================
// 5. Published Dossiers Data Integrity & Scenario Order
// =========================================================================
console.log("5. Testing data integrity for all 7 published Quality Watch dossiers...");

const publishedSymbols = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK", "ALC"];
const cohort = listOpportunityCohort();
assert.strictEqual(cohort.length, 7, "Published cohort length must be exactly 7");

for (const symbol of publishedSymbols) {
  const dossier = getOpportunityDossier(symbol);
  assert.strictEqual(dossier !== null, true, `Dossier for ${symbol} must be present`);
  if (!dossier) continue;

  // Verify WATCH disposition invariant
  assert.strictEqual(dossier.central_assessment.disposition, "WATCH_CONDITIONS");
  assert.strictEqual(dossier.central_assessment.disposition_label.toLowerCase().includes("watch"), true);
  assert.strictEqual(dossier.central_assessment.disposition_label.toLowerCase().includes("buy"), false);

  // Verify Summary Content availability
  const whyDeserves = dossier.selection_rationale?.why_this_business || dossier.central_assessment?.presentation_summary;
  assert.strictEqual(typeof whyDeserves === "string" && whyDeserves.length > 10, true);

  const mainUncertainty = dossier.central_assessment?.main_uncertainty;
  assert.strictEqual(typeof mainUncertainty === "string" && mainUncertainty.length > 10, true);

  const watchCondition = dossier.central_assessment?.price_watch_condition;
  assert.strictEqual(typeof watchCondition === "string" && watchCondition.length > 10, true);

  const assessmentChanges = dossier.selection_rationale?.assessment_changes || dossier.monitoring?.observable_invalidation_trigger;
  assert.strictEqual(typeof assessmentChanges === "string" && assessmentChanges.length > 10, true);

  // Verify Reference Price & Scenario Values
  const refPrice = dossier.valuation?.reference_price;
  assert.strictEqual(typeof refPrice === "number" && refPrice > 0, true);

  const scenarios = dossier.valuation?.scenarios;
  assert.strictEqual(scenarios !== undefined, true);
  if (scenarios) {
    const bearVal = scenarios.conservative?.per_share_value;
    const baseVal = (scenarios.base || scenarios.base_dcf || scenarios.base_multiple)?.per_share_value;
    const bullVal = scenarios.optimistic?.per_share_value;

    assert.strictEqual(typeof bearVal === "number" && bearVal > 0, true);
    assert.strictEqual(typeof baseVal === "number" && baseVal > 0, true);
    assert.strictEqual(typeof bullVal === "number" && bullVal > 0, true);

    // Verify Bear <= Base <= Bull value ordering
    assert.strictEqual(bearVal! <= baseVal!, true);
    assert.strictEqual(baseVal! <= bullVal!, true);

    // Verify calculated diffs
    const bearDiff = calculateScenarioPriceDiffPct(bearVal, refPrice);
    const baseDiff = calculateScenarioPriceDiffPct(baseVal, refPrice);
    const bullDiff = calculateScenarioPriceDiffPct(bullVal, refPrice);

    assert.strictEqual(bearDiff !== null && isFinite(bearDiff), true);
    assert.strictEqual(baseDiff !== null && isFinite(baseDiff), true);
    assert.strictEqual(bullDiff !== null && isFinite(bullDiff), true);
  }
}

console.log("   [OK] All 7 published dossiers have complete summary content and verified scenario calculations");

console.log("\nALL RESEARCH SUMMARY & SCENARIO PRESENTATION TESTS PASSED SUCCESSFULLY!");
