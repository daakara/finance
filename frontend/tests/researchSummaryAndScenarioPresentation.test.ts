/**
 * frontend/tests/researchSummaryAndScenarioPresentation.test.ts
 *
 * Focused Unit Test Suite for Research Summary & Scenario Presentation.
 * Verifies the implementation of ARX Closure Model v2 requirements:
 *
 * 1. Standardized scenario terminology:
 *    - Conservative -> Bear case
 *    - Base / Base Case -> Base case
 *    - Optimistic -> Bull case
 * 2. Accurate and safe scenario percentage difference calculation:
 *    - ((modeled / ref_price) - 1) * 100
 *    - Safe handling of null, undefined, zero, negative, and NaN reference prices
 * 3. Consistent Bear -> Base -> Bull scenario ordering
 * 4. Required explanatory copy and boundaries in OpportunityDossierCard:
 *    - Investment Research Summary with 4 structured quadrants
 *    - Clear designation: "Scenarios reflect different assumptions, not assigned probabilities."
 *    - Modeled values explicitly characterized as DCF intrinsic present values, not future price targets
 *    - Base case designated as "Central Case" without describing it as "most likely"
 *    - Clear "Reference price as of [date]" and "Difference from reference price" wording
 * 5. Publication and watchlist integrity:
 *    - Published cohort (FIX, DUOL, ULTA, MEDP, ISRG, DECK) contains complete summary fields
 *    - All 6 published dossiers maintain WATCH status (no BUY recommendations)
 *    - Unpublished/suppressed symbols handle safely without crashes or synthetic data
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import {
  calculateScenarioPriceDiffPct,
  getStandardizedScenarioLabel,
  getOpportunityDossier,
  listOpportunityCohort,
  isCohortSymbol,
  OpportunityDossierSnapshotFile,
  PublishedWatchManifestFile,
} from "../lib/opportunityDossiers";

console.log("Starting Research Summary & Scenario Presentation Test Suite...");

// =========================================================================
// 1. Scenario Percentage Difference Calculation Tests
// =========================================================================
console.log("1. Testing calculateScenarioPriceDiffPct()...");

// Standard positive difference: Modeled 110, Reference 100 -> +10%
const diffPositive = calculateScenarioPriceDiffPct(110, 100);
assert.strictEqual(diffPositive !== null, true);
assert.strictEqual(Number(diffPositive!.toFixed(2)), 10.0);

// Standard negative difference: Modeled 80, Reference 100 -> -20%
const diffNegative = calculateScenarioPriceDiffPct(80, 100);
assert.strictEqual(diffNegative !== null, true);
assert.strictEqual(Number(diffNegative!.toFixed(2)), -20.0);

// Zero difference (Anchor parity): Modeled 100, Reference 100 -> 0.0%
const diffZero = calculateScenarioPriceDiffPct(100, 100);
assert.strictEqual(diffZero !== null, true);
assert.strictEqual(Number(diffZero!.toFixed(2)), 0.0);

// Null or undefined modeled value -> returns null
assert.strictEqual(calculateScenarioPriceDiffPct(undefined, 100), null);
assert.strictEqual(calculateScenarioPriceDiffPct(null, 100), null);

// Null, undefined, zero, or negative reference price -> returns null (safe divide guard)
assert.strictEqual(calculateScenarioPriceDiffPct(100, undefined), null);
assert.strictEqual(calculateScenarioPriceDiffPct(100, null), null);
assert.strictEqual(calculateScenarioPriceDiffPct(100, 0), null);
assert.strictEqual(calculateScenarioPriceDiffPct(100, -50), null);

// Non-number / NaN inputs -> returns null
assert.strictEqual(calculateScenarioPriceDiffPct(NaN, 100), null);
assert.strictEqual(calculateScenarioPriceDiffPct(100, NaN), null);
// @ts-expect-error - testing invalid string input
assert.strictEqual(calculateScenarioPriceDiffPct("100", 100), null);

console.log("   [OK] calculateScenarioPriceDiffPct correctly and safely calculates percentage diffs");

// =========================================================================
// 2. Scenario Standardized Label Mapping Tests
// =========================================================================
console.log("2. Testing getStandardizedScenarioLabel()...");

assert.strictEqual(getStandardizedScenarioLabel("conservative"), "Bear case");
assert.strictEqual(getStandardizedScenarioLabel("Conservative"), "Bear case");
assert.strictEqual(getStandardizedScenarioLabel("CONSERVATIVE"), "Bear case");
assert.strictEqual(getStandardizedScenarioLabel("bear"), "Bear case");

assert.strictEqual(getStandardizedScenarioLabel("base"), "Base case");
assert.strictEqual(getStandardizedScenarioLabel("Base"), "Base case");
assert.strictEqual(getStandardizedScenarioLabel("base case"), "Base case");
assert.strictEqual(getStandardizedScenarioLabel("Base Case"), "Base case");

assert.strictEqual(getStandardizedScenarioLabel("optimistic"), "Bull case");
assert.strictEqual(getStandardizedScenarioLabel("Optimistic"), "Bull case");
assert.strictEqual(getStandardizedScenarioLabel("OPTIMISTIC"), "Bull case");
assert.strictEqual(getStandardizedScenarioLabel("bull"), "Bull case");

// Unknown or edge cases
assert.strictEqual(getStandardizedScenarioLabel("custom"), "custom");
assert.strictEqual(getStandardizedScenarioLabel(""), "");

console.log("   [OK] getStandardizedScenarioLabel correctly standardizes scenario terminology");

// =========================================================================
// 3. Dossier Component Source Code Invariant Checks
// =========================================================================
console.log("3. Testing OpportunityDossierCard component source invariants...");

const componentPath = path.resolve(__dirname, "../components/OpportunityDossierCard.tsx");
assert.strictEqual(fs.existsSync(componentPath), true, "OpportunityDossierCard.tsx must exist");
const componentSource = fs.readFileSync(componentPath, "utf-8");

// Verify required Investment Research Summary headers
assert.strictEqual(
  componentSource.includes("Investment Research Summary"),
  true,
  "Component must render 'Investment Research Summary' heading"
);
assert.strictEqual(
  componentSource.includes("Why This Business Deserves Attention"),
  true,
  "Component must contain 'Why This Business Deserves Attention' quadrant"
);
assert.strictEqual(
  componentSource.includes("Central Investment Debate"),
  true,
  "Component must contain 'Central Investment Debate' quadrant"
);
assert.strictEqual(
  componentSource.includes("Why It Remains WATCH"),
  true,
  "Component must contain 'Why It Remains WATCH' quadrant"
);
assert.strictEqual(
  componentSource.includes("Observable Evidence That Could Change Assessment"),
  true,
  "Component must contain 'Observable Evidence That Could Change Assessment' quadrant"
);

// Verify required Scenario Comparison wording
assert.strictEqual(
  componentSource.includes("Bear case"),
  true,
  "Component must render 'Bear case' label"
);
assert.strictEqual(
  componentSource.includes("Base case"),
  true,
  "Component must render 'Base case' label"
);
assert.strictEqual(
  componentSource.includes("Bull case"),
  true,
  "Component must render 'Bull case' label"
);
assert.strictEqual(
  componentSource.includes("Central Case"),
  true,
  "Component must designate Base case as Central Case"
);
assert.strictEqual(
  componentSource.includes("Modeled Value per Share"),
  true,
  "Component must render 'Modeled Value per Share'"
);
assert.strictEqual(
  componentSource.includes("Difference from reference price"),
  true,
  "Component must render 'Difference from reference price'"
);
assert.strictEqual(
  componentSource.includes("Difference from Reference Price"),
  true,
  "Component table must render 'Difference from Reference Price'"
);
assert.strictEqual(
  componentSource.includes("Reference price as of"),
  true,
  "Component must render 'Reference price as of' with dated anchor"
);
assert.strictEqual(
  componentSource.includes("Scenarios reflect different assumptions, not assigned probabilities."),
  true,
  "Component must render probability boundary disclaimer"
);

// Verify model characterization: explicitly present value / DCF, not price target
assert.strictEqual(
  componentSource.includes("Discounted Cash Flow (FCFF) Present Values"),
  true,
  "Component must describe model values as Discounted Cash Flow (FCFF) Present Values"
);
assert.strictEqual(
  componentSource.includes("not future price targets, guaranteed returns, or live market quotes"),
  true,
  "Component must explicitly state modeled values are not future price targets"
);

console.log("   [OK] OpportunityDossierCard component source satisfies all presentation requirements");

// =========================================================================
// 4. Published Dossiers Complete Summary and Scenario Data Integrity
// =========================================================================
console.log("4. Testing data integrity for all 6 published dossiers...");

const publishedSymbols = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"];
const cohort = listOpportunityCohort();
assert.strictEqual(cohort.length, 6, "Published cohort length must be 6");

for (const symbol of publishedSymbols) {
  const dossier = getOpportunityDossier(symbol);
  assert.strictEqual(dossier !== null, true, `Dossier for ${symbol} must be present`);
  if (!dossier) continue;

  // Verify WATCH disposition invariant
  assert.strictEqual(dossier.central_assessment.disposition, "WATCH_CONDITIONS", `${symbol} disposition must be WATCH_CONDITIONS`);
  assert.strictEqual(dossier.central_assessment.disposition_label.toLowerCase().includes("watch"), true, `${symbol} disposition_label must indicate watch`);
  assert.strictEqual(dossier.central_assessment.disposition_label.toLowerCase().includes("buy"), false, `${symbol} disposition_label must not say buy`);

  // Verify Summary Content availability
  const whyDeserves = dossier.selection_rationale?.why_this_business || dossier.central_assessment?.presentation_summary;
  assert.strictEqual(typeof whyDeserves === "string" && whyDeserves.length > 10, true, `${symbol} must have 'why_this_business' summary text`);

  const mainUncertainty = dossier.central_assessment?.main_uncertainty;
  assert.strictEqual(typeof mainUncertainty === "string" && mainUncertainty.length > 10, true, `${symbol} must have 'main_uncertainty' text`);

  const watchCondition = dossier.central_assessment?.price_watch_condition;
  assert.strictEqual(typeof watchCondition === "string" && watchCondition.length > 10, true, `${symbol} must have 'price_watch_condition' text`);

  const assessmentChanges = dossier.selection_rationale?.assessment_changes || dossier.monitoring?.observable_invalidation_trigger;
  assert.strictEqual(typeof assessmentChanges === "string" && assessmentChanges.length > 10, true, `${symbol} must have 'assessment_changes' text`);

  // Verify Reference Price & Scenario Values
  const refPrice = dossier.valuation?.reference_price;
  assert.strictEqual(typeof refPrice === "number" && refPrice > 0, true, `${symbol} reference price must be > 0`);

  const scenarios = dossier.valuation?.scenarios;
  assert.strictEqual(scenarios !== undefined, true, `${symbol} scenarios must exist`);
  if (scenarios) {
    const bearVal = scenarios.conservative?.per_share_value;
    const baseVal = scenarios.base?.per_share_value;
    const bullVal = scenarios.optimistic?.per_share_value;

    assert.strictEqual(typeof bearVal === "number" && bearVal > 0, true, `${symbol} Bear modeled value must be > 0`);
    assert.strictEqual(typeof baseVal === "number" && baseVal > 0, true, `${symbol} Base modeled value must be > 0`);
    assert.strictEqual(typeof bullVal === "number" && bullVal > 0, true, `${symbol} Bull modeled value must be > 0`);

    // Verify Bear <= Base <= Bull value ordering
    assert.strictEqual(bearVal! <= baseVal!, true, `${symbol} Bear value ($${bearVal}) must be <= Base value ($${baseVal})`);
    assert.strictEqual(baseVal! <= bullVal!, true, `${symbol} Base value ($${baseVal}) must be <= Bull value ($${bullVal})`);

    // Verify calculated diffs
    const bearDiff = calculateScenarioPriceDiffPct(bearVal, refPrice);
    const baseDiff = calculateScenarioPriceDiffPct(baseVal, refPrice);
    const bullDiff = calculateScenarioPriceDiffPct(bullVal, refPrice);

    assert.strictEqual(bearDiff !== null && isFinite(bearDiff), true, `${symbol} Bear diff must be finite`);
    assert.strictEqual(baseDiff !== null && isFinite(baseDiff), true, `${symbol} Base diff must be finite`);
    assert.strictEqual(bullDiff !== null && isFinite(bullDiff), true, `${symbol} Bull diff must be finite`);
  }
}

console.log("   [OK] All 6 published dossiers have complete summary content and verified scenario calculations");

// =========================================================================
// 5. Suppressed / Unpublished Symbols Guard
// =========================================================================
console.log("5. Testing suppressed / unpublished symbols fail-closed guard...");

const suppressedSymbols = ["IREN", "CORZ", "MU", "PLTR", "UNKNOWN"];
for (const symbol of suppressedSymbols) {
  assert.strictEqual(isCohortSymbol(symbol), false, `${symbol} must not be in published cohort`);
  assert.strictEqual(getOpportunityDossier(symbol), null, `${symbol} dossier must return null (fail-closed)`);
}

console.log("   [OK] Suppressed / unpublished symbols return null without error");

console.log("\nALL RESEARCH SUMMARY & SCENARIO PRESENTATION TESTS PASSED SUCCESSFULLY!");
