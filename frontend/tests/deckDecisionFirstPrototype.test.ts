/**
 * frontend/tests/deckDecisionFirstPrototype.test.ts
 *
 * Comprehensive Behavioral & Rendered Component Test Suite for the
 * DECK Decision-First Research Page Prototype.
 * Governed by ARX Closure Model v2.
 *
 * Enforces:
 * 1. Authoritative Data Integrity with Control DECK Payload:
 *    - Reference price: $85.72 as of 2026-10-16
 *    - Reverse-DCF solved 5Y revenue CAGR: -2.34%
 *    - Bear / Base / Bull scenario values ($73.22, $127.49, $209.87) and differences (-14.6%, +48.7%, +144.8%)
 *    - Strict scenario ordering: Bear case precedes Base case, Base case precedes Bull case
 * 2. Complete Elimination of Invented Fallbacks & Hardcoded Numbers:
 *    - Missing solved CAGR renders "Unavailable" (never -2.34%)
 *    - Missing scenarios render "Unavailable" (never $73.22, etc.)
 *    - Nonfinite numbers (NaN, Infinity) render "Unavailable"
 *    - Valid numeric zero is preserved ("0.00%", "$0.00")
 *    - Changing valid financial inputs in the payload dynamically updates rendered UI
 *    - Valuation suppression renders "Valuation Suppressed" and prevents numeric fair values
 *    - Complete absence of historical fallback strings when fields are omitted
 * 3. Corrected Approved Research Interpretation:
 *    - DTC deceleration <10% is treated as a monitored weakening threshold requiring investigation, not proof of brand fatigue
 *    - Margin deterioration <50% is treated as monitored invalidation requiring investigation, not proof of brand equity loss
 *    - Wholesale channel contraction is NOT invented as a third formal invalidation condition
 *    - Gross margin thresholds keep distinct meanings (strengthening >55%, invalidation <50%)
 *    - Review trigger correctly identifies Q2 FY27 earnings release
 *    - Rejection of process tags ("A. Verdict", "Primary Research Disposition", etc.)
 * 4. Component Isolation & Disposition Contracts:
 *    - OpportunityDossierCard routes DECK with WATCH_CONDITIONS to DeckDecisionFirstView
 *    - Non-WATCH disposition safely falls back to standard OpportunityDossierCard presentation
 *    - DeckDecisionFirstView renders dynamic disposition badge without hardcoding WATCH
 *    - Non-DECK published dossiers (FIX, ULTA, etc.) strictly retain standard presentation
 *    - Unpublished symbols fail closed to null
 * 5. Accessibility & Semantics:
 *    - Interactive touch targets satisfy >= 44px (min-h-[44px])
 *    - Visible focus ring classes present (focus:ring-2 focus:ring-cyan-400)
 *    - Accordions contain aria-expanded and aria-controls attributes
 */

import assert from "node:assert";
import React from "react";
// @ts-ignore - react-dom/server type declaration mapping under node16
import { renderToStaticMarkup } from "react-dom/server";
import OpportunityDossierCard from "../components/OpportunityDossierCard";
import DeckDecisionFirstView from "../components/DeckDecisionFirstView";
import {
  getOpportunityDossier,
  listOpportunityCohort,
  calculateScenarioPriceDiffPct,
  OpportunityDossier,
} from "../lib/opportunityDossiers";

console.log("Starting DECK Decision-First Prototype Comprehensive Behavioral Test Suite...");

// =========================================================================
// 1. Control Test: Authoritative DECK Research Dossier Payload
// =========================================================================
console.log("1. Verifying authoritative DECK research dossier control payload...");

const deckDossier = getOpportunityDossier("DECK");
assert.ok(deckDossier, "DECK must be available as an accepted Quality Watch dossier");

assert.strictEqual(deckDossier.identity.symbol, "DECK");
assert.strictEqual(deckDossier.identity.company_name, "Deckers Outdoor Corp.");
assert.strictEqual(deckDossier.central_assessment.disposition, "WATCH_CONDITIONS");
assert.strictEqual(deckDossier.valuation.reference_price, 85.72);
assert.strictEqual(deckDossier.valuation.reference_date, "2026-10-16");
assert.strictEqual(deckDossier.valuation.solved_5y_cagr_pct, -2.34);

const scenarios = deckDossier.valuation.scenarios!;
assert.strictEqual(scenarios.conservative.per_share_value, 73.22);
assert.strictEqual(scenarios.base.per_share_value, 127.49);
assert.strictEqual(scenarios.optimistic.per_share_value, 209.87);

// Render control DECK through OpportunityDossierCard
const controlMarkup = renderToStaticMarkup(
  React.createElement(OpportunityDossierCard, { dossier: deckDossier })
);

// A. Core values rendered
assert.ok(controlMarkup.includes("$85.72"), "Must render reference price $85.72");
assert.ok(controlMarkup.includes("-2.34%"), "Must render solved 5Y CAGR -2.34%");
assert.ok(controlMarkup.includes("$73.22"), "Must render Bear case value $73.22");
assert.ok(controlMarkup.includes("-14.6%"), "Must render Bear difference -14.6%");
assert.ok(controlMarkup.includes("$127.49"), "Must render Base case value $127.49");
assert.ok(controlMarkup.includes("+48.7%"), "Must render Base difference +48.7%");
assert.ok(controlMarkup.includes("$209.87"), "Must render Bull case value $209.87");
assert.ok(controlMarkup.includes("+144.8%"), "Must render Bull difference +144.8%");

// B. Scenario ordering check: Bear case < Base case < Bull case in rendered DOM
const bearIdx = controlMarkup.indexOf("Bear case");
const baseIdx = controlMarkup.indexOf("Base case");
const bullIdx = controlMarkup.indexOf("Bull case");
assert.ok(bearIdx !== -1 && baseIdx !== -1 && bullIdx !== -1, "All three scenario labels must be rendered");
assert.ok(bearIdx < baseIdx, "Bear case must appear before Base case in DOM");
assert.ok(baseIdx < bullIdx, "Base case must appear before Bull case in DOM");

// C. Rejection of unapproved wording and process labels
assert.strictEqual(controlMarkup.includes("A. Verdict"), false, "Must not include 'A. Verdict' process label");
assert.strictEqual(controlMarkup.includes("B. One-Sentence Thesis"), false, "Must not include 'B. One-Sentence Thesis' label");
assert.strictEqual(controlMarkup.includes("C. What Is Priced In?"), false, "Must not include 'C. What Is Priced In?' label");
assert.strictEqual(controlMarkup.includes("Primary Research Disposition"), false, "Must not include 'Primary Research Disposition'");
assert.strictEqual(controlMarkup.includes("Disciplined Expectation Check"), false, "Must not include 'Disciplined Expectation Check'");
assert.strictEqual(controlMarkup.includes("3. Wholesale Channel Contraction"), false, "Must not invent third invalidation trigger");
assert.strictEqual(controlMarkup.includes("Wholesale revenue turning negative"), false, "Must not invent wholesale revenue turning negative invalidation threshold");
assert.strictEqual(controlMarkup.includes("insufficient margin of safety"), false, "Must not include unsupported claim of insufficient margin of safety");
assert.strictEqual(controlMarkup.includes("brand fatigue"), false, "Must not claim DTC deceleration proves brand fatigue");
assert.strictEqual(controlMarkup.includes("must hold above 50.0%–55.0%"), false, "Must not merge gross margin thresholds into ambiguous range");
assert.strictEqual(controlMarkup.includes("with strong gross margin contribution"), false, "Must not assert unevidenced channel gross margin contribution");
assert.strictEqual(controlMarkup.includes("What Could This Become?"), false, "Section H ('What Could This Become?') must remain omitted");

// D. Approved interpretation confirmed
assert.ok(
  controlMarkup.includes("Growth below the existing 10.0% YoY monitoring threshold would weaken the thesis and require reassessment; the underlying cause would need investigation."),
  "Must include approved non-causal meaning for DTC deceleration"
);
assert.ok(
  controlMarkup.includes("Consolidated gross margin falling below 50.0% is a monitored invalidation condition that would weaken the thesis and require investigation of underlying cost, inventory, and pricing drivers."),
  "Must include approved non-causal meaning for margin deterioration"
);
assert.ok(
  controlMarkup.includes("Q2 FY27 earnings release"),
  "Must correctly state Q2 FY27 earnings release as next monitoring trigger"
);
assert.ok(
  controlMarkup.includes("Decision-First Prototype (DECK)"),
  "Must display clear prototype badge"
);

console.log("[OK] Control DECK rendering and approved interpretations verified.");

// =========================================================================
// 2. Controlled Fixture Tests: No Invented Fallbacks & Dynamic Data Binding
// =========================================================================
console.log("2. Running controlled fixture tests for fallback elimination and edge cases...");

function cloneDossier(d: OpportunityDossier): OpportunityDossier {
  return JSON.parse(JSON.stringify(d));
}

// A. Missing solved CAGR
{
  const fixture = cloneDossier(deckDossier);
  fixture.valuation.solved_5y_cagr_pct = null;
  fixture.central_assessment.presentation_summary = "Deckers Outdoor displays balance sheet strength and continued HOKA momentum.";
  const html = renderToStaticMarkup(React.createElement(DeckDecisionFirstView, { dossier: fixture }));
  assert.ok(html.includes("Unavailable"), "Missing solved CAGR must render 'Unavailable'");
  assert.strictEqual(html.includes("-2.34%"), false, "Must not substitute historical -2.34% when CAGR is null");
}

// B. Missing scenario (conservative missing)
{
  const fixture = cloneDossier(deckDossier);
  delete (fixture.valuation.scenarios as any).conservative;
  const html = renderToStaticMarkup(React.createElement(DeckDecisionFirstView, { dossier: fixture }));
  assert.ok(html.includes("Unavailable"), "Missing conservative scenario must render 'Unavailable'");
  assert.strictEqual(html.includes("$73.22"), false, "Must not substitute historical $73.22 when scenario is missing");
  assert.strictEqual(html.includes("-14.6%"), false, "Must not substitute historical -14.6% when scenario is missing");
}

// C. Nonfinite / Invalid Reference Price
{
  const fixture = cloneDossier(deckDossier);
  fixture.valuation.reference_price = NaN;
  const html = renderToStaticMarkup(React.createElement(DeckDecisionFirstView, { dossier: fixture }));
  assert.ok(html.includes("Unavailable"), "NaN reference price must render 'Unavailable'");
  assert.strictEqual(html.includes("NaN"), false, "Must never render literal NaN");
  assert.strictEqual(html.includes("Infinity"), false, "Must never render literal Infinity");
}

// D. Valid Numeric Zero Preservation
{
  const fixture = cloneDossier(deckDossier);
  fixture.valuation.solved_5y_cagr_pct = 0;
  fixture.valuation.scenarios!.conservative.per_share_value = 0;
  fixture.financial_profile.total_financial_debt = 0;
  fixture.financial_profile.is_zero_debt = false;

  const html = renderToStaticMarkup(React.createElement(DeckDecisionFirstView, { dossier: fixture }));
  assert.ok(html.includes("0.00%"), "Valid numeric 0 CAGR must render as '0.00%', not 'Unavailable'");
  assert.ok(html.includes("$0.00"), "Valid numeric 0 per share value must render as '$0.00', not 'Unavailable'");
}

// =========================================================================
// Mandatory Control 1: Changed-Input Control (Dynamic Data Binding)
// =========================================================================
{
  const fixture = cloneDossier(deckDossier);
  // 1. Financial profile & balance sheet updates
  fixture.financial_profile.net_debt_bridge = -2000000000; // -$2,000.0M (Net Cash)
  fixture.financial_profile.total_financial_debt = 50000000; // $50.00M debt
  fixture.financial_profile.is_zero_debt = false;

  // 2. Valuation inputs & Reverse-DCF assumptions
  fixture.valuation.reference_price = 100.0;
  fixture.valuation.reference_date = "2026-11-15";
  fixture.valuation.solved_5y_cagr_pct = 5.5;
  fixture.valuation.scenarios!.conservative.per_share_value = 90.0;
  fixture.valuation.scenarios!.base.per_share_value = 140.0;
  fixture.valuation.scenarios!.optimistic.per_share_value = 220.0;
  if (!fixture.valuation.reverse_dcf_parameters) {
    fixture.valuation.reverse_dcf_parameters = {} as any;
  }
  fixture.valuation.reverse_dcf_parameters!.operating_margin_pct = 25.0;
  fixture.valuation.reverse_dcf_parameters!.wacc_pct = 10.5;

  // 3. Monitoring & presentation narratives
  fixture.monitoring.next_monitoring_trigger = "Q3 FY27 earnings call in February 2027";
  fixture.central_assessment.main_uncertainty = "Testing custom uncertainty around supply chain lead times";
  fixture.monitoring.price_watch_condition = "Custom price watch condition: pullback below $70.00";
  fixture.central_assessment.presentation_summary = "Custom dynamic presentation summary: HOKA direct-to-consumer acceleration testing.";

  const html = renderToStaticMarkup(React.createElement(DeckDecisionFirstView, { dossier: fixture }));

  // Assert all updated values are rendered
  assert.ok(html.includes("$100.00"), "Must render updated reference price $100.00");
  assert.ok(html.includes("2026-11-15"), "Must render updated reference date 2026-11-15");
  assert.ok(html.includes("5.50%"), "Must render updated solved CAGR 5.50%");
  assert.ok(html.includes("$90.00"), "Must render updated Bear value $90.00");
  assert.ok(html.includes("-10.0%"), "Must render calculated Bear diff -10.0%");
  assert.ok(html.includes("$140.00"), "Must render updated Base value $140.00");
  assert.ok(html.includes("+40.0%"), "Must render calculated Base diff +40.0%");
  assert.ok(html.includes("$220.00"), "Must render updated Bull value $220.00");
  assert.ok(html.includes("+120.0%"), "Must render calculated Bull diff +120.0%");

  assert.ok(html.includes("-$2000.0M (Net Cash)"), "Must render updated Net Debt Bridge -$2000.0M (Net Cash)");
  assert.ok(html.includes("$50.0M debt"), "Must render updated total debt $50.0M debt");
  assert.ok(html.includes("25.00% operating margin"), "Must render updated operating margin 25.00%");
  assert.ok(html.includes("10.50% WACC"), "Must render updated WACC 10.50%");
  assert.ok(html.includes("Q3 FY27 earnings call in February 2027"), "Must render updated next monitoring trigger");
  assert.ok(html.includes("Testing custom uncertainty around supply chain lead times"), "Must render updated decisive unknown");
  assert.ok(html.includes("Custom price watch condition: pullback below $70.00"), "Must render updated watch condition");
  assert.ok(html.includes("Custom dynamic presentation summary: HOKA direct-to-consumer acceleration testing."), "Must render updated presentation summary");

  // Assert ZERO occurrences of old numbers or historical fallbacks
  assert.strictEqual(html.includes("$85.72"), false, "Old reference price $85.72 must not appear");
  assert.strictEqual(html.includes("-2.34%"), false, "Old CAGR -2.34% must not appear");
  assert.strictEqual(html.includes("$73.22"), false, "Old Bear value $73.22 must not appear");
  assert.strictEqual(html.includes("$127.49"), false, "Old Base value $127.49 must not appear");
  assert.strictEqual(html.includes("$209.87"), false, "Old Bull value $209.87 must not appear");
  assert.strictEqual(html.includes("1,602.6M"), false, "Old net cash 1,602.6M must not appear");
  assert.strictEqual(html.includes("1.60B"), false, "Hardcoded 1.60B must not appear");
  assert.strictEqual(html.includes("zero funded debt"), false, "Old zero debt string must not appear when debt is present");
  assert.strictEqual(html.includes("Q2 FY27 earnings release"), false, "Old earnings trigger must not appear when replaced");
  assert.strictEqual(html.includes("+17.3%"), false, "Hardcoded historical DTC growth +17.3% must not appear");
  assert.strictEqual(html.includes("+2.9%"), false, "Hardcoded historical wholesale growth +2.9% must not appear");
}

// =========================================================================
// Mandatory Control 2: Missing-Content Control (Honest Unavailable / Unresolved states)
// =========================================================================
{
  const fixture = cloneDossier(deckDossier);
  // Omit optional / narrative fields
  fixture.central_assessment.presentation_summary = "";
  fixture.valuation.solved_5y_cagr_pct = null;
  fixture.valuation.scenarios = null;
  fixture.monitoring.next_monitoring_trigger = "";
  fixture.central_assessment.main_uncertainty = "";
  fixture.monitoring.price_watch_condition = "";
  fixture.financial_profile.net_debt_bridge = null;
  fixture.financial_profile.total_financial_debt = null;
  fixture.financial_profile.is_zero_debt = false;
  fixture.valuation.reverse_dcf_parameters = null;

  const html = renderToStaticMarkup(React.createElement(DeckDecisionFirstView, { dossier: fixture }));

  // Honest "Unavailable" / "Unresolved" fallback assertions
  assert.ok(
    html.includes("Decisive research presentation summary unavailable."),
    "Missing presentation summary must render 'Decisive research presentation summary unavailable.'"
  );
  assert.ok(
    html.includes("Next review trigger unresolved or unavailable."),
    "Missing next review trigger must render 'Next review trigger unresolved or unavailable.'"
  );
  assert.ok(
    html.includes("Key uncertainty unresolved or unavailable."),
    "Missing main uncertainty must render 'Key uncertainty unresolved or unavailable.'"
  );
  assert.ok(
    html.includes("Strengthening watch condition unresolved or unavailable."),
    "Missing price watch condition must render 'Strengthening watch condition unresolved or unavailable.'"
  );
  assert.ok(
    html.includes("Unavailable"),
    "Missing numerical fields must render 'Unavailable'"
  );

  // Total absence of manufactured historical sentences or numbers
  assert.strictEqual(html.includes("73.22"), false, "Historical 73.22 must not appear as a fallback");
  assert.strictEqual(html.includes("127.49"), false, "Historical 127.49 must not appear as a fallback");
  assert.strictEqual(html.includes("209.87"), false, "Historical 209.87 must not appear as a fallback");
  assert.strictEqual(html.includes("-2.34%"), false, "Historical -2.34% must not appear as a fallback");
  assert.strictEqual(html.includes("-14.6%"), false, "Historical -14.6% must not appear as a fallback");
  assert.strictEqual(html.includes("+48.7%"), false, "Historical +48.7% must not appear as a fallback");
  assert.strictEqual(html.includes("+144.8%"), false, "Historical +144.8% must not appear as a fallback");
  assert.strictEqual(html.includes("1.60B"), false, "Historical 1.60B must not appear as a fallback");
  assert.strictEqual(html.includes("1602.6"), false, "Historical 1602.6 must not appear as a fallback");
  assert.strictEqual(html.includes("+17.3%"), false, "Historical +17.3% must not appear as a fallback");
  assert.strictEqual(html.includes("+2.9%"), false, "Historical +2.9% must not appear as a fallback");
  assert.strictEqual(html.includes("Q2 FY27 earnings release"), false, "Historical Q2 FY27 release must not appear as a fallback");
}

// =========================================================================
// Mandatory Control 3: Complete Valuation Suppression Control
// =========================================================================
{
  const fixture = cloneDossier(deckDossier);
  fixture.valuation.valuation_status = "VALUATION_SUPPRESSED";
  fixture.valuation.suppression_reason = "Model suppressed due to restructuring";
  fixture.central_assessment.presentation_summary = "Valuation model suppressed pending restructuring audit.";

  // 1. Default Surface (collapsed progressive disclosures)
  const defaultHtml = renderToStaticMarkup(
    React.createElement(DeckDecisionFirstView, { dossier: fixture, initialValuationOpen: false })
  );

  assert.ok(defaultHtml.includes("Valuation Suppressed"), "Must render 'Valuation Suppressed' in default surface");
  assert.ok(defaultHtml.includes("Model suppressed due to restructuring"), "Must render suppression reason in default surface");
  assert.strictEqual(defaultHtml.includes("$73.22"), false, "Must not render numeric Bear value on default surface");
  assert.strictEqual(defaultHtml.includes("$127.49"), false, "Must not render numeric Base value on default surface");
  assert.strictEqual(defaultHtml.includes("$209.87"), false, "Must not render numeric Bull value on default surface");
  assert.strictEqual(defaultHtml.includes("-2.34%"), false, "Must not render solved CAGR when suppressed");
  assert.strictEqual(defaultHtml.includes("-14.6%"), false, "Must not render Bear difference percentage when suppressed");
  assert.strictEqual(defaultHtml.includes("+48.7%"), false, "Must not render Base difference percentage when suppressed");
  assert.strictEqual(defaultHtml.includes("+144.8%"), false, "Must not render Bull difference percentage when suppressed");

  // 2. Expanded Valuation Disclosure (initialValuationOpen = true)
  const expandedHtml = renderToStaticMarkup(
    React.createElement(DeckDecisionFirstView, { dossier: fixture, initialValuationOpen: true })
  );

  assert.ok(expandedHtml.includes("Valuation Model Outputs Suppressed"), "Must render suppression title in expanded disclosure");
  assert.ok(expandedHtml.includes("Model suppressed due to restructuring"), "Must render suppression reason in expanded disclosure");
  assert.ok(expandedHtml.includes("Solved 5Y Revenue CAGR: Suppressed (Model unavailable)"), "Must state CAGR is suppressed in expanded disclosure");
  assert.ok(expandedHtml.includes("Intrinsic Modeled Equity Values: Suppressed"), "Must state intrinsic values are suppressed in expanded disclosure");

  // The comparative scenario table MUST be completely suppressed in the expanded disclosure
  assert.strictEqual(expandedHtml.includes("<table"), false, "Comparative scenario table must be completely suppressed when valuation is suppressed");
  assert.strictEqual(expandedHtml.includes("$73.22"), false, "Must not render numeric Bear value in expanded disclosure");
  assert.strictEqual(expandedHtml.includes("$127.49"), false, "Must not render numeric Base value in expanded disclosure");
  assert.strictEqual(expandedHtml.includes("$209.87"), false, "Must not render numeric Bull value in expanded disclosure");
  assert.strictEqual(expandedHtml.includes("-2.34%"), false, "Must not render solved CAGR in expanded disclosure");
  assert.strictEqual(expandedHtml.includes("Roundtrip:"), false, "Must not render roundtrip value in expanded disclosure");

  // Non-modeled baseline facts (reference price) remain visible in expanded disclosure
  assert.ok(expandedHtml.includes("$85.72 USD"), "Reference price tape remains visible in expanded disclosure");
}

console.log("[OK] Controlled fixture and fallback elimination tests passed.");

// =========================================================================
// 3. Component Isolation & Non-WATCH Disposition Handling
// =========================================================================
console.log("3. Verifying component isolation and non-WATCH disposition contracts...");

// A. Non-WATCH disposition in OpportunityDossierCard: Falls back to standard card
{
  const fixture = cloneDossier(deckDossier);
  fixture.central_assessment.disposition = "NO_SUPPORTED_OPPORTUNITY";
  fixture.central_assessment.disposition_label = "No Opportunity Supported";

  const html = renderToStaticMarkup(React.createElement(OpportunityDossierCard, { dossier: fixture }));
  assert.strictEqual(
    html.includes("Decision-First Prototype (DECK)"),
    false,
    "Non-WATCH disposition must not render DeckDecisionFirstView prototype badge"
  );
  assert.ok(
    html.includes("No Opportunity Supported") || html.includes("NO_SUPPORTED_OPPORTUNITY"),
    "Standard card must render the real disposition label"
  );
}

// B. Non-WATCH disposition in DeckDecisionFirstView directly: Dynamically renders badge, no hardcoded WATCH
{
  const fixture = cloneDossier(deckDossier);
  fixture.central_assessment.disposition = "RESEARCH_FURTHER";
  fixture.central_assessment.disposition_label = "Research Further Required";

  const html = renderToStaticMarkup(React.createElement(DeckDecisionFirstView, { dossier: fixture }));
  assert.ok(html.includes("Research Further Required"), "Must dynamically display custom disposition label");
}

// C. Non-DECK published assets retain standard card
const otherSymbols = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG"];
for (const sym of otherSymbols) {
  const d = getOpportunityDossier(sym);
  assert.ok(d, `${sym} must load from published collection`);
  const html = renderToStaticMarkup(React.createElement(OpportunityDossierCard, { dossier: d }));
  assert.strictEqual(
    html.includes("Decision-First Prototype (DECK)"),
    false,
    `${sym} must never render DECK decision-first prototype badge`
  );
}

// D. Unpublished symbol returns null
const nullDossier = getOpportunityDossier("UNKNOWN_SYMBOL_XYZ");
assert.strictEqual(nullDossier, null, "Unpublished symbol must return null");

console.log("[OK] Component isolation and disposition contracts verified.");

// =========================================================================
// 4. Accessibility & Interactive Semantics
// =========================================================================
console.log("4. Verifying accessibility, touch targets, and ARIA state...");

assert.ok(controlMarkup.includes("min-h-[44px]"), "Interactive controls must enforce >= 44px touch targets");
assert.ok(controlMarkup.includes("focus:ring-2 focus:ring-cyan-400"), "Controls must have visible focus rings");
assert.ok(controlMarkup.includes("aria-expanded="), "Progressive disclosure controls must have aria-expanded");
assert.ok(controlMarkup.includes("aria-controls="), "Progressive disclosure controls must have aria-controls");

// Cohort size check
const cohort = listOpportunityCohort();
assert.strictEqual(cohort.length, 6, "Must have exactly 6 published Quality Watch assets");

console.log("===============================================================================");
console.log("ALL BEHAVIORAL AND FIXTURE TESTS PASSED SUCCESSFULLY! ✅");
console.log("===============================================================================");
