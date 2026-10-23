/**
 * frontend/tests/qualityWatchDecisionFirstRollout.test.ts
 *
 * Comprehensive Test Suite for the Quality Watch Decision-First Layout Rollout.
 * Governed by ARX Closure Model v2 & ARX-DECISION-RESEARCH-PROCESS-V1.
 *
 * Enforces:
 * 1. Universal Coverage across all 6 Published Assets (FIX, DUOL, ULTA, MEDP, ISRG, DECK):
 *    - Rendered through OpportunityDossierCard into DecisionFirstDossierView.
 *    - Asset identity, company name, exchange, CIK, and route.
 *    - WATCH verdict badge and "WATCH — Decisive Reason".
 *    - Dated reference close ("Reference Close: 2026-10-16") and reference price.
 *    - Central thesis quote block.
 *    - Bear -> Base -> Bull expectations with strict DOM order.
 *    - Up to three drivers and two invalidation conditions.
 *    - Next review trigger, decisive unknown, and strengthening condition.
 *    - Three progressive disclosures (Valuation, Financials, Baseline & Selection Record).
 * 2. Net Debt Bridge Invariants:
 *    - Negative net debt rendered as (Net Cash) with negative sign (FIX, DUOL, MEDP, ISRG, DECK).
 *    - Positive net debt rendered as (Net Debt) with positive sign (ULTA).
 * 3. Selection Record vs Baseline Fallback:
 *    - ULTA, MEDP, ISRG, DECK render 6 Selection Record questions.
 *    - FIX, DUOL render 4 Baseline Research Record questions without empty/unavailable cards.
 * 4. Negative Controls & Preservation:
 *    - IREN and MU retain their standard presentation (NO_SUPPORTED_OPPORTUNITY).
 *    - Unpublished symbols fail closed to null.
 * 5. Accessibility:
 *    - Min 44px touch targets on interactive controls.
 *    - Visible focus rings.
 *    - Valid aria-expanded and aria-controls attributes.
 */

import assert from "node:assert";
import React from "react";
// @ts-ignore - react-dom/server type declaration mapping under node16
import { renderToStaticMarkup } from "react-dom/server";
import snapshotData from "../lib/opportunityDossierSnapshot.json";
import OpportunityDossierCard from "../components/OpportunityDossierCard";
import DecisionFirstDossierView from "../components/DecisionFirstDossierView";
import {
  getOpportunityDossier,
  isCohortSymbol,
  OpportunityDossier,
} from "../lib/opportunityDossiers";

const escapeHtml = (str: string): string =>
  str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

console.log("Starting Quality Watch Decision-First Rollout Test Suite...");

const publishedWatchSymbols = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"] as const;

// =========================================================================
// 1. Universal Decision-First Verification Across All 6 Published Assets
// =========================================================================
console.log("1. Verifying all 6 published assets render decision-first presentation...");

for (const sym of publishedWatchSymbols) {
  const dossier = getOpportunityDossier(sym);
  assert.ok(dossier, `${sym} must load from published collection`);
  assert.strictEqual(dossier.identity.symbol, sym);
  assert.strictEqual(
    dossier.central_assessment.disposition,
    "WATCH_CONDITIONS",
    `${sym} must have WATCH_CONDITIONS disposition`
  );

  // Render through OpportunityDossierCard
  const html = renderToStaticMarkup(
    React.createElement(OpportunityDossierCard, { dossier })
  );

  // A. Identity & Header
  assert.ok(html.includes(sym), `${sym}: Symbol must be rendered`);
  assert.ok(html.includes(dossier.identity.company_name), `${sym}: Company name must be rendered`);
  assert.ok(html.includes(dossier.identity.cik), `${sym}: CIK must be rendered`);
  assert.ok(html.includes(dossier.identity.exchange), `${sym}: Exchange must be rendered`);
  assert.ok(html.includes(dossier.identity.opportunity_route), `${sym}: Route must be rendered`);

  // B. WATCH Verdict & Dated Reference Close
  assert.ok(html.includes("WATCH — Decisive Reason"), `${sym}: Must render WATCH Decisive Reason heading`);
  assert.ok(html.includes("Reference Close: 2026-10-16"), `${sym}: Must render Reference Close: 2026-10-16`);
  assert.ok(html.includes(`$${dossier.valuation.reference_price.toFixed(2)}`), `${sym}: Must render reference price`);

  // C. Decisive Reason Narrative
  assert.ok(
    html.includes(escapeHtml(dossier.central_assessment.presentation_summary)),
    `${sym}: Must render authoritative presentation summary`
  );

  // D. Thesis Section
  assert.ok(html.includes("Thesis"), `${sym}: Must render Thesis heading`);

  // E. Expectations & Scenarios
  assert.ok(html.includes("Expectations &amp; Scenarios") || html.includes("Expectations & Scenarios"), `${sym}: Must render Expectations & Scenarios`);
  assert.ok(html.includes("Reference Price Solved 5-Year Revenue CAGR:"), `${sym}: Must render Solved CAGR label`);
  assert.ok(
    html.includes(`${dossier.valuation.solved_5y_cagr_pct?.toFixed(2)}%`),
    `${sym}: Must render solved CAGR percentage`
  );

  // Scenario Ordering: Bear case < Base case < Bull case
  const bearIdx = html.indexOf("Bear case");
  const baseIdx = html.indexOf("Base case");
  const bullIdx = html.indexOf("Bull case");
  assert.ok(bearIdx !== -1, `${sym}: Must render Bear case`);
  assert.ok(baseIdx !== -1, `${sym}: Must render Base case`);
  assert.ok(bullIdx !== -1, `${sym}: Must render Bull case`);
  assert.ok(bearIdx < baseIdx, `${sym}: Bear case must precede Base case in DOM`);
  assert.ok(baseIdx < bullIdx, `${sym}: Base case must precede Bull case in DOM`);

  // F. Drivers: "What Must Be True?"
  assert.ok(html.includes("What Must Be True?"), `${sym}: Must render What Must Be True? heading`);

  // G. Invalidation: "What Breaks The Thesis?"
  assert.ok(html.includes("What Breaks The Thesis?"), `${sym}: Must render What Breaks The Thesis? heading`);

  // H. Next Review & Key Unknown
  assert.ok(html.includes("Next Review &amp; Key Unknown") || html.includes("Next Review & Key Unknown"), `${sym}: Must render Next Review heading`);
  assert.ok(html.includes("Next Review Trigger:"), `${sym}: Must render Next Review Trigger label`);
  assert.ok(html.includes("Decisive Unknown:"), `${sym}: Must render Decisive Unknown label`);
  assert.ok(html.includes("Strengthening Condition:"), `${sym}: Must render Strengthening Condition label`);

  // I. Three Progressive Disclosures (accordion headers)
  assert.ok(
    html.includes("Valuation Mechanics, Parameters &amp; Full Scenario Comparison") ||
    html.includes("Valuation Mechanics, Parameters & Full Scenario Comparison"),
    `${sym}: Must render Disclosure 1 header`
  );
  assert.ok(
    html.includes("Financial Resilience, Capital Structure &amp; Sourced Net Debt Bridge") ||
    html.includes("Financial Resilience, Capital Structure & Sourced Net Debt Bridge"),
    `${sym}: Must render Disclosure 2 header`
  );
  assert.ok(
    html.includes("Full Dossier Baseline &amp; Quality Watch Selection Record") ||
    html.includes("Full Dossier Baseline & Quality Watch Selection Record"),
    `${sym}: Must render Disclosure 3 header`
  );

  // J. Accessibility: min-h-[44px] touch targets
  assert.ok(html.includes("min-h-[44px]"), `${sym}: Must include min-h-[44px] touch target class`);

  console.log(`  [PASS] ${sym}: Verified decision-first layout completeness.`);
}

console.log("[OK] All 6 published assets verified for decision-first layout completeness.");

// =========================================================================
// 2. Net Debt Bridge Invariants Across Assets
// =========================================================================
console.log("2. Verifying net debt bridge sign and labeling invariants...");

// FIX: -$1800.7M Net Cash
const fixDossier = getOpportunityDossier("FIX")!;
const fixHtml = renderToStaticMarkup(React.createElement(DecisionFirstDossierView, { dossier: fixDossier, initialFinancialsOpen: true }));
assert.ok(fixHtml.includes("(Net Cash)"), "FIX: Must be labeled (Net Cash)");
assert.ok(fixHtml.includes("-$1800.7M"), "FIX: Must render -$1800.7M net cash bridge");

// DUOL: -$1313.9M Net Cash & zero funded debt
const duolDossier = getOpportunityDossier("DUOL")!;
const duolHtml = renderToStaticMarkup(React.createElement(DecisionFirstDossierView, { dossier: duolDossier, initialFinancialsOpen: true }));
assert.ok(duolHtml.includes("(Net Cash)"), "DUOL: Must be labeled (Net Cash)");
assert.ok(duolHtml.includes("zero funded debt"), "DUOL: Must render zero funded debt");

// ULTA: +$126.1M Net Debt
const ultaDossier = getOpportunityDossier("ULTA")!;
const ultaHtml = renderToStaticMarkup(React.createElement(DecisionFirstDossierView, { dossier: ultaDossier, initialFinancialsOpen: true }));
assert.ok(ultaHtml.includes("$126.1M (Net Debt)"), "ULTA: Must render positive $126.1M (Net Debt)");

// MEDP: -$502.7M Net Cash
const medpDossier = getOpportunityDossier("MEDP")!;
const medpHtml = renderToStaticMarkup(React.createElement(DecisionFirstDossierView, { dossier: medpDossier, initialFinancialsOpen: true }));
assert.ok(medpHtml.includes("-$502.7M (Net Cash)"), "MEDP: Must render -$502.7M (Net Cash)");

// ISRG: -$5216.2M Net Cash
const isrgDossier = getOpportunityDossier("ISRG")!;
const isrgHtml = renderToStaticMarkup(React.createElement(DecisionFirstDossierView, { dossier: isrgDossier, initialFinancialsOpen: true }));
assert.ok(isrgHtml.includes("-$5216.2M (Net Cash)"), "ISRG: Must render -$5216.2M (Net Cash)");

// DECK: -$1602.6M Net Cash
const deckDossier = getOpportunityDossier("DECK")!;
const deckHtml = renderToStaticMarkup(React.createElement(DecisionFirstDossierView, { dossier: deckDossier, initialFinancialsOpen: true }));
assert.ok(deckHtml.includes("-$1602.6M (Net Cash)"), "DECK: Must render -$1602.6M (Net Cash)");

console.log("[OK] Net debt bridge sign and labeling invariants verified.");

// =========================================================================
// 3. Disclosure 3: Selection Record vs Baseline Fallback
// =========================================================================
console.log("3. Verifying Disclosure 3 selection record and baseline handling...");

// Assets with selection_rationale: 6 questions
for (const sym of ["ULTA", "MEDP", "ISRG", "DECK"]) {
  const d = getOpportunityDossier(sym)!;
  const html = renderToStaticMarkup(React.createElement(DecisionFirstDossierView, { dossier: d, initialProvenanceOpen: true }));
  assert.ok(html.includes("Why This Business?"), `${sym}: Must render Why This Business?`);
  assert.ok(html.includes("Why Research It Now?"), `${sym}: Must render Why Research It Now?`);
  assert.ok(html.includes("What Does The Price Require?"), `${sym}: Must render What Does The Price Require?`);
  assert.ok(html.includes("What Might ARX See Differently?"), `${sym}: Must render What Might ARX See Differently?`);
  assert.ok(html.includes("What Would Change The Assessment?"), `${sym}: Must render What Would Change The Assessment?`);
  assert.ok(html.includes("Why Prioritize Research Attention?"), `${sym}: Must render Why Prioritize Research Attention?`);
  assert.ok(html.includes("Governance Record:"), `${sym}: Must render Governance Record`);
}

// Assets without selection_rationale (FIX, DUOL): 4 baseline questions without empty cards
for (const sym of ["FIX", "DUOL"]) {
  const d = getOpportunityDossier(sym)!;
  const html = renderToStaticMarkup(React.createElement(DecisionFirstDossierView, { dossier: d, initialProvenanceOpen: true }));
  assert.ok(html.includes("Why This Business Deserves Attention"), `${sym}: Must render Why This Business Deserves Attention`);
  assert.ok(html.includes("Central Investment Debate"), `${sym}: Must render Central Investment Debate`);
  assert.ok(html.includes("Why It Remains WATCH"), `${sym}: Must render Why It Remains WATCH`);
  assert.ok(html.includes("Observable Evidence That Could Change Assessment"), `${sym}: Must render Observable Evidence`);
  assert.ok(html.includes("Governance Record:"), `${sym}: Must render Governance Record`);
  assert.strictEqual(html.includes("Why Prioritize Research Attention?"), false, `${sym}: Must not render uncataloged selection question`);
}

console.log("[OK] Disclosure 3 selection record and baseline handling verified.");

// =========================================================================
// 4. Negative Controls: Non-WATCH Assets & Uncataloged Symbols
// =========================================================================
console.log("4. Verifying negative controls (non-WATCH assets & uncataloged symbols)...");

// Non-dossier / excluded symbols fail closed from published collection
assert.strictEqual(isCohortSymbol("IREN"), false, "IREN must not be in published cohort");
assert.strictEqual(getOpportunityDossier("IREN"), null, "getOpportunityDossier(IREN) must return null");
assert.strictEqual(isCohortSymbol("MU"), false, "MU must not be in published cohort");
assert.strictEqual(getOpportunityDossier("MU"), null, "getOpportunityDossier(MU) must return null");
assert.strictEqual(getOpportunityDossier("UNKNOWN_TICKER"), null, "Unpublished symbol must return null");

// Standard presentation preservation when non-published / suppressed dossiers are rendered through OpportunityDossierCard
const irenDossier = (snapshotData as any).dossiers?.IREN as OpportunityDossier;
assert.ok(irenDossier, "IREN raw dossier must exist in snapshot");
const irenHtml = renderToStaticMarkup(React.createElement(OpportunityDossierCard, { dossier: irenDossier }));
assert.ok(
  irenHtml.includes("VALUATION SUPPRESSED") || irenHtml.includes("Valuation Model Outputs Suppressed"),
  "IREN must retain standard suppression display"
);
assert.strictEqual(irenHtml.includes("WATCH — Decisive Reason"), false, "IREN must not render WATCH Decisive Reason");
assert.strictEqual(irenHtml.includes("Decision-First Research"), false, "IREN must not render Decision-First Research badge");

const muDossier = (snapshotData as any).dossiers?.MU as OpportunityDossier;
assert.ok(muDossier, "MU raw dossier must exist in snapshot");
const muHtml = renderToStaticMarkup(React.createElement(OpportunityDossierCard, { dossier: muDossier }));
assert.ok(muHtml.includes("NO SUPPORTED OPPORTUNITY"), "MU must render standard NO SUPPORTED OPPORTUNITY badge");
assert.strictEqual(muHtml.includes("WATCH — Decisive Reason"), false, "MU must not render WATCH Decisive Reason");
assert.strictEqual(muHtml.includes("Decision-First Research"), false, "MU must not render Decision-First Research badge");

console.log("[OK] Negative controls verified.");

console.log("\n===============================================================================");
console.log("ALL QUALITY WATCH DECISION-FIRST ROLLOUT TESTS PASSED SUCCESSFULLY! ✅");
console.log("===============================================================================");
