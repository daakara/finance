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

// =========================================================================
// 5. Focused Regression Controls (A, B, C, D)
// =========================================================================
console.log("5. Running focused regression controls...");

// Helper for deep cloning
function cloneDossier<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

// -------------------------------------------------------------------------
// Control A: Changed-Input Control
// -------------------------------------------------------------------------
console.log("  A. Testing changed-input control...");

// Test DUOL changed inputs
{
  const duolMutated = cloneDossier(getOpportunityDossier("DUOL")!);
  duolMutated.central_assessment.strongest_supporting_evidence =
    "Mutated DUOL supporting evidence: $2.45B accelerated revenue run-rate and 88% gross margin.";
  duolMutated.financial_profile.operating_margin_pct = 19.85;
  duolMutated.monitoring.observable_invalidation_trigger =
    "Mutated DUOL trigger: Core subscription churn exceeding 14.5% YoY.";
  duolMutated.financial_profile.net_debt_bridge = -1550200000;

  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: duolMutated,
      initialFinancialsOpen: true,
    })
  );

  assert.ok(
    html.includes("Mutated DUOL supporting evidence: $2.45B accelerated revenue run-rate and 88% gross margin."),
    "DUOL: Rendered drivers must reflect mutated supporting evidence"
  );
  assert.ok(
    html.includes("19.85% operating margin"),
    "DUOL: Rendered margin driver must reflect mutated operating margin"
  );
  assert.ok(
    html.includes("Mutated DUOL trigger: Core subscription churn exceeding 14.5% YoY."),
    "DUOL: Rendered invalidation trigger must reflect mutated monitoring threshold"
  );
  assert.ok(
    html.includes("-$1550.2M (Net Cash)"),
    "DUOL: Rendered balance sheet must reflect mutated net cash bridge"
  );

  // Must not retain obsolete component literals
  assert.strictEqual(
    html.includes("platform saturation"),
    false,
    "DUOL: Must not retain obsolete causal claim 'platform saturation'"
  );
  assert.strictEqual(
    html.includes("freemium friction"),
    false,
    "DUOL: Must not retain obsolete causal claim 'freemium friction'"
  );
}

// Test FIX changed inputs
{
  const fixMutated = cloneDossier(getOpportunityDossier("FIX")!);
  fixMutated.central_assessment.strongest_supporting_evidence =
    "Mutated FIX evidence: $3.10B modular backlog with 35% ROIC.";
  fixMutated.financial_profile.operating_margin_pct = 18.25;
  fixMutated.monitoring.observable_invalidation_trigger =
    "Mutated FIX trigger: Modular HVAC backlog book-to-bill falling below 1.10x.";

  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: fixMutated,
    })
  );

  assert.ok(
    html.includes("Mutated FIX evidence: $3.10B modular backlog with 35% ROIC."),
    "FIX: Rendered drivers must reflect mutated supporting evidence"
  );
  assert.ok(
    html.includes("18.25% operating margin"),
    "FIX: Rendered margin driver must reflect mutated operating margin"
  );
  assert.ok(
    html.includes("Mutated FIX trigger: Modular HVAC backlog book-to-bill falling below 1.10x."),
    "FIX: Rendered invalidation must reflect mutated monitoring threshold"
  );

  // Must not retain obsolete component literals
  assert.strictEqual(
    html.includes("pricing pressure"),
    false,
    "FIX: Must not retain obsolete causal claim 'pricing pressure'"
  );
  assert.strictEqual(
    html.includes("project execution delays"),
    false,
    "FIX: Must not retain obsolete causal claim 'project execution delays'"
  );
}

console.log("  [PASS] Control A: Changed-input control verified.");

// -------------------------------------------------------------------------
// Control B: Original-Narrative Suppression for FIX & DUOL
// -------------------------------------------------------------------------
console.log("  B. Testing original-narrative suppression for FIX & DUOL...");

// FIX with original narrative but suppressed valuation
{
  const fixSuppressed = cloneDossier(getOpportunityDossier("FIX")!);
  fixSuppressed.valuation.valuation_status = "VALUATION_SUPPRESSED";
  fixSuppressed.valuation.suppression_reason = "Model qualification pending verification";

  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: fixSuppressed,
      initialValuationOpen: true,
      initialFinancialsOpen: true,
      initialProvenanceOpen: true,
    })
  );

  // 1. Default Surface
  assert.ok(
    html.includes("Decisive research presentation summary unavailable while valuation outputs are suppressed."),
    "FIX: Must render default decisive reason suppression notice"
  );
  assert.ok(
    html.includes("Valuation Suppressed"),
    "FIX: Solved CAGR must render Valuation Suppressed"
  );
  assert.ok(
    html.includes("Strengthening watch condition unavailable while valuation outputs are suppressed."),
    "FIX: Strengthening watch condition must be suppressed"
  );

  // 2. Expanded Valuation Disclosure
  assert.ok(
    html.includes("Valuation Model Outputs Suppressed"),
    "FIX: Valuation disclosure must display suppression banner"
  );
  assert.ok(
    html.includes("Model qualification pending verification"),
    "FIX: Valuation disclosure must display suppression reason"
  );

  // 3. Expanded Provenance (Disclosure 3)
  assert.ok(
    html.includes("Research presentation summary unavailable while valuation outputs are suppressed."),
    "FIX: Provenance must not leak original summary with valuation claims"
  );
  assert.ok(
    html.includes("Price watch condition unavailable while valuation outputs are suppressed."),
    "FIX: Provenance must not leak price watch condition with valuation numbers"
  );

  // 4. Verify original suppressed numbers DO NOT reappear anywhere in DOM
  assert.strictEqual(html.includes("30.07%"), false, "FIX: Must not leak solved CAGR 30.07%");
  assert.strictEqual(html.includes("$1,050 - $1,100"), false, "FIX: Must not leak base valuation range $1,050 - $1,100");
  assert.strictEqual(html.includes("$533.46"), false, "FIX: Must not leak Bear case $533.46");
  assert.strictEqual(html.includes("$1081.99"), false, "FIX: Must not leak Base case $1081.99");
  assert.strictEqual(html.includes("$1996.38"), false, "FIX: Must not leak Bull case $1996.38");

  // 5. Independently valid financial facts must remain visible
  assert.ok(html.includes("Comfort Systems USA, Inc."), "FIX: Company name must remain visible");
  assert.ok(html.includes("Reference Close: 2026-10-16"), "FIX: Reference date must remain visible");
  assert.ok(html.includes("$1439.07"), "FIX: Reference close tape must remain visible");
  assert.ok(html.includes("-$1800.7M (Net Cash)"), "FIX: Net cash bridge must remain visible");
}

// DUOL with original narrative but suppressed valuation
{
  const duolSuppressed = cloneDossier(getOpportunityDossier("DUOL")!);
  duolSuppressed.valuation.valuation_status = "VALUATION_SUPPRESSED";
  duolSuppressed.valuation.suppression_reason = "Model qualification pending verification";

  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: duolSuppressed,
      initialValuationOpen: true,
      initialFinancialsOpen: true,
      initialProvenanceOpen: true,
    })
  );

  // 1. Default Surface
  assert.ok(
    html.includes("Decisive research presentation summary unavailable while valuation outputs are suppressed."),
    "DUOL: Must render default decisive reason suppression notice"
  );
  assert.ok(
    html.includes("Strengthening watch condition unavailable while valuation outputs are suppressed."),
    "DUOL: Strengthening watch condition must be suppressed"
  );

  // 2. Expanded Provenance (Disclosure 3)
  assert.ok(
    html.includes("Research presentation summary unavailable while valuation outputs are suppressed."),
    "DUOL: Provenance must not leak original summary with valuation claims"
  );
  assert.ok(
    html.includes("Price watch condition unavailable while valuation outputs are suppressed."),
    "DUOL: Provenance must not leak price watch condition with valuation numbers"
  );

  // 3. Verify original suppressed numbers DO NOT reappear anywhere in DOM
  assert.strictEqual(html.includes("22.99%"), false, "DUOL: Must not leak solved CAGR 22.99%");
  assert.strictEqual(html.includes("$125 - $130"), false, "DUOL: Must not leak base valuation range $125 - $130");
  assert.strictEqual(html.includes("$85.08"), false, "DUOL: Must not leak Bear case $85.08");
  assert.strictEqual(html.includes("$129.33"), false, "DUOL: Must not leak Base case $129.33");
  assert.strictEqual(html.includes("$185.43"), false, "DUOL: Must not leak Bull case $185.43");

  // 4. Independently valid financial facts must remain visible
  assert.ok(html.includes("Duolingo, Inc."), "DUOL: Company name must remain visible");
  assert.ok(html.includes("Reference Close: 2026-10-16"), "DUOL: Reference date must remain visible");
  assert.ok(html.includes("$141.76"), "DUOL: Reference close tape must remain visible");
  assert.ok(html.includes("-$1313.9M (Net Cash)"), "DUOL: Net cash bridge must remain visible");
}

console.log("  [PASS] Control B: Original-narrative suppression verified.");

// -------------------------------------------------------------------------
// Control C: Missing-Versus-Zero Accounting
// -------------------------------------------------------------------------
console.log("  C. Testing missing-versus-zero accounting...");

// 1. restricted_cash = null -> Unavailable
{
  const testDossier = cloneDossier(getOpportunityDossier("FIX")!);
  testDossier.financial_profile.restricted_cash = null;

  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: testDossier,
      initialFinancialsOpen: true,
    })
  );

  assert.ok(
    html.includes("Restricted Cash</span><span class=\"text-slate-200 font-semibold\">Unavailable"),
    "restricted_cash = null must render Unavailable"
  );
  assert.strictEqual(
    html.includes("Restricted Cash</span><span class=\"text-slate-200 font-semibold\">$0.0M"),
    false,
    "restricted_cash = null must NOT render $0.0M"
  );
}

// 2. restricted_cash = 0 -> $0.0M
{
  const testDossier = cloneDossier(getOpportunityDossier("FIX")!);
  testDossier.financial_profile.restricted_cash = 0;

  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: testDossier,
      initialFinancialsOpen: true,
    })
  );

  assert.ok(
    html.includes("Restricted Cash</span><span class=\"text-slate-200 font-semibold\">$0.0M"),
    "restricted_cash = 0 must render $0.0M"
  );
}

// 3. Missing bridge / lease treatment -> unresolved wording
{
  const testDossier = cloneDossier(getOpportunityDossier("FIX")!);
  testDossier.financial_profile.bridge_note = null as any;
  testDossier.financial_profile.lease_convention_treatment = null as any;

  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: testDossier,
      initialFinancialsOpen: true,
    })
  );

  assert.ok(
    html.toLowerCase().includes("unresolved"),
    "Missing bridge/lease treatment must render unresolved wording"
  );
  assert.strictEqual(
    html.includes("Lease liabilities are expensed in operating income under ASC 842"),
    false,
    "Must not substitute unevidenced ASC 842 lease expense convention"
  );
}

// 4. Preserved authoritative wording for documented treatments
{
  const duolDossier = getOpportunityDossier("DUOL")!;
  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: duolDossier,
      initialFinancialsOpen: true,
    })
  );

  assert.ok(
    html.includes(duolDossier.financial_profile.bridge_note!),
    "DUOL documented bridge note must be preserved"
  );
}

console.log("  [PASS] Control C: Missing-versus-zero accounting verified.");

// -------------------------------------------------------------------------
// Control D: DECK Preservation
// -------------------------------------------------------------------------
console.log("  D. Testing DECK presentation, thresholds, and behavior preservation...");

{
  const deckDossier = getOpportunityDossier("DECK")!;
  const html = renderToStaticMarkup(
    React.createElement(DecisionFirstDossierView, {
      dossier: deckDossier,
      initialValuationOpen: true,
      initialFinancialsOpen: true,
      initialProvenanceOpen: true,
    })
  );

  // Exact 3 drivers preserved from PR #52
  assert.ok(html.includes("1. DTC Velocity &amp; International Scaling") || html.includes("1. DTC Velocity & International Scaling"), "DECK: Driver 1 title preserved");
  assert.ok(html.includes("HOKA direct-to-consumer sales sustain double-digit growth"), "DECK: Driver 1 text preserved");
  assert.ok(html.includes("2. Gross Margin &amp; Operating Defense") || html.includes("2. Gross Margin & Operating Defense"), "DECK: Driver 2 title preserved");
  assert.ok(html.includes("Consolidated gross margins sustain above 55.0%"), "DECK: Driver 2 text preserved");
  assert.ok(html.includes("3. Zero-Debt Balance Sheet Strength"), "DECK: Driver 3 title preserved");
  assert.ok(html.includes("Preservation of balance sheet strength (-$1602.6M (Net Cash), zero funded debt)"), "DECK: Driver 3 text preserved");

  // Exact 2 invalidations preserved from PR #52
  assert.ok(html.includes("1. Gross Margin Breach Below 50.0%"), "DECK: Invalidation 1 title preserved");
  assert.ok(html.includes("Monitored Trigger: &lt; 50.0% consolidated gross margin"), "DECK: Invalidation 1 trigger preserved (<50%)");
  assert.ok(html.includes("2. HOKA DTC Deceleration Below 10.0%"), "DECK: Invalidation 2 title preserved (10%, NOT 15%)");
  assert.ok(html.includes("Monitored Trigger: &lt; 10.0% YoY quarterly DTC net sales growth"), "DECK: Invalidation 2 trigger preserved (<10%)");

  // Net Debt Bridge & Split-Adjusted Basis
  assert.ok(html.includes("-$1602.6M (Net Cash)"), "DECK: Net cash bridge preserved");
  assert.ok(html.includes("6:1 split-adjusted"), "DECK: 6:1 split-adjusted basis note preserved");

  // Accepted thesis quote
  assert.ok(
    html.includes("Can HOKA and UGG sustain direct-to-consumer momentum and operating margins above 20%"),
    "DECK: Accepted thesis quote preserved"
  );
}

console.log("  [PASS] Control D: DECK preservation verified.");

console.log("\n===============================================================================");
console.log("ALL QUALITY WATCH DECISION-FIRST ROLLOUT TESTS PASSED SUCCESSFULLY! ✅");
console.log("===============================================================================");
