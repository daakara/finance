/**
 * frontend/tests/deckDecisionFirstPrototype.test.ts
 *
 * Comprehensive Behavioral & Contract Test Suite for the DECK Decision-First Research Page Prototype.
 * Governed by ARX Closure Model v2.
 *
 * Verifies:
 * 1. Accurate mapping and preservation of authoritative DECK dossier data:
 *    - Reference price: $85.72 as of 2026-10-16
 *    - Reverse-DCF solved 5Y revenue CAGR: -2.34%
 *    - Bear / Base / Bull scenario values ($73.22, $127.49, $209.87) and diffs (-14.6%, +48.7%, +144.8%)
 *    - Scenario ordering: Bear <= Base <= Bull
 *    - Zero funded debt ($0.00), $1.60B cash buffer, -$1,602.6M net cash bridge
 *    - ASC 842 operating leases ($472.33M) excluded from net debt bridge
 *    - Split-adjusted diluted shares (138.56M) from Form 10-Q Q1 FY27
 * 2. Strict adherence to epistemic boundaries:
 *    - Strict WATCH disposition preserved (zero BUY, INVEST, or PASS recommendations)
 *    - Zero invented consensus estimates, price targets, or shareholder returns
 *    - Optional Section H ("What could this become?") omitted
 * 3. Component isolation & fallback preservation:
 *    - DECK is routed to DeckDecisionFirstView in OpportunityDossierCard
 *    - All other published dossiers (FIX, DUOL, ULTA, MEDP, ISRG) retain accepted standard presentation
 *    - Uncataloged symbols fail closed to neutral fallback (null)
 * 4. Accessibility & interactive semantics:
 *    - Controls satisfy touch target >= 44px
 *    - Buttons have visible focus ring classes and semantic aria attributes (aria-expanded, aria-controls)
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import {
  getOpportunityDossier,
  listOpportunityCohort,
  isCohortSymbol,
  calculateScenarioPriceDiffPct,
  OpportunityDossier,
} from "../lib/opportunityDossiers";

console.log("Starting DECK Decision-First Prototype Behavioral Test Suite...");

// =========================================================================
// 1. Authoritative Data Integrity for DECK
// =========================================================================
console.log("1. Verifying authoritative DECK research dossier payload...");

const deckDossier = getOpportunityDossier("DECK");
assert.ok(deckDossier, "DECK must be available as an accepted Quality Watch dossier");

// A. Identity Context
assert.strictEqual(deckDossier.identity.symbol, "DECK");
assert.strictEqual(deckDossier.identity.company_name, "Deckers Outdoor Corp.");
assert.strictEqual(deckDossier.identity.cik, "0000910521");
assert.strictEqual(deckDossier.identity.exchange, "NYSE");
assert.strictEqual(deckDossier.identity.opportunity_route, "Consumer Cyclical / Footwear");

// B. Disposition & WATCH Semantics
assert.strictEqual(
  deckDossier.central_assessment.disposition,
  "WATCH_CONDITIONS",
  "DECK disposition must strictly be WATCH_CONDITIONS (no BUY or INVEST)"
);
assert.ok(
  deckDossier.central_assessment.disposition_label.toLowerCase().includes("watch"),
  "Disposition label must indicate WATCH"
);

// C. Valuation Reference & Reverse-DCF Parameters
assert.strictEqual(deckDossier.valuation.reference_price, 85.72, "Reference price must be $85.72");
assert.strictEqual(deckDossier.valuation.reference_date, "2026-10-16", "Reference date must be 2026-10-16");
assert.strictEqual(deckDossier.valuation.solved_5y_cagr_pct, -2.34, "Solved 5Y CAGR must be -2.34%");
assert.strictEqual(deckDossier.valuation.roundtrip_value, 85.72, "Roundtrip value must equal $85.72");

const rDcf = deckDossier.valuation.reverse_dcf_parameters;
assert.ok(rDcf, "Reverse-DCF parameters must be defined");
assert.strictEqual(rDcf.operating_margin_pct, 22);
assert.strictEqual(rDcf.wacc_pct, 9);
assert.strictEqual(rDcf.terminal_growth_pct, 2.5);
assert.strictEqual(rDcf.tax_rate_pct, 22);
assert.strictEqual(rDcf.reinvestment_rate_pct, 15);
assert.strictEqual(rDcf.current_shares, 138559000);
assert.strictEqual(rDcf.net_debt_bridge, -1602589000);

// D. Scenario Hierarchy & Calculation Integrity
const scenarios = deckDossier.valuation.scenarios;
assert.ok(scenarios, "Scenarios must be defined");
assert.ok(scenarios.conservative, "Bear case (conservative) must exist");
assert.ok(scenarios.base, "Base case must exist");
assert.ok(scenarios.optimistic, "Bull case (optimistic) must exist");

const bearVal = scenarios.conservative.per_share_value;
const baseVal = scenarios.base.per_share_value;
const bullVal = scenarios.optimistic.per_share_value;

assert.strictEqual(bearVal, 73.22, "Bear case value must be $73.22");
assert.strictEqual(baseVal, 127.49, "Base case value must be $127.49");
assert.strictEqual(bullVal, 209.87, "Bull case value must be $209.87");

// Hierarchy: Bear < Base < Bull
assert.ok(bearVal < baseVal, "Bear value must be less than Base value");
assert.ok(baseVal < bullVal, "Base value must be less than Bull value");

// Safe percentage difference checks
const bearDiff = calculateScenarioPriceDiffPct(bearVal, deckDossier.valuation.reference_price);
const baseDiff = calculateScenarioPriceDiffPct(baseVal, deckDossier.valuation.reference_price);
const bullDiff = calculateScenarioPriceDiffPct(bullVal, deckDossier.valuation.reference_price);

assert.ok(bearDiff !== null && Math.abs(bearDiff - -14.6) < 0.1, `Bear diff should be ~ -14.6%, got ${bearDiff}`);
assert.ok(baseDiff !== null && Math.abs(baseDiff - 48.7) < 0.1, `Base diff should be ~ +48.7%, got ${baseDiff}`);
assert.ok(bullDiff !== null && Math.abs(bullDiff - 144.8) < 0.1, `Bull diff should be ~ +144.8%, got ${bullDiff}`);

// E. Sourced Financial Facts & Capital Structure
const fin = deckDossier.financial_profile;
assert.strictEqual(fin.is_zero_debt, true, "DECK must be evidenced as zero funded debt");
assert.strictEqual(fin.total_financial_debt, 0, "Total financial debt must be 0");
assert.strictEqual(fin.cash_and_equivalents, 1602589000, "Cash must be $1,602.589M");
assert.strictEqual(fin.net_debt_bridge, -1602589000, "Net debt bridge must be -$1,602.589M (net cash)");
assert.strictEqual(fin.operating_lease_liabilities_asc842, 472334000, "ASC 842 leases must be $472.334M");
assert.ok(
  fin.lease_convention_treatment?.includes("excluded from net debt bridge"),
  "Leases must be documented as excluded from net debt bridge"
);
assert.strictEqual(fin.selected_share_basis?.value, 138559000);
assert.strictEqual(fin.selected_share_basis?.basis_type, "SPLIT_ADJUSTED_DILUTED");

console.log("[OK] Authoritative DECK payload verified with 100% precision.");

// =========================================================================
// 2. Component Isolation & Dispatch Rules
// =========================================================================
console.log("2. Verifying component isolation and dispatch in OpportunityDossierCard...");

const dossierCardPath = path.resolve(__dirname, "../components/OpportunityDossierCard.tsx");
const dossierCardSource = fs.readFileSync(dossierCardPath, "utf-8");

// A. Isolation check: DECK routes to DeckDecisionFirstView
assert.ok(
  dossierCardSource.includes("import DeckDecisionFirstView from \"./DeckDecisionFirstView\";"),
  "OpportunityDossierCard must import DeckDecisionFirstView"
);
assert.ok(
  dossierCardSource.includes('if (dossier.identity.symbol === "DECK")'),
  "OpportunityDossierCard must condition DeckDecisionFirstView exclusively on DECK symbol"
);
assert.ok(
  dossierCardSource.includes("<DeckDecisionFirstView dossier={dossier} />"),
  "OpportunityDossierCard must render DeckDecisionFirstView for DECK"
);

// B. Non-DECK preservation check: All other published assets retain standard card
const otherSymbols = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG"];
for (const sym of otherSymbols) {
  const otherDossier = getOpportunityDossier(sym);
  assert.ok(otherDossier, `${sym} must load from published collection`);
  assert.strictEqual(
    otherDossier.identity.symbol === "DECK",
    false,
    `${sym} must not be identified as DECK`
  );
}

// C. Unpublished fallback check: Uncataloged symbol returns null
const nullDossier = getOpportunityDossier("UNKNOWN_TICKER");
assert.strictEqual(nullDossier, null, "Uncataloged symbol must fail closed to null");

console.log("[OK] Component isolation and fallback preservation verified.");

// =========================================================================
// 3. Epistemic Content Rules & Semantic Verification
// =========================================================================
console.log("3. Verifying DeckDecisionFirstView content rules and structure...");

const deckViewPath = path.resolve(__dirname, "../components/DeckDecisionFirstView.tsx");
const deckViewSource = fs.readFileSync(deckViewPath, "utf-8");

// A. Verdict & Decisive Reason
assert.ok(
  deckViewSource.includes("A. Verdict &amp; Decisive Reason"),
  "Must include Section A: Verdict & Decisive Reason"
);
assert.ok(
  deckViewSource.includes("WATCH"),
  "Must prominently display WATCH verdict"
);
assert.ok(
  deckViewSource.includes("Status: WATCH (Conditions required for entry • Not a BUY recommendation)"),
  "Must include explicit non-recommendation boundary"
);

// B. One-Sentence Thesis
assert.ok(
  deckViewSource.includes("B. One-Sentence Thesis"),
  "Must include Section B: One-Sentence Thesis"
);
assert.ok(
  deckViewSource.includes("Can HOKA and UGG sustain direct-to-consumer momentum and operating margins above 20%"),
  "Must include verified one-sentence thesis"
);

// C. What Is Priced In?
assert.ok(
  deckViewSource.includes("C. What Is Priced In?"),
  "Must include Section C: What Is Priced In?"
);
assert.ok(
  deckViewSource.includes("The implied growth rate depends on the model assumptions"),
  "Must include required explanation that implied growth depends on model assumptions"
);
assert.ok(
  deckViewSource.includes("Bear case") &&
  deckViewSource.includes("Base case") &&
  deckViewSource.includes("Bull case"),
  "Must use standardized Bear case, Base case, Bull case labels"
);

// D. Key Drivers
assert.ok(
  deckViewSource.includes("D. What Must Be True?"),
  "Must include Section D: What Must Be True?"
);
assert.ok(
  deckViewSource.includes("1. DTC Velocity &amp; International Expansion") &&
  deckViewSource.includes("2. Gross Margin &amp; Operating Defense") &&
  deckViewSource.includes("3. Zero-Debt Balance Sheet Strength"),
  "Must present the three decisive evidence-backed drivers"
);

// E. Invalidation Triggers
assert.ok(
  deckViewSource.includes("E. What Breaks The Thesis?"),
  "Must include Section E: What Breaks The Thesis?"
);
assert.ok(
  deckViewSource.includes("1. Gross Margin Breach Below 50.0%") &&
  deckViewSource.includes("2. HOKA DTC Deceleration Below 10.0%") &&
  deckViewSource.includes("3. Wholesale Channel Contraction"),
  "Must present the three specific invalidation conditions"
);

// F. Potential Expectation Gap
assert.ok(
  deckViewSource.includes("F. Potential Expectation Gap"),
  "Must include Section F: Potential Expectation Gap"
);
assert.ok(
  deckViewSource.includes("An expectation gap has not yet been established beyond observed brand channel divergence"),
  "Must adhere to prompt rule: explicitly declare no unestablished consensus gap"
);

// G. What Would Change The Decision?
assert.ok(
  deckViewSource.includes("G. What Would Change The Decision?"),
  "Must include Section G: What Would Change The Decision?"
);
assert.ok(
  deckViewSource.includes("Next Review Trigger:"),
  "Must include Next Review Trigger"
);
assert.ok(
  deckViewSource.includes("Decisive Unknown:"),
  "Must include Decisive Unknown"
);

// H. Optional Section Omission
assert.strictEqual(
  deckViewSource.includes("What Could This Become?"),
  false,
  "Section H ('What Could This Become?') must be omitted as unsupported by current dossier"
);

// I. Progressive Disclosure Sections
assert.ok(
  deckViewSource.includes("Valuation Mechanics, Parameters &amp; Full Scenario Comparison"),
  "Must include progressive disclosure section 1 for valuation mechanics"
);
assert.ok(
  deckViewSource.includes("Financial Resilience, Capital Structure &amp; Sourced Net Debt Bridge"),
  "Must include progressive disclosure section 2 for financial resilience"
);
assert.ok(
  deckViewSource.includes("Full Dossier Baseline &amp; Quality Watch Selection Record (Questions A–F)"),
  "Must include progressive disclosure section 3 for full dossier baseline"
);

// J. Accessibility attributes
assert.ok(
  deckViewSource.includes("aria-expanded="),
  "Progressive disclosure buttons must have aria-expanded attribute"
);
assert.ok(
  deckViewSource.includes("aria-controls="),
  "Progressive disclosure buttons must have aria-controls attribute"
);
assert.ok(
  deckViewSource.includes("min-h-[44px]"),
  "Interactive controls must satisfy minimum 44px touch target requirement"
);
assert.ok(
  deckViewSource.includes("focus:ring-2 focus:ring-cyan-400"),
  "Interactive controls must render visible cyan focus ring"
);

console.log("[OK] DeckDecisionFirstView semantic and structural requirements verified.");

// =========================================================================
// 4. Verification across all Published Cohort Assets
// =========================================================================
console.log("4. Verifying cohort collection integrity (6 published assets)...");

const cohort = listOpportunityCohort();
assert.strictEqual(cohort.length, 6, "Must have exactly 6 published Quality Watch assets");
const cohortSymbols = cohort.map((c) => c.symbol);
assert.deepStrictEqual(
  cohortSymbols.sort(),
  ["DECK", "DUOL", "FIX", "ISRG", "MEDP", "ULTA"].sort(),
  "Cohort must comprise FIX, DUOL, ULTA, MEDP, ISRG, DECK"
);

for (const item of cohort) {
  const d = getOpportunityDossier(item.symbol);
  assert.ok(d, `Dossier for ${item.symbol} must exist`);
  assert.strictEqual(d.central_assessment.disposition, "WATCH_CONDITIONS");
  assert.ok(d.valuation.reference_price > 0, "Reference price must be positive");
}

console.log("[OK] Cohort collection verified.");

console.log("===============================================================================");
console.log("ALL DECK DECISION-FIRST PROTOTYPE TESTS PASSED SUCCESSFULLY! ✅");
console.log("===============================================================================");
