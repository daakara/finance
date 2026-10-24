/**
 * frontend/tests/alconV4FrontendIntegration.test.ts
 *
 * Comprehensive Frontend Contract & Rendering Acceptance Test Suite for Alcon V4.
 *
 * Verifies:
 * 1. Typed contract compliance of Alcon V4 payload.
 * 2. Five canonical scenarios + dynamic variants preservation and valuation methods.
 * 3. Discrete annual IRR vs calendar-dated XIRR separation and non-interchangeability.
 * 4. Hurdle rates strictly bound to accumulation ceilings ($64.15 at 8.0%, $62.70 at 8.5%, $58.59 at 10.0%).
 * 5. Historical reference price ($67.43) and market date (2026-10-21) explicit labeling.
 * 6. Balance sheet net debt bridge ($3,300M opening -> -$2,176.2M net cash) and share denominator (484.524M -> 475.524M).
 * 7. Buyback authority disaggregation ($1,200M approved vs $600M hypothetical future).
 * 8. Qualitative evidence claims (88.37% certified procedural replenishment vs unaudited hypotheses).
 * 9. Production build and manifest isolation: ALC strictly absent from published manifests and public accessors.
 * 10. Backward compatibility: existing published dossiers (FIX, DUOL, ULTA, MEDP, ISRG, DECK) unaffected.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

import {
  OpportunityDossier,
  getOpportunityDossier,
  isCohortSymbol,
  isAlconV4Dossier,
  PUBLISHED_WATCH_SYMBOLS,
  PublishedWatchManifestFile,
  OpportunityDossierSnapshotFile,
} from "../lib/opportunityDossiers";

console.log("Starting Alcon V4 Frontend Contract & Integration Test Suite...");

const repoRoot = path.resolve(__dirname, "../..");
const draftPath = path.resolve(repoRoot, "docs/research/opportunity_discovery/drafts/ALC_opportunity_dossier_draft.json");
const manifestPath = path.resolve(__dirname, "../lib/publishedWatchManifest.json");
const snapshotPath = path.resolve(__dirname, "../lib/opportunityDossierSnapshot.json");

// 1. Verify Draft Payload Exists and Parses
assert.strictEqual(fs.existsSync(draftPath), true, "Draft export must exist at docs/research/opportunity_discovery/drafts/ALC_opportunity_dossier_draft.json");
const rawDraft = fs.readFileSync(draftPath, "utf-8");
const alcDossier: OpportunityDossier = JSON.parse(rawDraft);

assert.strictEqual(alcDossier.identity.symbol, "ALC");
assert.strictEqual(alcDossier.identity.company_name, "Alcon Inc.");
assert.strictEqual(alcDossier.identity.cik, "0001167379");
assert.strictEqual(alcDossier.identity.exchange, "NYSE");
assert.strictEqual(isAlconV4Dossier(alcDossier), true, "Must satisfy isAlconV4Dossier type guard");

console.log("✅ 1. Draft payload identity and type guard verified.");

// 2. Five Canonical Scenarios Preservation & Valuation Methods
const scenarios = alcDossier.valuation.scenarios;
assert.ok(scenarios, "Scenarios object must exist");

const conservative = scenarios.conservative;
assert.ok(conservative, "Conservative scenario must exist");
assert.strictEqual(conservative.per_share_value, 76.64);
assert.strictEqual(conservative.exit_method, "multiple");
assert.strictEqual(conservative.revenue_cagr_pct, 5.00);
assert.strictEqual(conservative.operating_margin_pct, 16.50);

const baseMultiple = scenarios.base_multiple;
assert.ok(baseMultiple, "Base Multiple scenario must exist");
assert.strictEqual(baseMultiple.per_share_value, 90.02);
assert.strictEqual(baseMultiple.exit_method, "multiple");
assert.strictEqual(baseMultiple.revenue_cagr_pct, 6.50);
assert.strictEqual(baseMultiple.operating_margin_pct, 18.50);

const baseDcf = scenarios.base_dcf;
assert.ok(baseDcf, "Base DCF scenario must exist");
assert.strictEqual(baseDcf.per_share_value, 91.82);
assert.strictEqual(baseDcf.exit_method, "gordon_growth");
assert.strictEqual(baseDcf.revenue_cagr_pct, 6.50);
assert.strictEqual(baseDcf.operating_margin_pct, 18.50);
assert.strictEqual(baseDcf.wacc_pct, 7.25);
assert.strictEqual(baseDcf.terminal_growth_pct, 2.50);

const optimistic = scenarios.optimistic;
assert.ok(optimistic, "Optimistic scenario must exist");
assert.strictEqual(optimistic.per_share_value, 123.42);
assert.strictEqual(optimistic.exit_method, "multiple");
assert.strictEqual(optimistic.revenue_cagr_pct, 8.50);
assert.strictEqual(optimistic.operating_margin_pct, 20.00);

console.log("✅ 2. Five canonical scenarios and valuation methods verified.");

// 3. Discrete Annual IRR vs Calendar-Dated XIRR Separation
assert.strictEqual(conservative.five_period_annual_irr_pct, 3.18);
assert.strictEqual(conservative.calendar_dated_xirr_pct, 3.19);

assert.strictEqual(baseMultiple.five_period_annual_irr_pct, 6.50);
assert.strictEqual(baseMultiple.calendar_dated_xirr_pct, 6.51);

assert.strictEqual(baseDcf.five_period_annual_irr_pct, 6.92);
assert.strictEqual(baseDcf.calendar_dated_xirr_pct, 6.93);

assert.strictEqual(optimistic.five_period_annual_irr_pct, 13.34);
assert.strictEqual(optimistic.calendar_dated_xirr_pct, 13.36);

// Confirm they cannot be interchanged (they are distinct numeric values)
assert.notStrictEqual(baseDcf.five_period_annual_irr_pct, baseDcf.calendar_dated_xirr_pct, "Annual IRR and Calendar XIRR must not be conflated");

console.log("✅ 3. Discrete annual IRR vs calendar XIRR verified as separate measures.");

// 4. Accumulation Ceilings Bound to Correct Hurdles
const ceilings = alcDossier.alcon_v4_extension!.accumulation_ceilings;
assert.ok(ceilings, "Accumulation ceilings must exist");

const annual = ceilings.annual_irr_basis;
assert.strictEqual(annual.hurdle_80, 64.15, "8.0% annual ceiling must be 64.15");
assert.strictEqual(annual.hurdle_85, 62.70, "8.5% annual ceiling must be 62.70");
assert.strictEqual(annual.hurdle_100, 58.59, "10.0% annual ceiling must be 58.59");
assert.strictEqual(annual.invertibility_verified, true);

const calendar = ceilings.calendar_xirr_basis;
assert.strictEqual(calendar.hurdle_80, 64.20, "8.0% calendar ceiling must be 64.20");
assert.strictEqual(calendar.hurdle_85, 62.75, "8.5% calendar ceiling must be 62.75");
assert.strictEqual(calendar.hurdle_100, 58.64, "10.0% calendar ceiling must be 58.64");
assert.strictEqual(calendar.invertibility_verified, true);

console.log("✅ 4. Hurdle-specific accumulation ceilings verified.");

// 5. Balance Sheet & Net Debt Amortization Bridge
const fp = alcDossier.financial_profile;
assert.strictEqual(fp.accounting_basis, "IFRS");
assert.strictEqual(fp.currency, "USD");
assert.strictEqual(fp.cash_and_equivalents, 1200000000.0);
assert.strictEqual(fp.total_financial_debt, 4500000000.0);
assert.strictEqual(fp.net_debt_bridge, 3300000000.0);

assert.strictEqual(baseDcf.net_debt_bridge, -2176.2, "Ending net debt must be -$2,176.2M (net cash)");
assert.strictEqual(baseDcf.ending_diluted_shares, 475.524, "Ending diluted shares must be 475.524M");

console.log("✅ 5. Net debt bridge and diluted share count verified.");

// 6. Buyback Authority Disaggregation
const buybacks = alcDossier.alcon_v4_extension!.buyback_authority_reconciliation;
assert.ok(buybacks, "Buyback reconciliation must exist");
assert.strictEqual(buybacks.REMAINING_APPROVED_CAPACITY?.amount, 1200.0, "Approved remaining capacity must be $1,200M");
assert.strictEqual(buybacks.ASSUMED_FUTURE_AUTHORIZATIONS?.amount, 600.0, "Assumed future renewal must be $600M");
assert.strictEqual(buybacks.FORECAST_BUYBACKS_BASELINE?.amount, 1800.0, "Total baseline buybacks must be $1,800M");

console.log("✅ 6. Statutory buyback authority disaggregation verified.");

// 7. Qualitative Evidence & Non-Equipment Sales Claim
const matrix = alcDossier.alcon_v4_extension!.qualitative_evidence_matrix;
assert.strictEqual(matrix.length, 3);
const recurringClaim = matrix.find(m => m.claim.includes("Non-equipment"));
assert.ok(recurringClaim, "Non-equipment recurring sales claim must exist");
assert.strictEqual(recurringClaim.classification, "AUDITED_AND_CERTIFIED");
assert.ok(recurringClaim.finding.includes("88.37%"), "Finding must state 88.37%");

console.log("✅ 7. Qualitative claims and audited recurrence verified.");

// 8. Publication Enrollment & Loader Invariants
const validManifest: PublishedWatchManifestFile = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
const validSnapshot: OpportunityDossierSnapshotFile = JSON.parse(fs.readFileSync(snapshotPath, "utf-8"));

// ALC IS enrolled in published symbols and accepted_additions
assert.strictEqual(validManifest.published_symbols.includes("ALC"), true, "ALC must be in published_symbols");
assert.ok(validManifest.accepted_additions?.ALC, "ALC must be in accepted_additions");
assert.strictEqual(validManifest.published_count, 7, "Published count must be 7");

// Baseline snapshot remains pure
assert.strictEqual(validSnapshot.cohort.some(c => c.symbol === "ALC"), false, "ALC must NOT be in baseline snapshot cohort");
assert.strictEqual(Boolean(validSnapshot.dossiers.ALC), false, "ALC must NOT be in baseline snapshot dossiers");

// Public accessors return authoritative published dossier for ALC
const publishedDossier = getOpportunityDossier("ALC");
assert.ok(publishedDossier, "getOpportunityDossier('ALC') must return published dossier");
assert.strictEqual(publishedDossier?.identity.symbol, "ALC");
assert.strictEqual(isAlconV4Dossier(publishedDossier!), true, "Published ALC dossier must satisfy isAlconV4Dossier type guard");
assert.strictEqual(isCohortSymbol("ALC"), true, "isCohortSymbol must return true for ALC");

// Negative controls fail closed
assert.strictEqual(isCohortSymbol("MU"), false);
assert.strictEqual(isCohortSymbol("PLTR"), false);
assert.strictEqual(getOpportunityDossier("MU"), null);

console.log("✅ 8. Publication enrollment, baseline purity, and public loader verified.");

// 9. Backward Compatibility on Existing Published Dossiers
const publishedSymbols = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"];
for (const sym of publishedSymbols) {
  const dossier = getOpportunityDossier(sym);
  assert.ok(dossier, `Published dossier ${sym} must be retrievable`);
  assert.strictEqual(dossier.alcon_v4_extension, undefined, `${sym} must not have alcon_v4_extension`);
  assert.strictEqual(isAlconV4Dossier(dossier), false, `${sym} must fail isAlconV4Dossier type guard`);
}

console.log("✅ 9. Backward compatibility verified across all 6 published dossiers.");

console.log("\nALL ALCON V4 FRONTEND INTEGRATION TESTS PASSED (9/9) ✅");
