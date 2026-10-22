/**
 * frontend/tests/opportunityDossierPilotDraftIntegrity.test.ts
 *
 * Frontend Staging & Draft Isolation Test Suite for ARX Opportunity Dossier Scaling Pilot.
 * Verifies:
 * 1. Default runtime accessor fails closed to baseline 5 without preview flag.
 * 2. Uncataloged additions return null from default accessor and false from isCohortSymbol().
 * 3. Draft staging file contains all 20 dossiers with structural completeness.
 * 4. Honest review metadata:
 *    - Published baseline carries lifecycle_status=PUBLISHED_PRODUCTION, review_status=RESEARCH_REVIEW_UNRECORDED.
 *    - Draft additions carry lifecycle_status=DRAFT_STAGING, review_status=PENDING_RESEARCH_REVIEW,
 *      reviewer_id=null, reviewed_at_utc=null, dossier_version=0.1.0-draft.
 * 5. Representative ULTA deep evidence tracing & capital structure (ASC 842 lease vs zero funded debt).
 * 6. Capital structure classification across all 20 assets (10 zero-debt, 10 funded-debt).
 * 7. Truly unsupported symbols fail closed across all accessors.
 * 8. Build-time isolation boundary: child process with NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS="true"
 *    unlocks preview accessor while keeping default accessor strictly isolated.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";

import {
  getOpportunityDossier,
  getPilotDraftDossier,
  listOpportunityCohort,
  listPilotDraftCohort,
  isCohortSymbol,
  isPilotDraftSymbol,
  ODP05_COHORT_SYMBOLS,
  PILOT_EXPANSION_SYMBOLS,
  TOTAL_PILOT_SYMBOLS,
  IS_PILOT_DRAFTS_BUILD_ENABLED,
} from "../lib/opportunityDossiers";

console.log(`Starting Opportunity Dossier Pilot Draft Integrity Suite (IS_PILOT_DRAFTS_BUILD_ENABLED=${IS_PILOT_DRAFTS_BUILD_ENABLED})...`);

const repoRoot = path.resolve(__dirname, "../..");
const draftSnapshotPath = path.resolve(__dirname, "../lib/opportunityDossierPilotDrafts.json");
const publishedSnapshotPath = path.resolve(__dirname, "../lib/opportunityDossierSnapshot.json");
const manifestPath = path.resolve(repoRoot, "docs/research/opportunity_discovery/pilot_scaling_manifest.json");
const EXPECTED_ODP05_LEDGER_SHA256 = "a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d";

// 1. Verify Draft Staging Snapshot Exists and Parses
assert.strictEqual(fs.existsSync(draftSnapshotPath), true, "Draft staging snapshot file must exist in frontend/lib/");
const rawDraft = fs.readFileSync(draftSnapshotPath, "utf-8");
const draftData = JSON.parse(rawDraft);
assert.strictEqual(draftData.snapshot_metadata.cohort_count, 19, "Draft staging snapshot must contain exactly 19 companies");
assert.strictEqual(draftData.snapshot_metadata.is_draft_staging, true, "is_draft_staging must be true");
assert.strictEqual(draftData.snapshot_metadata.review_gate, "PENDING_INDEPENDENT_RESEARCH_REVIEW", "review_gate must be PENDING_INDEPENDENT_RESEARCH_REVIEW");
console.log("[OK] Draft snapshot loaded and verified (19 companies, review_gate=PENDING_INDEPENDENT_RESEARCH_REVIEW)");

// 2. Verify Manifest References Canonical Baseline
if (fs.existsSync(manifestPath)) {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
  assert.strictEqual(manifest.odp05_baseline_sha256, EXPECTED_ODP05_LEDGER_SHA256, "Manifest must reference canonical ODP-05 ledger hash");
  assert.ok(manifest.pilot_ledger_sha256, "Manifest must record pilot_ledger_sha256");
  assert.strictEqual(manifest.cohort_count, 19, "Manifest cohort_count must be 19");
  console.log("[OK] Pilot manifest verified against canonical baseline ODP-05 hash");
}

// 3. Verify Baseline Snapshot Metadata & Frozen State
assert.strictEqual(fs.existsSync(publishedSnapshotPath), true, "Published snapshot must exist in frontend/lib/");
const publishedData = JSON.parse(fs.readFileSync(publishedSnapshotPath, "utf-8"));
assert.strictEqual(publishedData.snapshot_metadata.source_ledger_sha256, EXPECTED_ODP05_LEDGER_SHA256, "Published snapshot must source from canonical ODP-05 ledger");
assert.strictEqual(publishedData.snapshot_metadata.review_timestamp_utc, null, "Published snapshot review_timestamp_utc must be unrecorded (null)");
console.log("[OK] Published baseline snapshot verified against canonical ODP-05 SHA-256 and unrecorded review timestamp");

// 4. Draft Staging Publication Metadata Honesty
// Check baseline assets in staging draft
for (const sym of ODP05_COHORT_SYMBOLS) {
  const d = draftData.dossiers[sym];
  assert.ok(d, `Baseline dossier for ${sym} must exist in staging JSON`);
  const meta = d.publication_metadata;
  assert.ok(meta, `Baseline dossier for ${sym} must carry publication_metadata`);
  assert.strictEqual(meta.lifecycle_status, "PUBLISHED_PRODUCTION", `${sym} lifecycle_status must be PUBLISHED_PRODUCTION`);
  assert.strictEqual(meta.review_status, "RESEARCH_REVIEW_UNRECORDED", `${sym} review_status must be RESEARCH_REVIEW_UNRECORDED`);
  assert.strictEqual(meta.reviewer_id, null, `${sym} reviewer_id must be null`);
  assert.strictEqual(meta.reviewed_at_utc, null, `${sym} reviewed_at_utc must be null`);
  assert.strictEqual(meta.dossier_version, "1.0.0-production", `${sym} dossier_version must be 1.0.0-production`);
}

// Check draft additions in staging draft
for (const sym of PILOT_EXPANSION_SYMBOLS) {
  const d = draftData.dossiers[sym];
  assert.ok(d, `Draft dossier for ${sym} must exist in staging JSON`);
  assert.strictEqual(d.identity.symbol, sym, `Dossier symbol must match ${sym}`);

  // Required sections verification
  assert.ok(d.central_assessment, `${sym} must have central_assessment`);
  assert.ok(d.valuation, `${sym} must have valuation`);
  assert.ok(d.financial_profile, `${sym} must have financial_profile`);
  assert.ok(d.monitoring, `${sym} must have monitoring`);
  assert.ok(d.navigation, `${sym} must have navigation`);

  // Primary filings verification
  assert.ok(d.financial_profile.primary_filings.length >= 1, `${sym} must have at least one primary filing`);
  for (const f of d.financial_profile.primary_filings) {
    assert.ok(f.accession_number, `${sym} filing must have accession_number`);
    assert.ok(f.url, `${sym} filing must have direct SEC EDGAR url`);
  }

  // Publication metadata honesty
  const meta = d.publication_metadata;
  assert.ok(meta, `${sym} must have publication_metadata`);
  assert.strictEqual(meta.lifecycle_status, "DRAFT_STAGING", `${sym} lifecycle_status must be DRAFT_STAGING`);
  assert.strictEqual(meta.review_status, "PENDING_RESEARCH_REVIEW", `${sym} review_status must be PENDING_RESEARCH_REVIEW`);
  assert.strictEqual(meta.reviewer_id, null, `${sym} reviewer_id must be null`);
  assert.strictEqual(meta.reviewed_at_utc, null, `${sym} reviewed_at_utc must be null`);
  assert.strictEqual(meta.dossier_version, "0.1.0-draft", `${sym} dossier_version must be 0.1.0-draft`);
}
console.log("[OK] Staging publication metadata verified: Honest unrecorded review status for baseline and pending review for drafts");

// 5. Representative ULTA Deep Sourced Evidence Tracing Verification
const ulta = draftData.dossiers["ULTA"];
assert.ok(ulta, "ULTA draft dossier must exist in staging snapshot");
assert.ok(ulta.source_evidence_tracing, "ULTA must include explicit source_evidence_tracing");
const tracing = ulta.source_evidence_tracing;
assert.strictEqual(tracing.company_identity.cik, "0001403568", "ULTA CIK must be 0001403568");
assert.strictEqual(tracing.revenue.reported_ttm_value_usd, 12955517000, "ULTA TTM revenue must match derived fact");
assert.strictEqual(tracing.operating_income.reported_ttm_value_usd, 1614258000, "ULTA TTM operating income must match derived fact");
assert.strictEqual(tracing.funded_debt.is_zero_debt, false, "ULTA must be classified as funded debt under Note 7");
assert.strictEqual(tracing.funded_debt.total_financial_debt_usd, 339578000, "ULTA total_financial_debt must be $339,578,000");
assert.strictEqual(tracing.lease_treatment.total_lease_liabilities_usd, 2184453000, "ULTA total ASC 842 lease liabilities must match filing ($2.18B)");
assert.strictEqual(tracing.reference_price.reference_price, 592.94, "ULTA reference price must be $592.94");
assert.strictEqual(tracing.valuation_model.solved_5y_revenue_cagr_pct, 15.45, "ULTA solved 5Y revenue CAGR must be 15.45%");
assert.strictEqual(tracing.valuation_model.roundtrip_value, 593.00, "ULTA roundtrip value must be $593.00");
assert.strictEqual(ulta.financial_profile.is_zero_debt, false, "ULTA financial profile must mark is_zero_debt=false");
assert.strictEqual(ulta.financial_profile.total_financial_debt, 339578000, "ULTA financial profile total_financial_debt must be 339578000");
assert.strictEqual(ulta.financial_profile.net_debt_bridge, 126127000, "ULTA financial profile net_debt_bridge must be 126127000");
assert.strictEqual(ulta.financial_profile.operating_lease_liabilities_asc842, 1871805000, "ULTA financial profile must report noncurrent lease liabilities");
assert.strictEqual(ulta.valuation.reference_price, 592.94, "ULTA valuation reference price must match");
assert.strictEqual(ulta.valuation.solved_5y_cagr_pct, 15.45, "ULTA valuation solved CAGR must match");
assert.ok(ulta.source_evidence_tracing.selection_rationale, "ULTA must include structured selection rationale");
assert.ok(ulta.source_evidence_tracing.selection_rationale.why_this_business, "ULTA must answer why_this_business");
console.log("[OK] Representative ULTA deep evidence tracing verified (CIK 0001403568, $592.94 ref price, 15.45% CAGR, Note 7 debt $339.6M, ASC 842 lease separated)");

// 5B. Expansion Additions (MEDP, ISRG, DECK) Evidence Tracing & Share Measurement Period
const expectedQ2Period = "Three months ended June 30, 2026";

// MEDP
const medp = draftData.dossiers["MEDP"];
assert.ok(medp, "MEDP draft dossier must exist in staging snapshot");
assert.ok(medp.source_evidence_tracing, "MEDP must include explicit source_evidence_tracing");
assert.strictEqual(medp.financial_profile.selected_share_basis.measurement_period, expectedQ2Period, "MEDP share basis measurement period must be Q2");
assert.strictEqual(medp.source_evidence_tracing.shares_outstanding.measurement_period, expectedQ2Period, "MEDP source tracing share measurement period must be Q2");
assert.strictEqual(medp.financial_profile.is_zero_debt, true, "MEDP is_zero_debt must be true");
assert.strictEqual(medp.financial_profile.total_financial_debt, 0, "MEDP total_financial_debt must be 0");
assert.strictEqual(medp.valuation.reference_price, 618.47, "MEDP reference price must be $618.47");
assert.strictEqual(medp.valuation.solved_5y_cagr_pct, 29.41, "MEDP solved CAGR must be 29.41%");
assert.ok(medp.source_evidence_tracing.selection_rationale.why_this_business, "MEDP must answer why_this_business");

// ISRG
const isrg = draftData.dossiers["ISRG"];
assert.ok(isrg, "ISRG draft dossier must exist in staging snapshot");
assert.ok(isrg.source_evidence_tracing, "ISRG must include explicit source_evidence_tracing");
assert.strictEqual(isrg.financial_profile.selected_share_basis.measurement_period, expectedQ2Period, "ISRG share basis measurement period must be Q2");
assert.strictEqual(isrg.source_evidence_tracing.shares_outstanding.measurement_period, expectedQ2Period, "ISRG source tracing share measurement period must be Q2");
assert.strictEqual(isrg.financial_profile.is_zero_debt, true, "ISRG is_zero_debt must be true");
assert.strictEqual(isrg.financial_profile.total_financial_debt, 0, "ISRG total_financial_debt must be 0");
assert.strictEqual(isrg.valuation.reference_price, 426.70, "ISRG reference price must be $426.70");
assert.strictEqual(isrg.valuation.solved_5y_cagr_pct, 36.87, "ISRG solved CAGR must be 36.87%");
assert.ok(!isrg.central_assessment.presentation_summary.includes("unassailable"), "ISRG must not use hyperbolic language 'unassailable'");
assert.ok(isrg.central_assessment.strongest_supporting_evidence.includes("85%"), "ISRG must state 85% recurring revenue definition");
assert.ok(isrg.source_evidence_tracing.selection_rationale.why_this_business, "ISRG must answer why_this_business");

// DECK
const deck = draftData.dossiers["DECK"];
assert.ok(deck, "DECK draft dossier must exist in staging snapshot");
assert.ok(deck.source_evidence_tracing, "DECK must include explicit source_evidence_tracing");
assert.strictEqual(deck.financial_profile.selected_share_basis.measurement_period, expectedQ2Period, "DECK share basis measurement period must be Q2");
assert.strictEqual(deck.source_evidence_tracing.shares_outstanding.measurement_period, expectedQ2Period, "DECK source tracing share measurement period must be Q2");
assert.strictEqual(deck.financial_profile.is_zero_debt, true, "DECK is_zero_debt must be true");
assert.strictEqual(deck.financial_profile.total_financial_debt, 0, "DECK total_financial_debt must be 0");
assert.strictEqual(deck.valuation.reference_price, 85.72, "DECK reference price must be $85.72");
assert.strictEqual(deck.valuation.solved_5y_cagr_pct, -2.34, "DECK solved CAGR must be -2.34%");
assert.ok(!deck.central_assessment.strongest_supporting_evidence.includes("market share gains (>25% YoY)"), "DECK must not claim market share gains >25% without data");
assert.ok(deck.central_assessment.strongest_supporting_evidence.includes("17.3%"), "DECK must cite HOKA DTC +17.3% YoY");
assert.ok(deck.source_evidence_tracing.selection_rationale.why_this_business, "DECK must answer why_this_business");
console.log("[OK] Expansion additions (MEDP, ISRG, DECK) deep evidence tracing and share measurement periods verified");

// 6. Zero-Debt vs Funded Debt Invariants Across All 19 Assets
const zeroDebtCohort = ["DUOL", "MEDP", "DECK", "LULU", "EME", "ANET", "ISRG", "CDNS"];
for (const sym of zeroDebtCohort) {
  const d = draftData.dossiers[sym];
  assert.ok(d, `Dossier for ${sym} must exist`);
  assert.strictEqual(d.financial_profile.is_zero_debt, true, `${sym} must be marked is_zero_debt=true`);
  assert.strictEqual(d.financial_profile.total_financial_debt, 0, `${sym} total_financial_debt must be 0`);
  assert.ok(d.financial_profile.zero_debt_support, `${sym} must document zero_debt_support`);
}

const fundedDebtCohort = ["FIX", "ULTA", "KLAC", "VRT", "PWR", "SNPS", "NOW", "ELF", "IREN", "CORZ", "MU"];
for (const sym of fundedDebtCohort) {
  const d = draftData.dossiers[sym];
  assert.ok(d, `Dossier for ${sym} must exist`);
  assert.strictEqual(d.financial_profile.is_zero_debt, false, `${sym} must be marked is_zero_debt=false`);
  assert.ok(d.financial_profile.total_financial_debt! > 0, `${sym} total_financial_debt must be positive`);
}
console.log("[OK] Capital structure invariants verified: 8 zero-debt vs 11 funded-debt cleanly partitioned");

// 7. Unsupported and Excluded Symbols Fail-Closed Invariant
for (const sym of ["AAPL", "NVDA", "JPM", "BAC", "UNKNOWN", "MU", "PLTR"]) {
  assert.strictEqual(isCohortSymbol(sym), false, `${sym} must not be in published Quality Watch cohort`);
  if (!IS_PILOT_DRAFTS_BUILD_ENABLED) {
    assert.strictEqual(isPilotDraftSymbol(sym), false, `${sym} must not be in pilot cohort`);
  }
  assert.strictEqual(getOpportunityDossier(sym), null, `Default accessor must return null for ${sym}`);
  assert.strictEqual(getOpportunityDossier(sym, { previewDrafts: true }), null, `Draft accessor must return null for ${sym}`);
}
console.log("[OK] Truly unsupported and excluded symbols (including MU and PLTR) fail-closed to null across both production and draft modes");

// 8. Runtime Behavior & Isolation Branches
const publishedAdditions = ["ULTA", "MEDP", "ISRG", "DECK"];
const unselectedDrafts = PILOT_EXPANSION_SYMBOLS.filter(s => !publishedAdditions.includes(s));

if (!IS_PILOT_DRAFTS_BUILD_ENABLED) {
  // Mode A: Default Production Mode (NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS is unset or false)
  // Verify published additions are active
  for (const sym of publishedAdditions) {
    assert.strictEqual(isCohortSymbol(sym), true, `${sym} must be in published Quality Watch cohort`);
    const publishedDossier = getOpportunityDossier(sym);
    assert.ok(publishedDossier, `Default getOpportunityDossier(${sym}) must return published dossier`);
    assert.strictEqual(publishedDossier.identity.symbol, sym);
  }

  // Verify unselected draft additions remain isolated
  for (const sym of unselectedDrafts) {
    assert.strictEqual(isCohortSymbol(sym), false, `${sym} must NOT be in published Quality Watch cohort`);
    assert.strictEqual(isPilotDraftSymbol(sym), false, `${sym} must NOT be recognized as pilot symbol when drafts disabled`);
    const publishedDossier = getOpportunityDossier(sym);
    assert.strictEqual(publishedDossier, null, `Default getOpportunityDossier(${sym}) must return null in production mode`);
    const draftAttempt = getPilotDraftDossier(sym);
    assert.strictEqual(draftAttempt, null, `getPilotDraftDossier(${sym}) must return null when build flag disabled`);
  }
  assert.strictEqual(listOpportunityCohort().length, 6, "listOpportunityCohort must return exactly 6 published Quality Watch companies (FIX, DUOL, ULTA, MEDP, ISRG, DECK)");
  assert.strictEqual(listPilotDraftCohort().length, 6, "listPilotDraftCohort must fall back to 6 companies when drafts disabled");
  console.log("[OK] Production baseline isolation verified: unselected draft additions return null from all accessors while additions are published");

  // Mode B: Execute Subprocess with NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS="true" to verify preview branch
  console.log("Spawning subprocess to verify preview mode under NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS=true...");
  const npxCmd = process.platform === "win32" ? "npx.cmd" : "npx";
  execSync(`${npxCmd} tsx tests/opportunityDossierPilotDraftIntegrity.test.ts`, {
    cwd: path.resolve(__dirname, ".."),
    env: { ...process.env, NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS: "true" },
    stdio: "inherit",
  });
  console.log("[OK] Both production-mode and preview-mode test passes completed successfully!");
} else {
  // Mode B (Child / Preview Mode): NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS is "true"
  for (const sym of TOTAL_PILOT_SYMBOLS) {
    assert.strictEqual(isPilotDraftSymbol(sym), true, `${sym} must be recognized as pilot symbol when drafts enabled`);
  }
  // 2 baseline watch (FIX, DUOL) + 2 held baseline (IREN, CORZ) + 14 additions = 18 watch assets (MU and PLTR excluded)
  assert.strictEqual(listPilotDraftCohort().length, 18, "listPilotDraftCohort must return 18 when drafts enabled (MU and PLTR excluded)");

  // Default accessor without previewDrafts option MUST STILL return null for unselected draft additions!
  for (const sym of unselectedDrafts) {
    assert.strictEqual(getOpportunityDossier(sym), null, `Default getOpportunityDossier(${sym}) must STILL return null even when build flag is active`);
    const preview = getOpportunityDossier(sym, { previewDrafts: true });
    assert.ok(preview, `Preview getOpportunityDossier(${sym}, { previewDrafts: true }) must return dossier`);
    assert.strictEqual(preview.identity.symbol, sym);
  }
  console.log("[OK] Preview-enabled accessor verified for all 18 watch dossiers while unselected drafts remain isolated from default accessor");
}

console.log("ALL OPPORTUNITY DOSSIER PILOT DRAFT INTEGRITY TESTS PASSED ✅");
