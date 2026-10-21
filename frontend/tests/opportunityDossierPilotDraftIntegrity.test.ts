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
assert.strictEqual(draftData.snapshot_metadata.cohort_count, 20, "Draft staging snapshot must contain exactly 20 companies");
assert.strictEqual(draftData.snapshot_metadata.is_draft_staging, true, "is_draft_staging must be true");
assert.strictEqual(draftData.snapshot_metadata.review_gate, "PENDING_INDEPENDENT_RESEARCH_REVIEW", "review_gate must be PENDING_INDEPENDENT_RESEARCH_REVIEW");
console.log("[OK] Draft snapshot loaded and verified (20 companies, review_gate=PENDING_INDEPENDENT_RESEARCH_REVIEW)");

// 2. Verify Manifest References Canonical Baseline
if (fs.existsSync(manifestPath)) {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
  assert.strictEqual(manifest.odp05_baseline_sha256, EXPECTED_ODP05_LEDGER_SHA256, "Manifest must reference canonical ODP-05 ledger hash");
  assert.ok(manifest.pilot_ledger_sha256, "Manifest must record pilot_ledger_sha256");
  assert.strictEqual(manifest.cohort_count, 20, "Manifest cohort_count must be 20");
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
console.log("[OK] Representative ULTA deep evidence tracing verified (CIK 0001403568, $592.94 ref price, 15.45% CAGR, Note 7 debt $339.6M, ASC 842 lease separated)");

// 6. Zero-Debt vs Funded Debt Invariants Across All 20 Assets
const zeroDebtCohort = ["DUOL", "MEDP", "DECK", "LULU", "EME", "ANET", "PLTR", "ISRG", "CDNS"];
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
console.log("[OK] Capital structure invariants verified: 9 zero-debt vs 11 funded-debt cleanly partitioned");

// 7. Unsupported Symbols Fail-Closed Invariant
for (const sym of ["AAPL", "NVDA", "JPM", "BAC", "UNKNOWN"]) {
  assert.strictEqual(isCohortSymbol(sym), false, `${sym} must not be in published cohort`);
  if (!IS_PILOT_DRAFTS_BUILD_ENABLED) {
    assert.strictEqual(isPilotDraftSymbol(sym), false, `${sym} must not be in pilot cohort`);
  }
  assert.strictEqual(getOpportunityDossier(sym), null, `Default accessor must return null for ${sym}`);
  assert.strictEqual(getOpportunityDossier(sym, { previewDrafts: true }), null, `Draft accessor must return null for ${sym}`);
}
console.log("[OK] Truly unsupported symbols fail-closed to null across both production and draft modes");

// 8. Runtime Behavior & Isolation Branches
if (!IS_PILOT_DRAFTS_BUILD_ENABLED) {
  // Mode A: Default Production Mode (NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS is unset or false)
  for (const sym of PILOT_EXPANSION_SYMBOLS) {
    assert.strictEqual(isCohortSymbol(sym), false, `${sym} must NOT be in published ODP-05 cohort`);
    assert.strictEqual(isPilotDraftSymbol(sym), false, `${sym} must NOT be recognized as pilot symbol when drafts disabled`);
    const publishedDossier = getOpportunityDossier(sym);
    assert.strictEqual(publishedDossier, null, `Default getOpportunityDossier(${sym}) must return null in production mode`);
    const draftAttempt = getPilotDraftDossier(sym);
    assert.strictEqual(draftAttempt, null, `getPilotDraftDossier(${sym}) must return null when build flag disabled`);
  }
  assert.strictEqual(listOpportunityCohort().length, 5, "listOpportunityCohort must return exactly 5 published companies");
  assert.strictEqual(listPilotDraftCohort().length, 5, "listPilotDraftCohort must fall back to 5 companies when drafts disabled");
  console.log("[OK] Production baseline isolation verified: draft additions return null from all accessors");

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
  assert.strictEqual(listPilotDraftCohort().length, 20, "listPilotDraftCohort must return 20 when drafts enabled");

  // Default accessor without previewDrafts option MUST STILL return null for draft additions!
  for (const sym of PILOT_EXPANSION_SYMBOLS) {
    assert.strictEqual(getOpportunityDossier(sym), null, `Default getOpportunityDossier(${sym}) must STILL return null even when build flag is active`);
    const preview = getOpportunityDossier(sym, { previewDrafts: true });
    assert.ok(preview, `Preview getOpportunityDossier(${sym}, { previewDrafts: true }) must return dossier`);
    assert.strictEqual(preview.identity.symbol, sym);
  }
  console.log("[OK] Preview-enabled accessor verified for all 20 dossiers while default path remains isolated");
}

console.log("ALL OPPORTUNITY DOSSIER PILOT DRAFT INTEGRITY TESTS PASSED ✅");
