/**
 * frontend/tests/opportunityDossierPilotDraftIntegrity.test.ts
 *
 * Frontend Staging & Draft Isolation Test Suite for ARX Opportunity Dossier Scaling Pilot.
 * Verifies:
 * 1. Default accessor getOpportunityDossier() remains strictly isolated to the 5-company baseline.
 * 2. Uncataloged additions return null from default accessor and false from isCohortSymbol().
 * 3. Draft accessor getPilotDraftDossier() / previewDrafts option cleanly accesses all 20 pilot dossiers.
 * 4. Required sections exist across all 20 dossiers in the pilot draft snapshot.
 * 5. All draft additions carry explicit draft lifecycle states and zero manufactured reviewer metadata.
 * 6. Truly unsupported symbols (e.g. AAPL) return null from both accessors.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

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
} from "../lib/opportunityDossiers";

console.log("Starting Opportunity Dossier Pilot Draft Integrity Suite...");

const repoRoot = path.resolve(__dirname, "../..");
const draftSnapshotPath = path.resolve(__dirname, "../lib/opportunityDossierPilotDrafts.json");
const publishedSnapshotPath = path.resolve(__dirname, "../lib/opportunityDossierSnapshot.json");

// 1. Verify Draft Staging Snapshot Exists and Parses
assert.strictEqual(fs.existsSync(draftSnapshotPath), true, "Draft staging snapshot file must exist in frontend/lib/");
const rawDraft = fs.readFileSync(draftSnapshotPath, "utf-8");
const draftData = JSON.parse(rawDraft);
assert.strictEqual(draftData.snapshot_metadata.cohort_count, 20, "Draft staging snapshot must contain exactly 20 companies");
console.log("[OK] Draft snapshot loaded and verified (20 companies)");

// 2. Strict Production Isolation: Default accessor returns null for draft-only additions
for (const sym of PILOT_EXPANSION_SYMBOLS) {
  assert.strictEqual(isCohortSymbol(sym), false, `${sym} must NOT be in published ODP-05 cohort`);
  const publishedDossier = getOpportunityDossier(sym);
  assert.strictEqual(publishedDossier, null, `Default getOpportunityDossier(${sym}) must return null in production mode`);
}
console.log("[OK] Production baseline isolation verified: draft additions return null from default accessor");

// 3. Draft Accessor Verification: Returns complete dossier when previewDrafts is enabled
for (const sym of TOTAL_PILOT_SYMBOLS) {
  assert.strictEqual(isPilotDraftSymbol(sym), true, `${sym} must be recognized as a pilot draft symbol`);
  const draftDossier = getOpportunityDossier(sym, { previewDrafts: true });
  assert.ok(draftDossier, `Draft dossier must be returned for ${sym}`);
  assert.strictEqual(draftDossier.identity.symbol, sym, `Dossier symbol must match ${sym}`);

  // Required sections verification
  assert.ok(draftDossier.central_assessment, `${sym} must have central_assessment`);
  assert.ok(draftDossier.valuation, `${sym} must have valuation`);
  assert.ok(draftDossier.financial_profile, `${sym} must have financial_profile`);
  assert.ok(draftDossier.monitoring, `${sym} must have monitoring`);
  assert.ok(draftDossier.navigation, `${sym} must have navigation`);

  // Primary filings verification
  assert.ok(draftDossier.financial_profile.primary_filings.length >= 1, `${sym} must have at least one primary filing`);
  for (const f of draftDossier.financial_profile.primary_filings) {
    assert.ok(f.accession_number, `${sym} filing must have accession_number`);
    assert.ok(f.url, `${sym} filing must have direct SEC EDGAR url`);
  }
}
console.log("[OK] All 20 pilot dossiers accessible and structurally complete in draft preview mode");

// 4. Draft Publication Metadata & Absence of Manufactured Reviewers
for (const sym of PILOT_EXPANSION_SYMBOLS) {
  const d = getPilotDraftDossier(sym);
  assert.ok(d, `Draft dossier for ${sym} must exist`);
  const meta = (d as any).publication_metadata;
  assert.ok(meta, `${sym} must have publication_metadata`);
  assert.strictEqual(meta.lifecycle_status, "DRAFT", `${sym} lifecycle_status must be DRAFT`);
  assert.strictEqual(meta.review_status, "PENDING_RESEARCH_REVIEW", `${sym} review_status must be PENDING_RESEARCH_REVIEW`);
  assert.strictEqual(meta.reviewer_id, null, `${sym} reviewer_id must be null`);
  assert.strictEqual(meta.reviewed_at_utc, null, `${sym} reviewed_at_utc must be null`);
}
console.log("[OK] Publication metadata verified: No manufactured reviewer identities in draft additions");

// 5. Zero-Debt vs Funded Debt Invariants
const zeroDebtCohort = ["DUOL", "ULTA", "MEDP", "DECK", "LULU", "EME", "ANET", "PLTR", "ISRG", "CDNS"];
for (const sym of zeroDebtCohort) {
  const d = getPilotDraftDossier(sym);
  assert.ok(d, `Dossier for ${sym} must exist`);
  assert.strictEqual(d.financial_profile.is_zero_debt, true, `${sym} must be marked is_zero_debt=true`);
  assert.strictEqual(d.financial_profile.total_financial_debt, 0, `${sym} total_financial_debt must be 0`);
  assert.ok(d.financial_profile.zero_debt_support, `${sym} must document zero_debt_support`);
}

const fundedDebtCohort = ["FIX", "KLAC", "VRT", "PWR", "SNPS", "NOW", "ELF", "IREN", "CORZ", "MU"];
for (const sym of fundedDebtCohort) {
  const d = getPilotDraftDossier(sym);
  assert.ok(d, `Dossier for ${sym} must exist`);
  assert.strictEqual(d.financial_profile.is_zero_debt, false, `${sym} must be marked is_zero_debt=false`);
  assert.ok(d.financial_profile.total_financial_debt! > 0, `${sym} total_financial_debt must be positive`);
}
console.log("[OK] Capital structure invariants verified: Clean separation of zero-debt from funded debt");

// 6. Unsupported Symbols Fail-Closed Invariant
for (const sym of ["AAPL", "NVDA", "JPM", "BAC", "UNKNOWN"]) {
  assert.strictEqual(isCohortSymbol(sym), false, `${sym} must not be in published cohort`);
  assert.strictEqual(isPilotDraftSymbol(sym), false, `${sym} must not be in pilot cohort`);
  assert.strictEqual(getOpportunityDossier(sym), null, `Default accessor must return null for ${sym}`);
  assert.strictEqual(getOpportunityDossier(sym, { previewDrafts: true }), null, `Draft accessor must return null for ${sym}`);
}
console.log("[OK] Truly unsupported symbols fail-closed to null across both production and draft modes");

console.log("ALL OPPORTUNITY DOSSIER PILOT DRAFT INTEGRITY TESTS PASSED ✅");
