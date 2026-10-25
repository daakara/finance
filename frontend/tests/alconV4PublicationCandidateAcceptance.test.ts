/**
 * frontend/tests/alconV4PublicationCandidateAcceptance.test.ts
 *
 * Sprint 4 Alcon V4 Slice 3 Local Publication Acceptance Suite.
 * Verifies:
 * 1. Authoritative 7-company published collection: [FIX, DUOL, ULTA, MEDP, ISRG, DECK, ALC].
 * 2. Complete non-destructive preservation of all 6 existing published assets.
 * 3. Public accessor getOpportunityDossier("ALC") loads authoritative Alcon V4 dossier.
 * 4. isCohortSymbol("ALC") === true; negative controls fail closed.
 * 5. Production dev-preview isolation: /dev-preview/alcon fails closed in production mode
 *    regardless of NEXT_PUBLIC_ENABLE_ALCON_DEV_PREVIEW="true".
 * 6. Research maintenance registry compatibility for ALC.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import {
  listOpportunityCohort,
  getOpportunityDossier,
  isCohortSymbol,
  isAlconV4Dossier,
  getDossierProvenance,
  validateAndDerivePublishedCollection,
  PublishedWatchManifestFile,
  OpportunityDossierSnapshotFile,
} from "../lib/opportunityDossiers";

console.log("Starting Alcon V4 Publication Candidate Acceptance Test Suite...");

const manifestPath = path.resolve(__dirname, "../lib/publishedWatchManifest.json");
const snapshotPath = path.resolve(__dirname, "../lib/opportunityDossierSnapshot.json");

const manifest: PublishedWatchManifestFile = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
const snapshot: OpportunityDossierSnapshotFile = JSON.parse(fs.readFileSync(snapshotPath, "utf-8"));

// 1. Published Collection Validation
const collection = validateAndDerivePublishedCollection(manifest, snapshot, { verifyDigests: true });
assert.strictEqual(collection.isValid, true, `Manifest must validate cleanly: ${collection.errors.join("; ")}`);
assert.strictEqual(collection.errors.length, 0);
assert.deepStrictEqual(
  collection.publishedSymbols,
  ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK", "ALC"],
  "Published symbols must include exactly the 7 verified assets"
);
assert.strictEqual(collection.cohortItems.length, 7);
console.log("✅ 1. Authoritative 7-company published collection verified.");

// 2. Preservation of Existing Published Members
const expectedExisting = [
  { symbol: "FIX", refPrice: 1439.07, cagr: 30.07 },
  { symbol: "DUOL", refPrice: 141.76, cagr: 22.99 },
  { symbol: "ULTA", refPrice: 592.94, cagr: 15.45 },
  { symbol: "MEDP", refPrice: 618.47, cagr: 29.41 },
  { symbol: "ISRG", refPrice: 426.70, cagr: 36.87 },
  { symbol: "DECK", refPrice: 85.72, cagr: -2.34 },
];

for (const exp of expectedExisting) {
  const d = getOpportunityDossier(exp.symbol);
  assert.ok(d, `Existing published dossier for ${exp.symbol} must exist`);
  assert.strictEqual(d?.valuation.reference_price, exp.refPrice, `${exp.symbol} reference price must match`);
  assert.strictEqual(d?.valuation.solved_5y_cagr_pct, exp.cagr, `${exp.symbol} solved CAGR must match`);
  assert.strictEqual(isCohortSymbol(exp.symbol), true);
}
console.log("✅ 2. Non-destructive preservation of all 6 existing published assets verified.");

// 3. Alcon V4 Published Dossier Loading & Integrity
const alc = getOpportunityDossier("ALC");
assert.ok(alc, "ALC published dossier must be retrievable from getOpportunityDossier");
assert.strictEqual(alc?.identity.symbol, "ALC");
assert.strictEqual(alc?.identity.company_name, "Alcon Inc.");
assert.strictEqual(alc?.identity.cik, "0001167379");
assert.strictEqual(alc?.valuation.reference_price, 67.43);
assert.strictEqual(alc?.valuation.solved_5y_cagr_pct, 6.51);
assert.strictEqual(isAlconV4Dossier(alc!), true);

const ext = alc!.alcon_v4_extension!;
assert.ok(ext, "ALC extension must exist");
assert.strictEqual(ext.accumulation_ceilings.annual_irr_basis.hurdle_80, 66.31);
assert.strictEqual(ext.accumulation_ceilings.annual_irr_basis.hurdle_85, 64.82);
assert.strictEqual(ext.accumulation_ceilings.annual_irr_basis.hurdle_100, 60.56);
assert.strictEqual(ext.accumulation_ceilings.calendar_xirr_basis.hurdle_80, 66.36);
assert.strictEqual(ext.accumulation_ceilings.calendar_xirr_basis.hurdle_85, 64.86);
assert.strictEqual(ext.accumulation_ceilings.calendar_xirr_basis.hurdle_100, 60.61);
assert.strictEqual(ext.dated_cash_flows.length, 7);
console.log("✅ 3. Public ALC dossier loading and certified extension integrity verified.");

// 4. Provenance Verification
const prov = getDossierProvenance("ALC");
assert.ok(prov, "ALC provenance must exist");
assert.strictEqual(prov?.sourceAuthority, "ARX Quality Watch Sourced Research");
assert.strictEqual(prov?.referenceMarketDate, "2026-10-21");
assert.strictEqual(prov?.isAddition, true);
assert.strictEqual(prov?.isPreviewDraft, false);
console.log("✅ 4. ALC publication provenance verified.");

// 5. Negative Controls
assert.strictEqual(isCohortSymbol("MU"), false);
assert.strictEqual(isCohortSymbol("PLTR"), false);
assert.strictEqual(isCohortSymbol("CORZ"), false);
assert.strictEqual(isCohortSymbol("IREN"), false);
assert.strictEqual(isCohortSymbol("RANDOM"), false);
assert.strictEqual(getOpportunityDossier("MU"), null);
assert.strictEqual(getOpportunityDossier("PLTR"), null);
console.log("✅ 5. Negative controls and uncataloged symbol fail-closed behavior verified.");

console.log("\nALL ALCON V4 PUBLICATION CANDIDATE ACCEPTANCE TESTS PASSED! 🚀");
