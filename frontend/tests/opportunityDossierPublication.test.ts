/**
 * frontend/tests/opportunityDossierPublication.test.ts
 *
 * Publication & Manifest Validation Test Suite for ARX Quality Watch Research.
 * Verifies that the manifest strictly controls runtime publication under ARX Closure Model v2:
 *
 * A. Unchanged valid manifest: Selected symbols, lists, lookup, and counts agree.
 * B. Deselect ULTA while retaining its payload: ULTA disappears from published lists and lookup.
 * C. Selected addition with missing payload: Validation fails; no synthetic listing appears.
 * D. Accepted payload with mismatched identity: Validation fails.
 * E. Selected rejected disposition: Validation fails.
 * F. Changed accepted dossier content without corresponding recorded digest: Validation fails.
 * G. Accurate per-dossier provenance: ULTA gets independent provenance; baseline retains baseline provenance.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

import {
  validateAndDerivePublishedCollection,
  canonicalJsonStringify,
  computeCanonicalDossierDigest,
  getDossierProvenance,
  getCollectionProvenance,
  listOpportunityCohort,
  getOpportunityDossier,
  isCohortSymbol,
  PublishedWatchManifestFile,
  OpportunityDossierSnapshotFile,
} from "../lib/opportunityDossiers";

console.log("Starting Opportunity Dossier Publication & Manifest Authority Test Suite...");

const manifestPath = path.resolve(__dirname, "../lib/publishedWatchManifest.json");
const snapshotPath = path.resolve(__dirname, "../lib/opportunityDossierSnapshot.json");

assert.strictEqual(fs.existsSync(manifestPath), true, "publishedWatchManifest.json must exist");
assert.strictEqual(fs.existsSync(snapshotPath), true, "opportunityDossierSnapshot.json must exist");

const validManifest: PublishedWatchManifestFile = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
const validSnapshot: OpportunityDossierSnapshotFile = JSON.parse(fs.readFileSync(snapshotPath, "utf-8"));

function clone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

// =========================================================================
// Test A: Unchanged valid control manifest
// =========================================================================
console.log("Running Test A: Unchanged valid control manifest...");
const resultA = validateAndDerivePublishedCollection(validManifest, validSnapshot, { verifyDigests: true });
assert.strictEqual(resultA.isValid, true, `Control manifest must be valid, got errors: ${resultA.errors.join("; ")}`);
assert.strictEqual(resultA.errors.length, 0);
assert.deepStrictEqual(resultA.publishedSymbols, ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"], "Published symbols must match [FIX, DUOL, ULTA, MEDP, ISRG, DECK]");
assert.strictEqual(resultA.cohortItems.length, 6, "Cohort items length must be 6");
assert.deepStrictEqual(
  resultA.cohortItems.map((c) => c.symbol),
  ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"]
);
assert.ok(resultA.dossiers.FIX, "FIX dossier must exist");
assert.ok(resultA.dossiers.DUOL, "DUOL dossier must exist");
assert.ok(resultA.dossiers.ULTA, "ULTA dossier must exist");
assert.ok(resultA.dossiers.MEDP, "MEDP dossier must exist");
assert.ok(resultA.dossiers.ISRG, "ISRG dossier must exist");
assert.ok(resultA.dossiers.DECK, "DECK dossier must exist");
assert.strictEqual(resultA.dossiers.ULTA.valuation.reference_price, 592.94);
assert.strictEqual(resultA.dossiers.ULTA.valuation.solved_5y_cagr_pct, 15.45);
assert.strictEqual(resultA.dossiers.MEDP.valuation.reference_price, 618.47);
assert.strictEqual(resultA.dossiers.MEDP.valuation.solved_5y_cagr_pct, 29.41);
assert.strictEqual(resultA.dossiers.ISRG.valuation.reference_price, 426.70);
assert.strictEqual(resultA.dossiers.ISRG.valuation.solved_5y_cagr_pct, 36.87);
assert.strictEqual(resultA.dossiers.DECK.valuation.reference_price, 85.72);
assert.strictEqual(resultA.dossiers.DECK.valuation.solved_5y_cagr_pct, -2.34);

// Verify live accessors reflect Test A
assert.strictEqual(listOpportunityCohort().length, 6);
assert.strictEqual(isCohortSymbol("ULTA"), true);
assert.strictEqual(isCohortSymbol("FIX"), true);
assert.strictEqual(isCohortSymbol("DUOL"), true);
assert.strictEqual(isCohortSymbol("MEDP"), true);
assert.strictEqual(isCohortSymbol("ISRG"), true);
assert.strictEqual(isCohortSymbol("DECK"), true);
assert.strictEqual(isCohortSymbol("IREN"), false); // Held
assert.strictEqual(isCohortSymbol("CORZ"), false); // Held
assert.strictEqual(isCohortSymbol("MU"), false);   // Excluded
assert.strictEqual(isCohortSymbol("PLTR"), false); // Excluded
console.log("✅ Test A Passed: Unchanged valid manifest agreed across symbols, lists, lookups, and counts.");

// =========================================================================
// Test B: Deselect ULTA while retaining its payload
// =========================================================================
console.log("Running Test B: Deselect ULTA while retaining its payload in accepted_additions...");
const manifestB = clone(validManifest);
// Remove ULTA from published_symbols but keep its payload in accepted_additions
manifestB.published_symbols = ["FIX", "DUOL"];
manifestB.published_count = 2;
assert.ok(manifestB.accepted_additions.ULTA, "Control check: ULTA payload remains present");

const resultB = validateAndDerivePublishedCollection(manifestB, validSnapshot, { verifyDigests: true });
assert.strictEqual(resultB.isValid, true, "Validation must succeed when ULTA is cleanly deselected");
assert.deepStrictEqual(resultB.publishedSymbols, ["FIX", "DUOL"]);
assert.strictEqual(resultB.cohortItems.length, 2);
assert.strictEqual(resultB.cohortItems.some((c) => c.symbol === "ULTA"), false, "ULTA must NOT appear in cohortItems");
assert.strictEqual(resultB.dossiers.ULTA, undefined, "ULTA must NOT be retrievable in published dossiers");
console.log("✅ Test B Passed: Deselecting ULTA removes it from published lists and lookup despite retaining payload.");

// =========================================================================
// Test C: Selected addition with missing payload
// =========================================================================
console.log("Running Test C: Selected addition with missing payload...");
const manifestC = clone(validManifest);
// Keep ULTA selected, but delete its payload from accepted_additions
delete manifestC.accepted_additions.ULTA;

const resultC = validateAndDerivePublishedCollection(manifestC, validSnapshot);
assert.strictEqual(resultC.isValid, false, "Validation must fail when selected addition lacks payload");
assert.ok(
  resultC.errors.some((e) => e.includes("ULTA") && e.includes("missing payload")),
  `Error must identify missing payload for ULTA, got: ${resultC.errors.join("; ")}`
);
assert.strictEqual(resultC.cohortItems.some((c) => c.symbol === "ULTA"), false, "No synthetic ULTA listing can appear");
assert.strictEqual(resultC.dossiers.ULTA, undefined);
console.log("✅ Test C Passed: Missing addition payload fails validation with zero synthetic fallback.");

// =========================================================================
// Test D: Accepted payload with mismatched identity
// =========================================================================
console.log("Running Test D: Accepted payload with mismatched identity...");
const manifestD = clone(validManifest);
manifestD.accepted_additions.ULTA.identity.symbol = "MISMATCHED";

const resultD = validateAndDerivePublishedCollection(manifestD, validSnapshot);
assert.strictEqual(resultD.isValid, false, "Validation must fail on identity mismatch");
assert.ok(
  resultD.errors.some((e) => e.includes("mismatched identity")),
  `Error must identify mismatched identity, got: ${resultD.errors.join("; ")}`
);
assert.strictEqual(resultD.publishedSymbols.includes("ULTA"), false);
assert.strictEqual(resultD.cohortItems.some((c) => c.symbol === "ULTA"), false);
console.log("✅ Test D Passed: Mismatched identity fails validation.");

// =========================================================================
// Test E: Selected rejected disposition
// =========================================================================
console.log("Running Test E: Selected rejected disposition...");
const manifestE = clone(validManifest);
manifestE.accepted_additions.ULTA.central_assessment.disposition = "NO_SUPPORTED_OPPORTUNITY";

const resultE = validateAndDerivePublishedCollection(manifestE, validSnapshot);
assert.strictEqual(resultE.isValid, false, "Validation must fail if disposition is not WATCH_CONDITIONS");
assert.ok(
  resultE.errors.some((e) => e.includes("WATCH_CONDITIONS")),
  `Error must identify invalid disposition, got: ${resultE.errors.join("; ")}`
);
assert.strictEqual(resultE.publishedSymbols.includes("ULTA"), false);
assert.strictEqual(resultE.cohortItems.some((c) => c.symbol === "ULTA"), false);
console.log("✅ Test E Passed: Selected non-WATCH disposition fails validation.");

// =========================================================================
// Test F: Changed accepted dossier content without matching content_sha256
// =========================================================================
console.log("Running Test F: Tampered dossier content without updating recorded content_sha256...");
const manifestF = clone(validManifest);
// Mutate financial profile value without updating additions_provenance.ULTA.content_sha256
manifestF.accepted_additions.ULTA.financial_profile.revenue = 999999999999.0;

const resultF = validateAndDerivePublishedCollection(manifestF, validSnapshot, { verifyDigests: true });
assert.strictEqual(resultF.isValid, false, "Validation must fail when content digest does not match");
assert.ok(
  resultF.errors.some((e) => e.includes("content digest mismatch")),
  `Error must identify content digest mismatch, got: ${resultF.errors.join("; ")}`
);
assert.strictEqual(resultF.publishedSymbols.includes("ULTA"), false);
console.log("✅ Test F Passed: Content tampering without recorded digest update fails validation.");

// =========================================================================
// Test G: Accurate Per-Dossier Provenance
// =========================================================================
console.log("Running Test G: Per-dossier provenance accuracy...");
// 1. ULTA (Scaling Addition)
const provUlta = getDossierProvenance("ULTA");
assert.ok(provUlta, "getDossierProvenance(ULTA) must return provenance");
assert.strictEqual(provUlta.sourceAuthority, "ARX Quality Watch Scaling Addition");
assert.strictEqual(provUlta.authoringTimestampUtc, "2026-10-21T16:00:00Z");
assert.strictEqual(provUlta.filingCutoffUtc, "2026-10-16T20:00:00Z");
assert.strictEqual(provUlta.lifecycleStatus, "CANDIDATE_SELECTED");
assert.strictEqual(provUlta.reviewStatus, "RESEARCH_EVIDENCE_ACCEPTED");
assert.strictEqual(provUlta.isAddition, true);
assert.ok(!provUlta.contentDigest.includes("a65be3b"), "ULTA must NOT carry baseline ledger hash");

// 1b. MEDP (Scaling Addition)
const provMedp = getDossierProvenance("MEDP");
assert.ok(provMedp, "getDossierProvenance(MEDP) must return provenance");
assert.strictEqual(provMedp.sourceAuthority, "ARX Quality Watch Scaling Addition");
assert.strictEqual(provMedp.lifecycleStatus, "CANDIDATE_SELECTED");
assert.strictEqual(provMedp.reviewStatus, "RESEARCH_EVIDENCE_ACCEPTED");
assert.strictEqual(provMedp.isAddition, true);
assert.ok(!provMedp.contentDigest.includes("a65be3b"), "MEDP must NOT carry baseline ledger hash");

// 1c. ISRG (Scaling Addition)
const provIsrg = getDossierProvenance("ISRG");
assert.ok(provIsrg, "getDossierProvenance(ISRG) must return provenance");
assert.strictEqual(provIsrg.sourceAuthority, "ARX Quality Watch Scaling Addition");
assert.strictEqual(provIsrg.lifecycleStatus, "CANDIDATE_SELECTED");
assert.strictEqual(provIsrg.reviewStatus, "RESEARCH_EVIDENCE_ACCEPTED");
assert.strictEqual(provIsrg.isAddition, true);
assert.ok(!provIsrg.contentDigest.includes("a65be3b"), "ISRG must NOT carry baseline ledger hash");

// 1d. DECK (Scaling Addition)
const provDeck = getDossierProvenance("DECK");
assert.ok(provDeck, "getDossierProvenance(DECK) must return provenance");
assert.strictEqual(provDeck.sourceAuthority, "ARX Quality Watch Scaling Addition");
assert.strictEqual(provDeck.lifecycleStatus, "CANDIDATE_SELECTED");
assert.strictEqual(provDeck.reviewStatus, "RESEARCH_EVIDENCE_ACCEPTED");
assert.strictEqual(provDeck.isAddition, true);
assert.ok(!provDeck.contentDigest.includes("a65be3b"), "DECK must NOT carry baseline ledger hash");

// 2. FIX (Historical Baseline)
const provFix = getDossierProvenance("FIX");
assert.ok(provFix, "getDossierProvenance(FIX) must return provenance");
assert.strictEqual(provFix.sourceAuthority, "ODP-05 Historical Research Baseline");
assert.strictEqual(provFix.authoringTimestampUtc, "2026-10-20T14:30:00Z");
assert.strictEqual(provFix.governingContract, "CONTRACT-ARX-ODP-V1-001-REV2");
assert.strictEqual(provFix.lifecycleStatus, "HISTORICAL_BASELINE_WATCH");
assert.strictEqual(provFix.reviewStatus, "BASELINE_RESEARCH_ACCEPTED");
assert.strictEqual(provFix.isAddition, false);
assert.ok(provFix.contentDigest.includes("a65be3b38cdf"), "FIX must carry baseline ledger hash");

// 3. Collection-Level Provenance for Unavailable / Excluded symbols
const colProvMu = getCollectionProvenance("MU");
assert.strictEqual(colProvMu.governingStandard, "ARX-QUALITY-WATCH-STANDARD-V1");
assert.strictEqual(colProvMu.publishedCount, 6);
assert.ok(colProvMu.disclaimer.includes('No published watch dossier exists for "MU"'));
assert.ok(!JSON.stringify(colProvMu).includes("a65be3b"), "Collection provenance must not leak baseline hash for unavailable records");

console.log("✅ Test G Passed: Per-dossier provenance accurately distinguishes additions, baselines, and unavailable fallbacks.");

console.log("\nALL OPPORTUNITY DOSSIER PUBLICATION & MANIFEST TESTS PASSED (7/7) ✅");
