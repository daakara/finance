/**
 * frontend/scripts/verify-dossier-integrity.ts
 *
 * Pre-build and release integrity gate for Opportunity Dossier snapshot.
 * Enforces cryptographic digest matching, schema invariants, and research parity
 * directly in frontend CI and build pipelines without Python runtime dependency.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

const EXPECTED_SOURCE_LEDGER_SHA256 = "a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d";
const EXPECTED_COHORT = ["FIX", "DUOL", "IREN", "CORZ", "MU"];

function computeSha256(filePath: string): string {
  const content = fs.readFileSync(filePath);
  return crypto.createHash("sha256").update(content).digest("hex");
}

function verifySnapshotIntegrity() {
  console.log("🔍 [INTEGRITY GATE] Verifying Opportunity Dossier Snapshot...");

  const repoRoot = path.resolve(__dirname, "../..");
  const snapshotPath = path.resolve(__dirname, "../lib/opportunityDossierSnapshot.json");
  const sourceLedgerPath = path.resolve(repoRoot, "docs/research/opportunity_discovery/odp05_discovery_ledger.json");

  // 1. Verify snapshot file existence
  if (!fs.existsSync(snapshotPath)) {
    console.error(`❌ FATAL: Opportunity Dossier Snapshot not found at: ${snapshotPath}`);
    process.exit(1);
  }

  // 2. Read and parse snapshot
  let snapshot: any;
  try {
    const raw = fs.readFileSync(snapshotPath, "utf-8");
    snapshot = JSON.parse(raw);
  } catch (err: any) {
    console.error(`❌ FATAL: Failed to parse snapshot JSON: ${err.message}`);
    process.exit(1);
  }

  // 3. Verify metadata
  const meta = snapshot.snapshot_metadata;
  if (!meta) {
    console.error("❌ FATAL: snapshot_metadata missing from snapshot");
    process.exit(1);
  }

  if (meta.source_ledger_sha256 !== EXPECTED_SOURCE_LEDGER_SHA256) {
    console.error(`❌ SHA-256 MISMATCH in metadata: expected ${EXPECTED_SOURCE_LEDGER_SHA256}, got ${meta.source_ledger_sha256}`);
    process.exit(1);
  }

  // If source ledger exists locally, verify its actual hash
  if (fs.existsSync(sourceLedgerPath)) {
    const actualSourceSha = computeSha256(sourceLedgerPath);
    if (actualSourceSha !== EXPECTED_SOURCE_LEDGER_SHA256) {
      console.error(`❌ SOURCE LEDGER TAMPERED: actual hash ${actualSourceSha} != expected ${EXPECTED_SOURCE_LEDGER_SHA256}`);
      process.exit(1);
    }
    console.log(`✅ Source ledger hash verified: ${actualSourceSha.substring(0, 16)}...`);
  }

  // 4. Verify dossiers and cohort completeness
  const dossiers = snapshot.dossiers;
  if (!dossiers || typeof dossiers !== "object") {
    console.error("❌ FATAL: dossiers object missing or invalid");
    process.exit(1);
  }

  for (const sym of EXPECTED_COHORT) {
    if (!dossiers[sym]) {
      console.error(`❌ MISSING COHORT ASSET: Symbol ${sym} not found in snapshot dossiers`);
      process.exit(1);
    }
  }

  // 5. Invariant Checks for Each Symbol
  // FIX: computed verified
  const fix = dossiers.FIX;
  if (fix.valuation.valuation_status !== "COMPUTED_VERIFIED" || fix.valuation.solved_5y_cagr_pct !== 30.07) {
    console.error("❌ FIX valuation invariant failure");
    process.exit(1);
  }
  if (!fix.valuation.reverse_dcf_parameters || fix.valuation.reverse_dcf_parameters.solved_revenue_cagr_pct !== 30.07) {
    console.error("❌ FIX reverse_dcf_parameters missing or invalid");
    process.exit(1);
  }

  // DUOL: zero debt invariants
  const duol = dossiers.DUOL;
  if (!duol.financial_profile.is_zero_debt || duol.financial_profile.cash_and_equivalents === null) {
    console.error("❌ DUOL zero-debt invariant failure");
    process.exit(1);
  }
  if (!duol.financial_profile.bridge_note || !duol.financial_profile.lease_convention_treatment) {
    console.error("❌ DUOL bridge notes missing");
    process.exit(1);
  }

  // IREN: valuation suppressed + restricted cash
  const iren = dossiers.IREN;
  if (iren.valuation.valuation_status !== "VALUATION_SUPPRESSED" || iren.valuation.solved_5y_cagr_pct !== null) {
    console.error("❌ IREN valuation suppression invariant failure");
    process.exit(1);
  }
  if (iren.financial_profile.restricted_cash !== 1723900000.0) {
    console.error(`❌ IREN restricted cash mismatch: ${iren.financial_profile.restricted_cash} != 1723900000.0`);
    process.exit(1);
  }

  // CORZ: valuation suppressed + raw revenue parity check (Finding D)
  const corz = dossiers.CORZ;
  if (corz.valuation.valuation_status !== "VALUATION_SUPPRESSED") {
    console.error("❌ CORZ valuation suppression invariant failure");
    process.exit(1);
  }
  if (corz.financial_profile.revenue !== 164201000.0) {
    console.error(`❌ CORZ revenue mismatch: ${corz.financial_profile.revenue} != 164201000.0`);
    process.exit(1);
  }
  if (corz.financial_profile.digital_assets !== 49675000.0) {
    console.error(`❌ CORZ digital_assets mismatch: ${corz.financial_profile.digital_assets} != 49675000.0`);
    process.exit(1);
  }

  // MU: qualified valuation + raw revenue parity check (Finding D)
  const mu = dossiers.MU;
  if (mu.valuation.valuation_status !== "COMPUTED_QUALIFIED") {
    console.error("❌ MU valuation qualification invariant failure");
    process.exit(1);
  }
  if (mu.financial_profile.revenue !== 133188000000.0) {
    console.error(`❌ MU revenue mismatch: ${mu.financial_profile.revenue} != 133188000000.0`);
    process.exit(1);
  }
  if (!mu.valuation.reverse_dcf_parameters || mu.valuation.reverse_dcf_parameters.solved_revenue_cagr_pct !== 21.29) {
    console.error("❌ MU reverse_dcf_parameters missing or invalid");
    process.exit(1);
  }

  console.log("✅ [INTEGRITY GATE PASSED] All Opportunity Dossier snapshot invariants verified successfully.");
}

verifySnapshotIntegrity();
