import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

console.log("Starting Opportunity Dossier Snapshot & Research Parity Integrity Suite...");

const EXPECTED_LEDGER_SHA256 = "a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d";
const EXPECTED_COHORT = ["FIX", "DUOL", "IREN", "CORZ", "MU"];

const repoRoot = path.resolve(__dirname, "../..");
const snapshotPath = path.resolve(__dirname, "../lib/opportunityDossierSnapshot.json");
const sourceLedgerPath = path.resolve(repoRoot, "docs/research/opportunity_discovery/odp05_discovery_ledger.json");

// 1. Snapshot existence and JSON parse
assert.strictEqual(fs.existsSync(snapshotPath), true, "Snapshot file must exist in frontend/lib/");
const rawSnapshot = fs.readFileSync(snapshotPath, "utf-8");
const snapshot = JSON.parse(rawSnapshot);
console.log("[OK] Snapshot loaded and parsed successfully");

// 2. Metadata verification
const meta = snapshot.snapshot_metadata;
assert.ok(meta, "Snapshot must have snapshot_metadata");
assert.strictEqual(meta.source_ledger_sha256, EXPECTED_LEDGER_SHA256, "Source ledger SHA-256 in metadata must match accepted baseline");
assert.strictEqual(meta.market_session_date, "2026-10-16", "Market session date must be 2026-10-16");
assert.ok(meta.filing_availability_cutoff_utc, "filing_availability_cutoff_utc must be present");
assert.ok(meta.authoring_timestamp_utc, "authoring_timestamp_utc must be present");
console.log("[OK] Snapshot metadata verified");

// 3. Source Ledger Hash Parity (if source ledger is present in repository)
if (fs.existsSync(sourceLedgerPath)) {
  const sourceRaw = fs.readFileSync(sourceLedgerPath);
  const actualHash = crypto.createHash("sha256").update(sourceRaw).digest("hex");
  assert.strictEqual(actualHash, EXPECTED_LEDGER_SHA256, "Actual source ledger file SHA-256 must match authoritative baseline");
  console.log(`[OK] Source ledger file verified against canonical SHA-256 (${actualHash.substring(0, 16)}...)`);
}

// 4. Cohort Completeness
const dossiers = snapshot.dossiers;
assert.ok(dossiers && typeof dossiers === "object", "Dossiers dictionary must exist");
for (const sym of EXPECTED_COHORT) {
  assert.ok(dossiers[sym], `Cohort symbol ${sym} must exist in dossiers dictionary`);
}
console.log("[OK] All 5 cohort assets present");

// 5. Finding D Parity: Raw Revenue Direct Ledger Sourcing
const corz = dossiers["CORZ"];
assert.strictEqual(corz.financial_profile.revenue, 164201000.0, "CORZ revenue must match ledger q2_2026_revenue 164201000.0 exactly (Finding D)");
assert.notStrictEqual(corz.financial_profile.revenue, 164200000.0, "CORZ must not have rounded/hardcoded 164200000.0 value");

const mu = dossiers["MU"];
assert.strictEqual(mu.financial_profile.revenue, 133188000000.0, "MU revenue must match ledger fy2026_revenue 133188000000.0 exactly (Finding D)");
console.log("[OK] Finding D: Raw revenue parity verified against accepted ledger");

// 6. Finding B Parity: Scenario Assumptions & Reverse-DCF Parameters
for (const sym of ["FIX", "DUOL", "MU"]) {
  const d = dossiers[sym];
  const params = d.valuation.reverse_dcf_parameters;
  assert.ok(params, `${sym} must have reverse_dcf_parameters`);
  assert.strictEqual(typeof params.solved_revenue_cagr_pct, "number", `${sym} must have numeric solved_revenue_cagr_pct`);
  assert.strictEqual(typeof params.operating_margin_pct, "number", `${sym} must have numeric operating_margin_pct`);
  assert.strictEqual(typeof params.wacc_pct, "number", `${sym} must have numeric wacc_pct`);
  assert.strictEqual(typeof params.terminal_growth_pct, "number", `${sym} must have numeric terminal_growth_pct`);
  assert.strictEqual(typeof params.tax_rate_pct, "number", `${sym} must have numeric tax_rate_pct`);
  assert.strictEqual(typeof params.reinvestment_rate_pct, "number", `${sym} must have numeric reinvestment_rate_pct`);
  assert.ok(params.current_shares > 0, `${sym} must have positive current_shares`);
  assert.strictEqual(typeof params.net_debt_bridge, "number", `${sym} must have numeric net_debt_bridge`);

  const scenarios = d.valuation.scenarios;
  assert.ok(scenarios && scenarios.conservative && scenarios.base && scenarios.optimistic, `${sym} must have all 3 scenarios`);
  for (const scenName of ["conservative", "base", "optimistic"]) {
    const s = scenarios[scenName];
    assert.strictEqual(typeof s.revenue_cagr_pct, "number", `${sym} ${scenName} must have revenue_cagr_pct`);
    assert.strictEqual(typeof s.operating_margin_pct, "number", `${sym} ${scenName} must have operating_margin_pct`);
    assert.strictEqual(typeof s.wacc_pct, "number", `${sym} ${scenName} must have wacc_pct`);
    assert.strictEqual(typeof s.terminal_growth_pct, "number", `${sym} ${scenName} must have terminal_growth_pct`);
    assert.strictEqual(typeof s.tax_rate_pct, "number", `${sym} ${scenName} must have tax_rate_pct`);
    assert.strictEqual(typeof s.reinvestment_rate_pct, "number", `${sym} ${scenName} must have reinvestment_rate_pct`);
    assert.strictEqual(typeof s.enterprise_value, "number", `${sym} ${scenName} must have enterprise_value`);
    assert.strictEqual(typeof s.equity_value, "number", `${sym} ${scenName} must have equity_value`);
    assert.strictEqual(typeof s.per_share_value, "number", `${sym} ${scenName} must have per_share_value`);
  }
}

for (const sym of ["IREN", "CORZ"]) {
  const d = dossiers[sym];
  assert.strictEqual(d.valuation.valuation_status, "VALUATION_SUPPRESSED", `${sym} must be VALUATION_SUPPRESSED`);
  assert.strictEqual(d.valuation.reverse_dcf_parameters, null, `${sym} must have null reverse_dcf_parameters`);
  assert.strictEqual(d.valuation.scenarios, null, `${sym} must have null scenarios`);
  assert.ok(d.valuation.resume_conditions.length > 0, `${sym} must have fail-closed resume conditions`);
}
console.log("[OK] Finding B: Scenario assumptions and reverse-DCF solves verified");

// 7. Finding C Parity: Sourced Capital Structure & Net Debt Bridge Breakdown
const duol = dossiers["DUOL"];
assert.strictEqual(duol.financial_profile.is_zero_debt, true, "DUOL must be zero debt");
assert.strictEqual(duol.financial_profile.operating_lease_liabilities_asc842, 86136000.0, "DUOL must report ASC 842 operating lease liabilities");
assert.ok(duol.financial_profile.lease_convention_treatment.includes("operating expense"), "DUOL lease treatment must note operating expense convention");
assert.ok(duol.financial_profile.bridge_note, "DUOL must have bridge note");

const iren = dossiers["IREN"];
assert.strictEqual(iren.financial_profile.restricted_cash, 1723900000.0, "IREN restricted cash must be 1723900000.0");
assert.ok(iren.financial_profile.restriction_treatment.includes("Restricted cash"), "IREN restriction treatment must document exclusion");

assert.strictEqual(corz.financial_profile.digital_assets, 49675000.0, "CORZ digital assets must be 49675000.0");
console.log("[OK] Finding C: Accounting bridge and capital structure disclosures verified");

console.log("ALL OPPORTUNITY DOSSIER SNAPSHOT INTEGRITY TESTS PASSED ✅");
