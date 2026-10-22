"""
tests/test_prepare_opportunity_dossier.py

Focused Invariant, Integrity, and Negative Test Suite for Opportunity Dossier Scaling Pilot.
Verifies:
1. Frozen ODP-05 ledger baseline preservation (exact SHA-256 digest match).
2. Preparation of the full 20-company pilot cohort (5 baseline + 15 additions).
3. Deterministic Reverse-DCF solving and roundtrip forward model reconciliation (< $0.05).
4. Evidence provenance integrity (valid SEC EDGAR accessions, primary URLs, reporting dates).
5. Capital structure accounting: documented zero debt vs funded debt.
6. Lease-as-operating-expense convention (ASC 842) for retail brands.
7. Explicit valuation suppression (IREN, CORZ) and qualification (MU, PLTR, KLAC, SNPS).
8. Absence of manufactured reviewer identities or review timestamps; honest review metadata.
9. Verification routine passes across authoritative pilot artifacts.
10. Representative candidate (ULTA) complete source-to-input evidence tracing.
11. Negative: Tampered financial inputs fail validation.
12. Negative: Tampered narrative claims fail validation.
13. Negative: Unauthorized review status upgrade fails validation.
14. Negative: Manifest ledger hash mismatch fails validation.
15. Negative: Missing evidence mapping on representative dossier fails validation.
16. Negative: Output path targeting production snapshot fails fail-closed boundary.
"""

import copy
import hashlib
import json
import shutil
import pytest
from pathlib import Path

from scripts.research.prepare_opportunity_dossier import (
    REPO_ROOT,
    FROZEN_ODP05_LEDGER_PATH,
    FROZEN_ODP05_SHA256,
    BASELINE_COHORT,
    PILOT_ADDITIONS_COHORT,
    TOTAL_PILOT_COHORT,
    PUBLISHED_WATCH_SYMBOLS,
    BASELINE_WATCH_COHORT,
    EXCLUDED_HISTORICAL_SYMBOLS,
    EXCLUDED_DRAFT_SYMBOLS,
    DEFAULT_PILOT_LEDGER_PATH,
    DEFAULT_PILOT_MANIFEST_PATH,
    DEFAULT_FRONTEND_DRAFT_PATH,
    DEFAULT_DOC_MATRIX_PATH,
    DEFAULT_PUBLISHED_MANIFEST_PATH,
    DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH,
    prepare_pilot_scaling_corpus,
    validate_existing_pilot_artifacts,
    PILOT_SPECS,
)
from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    solve_reverse_dcf,
)


@pytest.fixture
def valid_pilot_artifacts(tmp_path):
    """
    Creates a valid, isolated sandbox containing unmodified pilot artifacts
    with exact matching hashes and documentation.
    """
    ledger_path = tmp_path / "pilot_scaling_ledger.json"
    manifest_path = tmp_path / "pilot_scaling_manifest.json"
    frontend_path = tmp_path / "opportunityDossierPilotDrafts.json"
    doc_path = tmp_path / "pilot_scaling_cohort_matrix.md"
    pub_manifest_path = tmp_path / "published_watch_manifest.json"
    fe_pub_manifest_path = tmp_path / "publishedWatchManifest.json"

    shutil.copyfile(DEFAULT_PILOT_LEDGER_PATH, ledger_path)
    shutil.copyfile(DEFAULT_PILOT_MANIFEST_PATH, manifest_path)
    shutil.copyfile(DEFAULT_FRONTEND_DRAFT_PATH, frontend_path)
    shutil.copyfile(DEFAULT_DOC_MATRIX_PATH, doc_path)
    shutil.copyfile(DEFAULT_PUBLISHED_MANIFEST_PATH, pub_manifest_path)
    shutil.copyfile(DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH, fe_pub_manifest_path)

    return {
        "ledger_path": ledger_path,
        "manifest_path": manifest_path,
        "frontend_draft_path": frontend_path,
        "doc_matrix_path": doc_path,
        "published_manifest_path": pub_manifest_path,
        "frontend_published_manifest_path": fe_pub_manifest_path,
    }


def test_1_frozen_odp05_baseline_unmodified():
    """Verify that ODP-05 canonical ledger is untouched and matches accepted SHA-256 digest."""
    assert FROZEN_ODP05_LEDGER_PATH.exists(), f"ODP-05 ledger missing at {FROZEN_ODP05_LEDGER_PATH}"
    with open(FROZEN_ODP05_LEDGER_PATH, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == FROZEN_ODP05_SHA256, f"ODP-05 ledger tampered: {actual_sha} != {FROZEN_ODP05_SHA256}"


def test_2_cohort_completeness_and_sizing():
    """Verify that the scaling pilot cohort defines exactly 19 companies (5 baseline + 14 additions, with PLTR excluded per product owner direction)."""
    assert len(BASELINE_COHORT) == 5
    assert len(PILOT_ADDITIONS_COHORT) == 14
    assert len(TOTAL_PILOT_COHORT) == 19
    assert len(set(TOTAL_PILOT_COHORT)) == 19, "Cohort symbols must be unique"
    for sym in PILOT_ADDITIONS_COHORT:
        assert sym in PILOT_SPECS, f"Addition {sym} missing from PILOT_SPECS"


def test_3_reverse_dcf_roundtrip_verification():
    """Verify that every numeric valuation in the 15 additions solves and roundtrips within canonical tolerance (max(0.05, ref_price * 0.0005))."""
    for sym, spec in PILOT_SPECS.items():
        val_status = spec.get("valuation_status")
        if val_status == "VALUATION_SUPPRESSED":
            continue

        ref_price = spec["reference_price"]
        fin = spec["financial_profile"]
        val_spec = spec["valuation_spec"]

        solved_cagr, status = solve_reverse_dcf(
            target_price=ref_price,
            base_revenue=fin["revenue"],
            current_shares=fin["diluted_shares"],
            net_debt_bridge=fin["net_debt_bridge"],
            operating_margin_pct=val_spec["margin"],
            wacc_pct=val_spec["wacc"],
            terminal_growth_pct=val_spec["terminal_g"],
            tax_rate_pct=val_spec["tax_rate"],
            reinvestment_rate_pct=val_spec["reinvestment_rate"],
            low_bound_pct=-40.0,
            high_bound_pct=100.0,
        )

        assert solved_cagr is not None, f"Failed to solve reverse DCF for {sym}: {status}"
        assert "VERIFIED" in status, f"Roundtrip status not verified for {sym}: {status}"

        # Explicit roundtrip check
        roundtrip = run_fcff_model(
            base_revenue=fin["revenue"],
            current_shares=fin["diluted_shares"],
            net_debt_bridge=fin["net_debt_bridge"],
            growth_5y_pct=solved_cagr,
            operating_margin_pct=val_spec["margin"],
            wacc_pct=val_spec["wacc"],
            terminal_growth_pct=val_spec["terminal_g"],
            tax_rate_pct=val_spec["tax_rate"],
            reinvestment_rate_pct=val_spec["reinvestment_rate"],
        )
        diff = abs(roundtrip["per_share"] - ref_price)
        threshold = max(0.05, ref_price * 0.0005)
        assert diff <= threshold, f"Roundtrip diff ${diff:.3f} exceeds threshold ${threshold:.3f} for {sym}"


def test_4_evidence_provenance_and_sec_filings():
    """Verify that every company dossier has valid primary SEC filings with accession numbers and direct URLs."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    for sym, d in dossiers.items():
        filings = d["financial_profile"]["primary_filings"]
        assert len(filings) >= 1, f"No filings for {sym}"
        for f in filings:
            accn = f["accession_number"]
            assert len(accn) == 20 and accn.count("-") == 2, f"Malformed accession for {sym}: {accn}"
            assert f["url"].startswith("https://www.sec.gov/Archives/edgar/data/"), f"Invalid EDGAR URL for {sym}: {f['url']}"
            assert f["period_end"], f"Missing period_end for {sym}"
            assert f["form"] in ["10-K", "10-Q", "6-K", "8-K"], f"Invalid form for {sym}: {f['form']}"


def test_5_capital_structure_and_zero_debt_mechanics():
    """Verify that documented zero-debt companies are clearly separated from debt-bearing companies."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    # Documented zero-debt companies
    zero_debt_symbols = ["DUOL", "MEDP", "DECK", "LULU", "EME", "ANET", "ISRG", "CDNS"]
    for sym in zero_debt_symbols:
        d = dossiers[sym]
        fp = d["financial_profile"]
        assert fp["is_zero_debt"] is True, f"{sym} must be marked is_zero_debt=True"
        assert fp["total_financial_debt"] == 0.0, f"{sym} total_financial_debt must be 0.0"
        assert fp["zero_debt_support"] is not None, f"{sym} must have documented zero_debt_support"

    # Debt-bearing companies
    debt_symbols = ["FIX", "ULTA", "KLAC", "VRT", "PWR", "SNPS", "NOW", "ELF", "IREN", "CORZ", "MU"]
    for sym in debt_symbols:
        d = dossiers[sym]
        fp = d["financial_profile"]
        assert fp["is_zero_debt"] is False, f"{sym} must be marked is_zero_debt=False"
        assert fp["total_financial_debt"] > 0.0, f"{sym} total_financial_debt must be strictly positive"


def test_6_retail_lease_convention_asc842():
    """Verify that retail brands apply Lease-as-Operating-Expense convention under ASC 842."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    for sym in ["ULTA", "LULU", "DECK"]:
        d = dossiers[sym]
        fp = d["financial_profile"]
        assert fp.get("operating_lease_liabilities_asc842") is not None, f"{sym} must report ASC 842 lease liabilities"
        assert fp.get("lease_convention_treatment") is not None, f"{sym} must document lease convention treatment"
        assert "operating expense" in fp["lease_convention_treatment"].lower(), f"{sym} must apply operating expense convention"


def test_7_valuation_suppression_and_qualification_reasons():
    """Verify that valuation suppression and qualification statuses have justified, explicit notices."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    # Suppressed
    for sym in ["IREN", "CORZ"]:
        d = dossiers[sym]
        v = d["valuation"]
        assert v["valuation_status"] == "VALUATION_SUPPRESSED"
        assert v["suppression_reason"] is not None
        assert len(v["resume_conditions"]) >= 1

    # Qualified
    for sym in ["MU", "KLAC", "SNPS"]:
        d = dossiers[sym]
        v = d["valuation"]
        assert v["valuation_status"] == "COMPUTED_QUALIFIED"
        assert v["qualification_notice"] is not None


def test_8_honest_review_and_publication_metadata():
    """Verify that review statuses are honest: baseline unrecorded, draft pending review, zero invented sign-offs."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    for sym in BASELINE_COHORT:
        meta = dossiers[sym]["publication_metadata"]
        assert meta["lifecycle_status"] == "PUBLISHED_PRODUCTION"
        assert meta["review_status"] == "RESEARCH_REVIEW_UNRECORDED"
        assert meta["reviewer_id"] is None
        assert meta["reviewed_at_utc"] is None

    for sym in PILOT_ADDITIONS_COHORT:
        meta = dossiers[sym]["publication_metadata"]
        assert meta["lifecycle_status"] == "DRAFT_STAGING"
        assert meta["review_status"] == "PENDING_RESEARCH_REVIEW"
        assert meta["reviewer_id"] is None
        assert meta["reviewed_at_utc"] is None


def test_9_pilot_scaling_artifacts_validation():
    """Verify that the read-only validation routine passes across generated pilot artifacts."""
    assert validate_existing_pilot_artifacts() is True


def test_10_representative_dossier_ulta_evidence_tracing():
    """Verify that ULTA representative dossier contains complete, granular source-to-input tracing."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    ulta = ledger["dossiers"]["ULTA"]
    tracing = ulta["financial_profile"].get("source_evidence_tracing")
    assert tracing is not None, "ULTA missing source_evidence_tracing"

    # 1. Company identity
    assert tracing["company_identity"]["fact_status"] == "VERIFIED_REPORTED_FACT"
    assert tracing["company_identity"]["cik"] == "0001403568"

    # 2. Revenue derivation
    rev = tracing["revenue"]
    assert rev["fact_status"] == "DERIVED_CALCULATION"
    assert rev["reported_ttm_value_usd"] == 12955517000.0
    assert len(rev["filing_sources"]) == 2

    # 3. Operating income derivation
    op = tracing["operating_income"]
    assert op["fact_status"] == "DERIVED_CALCULATION"
    assert op["reported_ttm_value_usd"] == 1614258000.0

    # 4. Cash and Note 7 funded debt
    assert tracing["cash_and_investments"]["fact_status"] == "VERIFIED_REPORTED_FACT"
    assert tracing["cash_and_investments"]["total_cash_and_investments_usd"] == 213451000.0
    assert tracing["funded_debt"]["fact_status"] == "VERIFIED_REPORTED_FACT"
    assert tracing["funded_debt"]["total_financial_debt_usd"] == 339578000.0
    assert tracing["funded_debt"]["is_zero_debt"] is False
    assert ulta["financial_profile"]["total_financial_debt"] == 339578000.0
    assert ulta["financial_profile"]["is_zero_debt"] is False
    assert ulta["financial_profile"]["net_debt_bridge"] == 126127000.0

    # 5. ASC 842 Lease treatment
    assert tracing["lease_treatment"]["fact_status"] == "VERIFIED_REPORTED_FACT"
    assert tracing["lease_treatment"]["accounting_standard"] == "ASC 842"
    assert tracing["lease_treatment"]["total_lease_liabilities_usd"] == 2184453000.0

    # 6. Diluted share count
    assert tracing["shares_outstanding"]["fact_status"] == "VERIFIED_REPORTED_FACT"
    assert tracing["shares_outstanding"]["share_count"] == 43062000.0
    assert tracing["shares_outstanding"]["measurement_period"] == "Three months ended 2026-08-01"
    assert ulta["financial_profile"]["selected_share_basis"]["measurement_period"] == "Three months ended 2026-08-01"

    # 7. Reference price
    assert tracing["reference_price"]["fact_status"] == "VERIFIED_REPORTED_FACT"
    assert tracing["reference_price"]["reference_price"] == 592.94

    # 8. Solved CAGR & roundtrip
    assert tracing["valuation_model"]["fact_status"] == "DERIVED_CALCULATION"
    assert tracing["valuation_model"]["solved_5y_revenue_cagr_pct"] == 15.45
    assert tracing["valuation_model"]["roundtrip_value"] == 593.00
    assert ulta["valuation"]["solved_5y_cagr_pct"] == 15.45

    # 9. Narrative claims
    assert len(tracing["narrative_claims"]) >= 3


def test_11_negative_financial_input_tampering(tmp_path):
    """Negative test: Tampering with financial metrics on disk fails validation."""
    with open(DEFAULT_PILOT_LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    with open(DEFAULT_PILOT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(DEFAULT_FRONTEND_DRAFT_PATH, "r", encoding="utf-8") as f:
        frontend = json.load(f)

    # Tamper with ULTA revenue in ledger
    tampered_ledger = copy.deepcopy(ledger)
    tampered_ledger["dossiers"]["ULTA"]["financial_profile"]["revenue"] = 9999999999.0

    t_ledger_path = tmp_path / "pilot_scaling_ledger.json"
    t_manifest_path = tmp_path / "pilot_scaling_manifest.json"
    t_fe_path = tmp_path / "opportunityDossierPilotDrafts.json"

    with open(t_ledger_path, "w", encoding="utf-8") as f:
        json.dump(tampered_ledger, f)
    with open(t_fe_path, "w", encoding="utf-8") as f:
        json.dump(frontend, f)

    # Compute hash of tampered file so manifest hash passes, forcing in-memory check to catch it
    with open(t_ledger_path, "rb") as f:
        t_sha = hashlib.sha256(f.read()).hexdigest()
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["pilot_ledger_sha256"] = t_sha

    with open(t_manifest_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f)

    # Validation must detect financial profile tampering
    result = validate_existing_pilot_artifacts(
        ledger_path=t_ledger_path,
        manifest_path=t_manifest_path,
        frontend_draft_path=t_fe_path,
    )
    assert result is False, "Validation must fail when financial inputs are tampered"


def test_12_negative_narrative_tampering(tmp_path):
    """Negative test: Tampering with narrative thesis on disk fails validation."""
    with open(DEFAULT_PILOT_LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    with open(DEFAULT_PILOT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(DEFAULT_FRONTEND_DRAFT_PATH, "r", encoding="utf-8") as f:
        frontend = json.load(f)

    tampered_ledger = copy.deepcopy(ledger)
    tampered_ledger["dossiers"]["ULTA"]["central_assessment"]["presentation_summary"] = "Tampered summary."

    t_ledger_path = tmp_path / "pilot_scaling_ledger.json"
    t_manifest_path = tmp_path / "pilot_scaling_manifest.json"
    t_fe_path = tmp_path / "opportunityDossierPilotDrafts.json"

    with open(t_ledger_path, "w", encoding="utf-8") as f:
        json.dump(tampered_ledger, f)
    with open(t_fe_path, "w", encoding="utf-8") as f:
        json.dump(frontend, f)

    with open(t_ledger_path, "rb") as f:
        t_sha = hashlib.sha256(f.read()).hexdigest()
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["pilot_ledger_sha256"] = t_sha

    with open(t_manifest_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f)

    result = validate_existing_pilot_artifacts(
        ledger_path=t_ledger_path,
        manifest_path=t_manifest_path,
        frontend_draft_path=t_fe_path,
    )
    assert result is False, "Validation must fail when narrative thesis is tampered"


def test_13_negative_unauthorized_review_status_upgrade(tmp_path):
    """Negative test: Unauthorized upgrade of draft to RESEARCH_REVIEWED fails validation."""
    with open(DEFAULT_PILOT_LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    with open(DEFAULT_PILOT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(DEFAULT_FRONTEND_DRAFT_PATH, "r", encoding="utf-8") as f:
        frontend = json.load(f)

    tampered_ledger = copy.deepcopy(ledger)
    tampered_ledger["dossiers"]["ULTA"]["publication_metadata"]["review_status"] = "RESEARCH_REVIEWED"

    t_ledger_path = tmp_path / "pilot_scaling_ledger.json"
    t_manifest_path = tmp_path / "pilot_scaling_manifest.json"
    t_fe_path = tmp_path / "opportunityDossierPilotDrafts.json"

    with open(t_ledger_path, "w", encoding="utf-8") as f:
        json.dump(tampered_ledger, f)
    with open(t_fe_path, "w", encoding="utf-8") as f:
        json.dump(frontend, f)

    with open(t_ledger_path, "rb") as f:
        t_sha = hashlib.sha256(f.read()).hexdigest()
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["pilot_ledger_sha256"] = t_sha

    with open(t_manifest_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f)

    result = validate_existing_pilot_artifacts(
        ledger_path=t_ledger_path,
        manifest_path=t_manifest_path,
        frontend_draft_path=t_fe_path,
    )
    assert result is False, "Validation must reject unauthorized review status upgrade"


def test_14_negative_manifest_hash_mismatch(tmp_path):
    """Negative test: Manifest ledger hash mismatch fails validation immediately."""
    with open(DEFAULT_PILOT_LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    with open(DEFAULT_PILOT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(DEFAULT_FRONTEND_DRAFT_PATH, "r", encoding="utf-8") as f:
        frontend = json.load(f)

    t_ledger_path = tmp_path / "pilot_scaling_ledger.json"
    t_manifest_path = tmp_path / "pilot_scaling_manifest.json"
    t_fe_path = tmp_path / "opportunityDossierPilotDrafts.json"

    with open(t_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f)
    with open(t_fe_path, "w", encoding="utf-8") as f:
        json.dump(frontend, f)

    # Corrupt the recorded ledger hash in manifest
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["pilot_ledger_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"

    with open(t_manifest_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f)

    result = validate_existing_pilot_artifacts(
        ledger_path=t_ledger_path,
        manifest_path=t_manifest_path,
        frontend_draft_path=t_fe_path,
    )
    assert result is False, "Validation must fail when manifest hash does not match actual ledger file"


def test_15_negative_missing_evidence_mapping(tmp_path):
    """Negative test: Missing source_evidence_tracing on representative candidate fails validation."""
    with open(DEFAULT_PILOT_LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    with open(DEFAULT_PILOT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(DEFAULT_FRONTEND_DRAFT_PATH, "r", encoding="utf-8") as f:
        frontend = json.load(f)

    tampered_ledger = copy.deepcopy(ledger)
    tampered_ledger["dossiers"]["ULTA"]["financial_profile"]["source_evidence_tracing"] = None
    tampered_ledger["dossiers"]["ULTA"]["source_evidence_tracing"] = None

    t_ledger_path = tmp_path / "pilot_scaling_ledger.json"
    t_manifest_path = tmp_path / "pilot_scaling_manifest.json"
    t_fe_path = tmp_path / "opportunityDossierPilotDrafts.json"

    with open(t_ledger_path, "w", encoding="utf-8") as f:
        json.dump(tampered_ledger, f)
    with open(t_fe_path, "w", encoding="utf-8") as f:
        json.dump(frontend, f)

    with open(t_ledger_path, "rb") as f:
        t_sha = hashlib.sha256(f.read()).hexdigest()
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["pilot_ledger_sha256"] = t_sha

    with open(t_manifest_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f)

    result = validate_existing_pilot_artifacts(
        ledger_path=t_ledger_path,
        manifest_path=t_manifest_path,
        frontend_draft_path=t_fe_path,
    )
    assert result is False, "Validation must fail when ULTA evidence mapping is missing"


def test_16_negative_frontend_financial_tampering(valid_pilot_artifacts):
    """Negative test: Tampering with financial metrics in opportunityDossierPilotDrafts.json fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["financial_profile"]["total_financial_debt"] = 0.0
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_17_negative_frontend_narrative_tampering(valid_pilot_artifacts):
    """Negative test: Tampering with narrative synthesis in opportunityDossierPilotDrafts.json fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["central_assessment"]["presentation_summary"] = "Tampered summary."
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_18_negative_frontend_scenario_tampering(valid_pilot_artifacts):
    """Negative test: Tampering with valuation scenario in opportunityDossierPilotDrafts.json fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["valuation"]["scenarios"]["base"]["per_share_value"] = 999.99
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_19_negative_frontend_review_status_tampering(valid_pilot_artifacts):
    """Negative test: Unauthorized review status promotion in opportunityDossierPilotDrafts.json fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["publication_metadata"]["review_status"] = "RESEARCH_REVIEWED"
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_20_negative_docs_freshness_mismatch(valid_pilot_artifacts):
    """Negative test: Tampering with pilot_scaling_cohort_matrix.md fails doc freshness check."""
    doc_path = valid_pilot_artifacts["doc_matrix_path"]
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("# Tampered Doc Matrix\n")

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_21_negative_frontend_primary_filing_url_mutation(valid_pilot_artifacts):
    """Negative test: Mutating primary filing URL in frontend projection fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["financial_profile"]["primary_filings"][0]["url"] = "https://tampered.sec.gov/filing"
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_22_negative_frontend_wacc_mutation(valid_pilot_artifacts):
    """Negative test: Mutating reverse-DCF WACC in frontend projection fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["valuation"]["reverse_dcf_parameters"]["wacc_pct"] = 90.0
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_23_negative_frontend_scenario_revenue_growth_mutation(valid_pilot_artifacts):
    """Negative test: Mutating scenario revenue growth assumption in frontend projection fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["valuation"]["scenarios"]["base"]["revenue_cagr_pct"] = 80.0
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_24_negative_frontend_unexpected_added_field(valid_pilot_artifacts):
    """Negative test: Injecting an unexpected extra field into frontend projection fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["unexpected_extra_field"] = "injected"
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_25_negative_frontend_missing_field(valid_pilot_artifacts):
    """Negative test: Deleting a required field from frontend projection fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    del fe["dossiers"]["ULTA"]["financial_profile"]["primary_filings"]
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_26_negative_validation_failure_leaves_artifact_bytes_unchanged(valid_pilot_artifacts):
    """Negative test: Validation failure leaves all artifact files strictly byte-identical (read-only guarantee)."""
    # Mutate frontend projection
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ULTA"]["valuation"]["reverse_dcf_parameters"]["wacc_pct"] = 90.0
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    # Compute hashes of all 4 artifacts before validation call
    hashes_before = {}
    for key, path in valid_pilot_artifacts.items():
        with open(path, "rb") as f:
            hashes_before[key] = hashlib.sha256(f.read()).hexdigest()

    result = validate_existing_pilot_artifacts(**valid_pilot_artifacts)
    assert result is False

    # Verify hashes after validation call are 100% byte-identical
    for key, path in valid_pilot_artifacts.items():
        with open(path, "rb") as f:
            hash_after = hashlib.sha256(f.read()).hexdigest()
        assert hash_after == hashes_before[key], f"Artifact {path} was modified during failed validation!"


def test_27_control_valid_pilot_artifacts_passes(valid_pilot_artifacts):
    """Control test: Unmodified complete pilot artifacts pass validation cleanly."""
    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is True


def test_28_published_watch_manifest_integrity():
    """Verify that published watch manifest contains exactly 3 published WATCH assets (FIX, DUOL, ULTA), holds IREN and CORZ, and excludes MU and PLTR."""
    assert DEFAULT_PUBLISHED_MANIFEST_PATH.exists(), f"Published manifest missing at {DEFAULT_PUBLISHED_MANIFEST_PATH}"
    assert DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH.exists(), f"Frontend published manifest missing at {DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH}"

    with open(DEFAULT_PUBLISHED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        m2 = json.load(f)

    assert m1 == m2, "Docs and frontend published watch manifests must be identical"
    assert m1["schema_version"] == "ARX-PUBLISHED-WATCH-MANIFEST-V1"
    assert m1["governing_standard"] == "ARX-QUALITY-WATCH-STANDARD-V1"
    assert m1["published_symbols"] == ["FIX", "DUOL", "ULTA"]
    assert m1["published_count"] == 3
    assert m1["baseline_watch_symbols"] == ["FIX", "DUOL"]
    assert m1["held_historical_symbols"] == ["IREN", "CORZ"]
    assert m1["excluded_historical_symbols"] == ["MU"]
    assert m1["excluded_draft_symbols"] == ["PLTR"]
    assert "ULTA" in m1["accepted_additions"]
    assert m1["accepted_additions"]["ULTA"]["publication_metadata"]["lifecycle_status"] == "CANDIDATE_SELECTED"
    assert m1["accepted_additions"]["ULTA"]["publication_metadata"]["dossier_version"] == "1.0.0-rc1"
    assert m1["accepted_additions"]["ULTA"]["publication_metadata"]["review_status"] == "RESEARCH_EVIDENCE_ACCEPTED"
    assert m1["accepted_additions"]["ULTA"]["valuation"]["solved_5y_cagr_pct"] == 15.45
    assert m1["accepted_additions"]["ULTA"]["financial_profile"]["total_financial_debt"] == 339578000.0
    assert m1["accepted_additions"]["ULTA"]["financial_profile"]["is_zero_debt"] is False
    assert m1["additions_provenance"]["ULTA"]["acceptance_basis"] == "REUSED_ACCEPTED_RESEARCH_EVIDENCE"
    assert m1["additions_provenance"]["ULTA"]["reference_price"] == 592.94
    assert len(m1["additions_provenance"]["ULTA"]["content_sha256"]) == 64

