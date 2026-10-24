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

    # 10. Selection rationale
    assert "selection_rationale" in tracing
    assert len(tracing["selection_rationale"]) == 6


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
    """Verify that published watch manifest contains exactly 6 published WATCH assets (FIX, DUOL, ULTA, MEDP, ISRG, DECK), holds IREN and CORZ, and excludes MU and PLTR."""
    assert DEFAULT_PUBLISHED_MANIFEST_PATH.exists(), f"Published manifest missing at {DEFAULT_PUBLISHED_MANIFEST_PATH}"
    assert DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH.exists(), f"Frontend published manifest missing at {DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH}"

    with open(DEFAULT_PUBLISHED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        m2 = json.load(f)

    assert m1 == m2, "Docs and frontend published watch manifests must be identical"
    assert m1["schema_version"] == "ARX-PUBLISHED-WATCH-MANIFEST-V1"
    assert m1["governing_standard"] == "ARX-QUALITY-WATCH-STANDARD-V1"
    assert m1["published_symbols"] == ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"]
    assert m1["published_count"] == 6
    assert m1["baseline_watch_symbols"] == ["FIX", "DUOL"]
    assert m1["held_historical_symbols"] == ["IREN", "CORZ"]
    assert m1["excluded_historical_symbols"] == ["MU"]
    assert m1["excluded_draft_symbols"] == ["PLTR"]

    # ULTA
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

    # MEDP
    assert "MEDP" in m1["accepted_additions"]
    assert m1["accepted_additions"]["MEDP"]["publication_metadata"]["lifecycle_status"] == "CANDIDATE_SELECTED"
    assert m1["accepted_additions"]["MEDP"]["publication_metadata"]["review_status"] == "RESEARCH_EVIDENCE_ACCEPTED"
    assert m1["accepted_additions"]["MEDP"]["valuation"]["solved_5y_cagr_pct"] == 29.41
    assert m1["accepted_additions"]["MEDP"]["financial_profile"]["total_financial_debt"] == 0.0
    assert m1["accepted_additions"]["MEDP"]["financial_profile"]["is_zero_debt"] is True
    assert m1["additions_provenance"]["MEDP"]["reference_price"] == 618.47
    assert len(m1["additions_provenance"]["MEDP"]["content_sha256"]) == 64

    # ISRG
    assert "ISRG" in m1["accepted_additions"]
    assert m1["accepted_additions"]["ISRG"]["publication_metadata"]["lifecycle_status"] == "CANDIDATE_SELECTED"
    assert m1["accepted_additions"]["ISRG"]["publication_metadata"]["review_status"] == "RESEARCH_EVIDENCE_ACCEPTED"
    assert m1["accepted_additions"]["ISRG"]["valuation"]["solved_5y_cagr_pct"] == 36.87
    assert m1["accepted_additions"]["ISRG"]["financial_profile"]["total_financial_debt"] == 0.0
    assert m1["accepted_additions"]["ISRG"]["financial_profile"]["is_zero_debt"] is True
    assert m1["additions_provenance"]["ISRG"]["reference_price"] == 426.70
    assert len(m1["additions_provenance"]["ISRG"]["content_sha256"]) == 64

    # DECK
    assert "DECK" in m1["accepted_additions"]
    assert m1["accepted_additions"]["DECK"]["publication_metadata"]["lifecycle_status"] == "CANDIDATE_SELECTED"
    assert m1["accepted_additions"]["DECK"]["publication_metadata"]["review_status"] == "RESEARCH_EVIDENCE_ACCEPTED"
    assert m1["accepted_additions"]["DECK"]["valuation"]["solved_5y_cagr_pct"] == -2.34
    assert m1["accepted_additions"]["DECK"]["financial_profile"]["total_financial_debt"] == 0.0
    assert m1["accepted_additions"]["DECK"]["financial_profile"]["is_zero_debt"] is True
    assert m1["accepted_additions"]["DECK"]["financial_profile"]["operating_lease_liabilities_asc842"] == 472334000.0
    assert m1["additions_provenance"]["DECK"]["reference_price"] == 85.72
    assert len(m1["additions_provenance"]["DECK"]["content_sha256"]) == 64

    # Remaining expansion queue & selection rationale in manifest
    assert "remaining_expansion_queue" in m1
    assert len(m1["remaining_expansion_queue"]) == 10
    assert "KLAC" in m1["remaining_expansion_queue"]
    assert "ELF" in m1["remaining_expansion_queue"]
    for sym in ["ULTA", "MEDP", "ISRG", "DECK"]:
        assert "selection_rationale" in m1["additions_provenance"][sym]
        assert len(m1["additions_provenance"][sym]["selection_rationale"]) == 6


def test_29_expansion_additions_evidence_tracing_and_share_periods():
    """Verify that expansion additions MEDP, ISRG, and DECK contain complete evidence tracing and exact quarterly share measurement periods."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    for sym in ["MEDP", "ISRG", "DECK"]:
        d = dossiers[sym]
        tracing = d["financial_profile"].get("source_evidence_tracing")
        assert tracing is not None, f"{sym} missing source_evidence_tracing"
        assert tracing["shares_outstanding"]["measurement_period"] == "Three months ended June 30, 2026"
        assert d["financial_profile"]["selected_share_basis"]["measurement_period"] == "Three months ended June 30, 2026"
        assert "diluted" in d["financial_profile"]["selected_share_basis"]["rationale"].lower()


def test_30_deck_growth_claim_accuracy():
    """Verify that DECK brand and channel growth claims cite primary Q1 FY27 disclosures ($703.5M total HOKA, $1,019.5M net sales) without stale or unsupported claims."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    deck = ledger["dossiers"]["DECK"]
    supp = deck["central_assessment"]["strongest_supporting_evidence"]

    assert "market share gains (>25% YoY)" not in supp, "DECK must not claim market share gains >25% YoY without market data"
    assert "17.3%" in supp, "DECK must cite HOKA DTC net sales growth (+17.3% YoY)"
    assert "2.9%" in supp, "DECK must cite HOKA wholesale net sales growth (+2.9% YoY)"
    assert "703.5M" in supp, "DECK must cite verified total HOKA net sales ($703.5M)"
    assert "1,019.5M" in supp, "DECK must cite verified consolidated net sales ($1,019.5M)"
    for stale in ["545.0M", "151.0M", "394.0M", "825.3M"]:
        assert stale not in supp, f"DECK must not contain stale dollar figure: {stale}"


def test_31_isrg_recurring_revenue_and_measured_language():
    """Verify that ISRG recurring revenue definition includes operating leases (85%) and avoids hyperbolic adjectives."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    isrg = ledger["dossiers"]["ISRG"]
    summary = isrg["central_assessment"]["presentation_summary"]
    supp = isrg["central_assessment"]["strongest_supporting_evidence"]

    assert "unassailable" not in summary.lower(), "ISRG presentation_summary must not use 'unassailable'"
    assert "unassailable" not in supp.lower(), "ISRG strongest_supporting_evidence must not use 'unassailable'"
    assert "85%" in supp or "85%" in summary, "ISRG must state 85% recurring revenue definition"


def test_32_medp_operating_profitability_and_cash_generation():
    """Verify that MEDP grounds its quality moat in verified 20.97% operating margin and $502.7M net cash, with unsupported ROIC claims removed."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    medp = ledger["dossiers"]["MEDP"]
    tracing = medp["financial_profile"]["source_evidence_tracing"]
    supp = medp["central_assessment"]["strongest_supporting_evidence"]
    why = medp["selection_rationale"]["why_this_business"]

    # Positive assertions on verified facts
    assert "20.97%" in supp, "MEDP must cite verified 20.97% operating margin in supporting evidence"
    assert "$502.7M" in supp, "MEDP must cite $502.7M cash in supporting evidence"
    assert "zero funded debt" in supp.lower(), "MEDP must cite zero funded debt"
    assert "positive operating cash generation" in supp.lower(), "MEDP must cite positive operating cash generation"
    assert "20.97%" in why, "MEDP why_this_business must cite 20.97% operating margin"

    # Negative assertions: Unsupported ROIC and unsupported equity/funding claims removed
    assert "ROIC" not in supp, "MEDP strongest_supporting_evidence must not make unsupported ROIC claims"
    assert "75.3%" not in supp and "41.3%" not in supp, "MEDP must not cite unverified ROIC percentages"
    assert "100% self-funded" not in supp, "MEDP strongest_supporting_evidence must not make unsupported '100% self-funded' claims"
    assert "100% self-funded" not in str(medp), "MEDP dossier must not cite '100% self-funded'"
    assert "ROIC" not in why, "MEDP why_this_business must not cite unsupported ROIC"
    assert "1,114.6" not in str(tracing), "MEDP tracing must not cite non-existent $1,114.6M equity"


def test_33_structured_selection_rationale_completeness():
    """Verify that all published additions (ULTA, MEDP, ISRG, DECK) contain structured selection records answering A-F."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    required_keys = ["why_this_business", "why_research_now", "price_requirements", "differentiated_view", "assessment_changes", "queue_priority"]
    for sym in ["ULTA", "MEDP", "ISRG", "DECK"]:
        d = dossiers[sym]
        rationale = d.get("selection_rationale")
        assert rationale is not None, f"{sym} missing selection_rationale"
        for k in required_keys:
            assert rationale.get(k), f"{sym} selection_rationale missing {k}"


def test_34_negative_deck_growth_claim_tampering(valid_pilot_artifacts):
    """Negative test: Tampering with DECK narrative to restore unsupported >25% market share claim fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["DECK"]["central_assessment"]["strongest_supporting_evidence"] = "HOKA brand momentum continues rapid global wholesale and DTC market share gains (>25% YoY)."
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_35_negative_share_measurement_period_tampering(valid_pilot_artifacts):
    """Negative test: Mislabeling MEDP share measurement period as TTM fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["MEDP"]["financial_profile"]["selected_share_basis"]["measurement_period"] = "TTM ended 2026-06-30"
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_36_negative_isrg_hyperbolic_wording(valid_pilot_artifacts):
    """Negative test: Injecting 'unassailable' hyperbole into ISRG presentation summary fails validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)
    fe["dossiers"]["ISRG"]["central_assessment"]["presentation_summary"] = "Intuitive Surgical maintains an unassailable surgical robotics ecosystem."
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_37_financial_reconciliation_primary_sources_to_derived_metrics():
    """Verify that expansion additions MEDP, ISRG, and DECK reconcile perfectly against independent primary source fixtures."""
    from scripts.research.prepare_opportunity_dossier import run_fcff_model

    fixtures = {
        "MEDP": {
            "rev_10k": 2530234000.0,
            "rev_prior_10q": 1161881000.0,
            "rev_current_10q": 1413935000.0,
            "opinc_10k": 534935000.0,
            "opinc_prior_10q": 239858000.0,
            "opinc_current_10q": 288327000.0,
            "cash": 502688000.0,
            "st_investments": 0.0,
            "funded_debt": 0.0,
            "lease_liabilities": 144112000.0,
            "diluted_shares": 28554000.0,
            "expected_cagr": 29.41,
        },
        "ISRG": {
            "rev_10k": 10064700000.0,
            "rev_prior_10q": 4693400000.0,
            "rev_current_10q": 5663100000.0,
            "opinc_10k": 2945500000.0,
            "opinc_prior_10q": 1321500000.0,
            "opinc_current_10q": 1827200000.0,
            "cash": 2760400000.0,
            "st_investments": 2455800000.0,
            "funded_debt": 0.0,
            "lease_liabilities": 170900000.0,
            "diluted_shares": 357300000.0,
            "expected_cagr": 36.87,
        },
        "DECK": {
            "rev_10k": 5472296000.0,
            "rev_prior_10q": 964538000.0,
            "rev_current_10q": 1019531000.0,
            "opinc_10k": 1262903000.0,
            "opinc_prior_10q": 165287000.0,
            "opinc_current_10q": 155301000.0,
            "cash": 1602589000.0,
            "st_investments": 0.0,
            "funded_debt": 0.0,
            "lease_liabilities": 472334000.0,
            "diluted_shares": 138559000.0,
            "expected_cagr": -2.34,
        },
    }

    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    for sym, fix in fixtures.items():
        d = dossiers[sym]
        fin = d["financial_profile"]
        tracing = fin["source_evidence_tracing"]

        # Independent arithmetic derivation
        calc_ttm_rev = fix["rev_10k"] - fix["rev_prior_10q"] + fix["rev_current_10q"]
        calc_ttm_opinc = fix["opinc_10k"] - fix["opinc_prior_10q"] + fix["opinc_current_10q"]
        calc_margin = round((calc_ttm_opinc / calc_ttm_rev) * 100, 2)
        calc_total_cash = fix["cash"] + fix["st_investments"]
        calc_net_debt = fix["funded_debt"] - calc_total_cash

        # Dossier and financial profile reconciliation
        assert fin["revenue"] == calc_ttm_rev, f"{sym} revenue mismatch: {fin['revenue']} vs {calc_ttm_rev}"
        assert fin["operating_income"] == calc_ttm_opinc, f"{sym} operating income mismatch: {fin['operating_income']} vs {calc_ttm_opinc}"
        assert fin["operating_margin_pct"] == calc_margin, f"{sym} margin mismatch: {fin['operating_margin_pct']} vs {calc_margin}"
        assert fin["total_cash_and_investments"] == calc_total_cash, f"{sym} cash mismatch"
        assert fin["total_financial_debt"] == fix["funded_debt"], f"{sym} debt mismatch"
        assert fin["net_debt_bridge"] == calc_net_debt, f"{sym} net debt bridge mismatch"
        assert fin["selected_share_basis"]["value"] == fix["diluted_shares"], f"{sym} shares mismatch"

        # Source evidence tracing reconciliation
        rev_sources = tracing["revenue"]["filing_sources"]
        assert len(rev_sources) == 3, f"{sym} revenue tracing must include 3 filing sources (10-K, prior 10-Q, current 10-Q)"
        assert rev_sources[0]["reported_value"] == fix["rev_10k"]
        assert rev_sources[1]["reported_value"] == fix["rev_prior_10q"]
        assert rev_sources[2]["reported_value"] == fix["rev_current_10q"]

        opinc_sources = tracing["operating_income"]["filing_sources"]
        assert len(opinc_sources) == 3, f"{sym} operating income tracing must include 3 filing sources"
        assert opinc_sources[0]["reported_value"] == fix["opinc_10k"]
        assert opinc_sources[1]["reported_value"] == fix["opinc_prior_10q"]
        assert opinc_sources[2]["reported_value"] == fix["opinc_current_10q"]

        assert tracing["lease_treatment"]["total_lease_liabilities_usd"] == fix["lease_liabilities"]
        assert tracing["shares_outstanding"]["share_count"] == fix["diluted_shares"]


def test_38_negative_financial_component_tampering(valid_pilot_artifacts):
    """Negative test: Tampering with any filing source component in source_evidence_tracing fails artifact validation."""
    fe_path = valid_pilot_artifacts["frontend_draft_path"]
    with open(fe_path, "r", encoding="utf-8") as f:
        fe = json.load(f)

    # Tamper with DECK 10-K reported revenue
    fe["dossiers"]["DECK"]["financial_profile"]["source_evidence_tracing"]["revenue"]["filing_sources"][0]["reported_value"] = 9999999999.0
    with open(fe_path, "w", encoding="utf-8") as f:
        json.dump(fe, f)

    assert validate_existing_pilot_artifacts(**valid_pilot_artifacts) is False


def test_39_valuation_forward_model_roundtrip_verification():
    """Verify that all additions (ULTA, MEDP, ISRG, DECK) execute forward FCFF model roundtrip matching machine model output."""
    from scripts.research.prepare_opportunity_dossier import run_fcff_model, PILOT_SPECS

    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    for sym in ["ULTA", "MEDP", "ISRG", "DECK"]:
        d = dossiers[sym]
        fin = d["financial_profile"]
        val = d["valuation"]
        val_spec = PILOT_SPECS[sym]["valuation_spec"]

        shares = fin["selected_share_basis"]["value"]
        rt = run_fcff_model(
            base_revenue=fin["revenue"],
            current_shares=shares,
            net_debt_bridge=fin["net_debt_bridge"],
            growth_5y_pct=val["solved_5y_cagr_pct"],
            operating_margin_pct=val_spec["margin"],
            wacc_pct=val_spec["wacc"],
            terminal_growth_pct=val_spec["terminal_g"],
            tax_rate_pct=val_spec["tax_rate"],
            reinvestment_rate_pct=val_spec["reinvestment_rate"],
        )

        model_per_share = rt["per_share"]
        model_delta_cents = round(abs(model_per_share - val["reference_price"]) * 100, 2)

        assert val["roundtrip_value"] == round(model_per_share, 2), f"{sym} roundtrip_value must equal machine model per_share rounded to 2 decimal places"
        assert val["roundtrip_delta_cents"] == model_delta_cents, f"{sym} roundtrip_delta_cents must match machine calculation"
        assert model_delta_cents < 10.0, f"{sym} roundtrip delta ({model_delta_cents} cents) must be within 10 cent tolerance"



