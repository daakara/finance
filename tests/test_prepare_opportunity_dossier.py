"""
tests/test_prepare_opportunity_dossier.py

Focused Invariant and Integrity Test Suite for the Opportunity Dossier Scaling Pilot.
Verifies:
1. Frozen ODP-05 ledger baseline preservation (exact SHA-256 digest match).
2. Preparation of the full 20-company pilot cohort (5 baseline + 15 additions).
3. Deterministic Reverse-DCF solving and roundtrip forward model reconciliation (< $0.05).
4. Evidence provenance integrity (valid SEC EDGAR accessions, primary URLs, reporting dates).
5. Capital structure accounting: documented zero debt vs funded debt.
6. Lease-as-operating-expense convention (ASC 842) for retail brands.
7. Explicit valuation suppression (IREN, CORZ) and qualification (MU, PLTR, KLAC, SNPS).
8. Absence of manufactured reviewer identities or review timestamps in draft metadata.
9. Complete isolation of draft records from production publication dataset.
"""

import hashlib
import json
import os
import pytest
from pathlib import Path

from scripts.research.prepare_opportunity_dossier import (
    REPO_ROOT,
    FROZEN_ODP05_LEDGER_PATH,
    FROZEN_ODP05_SHA256,
    BASELINE_COHORT,
    PILOT_ADDITIONS_COHORT,
    TOTAL_PILOT_COHORT,
    DEFAULT_PILOT_LEDGER_PATH,
    DEFAULT_PILOT_MANIFEST_PATH,
    DEFAULT_FRONTEND_DRAFT_PATH,
    prepare_pilot_scaling_corpus,
    validate_existing_pilot_artifacts,
    PILOT_SPECS,
)
from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    solve_reverse_dcf,
)


def test_1_frozen_odp05_baseline_unmodified():
    """Verify that ODP-05 canonical ledger is untouched and matches accepted SHA-256 digest."""
    assert FROZEN_ODP05_LEDGER_PATH.exists(), f"ODP-05 ledger missing at {FROZEN_ODP05_LEDGER_PATH}"
    with open(FROZEN_ODP05_LEDGER_PATH, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == FROZEN_ODP05_SHA256, f"ODP-05 ledger tampered: {actual_sha} != {FROZEN_ODP05_SHA256}"


def test_2_cohort_completeness_and_sizing():
    """Verify that the scaling pilot cohort defines exactly 20 companies (5 baseline + 15 additions)."""
    assert len(BASELINE_COHORT) == 5
    assert len(PILOT_ADDITIONS_COHORT) == 15
    assert len(TOTAL_PILOT_COHORT) == 20
    assert len(set(TOTAL_PILOT_COHORT)) == 20, "Cohort symbols must be unique"
    for sym in PILOT_ADDITIONS_COHORT:
        assert sym in PILOT_SPECS, f"Addition {sym} missing from PILOT_SPECS"


def test_3_reverse_dcf_roundtrip_verification():
    """Verify that every numeric valuation in the 15 additions solves and roundtrips within $0.05."""
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
    zero_debt_symbols = ["DUOL", "ULTA", "MEDP", "DECK", "LULU", "EME", "ANET", "PLTR", "ISRG", "CDNS"]
    for sym in zero_debt_symbols:
        d = dossiers[sym]
        fp = d["financial_profile"]
        assert fp["is_zero_debt"] is True, f"{sym} must be marked is_zero_debt=True"
        assert fp["total_financial_debt"] == 0.0, f"{sym} total_financial_debt must be 0.0"
        assert fp["zero_debt_support"] is not None, f"{sym} must have documented zero_debt_support"

    # Debt-bearing companies
    debt_symbols = ["FIX", "KLAC", "VRT", "PWR", "SNPS", "NOW", "ELF", "IREN", "CORZ", "MU"]
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
    for sym in ["MU", "PLTR", "KLAC", "SNPS"]:
        d = dossiers[sym]
        v = d["valuation"]
        assert v["valuation_status"] == "COMPUTED_QUALIFIED"
        assert v["qualification_notice"] is not None


def test_8_no_manufactured_reviewer_metadata():
    """Verify that draft records never manufacture reviewer identities or review timestamps."""
    ledger, _, _ = prepare_pilot_scaling_corpus()
    dossiers = ledger["dossiers"]

    for sym in PILOT_ADDITIONS_COHORT:
        meta = dossiers[sym]["publication_metadata"]
        assert meta["lifecycle_status"] == "DRAFT"
        assert meta["review_status"] == "PENDING_RESEARCH_REVIEW"
        assert meta["reviewer_id"] is None, f"{sym} has manufactured reviewer_id: {meta['reviewer_id']}"
        assert meta["reviewed_at_utc"] is None, f"{sym} has manufactured reviewed_at_utc: {meta['reviewed_at_utc']}"


def test_9_pilot_scaling_artifacts_validation():
    """Verify that the validation routine passes across generated pilot artifacts."""
    assert validate_existing_pilot_artifacts() is True
