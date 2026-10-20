"""
tests/test_generate_opportunity_dossier_snapshot.py

Unit and parity tests for the Opportunity Dossier snapshot generator.
Verifies cryptographic digest matching, deterministic output, fail-closed validation,
and semantic preservation across the 5-company research cohort.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "research" / "generate_opportunity_dossier_snapshot.py"
SOURCE_LEDGER = REPO_ROOT / "docs" / "research" / "opportunity_discovery" / "odp05_discovery_ledger.json"
SNAPSHOT_PATH = REPO_ROOT / "frontend" / "lib" / "opportunityDossierSnapshot.json"

EXPECTED_SHA256 = "a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d"


def test_generator_check_mode_passes():
    """Verifies that running the generator with --check passes on clean repo state."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "CHECK PASS" in result.stdout


def test_generator_detects_stale_output(tmp_path):
    """Verifies that --check exits 1 if the output file is altered or stale."""
    tampered_file = tmp_path / "tampered_snapshot.json"
    tampered_file.write_text('{"tampered": true}\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--output", str(tampered_file), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "CHECK FAIL" in result.stderr


def test_generator_fails_on_missing_source(tmp_path):
    """Verifies that generator exits 1 when source ledger does not exist."""
    missing_source = tmp_path / "nonexistent_ledger.json"
    out_file = tmp_path / "out.json"

    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--source", str(missing_source), "--output", str(out_file)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FATAL: Source ledger not found" in result.stderr


def test_generator_fails_on_tampered_source_hash(tmp_path):
    """Verifies that generator enforces the exact expected source SHA-256."""
    tampered_source = tmp_path / "tampered_ledger.json"
    tampered_source.write_text('{"metadata": {}, "assessments": {}}\n', encoding="utf-8")
    out_file = tmp_path / "out.json"

    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--source", str(tampered_source), "--output", str(out_file)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "SHA-256 mismatch" in result.stderr


def test_snapshot_content_semantic_invariants():
    """Verifies that the generated snapshot adheres to all accepted research invariants."""
    assert SNAPSHOT_PATH.exists()
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    # Metadata
    meta = snapshot["snapshot_metadata"]
    assert meta["source_ledger_sha256"] == EXPECTED_SHA256
    assert meta["market_session_date"] == "2026-10-16"
    assert meta["cohort_symbols"] == ["FIX", "DUOL", "IREN", "CORZ", "MU"]

    dossiers = snapshot["dossiers"]
    assert set(dossiers.keys()) == {"FIX", "DUOL", "IREN", "CORZ", "MU"}

    # FIX verification
    fix = dossiers["FIX"]
    assert fix["identity"]["cik"] == "0001035983"
    assert fix["central_assessment"]["disposition"] == "WATCH_CONDITIONS"
    assert fix["valuation"]["valuation_status"] == "COMPUTED_VERIFIED"
    assert fix["valuation"]["reference_price"] == 1439.07
    assert fix["valuation"]["solved_5y_cagr_pct"] == 30.07
    assert fix["valuation"]["reference_price_label"] == "Reference price used in reverse DCF"
    assert fix["financial_profile"]["debt_display"] == "$54.1M"
    assert fix["financial_profile"]["is_zero_debt"] is False
    assert fix["financial_profile"]["revenue_period"] == "TTM (Ended 2026-06-30)"

    # DUOL verification (Zero debt semantics)
    duol = dossiers["DUOL"]
    assert duol["identity"]["cik"] == "0001562088"
    assert duol["central_assessment"]["disposition"] == "WATCH_CONDITIONS"
    assert duol["valuation"]["valuation_status"] == "COMPUTED_VERIFIED"
    assert duol["valuation"]["reference_price"] == 141.76
    assert duol["valuation"]["solved_5y_cagr_pct"] == 22.99
    assert duol["financial_profile"]["is_zero_debt"] is True
    assert duol["financial_profile"]["debt_display"] == "$0.00 (Zero Funded Debt Evidenced)"
    assert "WACC 10.0%" in duol["valuation"]["convention_summary"]
    assert "Terminal g 3.0%" in duol["valuation"]["convention_summary"]

    # IREN verification (Valuation suppression)
    iren = dossiers["IREN"]
    assert iren["identity"]["cik"] == "0001878848"
    assert iren["central_assessment"]["disposition"] == "WATCH_CONDITIONS"
    assert iren["valuation"]["valuation_status"] == "VALUATION_SUPPRESSED"
    assert iren["valuation"]["solved_5y_cagr_pct"] is None
    assert iren["valuation"]["scenarios"] is None
    assert iren["valuation"]["reference_price_label"] == "Research reference price"
    assert "Hyperscale data center construction" in iren["valuation"]["suppression_reason"]
    assert len(iren["valuation"]["resume_conditions"]) == 3
    assert iren["financial_profile"]["revenue_period"] == "FY2026 (Ended 2026-06-30)"

    # CORZ verification (Valuation suppression + CoreWeave $6.7B, not drifted $8.7B)
    corz = dossiers["CORZ"]
    assert corz["identity"]["cik"] == "0001839341"
    assert corz["central_assessment"]["disposition"] == "WATCH_CONDITIONS"
    assert corz["valuation"]["valuation_status"] == "VALUATION_SUPPRESSED"
    assert corz["valuation"]["solved_5y_cagr_pct"] is None
    assert "two consecutive quarters of positive free cash flow post total capital expenditures" in corz["valuation"]["resume_conditions"][0]
    assert "Construction cost overruns forcing emergency dilutive equity" in corz["monitoring"]["observable_invalidation_trigger"]
    assert corz["financial_profile"]["revenue_period"] == "Q2 2026 (Quarter Ended 2026-06-30)"

    # MU verification (Qualified negative conclusion)
    mu = dossiers["MU"]
    assert mu["identity"]["cik"] == "0000723125"
    assert mu["central_assessment"]["disposition"] == "NO_SUPPORTED_OPPORTUNITY"
    assert mu["valuation"]["valuation_status"] == "COMPUTED_QUALIFIED"
    assert mu["valuation"]["solved_5y_cagr_pct"] == 21.29
    assert mu["valuation"]["qualification_notice"] is not None
    assert "COMPUTED_QUALIFIED" in mu["valuation"]["qualification_notice"]
    assert "DRAM contract price declines or sequential drop in blended ASPs" in mu["monitoring"]["observable_invalidation_trigger"]
    assert "below $500" in mu["monitoring"]["price_watch_condition"]
    assert mu["financial_profile"]["revenue_period"] == "FY2026 (Ended 2026-09-03)"
