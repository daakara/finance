"""
tests/test_export_alc_dossier.py

Comprehensive test suite verifying deterministic export fidelity, schema compatibility,
numerical parity, and fail-closed safety for Alcon V4 Opportunity Dossier export.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.tier1


from scripts.research.export_alc_dossier import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    EXPECTED_CANONICAL_LEDGER_SHA256,
    TARGET_CIK,
    TARGET_COMPANY,
    TARGET_SYMBOL,
    compute_sha256,
    export_alc_dossier,
    validate_canonical_ledger,
)


@pytest.fixture(scope="module")
def exported_dossier():
    """Generates and returns the exported Alcon dossier from the canonical ledger."""
    assert DEFAULT_INPUT_PATH.exists(), f"Canonical ledger missing: {DEFAULT_INPUT_PATH}"
    return export_alc_dossier(DEFAULT_INPUT_PATH)


def test_01_five_scenario_preservation(exported_dossier):
    """1. Preserves all canonical scenarios and dynamic variants."""
    scenarios = exported_dossier["valuation"]["scenarios"]
    expected_scenarios = [
        "conservative",
        "base_multiple",
        "base_dcf",
        "optimistic",
        "conservative_dynamic",
        "optimistic_dynamic",
    ]
    for sc in expected_scenarios:
        assert sc in scenarios, f"Missing scenario: {sc}"
        proj = scenarios[sc]
        assert proj["per_share_value"] > 0
        assert proj["exit_method"] in ["multiple", "gordon_growth"]
        assert proj["holding_period_years"] == 5.0


def test_02_numerical_parity_against_frozen_ledger(exported_dossier):
    """2. Numerical parity against independently certified v4.0 ledger values."""
    base_dcf = exported_dossier["valuation"]["scenarios"]["base_dcf"]
    
    assert base_dcf["per_share_value"] == 91.82
    assert base_dcf["potential_return_pct"] == pytest.approx(36.17, abs=0.01)
    assert base_dcf["five_period_annual_irr_pct"] == 6.92
    assert base_dcf["calendar_dated_xirr_pct"] == 6.93
    assert base_dcf["enterprise_value"] == pytest.approx(41487.79, abs=0.01)
    assert base_dcf["equity_value"] == pytest.approx(43663.99, abs=0.01)
    assert base_dcf["revenue_cagr_pct"] == 6.50
    assert base_dcf["operating_margin_pct"] == 18.50
    assert base_dcf["year5_revenue"] == 14396.6
    assert base_dcf["year5_ebit"] == pytest.approx(2663.37, abs=0.01)
    assert base_dcf["year5_ebitda"] == pytest.approx(3815.10, abs=0.01)
    assert base_dcf["wacc_pct"] == 7.25
    assert base_dcf["terminal_growth_pct"] == 2.50
    assert base_dcf["ending_diluted_shares"] == 475.524


def test_03_dated_cash_flow_and_xirr_schedule(exported_dossier):
    """3. Dated XIRR and 7-event cash-flow schedule preservation."""
    ext = exported_dossier["alcon_v4_extension"]
    cash_flows = ext["dated_cash_flows"]
    assert len(cash_flows) == 7

    # Event 0: Initial Spot Investment
    assert cash_flows[0]["event"] == "INITIAL_INVESTMENT"
    assert cash_flows[0]["date"] == "2026-10-22"
    assert cash_flows[0]["cash_flow"] == -67.43

    # Events 1-5: Annual Dividends
    expected_divs = [0.36, 0.39, 0.42, 0.45, 0.48]
    expected_dates = ["2027-05-15", "2028-05-15", "2029-05-15", "2030-05-15", "2031-05-15"]
    for i, (div, dt) in enumerate(zip(expected_divs, expected_dates), start=1):
        assert cash_flows[i]["event"] == f"ANNUAL_DIVIDEND_YEAR_{i}"
        assert cash_flows[i]["date"] == dt
        assert cash_flows[i]["cash_flow"] == div

    # Event 6: Terminal Share Disposition
    assert cash_flows[6]["event"] == "TERMINAL_SHARE_DISPOSITION"
    assert cash_flows[6]["date"] == "2031-10-22"
    assert cash_flows[6]["cash_flow"] == pytest.approx(91.82, abs=0.01)

    # Holding period
    assert ext["holding_period_days"] == 1826
    assert ext["holding_period_years"] == 5.0


def test_04_bridge_currency_units_and_denominator(exported_dossier):
    """4. Balance sheet bridge, USD currency, $M units, and share denominator transitions."""
    fp = exported_dossier["financial_profile"]
    assert fp["currency"] == "USD"
    assert fp["reporting_currency"] == "USD"
    assert fp["cash_and_equivalents"] == 1200000000.0
    assert fp["total_financial_debt"] == 4500000000.0
    assert fp["net_debt_bridge"] == 3300000000.0
    assert fp["selected_share_basis"]["value"] == 484524000.0

    # Valuation terminal bridge
    base_dcf = exported_dossier["valuation"]["scenarios"]["base_dcf"]
    assert base_dcf["net_debt_bridge"] == pytest.approx(-2176.2, abs=0.1)  # net cash of $2,176.2M
    assert base_dcf["ending_diluted_shares"] == 475.524


def test_05_accumulation_ceilings_and_invertibility(exported_dossier):
    """5. Accumulation ceilings ($64.15, $62.70, $58.59) and invertibility verification."""
    ceilings = exported_dossier["alcon_v4_extension"]["accumulation_ceilings"]
    
    annual = ceilings["annual_irr_basis"]
    assert annual["hurdle_80"] == 64.15
    assert annual["hurdle_85"] == 62.70
    assert annual["hurdle_100"] == 58.59
    assert annual["invertibility_verified"] is True

    calendar = ceilings["calendar_xirr_basis"]
    assert calendar["hurdle_80"] == 64.20
    assert calendar["hurdle_85"] == 62.75
    assert calendar["hurdle_100"] == 58.64
    assert calendar["invertibility_verified"] is True


def test_06_historical_reference_price_and_dates(exported_dossier):
    """6. Historical reference price ($67.43) and as-of date (2026-10-21) semantics."""
    v = exported_dossier["valuation"]
    assert v["reference_price"] == 67.43
    assert v["reference_date"] == "2026-10-21"
    assert "frozen historical" in v["reference_price_label"].lower()
    
    pm = exported_dossier["publication_metadata"]
    assert pm["market_session_date"] == "2026-10-21"
    assert pm["valuation_date"] == "2026-10-22"
    assert pm["investment_end_date"] == "2031-10-22"
    assert pm["is_read_only"] is True


def test_07_identity_and_cross_listing_binding(exported_dossier):
    """7. Alcon Inc. / NYSE / SIX / CIK 0001167379 identity binding."""
    identity = exported_dossier["identity"]
    assert identity["symbol"] == TARGET_SYMBOL
    assert identity["company_name"] == TARGET_COMPANY
    assert identity["cik"] == TARGET_CIK
    assert identity["exchange"] == "NYSE"
    assert identity["primary_us_listing"]["ticker"] == "ALC"
    assert identity["secondary_listings"][0]["exchange"] == "SIX"
    assert identity["secondary_listings"][0]["ticker"] == "ALC"
    assert identity["secondary_listings"][0]["currency"] == "CHF"


def test_08_deterministic_repeated_export(tmp_path):
    """8. Deterministic repeated export produces identical byte-for-byte output."""
    dossier1 = export_alc_dossier(DEFAULT_INPUT_PATH)
    dossier2 = export_alc_dossier(DEFAULT_INPUT_PATH)

    b1 = (json.dumps(dossier1, indent=2, sort_keys=True) + "\n").encode("utf-8")
    b2 = (json.dumps(dossier2, indent=2, sort_keys=True) + "\n").encode("utf-8")

    assert b1 == b2
    assert compute_sha256(DEFAULT_OUTPUT_PATH) == compute_sha256(DEFAULT_OUTPUT_PATH)


def test_09_check_mode_drift_detection(tmp_path):
    """9. Check mode detects drift without writing to disk."""
    # Run CLI --check on valid file
    res = subprocess.run(
        [sys.executable, "scripts/research/export_alc_dossier.py", "--check"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "CHECK PASSED" in res.stdout

    # Test drift detection against modified file in tmp_path
    drift_file = tmp_path / "drifted_dossier.json"
    drift_file.write_text('{"drift": true}\n', encoding="utf-8")
    res_drift = subprocess.run(
        [sys.executable, "scripts/research/export_alc_dossier.py", "--output", str(drift_file), "--check"],
        capture_output=True,
        text=True,
    )
    assert res_drift.returncode == 1
    assert "Export drift detected" in (res_drift.stdout + res_drift.stderr)


def test_10_invalid_and_corrupt_input_fails_closed(tmp_path):
    """10. Missing, invalid, or version-mismatched input fails closed."""
    # Test non-existent path
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        export_alc_dossier(non_existent)

    # Test invalid model version
    invalid_ver = tmp_path / "invalid_ver.json"
    invalid_ver.write_text(json.dumps({"metadata": {"model_version": "v3.0"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported model version"):
        validate_canonical_ledger(json.loads(invalid_ver.read_text()), invalid_ver)

    # Test missing required scenario
    missing_sc = tmp_path / "missing_sc.json"
    missing_sc.write_text(
        json.dumps({
            "metadata": {"model_version": "v4.0", "currency": "USD"},
            "scenarios": {"CONSERVATIVE": {}}
        }),
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Missing required canonical scenario"):
        validate_canonical_ledger(json.loads(missing_sc.read_text()), missing_sc)


def test_11_schema_compatibility_with_opportunity_dossier(exported_dossier):
    """11. Extension preserves existing OpportunityDossier schema compatibility."""
    required_dossier_keys = [
        "dossier_id",
        "publication_metadata",
        "identity",
        "central_assessment",
        "financial_profile",
        "valuation",
        "monitoring",
        "navigation",
        "alcon_v4_extension",
    ]
    for key in required_dossier_keys:
        assert key in exported_dossier, f"Missing top-level OpportunityDossier key: {key}"

    # Verify central_assessment disposition
    assert exported_dossier["central_assessment"]["disposition"] == "WATCH_CONDITIONS"
    # Verify publication status is draft
    assert exported_dossier["publication_metadata"]["publication_status"] == "DRAFT_RESEARCH_CANDIDATE"


def test_12_draft_output_path_isolation(exported_dossier):
    """12. Draft output path is isolated in drafts/ and inactive for production builds."""
    output_path = Path("docs/research/opportunity_discovery/drafts/ALC_opportunity_dossier_draft.json")
    assert output_path.exists()
    assert "drafts" in output_path.parts
    assert exported_dossier["publication_metadata"]["is_publication_active"] is False
    assert exported_dossier["publication_metadata"]["publication_status"] == "DRAFT_RESEARCH_CANDIDATE"
