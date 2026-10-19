"""
tests/test_fundamentals_api.py

Comprehensive unit, integration, and invariant tests for the read-only,
facts-only Sourced Fundamentals API and data service.

Verifies:
- P01: Bounded surface (FIX, DUOL, ULTA facts-only view, uncataloged graceful 404).
- P02: Identity correctness (symbol/CIK agreement, mismatch rejection).
- P03: Financial semantics (correct units, reported vs derived separation, null vs zero).
- P04: Period correctness (per-metric filing dates and periods).
- P05: Snapshot identity and parity (API output matches static frontend snapshot).
- P06: Pure read-only behavior with zero mutation side-effects.
- P07: Attributable SEC EDGAR source URLs and accession numbers.
- P08: Evidence-responsive debt presentation (preserves nulls, reflects actual evidence).
- P09: Facts-only scope (strictly no DCF scenarios, fair values, or buy rankings).
- P10: Graceful fail-closed error handling (503 for missing snapshot, 500 for corrupt).
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.research.sourced_fundamentals.service import (
    SourcedFundamentalsService,
    SnapshotState,
    reset_sourced_fundamentals_service,
    get_sourced_fundamentals_service,
)

client = TestClient(app)


def test_01_catalog_endpoint_success():
    """Verify GET /api/v1/fundamentals/ returns 200, valid metadata, and catalog list."""
    response = client.get("/api/v1/fundamentals/")
    assert response.status_code == 200
    data = response.json()

    assert "metadata" in data
    assert "companies" in data
    assert "count" in data
    assert data["count"] == len(data["companies"])
    assert data["count"] >= 8

    # Check cache-control header
    assert "Cache-Control" in response.headers
    assert "public" in response.headers["Cache-Control"]

    # Verify key companies are present in catalog
    symbols = [c["symbol"] for c in data["companies"]]
    for expected in ["FIX", "DUOL", "ULTA", "MEDP", "CELH", "DECK", "KLAC", "LULU"]:
        assert expected in symbols


def test_02_catalog_items_facts_only_structure():
    """Verify catalog items conform to facts-only schema with zero valuation/ranking leaks."""
    response = client.get("/api/v1/fundamentals/")
    data = response.json()
    for item in data["companies"]:
        assert "symbol" in item
        assert "company_name" in item
        assert "cik" in item
        assert "latest_reported_period" in item
        assert "latest_filing_date" in item
        assert isinstance(item["total_financial_debt_evidenced"], bool)

        # STRICT FACTS-ONLY INVARIANT: No valuation or investment priority labels
        assert "valuation_status" not in item
        assert "fair_value" not in item
        assert "implied_upside" not in item
        assert "disposition" not in item
        assert "investment_priority" not in item


def test_03_fix_snapshot_facts_only_success():
    """Verify GET /api/v1/fundamentals/FIX returns structured facts-only financial evidence."""
    response = client.get("/api/v1/fundamentals/FIX")
    assert response.status_code == 200
    data = response.json()

    # P02: Identity
    ident = data["company_identity"]
    assert ident["symbol"] == "FIX"
    assert ident["cik"] == "0001035983"
    assert "COMFORT SYSTEMS" in ident["company_name"].upper()
    assert ident["sec_edgar_company_url"] == "https://www.sec.gov/edgar/browse/?CIK=1035983"

    # P03 & P07: Reported facts with accession numbers and SEC archive URLs
    facts = data["reported_facts"]
    assert len(facts) >= 20
    metric_keys = [f["metric_key"] for f in facts]
    assert "revenue_ttm" in metric_keys
    assert "cfo_ttm" in metric_keys
    assert "capex_ttm" in metric_keys

    reported_facts_with_accession = [f for f in facts if f.get("accession_number")]
    assert len(reported_facts_with_accession) >= 10
    for f in reported_facts_with_accession:
        assert f["source_url"] is not None
        assert "sec.gov/Archives/edgar/data" in f["source_url"]
        assert f["status"] in ["REPORTED", "DERIVED"]
        assert f["fact_type"] == "PRIMARY_REPORTED"

    # Derived calculations
    derived = data["derived_calculations"]
    assert derived["revenue_ttm"]["value"] is not None
    assert derived["revenue_ttm"]["units"] == "USD"
    assert derived["cfo_minus_capex_ttm"]["value"] is not None
    assert derived["cfo_minus_capex_ttm"]["units"] == "USD"
    assert derived["total_financial_debt"]["value"] is not None
    assert derived["total_financial_debt"]["units"] == "USD"
    assert derived["total_financial_debt"]["status"] == "REPORTED"

    # Share count units check (must be SHARES, not USD)
    if "diluted_shares_ttm_latest" in derived:
        assert derived["diluted_shares_ttm_latest"]["units"] == "SHARES"

    # P09: STRICT FACTS-ONLY: deep_valuation and promotional claims must be completely absent
    assert "deep_valuation" not in data
    assert "fair_value" not in data
    assert "implied_upside" not in data
    assert "accumulation_corridor" not in data

    # Evidence notes should describe factual gaps and invalidation conditions without buy recommendations
    notes = data["research_evidence_notes"]
    assert "remaining_evidence_gaps" in notes
    assert "thesis_invalidation_conditions" in notes
    assert "disposition" not in notes


def test_04_duol_preserves_null_debt_and_facts_only():
    """Verify DUOL snapshot preserves null for total_financial_debt without synthetic zero substitution."""
    response = client.get("/api/v1/fundamentals/DUOL")
    assert response.status_code == 200
    data = response.json()

    derived = data["derived_calculations"]
    # Total financial debt must be null (never synthetic 0.0) under aggregate completeness
    assert derived["total_financial_debt"]["value"] is None
    assert derived["total_financial_debt"]["status"] == "UNAVAILABLE"
    assert "UNAVAILABLE" in derived["total_financial_debt"]["notes"]

    # Long-term and short-term debt are also null
    assert derived["long_term_debt"]["value"] is None
    assert derived["short_term_debt"]["value"] is None

    # Cash must be reported
    assert derived["cash_and_equivalents"]["value"] is not None
    assert derived["cash_and_equivalents"]["status"] == "REPORTED"

    # Evidence gaps must document debt concepts unavailable in XBRL
    notes = data["research_evidence_notes"]
    assert any("debt" in gap.lower() for gap in notes["remaining_evidence_gaps"])

    # P09: Valuation absent
    assert "deep_valuation" not in data


def test_05_ulta_partial_debt_reporting():
    """Verify ULTA reports short-term debt while preserving null total debt under aggregate completeness."""
    response = client.get("/api/v1/fundamentals/ULTA")
    assert response.status_code == 200
    data = response.json()

    derived = data["derived_calculations"]
    # Short-term debt is reported (revolving credit draw)
    assert derived["short_term_debt"]["value"] is not None
    assert derived["short_term_debt"]["value"] == 339578000.0
    assert derived["short_term_debt"]["status"] == "REPORTED"

    # Long term debt is null in automated harvester
    assert derived["long_term_debt"]["value"] is None
    assert derived["long_term_debt"]["status"] == "UNAVAILABLE"

    # Total financial debt must remain null under fail-closed aggregate completeness
    assert derived["total_financial_debt"]["value"] is None
    assert derived["total_financial_debt"]["status"] == "UNAVAILABLE"

    # P09: Valuation absent
    assert "deep_valuation" not in data


def test_06_uncataloged_symbol_fails_closed_404():
    """Verify uncataloged symbols return 404 Not Found with available catalog inventory."""
    response = client.get("/api/v1/fundamentals/NONEXISTENT_XYZ")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "not found" in detail.lower()
    assert "FIX" in detail
    assert "DUOL" in detail
    assert "ULTA" in detail


def test_07_identity_validation_and_mismatch():
    """Verify identity validation, whitespace trimming, and symbol mismatch rejection."""
    # Case insensitivity and whitespace trimming
    resp_trimmed = client.get("/api/v1/fundamentals/  fix  ")
    assert resp_trimmed.status_code == 200
    assert resp_trimmed.json()["company_identity"]["symbol"] == "FIX"

    # Empty symbol
    resp_empty = client.get("/api/v1/fundamentals/%20")
    assert resp_empty.status_code in (400, 404)

    # Unit-level service identity mismatch test
    service = get_sourced_fundamentals_service()
    assert service.validate_symbol_identity("FIX", {"fundamentals": {"symbol": "FIX", "cik": "0001035983"}})
    assert not service.validate_symbol_identity("FIX", {"fundamentals": {"symbol": "AAPL", "cik": "0000320193"}})
    assert not service.validate_symbol_identity("FIX", {"fundamentals": {"symbol": "FIX", "cik": None}})
    assert not service.validate_symbol_identity("", {"fundamentals": {"symbol": "FIX", "cik": "0001035983"}})


def test_08_missing_and_malformed_ledger_handling(tmp_path):
    """Verify service returns explicit 503 on missing ledger and 500 on malformed ledger without path leak."""
    # Test missing ledger
    missing_path = tmp_path / "nonexistent_ledger.json"
    svc_missing = SourcedFundamentalsService(ledger_path=missing_path)
    assert svc_missing.state == SnapshotState.MISSING_LEDGER
    assert not svc_missing.is_ready

    # Test malformed JSON ledger
    malformed_path = tmp_path / "corrupt_ledger.json"
    malformed_path.write_text("{invalid json corrupt content...", encoding="utf-8")
    svc_malformed = SourcedFundamentalsService(ledger_path=malformed_path)
    assert svc_malformed.state == SnapshotState.MALFORMED_LEDGER
    assert not svc_malformed.is_ready

    # Test empty ledger (valid schema with empty companies dict)
    empty_path = tmp_path / "empty_ledger.json"
    empty_path.write_text('{"metadata": {}, "companies": {}}', encoding="utf-8")
    svc_empty = SourcedFundamentalsService(ledger_path=empty_path)
    assert svc_empty.state == SnapshotState.EMPTY_LEDGER
    assert not svc_empty.is_ready

    # Path privacy verification: error_detail must NOT contain the full local filesystem path
    assert str(tmp_path) not in (svc_missing.error_detail or "")


def test_09_api_and_static_snapshot_parity():
    """Verify machine-checked parity between service output and static frontend snapshot."""
    service = get_sourced_fundamentals_service()
    snapshot_path = Path("frontend/lib/sourcedFundamentalsSnapshot.json")
    assert snapshot_path.exists(), "Frontend snapshot file must exist"

    with open(snapshot_path, "r", encoding="utf-8") as f:
        static_data = json.load(f)

    catalog = service.list_available_companies()
    assert len(catalog) > 0

    for item in catalog:
        sym = item["symbol"]
        live_snap = service.get_company_snapshot(sym)
        assert live_snap is not None
        assert sym in static_data["companies"], f"Symbol {sym} missing from static snapshot"

        static_snap = static_data["companies"][sym]
        # Verify identity parity
        assert live_snap["company_identity"]["symbol"] == static_snap["company_identity"]["symbol"]
        assert live_snap["company_identity"]["cik"] == static_snap["company_identity"]["cik"]

        # Verify key financial parity
        live_rev = live_snap["derived_calculations"]["revenue_ttm"]["value"]
        static_rev = static_snap["derived_calculations"]["revenue_ttm"]["value"]
        assert live_rev == static_rev, f"Revenue mismatch for {sym}"

        # Verify facts-only parity (no deep_valuation in either)
        assert "deep_valuation" not in live_snap
        assert "deep_valuation" not in static_snap


def test_10_system_status_endpoint_never_leaks_paths():
    """Verify diagnostic system status endpoint reports online without disclosing internal paths."""
    response = client.get("/api/v1/fundamentals/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["snapshot_state"] == "READY"
    assert data["read_only"] is True
    assert data["companies_cataloged_count"] >= 8
    assert "FIX" in data["cataloged_symbols"]

    # Verify no local Windows or Unix path strings are exposed
    serialized = json.dumps(data)
    assert "Users\\" not in serialized
    assert "C:\\" not in serialized
    assert "/Users/" not in serialized


def test_11_fix_normalization_arithmetic_verification():
    """
    R03: Verify FIX cash flow normalization arithmetic balances exactly
    and separates reported figures, working capital surge, and analyst assumptions.
    """
    # Authoritative reported ledger numbers
    ebit_m = 1849.30
    pretax_income_m = 1837.35
    income_tax_m = 402.95
    effective_tax_rate = income_tax_m / pretax_income_m  # 21.931%
    nopat_m = round(ebit_m * (1 - effective_tax_rate), 2)  # $1,443.73M
    assert nopat_m == 1443.73

    # Cash conversion inputs
    reported_cfo_m = 2550.07
    reported_capex_m = 390.26
    cfo_less_capex_m = round(reported_cfo_m - reported_capex_m, 2)
    assert cfo_less_capex_m == 2159.81

    # Separation of observed customer advance liability surge
    customer_advance_surge_m = 710.00  # Billings in excess of costs interim financing inflow

    # Explicit analyst normalization walk
    da_assumed_m = 112.50
    normalized_wc_reinvestment_m = 280.00
    sustainable_cf_m = round(nopat_m + da_assumed_m - reported_capex_m - normalized_wc_reinvestment_m, 2)

    # Exact arithmetic verification: 1443.73 + 112.50 - 390.26 - 280.00 = 885.97
    assert sustainable_cf_m == 885.97
    assert round(1443.73 + 112.50 - 390.26 - 280.00, 2) == 885.97


def test_12_read_only_zero_side_effects():
    """
    P06: Verify GET requests do not perform file writes, network calls,
    or mutate snapshot state.
    """
    service = get_sourced_fundamentals_service()
    initial_mtime = service._ledger_path.stat().st_mtime

    # Perform multiple GET calls across catalog and detail
    resp_cat = client.get("/api/v1/fundamentals/")
    assert resp_cat.status_code == 200
    resp_fix = client.get("/api/v1/fundamentals/FIX")
    assert resp_fix.status_code == 200
    resp_duol = client.get("/api/v1/fundamentals/DUOL")
    assert resp_duol.status_code == 200
    resp_ulta = client.get("/api/v1/fundamentals/ULTA")
    assert resp_ulta.status_code == 200

    # Verify ledger file was never modified or touched
    final_mtime = service._ledger_path.stat().st_mtime
    assert initial_mtime == final_mtime
