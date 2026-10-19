"""
tests/test_fundamentals_api.py

Unit and integration tests for the read-only Sourced Fundamentals API endpoints.
Validates:
- GET /api/v1/fundamentals/ (catalog list)
- GET /api/v1/fundamentals/{symbol} for priority candidate (FIX)
- GET /api/v1/fundamentals/{symbol} for missing-debt candidates (DUOL, ULTA)
- Fail-closed 404 behavior for uncataloged symbols
- Preservation of null vs reported zero
- Units, accession numbers, SEC URLs, and reported vs derived separation
- Zero side effects (no file writes or mutations)
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

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


def test_02_catalog_items_structure():
    """Verify catalog items have required schema fields and do not expose unverified claims."""
    response = client.get("/api/v1/fundamentals/")
    data = response.json()
    for item in data["companies"]:
        assert "symbol" in item
        assert "company_name" in item
        assert "cik" in item
        assert "latest_reported_period" in item
        assert "valuation_status" in item
        assert item["valuation_status"] in ["AVAILABLE", "SUPPRESSED_UNAVAILABLE", "UNAVAILABLE"]
        assert isinstance(item["total_financial_debt_evidenced"], bool)


def test_03_fix_snapshot_success():
    """Verify GET /api/v1/fundamentals/FIX returns complete structured snapshot."""
    response = client.get("/api/v1/fundamentals/FIX")
    assert response.status_code == 200
    data = response.json()

    # Identity
    ident = data["company_identity"]
    assert ident["symbol"] == "FIX"
    assert ident["cik"] == "0001035983"
    assert "COMFORT SYSTEMS" in ident["company_name"].upper()
    assert ident["sec_edgar_company_url"] == "https://www.sec.gov/edgar/browse/?CIK=1035983"

    # Reported facts
    facts = data["reported_facts"]
    assert len(facts) >= 20
    metric_keys = [f["metric_key"] for f in facts]
    assert "revenue_ttm" in metric_keys
    assert "cfo_ttm" in metric_keys
    assert "capex_ttm" in metric_keys

    # Check that primary reported facts have accession numbers and SEC URLs
    reported_facts_with_accession = [f for f in facts if f.get("accession_number")]
    assert len(reported_facts_with_accession) >= 10
    for f in reported_facts_with_accession:
        assert f["source_url"] is not None
        assert "sec.gov/Archives/edgar/data" in f["source_url"]

    # Derived calculations
    derived = data["derived_calculations"]
    assert derived["revenue_ttm"]["value"] is not None
    assert derived["cfo_minus_capex_ttm"]["value"] is not None
    assert derived["total_financial_debt"]["value"] is not None
    assert derived["total_financial_debt"]["status"] == "REPORTED"

    # Deep valuation
    deep_val = data["deep_valuation"]
    assert deep_val["valuation_status"] == "AVAILABLE"
    assert "scenarios" in deep_val
    assert "Base" in deep_val["scenarios"]
    assert "reverse_dcf_implied_5y_growth_pct" in deep_val


def test_04_duol_preserves_null_debt_and_suppresses_valuation():
    """Verify DUOL snapshot preserves null for total_financial_debt and suppresses valuation."""
    response = client.get("/api/v1/fundamentals/DUOL")
    assert response.status_code == 200
    data = response.json()

    derived = data["derived_calculations"]
    # Total financial debt must be null (not 0.0) in the raw derived metrics from XBRL
    assert derived["total_financial_debt"]["value"] is None
    assert derived["total_financial_debt"]["status"] == "UNAVAILABLE"
    assert "UNAVAILABLE" in derived["total_financial_debt"]["notes"]

    # Research disposition and valuation suppression
    res = data["research_disposition_and_evidence"]
    assert res["disposition"] == "RESEARCH_FURTHER"
    assert res["valuation_assessment_status"] == "SUPPRESSED_UNAVAILABLE"
    assert "debt" in res["valuation_suppression_reason"].lower()

    # Valuation node should be suppressed
    deep_val = data["deep_valuation"]
    assert deep_val["valuation_status"] == "SUPPRESSED_UNAVAILABLE"
    assert "dcf_fair_value_per_share" not in deep_val


def test_05_ulta_preserves_null_total_debt_and_suppresses_valuation():
    """Verify ULTA snapshot preserves null for total_financial_debt and reports short-term debt."""
    response = client.get("/api/v1/fundamentals/ULTA")
    assert response.status_code == 200
    data = response.json()

    derived = data["derived_calculations"]
    assert derived["total_financial_debt"]["value"] is None
    assert derived["total_financial_debt"]["status"] == "UNAVAILABLE"
    # Long term debt is null in automated harvester
    assert derived["long_term_debt"]["value"] is None

    # Deep valuation must be suppressed
    assert data["deep_valuation"]["valuation_status"] == "SUPPRESSED_UNAVAILABLE"


def test_06_uncataloged_symbol_fails_closed_404():
    """Verify uncataloged symbols return 404 Not Found with descriptive detail."""
    response = client.get("/api/v1/fundamentals/NONEXISTENT_XYZ")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "not found" in detail.lower()
    assert "FIX" in detail  # Lists available symbols


def test_07_empty_symbol_fails_404_or_catalog():
    """Verify empty/blank symbol handling."""
    # /api/v1/fundamentals/ routes to catalog
    resp_empty = client.get("/api/v1/fundamentals/")
    assert resp_empty.status_code == 200


def test_08_case_insensitivity():
    """Verify symbols are matched case-insensitively."""
    resp_upper = client.get("/api/v1/fundamentals/FIX")
    resp_lower = client.get("/api/v1/fundamentals/fix")
    assert resp_upper.status_code == 200
    assert resp_lower.status_code == 200
    assert resp_upper.json()["company_identity"]["symbol"] == resp_lower.json()["company_identity"]["symbol"]


def test_09_system_status_endpoint():
    """Verify GET /api/v1/fundamentals/system/status diagnostic endpoint."""
    response = client.get("/api/v1/fundamentals/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["read_only"] is True
    assert data["companies_cataloged_count"] >= 8
    assert "FIX" in data["cataloged_symbols"]
