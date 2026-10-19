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

from analyst_dashboard.research.sourced_fundamentals.identity import (
    verify_company_identity,
    validate_cik_format,
    ATTRIBUTABLE_SEC_IDENTITY_AUTHORITY,
)
from analyst_dashboard.research.sourced_fundamentals.normalization import (
    calculate_fix_normalized_cash_flow,
    NormalizationAssumptions,
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
    facts_by_key = {f["metric_key"]: f for f in facts}
    assert "revenue_ttm" in facts_by_key
    assert "cfo_ttm" in facts_by_key
    assert "capex_ttm" in facts_by_key
    assert "cash_and_equivalents" in facts_by_key

    # F02: Check fact status preservation
    assert facts_by_key["cash_and_equivalents"]["status"] == "REPORTED"
    assert facts_by_key["cash_and_equivalents"]["fact_type"] == "PRIMARY_REPORTED"
    assert facts_by_key["cash_and_equivalents"]["source_url"] is not None

    assert facts_by_key["revenue_ttm"]["status"] == "DERIVED"
    assert facts_by_key["revenue_ttm"]["fact_type"] == "DERIVED_CALCULATION"
    assert facts_by_key["revenue_ttm"]["source_url"] is not None

    reported_facts_with_accession = [f for f in facts if f.get("accession_number")]
    assert len(reported_facts_with_accession) >= 10
    for f in reported_facts_with_accession:
        assert f["source_url"] is not None
        assert "sec.gov/Archives/edgar/data" in f["source_url"]
        assert f["status"] in ["REPORTED", "DERIVED"]

    # F03: Derived calculations
    derived = data["derived_calculations"]
    assert derived["revenue_ttm"]["value"] is not None
    assert derived["revenue_ttm"]["units"] == "USD"
    assert derived["cfo_minus_capex_ttm"]["value"] is not None
    assert derived["cfo_minus_capex_ttm"]["units"] == "USD"

    # FIX total debt is derived from ST ($0.22M) + LT ($53.85M) = $54.06M debt
    assert derived["short_term_debt"]["value"] == 215000.0
    assert derived["long_term_debt"]["value"] == 53849000.0
    assert derived["total_financial_debt"]["value"] == 54064000.0
    assert derived["total_financial_debt"]["units"] == "USD"
    assert derived["total_financial_debt"]["status"] == "DERIVED"
    assert "Short-Term Debt" in derived["total_financial_debt"]["formula"]
    assert "Long-Term Debt" in derived["total_financial_debt"]["formula"]

    # Operating leases labeled as noncurrent
    assert derived["operating_lease_liabilities"]["label"] == "Operating Lease Liabilities (Noncurrent)"
    assert derived["operating_lease_liabilities"]["status"] == "REPORTED"

    # Diluted shares expose distinct period_start and period_end ISO dates
    shares = derived["diluted_shares_weighted_avg"]
    assert shares["value"] == 35254000.0
    assert shares["units"] == "shares"
    assert shares["period_start"] == "2026-04-01"
    assert shares["period_end"] == "2026-06-30"
    assert shares["period_type"] == "QUARTERLY_WEIGHTED_AVERAGE"

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
    """Verify identity validation, CIK format checking, attributable binding checks, and route 500 handling on corruption."""
    # 1. Case insensitivity and whitespace trimming
    resp_trimmed = client.get("/api/v1/fundamentals/  fix  ")
    assert resp_trimmed.status_code == 200
    assert resp_trimmed.json()["company_identity"]["symbol"] == "FIX"

    # 2. Empty symbol
    resp_empty = client.get("/api/v1/fundamentals/%20")
    assert resp_empty.status_code in (400, 404)

    # 3. Format validation
    assert validate_cik_format("0001035983") is True
    assert validate_cik_format("1035983") is False  # Missing leading zeros / not 10 digits
    assert validate_cik_format("NOT_A_CIK") is False
    assert validate_cik_format("000103598X") is False
    assert validate_cik_format("") is False
    assert validate_cik_format(None) is False

    # 4. Attributable identity authority checks
    res_valid = verify_company_identity("FIX", "FIX", "0001035983")
    assert res_valid.is_valid is True
    assert res_valid.error is None
    assert res_valid.binding.company_name == "COMFORT SYSTEMS USA INC"

    # Mismatched CIK (e.g. FIX bound to Apple CIK)
    res_mismatch = verify_company_identity("FIX", "FIX", "0000320193")
    assert res_mismatch.is_valid is False
    assert "mismatch" in res_mismatch.error.lower()

    # Mismatched record symbol
    res_sym_mismatch = verify_company_identity("FIX", "AAPL", "0001035983")
    assert res_sym_mismatch.is_valid is False
    assert "symbol mismatch" in res_sym_mismatch.error.lower()

    # Malformed CIK in record
    res_malformed = verify_company_identity("FIX", "FIX", "NOT_A_CIK")
    assert res_malformed.is_valid is False
    assert "malformed cik" in res_malformed.error.lower()

    # Uncataloged symbol
    res_uncataloged = verify_company_identity("NONEXISTENT", "NONEXISTENT", "0001035983")
    assert res_uncataloged.is_valid is False
    assert "not cataloged" in res_uncataloged.error.lower()

    # 5. Service level identity validation
    service = get_sourced_fundamentals_service()
    assert service.validate_symbol_identity("FIX", {"fundamentals": {"symbol": "FIX", "cik": "0001035983"}})
    assert not service.validate_symbol_identity("FIX", {"fundamentals": {"symbol": "AAPL", "cik": "0000320193"}})
    assert not service.validate_symbol_identity("FIX", {"fundamentals": {"symbol": "FIX", "cik": "0000320193"}})
    assert not service.validate_symbol_identity("FIX", {"fundamentals": {"symbol": "FIX", "cik": None}})
    assert not service.validate_symbol_identity("", {"fundamentals": {"symbol": "FIX", "cik": "0001035983"}})

    # 6. Route-level handling of internal identity violation returns 500 (not 404)
    original_data = service._data
    try:
        corrupted_data = json.loads(json.dumps(original_data))
        # Corrupt FIX's CIK to Apple's CIK
        corrupted_data["companies"]["FIX"]["fundamentals"]["cik"] = "0000320193"
        service._data = corrupted_data

        resp_corrupt = client.get("/api/v1/fundamentals/FIX")
        assert resp_corrupt.status_code == 500
        assert "Internal identity integrity violation" in resp_corrupt.json()["detail"]
    finally:
        service._data = original_data


def test_08_missing_and_malformed_ledger_handling(tmp_path):
    """Verify service returns explicit 503 on missing ledger and 500 on malformed ledger without path leak."""
    from unittest.mock import patch

    # 1. Test missing ledger
    missing_path = tmp_path / "nonexistent_ledger.json"
    svc_missing = SourcedFundamentalsService(ledger_path=missing_path)
    assert svc_missing.state == SnapshotState.MISSING_LEDGER
    assert not svc_missing.is_ready

    # 2. Test malformed JSON ledger
    malformed_path = tmp_path / "corrupt_ledger.json"
    malformed_path.write_text("{invalid json corrupt content...", encoding="utf-8")
    svc_malformed = SourcedFundamentalsService(ledger_path=malformed_path)
    assert svc_malformed.state == SnapshotState.MALFORMED_LEDGER
    assert not svc_malformed.is_ready

    # 3. Test empty ledger (valid schema with empty companies dict)
    empty_path = tmp_path / "empty_ledger.json"
    empty_path.write_text('{"metadata": {}, "companies": {}}', encoding="utf-8")
    svc_empty = SourcedFundamentalsService(ledger_path=empty_path)
    assert svc_empty.state == SnapshotState.EMPTY_LEDGER
    assert not svc_empty.is_ready

    # 4. Null metadata (F04)
    null_meta_path = tmp_path / "null_meta.json"
    null_meta_path.write_text('{"metadata": null, "companies": {"FIX": {}}}', encoding="utf-8")
    svc_null_meta = SourcedFundamentalsService(ledger_path=null_meta_path)
    assert svc_null_meta.state == SnapshotState.MALFORMED_LEDGER
    assert not svc_null_meta.is_ready

    # 5. Null company record (F04)
    null_comp_path = tmp_path / "null_company.json"
    null_comp_path.write_text('{"metadata": {}, "companies": {"FIX": null}}', encoding="utf-8")
    svc_null_comp = SourcedFundamentalsService(ledger_path=null_comp_path)
    assert svc_null_comp.state == SnapshotState.MALFORMED_LEDGER
    assert not svc_null_comp.is_ready

    # 6. Missing fundamentals object (F04)
    missing_fund_path = tmp_path / "missing_fundamentals.json"
    missing_fund_path.write_text('{"metadata": {}, "companies": {"FIX": {"derived_metrics": {}}}}', encoding="utf-8")
    svc_missing_fund = SourcedFundamentalsService(ledger_path=missing_fund_path)
    assert svc_missing_fund.state == SnapshotState.MALFORMED_LEDGER
    assert not svc_missing_fund.is_ready

    # 7. Path privacy verification: error_detail must NOT contain the full local filesystem path
    assert str(tmp_path) not in (svc_missing.error_detail or "")
    assert str(tmp_path) not in (svc_malformed.error_detail or "")

    # 8. Route-level verification for missing and malformed services (F04)
    with patch("api.routes.fundamentals.get_sourced_fundamentals_service", return_value=svc_missing):
        resp_m_cat = client.get("/api/v1/fundamentals/")
        assert resp_m_cat.status_code == 503
        assert "dataset is missing" in resp_m_cat.json()["detail"]
        assert str(tmp_path) not in resp_m_cat.json()["detail"]

        resp_m_item = client.get("/api/v1/fundamentals/FIX")
        assert resp_m_item.status_code == 503

    with patch("api.routes.fundamentals.get_sourced_fundamentals_service", return_value=svc_malformed):
        resp_f_cat = client.get("/api/v1/fundamentals/")
        assert resp_f_cat.status_code == 500
        assert "dataset is invalid" in resp_f_cat.json()["detail"]
        assert str(tmp_path) not in resp_f_cat.json()["detail"]

        resp_f_item = client.get("/api/v1/fundamentals/FIX")
        assert resp_f_item.status_code == 500

    with patch("api.routes.fundamentals.get_sourced_fundamentals_service", return_value=svc_empty):
        resp_e_cat = client.get("/api/v1/fundamentals/")
        assert resp_e_cat.status_code == 500

        resp_e_item = client.get("/api/v1/fundamentals/FIX")
        assert resp_e_item.status_code == 500


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
    R03 & F06: Verify FIX cash flow normalization arithmetic balances exactly,
    separates reported figures, working capital surge, and analyst assumptions,
    and dynamically alters output when input parameters/assumptions change.
    """
    # Authoritative reported ledger numbers
    ebit_raw = 1849302000.0  # $1,849.30M
    pretax_income_raw = 1837348000.0  # $1,837.35M
    income_tax_raw = 402949000.0  # $402.95M
    effective_tax_rate_pct = (income_tax_raw / pretax_income_raw) * 100.0  # 21.93106%
    capex_raw = 390259000.0  # $390.26M
    reported_cfo_raw = 2550065000.0  # $2,550.07M

    # 1. Base model run through normalization engine
    result = calculate_fix_normalized_cash_flow(
        ebit_raw=ebit_raw,
        effective_tax_rate_pct=effective_tax_rate_pct,
        capex_raw=capex_raw,
        reported_cfo_raw=reported_cfo_raw,
    )

    norm_walk = result["normalization_walk"]
    assert norm_walk["nopat"] == 1443.73
    assert norm_walk["plus_da"] == 112.50
    assert norm_walk["minus_capex"] == 390.26
    assert norm_walk["minus_normalized_wc_reinvestment"] == 280.00
    # Exact arithmetic verification: 1443.73 + 112.50 - 390.26 - 280.00 = 885.97
    assert norm_walk["sustainable_cash_flow_millions"] == 885.97

    # 2. Dynamic behavior verification (F06): altering assumptions changes output
    higher_reinvest = calculate_fix_normalized_cash_flow(
        ebit_raw=ebit_raw,
        effective_tax_rate_pct=effective_tax_rate_pct,
        capex_raw=capex_raw,
        reported_cfo_raw=reported_cfo_raw,
        assumptions=NormalizationAssumptions(normalized_wc_reinvestment_millions=350.00),
    )
    assert higher_reinvest["normalization_walk"]["sustainable_cash_flow_millions"] == 815.97
    assert higher_reinvest["normalization_walk"]["sustainable_cash_flow_millions"] != norm_walk["sustainable_cash_flow_millions"]

    # 3. Dynamic behavior on input alteration (lower EBIT)
    lower_ebit = calculate_fix_normalized_cash_flow(
        ebit_raw=1500000000.0,
        effective_tax_rate_pct=effective_tax_rate_pct,
        capex_raw=capex_raw,
        reported_cfo_raw=reported_cfo_raw,
    )
    assert lower_ebit["normalization_walk"]["sustainable_cash_flow_millions"] < norm_walk["sustainable_cash_flow_millions"]


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


def test_13_route_read_only_side_effects_traps():
    """
    P06 & F07: Comprehensive spy/trap verification proving GET endpoints
    perform zero disk writes and zero external network calls.
    """
    from unittest.mock import patch
    import builtins
    import urllib.request
    import http.client

    # Trap open calls that attempt to write
    original_open = builtins.open
    write_traps = []

    def trapped_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ["w", "a", "x", "+"]):
            write_traps.append((str(file), mode))
            raise IOError(f"Write violation during read-only operation: open({file}, {mode})")
        return original_open(file, mode, *args, **kwargs)

    # Trap HTTP/HTTPS connections
    network_traps = []

    def trapped_http_connect(*args, **kwargs):
        network_traps.append(("http", args))
        raise IOError("Outbound HTTP network violation during read-only operation")

    def trapped_https_connect(*args, **kwargs):
        network_traps.append(("https", args))
        raise IOError("Outbound HTTPS network violation during read-only operation")

    with patch("builtins.open", side_effect=trapped_open), \
         patch.object(http.client.HTTPConnection, "connect", side_effect=trapped_http_connect), \
         patch.object(http.client.HTTPSConnection, "connect", side_effect=trapped_https_connect):

        # Test catalog
        resp_cat = client.get("/api/v1/fundamentals/")
        assert resp_cat.status_code == 200

        # Test individual assets
        for sym in ["FIX", "DUOL", "ULTA"]:
            resp = client.get(f"/api/v1/fundamentals/{sym}")
            assert resp.status_code == 200

        # Test fallback uncataloged
        resp_404 = client.get("/api/v1/fundamentals/NONEXISTENT")
        assert resp_404.status_code == 404

    assert len(write_traps) == 0, f"Write operations attempted: {write_traps}"
    assert len(network_traps) == 0, f"Network operations attempted: {network_traps}"
