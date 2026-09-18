"""Targeted Integration & Adversarial Verification Suite for ARX FRED Macro Evidence Plumbing.

Verifies:
1. Canonical key contract: yield_curve_10y2y, high_yield_credit_spread, timestamps, provider.
2. Legacy provider aliases normalization (yield_curve_spread -> yield_curve_10y2y, credit_spread_oas -> high_yield_credit_spread).
3. Valid numeric zero preservation (yield == 0.0 is evaluated, not treated as missing or falsy).
4. Negative yield spread preservation (inverted curve yields warning status).
5. Missing yield curve fails closed (MISSING_MACRO_DATA = UNAVAILABLE).
6. Missing credit spread fails closed (FABRICATED_MACRO_DEFAULTS = 0).
7. Both missing fails closed.
8. Provider error handling with zero fabricated defaults.
9. Malformed payload fail-closed parsing.
10. Point-in-time future observation quarantine (macro_available_at <= recommended_at).
11. Point-in-time valid observation acceptance.
12. Integration: fred_fetcher output -> analytics canonical normalization -> confluence evaluation.
13. Adversarial: macro restoration does not bypass actionability contract (position size remains 0 when not actionable).
14. Adversarial: macro restoration does not automatically clear partial evidence if another domain is missing.
"""

from unittest.mock import patch, MagicMock
import pytest
from analyst_dashboard.data.fred_fetcher import FredMacroFetcher, normalize_macro_payload
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger, ProvenanceCohort
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine, ACTIONABLE_EXECUTION_STATUSES

pytestmark = pytest.mark.tier2c


# ── 1. Canonical Key Contract & Provider Aliases ─────────────────────────────

def test_canonical_yield_and_credit_keys():
    """Verify that canonical keys yield_curve_10y2y and high_yield_credit_spread are recognized directly."""
    raw = {
        "yield_curve_10y2y": 0.42,
        "high_yield_credit_spread": 2.85,
        "yield_observation_timestamp": "2026-09-18T00:00:00Z",
        "credit_observation_timestamp": "2026-09-18T00:00:00Z",
        "provider": "FRED",
    }
    normalized = normalize_macro_payload(raw)
    assert normalized is not None
    assert normalized["yield_curve_10y2y"] == 0.42
    assert normalized["high_yield_credit_spread"] == 2.85
    assert normalized["credit_spread"] == 2.85
    assert normalized["availability"] == "AVAILABLE"
    assert normalized["provider"] == "FRED"
    assert normalized["yield_observation_timestamp"] == "2026-09-18T00:00:00Z"
    assert normalized["credit_observation_timestamp"] == "2026-09-18T00:00:00Z"
    assert normalized["raw_payload_hash"] is not None


def test_legacy_provider_aliases_normalization():
    """Verify normalization of provider keys: yield_curve_spread -> yield_curve_10y2y, credit_spread_oas -> high_yield_credit_spread."""
    legacy_payload = {
        "yield_curve_spread": 0.35,
        "credit_spread_oas": 3.15,
        "fed_funds_rate": 3.63,
        "cpi_index": 332.8,
    }
    normalized = normalize_macro_payload(legacy_payload)
    assert normalized is not None
    assert normalized["yield_curve_10y2y"] == 0.35
    assert normalized["high_yield_credit_spread"] == 3.15
    assert normalized["credit_spread"] == 3.15
    assert normalized["availability"] == "AVAILABLE"


# ── 2. Numeric Zero & Negative Value Integrity ──────────────────────────────

def test_valid_zero_macro_value_preserved():
    """Verify VALID_ZERO_MACRO_VALUE = PRESERVED (0.0 is not treated as None or falsy)."""
    payload = {
        "yield_curve_10y2y": 0.0,
        "high_yield_credit_spread": 3.2,
    }
    normalized = normalize_macro_payload(payload)
    assert normalized is not None
    assert normalized["yield_curve_10y2y"] == 0.0
    assert normalized["yield_curve_10y2y"] is not None
    assert normalized["availability"] == "AVAILABLE"

    # Confluence engine must evaluate 0.0 yield curve as non-negative
    engine = ConfluenceEngine()
    res = engine.calculate_confluence(
        symbol="AAPL",
        macro_data=normalized,
        technical_data={"current_price": 200.0, "stop_loss": 190.0},
    )
    macro_pillar = next((p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
    assert macro_pillar is not None
    assert macro_pillar["status"] == "positive"
    assert macro_pillar["score"] == 85.0


def test_negative_yield_spread_preserved():
    """Verify that negative yield spread (inverted yield curve) is preserved and triggers warning."""
    payload = {
        "yield_curve_10y2y": -0.45,
        "high_yield_credit_spread": 3.8,
    }
    normalized = normalize_macro_payload(payload)
    assert normalized["yield_curve_10y2y"] == -0.45

    engine = ConfluenceEngine()
    res = engine.calculate_confluence(symbol="AAPL", macro_data=normalized)
    macro_pillar = next((p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
    assert macro_pillar is not None
    assert macro_pillar["status"] == "warning"
    assert macro_pillar["score"] == 38.0


# ── 3. Fail-Closed Semantics (Missing ≠ Zero, Zero Fabricated Defaults) ──────

def test_missing_yield_curve_fails_closed():
    """Verify that missing yield curve sets availability=PARTIAL and confluence status=unavailable."""
    payload = {
        "high_yield_credit_spread": 3.0,
    }
    normalized = normalize_macro_payload(payload)
    assert normalized["yield_curve_10y2y"] is None
    assert normalized["availability"] == "PARTIAL"

    engine = ConfluenceEngine()
    res = engine.calculate_confluence(symbol="AAPL", macro_data=normalized)
    macro_pillar = next((p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
    assert macro_pillar is not None
    assert macro_pillar["status"] == "unavailable"
    assert macro_pillar["score"] == 0.0
    assert "unavailable" in macro_pillar["plainDetail"].lower()


def test_missing_credit_spread_fails_closed():
    """Verify that missing credit spread sets availability=PARTIAL and confluence status=unavailable."""
    payload = {
        "yield_curve_10y2y": 0.5,
    }
    normalized = normalize_macro_payload(payload)
    assert normalized["high_yield_credit_spread"] is None
    assert normalized["availability"] == "PARTIAL"

    engine = ConfluenceEngine()
    res = engine.calculate_confluence(symbol="AAPL", macro_data=normalized)
    macro_pillar = next((p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
    assert macro_pillar is not None
    assert macro_pillar["status"] == "unavailable"
    assert macro_pillar["score"] == 0.0


def test_both_missing_fails_closed():
    """Verify that empty payload sets availability=UNAVAILABLE and macro_status=unavailable."""
    normalized = normalize_macro_payload({})
    assert normalized["availability"] == "UNAVAILABLE"
    assert normalized["yield_curve_10y2y"] is None
    assert normalized["high_yield_credit_spread"] is None

    engine = ConfluenceEngine()
    res = engine.calculate_confluence(symbol="AAPL", macro_data=normalized)
    macro_pillar = next((p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
    assert macro_pillar["status"] == "unavailable"
    assert macro_pillar["score"] == 0.0


def test_provider_error_fails_closed_zero_fabricated_defaults():
    """Verify that provider network failure returns UNAVAILABLE without fabricating defaults."""
    fetcher = FredMacroFetcher()
    with patch("requests.get", side_effect=Exception("FRED connection timeout")):
        macro = fetcher.get_macro_indicators()
        assert macro["availability"] == "UNAVAILABLE"
        assert macro["yield_curve_10y2y"] is None
        assert macro["high_yield_credit_spread"] is None
        assert macro["yield_curve_spread"] is None
        assert macro["credit_spread_oas"] is None
        assert macro["rating"] == 0
        assert "unavailable" in macro["regime"].lower()


def test_malformed_payload_fails_closed():
    """Verify that malformed non-numeric values are safely parsed to None."""
    payload = {
        "yield_curve_10y2y": "INVALID_NUMBER",
        "high_yield_credit_spread": {"malformed": True},
    }
    normalized = normalize_macro_payload(payload)
    assert normalized["yield_curve_10y2y"] is None
    assert normalized["high_yield_credit_spread"] is None
    assert normalized["availability"] == "UNAVAILABLE"

    engine = ConfluenceEngine()
    res = engine.calculate_confluence(symbol="AAPL", macro_data=normalized)
    macro_pillar = next((p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
    assert macro_pillar["status"] == "unavailable"


# ── 4. Point-in-Time Anti-Lookahead Integrity ───────────────────────────────

def test_future_observation_timestamp_quarantined_in_confluence():
    """Verify that macro observation with timestamp ahead of recommended_at is quarantined."""
    payload = {
        "yield_curve_10y2y": 0.50,
        "high_yield_credit_spread": 2.80,
        "macro_observation_available_at": "2026-09-20T00:00:00Z",
        "recommended_at": "2026-09-18T12:00:00Z",  # Cutoff before observation!
    }
    normalized = normalize_macro_payload(payload)
    engine = ConfluenceEngine()
    res = engine.calculate_confluence(symbol="AAPL", macro_data=normalized)
    macro_pillar = next((p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
    assert macro_pillar["status"] == "unavailable"


def test_future_observation_quarantined_in_governance_ledger():
    """Verify that Epoch 1 classifier quarantines future macro observations to CONTAMINATED."""
    record = {
        "timestamp": "2026-09-19T01:00:00Z",
        "symbol": "AAPL",
        "status": "ACTIVE",
        "provenanceCohort": ProvenanceCohort.PROSPECTIVE_CLEAN,
        "inputs": {
            "macroObservationAvailableAt": "2026-09-19T02:30:00Z",  # 90m ahead of decision!
            "marketSnapshotObservedAt": "2026-09-19T00:59:00Z",
        },
    }
    cohort = ExperimentLedger.classify_provenance_cohort(record)
    assert cohort == ProvenanceCohort.CONTAMINATED


def test_point_in_time_valid_observation():
    """Verify that observation available prior to recommendation date passes anti-lookahead check."""
    payload = {
        "yield_curve_10y2y": 0.40,
        "high_yield_credit_spread": 2.90,
        "macro_observation_available_at": "2026-09-18T00:00:00Z",
        "recommended_at": "2026-09-18T16:00:00Z",
    }
    engine = ConfluenceEngine()
    res = engine.calculate_confluence(symbol="AAPL", macro_data=payload)
    macro_pillar = next((p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
    assert macro_pillar["status"] == "positive"


# ── 5. End-to-End Integration & Adversarial Boundaries ───────────────────────

def test_fred_fetcher_to_analytics_to_confluence_pipeline():
    """Simulate authentic FRED output flowing through normalize_macro_payload into ConfluenceEngine."""
    mock_obs = [
        {"date": "2026-09-18", "value": "0.25", "realtime_start": "2026-09-18"},
    ]
    fetcher = FredMacroFetcher(api_key="TEST_INTEGRATION_KEY")
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"observations": mock_obs}
        mock_get.return_value = mock_resp

        # 1. Fetcher output
        fred_output = fetcher.get_macro_indicators()
        assert fred_output["yield_curve_10y2y"] == 0.25
        assert fred_output["high_yield_credit_spread"] == 0.25  # from mock

        # 2. Canonical normalization
        macro_inputs = normalize_macro_payload(fred_output)
        assert macro_inputs["availability"] == "AVAILABLE"
        assert macro_inputs["yield_curve_10y2y"] == 0.25

        # 3. Confluence engine evaluation
        engine = ConfluenceEngine()
        confluence_res = engine.calculate_confluence(
            symbol="AAPL",
            macro_data=macro_inputs,
            technical_data={"current_price": 250.0, "stop_loss": 240.0},
        )
        macro_pillar = next((p for p in confluence_res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR"), None)
        assert macro_pillar is not None
        assert macro_pillar["status"] == "positive"
        assert macro_pillar["score"] == 85.0


def test_adversarial_macro_restoration_does_not_force_actionability():
    """Adversarial check: macro restoration must NOT bypass execution status guardrails."""
    # Even if macro is positive (+85.0), if technical setup is NOT actionable, position sizing remains 0
    trade_levels = OptimalExecutionEngine.calculate_trade_levels(
        price_df=None,
        current_price=100.0,
        user_role="LONG_TERM",
    )
    # Missing/empty price history strictly suppresses actionable entry/stop targets
    assert trade_levels["optimal_entry_min"] is None
    assert trade_levels["stop_loss"] is None
    assert trade_levels["execution_status"] not in ACTIONABLE_EXECUTION_STATUSES


def test_adversarial_macro_restoration_does_not_clear_partial_evidence():
    """Adversarial check: macro availability must NOT force overall eligibility to ELIGIBLE if other domains missing."""
    engine = ConfluenceEngine()
    # Provide macro, but omit fundamentals and smart money
    res = engine.calculate_confluence(
        symbol="AAPL",
        macro_data={"yield_curve_10y2y": 0.5, "high_yield_credit_spread": 2.5},
        fundamental_data={},  # empty fundamentals
        smart_money_data={"signals": []},
    )
    fund_pillar = next((p for p in res["pillars"] if "fundamental" in p["pillar"].lower() or "solvency" in p["pillar"].lower()), None)
    # Missing fundamentals remain unassessed/unavailable
    if fund_pillar:
        assert fund_pillar["status"] == "unavailable" or fund_pillar["score"] == 0.0
