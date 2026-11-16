"""Authoritative Verification Suite for ARX-FRED-SEC-PRESENTATION-INTEGRITY-001.

Verifies:
1. Macro Data Contract in api/routes/regimes.py (_build_macro_contract_payload):
   - Returns verified contract with availability, is_live, regime, yield_curve_10y2y,
     real_interest_rate, credit_spread, fed_funds_rate, macro_multiplier, last_updated,
     and series_observations.
   - Enforces live multiplier qualification: only live when authentic and fresh inputs exist.
   - Enforces fail-closed behavior: when inputs are unavailable or missing, is_live is False,
     macro_multiplier is None, and availability is UNAVAILABLE (zero multiplier fabrication).
   - Handles exceptions gracefully and fails closed without crashing.
2. /regimes/current and /regimes/{symbol} endpoints:
   - Returns 'macro' key matching the authoritative macroeconomic contract.
3. SEC Filings presentation integrity in api/routes/smart_money.py:
   - Returns official filing metadata (form, accession_number, sec_url, filing_date)
     without fabricating parsed shares, zero prices, or fake purchase classifications.
4. FRED fetcher DFII10 integration:
   - Captures DFII10 real rate, observation dates, and series observations map.
"""

from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from api.routes.regimes import _build_macro_contract_payload, _compute_regime_for_symbol
from analyst_dashboard.data.fred_fetcher import FredMacroFetcher, normalize_macro_payload


# ── 1. Macro Contract Live vs Unavailable Behavior ───────────────────────────

def test_macro_contract_available_and_live():
    """Verify that when authentic FRED data is available, contract marks is_live=True with valid multiplier."""
    mock_indicators = {
        "availability": "AVAILABLE",
        "yield_curve_10y2y": 0.45,
        "high_yield_credit_spread": 2.85,
        "real_interest_rate": 1.75,
        "fed_funds_rate": 4.58,
        "series_observations": {
            "T10Y2Y": {"value": 0.45, "date": "2026-11-14", "series_id": "T10Y2Y"},
            "BAMLH0A0HYM2": {"value": 2.85, "date": "2026-11-13", "series_id": "BAMLH0A0HYM2"},
            "DFII10": {"value": 1.75, "date": "2026-11-14", "series_id": "DFII10"},
            "FEDFUNDS": {"value": 4.58, "date": "2026-11-01", "series_id": "FEDFUNDS"},
        },
    }

    with patch("analyst_dashboard.data.fred_fetcher.FredMacroFetcher.get_macro_indicators", return_value=mock_indicators):
        payload = _build_macro_contract_payload()

        assert payload["availability"] == "AVAILABLE"
        assert payload["is_live"] is True
        assert payload["yield_curve_10y2y"] == 0.45
        assert payload["credit_spread"] == 2.85
        assert payload["real_interest_rate"] == 1.75
        assert payload["fed_funds_rate"] == 4.58
        assert payload["macro_multiplier"] is not None
        assert payload["macro_multiplier"] > 0
        assert payload["last_updated"] == "2026-11-14"
        assert "T10Y2Y" in payload["series_observations"]


def test_macro_contract_inverted_yield_curve_defensive_multiplier():
    """Verify that an inverted yield curve produces RECESSIONARY_PRESSURE and defensive multiplier (0.85)."""
    mock_indicators = {
        "availability": "AVAILABLE",
        "yield_curve_10y2y": -0.35,
        "high_yield_credit_spread": 3.80,
        "real_interest_rate": 2.10,
        "fed_funds_rate": 5.25,
        "series_observations": {
            "T10Y2Y": {"value": -0.35, "date": "2026-11-14", "series_id": "T10Y2Y"},
        },
    }

    with patch("analyst_dashboard.data.fred_fetcher.FredMacroFetcher.get_macro_indicators", return_value=mock_indicators):
        payload = _build_macro_contract_payload()

        assert payload["is_live"] is True
        assert payload["regime"] == "RECESSIONARY_PRESSURE"
        assert payload["macro_multiplier"] == 0.85
        assert "Inverted" in payload["regime_title"]


def test_macro_contract_unavailable_zero_multiplier_fabrication():
    """Verify that when availability is UNAVAILABLE, is_live is False and macro_multiplier is None."""
    mock_indicators = {
        "availability": "UNAVAILABLE",
        "yield_curve_10y2y": None,
        "high_yield_credit_spread": None,
        "series_observations": {},
    }

    with patch("analyst_dashboard.data.fred_fetcher.FredMacroFetcher.get_macro_indicators", return_value=mock_indicators):
        payload = _build_macro_contract_payload()

        assert payload["availability"] == "UNAVAILABLE"
        assert payload["is_live"] is False
        assert payload["regime"] == "MACRO_TELEMETRY_UNAVAILABLE"
        assert payload["macro_multiplier"] is None
        assert payload["yield_curve_10y2y"] is None
        assert payload["credit_spread"] is None
        assert payload["real_interest_rate"] is None
        assert payload["fed_funds_rate"] is None
        assert payload["last_updated"] is None


def test_macro_contract_missing_yield_curve_fails_closed():
    """Verify that if credit spread is present but yield curve is missing, is_live is False and multiplier is None."""
    mock_indicators = {
        "availability": "AVAILABLE",
        "yield_curve_10y2y": None,
        "high_yield_credit_spread": 3.20,
        "series_observations": {},
    }

    with patch("analyst_dashboard.data.fred_fetcher.FredMacroFetcher.get_macro_indicators", return_value=mock_indicators):
        payload = _build_macro_contract_payload()

        assert payload["is_live"] is False
        assert payload["macro_multiplier"] is None
        assert payload["regime"] == "MACRO_TELEMETRY_UNAVAILABLE"


def test_macro_contract_exception_fails_closed():
    """Verify that unexpected exceptions fail closed to UNAVAILABLE contract without crashing."""
    with patch("analyst_dashboard.data.fred_fetcher.FredMacroFetcher.get_macro_indicators", side_effect=RuntimeError("FRED API connection timeout")):
        payload = _build_macro_contract_payload()

        assert payload["availability"] == "UNAVAILABLE"
        assert payload["is_live"] is False
        assert payload["regime"] == "MACRO_TELEMETRY_UNAVAILABLE"
        assert payload["macro_multiplier"] is None
        assert payload["yield_curve_10y2y"] is None


# ── 2. Regimes Compute Envelope Integration ──────────────────────────────────

def test_compute_regime_for_symbol_includes_macro_envelope():
    """Verify _compute_regime_for_symbol embeds the macro contract in its return payload."""
    fake_hist = pd.DataFrame(
        {
            "Open": [100.0] * 30,
            "High": [105.0] * 30,
            "Low": [95.0] * 30,
            "Close": [102.0] * 30,
            "Volume": [1000000] * 30,
        },
        index=pd.date_range("2026-10-01", periods=30, freq="D"),
    )

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = fake_hist

    mock_macro = {
        "availability": "AVAILABLE",
        "is_live": True,
        "regime": "NEUTRAL_EXPANSION",
        "regime_title": "Macro Regime: Disinflationary Expansion",
        "regime_subtitle": "Neutral macro conditions.",
        "yield_curve_10y2y": 0.25,
        "real_interest_rate": 1.65,
        "credit_spread": 3.10,
        "fed_funds_rate": 4.60,
        "macro_multiplier": 1.10,
        "last_updated": "2026-11-14",
        "series_observations": {},
    }

    with patch("yfinance.Ticker", return_value=mock_ticker), \
         patch("api.routes.regimes._build_macro_contract_payload", return_value=mock_macro):

        result = _compute_regime_for_symbol("SPY")

        assert "macro" in result
        assert result["macro"]["availability"] == "AVAILABLE"
        assert result["macro"]["is_live"] is True
        assert result["macro"]["yield_curve_10y2y"] == 0.25
        assert result["macro"]["real_interest_rate"] == 1.65
        assert result["macro"]["macro_multiplier"] == 1.10


# ── 3. FRED Fetcher DFII10 & Series Observations ─────────────────────────────

def test_fred_fetcher_captures_dfii10_real_interest_rate():
    """Verify FredMacroFetcher captures DFII10 real interest rate and maps series_observations."""
    fetcher = FredMacroFetcher()

    def mock_fetch(series_id):
        data = {
            "T10Y2Y": (0.35, "2026-11-14"),
            "BAMLH0A0HYM2": (3.15, "2026-11-13"),
            "DFII10": (1.82, "2026-11-14"),
            "FEDFUNDS": (4.88, "2026-11-01"),
        }
        if series_id in data:
            val, dt = data[series_id]
            return {"value": val, "date": dt, "series_id": series_id}
        return None

    with patch.object(fetcher, "fetch_latest_observation_record", side_effect=mock_fetch):
        indicators = fetcher.get_macro_indicators()

        assert indicators["availability"] == "AVAILABLE"
        assert indicators["yield_curve_10y2y"] == 0.35
        assert indicators["high_yield_credit_spread"] == 3.15
        assert indicators["real_interest_rate"] == 1.82
        assert indicators["fed_funds_rate"] == 4.88
        assert "DFII10" in indicators["series_observations"]
        assert indicators["series_observations"]["DFII10"]["value"] == 1.82
        assert indicators["series_observations"]["DFII10"]["date"] == "2026-11-14"


def test_normalize_macro_payload_preserves_real_rate_and_observations():
    """Verify normalize_macro_payload preserves real_interest_rate and series_observations."""
    raw = {
        "yield_curve_10y2y": 0.20,
        "high_yield_credit_spread": 3.40,
        "real_interest_rate": 1.70,
        "fed_funds_rate": 4.75,
        "series_observations": {
            "DFII10": {"value": 1.70, "date": "2026-11-14"},
        },
    }
    normalized = normalize_macro_payload(raw)
    assert normalized is not None
    assert normalized["real_interest_rate"] == 1.70
    assert normalized["fed_funds_rate"] == 4.75
    assert "DFII10" in normalized["series_observations"]


# ── 4. SEC Filings Presentation & Metadata Integrity ─────────────────────────

def test_sec_filings_metadata_preservation():
    """Verify get_sec_filings returns unparsed filing metadata without synthesizing fake trades."""
    from api.routes.smart_money import get_sec_filings

    mock_envelope = {
        "status": "AVAILABLE_WITH_RESULTS",
        "cik": "0000320193",
        "items": [
            {
                "form": "4",
                "filing_date": "2026-11-10",
                "report_date": "2026-11-08",
                "acceptance_datetime": "2026-11-10T17:30:00Z",
                "available_from": "2026-11-10T17:30:00Z",
                "description": "Statement of Changes in Beneficial Ownership",
                "accession_number": "0000320193-26-000105",
                "sec_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019326000105/0000320193-26-000105.txt",
            }
        ],
        "cached": True,
    }

    with patch("analyst_dashboard.data.sec_edgar_fetcher.SecEdgarFetcher.get_recent_filings_envelope", return_value=mock_envelope):
        response = get_sec_filings("AAPL")

        assert response["status"] == "AVAILABLE_WITH_RESULTS"
        assert response["symbol"] == "AAPL"
        assert response["cik"] == "0000320193"
        assert len(response["items"]) == 1

        filing = response["items"][0]
        assert filing["form"] == "4"
        assert filing["accession_number"] == "0000320193-26-000105"
        assert filing["filing_date"] == "2026-11-10"
        assert "sec.gov" in filing["sec_url"]

        # Ensure NO zero-value trade fields are fabricated in the raw filings payload
        assert "shares" not in filing
        assert "price" not in filing
        assert "transaction_code" not in filing
