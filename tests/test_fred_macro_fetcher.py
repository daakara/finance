"""Tests for FRED Macroeconomic Fetcher and MDR Regime Classification."""

from unittest.mock import patch, MagicMock
import pytest
from analyst_dashboard.data.fred_fetcher import FredMacroFetcher


def test_fred_macro_fetcher_indicators():
    fetcher = FredMacroFetcher()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "observations": [{"value": "1.25"}, {"value": "1.20"}]
    }
    with patch("requests.get", return_value=mock_resp):
        macro = fetcher.get_macro_indicators()

        assert "yield_curve_spread" in macro
        assert "fed_funds_rate" in macro
        assert "credit_spread_oas" in macro
        assert "cpi_yoy" in macro
        assert "rating" in macro
        assert "regime" in macro
        assert 1 <= macro["rating"] <= 5
        assert isinstance(macro["yield_curve_spread"], float)
        assert isinstance(macro["fed_funds_rate"], float)


def test_fred_macro_fetcher_fallback_on_invalid_key():
    fetcher = FredMacroFetcher(api_key="INVALID_KEY_TEST")
    val = fetcher.fetch_latest_observation("T10Y2Y", default_val=0.45)
    assert val == 0.45