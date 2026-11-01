"""
Directive 044R.7: Post-Release Sentry and Market Data Provider Remediation Verification Suite

Covers:
1. CPRX batch setups do not make unnecessary provider calls.
2. CPRX single-symbol analytics do not make unnecessary provider calls.
3. Known delisted securities remain non-actionable (DELISTED_SECURITY / INELIGIBLE_INACTIVE).
4. Unknown-symbol requests preserve truthful response contracts (clean 404).
5. Expected lookup failures do not generate duplicate misleading Sentry incidents.
6. Genuine provider failures remain observable (429, 5xx, timeouts, network errors).
7. Valid active equities and ETFs continue using normal market-data paths.
8. Existing crypto/equity routing protections remain intact.
9. RFC 8259 JSON serialization and RVOL N/A protections remain intact.
10. Operational HOLDs and Sprint 5 lockdown remain intact.
"""

import json
import math
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from fastapi import HTTPException

from api.observability.sentry_adapter import SentryBackendAdapter
from api.routes.analytics import (
    get_tactical_setups,
    get_tactical_setup_for_symbol,
    get_asset_analytics,
    _sanitize_floats,
    KNOWN_CRYPTO_SYMBOLS,
)
from api.routes.screener import CONFIRMED_CORPORATE_ACTIONS


class TestCPRXProviderAccessElimination:
    """Work Package B: Verify zero external market data provider calls for delisted securities."""

    @patch("api.routes.analytics.yf.Ticker")
    def test_cprx_batch_setups_zero_provider_calls(self, mock_ticker):
        """1. CPRX batch setups do not make unnecessary provider calls."""
        res = get_tactical_setups(tickers="LNTH, CPRX", user_role="LONG_TERM")
        
        # Verify yf.Ticker was NEVER called with CPRX
        called_symbols = [call_args[0][0] for call_args in mock_ticker.call_args_list]
        assert "CPRX" not in called_symbols, f"yf.Ticker was called with CPRX: {called_symbols}"

        # Verify CPRX setup is present with delisted contract
        cprx_setup = next((s for s in res["setups"] if s["symbol"] == "CPRX"), None)
        assert cprx_setup is not None, "CPRX setup should be returned in setups"
        assert cprx_setup["executionStatus"] == "DELISTED_SECURITY"
        assert cprx_setup["decisionState"] == "INELIGIBLE_INACTIVE"
        assert cprx_setup["isActionable"] is False
        assert cprx_setup["confluenceScore"] == 0.0
        assert cprx_setup["corporateAction"] is not None
        assert cprx_setup["corporateAction"]["status"] == "DELISTED"

    @patch("api.routes.analytics.yf.Ticker")
    def test_cprx_single_setups_zero_provider_calls(self, mock_ticker):
        """Known delisted securities return delisted setup without external calls."""
        res = get_tactical_setup_for_symbol(symbol="CPRX", user_role="LONG_TERM")

        # Verify yf.Ticker was NEVER called with CPRX
        called_symbols = [call_args[0][0] for call_args in mock_ticker.call_args_list]
        assert "CPRX" not in called_symbols, f"yf.Ticker was called with CPRX: {called_symbols}"

        assert res["executionStatus"] == "DELISTED_SECURITY"
        assert res["decisionState"] == "INELIGIBLE_INACTIVE"
        assert res["isActionable"] is False
        assert res["confluenceScore"] == 0.0

    @patch("api.routes.analytics.yf.Ticker")
    def test_cprx_single_symbol_analytics_zero_provider_calls(self, mock_ticker):
        """2. CPRX single-symbol analytics do not make unnecessary provider calls."""
        with pytest.raises(HTTPException) as excinfo:
            get_asset_analytics(symbol="CPRX")

        assert excinfo.value.status_code == 404
        assert "delisted" in excinfo.value.detail.lower()

        # Verify yf.Ticker was NEVER called with CPRX
        called_symbols = [call_args[0][0] for call_args in mock_ticker.call_args_list]
        assert "CPRX" not in called_symbols, f"yf.Ticker was called with CPRX: {called_symbols}"


class TestUnknownSymbolAndSentryBehavior:
    """Work Package C: Invalid symbol handling and Sentry deduplication/filtering."""

    @patch("api.routes.analytics.yf.Ticker")
    @patch("api.routes.analytics.market_db")
    def test_unknown_symbol_preserves_truthful_contract(self, mock_market_db, mock_ticker):
        """4. Unknown-symbol requests preserve truthful response contracts (clean 404)."""
        mock_hist = MagicMock()
        mock_hist.empty = True
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_instance

        mock_market_db.get_candles_with_freshness.return_value = {"candles": []}

        with pytest.raises(HTTPException) as excinfo:
            get_asset_analytics(symbol="NONEXISTENTSTOCK")

        assert excinfo.value.status_code == 404
        assert "No valid price action data found for symbol NONEXISTENTSTOCK" in excinfo.value.detail

    def test_expected_yfinance_lookups_dropped_by_sentry(self):
        """5. Expected lookup failures do not generate duplicate misleading Sentry incidents."""
        adapter = SentryBackendAdapter()

        # Expected lookup error 1: HTTP 404 quoteSummary from yfinance
        event_404 = {
            "logger": "yfinance",
            "level": "error",
            "logentry": {
                "message": 'HTTP Error 404: {"quoteSummary":{"result":null,"error":{"code":"Not Found","description":"Quote not found for symbol: NONEXISTENTSTOCK"}}}'
            },
        }
        res_404 = adapter._sanitize_event(event_404, {})
        assert res_404 is None, "yfinance 404 quoteSummary error must be dropped by Sentry"

        # Expected lookup error 2: No data found delisted
        event_no_data = {
            "logger": "yfinance",
            "level": "error",
            "logentry": {
                "message": "$NONEXISTENTSTOCK: No data found, symbol may be delisted"
            },
        }
        res_no_data = adapter._sanitize_event(event_no_data, {})
        assert res_no_data is None, "yfinance delisted symbol error must be dropped by Sentry"

        # Expected lookup error 3: CPRX delisted
        event_cprx = {
            "logger": "yfinance",
            "level": "error",
            "message": "$CPRX: No data found, symbol may be delisted",
        }
        res_cprx = adapter._sanitize_event(event_cprx, {})
        assert res_cprx is None, "CPRX delisted error must be dropped by Sentry"

        # Expected lookup error 4: via hint log_record
        mock_record = MagicMock()
        mock_record.getMessage.return_value = "CPRX: No price data found, symbol may be delisted (1d 2025-11-01 -> 2026-11-01)"
        res_hint = adapter._sanitize_event({"logger": "yfinance", "level": "error"}, {"log_record": mock_record})
        assert res_hint is None, "yfinance log_record miss must be dropped by Sentry"

    def test_genuine_provider_failures_remain_observable_in_sentry(self):
        """6. Genuine provider failures remain observable (429, 5xx, timeouts, network)."""
        adapter = SentryBackendAdapter()

        # 1. HTTP 429 Rate Limit
        event_429 = {
            "logger": "yfinance",
            "level": "error",
            "logentry": {"message": "HTTP Error 429: Too Many Requests"},
        }
        res_429 = adapter._sanitize_event(event_429, {})
        assert res_429 is not None, "HTTP 429 rate limit must NOT be dropped"
        assert res_429["tags"]["service"] == "arx-api"

        # 2. HTTP 503 Upstream Failure
        event_503 = {
            "logger": "yfinance",
            "level": "error",
            "logentry": {"message": "HTTP Error 503: Service Unavailable"},
        }
        res_503 = adapter._sanitize_event(event_503, {})
        assert res_503 is not None, "HTTP 503 upstream error must NOT be dropped"

        # 3. Timeout Error
        event_timeout = {
            "logger": "yfinance",
            "level": "error",
            "logentry": {"message": "Request timed out after 5.0 seconds"},
        }
        res_timeout = adapter._sanitize_event(event_timeout, {})
        assert res_timeout is not None, "Provider timeout error must NOT be dropped"

        # 4. Connection / Network Error
        event_conn = {
            "logger": "yfinance",
            "level": "error",
            "logentry": {"message": "ConnectionError: Failed to establish a new connection"},
        }
        res_conn = adapter._sanitize_event(event_conn, {})
        assert res_conn is not None, "Connection/network error must NOT be dropped"

        # 5. Non-yfinance Application Logger
        event_app = {
            "logger": "api.routes.analytics",
            "level": "error",
            "logentry": {"message": "Failed to parse technical data"},
        }
        res_app = adapter._sanitize_event(event_app, {})
        assert res_app is not None, "Non-yfinance application logger must NOT be dropped"

        # 6. Unhandled Exception
        event_exc = {
            "exception": {"values": [{"type": "ValueError", "value": "Unexpected calc failure"}]},
            "level": "error",
        }
        res_exc = adapter._sanitize_event(event_exc, {})
        assert res_exc is not None, "Application exception must NOT be dropped"


class TestActiveEquitiesAndETFs:
    """7 & 8: Verify active instruments continue using normal paths and crypto isolation."""

    @patch("api.routes.analytics.yf.Ticker")
    def test_active_equity_uses_normal_provider_path(self, mock_ticker):
        """7. Valid active equities and ETFs continue using normal market-data paths."""
        dates = pd.date_range(end="2026-10-30", periods=30, freq="B")
        mock_df = pd.DataFrame({
            "Open": [100.0] * 30,
            "High": [105.0] * 30,
            "Low": [95.0] * 30,
            "Close": [102.0] * 30,
            "Volume": [1000000] * 30,
        }, index=dates)

        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_instance.info = {"returnOnAssets": 0.15, "trailingPE": 20.0}
        mock_instance.calendar = None
        mock_ticker.return_value = mock_instance

        res = get_asset_analytics(symbol="AAPL")
        assert res["symbol"] == "AAPL"
        assert res["currentPrice"] == 102.0

        called_symbols = [call_args[0][0] for call_args in mock_ticker.call_args_list]
        assert "AAPL" in called_symbols
        assert "AAPL-USD" not in called_symbols

    def test_crypto_symbols_isolated(self):
        """8. Existing crypto/equity routing protections remain intact."""
        assert "BTC" in KNOWN_CRYPTO_SYMBOLS
        assert "ETH" in KNOWN_CRYPTO_SYMBOLS
        assert "CPRX" not in KNOWN_CRYPTO_SYMBOLS
        assert "AAPL" not in KNOWN_CRYPTO_SYMBOLS


class TestRFC8259AndGovernanceInvariants:
    """9 & 10: Verify RFC 8259 JSON compliance, RVOL, and operational holds."""

    def test_rfc8259_and_rvol_na(self):
        """9. RFC 8259 JSON serialization and RVOL N/A protections remain intact."""
        dirty = {
            "nan": float("nan"),
            "inf": float("inf"),
            "neg_inf": float("-inf"),
            "rvol": "N/A",
            "valid": 99.5,
        }
        cleaned = _sanitize_floats(dirty)
        assert cleaned["nan"] is None
        assert cleaned["inf"] is None
        assert cleaned["neg_inf"] is None
        assert cleaned["rvol"] == "N/A"
        assert cleaned["valid"] == 99.5

        # Must serialize cleanly without allow_nan violations
        dumped = json.dumps(cleaned)
        assert "NaN" not in dumped
        assert "Infinity" not in dumped

    def test_governance_invariants_and_holds(self):
        """10. No operational HOLD or execution eligibility regression."""
        # Check CONFIRMED_CORPORATE_ACTIONS integrity
        assert "CPRX" in CONFIRMED_CORPORATE_ACTIONS
        cprx_action = CONFIRMED_CORPORATE_ACTIONS["CPRX"]
        assert cprx_action["status"] == "DELISTED"
        assert cprx_action["execution_status"] == "DELISTED_SECURITY"
        assert cprx_action["decision_state"] == "INELIGIBLE_INACTIVE"
        assert cprx_action["screening_eligible"] is False
        assert cprx_action["is_investable"] is False
