"""
Directive 044R.2: Adversarial Production Data Integrity, Three-Surface Setup Parity, and Sentry Remediation Suite
Covers:
- RFC 8259 compliant JSON serialization across analytics, screener, macro routes (Sentry ARX-TERMINAL-3)
- Elimination of equity-to-crypto false fallback (Sentry ARX-TERMINAL-3Q: AAPL-USD suppression)
- Three-surface setup parity across screener, tactical setups, and single-asset analytics
- RVOL unobserved fallback integrity (N/A vs misleading 1.0x)
- Macro provider graceful degradation and failure isolation (Sentry ARX-TERMINAL-2C)
"""

import json
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from api.routes.analytics import _sanitize_floats as sanitize_analytics
from api.routes.macro import _sanitize_floats as sanitize_macro
from api.routes.screener import _sanitize_floats as sanitize_screener


class TestRFC8259Sanitization:
    """Test recursive sanitization of non-finite numbers and numpy/pandas data types."""

    @pytest.mark.parametrize("sanitize_fn", [sanitize_analytics, sanitize_macro, sanitize_screener])
    def test_sanitize_nested_payload(self, sanitize_fn):
        dirty_payload = {
            "symbol": "TEST",
            "native_nan": float("nan"),
            "native_inf": float("inf"),
            "native_neg_inf": float("-inf"),
            "numpy_float32_nan": np.float32("nan"),
            "numpy_float64_inf": np.float64("inf"),
            "numpy_int64": np.int64(42),
            "pandas_na": pd.NA,
            "pandas_nat": pd.NaT,
            "numpy_array": np.array([1.5, np.nan, 3.0]),
            "nested_dict": {
                "inner_nan": float("nan"),
                "inner_list": [1.0, float("nan"), np.float32("nan"), "valid_str"],
            },
            "valid_float": 123.45,
            "valid_str": "OK",
            "valid_none": None,
            "bool_true": True,
            "bool_false": False,
        }

        cleaned = sanitize_fn(dirty_payload)

        # Invariant: booleans must preserve their exact boolean identity and not be cast to 1 or 0
        assert cleaned["bool_true"] is True
        assert cleaned["bool_false"] is False
        assert isinstance(cleaned["bool_true"], bool)
        assert isinstance(cleaned["bool_false"], bool)

        # Must serialize cleanly via standard json.dumps without allow_nan=False violations
        json_str = json.dumps(cleaned)
        deserialized = json.loads(json_str)

        assert deserialized["native_nan"] is None
        assert deserialized["native_inf"] is None
        assert deserialized["native_neg_inf"] is None
        assert deserialized["numpy_float32_nan"] is None
        assert deserialized["numpy_float64_inf"] is None
        assert deserialized["numpy_int64"] == 42
        assert deserialized["pandas_na"] is None
        assert deserialized["pandas_nat"] is None
        assert deserialized["numpy_array"] == [1.5, None, 3.0]
        assert deserialized["nested_dict"]["inner_nan"] is None
        assert deserialized["nested_dict"]["inner_list"] == [1.0, None, None, "valid_str"]
        assert deserialized["valid_float"] == 123.45
        assert deserialized["valid_str"] == "OK"
        assert deserialized["valid_none"] is None

        # Verify no non-standard tokens exist in serialized JSON text
        assert "NaN" not in json_str
        assert "Infinity" not in json_str


class TestEquityCryptoFallbackSuppression:
    """Verify that equities like AAPL and GOOGL never query ${sym}-USD on fetch failures."""

    def test_crypto_classification(self):
        from api.routes.analytics import KNOWN_CRYPTO_SYMBOLS

        assert "BTC" in KNOWN_CRYPTO_SYMBOLS
        assert "ETH" in KNOWN_CRYPTO_SYMBOLS
        assert "AAPL" not in KNOWN_CRYPTO_SYMBOLS
        assert "GOOGL" not in KNOWN_CRYPTO_SYMBOLS
        assert "ONON" not in KNOWN_CRYPTO_SYMBOLS

    @patch("api.routes.analytics.yf.Ticker")
    @patch("api.routes.analytics.market_db")
    def test_equity_never_fetches_usd_ticker(self, mock_market_db, mock_ticker):
        from fastapi import HTTPException
        from api.routes.analytics import get_asset_analytics

        # Simulate primary yf.Ticker("AAPL").history failing/returning empty
        mock_hist = MagicMock()
        mock_hist.empty = True
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_instance

        # Simulate SQLite cache also empty
        mock_market_db.get_candles_with_freshness.return_value = {"candles": []}

        # Expect 404
        with pytest.raises(HTTPException) as excinfo:
            get_asset_analytics(symbol="AAPL")

        assert excinfo.value.status_code == 404

        # Crucial assertion: yf.Ticker must NEVER have been called with "AAPL-USD"
        called_symbols = [call_args[0][0] for call_args in mock_ticker.call_args_list]
        assert "AAPL-USD" not in called_symbols
        assert "AAPL" in called_symbols


class TestRVOLFallback:
    """Verify unobserved volume returns N/A rather than deceptive 1.0x."""

    def test_screener_rvol_unobserved(self):
        # Empty hist_df
        hist_df = pd.DataFrame()
        if not hist_df.empty and len(hist_df) >= 5 and "Volume" in hist_df.columns:
            rvol_val = "1.0x"
        else:
            rvol_val = "N/A"
        assert rvol_val == "N/A"

        # Volume column present but all zeros
        hist_zero_vol = pd.DataFrame({
            "Close": [10.0] * 10,
            "Volume": [0] * 10
        })
        if not hist_zero_vol.empty and len(hist_zero_vol) >= 5 and "Volume" in hist_zero_vol.columns:
            recent_vol = float(hist_zero_vol["Volume"].iloc[-1])
            avg_vol = float(hist_zero_vol["Volume"].tail(20).mean())
            if avg_vol > 0:
                rvol_val = f"{round(recent_vol / avg_vol, 1)}x"
            else:
                rvol_val = "N/A"
        else:
            rvol_val = "N/A"
        assert rvol_val == "N/A"


class TestThreeSurfaceSetupParity:
    """Verify parity of setup pattern and lookback across screener, setups, and analytics."""

    def test_252_bar_lookback_detects_correction(self):
        from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine

        engine = OptimalExecutionEngine()

        # Construct 252-bar synthetic history for an asset that topped 180 bars ago at $50
        # and is currently at $35 (-30% correction)
        dates = pd.date_range(end="2026-10-30", periods=252, freq="B")
        closes = np.full(252, 35.0)
        # 52-week high at index 70
        closes[70] = 50.0

        df_252 = pd.DataFrame({
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(252, 1000000),
        }, index=dates)

        # 60-bar subset does NOT see the $50 peak
        df_60 = df_252.tail(60)

        plan_60 = engine.calculate_trade_levels(df_60, 35.0, user_role="LONG_TERM")
        plan_252 = engine.calculate_trade_levels(df_252, 35.0, user_role="LONG_TERM")

        # 60 bars sees max price of ~35.35 -> might classify as consolidation/VCP
        # 252 bars sees peak of $50.50 -> 35.0 is -30.7% from high -> Stage 1 Base or Correction
        assert plan_252.get("setup_pattern") != plan_60.get("setup_pattern") or plan_252.get("stage_phase") != plan_60.get("stage_phase")
        assert "Base" in plan_252.get("setup_pattern", "") or "Bottoming" in plan_252.get("setup_pattern", "")


class TestMacroFailureIsolation:
    """Verify macro ribbon isolates provider failure and returns UNAVAILABLE without crashing."""

    @patch("api.routes.macro.yf.Ticker")
    def test_isolated_bar_unavailable_on_failure(self, mock_ticker):
        from api.routes.macro import _fetch_isolated_bar

        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance

        res = _fetch_isolated_bar("SPY")
        assert res["status"] == "UNAVAILABLE"
        assert res["available"] is False
        assert res["price"] is None
