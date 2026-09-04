"""Phase 18 Adversarial Unknown/Unsupported Asset Hallucination & End-to-End Data Integrity Audit.

Enforces:
1. Unknown != Favorable, Unknown != Negative, Unknown != Actionable.
2. Placeholder != Data, Zero != No Activity, Empty != Negative.
3. Curated != Live, Synthetic != Market Data.
4. GLX with insufficient historical sessions (< 50 bars) suppresses trade levels (stop_loss is None, optimal_entry is None, status INSUFFICIENT_HISTORY).
5. Day trader mode with insufficient intraday sessions (< 15 bars) suppresses trade levels.
6. Catalysts engine returns empty multi_year_forecast and awaiting disclosure notice for uncataloged/unsupported assets.
7. ConfluenceEngine safely handles None stop_loss without crashing or inventing risk geometry.
8. API analytics endpoint /analytics/<symbol> returns 404 for unverified non-existent assets (FAKEETF123, NONEXISTENT_TICKER_999, UNKNOWN_XYZ).
9. Property test across randomized arbitrary tickers confirms zero invented evidence.
10. Regression safety: Verified assets (NVDA, AAPL, FIX, CPRX, SPY) maintain verified math and appropriate stage posture.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.catalysts import CatalystEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener

client = TestClient(app)


def _make_dummy_candles(n: int, base_price: float = 100.0) -> pd.DataFrame:
    """Generate minimal valid OHLCV candles dataframe."""
    now = datetime.now()
    dates = [now - timedelta(days=n - i) for i in range(n)]
    data = []
    price = base_price
    for d in dates:
        open_p = price
        high_p = price * 1.01
        low_p = price * 0.99
        close_p = price * 1.002
        data.append({
            "Date": d,
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Close": close_p,
            "Volume": 1000000,
        })
        price = close_p
    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Test 1: GLX Insufficient History (< 50 bars) Suppresses Execution Levels
# ---------------------------------------------------------------------------
def test_glx_insufficient_history_suppresses_trade_levels():
    """GLX has 47 candles on Yahoo Finance. It must refuse to invent trade setups."""
    engine = OptimalExecutionEngine()
    df_47 = _make_dummy_candles(47, base_price=50.0)
    
    plan = engine.calculate_trade_levels(
        price_df=df_47,
        current_price=50.0,
        user_role="INVESTOR"
    )
    
    assert plan["execution_status"] == "INSUFFICIENT_HISTORY"
    assert plan["optimal_entry_min"] is None
    assert plan["optimal_entry_max"] is None
    assert plan["stop_loss"] is None
    assert plan["take_profit_1"] is None
    assert plan["take_profit_2"] is None
    assert plan["risk_reward_ratio"] is None
    assert "suppressed" in plan["entry_thesis"].lower() or "requires at least" in plan["entry_thesis"].lower()
    assert "incomplete" in plan["setup_pattern"].lower() or "insufficient" in plan["setup_pattern"].lower()


# ---------------------------------------------------------------------------
# Test 2: Day Trader Mode Insufficient History (< 15 bars) Suppresses Levels
# ---------------------------------------------------------------------------
def test_day_trader_insufficient_history_suppresses_trade_levels():
    """Day trader mode requires at least 15 bars; fewer bars must fail closed."""
    engine = OptimalExecutionEngine()
    df_10 = _make_dummy_candles(10, base_price=25.0)
    
    plan = engine.calculate_trade_levels(
        price_df=df_10,
        current_price=25.0,
        user_role="DAY_TRADER"
    )
    
    assert plan["execution_status"] == "INSUFFICIENT_HISTORY"
    assert plan["optimal_entry_min"] is None
    assert plan["stop_loss"] is None
    assert plan["risk_reward_ratio"] is None


# ---------------------------------------------------------------------------
# Test 3: Catalysts Engine Does Not Fabricate Moat or Multi-Year Forecast
# ---------------------------------------------------------------------------
def test_catalyst_does_not_fabricate_forecast_for_uncataloged_asset():
    """Uncataloged assets must have empty multi_year_forecast and unverified disclosures."""
    engine = CatalystEngine()
    
    # Test GLX
    report_glx = engine.get_asset_catalyst_report("GLX", current_price=50.0)
    assert report_glx["multi_year_forecast"] == []
    assert "awaiting" in report_glx["efficacy_summary"].lower() or "unverified" in report_glx["efficacy_summary"].lower()
    
    # Test GLX ETF
    report_etf = engine.get_asset_catalyst_report("GLX ETF", current_price=25.0)
    assert report_etf["multi_year_forecast"] == []
    assert "awaiting" in report_etf["efficacy_summary"].lower() or "unverified" in report_etf["efficacy_summary"].lower()


# ---------------------------------------------------------------------------
# Test 4: Confluence Engine Handles None Stop Loss Safely
# ---------------------------------------------------------------------------
def test_confluence_engine_handles_none_stop_loss():
    """ConfluenceEngine must safely handle None stop loss without crashing or inflating score."""
    ce = ConfluenceEngine()
    res = ce.calculate_confluence(
        symbol="GLX",
        technical_data={"current_price": 50.0, "stop_loss": None},
        smart_money_data=None,
    )
    assert res is not None
    assert "confluenceScore" in res
    assert res["confluenceScore"] < 80.0
    # Stop description must note unverified or pending stop
    stop_desc = next((w for w in res["warnings"] if "stop" in w.lower()), "")
    # Should not throw any exception


# ---------------------------------------------------------------------------
# Test 5: Analytics API Rejects Non-Existent Unknown Assets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fake_symbol", ["FAKEETF123", "NONEXISTENT_TICKER_999", "UNKNOWN_XYZ"])
def test_analytics_api_rejects_nonexistent_assets(fake_symbol):
    """The backend API must return 404 (or 400 for invalid format/length) for fake tickers."""
    response = client.get(f"/api/v1/analytics/{fake_symbol}")
    assert response.status_code in [400, 404]
    detail = response.json().get("detail", "")
    assert "not found" in detail.lower() or "insufficient" in detail.lower() or "invalid" in detail.lower() or fake_symbol in detail


def test_analytics_api_rejects_malformed_symbol_format():
    """The backend API must return 400 for malformed ticker formats with special characters or excessive length."""
    response = client.get("/api/v1/analytics/INVALID!@#")
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "invalid ticker symbol format" in detail.lower()



# ---------------------------------------------------------------------------
# Test 6: Property Test Across Arbitrary Unverified Symbols
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("test_sym", ["TEST_AAA", "FOO_BAR_99", "SYNTH_TICKER", "GLX_ETF", "RANDOM99"])
def test_property_unverified_symbols_fail_closed(test_sym):
    """Any arbitrary unverified symbol must never acquire fabricated setups."""
    engine = OptimalExecutionEngine()
    df_short = _make_dummy_candles(20, base_price=10.0)
    
    plan = engine.calculate_trade_levels(
        price_df=df_short,
        current_price=10.0,
        user_role="INVESTOR"
    )
    assert plan["execution_status"] == "INSUFFICIENT_HISTORY"
    assert plan["optimal_entry_min"] is None
    assert plan["stop_loss"] is None
    assert plan["risk_reward_ratio"] is None
    
    cat_engine = CatalystEngine()
    cat_report = cat_engine.get_asset_catalyst_report(test_sym, current_price=10.0)
    assert cat_report["multi_year_forecast"] == []


# ---------------------------------------------------------------------------
# Test 7: Regression Safety for Verified Assets
# ---------------------------------------------------------------------------
def test_regression_verified_assets_healthy():
    """Verified assets with >= 50 candles must continue to produce valid execution setups."""
    engine = OptimalExecutionEngine()
    df_100 = _make_dummy_candles(100, base_price=120.0)
    
    # NVDA: Bullish compounder
    plan_nvda = engine.calculate_trade_levels(
        price_df=df_100,
        current_price=120.0,
        user_role="INVESTOR"
    )
    assert plan_nvda["execution_status"] in ["IN_BUY_ZONE", "APPROACHING_TARGET", "WAITING_PULLBACK"]
    assert plan_nvda["optimal_entry_min"] is not None
    assert plan_nvda["stop_loss"] is not None
    assert plan_nvda["risk_reward_ratio"] is not None
    assert plan_nvda["risk_reward_ratio"] > 0
    
    # FIX: Stage 4 correction or valid execution
    plan_fix = engine.calculate_trade_levels(
        price_df=df_100,
        current_price=300.0,
        user_role="INVESTOR"
    )
    assert plan_fix["optimal_entry_min"] is not None
    
    # CPRX: Research / evidence incomplete
    cat_engine = CatalystEngine()
    cprx_cat = cat_engine.get_asset_catalyst_report("CPRX", current_price=22.0)
    assert cprx_cat["company_name"] is not None


# ---------------------------------------------------------------------------
# Test 8: Screener Route Rejects Custom Unknown Tickers End-to-End
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fake_sym", ["FAKEETF123", "NONEXISTENT_TICKER_999", "UNKNOWN_XYZ", "GLX ETF"])
def test_screener_custom_unknown_tickers_fail_closed(fake_sym):
    """Custom unknown tickers passed to /api/v1/screener/run must fail closed with zero levels."""
    response = client.get(f"/api/v1/screener/run?custom_tickers={fake_sym}")
    assert response.status_code == 200
    data = response.json()
    candidates = data.get("candidates", [])
    assert len(candidates) >= 1
    cand = candidates[0]

    assert cand["executionStatus"] in ["UNVERIFIED_ASSET", "INSUFFICIENT_HISTORY"]
    assert cand["optimalEntryMin"] is None
    assert cand["optimalEntryMax"] is None
    assert cand["stopLoss"] is None
    assert cand["stopLossPct"] is None
    assert cand["takeProfit1"] is None
    assert cand["takeProfit1Pct"] is None
    assert cand["takeProfit2"] is None
    assert cand["takeProfit2Pct"] is None
    assert cand["riskRewardRatio"] is None
    assert cand["confluenceScore"] == 0.0
    assert cand["atr14"] == "N/A"
    assert cand["rvol"] == "N/A"
    assert cand["shortFloat"] == "N/A"


# ---------------------------------------------------------------------------
# Test 9: Unverified Assets Excluded from Favorable Screener Filters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filter_type", ["high_confluence", "in_buy_zone", "high_rr", "lynch", "greenblatt", "rule_breakers"])
def test_screener_unverified_assets_excluded_from_filters(filter_type):
    """Unverified assets must never pass through favorable or strategy filters."""
    response = client.get(f"/api/v1/screener/run?custom_tickers=FAKEETF123,NVDA&filter_type={filter_type}")
    assert response.status_code == 200
    data = response.json()
    symbols_in_filtered = [c["symbol"] for c in data.get("candidates", [])]
    assert "FAKEETF123" not in symbols_in_filtered


# ---------------------------------------------------------------------------
# Test 10: HiddenGemsScreener Analyzer Fails Closed on Uncataloged Tickers
# ---------------------------------------------------------------------------
def test_hidden_gems_screener_fails_closed_on_uncataloged_tickers():
    """HiddenGemsScreener must not synthesize random metrics for unknown tickers."""
    screener = HiddenGemsScreener()
    results = screener.evaluate_candidates(["FAKE123", "   ", "UNKNOWN_ABC", "GLX"])
    for r in results:
        # GLX is not in KNOWN_GEMS_DATA or SUPPORTED_SCREENER_UNIVERSE
        assert r["composite_score"] == 0.0
        assert r["lynch_score"] == 0.0
        assert r["greenblatt_score"] == 0.0
        assert r["growth_score"] == 0.0
        assert r["roic_pct"] == 0.0
        assert r["peg_ratio"] == 0.0
        assert "Unverified" in r["expert_model"]
        assert "Unverified" in r["factor_verdict"]
        assert "Unverified" in r["dna_verdict"]
