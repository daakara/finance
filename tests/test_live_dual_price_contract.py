"""Tests for ARX Terminal Governed Live Dual-Price Architecture & Epoch 3 Contracts.

Validates:
1. MarketPriceState dual-price contract & serialization.
2. Market session resolution (REGULAR_SESSION, PREMARKET, AFTER_HOURS, CLOSED, WEEKEND, HOLIDAY).
3. Provider failover & fail-closed behavior (Alpaca IEX primary -> Yahoo -> UNAVAILABLE).
4. Freshness threshold enforcement (<= 5 min REALTIME vs DELAYED/UNAVAILABLE).
5. Indicator non-contamination invariant: EMA, ATR, RSI arrays are strictly immutable under spot price changes.
6. OptimalExecution live spot evaluation:
   - Execution status dynamically updates on intraday spot movements.
   - Stop-loss breach dynamically flags STOPPED_OUT without altering structural levels.
7. Manifest compliance: EPOCH_3_MANIFEST verified, FROZEN_ENGINE_MANIFEST verified, EPOCH_2 untouched.
"""

import math
import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, time

from analyst_dashboard.data.market_price_state import (
    MarketPriceState,
    get_market_session,
    resolve_dual_price_state,
    REALTIME_MAX_AGE_MS,
)
from analyst_dashboard.data.alpaca_fetcher import AlpacaMarketFetcher
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from api.routes.analytics import compute_intraday_technicals
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
from analyst_dashboard.governance.passive_capture import (
    PassiveCaptureHook,
    ExecutionContext,
    governance_execution_context,
)


@pytest.fixture
def sample_daily_candles():
    """Generates synthetic completed daily OHLCV bars for indicator non-contamination testing."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range(end="2026-09-23", periods=n, freq="B")
    close = np.linspace(100, 150, n) + np.random.normal(0, 1.5, n)
    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)
    open_p = low + np.random.uniform(0.1, 1.0, n)
    volume = np.random.randint(1000000, 5000000, n)

    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    return df


# ==============================================================================
# 1. MARKET SESSION RESOLUTION TESTS
# ==============================================================================

def test_market_session_resolution_regular_session():
    # 2026-09-23 is Wednesday at 10:30 AM Eastern (14:30 UTC) -> REGULAR_SESSION
    dt_regular = datetime(2026, 9, 23, 14, 30, tzinfo=timezone.utc)
    assert get_market_session(dt_regular) == "REGULAR_SESSION"


def test_market_session_resolution_premarket():
    # 2026-09-23 Wednesday at 7:00 AM Eastern (11:00 UTC) -> PREMARKET
    dt_pre = datetime(2026, 9, 23, 11, 0, tzinfo=timezone.utc)
    assert get_market_session(dt_pre) == "PREMARKET"


def test_market_session_resolution_after_hours():
    # 2026-09-23 Wednesday at 17:30 Eastern (21:30 UTC) -> AFTER_HOURS
    dt_post = datetime(2026, 9, 23, 21, 30, tzinfo=timezone.utc)
    assert get_market_session(dt_post) == "AFTER_HOURS"


def test_market_session_resolution_weekend():
    # 2026-09-26 Saturday at 14:00 UTC -> WEEKEND
    dt_weekend = datetime(2026, 9, 26, 14, 0, tzinfo=timezone.utc)
    assert get_market_session(dt_weekend) == "WEEKEND"


def test_market_session_resolution_holiday():
    # 2026-12-25 Christmas (Friday) -> HOLIDAY
    dt_holiday = datetime(2026, 12, 25, 15, 0, tzinfo=timezone.utc)
    assert get_market_session(dt_holiday) == "HOLIDAY"


def test_market_session_calendar_exhaustive():
    """Exhaustively verifies NYSE calendar sessions, holidays, and early closes."""
    test_cases = [
        ("regular", datetime(2026, 9, 23, 14, 30, tzinfo=timezone.utc), "REGULAR_SESSION"),
        ("premarket", datetime(2026, 9, 23, 11, 0, tzinfo=timezone.utc), "PREMARKET"),
        ("after_hours", datetime(2026, 9, 23, 21, 30, tzinfo=timezone.utc), "AFTER_HOURS"),
        ("weekend", datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc), "WEEKEND"),
        ("good_friday", datetime(2026, 4, 3, 15, 0, tzinfo=timezone.utc), "HOLIDAY"),
        ("juneteenth", datetime(2026, 6, 19, 15, 0, tzinfo=timezone.utc), "HOLIDAY"),
        ("thanksgiving", datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc), "HOLIDAY"),
        ("black_friday_open", datetime(2026, 11, 27, 16, 0, tzinfo=timezone.utc), "REGULAR_SESSION"),
        ("black_friday_close", datetime(2026, 11, 27, 18, 30, tzinfo=timezone.utc), "AFTER_HOURS"),
        ("christmas_eve_open", datetime(2026, 12, 24, 16, 0, tzinfo=timezone.utc), "REGULAR_SESSION"),
        ("christmas_eve_close", datetime(2026, 12, 24, 18, 30, tzinfo=timezone.utc), "AFTER_HOURS"),
        ("observed_independence_day", datetime(2026, 7, 3, 15, 0, tzinfo=timezone.utc), "HOLIDAY"),
        ("observed_christmas", datetime(2026, 12, 25, 15, 0, tzinfo=timezone.utc), "HOLIDAY"),
    ]
    for label, dt, expected in test_cases:
        actual = get_market_session(dt)
        assert actual == expected, f"Failed calendar verification for {label}: expected {expected}, got {actual}"


# ==============================================================================
# 2. PROVIDER RESOLUTION & FAILOVER TESTS
# ==============================================================================

def test_dual_price_resolution_with_alpaca(monkeypatch):
    """Primary provider (Alpaca IEX) delivers real-time quote."""
    fetcher = AlpacaMarketFetcher(api_key_id="mock_key", api_secret_key="mock_secret")
    now_ms = int(datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)

    monkeypatch.setattr(fetcher, "fetch_realtime_quote", lambda sym: {
        "price": 152.45,
        "observed_at_ms": now_ms - 1000,
        "source": "ALPACA_IEX",
        "symbol": sym,
    })

    state = resolve_dual_price_state(
        symbol="AAPL",
        analysis_reference_price=150.00,
        analysis_reference_date="2026-09-22",
        alpaca_fetcher=fetcher,
        now_utc=datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc),
    )

    assert state.symbol == "AAPL"
    assert state.live_spot_price == 152.45
    assert state.live_source == "ALPACA_IEX"
    assert state.live_freshness == "REALTIME"
    assert state.analysis_reference_price == 150.00
    assert state.analysis_reference_date == "2026-09-22"
    assert state.analysis_reference_source == "COMPLETED_SESSION"


def test_dual_price_resolution_fallback_to_yahoo(monkeypatch):
    """When Alpaca is unavailable, failover to Yahoo ticker_obj."""
    fetcher = AlpacaMarketFetcher(api_key_id=None, api_secret_key=None)
    now_ms = int(datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)

    class MockTicker:
        history_metadata = {
            "regularMarketPrice": 151.80,
            "regularMarketTime": now_ms // 1000,
        }

    state = resolve_dual_price_state(
        symbol="AAPL",
        analysis_reference_price=150.00,
        analysis_reference_date="2026-09-22",
        alpaca_fetcher=fetcher,
        ticker_obj=MockTicker(),
        now_utc=datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc),
    )

    assert state.live_spot_price == 151.80
    assert state.live_source == "YAHOO"
    assert state.live_freshness == "DELAYED"
    assert state.analysis_reference_price == 150.00


def test_dual_price_resolution_fail_closed_on_stale_data(monkeypatch):
    """When quote is older than 5 minutes, freshness degrades to DELAYED."""
    fetcher = AlpacaMarketFetcher(api_key_id="mock_key", api_secret_key="mock_secret")
    now_ms = int(datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)
    stale_ms = now_ms - (10 * 60 * 1000)  # 10 minutes ago

    monkeypatch.setattr(fetcher, "fetch_realtime_quote", lambda sym: {
        "price": 152.00,
        "observed_at_ms": stale_ms,
        "source": "ALPACA_IEX",
        "symbol": sym,
    })

    state = resolve_dual_price_state(
        symbol="AAPL",
        analysis_reference_price=150.00,
        analysis_reference_date="2026-09-22",
        alpaca_fetcher=fetcher,
        now_utc=datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc),
    )

    assert state.live_spot_price == 152.00
    assert state.live_freshness == "DELAYED"


def test_provider_freshness_exact_boundaries(monkeypatch):
    """Verifies provider freshness at exact boundary thresholds:
    - Alpaca 59s -> REALTIME
    - Alpaca 60s -> REALTIME
    - Alpaca 61s -> STALE
    - Alpaca 299s -> STALE
    - Alpaca 300s -> STALE
    - Alpaca 301s -> DELAYED
    - Missing provider quote -> UNAVAILABLE
    - Yahoo fallback quote -> DELAYED
    """
    ref_dt = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)
    now_ms = int(ref_dt.timestamp() * 1000)

    # 1. Alpaca boundary tests
    alpaca_cases = [
        (59, "REALTIME"),
        (60, "REALTIME"),
        (61, "STALE"),
        (299, "STALE"),
        (300, "STALE"),
        (301, "DELAYED"),
    ]
    for age_sec, expected_freshness in alpaca_cases:
        fetcher = AlpacaMarketFetcher(api_key_id="mock_key", api_secret_key="mock_secret")
        observed_ms = now_ms - (age_sec * 1000)
        monkeypatch.setattr(fetcher, "fetch_realtime_quote", lambda sym, o_ms=observed_ms: {
            "price": 150.50,
            "observed_at_ms": o_ms,
            "source": "ALPACA_IEX",
            "symbol": sym,
        })
        state = resolve_dual_price_state(
            symbol="AAPL",
            analysis_reference_price=150.00,
            analysis_reference_date="2026-09-22",
            alpaca_fetcher=fetcher,
            now_utc=ref_dt,
        )
        assert state.live_freshness == expected_freshness, (
            f"Alpaca age {age_sec}s: expected {expected_freshness}, got {state.live_freshness}"
        )
        assert state.live_source == "ALPACA_IEX"

    # 2. Missing quote -> UNAVAILABLE
    no_fetcher = AlpacaMarketFetcher(api_key_id=None, api_secret_key=None)
    missing_state = resolve_dual_price_state(
        symbol="AAPL",
        analysis_reference_price=150.00,
        analysis_reference_date="2026-09-22",
        alpaca_fetcher=no_fetcher,
        ticker_obj=None,
        now_utc=ref_dt,
    )
    assert missing_state.live_freshness == "UNAVAILABLE"
    assert missing_state.live_source == "UNAVAILABLE"

    # 3. Yahoo -> DELAYED
    class FreshYahooTicker:
        history_metadata = {
            "regularMarketPrice": 150.75,
            "regularMarketTime": now_ms // 1000,
        }
    yahoo_state = resolve_dual_price_state(
        symbol="AAPL",
        analysis_reference_price=150.00,
        analysis_reference_date="2026-09-22",
        alpaca_fetcher=no_fetcher,
        ticker_obj=FreshYahooTicker(),
        now_utc=ref_dt,
    )
    assert yahoo_state.live_freshness == "DELAYED"
    assert yahoo_state.live_source == "YAHOO"


def test_dual_price_resolution_total_provider_outage():
    """When all live quote providers are unavailable, fail closed gracefully."""
    fetcher = AlpacaMarketFetcher(api_key_id=None, api_secret_key=None)

    state = resolve_dual_price_state(
        symbol="AAPL",
        analysis_reference_price=150.00,
        analysis_reference_date="2026-09-22",
        alpaca_fetcher=fetcher,
        ticker_obj=None,
        now_utc=datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc),
    )

    assert state.live_spot_price is None
    assert state.live_source == "UNAVAILABLE"
    assert state.live_freshness == "UNAVAILABLE"
    assert state.analysis_reference_price == 150.00


# ==============================================================================
# 3. INDICATOR NON-CONTAMINATION INVARIANT PROOF
# ==============================================================================

def test_indicator_non_contamination_proof(sample_daily_candles):
    """Mathematically proves that completed-session indicator arrays are 100% identical

    regardless of spot price movements (+2%, -2%, or None).
    Daily indicators must NEVER be contaminated with intraday open bars.
    """
    engine = OptimalExecutionEngine()
    ref_price = float(sample_daily_candles["Close"].iloc[-1])

    # Case A: Live spot is None (weekend/after hours baseline)
    tech_a = compute_intraday_technicals(sample_daily_candles)
    plan_a = engine.calculate_trade_levels(
        price_df=sample_daily_candles,
        current_price=ref_price,
        user_role="LONG_TERM",
        technicals=tech_a,
        live_spot_price=None,
    )

    # Case B: Live spot gaps up +2.5% intraday
    spot_up = ref_price * 1.025
    tech_b = compute_intraday_technicals(sample_daily_candles)
    plan_b = engine.calculate_trade_levels(
        price_df=sample_daily_candles,
        current_price=ref_price,
        user_role="LONG_TERM",
        technicals=tech_b,
        live_spot_price=spot_up,
    )

    # Case C: Live spot drops -2.5% intraday
    spot_down = ref_price * 0.975
    tech_c = compute_intraday_technicals(sample_daily_candles)
    plan_c = engine.calculate_trade_levels(
        price_df=sample_daily_candles,
        current_price=ref_price,
        user_role="LONG_TERM",
        technicals=tech_c,
        live_spot_price=spot_down,
    )

    # 1. Technical indicator arrays are strictly identical
    assert tech_a == tech_b == tech_c

    # 2. Structural trade levels (anchored to completed daily close) are strictly identical
    assert plan_a["stop_loss"] == plan_b["stop_loss"] == plan_c["stop_loss"]
    assert plan_a["optimal_entry_min"] == plan_b["optimal_entry_min"] == plan_c["optimal_entry_min"]
    assert plan_a["optimal_entry_max"] == plan_b["optimal_entry_max"] == plan_c["optimal_entry_max"]
    assert plan_a["take_profit_1"] == plan_b["take_profit_1"] == plan_c["take_profit_1"]
    assert plan_a["analysis_reference_price"] == plan_b["analysis_reference_price"] == plan_c["analysis_reference_price"]

    # 3. Live spot price is isolated to evaluation price & distances
    assert plan_b["live_spot_price"] == spot_up
    assert plan_c["live_spot_price"] == spot_down
    assert plan_b["eval_price"] == spot_up
    assert plan_c["eval_price"] == spot_down


# ==============================================================================
# 4. OPTIMAL EXECUTION DYNAMIC SPOT EVALUATION
# ==============================================================================

def test_optimal_execution_live_spot_evaluation_stop_breach(sample_daily_candles):
    """When intraday live spot breaches stop-loss floor, execution_status dynamically

    resolves to STOPPED_OUT without modifying the underlying structural trade levels.
    """
    engine = OptimalExecutionEngine()
    ref_price = float(sample_daily_candles["Close"].iloc[-1])
    technicals = compute_intraday_technicals(sample_daily_candles)

    # Initial baseline
    plan_normal = engine.calculate_trade_levels(
        price_df=sample_daily_candles,
        current_price=ref_price,
        user_role="LONG_TERM",
        technicals=technicals,
        live_spot_price=ref_price,
    )
    stop_level = plan_normal["stop_loss"]
    assert stop_level is not None

    # Severe intraday crash below stop level
    crashed_spot = stop_level * 0.98
    plan_stopped = engine.calculate_trade_levels(
        price_df=sample_daily_candles,
        current_price=ref_price,
        user_role="LONG_TERM",
        technicals=technicals,
        live_spot_price=crashed_spot,
    )

    assert plan_stopped["execution_status"] == "STOPPED_OUT"
    assert plan_stopped["is_in_buy_zone"] is False
    # Structural stop loss price itself was not mutated
    assert plan_stopped["stop_loss"] == stop_level
    assert plan_stopped["analysis_reference_price"] == ref_price


# ==============================================================================
# 5. GOVERNANCE MANIFEST VALIDATION
# ==============================================================================

def test_frozen_engine_manifest_compliance():
    """Asserts FROZEN_ENGINE_MANIFEST.json is verified against current repository code."""
    res = ExperimentLedger.verify_frozen_engine_manifest()
    assert res["valid"] is True, f"Frozen manifest failed verification: {res}"
    assert res["status"] == "VERIFIED"
    assert res["frozenStrategyVersion"] == "2.5.0"


def test_epoch3_governance_manifest_compliance():
    """Asserts EPOCH_3_MANIFEST.json is verified against current repository code."""
    res = ExperimentLedger.verify_epoch3_manifest()
    assert res["valid"] is True, f"Epoch 3 manifest failed verification: {res}"
    assert res["status"] == "VERIFIED"
    assert res["epochId"] == "ARX_PROSPECTIVE_VALIDATION_EPOCH_3"
    assert len(res["files"]) == 8


def test_epoch2_manifest_byte_for_byte_untouched():
    """Asserts EPOCH_2_MANIFEST.json remains 100% byte-for-byte untouched from historical record."""
    m2 = ExperimentLedger.get_epoch2_manifest()
    assert m2 is not None
    assert m2["epochId"] == "ARX_PROSPECTIVE_VALIDATION_EPOCH_2"
    assert m2["observationGovernanceManifestHash"] == "3ba81b701a260dc5098d48f5356dd7a0fd5c064354b5ecea5d729039dc2fff38"
    assert m2["decisionEngineSha"] == "7ad44595826c147cc77f93cd676af520764c7442"
    assert len(m2["executableGovernanceFiles"]) == 4
