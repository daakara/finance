import pytest
import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.liquidity_guard import LiquidityGuard
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine

pytestmark = pytest.mark.tier2b


def test_liquidity_guard_fallback():
    res = LiquidityGuard.evaluate_liquidity(pd.DataFrame(), 100.0)
    assert res["liquidity_grade"] == "UNKNOWN_LIQUIDITY"
    assert res["badge_color"] == "slate"
    assert res["suppress_buy_zone"] is False
    assert res["execution_hazard"] is False
    assert res["adv_20d_usd"] == 0.0

def test_liquidity_guard_high_trading_liquidity():
    # 30 days of high volume, low price volatility
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [100.0 + (i * 0.1) for i in range(30)]
    volumes = [500_000 for _ in range(30)] # $50M ADV
    df = pd.DataFrame({"Close": prices, "Volume": volumes, "High": [p + 0.5 for p in prices], "Low": [p - 0.5 for p in prices]}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 103.0)
    assert res["liquidity_grade"] == "HIGH_TRADING_LIQUIDITY"
    assert res["adv_20d_usd"] > 2_000_000.0
    assert res["suppress_buy_zone"] is False
    assert res["execution_hazard"] is False
    assert res["badge_color"] == "emerald"
    # Mathematical scaling verification
    assert abs(res["amihud_illiq_scaled"] - (res["amihud_illiq"] * 1e6)) < 1e-9

def test_liquidity_guard_execution_risk():
    # 30 days of micro volume ($20K ADV)
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [2.0 for _ in range(30)]
    volumes = [10_000 for _ in range(30)] # $20K ADV
    df = pd.DataFrame({"Close": prices, "Volume": volumes, "High": [2.1 for _ in range(30)], "Low": [1.9 for _ in range(30)]}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 2.0)
    assert res["liquidity_grade"] == "EXECUTION_RISK"
    assert res["suppress_buy_zone"] is False  # Shadow mode invariant: zero decision mutation
    assert res["execution_hazard"] is True
    assert res["market_order_warning"] is True
    assert res["badge_color"] == "rose"

def test_optimal_execution_non_interference_adversarial():
    """
    Adversarial non-interference test:
    Verify that an identical technical setup with low liquidity produces
    IDENTICAL optimal entry, stop loss, and take profit levels,
    proving LiquidityGuard does not tamper with model decisions.
    """
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [100.0 + (i * 0.1) for i in range(30)]
    highs = [p + 0.5 for p in prices]
    lows = [p - 0.5 for p in prices]

    # Setup A: Liquid ($50M ADV)
    df_liquid = pd.DataFrame({"Close": prices, "Volume": [500_000]*30, "High": highs, "Low": lows}, index=dates)
    plan_liquid = OptimalExecutionEngine.calculate_trade_levels(df_liquid, 103.0)

    # Setup B: Illiquid ($5K ADV), identical price history
    df_illiquid = pd.DataFrame({"Close": prices, "Volume": [50]*30, "High": highs, "Low": lows}, index=dates)
    plan_illiquid = OptimalExecutionEngine.calculate_trade_levels(df_illiquid, 103.0)

    # Core frozen model decisions MUST be 100% identical
    assert plan_liquid["optimal_entry_min"] == plan_illiquid["optimal_entry_min"]
    assert plan_liquid["optimal_entry_max"] == plan_illiquid["optimal_entry_max"]
    assert plan_liquid["stop_loss"] == plan_illiquid["stop_loss"]
    assert plan_liquid["take_profit_1"] == plan_illiquid["take_profit_1"]
    assert plan_liquid["take_profit_2"] == plan_illiquid["take_profit_2"]
    assert plan_liquid["risk_reward_ratio"] == plan_illiquid["risk_reward_ratio"]
    assert plan_liquid["setup_pattern"] == plan_illiquid["setup_pattern"]

    # Only observational execution metadata differs
    assert plan_liquid["liquidity_defense"]["liquidity_grade"] == "HIGH_TRADING_LIQUIDITY"
    assert plan_illiquid["liquidity_defense"]["liquidity_grade"] == "EXECUTION_RISK"
    assert plan_illiquid["execution_hazard"] is True

def test_extreme_cases_resilience():
    """Test extreme market data edge cases to ensure zero crashes or unhandled exceptions."""
    dates = pd.date_range("2026-01-01", periods=30)

    # 1. Zero volume series
    df_zero_vol = pd.DataFrame({"Close": [10.0]*30, "Volume": [0.0]*30}, index=dates)
    res1 = LiquidityGuard.evaluate_liquidity(df_zero_vol, 10.0)
    assert res1["liquidity_grade"] == "EXECUTION_RISK"
    assert res1["adv_20d_usd"] == 0.0

    # 2. Missing Volume column entirely
    df_no_vol = pd.DataFrame({"Close": [10.0]*30}, index=dates)
    res2 = LiquidityGuard.evaluate_liquidity(df_no_vol, 10.0)
    assert res2["liquidity_grade"] == "EXECUTION_RISK"

    # 3. NaN and Inf in prices and volumes
    df_nan = pd.DataFrame({
        "Close": [np.nan, np.inf, 10.0, 11.0, 12.0] * 6,
        "Volume": [np.inf, np.nan, 1000.0, 2000.0, 3000.0] * 6
    }, index=dates)
    res3 = LiquidityGuard.evaluate_liquidity(df_nan, 12.0)
    assert res3["liquidity_grade"] in ["HIGH_TRADING_LIQUIDITY", "MODERATE_TRADING_LIQUIDITY", "EXECUTION_RISK"]
    assert not np.isnan(res3["adv_20d_usd"])

    # 4. Single session DataFrame
    df_single = pd.DataFrame({"Close": [10.0], "Volume": [1000]}, index=pd.date_range("2026-01-01", periods=1))
    res4 = LiquidityGuard.evaluate_liquidity(df_single, 10.0)
    assert res4["liquidity_grade"] == "UNKNOWN_LIQUIDITY" # Safe fallback (< 3 sessions)
    assert res4["badge_color"] == "slate"

    # 5. Enormous 100x volume spike
    vols = [10_000]*29 + [1_000_000]
    df_spike = pd.DataFrame({"Close": [10.0]*30, "Volume": vols}, index=dates)
    res5 = LiquidityGuard.evaluate_liquidity(df_spike, 10.0)
    assert res5["is_volume_spike"] is True
    assert res5["volume_spike_ratio"] > 15.0

    # 6. Participation rate calculation
    rate = LiquidityGuard.estimate_participation_rate(order_size_usd=10_000, adv_20d_usd=1_000_000)
    assert abs(rate - 0.01) < 1e-6


def test_amihud_monotonicity_volume():
    """
    Mathematical Monotonicity Invariant 1:
    Holding price path and return constant, Amihud ILLIQ must be strictly decreasing in volume:
    Volume_High > Volume_Med > Volume_Low  =>  ILLIQ_High < ILLIQ_Med < ILLIQ_Low
    """
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [100.0 * (1.01 ** i) for i in range(30)] # ~1% daily return

    df_high_vol = pd.DataFrame({"Close": prices, "Volume": [500_000]*30}, index=dates)
    df_med_vol = pd.DataFrame({"Close": prices, "Volume": [50_000]*30}, index=dates)
    df_low_vol = pd.DataFrame({"Close": prices, "Volume": [5_000]*30}, index=dates)

    res_high = LiquidityGuard.evaluate_liquidity(df_high_vol, prices[-1])
    res_med = LiquidityGuard.evaluate_liquidity(df_med_vol, prices[-1])
    res_low = LiquidityGuard.evaluate_liquidity(df_low_vol, prices[-1])

    assert res_high["amihud_illiq"] < res_med["amihud_illiq"] < res_low["amihud_illiq"]
    assert res_high["amihud_illiq_scaled"] < res_med["amihud_illiq_scaled"] < res_low["amihud_illiq_scaled"]


def test_amihud_monotonicity_return():
    """
    Mathematical Monotonicity Invariant 2:
    Holding dollar volume constant, Amihud ILLIQ must be strictly increasing in absolute return:
    |Return_High| > |Return_Low|  =>  ILLIQ_High > ILLIQ_Low
    """
    dates = pd.date_range("2026-01-01", periods=30)
    vol = [10_000]*30

    # Low volatility price path: 0.1% daily return
    prices_low_vol = [100.0 * (1.001 ** i) for i in range(30)]
    df_low_ret = pd.DataFrame({"Close": prices_low_vol, "Volume": vol}, index=dates)

    # High volatility price path: 2.5% daily return
    prices_high_vol = [100.0 * (1.025 ** i) for i in range(30)]
    df_high_ret = pd.DataFrame({"Close": prices_high_vol, "Volume": vol}, index=dates)

    res_low = LiquidityGuard.evaluate_liquidity(df_low_ret, 100.0)
    res_high = LiquidityGuard.evaluate_liquidity(df_high_ret, 100.0)

    assert res_high["amihud_illiq"] > res_low["amihud_illiq"]
    assert res_high["amihud_illiq_scaled"] > res_low["amihud_illiq_scaled"]
