import pytest
import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.liquidity_guard import LiquidityGuard

def test_liquidity_guard_fallback():
    res = LiquidityGuard.evaluate_liquidity(pd.DataFrame(), 100.0)
    assert res["liquidity_grade"] == "INSTITUTIONAL"
    assert res["suppress_buy_zone"] is False
    assert res["execution_hazard"] is False

def test_liquidity_guard_institutional():
    # 30 days of high volume, low price volatility
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [100.0 + (i * 0.1) for i in range(30)]
    volumes = [500_000 for _ in range(30)] # $50M ADV
    df = pd.DataFrame({"Close": prices, "Volume": volumes}, index=dates)
    
    res = LiquidityGuard.evaluate_liquidity(df, 103.0)
    assert res["liquidity_grade"] == "INSTITUTIONAL"
    assert res["adv_20d_usd"] > 2_000_000.0
    assert res["suppress_buy_zone"] is False
    assert res["execution_hazard"] is False
    assert res["badge_color"] == "emerald"

def test_liquidity_guard_trap():
    # 30 days of micro volume ($50K ADV)
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [2.0 for _ in range(30)]
    volumes = [10_000 for _ in range(30)] # $20K ADV
    df = pd.DataFrame({"Close": prices, "Volume": volumes}, index=dates)
    
    res = LiquidityGuard.evaluate_liquidity(df, 2.0)
    assert res["liquidity_grade"] == "TRAP"
    assert res["suppress_buy_zone"] is True
    assert res["execution_hazard"] is True
    assert res["badge_color"] == "rose"
