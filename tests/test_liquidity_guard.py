import pytest
import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.liquidity_guard import LiquidityGuard
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine

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
    df = pd.DataFrame({"Close": prices, "Volume": volumes, "High": [p + 0.5 for p in prices], "Low": [p - 0.5 for p in prices]}, index=dates)
    
    res = LiquidityGuard.evaluate_liquidity(df, 103.0)
    assert res["liquidity_grade"] == "INSTITUTIONAL"
    assert res["adv_20d_usd"] > 2_000_000.0
    assert res["suppress_buy_zone"] is False
    assert res["execution_hazard"] is False
    assert res["badge_color"] == "emerald"

def test_liquidity_guard_trap():
    # 30 days of micro volume ($20K ADV)
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [2.0 for _ in range(30)]
    volumes = [10_000 for _ in range(30)] # $20K ADV
    df = pd.DataFrame({"Close": prices, "Volume": volumes, "High": [2.1 for _ in range(30)], "Low": [1.9 for _ in range(30)]}, index=dates)
    
    res = LiquidityGuard.evaluate_liquidity(df, 2.0)
    assert res["liquidity_grade"] == "TRAP"
    assert res["suppress_buy_zone"] is True
    assert res["execution_hazard"] is True
    assert res["badge_color"] == "rose"

def test_optimal_execution_integration_institutional():
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [100.0 + (i * 0.1) for i in range(30)]
    volumes = [500_000 for _ in range(30)]
    df = pd.DataFrame({"Close": prices, "Volume": volumes, "High": [p + 0.5 for p in prices], "Low": [p - 0.5 for p in prices]}, index=dates)
    
    plan = OptimalExecutionEngine.calculate_trade_levels(df, 103.0)
    assert "liquidity_defense" in plan
    assert plan["liquidity_defense"]["liquidity_grade"] == "INSTITUTIONAL"
    assert plan.get("execution_hazard") is not True

def test_optimal_execution_integration_trap():
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [2.0 for _ in range(30)]
    volumes = [5_000 for _ in range(30)] # $10K ADV -> TRAP
    df = pd.DataFrame({"Close": prices, "Volume": volumes, "High": [2.1 for _ in range(30)], "Low": [1.9 for _ in range(30)]}, index=dates)
    
    plan = OptimalExecutionEngine.calculate_trade_levels(df, 2.0)
    assert "liquidity_defense" in plan
    assert plan["liquidity_defense"]["liquidity_grade"] == "TRAP"
    assert plan["execution_hazard"] is True
