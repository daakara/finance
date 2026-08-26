"""Tests for OptimalExecutionEngine (Minervini VCP, Turtle ATR & Raschke 20 EMA levels)."""

import pytest
import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine

def test_optimal_execution_engine_fallback():
    empty_df = pd.DataFrame()
    plan = OptimalExecutionEngine.calculate_trade_levels(empty_df, current_price=100.0, user_role="LONG_TERM")
    assert plan["current_price"] == 100.0
    assert plan["stop_loss"] < 100.0
    assert plan["take_profit_1"] > 100.0
    assert plan["take_profit_2"] > plan["take_profit_1"]
    assert plan["risk_reward_ratio"] >= 1.0

def test_optimal_execution_engine_long_term():
    prices = [100 + i * 0.5 + np.sin(i) for i in range(60)]
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 1.0 for p in prices],
        "Low": [p - 1.0 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 60,
    })
    curr = float(prices[-1])
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=curr, user_role="LONG_TERM")
    assert plan["stop_loss"] < curr
    assert plan["take_profit_1"] > curr
    assert plan["take_profit_2"] > plan["take_profit_1"]
    assert "Minervini" in plan["setup_pattern"]
    assert plan["risk_reward_ratio"] >= 1.0

def test_optimal_execution_engine_day_trader():
    prices = [150 + i * 0.2 for i in range(30)]
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 0.5 for p in prices],
        "Low": [p - 0.5 for p in prices],
        "Close": prices,
        "Volume": [50000] * 30,
    })
    curr = float(prices[-1])
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=curr, user_role="DAY_TRADER")
    assert plan["stop_loss"] < curr
    assert plan["take_profit_1"] > curr
    assert "Raschke" in plan["setup_pattern"]
    assert plan["risk_reward_ratio"] >= 1.0