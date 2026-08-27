"""Unit tests for Screener Optimal Execution Scanner & Signal Classification."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine

try:
    import pandas as pd
except ImportError:
    pd = None


def test_optimal_execution_levels():
    engine = OptimalExecutionEngine()
    
    current_price = 106.80
    if pd is not None:
        dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "Open": [100.0 + (i * 0.2) for i in range(30)],
            "High": [102.0 + (i * 0.2) for i in range(30)],
            "Low": [99.0 + (i * 0.2) for i in range(30)],
            "Close": [101.0 + (i * 0.2) for i in range(30)],
            "Volume": [1000000 for _ in range(30)],
        }, index=dates)
    else:
        df = []

    levels = engine.calculate_trade_levels(df, current_price, user_role="LONG_TERM")

    assert "optimal_entry_min" in levels
    assert "optimal_entry_max" in levels
    assert "stop_loss" in levels
    assert "take_profit_1" in levels
    assert "take_profit_2" in levels
    assert "risk_reward_ratio" in levels

    assert levels["stop_loss"] < current_price
    assert levels["take_profit_1"] > current_price
    assert levels["risk_reward_ratio"] > 0


if __name__ == "__main__":
    test_optimal_execution_levels()
    print("[PASS] ALL SCREENER EXECUTION TESTS PASSED SUCCESSFULLY")
