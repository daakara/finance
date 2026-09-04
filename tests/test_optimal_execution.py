import pytest
import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine


def test_optimal_execution_engine_fallback():
    empty_df = pd.DataFrame()
    plan = OptimalExecutionEngine.calculate_trade_levels(empty_df, current_price=100.0, user_role="LONG_TERM")
    assert plan["current_price"] == 100.0
    assert plan["execution_status"] == "INSUFFICIENT_HISTORY"
    assert plan["stop_loss"] is None
    assert plan["take_profit_1"] is None
    assert plan["take_profit_2"] is None
    assert plan["risk_reward_ratio"] is None
    assert plan["optimal_entry_min"] is None
    assert plan["optimal_entry_max"] is None


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
    assert plan["stop_loss"] < plan["optimal_entry_min"]
    assert plan["optimal_entry_min"] <= plan["optimal_entry_max"]
    assert plan["optimal_entry_max"] < plan["take_profit_1"]
    assert plan["take_profit_1"] < plan["take_profit_2"]
    assert "Minervini" in plan["setup_pattern"]
    assert plan["risk_reward_ratio"] >= 1.85


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
    assert plan["stop_loss"] < plan["optimal_entry_min"]
    assert plan["optimal_entry_min"] <= plan["optimal_entry_max"]
    assert plan["optimal_entry_max"] < plan["take_profit_1"]
    assert "Raschke" in plan["setup_pattern"]
    assert plan["risk_reward_ratio"] >= 1.85


def test_calculate_execution_plan_convenience_api_multi_tier():
    """Verify execution ladders across price tiers: micro-cap, mid-cap, mega-cap, and crypto."""
    test_cases = [
        ("PENNY", 1.50, 0.05),
        ("MID", 45.00, 1.20),
        ("MEGA", 450.00, 8.50),
        ("BTC-USD", 62000.00, 2450.00),
    ]
    for sym, price, atr in test_cases:
        for role in ["LONG_TERM", "DAY_TRADER"]:
            plan = OptimalExecutionEngine.calculate_execution_plan(
                symbol=sym,
                price=price,
                atr_14=atr,
                user_role=role,
            )
            assert plan["stop_loss"] > 0, f"{sym} stop_loss must be positive"
            assert plan["stop_loss"] < plan["optimal_entry_min"], f"{sym} Stop < Entry_min"
            assert plan["optimal_entry_min"] <= plan["optimal_entry_max"], f"{sym} Entry_min <= Entry_max"
            assert plan["optimal_entry_max"] < plan["take_profit_1"], f"{sym} Entry_max < TP1"
            assert plan["take_profit_1"] < plan["take_profit_2"], f"{sym} TP1 < TP2"
            assert plan["stop_loss"] < plan["current_price"] < plan["take_profit_1"], f"{sym} Stop < Spot < TP1"
            assert plan["risk_reward_ratio"] >= 1.85, f"{sym} R:R must be >= 1.85 (got {plan['risk_reward_ratio']})"


def test_stage_4_markdown_breakout_pivot_reanchoring():
    """Verify that Stage 4 corrections re-anchor targets from the Breakout Pivot and enforce R:R."""
    prices = [150.0 - i * 0.8 for i in range(60)]  # Clear downtrend from 150 to ~102
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 1.5 for p in prices],
        "Low": [p - 1.5 for p in prices],
        "Close": prices,
        "Volume": [1500000] * 60,
    })
    curr = float(prices[-1])
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=curr, user_role="LONG_TERM")

    assert "Stage 4" in plan["stage_phase"]
    assert plan["breakout_pivot"] is not None
    assert plan["breakout_pivot"] >= curr
    assert plan["take_profit_1"] > plan["optimal_entry_max"]
    assert plan["take_profit_2"] > plan["take_profit_1"]
    assert plan["risk_reward_ratio"] >= 1.85


def test_enforce_execution_invariants_circuit_breaker():
    """Verify self-healing circuit breaker repairs corrupt or inverted inputs."""
    corrupted_plan = {
        "current_price": 100.0,
        "optimal_entry_min": 102.0,  # Corrupted: min > max
        "optimal_entry_max": 98.0,
        "stop_loss": 105.0,         # Corrupted: stop > spot
        "take_profit_1": 95.0,      # Corrupted: tp1 < spot
        "take_profit_2": 90.0,      # Corrupted: tp2 < tp1
        "risk_reward_ratio": 0.5,
    }
    healed = OptimalExecutionEngine._enforce_execution_invariants(corrupted_plan, user_role="LONG_TERM")
    assert healed["stop_loss"] < healed["optimal_entry_min"]
    assert healed["optimal_entry_min"] <= healed["optimal_entry_max"]
    assert healed["optimal_entry_max"] < healed["take_profit_1"]
    assert healed["take_profit_1"] < healed["take_profit_2"]
    assert healed["stop_loss"] < 100.0 < healed["take_profit_1"]
    assert healed["risk_reward_ratio"] >= 1.85


def test_sub_cent_and_non_positive_price_hardening():
    """Verify micro-cap sub-cent ($0.0001, $0.001, $0.01) and zero/negative prices uphold all invariants."""
    test_prices = [-10.0, 0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.50, 0.99]
    for p in test_prices:
        for role in ["LONG_TERM", "DAY_TRADER"]:
            plan = OptimalExecutionEngine.calculate_execution_plan("SUB_CENT", p, user_role=role)
            assert plan["stop_loss"] < plan["optimal_entry_min"], f"Failed for p={p}, role={role}"
            assert plan["optimal_entry_min"] <= plan["optimal_entry_max"], f"Failed for p={p}, role={role}"
            assert plan["optimal_entry_max"] < plan["take_profit_1"], f"Failed for p={p}, role={role}"
            assert plan["take_profit_1"] < plan["take_profit_2"], f"Failed for p={p}, role={role}"
            assert plan["stop_loss"] < plan["current_price"] < plan["take_profit_1"], f"Failed for p={p}, role={role}"
            assert plan["risk_reward_ratio"] >= 1.85, f"R:R below 1.85 for p={p}, role={role}"