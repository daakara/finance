import pytest
import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine


def _generate_synthetic_candles(
    count: int = 60,
    base_price: float = 100.0,
    daily_gain: float = 0.5,
    final_candle_type: str = "CONFIRMED_BOUNCE"
) -> pd.DataFrame:
    prices = [base_price + i * daily_gain for i in range(count)]
    opens = [p - 0.2 for p in prices]
    highs = [p + 0.8 for p in prices]
    lows = [p - 0.8 for p in prices]
    closes = list(prices)

    if final_candle_type == "FALLING_KNIFE_UNCONFIRMED":
        opens[-1] = closes[-2] + 0.5
        highs[-1] = opens[-1] + 0.2
        lows[-1] = closes[-2] - 2.5
        closes[-1] = lows[-1] + 0.05
    elif final_candle_type == "CONFIRMED_BOUNCE":
        opens[-1] = closes[-2] - 0.5
        lows[-1] = closes[-2] - 2.0
        highs[-1] = closes[-2] + 1.0
        closes[-1] = highs[-1] - 0.1

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": [1_000_000] * count,
    })


def test_pullback_confirmation_trigger_gates_actionable_setup():
    df_knife = _generate_synthetic_candles(60, base_price=100.0, final_candle_type="FALLING_KNIFE_UNCONFIRMED")
    curr_knife = float(df_knife["Close"].iloc[-1])
    plan_knife = OptimalExecutionEngine.calculate_trade_levels(df_knife, current_price=curr_knife, user_role="LONG_TERM")
    
    if plan_knife["optimal_entry_min"] <= curr_knife <= plan_knife["optimal_entry_max"]:
        assert plan_knife["execution_status"] == "IN_BUY_ZONE_AWAITING_TRIGGER"
        
        trace = DecisionTraceEngine.generate_trace(
            symbol="TEST_KNIFE",
            current_price=curr_knife,
            candle_count=60,
            freshness_status="FRESH_PROD",
            has_fundamentals=True,
            confluence={"confluenceScore": 82.0, "pillars": [], "warnings": []},
            technicals={"trend": "BULLISH"},
            optimal_execution=plan_knife,
        )
        assert trace["decisionState"] == DecisionState.VALID_SETUP.value
        assert "awaiting reversal/stabilization confirmation" in trace["disqualificationReason"]
        assert trace["gatekeeperVerdict"]["canSizeTrade"] is False

    df_bounce = _generate_synthetic_candles(60, base_price=100.0, final_candle_type="CONFIRMED_BOUNCE")
    curr_bounce = float(df_bounce["Close"].iloc[-1])
    plan_bounce = OptimalExecutionEngine.calculate_trade_levels(df_bounce, current_price=curr_bounce, user_role="LONG_TERM")
    
    if plan_bounce["optimal_entry_min"] <= curr_bounce <= plan_bounce["optimal_entry_max"]:
        assert plan_bounce["execution_status"] == "IN_BUY_ZONE"
        
        trace = DecisionTraceEngine.generate_trace(
            symbol="TEST_BOUNCE",
            current_price=curr_bounce,
            candle_count=60,
            freshness_status="FRESH_PROD",
            has_fundamentals=True,
            confluence={"confluenceScore": 82.0, "pillars": [], "warnings": []},
            technicals={"trend": "BULLISH"},
            optimal_execution=plan_bounce,
        )
        assert trace["decisionState"] == DecisionState.ACTIONABLE_SETUP.value
        assert trace["disqualificationReason"] is None
        assert trace["gatekeeperVerdict"]["canSizeTrade"] is True


def test_structural_stop_anchoring_and_realistic_targets():
    df = _generate_synthetic_candles(60, base_price=150.0, final_candle_type="CONFIRMED_BOUNCE")
    raw_spot = float(df["Close"].iloc[-1])
    initial_plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=raw_spot, user_role="LONG_TERM")
    
    # Test execution levels when entering inside the buy zone
    buy_zone_spot = (initial_plan["optimal_entry_min"] + initial_plan["optimal_entry_max"]) / 2.0
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=buy_zone_spot, user_role="LONG_TERM")

    stop = plan["stop_loss"]
    tp1 = plan["take_profit_1"]
    tp2 = plan["take_profit_2"]
    rr = plan["risk_reward_ratio"]

    # Invariants
    assert stop < plan["optimal_entry_min"] <= buy_zone_spot <= plan["optimal_entry_max"]
    assert plan["optimal_entry_max"] < tp1 < tp2
    assert rr >= 1.85

    # Risk distance when inside buy zone should be bounded between 3.5% and 6.5% for swing
    risk_pct = abs((stop - buy_zone_spot) / buy_zone_spot) * 100.0
    assert 3.5 <= risk_pct <= 6.5, f"Risk percent {risk_pct}% outside calibrated 3.5%-6.5% window"

    # TP1 should be realistically calibrated (typically <= 18% above spot for swing setups)
    tp1_pct = ((tp1 - buy_zone_spot) / buy_zone_spot) * 100.0
    assert 4.0 <= tp1_pct <= 18.0, f"TP1 percent {tp1_pct}% outside realistic 4%-18% envelope"


def test_boundary_corridor_state_transitions():
    df = _generate_synthetic_candles(60, base_price=200.0, final_candle_type="CONFIRMED_BOUNCE")
    spot = float(df["Close"].iloc[-1])
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=spot, user_role="LONG_TERM")
    
    emin = plan["optimal_entry_min"]
    emax = plan["optimal_entry_max"]

    below_plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=emin - 1.0, user_role="LONG_TERM")
    assert below_plan["execution_status"] == "WAITING_PULLBACK"

    mid_price = (emin + emax) / 2.0
    mid_plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=mid_price, user_role="LONG_TERM")
    assert mid_plan["execution_status"] == "IN_BUY_ZONE"

    above_plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=emax + 1.0, user_role="LONG_TERM")
    assert above_plan["execution_status"] == "APPROACHING_TARGET"
