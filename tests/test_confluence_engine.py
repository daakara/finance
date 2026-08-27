"""Unit tests for ConfluenceEngine and Position Sizer."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine


def test_confluence_scoring():
    engine = ConfluenceEngine()

    # 1. High Conviction Scenario
    high_res = engine.calculate_confluence(
        symbol="LNTH",
        technical_data={"executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.8},
        smart_money_data={"has_insider_buy": True, "has_congress_buy": True},
        fundamental_data={"roic": 36.5, "peg": 0.68, "piotroski_f": 9},
        catalyst_data={"days_to_earnings": 35},
    )

    assert high_res["confluenceScore"] >= 80.0
    assert "EXCEPTIONAL" in high_res["confluenceRating"] or "HIGH" in high_res["confluenceRating"]
    assert len(high_res["reasons"]) >= 3

    # 2. Binary Gap Risk Scenario
    risky_res = engine.calculate_confluence(
        symbol="DUOL",
        technical_data={"executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 1.5},
        smart_money_data={"has_insider_buy": False, "has_congress_buy": False},
        fundamental_data={"roic": 15.0, "peg": 1.5, "piotroski_f": 6},
        catalyst_data={"days_to_earnings": 1},  # Earnings in <24h
    )

    assert risky_res["confluenceScore"] < 50.0
    assert len(risky_res["warnings"]) >= 1


def test_position_sizing_math():
    engine = ConfluenceEngine()

    sizing = engine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,        # $250 max loss
        entry_price=100.0,
        stop_loss=95.0,      # $5 risk per share
        take_profit_1=110.0, # $10 reward per share
    )

    assert sizing["shares"] == 50  # $250 / $5 = 50 shares
    assert sizing["positionValue"] == 5000.0
    assert sizing["actualDollarRisk"] == 250.0
    assert sizing["projectedProfitTp1"] == 500.0
    assert sizing["riskRewardRatio"] == 2.0
    assert sizing["halfKellyOptimalPct"] > 0


def test_confluence_score_bounds():
    """Verify confluence score always strictly clamps between 0.0 and 100.0."""
    engine = ConfluenceEngine()

    # Minimum bound test (all negative/penalizing inputs)
    worst_res = engine.calculate_confluence(
        symbol="TRASH",
        technical_data={"executionStatus": "STOPPED_OUT", "riskRewardRatio": 0.5},
        smart_money_data={"has_insider_buy": False, "has_congress_buy": False},
        fundamental_data={"roic": -10.0, "peg": 4.0, "piotroski_f": 1},
        catalyst_data={"days_to_earnings": 1},
    )
    assert 0.0 <= worst_res["confluenceScore"] <= 100.0, "Score must stay >= 0"

    # Maximum bound test (all positive/boosting inputs)
    best_res = engine.calculate_confluence(
        symbol="GOLD",
        technical_data={"executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 5.0},
        smart_money_data={"has_insider_buy": True, "has_congress_buy": True},
        fundamental_data={"roic": 45.0, "peg": 0.5, "piotroski_f": 9},
        catalyst_data={"days_to_earnings": 60},
    )
    assert best_res["confluenceScore"] <= 100.0, "Score must not exceed 100"


def test_position_sizing_zero_division_guard():
    """Verify position sizer handles invalid/tight stop loss without crashing with zero-division error."""
    engine = ConfluenceEngine()

    # Stop loss equal to or above entry price
    tight_sizing = engine.calculate_position_size(
        account_equity=50000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=100.0, # Zero risk per share (invalid stop)
        take_profit_1=105.0,
    )
    assert "error" in tight_sizing, "Must return error on invalid/zero stop loss distance"
    assert tight_sizing["shares"] == 0


def test_position_sizing_kelly_capping():
    """Verify Fractional Kelly allocation never exceeds statutory 25% safety ceiling."""
    engine = ConfluenceEngine()

    huge_edge_sizing = engine.calculate_position_size(
        account_equity=100000.0,
        risk_pct=2.0,
        entry_price=50.0,
        stop_loss=48.0,
        take_profit_1=100.0, # 25:1 R:R ratio
    )
    assert huge_edge_sizing["halfKellyOptimalPct"] <= 25.0, "Half-Kelly must be capped at 25% portfolio ceiling"


if __name__ == "__main__":
    test_confluence_scoring()
    test_confluence_score_bounds()
    test_position_sizing_math()
    test_position_sizing_zero_division_guard()
    test_position_sizing_kelly_capping()
    print("[PASS] ALL CONFLUENCE & POSITION SIZING INVARIANT TESTS PASSED SUCCESSFULLY")

