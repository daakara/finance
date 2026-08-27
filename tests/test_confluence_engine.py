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


if __name__ == "__main__":
    test_confluence_scoring()
    test_position_sizing_math()
    print("[PASS] ALL CONFLUENCE & POSITION SIZING TESTS PASSED SUCCESSFULLY")
