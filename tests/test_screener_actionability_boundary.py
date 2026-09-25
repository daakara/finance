"""Targeted Contract Test: Screener Structural Discovery Boundary.

Verifies the architectural invariant:
1. Screener represents structural discovery / screening geometry only.
2. Screener CANNOT declare canonical trade actionability (isActionable=False, canSizeTrade=False).
3. Case A: Price strictly within canonical execution corridor.
4. Case B (LNTH Condition): Price outside canonical corridor (99.72 vs [100.15, 100.94])
   but inside screener discovery tolerance (abs(99.72 - 100.94) / 99.72 = 0.0122 <= 0.015).
   Requires: SCREENING_NEAR_ZONE = True, CANONICAL_ACTIONABLE = False.
5. Case C: Price well outside both corridors.
"""

import pytest
import pandas as pd
from unittest.mock import patch
from api.routes.screener import run_screener_get
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState

pytestmark = pytest.mark.tier2b


def test_screener_authority_boundary_cases():
    """Verify that screener discovery output never declares canonical actionability."""
    dates = pd.date_range("2026-01-01", periods=60)
    prices = [100.0 + (i * 0.1) for i in range(60)]
    candles = [
        {"time": str(d), "open": p, "high": p + 0.5, "low": p - 0.5, "close": p, "volume": 500000}
        for d, p in zip(dates, prices)
    ]

    # ── Case A: Price strictly within canonical corridor ─────────────────────
    # Screener output must still fail-closed on actionability because execution requires DecisionTrace
    mock_exec_a = {
        "optimal_entry_min": 100.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 97.0,
        "take_profit_1": 108.0,
        "take_profit_2": 114.0,
        "risk_reward_ratio": 2.5,
        "setup_pattern": "Minervini VCP",
        "entry_thesis": "Corridor breakout",
        "atr_14": 2.0,
        "execution_status": "IN_BUY_ZONE",
    }
    with patch("api.routes.screener.market_db.get_latest_price", return_value={"currentPrice": 101.0}):
        with patch("api.routes.screener.optimal_engine.calculate_trade_levels", return_value=mock_exec_a):
            data = run_screener_get(filter_type="all", custom_tickers="CASE_A")
            cand = data["candidates"][0]
            assert cand["isActionable"] is False, "Case A: Screener cannot declare isActionable=True"
            assert cand["canSizeTrade"] is False, "Case A: Screener cannot declare canSizeTrade=True"

    # ── Case B: The LNTH Condition (Outside canonical corridor, inside 1.5% screener tolerance)
    # Price = 99.72, Entry Min = 100.15, Entry Max = 100.94
    # Screener detects candidate within discovery tolerance, but CANONICAL actionability is FALSE
    mock_exec_b = {
        "optimal_entry_min": 100.15,
        "optimal_entry_max": 100.94,
        "stop_loss": 97.50,
        "take_profit_1": 108.0,
        "take_profit_2": 114.0,
        "risk_reward_ratio": 2.8,
        "setup_pattern": "Minervini VCP",
        "entry_thesis": "Stage 2 breakout",
        "atr_14": 2.5,
        "execution_status": "WAITING_PULLBACK",
    }
    with patch("api.routes.screener.market_db.get_latest_price", return_value={"currentPrice": 99.72}):
        with patch("api.routes.screener.optimal_engine.calculate_trade_levels", return_value=mock_exec_b):
            data = run_screener_get(filter_type="all", custom_tickers="LNTH")
            cand = data["candidates"][0]

            # Screener discovery context:
            assert cand["screeningGeometry"] == "WITHIN_TOLERANCE", "Case B: Must identify within discovery tolerance"
            assert cand["isActionable"] is False, "Case B: Screener cannot declare isActionable=True"
            assert cand["canSizeTrade"] is False, "Case B: Screener cannot declare canSizeTrade=True"
            assert cand["decisionState"] != "ACTIONABLE_SETUP", "Case B: Decision state must not be ACTIONABLE_SETUP"

            # Verify against Single-Asset Analysis Canonical Path:
            opt_engine = OptimalExecutionEngine()
            df = pd.DataFrame([{
                "Open": c["open"], "High": c["high"], "Low": c["low"], "Close": c["close"], "Volume": c["volume"]
            } for c in candles], index=pd.to_datetime([c["time"] for c in candles]))
            plan = opt_engine.calculate_trade_levels(df, 99.72, user_role="LONG_TERM")
            # In canonical execution, 99.72 is outside the buy zone corridor [100.15, 100.94]
            assert plan["execution_status"] in ["WAITING_PULLBACK", "PULLBACK_SUPPORT"], "Canonical execution status must be pullback/waiting"
            assert plan["is_actionable"] is False, "Canonical analysis is_actionable must be False"

    # ── Case C: Price well outside both corridors ─────────────────────────────
    mock_exec_c = {
        "optimal_entry_min": 100.15,
        "optimal_entry_max": 100.94,
        "stop_loss": 97.50,
        "take_profit_1": 108.0,
        "take_profit_2": 114.0,
        "risk_reward_ratio": 2.8,
        "setup_pattern": "Minervini VCP",
        "entry_thesis": "Stage 2 breakout",
        "atr_14": 2.5,
        "execution_status": "WAITING_PULLBACK",
    }
    with patch("api.routes.screener.market_db.get_latest_price", return_value={"currentPrice": 85.0}):
        with patch("api.routes.screener.optimal_engine.calculate_trade_levels", return_value=mock_exec_c):
            data = run_screener_get(filter_type="all", custom_tickers="CASE_C")
            cand = data["candidates"][0]
            assert cand["screeningGeometry"] == "OUTSIDE_TOLERANCE"
            assert cand["isActionable"] is False
            assert cand["canSizeTrade"] is False


def test_screener_never_declares_buy_zone_confirmed():
    """Verify that screener status labels and decision labels never claim confirmed buy zone."""
    data = run_screener_get(filter_type="all")
    for cand in data.get("candidates", []):
        assert cand["isActionable"] is False, f"{cand['symbol']} emitted isActionable=True"
        assert cand["canSizeTrade"] is False, f"{cand['symbol']} emitted canSizeTrade=True"
        assert "CONFIRMED" not in (cand.get("decisionStateLabel") or "").upper(), f"{cand['symbol']} claimed confirmed in decision label"
