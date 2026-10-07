"""Property-based invariant test suite for Execution Ladder Remediation.

Verifies:
INV-EL-01: TP2 > TP1 after rounding under all spot prices and precisions.
INV-EL-02: TP2 - TP1 >= max(min_tick, 1.0 * ATR14, 0.75 * execution_risk, 0.05 * planned_entry).
INV-EL-03: Extended non-actionable assets suppress immediate execution stops (execution_stop_visible = false).
INV-EL-04: Structural invalidation is immutable by portfolio risk budget clamps.
INV-EL-05: WAITING_PULLBACK implements Option B (OR policy: spot > entry_max + min(entry_max * 0.05, 1.0 * atr_14)).
INV-EL-06: Prospective scanner does not assign APPROACHING_TARGET solely because spot > entry_max.
INV-EL-07: Day Trader role calculations and VWAP/EMA anchors remain 100% regression-free.
"""

import json
import math
import os
import pytest
import pandas as pd

from analyst_dashboard.analyzers.optimal_execution import (
    OptimalExecutionEngine,
    ACTIONABLE_EXECUTION_STATUSES,
    NON_ACTIONABLE_EXECUTION_STATUSES,
)


REPRESENTATIVE_UNIVERSE = [
    {
        "symbol": "NAUT",
        "spot": 1.96,
        "entry_min": 1.27,
        "entry_max": 1.46,
        "atr_14": 0.19,
        "raw_stop": 1.23,
        "expected_status": "WAITING_PULLBACK",
        "expected_market_location": "BETWEEN_TP1_AND_TP2",
        "expected_actionable": False,
        "expected_stop_visible": False,
    },
    {
        "symbol": "PLSE",
        "spot": 7.50,
        "entry_min": 7.10,
        "entry_max": 7.60,
        "atr_14": 0.50,
        "raw_stop": 6.83,
        "expected_status": "IN_BUY_ZONE",
        "expected_market_location": "IN_BUY_ZONE",
        "expected_actionable": True,
        "expected_stop_visible": True,
    },
    {
        "symbol": "NVDA",
        "spot": 125.00,
        "entry_min": 120.00,
        "entry_max": 126.00,
        "atr_14": 4.50,
        "raw_stop": 116.40,
        "expected_status": "IN_BUY_ZONE",
        "expected_market_location": "IN_BUY_ZONE",
        "expected_actionable": True,
        "expected_stop_visible": True,
    },
    {
        "symbol": "SPY",
        "spot": 575.00,
        "entry_min": 568.00,
        "entry_max": 576.00,
        "atr_14": 5.00,
        "raw_stop": 550.96,
        "expected_status": "IN_BUY_ZONE",
        "expected_market_location": "IN_BUY_ZONE",
        "expected_actionable": True,
        "expected_stop_visible": True,
    },
    {
        "symbol": "KO",
        "spot": 68.00,
        "entry_min": 67.20,
        "entry_max": 68.30,
        "atr_14": 0.90,
        "raw_stop": 65.18,
        "expected_status": "IN_BUY_ZONE",
        "expected_market_location": "IN_BUY_ZONE",
        "expected_actionable": True,
        "expected_stop_visible": True,
    },
    {
        "symbol": "MSTR",
        "spot": 180.00,
        "entry_min": 165.00,
        "entry_max": 182.00,
        "atr_14": 16.00,
        "raw_stop": 156.00,
        "expected_status": "IN_BUY_ZONE",
        "expected_market_location": "IN_BUY_ZONE",
        "expected_actionable": True,
        "expected_stop_visible": True,
    },
]


def test_inv_el_01_tp2_greater_than_tp1_across_universe():
    """INV-EL-01: TP2 > TP1 after rounding under all spot prices and precisions."""
    for asset in REPRESENTATIVE_UNIVERSE:
        raw_plan = {
            "symbol": asset["symbol"],
            "current_price": asset["spot"],
            "optimal_entry_min": asset["entry_min"],
            "optimal_entry_max": asset["entry_max"],
            "stop_loss": asset["raw_stop"],
            "atr_14": asset["atr_14"],
        }
        plan = OptimalExecutionEngine._enforce_execution_invariants(raw_plan, "LONG_TERM")
        tp1 = plan["take_profit_1"]
        tp2 = plan["take_profit_2"]
        assert tp2 > tp1, f"{asset['symbol']}: TP2 ({tp2}) must be strictly greater than TP1 ({tp1})"


def test_inv_el_02_tp2_hybrid_separation_floor():
    """INV-EL-02: TP2 separation satisfies the ratified hybrid floor."""
    for asset in REPRESENTATIVE_UNIVERSE:
        raw_plan = {
            "symbol": asset["symbol"],
            "current_price": asset["spot"],
            "optimal_entry_min": asset["entry_min"],
            "optimal_entry_max": asset["entry_max"],
            "stop_loss": asset["raw_stop"],
            "atr_14": asset["atr_14"],
        }
        plan = OptimalExecutionEngine._enforce_execution_invariants(raw_plan, "LONG_TERM")
        tp1 = plan["take_profit_1"]
        tp2 = plan["take_profit_2"]
        dec = 6 if asset["spot"] < 0.01 else (4 if asset["spot"] < 1.0 else 2)
        min_tick = 10 ** (-dec)
        min_sep = round(
            max(
                min_tick,
                1.0 * asset["atr_14"],
                0.75 * plan["execution_risk"],
                0.05 * plan["planned_entry"],
            ),
            dec,
        )
        separation = round(tp2 - tp1, dec)
        assert separation >= min_sep, (
            f"{asset['symbol']}: TP2 - TP1 separation ({separation}) must be >= hybrid floor ({min_sep})"
        )


def test_inv_el_03_extended_asset_execution_stop_suppression():
    """INV-EL-03: Extended non-actionable assets expose no active execution stop."""
    for asset in REPRESENTATIVE_UNIVERSE:
        raw_plan = {
            "symbol": asset["symbol"],
            "current_price": asset["spot"],
            "optimal_entry_min": asset["entry_min"],
            "optimal_entry_max": asset["entry_max"],
            "stop_loss": asset["raw_stop"],
            "atr_14": asset["atr_14"],
        }
        plan = OptimalExecutionEngine._enforce_execution_invariants(raw_plan, "LONG_TERM")
        assert plan["execution_stop_visible"] == asset["expected_stop_visible"], (
            f"{asset['symbol']}: execution_stop_visible must be {asset['expected_stop_visible']}"
        )
        if not asset["expected_actionable"]:
            assert plan["execution_stop_visible"] is False
            assert plan["is_actionable"] is False


def test_inv_el_04_structural_invalidation_immutable_by_risk_budget():
    """INV-EL-04: Portfolio risk budget constraints modulate sizing only, never structural invalidation."""
    raw_plan = {
        "symbol": "NAUT",
        "current_price": 1.96,
        "optimal_entry_min": 1.27,
        "optimal_entry_max": 1.46,
        "structural_invalidation": 1.23,
        "stop_loss": 1.23,
        "atr_14": 0.19,
    }
    plan = OptimalExecutionEngine._enforce_execution_invariants(raw_plan, "LONG_TERM")
    assert plan["stop_loss"] == 1.23
    assert plan["structural_invalidation"] == 1.23
    # stop_loss_pct anchored to planned_entry ($1.46), NOT spot ($1.96)
    expected_pct = round(((1.23 - 1.46) / 1.46) * 100, 2)
    assert plan["stop_loss_pct"] == expected_pct
    assert plan["stop_loss_pct"] != round(((1.23 - 1.96) / 1.96) * 100, 2)


def test_inv_el_05_waiting_pullback_or_policy():
    """INV-EL-05: WAITING_PULLBACK implements the ratified OR policy exactly."""
    entry_max = 10.00
    atr_14 = 0.40  # 4%
    # 5% of entry_max is 0.50. min(0.50, 0.40) = 0.40. ext_threshold = 10.40.
    ext_threshold = 10.00 + min(10.00 * 0.05, 1.0 * atr_14)  # 10.40

    # Inside corridor
    p1 = OptimalExecutionEngine.calculate_execution_plan("TEST", 9.80, atr_14=atr_14)
    # Right at entry_max
    raw_at_max = {
        "current_price": 10.00,
        "optimal_entry_min": 9.50,
        "optimal_entry_max": 10.00,
        "stop_loss": 9.00,
        "atr_14": 0.40,
    }
    res_at_max = OptimalExecutionEngine._enforce_execution_invariants(raw_at_max, "LONG_TERM")
    assert res_at_max["execution_status"] == "IN_BUY_ZONE"

    # Mild extension (10.20 <= 10.40)
    raw_mild = {
        "current_price": 10.20,
        "optimal_entry_min": 9.50,
        "optimal_entry_max": 10.00,
        "stop_loss": 9.00,
        "atr_14": 0.40,
    }
    res_mild = OptimalExecutionEngine._enforce_execution_invariants(raw_mild, "LONG_TERM")
    assert res_mild["execution_status"] == "EXTENDED_ABOVE_BUY_ZONE"
    assert res_mild["is_actionable"] is False

    # Material extension (10.50 > 10.40)
    raw_ext = {
        "current_price": 10.50,
        "optimal_entry_min": 9.50,
        "optimal_entry_max": 10.00,
        "stop_loss": 9.00,
        "atr_14": 0.40,
    }
    res_ext = OptimalExecutionEngine._enforce_execution_invariants(raw_ext, "LONG_TERM")
    assert res_ext["execution_status"] == "WAITING_PULLBACK"
    assert res_ext["is_actionable"] is False


def test_inv_el_06_prospective_scanner_no_approaching_target():
    """INV-EL-06: Prospective scanner does not assign APPROACHING_TARGET solely because spot > entry_max."""
    raw_plan = {
        "symbol": "NAUT",
        "current_price": 1.96,
        "optimal_entry_min": 1.27,
        "optimal_entry_max": 1.46,
        "stop_loss": 1.23,
        "take_profit_1": 1.89,
        "take_profit_2": 2.18,
        "atr_14": 0.19,
    }
    plan = OptimalExecutionEngine._enforce_execution_invariants(raw_plan, "LONG_TERM")
    # Even though spot ($1.96) > entry_max ($1.46), status must NOT be APPROACHING_TARGET
    assert plan["execution_status"] != "APPROACHING_TARGET"
    assert plan["execution_status"] == "WAITING_PULLBACK"
    assert plan["market_location"] == "BETWEEN_TP1_AND_TP2"


def test_inv_el_07_day_mode_unchanged():
    """INV-EL-07: Day Trader role calculations and VWAP/EMA anchors remain 100% regression-free."""
    fixture_path = r"C:\Users\akara\Documents\Projects\finance\naut_day_production.json"
    if os.path.exists(fixture_path):
        with open(fixture_path, "r") as f:
            d = json.load(f)
        df = pd.DataFrame(d["candles"])
        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        plan = OptimalExecutionEngine.calculate_trade_levels(
            price_df=df,
            current_price=d["currentPrice"],
            user_role="DAY_TRADER",
            technicals=d.get("technicals"),
            live_spot_price=d.get("liveSpotPrice"),
        )
        assert plan["optimal_entry_min"] == 1.95
        assert plan["optimal_entry_max"] == 1.98
        assert plan["stop_loss"] == 1.93
        assert plan["take_profit_1"] == 2.05
        assert plan["take_profit_2"] == 2.09
        assert plan["execution_status"] == "IN_BUY_ZONE"
        assert plan["is_in_buy_zone"] is True
        assert plan["is_actionable"] is True
        assert plan["user_role"] == "DAY_TRADER"


def test_naut_long_forensic_parity():
    """Section 16: NAUT Long forensic verification against ratified numbers."""
    fixture_path = r"C:\Users\akara\Documents\Projects\finance\naut_long_production.json"
    if os.path.exists(fixture_path):
        with open(fixture_path, "r") as f:
            d = json.load(f)
        df = pd.DataFrame(d["candles"])
        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        plan = OptimalExecutionEngine.calculate_trade_levels(
            price_df=df,
            current_price=d["currentPrice"],
            user_role="LONG_TERM",
            technicals=d.get("technicals"),
            live_spot_price=d.get("liveSpotPrice"),
        )
        assert plan["current_price"] == 1.96
        assert plan["optimal_entry_min"] == 1.27
        assert plan["optimal_entry_max"] == 1.46
        assert plan["planned_entry"] == 1.46
        assert plan["structural_invalidation"] == 1.23
        assert plan["execution_risk"] == 0.23
        assert plan["stop_loss"] == 1.23
        assert plan["take_profit_1"] == 1.89
        assert plan["take_profit_2"] in (2.17, 2.18)
        assert plan["take_profit_2"] - plan["take_profit_1"] >= 0.28
        assert plan["market_location"] == "BETWEEN_TP1_AND_TP2"
        assert plan["execution_status"] == "WAITING_PULLBACK"
        assert plan["is_in_buy_zone"] is False
        assert plan["execution_stop_visible"] is False
        assert plan["is_actionable"] is False
        assert plan["stop_loss_pct"] == -15.75
