import os
import json
import pytest
import pandas as pd
from datetime import datetime, timezone
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger


class DummyDB:
    def __init__(self, candles_map):
        self.candles_map = candles_map

    def get_daily_candles(self, symbol, limit=100):
        return self.candles_map.get(symbol, [])


def test_experiment_ledger_registration_and_immutability(tmp_path):
    ledger_path = str(tmp_path / "test_ledger.json")
    
    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 94.0,
        "stop_loss_pct": -6.0,
        "take_profit_1": 112.0,
        "take_profit_1_pct": 12.0,
        "take_profit_2": 118.0,
        "take_profit_2_pct": 18.0,
        "risk_reward_ratio": 2.0,
        "atr_14": 3.0,
        "setup_pattern": "Minervini VCP",
        "stage_phase": "Stage 2 Advancing",
    }
    
    # 1. Register signal
    rec1 = ExperimentLedger.register_signal(
        symbol="TEST",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL", "sector": "TECH", "asset_class": "US_EQUITY"},
        engine_commit="4e36862",
        engine_tag="v2.4.0-phase24-freeze",
        ledger_path=ledger_path
    )
    
    assert rec1["symbol"] == "TEST"
    assert rec1["entryPrice"] == 100.0
    assert rec1["stopLoss"] == 94.0
    assert rec1["takeProfit1"] == 112.0
    assert rec1["status"] == "OPEN"
    assert rec1["engineVersion"] == "4e36862"
    assert rec1["engineTag"] == "v2.4.0-phase24-freeze"

    # 2. Immutability check: re-registering with different parameters must NOT overwrite
    rec2 = ExperimentLedger.register_signal(
        symbol="TEST",
        entry_price=105.0,  # Corrupted attempt
        opt_exec=opt_exec,
        confluence_score=90.0,
        inputs_meta={"market_regime": "BEAR"},
        ledger_path=ledger_path
    )
    assert rec2["entryPrice"] == 100.0  # Original price preserved
    assert rec2["confluenceScore"] == 85.0  # Original score preserved


def test_experiment_ledger_forward_update_tp1_win(tmp_path):
    ledger_path = str(tmp_path / "test_ledger.json")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 94.0,
        "stop_loss_pct": -6.0,
        "take_profit_1": 110.0,
        "take_profit_1_pct": 10.0,
        "take_profit_2": 118.0,
        "take_profit_2_pct": 18.0,
        "risk_reward_ratio": 2.0,
        "atr_14": 3.0,
    }
    
    ExperimentLedger.register_signal(
        symbol="WINNER",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        ledger_path=ledger_path
    )
    
    # Simulate 3 future days: Day 1 sideways, Day 2 rallies to 108, Day 3 touches 112 (TP1 hit)
    future_candles = [
        {"time": f"{today} 00:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
        {"time": "2099-01-01", "open": 100.5, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1000},
        {"time": "2099-01-02", "open": 101.0, "high": 108.0, "low": 100.0, "close": 107.0, "volume": 1000},
        {"time": "2099-01-03", "open": 107.0, "high": 113.0, "low": 106.0, "close": 111.0, "volume": 1000},
    ]
    
    db = DummyDB({"WINNER": future_candles})
    res = ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)
    assert res["updatedSignals"] == 1
    assert res["openSignals"] == 0  # Position is now resolved
    
    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    assert sig["status"] == "RESOLVED"
    assert sig["forwardTracking"]["tp1Hit"] is True
    assert sig["forwardTracking"]["resolvedOutcome"] == "TP1_WIN"
    assert sig["forwardTracking"]["realizedReturnPct"] == 10.0
    assert sig["forwardTracking"]["maxFavorableExcursionPct"] == 13.0
    assert sig["forwardTracking"]["maxAdverseExcursionPct"] == -2.0


def test_experiment_ledger_governance_scorecard(tmp_path):
    ledger_path = str(tmp_path / "test_ledger.json")
    
    # Create synthetic ledger with 1 Win (+10%) and 1 Loss (-5%)
    synthetic_ledger = {
        "version": "1.0.0",
        "totalActiveSignals": 0,
        "signals": [
            {
                "signalId": "WIN_1",
                "symbol": "WIN",
                "status": "RESOLVED",
                "forwardTracking": {
                    "resolvedOutcome": "TP1_WIN",
                    "realizedReturnPct": 10.0
                }
            },
            {
                "signalId": "LOSS_1",
                "symbol": "LOSS",
                "status": "RESOLVED",
                "forwardTracking": {
                    "resolvedOutcome": "STOP_LOSS",
                    "realizedReturnPct": -5.0
                }
            }
        ]
    }
    ExperimentLedger.save_ledger(synthetic_ledger, ledger_path)
    scorecard = ExperimentLedger.compute_governance_scorecard(ledger_path)
    
    assert scorecard["totalSignals"] == 2
    assert scorecard["resolvedSignals"] == 2
    assert scorecard["winRate"] == 50.0
    assert scorecard["stopRate"] == 50.0
    assert scorecard["avgWinPct"] == 10.0
    assert scorecard["avgLossPct"] == 5.0
    # Expectancy = (0.5 * 10) - (0.5 * 5) = 2.5%
    assert scorecard["expectancyPct"] == 2.5
    # Profit Factor = 10 / 5 = 2.0
    assert scorecard["profitFactor"] == 2.0
    assert scorecard["governanceGates"]["expectancyPositive"] is True
    assert scorecard["governanceGates"]["profitFactorAbove1_5"] is True
