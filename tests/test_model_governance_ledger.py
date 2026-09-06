import os
import json
import pytest
import pandas as pd
from datetime import datetime, timezone
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger

pytestmark = pytest.mark.tier3



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
    assert "governancePrinciple" in scorecard


def test_experiment_ledger_anti_lookahead_audit(tmp_path):
    """Verify that candles occurring before or during signal date cannot be contaminated by future data,
    and modifying future candles leaves initial decisions strictly immutable."""
    ledger_path = str(tmp_path / "test_ledger.json")
    today = "2026-09-04"

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

    # 1. Register signal with explicit signal_date
    rec = ExperimentLedger.register_signal(
        symbol="SHIELD",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        signal_date=today,
        ledger_path=ledger_path
    )

    # Verify initial decision immutability
    assert rec["entryPrice"] == 100.0
    assert rec["stopLoss"] == 94.0
    assert rec["takeProfit1"] == 110.0

    # 2. Feed future candles that drop in the future
    future_candles = [
        {"time": "2026-09-04", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
        {"time": "2026-09-05", "open": 95.0, "high": 96.0, "low": 50.0, "close": 52.0, "volume": 50000},
    ]
    db = DummyDB({"SHIELD": future_candles})
    ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)

    # Reload and assert: future crash resolved outcome as STOP_LOSS, but CANNOT retroactively change initial entry, stop, or confluence
    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    assert sig["entryPrice"] == 100.0
    assert sig["stopLoss"] == 94.0
    assert sig["takeProfit1"] == 110.0
    assert sig["confluenceScore"] == 85.0
    assert sig["status"] == "RESOLVED"
    assert sig["forwardTracking"]["resolvedOutcome"] == "STOP_LOSS"
    assert sig["forwardTracking"]["stopHit"] is True


def test_decision_snapshot_hash_tamper_detection(tmp_path):
    """Verify that tampering with any frozen decision variable raises GOVERNANCE_INTEGRITY_FAILURE."""
    ledger_path = str(tmp_path / "test_ledger.json")
    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 94.0,
        "take_profit_1": 110.0,
        "take_profit_2": 118.0,
        "risk_reward_ratio": 2.0,
        "atr_14": 3.0,
        "liquidity_defense": {
            "liquidity_grade": "HIGH_TRADING_LIQUIDITY",
            "adv_20d_usd": 50_000_000.0,
            "amihud_illiq": 1e-10,
            "amihud_illiq_scaled": 0.0001
        }
    }

    rec = ExperimentLedger.register_signal(
        symbol="HASH_GUARD",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        ledger_path=ledger_path
    )

    assert "decisionSnapshotHash" in rec
    assert rec["decisionSnapshotHash"] is not None
    assert "inputsSnapshotHash" in rec
    assert rec["inputsSnapshotHash"] is not None
    assert rec["liquidityAtSignal"]["liquidity_grade"] == "HIGH_TRADING_LIQUIDITY"

    # 1. Simulate unauthorized post-hoc tampering with stopLoss (decision parameter)
    ledger = ExperimentLedger.load_ledger(ledger_path)
    ledger["signals"][0]["stopLoss"] = 96.0  # Tampered!
    ExperimentLedger.save_ledger(ledger, ledger_path)

    future_candles = [{"time": "2099-01-01", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000}]
    db = DummyDB({"HASH_GUARD": future_candles})
    with pytest.raises(ValueError, match="GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch"):
        ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)

    # Restore stopLoss and simulate tampering with input features (e.g. atr14)
    ledger = ExperimentLedger.load_ledger(ledger_path)
    ledger["signals"][0]["stopLoss"] = 94.0  # Restored
    ledger["signals"][0]["inputs"]["atr14"] = 99.0  # Tampered input!
    ExperimentLedger.save_ledger(ledger, ledger_path)

    with pytest.raises(ValueError, match="GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch"):
        ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)


def test_liquidity_forward_observation_immutability(tmp_path):
    """Verify that forward liquidity updates append without mutating liquidityAtSignal or decision fields."""
    ledger_path = str(tmp_path / "test_ledger.json")
    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 94.0,
        "take_profit_1": 110.0,
        "take_profit_2": 118.0,
        "risk_reward_ratio": 2.0,
        "liquidity_defense": {
            "liquidity_grade": "HIGH_TRADING_LIQUIDITY",
            "adv_20d_usd": 50_000_000.0,
            "amihud_illiq": 1e-10,
            "amihud_illiq_scaled": 0.0001
        }
    }

    rec = ExperimentLedger.register_signal(
        symbol="TIME_OBS",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        ledger_path=ledger_path
    )

    # Record forward observation on session 5
    obs5 = ExperimentLedger.record_liquidity_forward_observation(
        signal_id=rec["signalId"],
        session_index=5,
        liquidity_metrics={
            "adv_20d_usd": 42_000_000.0,
            "amihud_illiq": 1.2e-10,
            "amihud_illiq_scaled": 0.00012,
            "liquidity_grade": "HIGH_TRADING_LIQUIDITY",
            "execution_hazard": False
        },
        ledger_path=ledger_path
    )
    assert obs5["sessionIndex"] == 5

    # Verify reload: liquidityAtSignal is untouched, liquidityForwardObservations has 1 entry
    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    assert sig["liquidityAtSignal"]["adv_20d_usd"] == 50_000_000.0
    assert len(sig["liquidityForwardObservations"]) == 1
    assert sig["liquidityForwardObservations"][0]["adv20dUsd"] == 42_000_000.0
    assert sig["entryPrice"] == 100.0
    assert sig["stopLoss"] == 94.0


def test_multi_horizon_tracking_and_benchmark_relative_returns(tmp_path):
    """Verify multi-horizon tracking at T+1, T+5, T+10, T+20 and benchmark relative returns."""
    ledger_path = str(tmp_path / "test_ledger.json")
    sig_date = "2026-09-01"

    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 85.0,  # Wide stop so it stays open across all 20 sessions
        "stop_loss_pct": -15.0,
        "take_profit_1": 150.0,
        "take_profit_1_pct": 50.0,
        "risk_reward_ratio": 3.33,
        "atr_14": 2.0,
    }

    ExperimentLedger.register_signal(
        symbol="MULTI",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=82.5,
        inputs_meta={"market_regime": "BULL", "sector": "TECHNOLOGY"},
        signal_date=sig_date,
        component_scores={"qualityScore": 90, "growthScore": 85, "technicalScore": 80},
        benchmarks={"spyPrice": 500.0, "rspPrice": 160.0, "sectorBenchmarkSymbol": "XLK"},
        ledger_path=ledger_path
    )

    # Generate 20 daily sessions for MULTI, SPY, RSP, XLK
    multi_candles = [{"time": f"2026-09-{d:02d}", "open": 100.0 + d, "high": 102.0 + d, "low": 99.0 + d, "close": 100.0 + d * 1.0, "volume": 1000} for d in range(2, 22)]
    spy_candles = [{"time": f"2026-09-{d:02d}", "open": 500.0 + d * 0.5, "high": 501.0 + d * 0.5, "low": 499.0 + d * 0.5, "close": 500.0 + d * 0.5, "volume": 10000} for d in range(1, 22)]
    rsp_candles = [{"time": f"2026-09-{d:02d}", "open": 160.0 + d * 0.1, "high": 161.0 + d * 0.1, "low": 159.0 + d * 0.1, "close": 160.0 + d * 0.1, "volume": 5000} for d in range(1, 22)]
    xlk_candles = [{"time": f"2026-09-{d:02d}", "open": 200.0 + d * 0.8, "high": 201.0 + d * 0.8, "low": 199.0 + d * 0.8, "close": 200.0 + d * 0.8, "volume": 8000} for d in range(1, 22)]

    db = DummyDB({
        "MULTI": multi_candles,
        "SPY": spy_candles,
        "RSP": rsp_candles,
        "XLK": xlk_candles,
    })

    res = ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)
    assert res["updatedSignals"] == 1

    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    ft = sig["forwardTracking"]

    # Verify multi-horizon raw returns
    assert ft["return1d"] == 2.0   # Day 2 close = 102.0 -> (102-100)/100 = +2%
    assert ft["return5d"] == 6.0   # Day 6 close = 106.0 -> (106-100)/100 = +6%
    assert ft["return10d"] == 11.0 # Day 11 close = 111.0 -> (111-100)/100 = +11%
    assert ft["return20d"] == 21.0 # Day 21 close = 121.0 -> (121-100)/100 = +21%

    # Verify signal quality separation
    sq = ft["signalQuality"]
    assert sq["directionalAccuracy1d"] is True
    assert sq["directionalAccuracy5d"] is True
    assert sq["directionalAccuracy20d"] is True
    assert sq["rawReturn20d"] == 21.0

    # Verify relative benchmark excess returns vs SPY and RSP
    assert ft["relativeReturns"]["vsSpy"]["20d"] is not None
    assert ft["relativeReturns"]["vsRsp"]["20d"] is not None


def test_signal_quality_vs_trade_construction_premature_stop_out(tmp_path):
    """Verify that a premature stop-out (stop hit, but price subsequently reached TP1)
    is properly identified, decoupling raw directional signal quality from stop execution."""
    ledger_path = str(tmp_path / "test_ledger.json")
    sig_date = "2026-09-01"

    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 95.0,
        "stop_loss_pct": -5.0,
        "take_profit_1": 115.0,
        "take_profit_1_pct": 15.0,
        "risk_reward_ratio": 3.0,
        "atr_14": 3.0,
    }

    ExperimentLedger.register_signal(
        symbol="SHAKE",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=88.0,
        inputs_meta={"market_regime": "BULL"},
        signal_date=sig_date,
        ledger_path=ledger_path
    )

    # Day 1: dips to 93 (stop hit at 95), then Day 2-5 rallies all the way to 120 (well past TP1 at 115)
    shake_candles = [
        {"time": "2026-09-02", "open": 98.0, "high": 99.0, "low": 93.0, "close": 94.0, "volume": 1000}, # Stop hit here
        {"time": "2026-09-03", "open": 96.0, "high": 105.0, "low": 95.0, "close": 104.0, "volume": 2000},
        {"time": "2026-09-04", "open": 106.0, "high": 118.0, "low": 105.0, "close": 116.0, "volume": 3000}, # High touches 118 (past TP1)
    ]
    db = DummyDB({"SHAKE": shake_candles})
    ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)

    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    ft = sig["forwardTracking"]

    # Trade construction outcome was STOP_LOSS
    assert sig["status"] == "RESOLVED"
    assert ft["resolvedOutcome"] == "STOP_LOSS"
    assert ft["realizedReturnPct"] == -5.0

    # BUT premature stop-out must be detected as True!
    assert ft["tradeConstruction"]["prematureStopOut"] is True
    # And MFE reflects the raw price potential (+18%)
    assert ft["maxFavorableExcursionPct"] == 18.0


def test_confluence_monotonicity_scorecard(tmp_path):
    """Verify that scorecard computes monotonicity across confluence score bands (<75, 75-79.9, 80-84.9, 85+)."""
    ledger_path = str(tmp_path / "test_ledger.json")

    # Construct synthetic signals across 3 confluence buckets with monotonic win rates
    synthetic_ledger = {
        "version": "1.0.0",
        "totalActiveSignals": 0,
        "signals": [
            # Bucket 75-79.9: 1 Win, 1 Loss -> 50% Win Rate
            {
                "signalId": "S_76_WIN",
                "symbol": "S1",
                "confluenceScore": 76.0,
                "status": "RESOLVED",
                "signalDate": "2026-09-01",
                "forwardTracking": {"resolvedOutcome": "TP1_WIN", "realizedReturnPct": 10.0, "maxFavorableExcursionPct": 12.0}
            },
            {
                "signalId": "S_76_LOSS",
                "symbol": "S2",
                "confluenceScore": 77.0,
                "status": "RESOLVED",
                "signalDate": "2026-09-01",
                "forwardTracking": {"resolvedOutcome": "STOP_LOSS", "realizedReturnPct": -6.0, "maxFavorableExcursionPct": 2.0}
            },
            # Bucket 80-84.9: 2 Wins, 1 Loss -> 66.7% Win Rate
            {
                "signalId": "S_82_WIN1",
                "symbol": "S3",
                "confluenceScore": 82.0,
                "status": "RESOLVED",
                "signalDate": "2026-09-02",
                "forwardTracking": {"resolvedOutcome": "TP1_WIN", "realizedReturnPct": 12.0, "maxFavorableExcursionPct": 14.0}
            },
            {
                "signalId": "S_82_WIN2",
                "symbol": "S4",
                "confluenceScore": 83.0,
                "status": "RESOLVED",
                "signalDate": "2026-09-02",
                "forwardTracking": {"resolvedOutcome": "TP1_WIN", "realizedReturnPct": 14.0, "maxFavorableExcursionPct": 15.0}
            },
            {
                "signalId": "S_82_LOSS",
                "symbol": "S5",
                "confluenceScore": 81.0,
                "status": "RESOLVED",
                "signalDate": "2026-09-02",
                "forwardTracking": {"resolvedOutcome": "STOP_LOSS", "realizedReturnPct": -5.0, "maxFavorableExcursionPct": 3.0}
            },
            # Bucket 85+: 2 Wins, 0 Losses -> 100% Win Rate
            {
                "signalId": "S_88_WIN1",
                "symbol": "S6",
                "confluenceScore": 88.0,
                "status": "RESOLVED",
                "signalDate": "2026-09-03",
                "forwardTracking": {"resolvedOutcome": "TP1_WIN", "realizedReturnPct": 18.0, "maxFavorableExcursionPct": 20.0}
            },
            {
                "signalId": "S_88_WIN2",
                "symbol": "S7",
                "confluenceScore": 91.0,
                "status": "RESOLVED",
                "signalDate": "2026-09-03",
                "forwardTracking": {"resolvedOutcome": "TP1_WIN", "realizedReturnPct": 16.0, "maxFavorableExcursionPct": 18.0}
            }
        ]
    }
    ExperimentLedger.save_ledger(synthetic_ledger, ledger_path)
    scorecard = ExperimentLedger.compute_governance_scorecard(ledger_path)

    mono = scorecard["confluenceMonotonicity"]
    assert mono["monotonicityStatus"] == "MONOTONIC_CONFIRMED"
    assert mono["isMonotonicWinRate"] is True
    assert mono["isMonotonicReturn"] is True

    # Check that buckets show increasing win rates: 50.0% -> 66.7% -> 100.0%
    b_75 = mono["buckets"]["75-79.9"]
    b_80 = mono["buckets"]["80-84.9"]
    b_85 = mono["buckets"]["85+"]
    assert b_75["winRate"] == 50.0
    assert b_80["winRate"] == 66.7
    assert b_85["winRate"] == 100.0
    assert b_85["meanRealizedReturn"] > b_80["meanRealizedReturn"] > b_75["meanRealizedReturn"]
