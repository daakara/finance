"""Phase 26 Prospective Liquidity Validation Adversarial & Invariant Test Suite.

Verifies the 12 mandatory Phase 26 governance & mathematical requirements:
1. Prospective observations cannot mutate the frozen decision.
2. Signal-time liquidity cannot be overwritten.
3. Future liquidity cannot affect historical returns.
4. Execution observations cannot alter signal outcomes.
5. Missing fill data is represented as missing (None), not fabricated.
6. Zero/negative/invalid prices are rejected safely.
7. Participation calculations remain mathematically correct.
8. Cohort assignment is deterministic.
9. No future information enters a signal-time observation (temporal anti-lookahead).
10. Cryptographic hashes detect post-hoc modification fail-closed.
11. Counterfactual gating does not mutate the actual signal.
12. Statistical calculations handle empty/small cohorts safely without NaNs or zero-division.
"""

import os
import json
import math
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
from analyst_dashboard.governance.liquidity_validation import Phase26ValidationEngine
from api.routes.screener import get_phase26_validation_report

pytestmark = pytest.mark.tier3



@pytest.fixture
def mock_signal_data():
    return {
        "symbol": "VALID8",
        "entry_price": 100.0,
        "opt_exec": {
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
            "stage_phase": "Stage 2 Advancing Growth Phase",
            "liquidity_defense": {
                "liquidity_grade": "HIGH_TRADING_LIQUIDITY",
                "badge_color": "emerald",
                "adv_20d_usd": 10_000_000.0,
                "adv_5d_usd": 10_000_000.0,
                "liquidity_trend": 1.0,
                "amihud_illiq": 1.0e-8,
                "amihud_illiq_scaled": 0.01,
                "volume_spike_ratio": 1.0,
                "is_volume_spike": False,
                "float_turnover_pct": None,
                "is_float_turnover_anomaly": False,
                "estimated_participation_rate": 0.0005,
                "execution_hazard": False,
                "market_order_warning": False,
                "suppress_buy_zone": False,
                "plain_label": "💧 High Trading Liquidity",
                "pro_label": "HIGH_TRADING_LIQUIDITY",
                "plain_summary": "High trading volume.",
                "pro_summary": "ADV exceeds $2M.",
                "spec_version": "LiquidityGuard Shadow Spec v1.0",
                "signal_timestamp": "2026-09-04T12:00:00Z",
            }
        },
        "confluence_score": 85.0,
        "inputs_meta": {
            "market_regime": "BULL",
            "sector": "TECHNOLOGY",
            "asset_class": "US_EQUITY"
        }
    }


# ── 1. Prospective observations cannot mutate the frozen decision ─────────────
def test_1_prospective_observations_cannot_mutate_frozen_decision(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]
    original_decision_hash = rec["decisionSnapshotHash"]

    # Record multiple prospective execution observations
    ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=100.25,
        execution_timestamp="2026-09-04T12:05:00Z",
        order_size_usd=5000.0,
        side="BUY",
        ledger_path=ledger_path
    )

    ledger_after = ExperimentLedger.load_ledger(ledger_path)
    sig_after = ledger_after["signals"][0]

    # Immutable frozen decision parameters
    assert sig_after["decisionSnapshotHash"] == original_decision_hash
    assert sig_after["entryPrice"] == 100.0
    assert sig_after["stopLoss"] == 94.0
    assert sig_after["takeProfit1"] == 112.0
    assert sig_after["takeProfit2"] == 118.0
    assert sig_after["decisionState"] == "ACTIONABLE_SETUP"
    assert sig_after["confluenceScore"] == 85.0


# ── 2. Signal-time liquidity cannot be overwritten ────────────────────────────
def test_2_signal_time_liquidity_cannot_be_overwritten(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]
    orig_liq_signal = dict(rec["liquidityAtSignal"])

    # Record execution observation with differing liquidity conditions
    ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=101.50,
        execution_timestamp="2026-09-04T12:05:00Z",
        order_size_usd=10000.0,
        side="BUY",
        ledger_path=ledger_path
    )

    # Record a deteriorating forward liquidity observation
    ExperimentLedger.record_liquidity_forward_observation(
        signal_id=sig_id,
        session_index=1,
        liquidity_metrics={
            "adv_20d_usd": 100_000.0,
            "amihud_illiq": 5.0e-5,
            "amihud_illiq_scaled": 50.0,
            "liquidity_grade": "EXECUTION_RISK",
            "execution_hazard": True
        },
        ledger_path=ledger_path
    )

    ledger_after = ExperimentLedger.load_ledger(ledger_path)
    sig_after = ledger_after["signals"][0]

    # Signal-time liquidity MUST remain identical to orig_liq_signal
    assert sig_after["liquidityAtSignal"] == orig_liq_signal
    assert sig_after["liquidityAtSignal"]["liquidity_grade"] == "HIGH_TRADING_LIQUIDITY"
    assert sig_after["liquidityAtSignal"]["adv_20d_usd"] == 10_000_000.0


# ── 3. Future liquidity cannot affect historical returns ──────────────────────
def test_3_future_liquidity_cannot_affect_historical_returns(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Mark as resolved with a +12.0% TP1 hit
    ledger = ExperimentLedger.load_ledger(ledger_path)
    s = ledger["signals"][0]
    s["status"] = "RESOLVED"
    s["forwardTracking"]["resolvedOutcome"] = "TP1_WIN"
    s["forwardTracking"]["realizedReturnPct"] = 12.0
    ExperimentLedger.save_ledger(ledger, ledger_path)

    # Append future liquidity collapse
    ExperimentLedger.record_liquidity_forward_observation(
        signal_id=sig_id,
        session_index=5,
        liquidity_metrics={"adv_20d_usd": 1000.0, "liquidity_grade": "EXECUTION_RISK", "execution_hazard": True},
        ledger_path=ledger_path
    )

    ledger_after = ExperimentLedger.load_ledger(ledger_path)
    assert ledger_after["signals"][0]["status"] == "RESOLVED"
    assert ledger_after["signals"][0]["forwardTracking"]["resolvedOutcome"] == "TP1_WIN"
    assert ledger_after["signals"][0]["forwardTracking"]["realizedReturnPct"] == 12.0


# ── 4. Execution observations cannot alter signal outcomes ────────────────────
def test_4_execution_observations_cannot_alter_signal_outcomes(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Record severe execution slippage (e.g. 500 bps)
    ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=105.0,  # 5% slippage
        execution_timestamp="2026-09-04T12:01:00Z",
        side="BUY",
        ledger_path=ledger_path
    )

    ledger_after = ExperimentLedger.load_ledger(ledger_path)
    s = ledger_after["signals"][0]
    assert s["status"] == "OPEN"
    assert s["entryPrice"] == 100.0  # entry price reference remains 100.0
    assert s["forwardTracking"]["tp1Hit"] is False
    assert s["forwardTracking"]["stopHit"] is False


# ── 5. Missing fill data is represented as missing, not fabricated ────────────
def test_5_missing_fill_data_is_represented_as_missing(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Record unfilled order
    obs = ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=None,  # Not filled
        execution_timestamp=None,
        side="BUY",
        order_size_usd=5000.0,
        ledger_path=ledger_path
    )

    assert obs["fillPrice"] is None
    assert obs["slippageBps"] is None
    assert obs["executionSource"] == "UNFILLED"
    assert obs["executionTimestamp"] is None


# ── 6. Zero/negative/invalid prices are rejected safely ───────────────────────
def test_6_zero_negative_invalid_prices_rejected_safely(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Invalid fill prices
    for invalid_p in [0.0, -10.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="Invalid fill price"):
            ExperimentLedger.record_execution_observation(
                signal_id=sig_id,
                fill_price=invalid_p,
                ledger_path=ledger_path
            )

    # Invalid order side
    with pytest.raises(ValueError, match="Invalid order side"):
        ExperimentLedger.record_execution_observation(
            signal_id=sig_id,
            fill_price=100.0,
            side="HOLD",
            ledger_path=ledger_path
        )


# ── 7. Participation calculations remain mathematically correct ───────────────
def test_7_participation_calculations_correct(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Order size: $50,000 on $10,000,000 ADV -> 0.005 (0.5%)
    obs = ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=100.05,
        order_size_usd=50_000.0,
        execution_timestamp="2026-09-04T12:05:00Z",
        ledger_path=ledger_path
    )
    assert obs["participationRate"] == 0.005
    assert obs["slippageBps"] == 5.0  # (100.05 - 100.0) / 100.0 * 10,000 = 5.0 bps


# ── 8. Cohort assignment is deterministic ─────────────────────────────────────
def test_8_cohort_assignment_is_deterministic():
    records = [
        {"liquidityGrade": "HIGH_TRADING_LIQUIDITY", "fillPrice": 100.1, "slippageBps": 10.0, "isSimulated": False},
        {"liquidityGrade": "HIGH_TRADING_LIQUIDITY", "fillPrice": 100.2, "slippageBps": 20.0, "isSimulated": False},
        {"liquidityGrade": "EXECUTION_RISK", "fillPrice": 100.8, "slippageBps": 80.0, "isSimulated": True},
        {"liquidityGrade": "EXECUTION_RISK", "fillPrice": 101.0, "slippageBps": 100.0, "isSimulated": True},
    ]

    report = Phase26ValidationEngine.evaluate_execution_friction(records)
    high_c = report["cohorts"]["HIGH_TRADING_LIQUIDITY"]
    risk_c = report["cohorts"]["EXECUTION_RISK"]

    assert high_c["sampleSize"] == 2
    assert high_c["meanSlippageBps"] == 15.0
    assert high_c["realFills"] == 2
    assert high_c["simulatedFills"] == 0

    assert risk_c["sampleSize"] == 2
    assert risk_c["meanSlippageBps"] == 90.0
    assert risk_c["realFills"] == 0
    assert risk_c["simulatedFills"] == 2


# ── 9. No future information enters a signal-time observation ─────────────────
def test_9_temporal_anti_lookahead_enforced(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Signal was created at 2026-09-04T12:00:00Z. Attempting execution at 2026-09-04T11:00:00Z must fail.
    with pytest.raises(ValueError, match="TEMPORAL_VIOLATION"):
        ExperimentLedger.record_execution_observation(
            signal_id=sig_id,
            fill_price=100.1,
            execution_timestamp="2026-09-04T11:00:00Z",
            ledger_path=ledger_path
        )


# ── 10. Cryptographic hashes detect post-hoc modification ─────────────────────
def test_10_cryptographic_hashes_detect_post_hoc_modification(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Tamper with entry price on disk
    with open(ledger_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["signals"][0]["entryPrice"] = 105.0  # Tampered!
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Attempting to record an execution observation must fail-closed
    with pytest.raises(ValueError, match="GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch"):
        ExperimentLedger.record_execution_observation(
            signal_id=sig_id,
            fill_price=105.1,
            ledger_path=ledger_path
        )


# ── 11. Counterfactual gating does not mutate the actual signal ───────────────
def test_11_counterfactual_gating_does_not_mutate_signal(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )

    ledger = ExperimentLedger.load_ledger(ledger_path)
    s = ledger["signals"][0]
    s["status"] = "RESOLVED"
    s["forwardTracking"]["resolvedOutcome"] = "TP1_WIN"
    s["forwardTracking"]["realizedReturnPct"] = 12.0
    ExperimentLedger.save_ledger(ledger, ledger_path)

    # Evaluate economic counterfactual
    loaded = ExperimentLedger.load_ledger(ledger_path)
    res = Phase26ValidationEngine.evaluate_economic_counterfactual(loaded, filter_criterion="HIGH_TRADING_LIQUIDITY")

    # Invariant: signals in ledger must remain unmodified
    assert res["tradeoffAnalysis"]["filteredTradesCount"] == 1
    assert res["tradeoffAnalysis"]["excludedWinnersCount"] == 1

    ledger_after = ExperimentLedger.load_ledger(ledger_path)
    assert ledger_after["signals"][0]["status"] == "RESOLVED"
    assert ledger_after["signals"][0]["forwardTracking"]["realizedReturnPct"] == 12.0


# ── 12. Statistical calculations handle empty/small cohorts safely ────────────
def test_12_statistical_calculations_handle_small_cohorts_safely():
    # Empty records
    empty_report = Phase26ValidationEngine.evaluate_execution_friction([])
    assert empty_report["status"] == "EMPTY_OBSERVATION_POOL"
    assert empty_report["pairwiseComparison"]["status"] == "INSUFFICIENT_SAMPLE_FOR_INFERENCE"
    assert empty_report["pairwiseComparison"]["differenceInMeansBps"] is None

    # Single observation cohort
    single_record = [
        {"liquidityGrade": "HIGH_TRADING_LIQUIDITY", "fillPrice": 100.1, "slippageBps": 10.0, "isSimulated": False}
    ]
    single_report = Phase26ValidationEngine.evaluate_execution_friction(single_record)
    high_stats = single_report["cohorts"]["HIGH_TRADING_LIQUIDITY"]
    assert high_stats["status"] == "SINGLE_OBSERVATION"
    assert high_stats["meanSlippageBps"] == 10.0
    assert high_stats["standardErrorBps"] is None
    assert high_stats["ci95"] is None
    assert single_report["pairwiseComparison"]["statisticallySignificant"] is False

    # Empty economic counterfactual
    empty_econ = Phase26ValidationEngine.evaluate_economic_counterfactual({"signals": []})
    assert empty_econ["status"] == "NO_RESOLVED_SIGNALS"
    assert empty_econ["ungatedPortfolio"]["tradeCount"] == 0
    assert empty_econ["gatedPortfolio"]["tradeCount"] == 0


# ── 13. API Endpoint Round-Trip ───────────────────────────────────────────────
def test_13_api_endpoint_phase26_validation_report():
    report = get_phase26_validation_report()
    assert report["document"] == "Phase 26 Prospective Liquidity Validation Report"
    assert report["specVersion"] == "LiquidityGuard Shadow Spec v1.0"
    assert "executionFrictionExperiment" in report
    assert "economicCounterfactualExperiment" in report
    assert "promotionGateReadiness" in report
    assert report["overallPromotionVerdict"] == "RETAIN_SHADOW_OBSERVATION"


# ── 14. Duplicate Execution Observation Rejected ──────────────────────────────
def test_14_duplicate_execution_observation_rejected(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # First fill record
    ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=100.25,
        execution_timestamp="2026-09-04T12:05:00Z",
        order_size_usd=5000.0,
        side="BUY",
        ledger_path=ledger_path
    )

    # Identical fill attempt must fail closed
    with pytest.raises(ValueError, match="DUPLICATE_EXECUTION_OBSERVATION"):
        ExperimentLedger.record_execution_observation(
            signal_id=sig_id,
            fill_price=100.25,
            execution_timestamp="2026-09-04T12:05:00Z",
            order_size_usd=5000.0,
            side="BUY",
            ledger_path=ledger_path
        )


# ── 15. Cross-Signal Contamination Rejected ───────────────────────────────────
def test_15_cross_signal_contamination_rejected(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Attempting to record an execution for WRONG symbol must fail closed
    with pytest.raises(ValueError, match="CROSS_SIGNAL_CONTAMINATION"):
        ExperimentLedger.record_execution_observation(
            signal_id=sig_id,
            symbol="WRONG_SYM",
            fill_price=100.25,
            execution_timestamp="2026-09-04T12:05:00Z",
            ledger_path=ledger_path
        )


# ── 16. Resolution Occurring Before Execution Rejected ────────────────────────
def test_16_resolution_occurring_before_execution_rejected(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=mock_signal_data["entry_price"],
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # Mark as resolved on 2026-09-04
    ledger = ExperimentLedger.load_ledger(ledger_path)
    s = ledger["signals"][0]
    s["status"] = "RESOLVED"
    s["forwardTracking"]["resolutionDate"] = "2026-09-04"
    ExperimentLedger.save_ledger(ledger, ledger_path)

    # Attempt to record execution dated subsequent week (2026-09-10)
    with pytest.raises(ValueError, match="TEMPORAL_VIOLATION.*occurs after trade resolution date"):
        ExperimentLedger.record_execution_observation(
            signal_id=sig_id,
            fill_price=100.25,
            execution_timestamp="2026-09-10T12:00:00Z",
            ledger_path=ledger_path
        )


# ── 17. Signed Slippage Directionality for BUY vs SELL ────────────────────────
def test_17_signed_slippage_directionality(tmp_path, mock_signal_data):
    ledger_path = str(tmp_path / "ledger.json")
    rec = ExperimentLedger.register_signal(
        symbol=mock_signal_data["symbol"],
        entry_price=100.0,
        opt_exec=mock_signal_data["opt_exec"],
        confluence_score=mock_signal_data["confluence_score"],
        inputs_meta=mock_signal_data["inputs_meta"],
        ledger_path=ledger_path
    )
    sig_id = rec["signalId"]

    # BUY with adverse fill (101.0 > 100.0) -> +100.0 bps signed slippage
    obs_buy_adverse = ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=101.0,
        execution_timestamp="2026-09-04T12:01:00Z",
        side="BUY",
        ledger_path=ledger_path
    )
    assert obs_buy_adverse["slippageBps"] == 100.0
    assert obs_buy_adverse["signedSlippageBps"] == 100.0

    # BUY with favorable fill (99.0 < 100.0) -> -100.0 bps signed slippage (price improvement)
    obs_buy_fav = ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=99.0,
        execution_timestamp="2026-09-04T12:02:00Z",
        side="BUY",
        ledger_path=ledger_path
    )
    assert obs_buy_fav["slippageBps"] == 100.0
    assert obs_buy_fav["signedSlippageBps"] == -100.0

    # SELL with adverse fill (99.0 < 100.0) -> +100.0 bps signed slippage (received less)
    obs_sell_adverse = ExperimentLedger.record_execution_observation(
        signal_id=sig_id,
        fill_price=99.0,
        execution_timestamp="2026-09-04T12:03:00Z",
        side="SELL",
        ledger_path=ledger_path
    )
    assert obs_sell_adverse["slippageBps"] == 100.0
    assert obs_sell_adverse["signedSlippageBps"] == 100.0


# ── 18. Real vs. Simulated Cohort Segregation ─────────────────────────────────
def test_18_real_vs_simulated_cohort_segregation():
    records = [
        {"liquidityGrade": "HIGH_TRADING_LIQUIDITY", "fillPrice": 100.1, "slippageBps": 10.0, "isSimulated": False, "executionSource": "BROKER_FILL"},
        {"liquidityGrade": "HIGH_TRADING_LIQUIDITY", "fillPrice": 100.2, "slippageBps": 20.0, "isSimulated": False, "executionSource": "BROKER_FILL"},
        {"liquidityGrade": "EXECUTION_RISK", "fillPrice": 101.0, "slippageBps": 100.0, "isSimulated": True, "executionSource": "SIMULATED_FILL"},
        {"liquidityGrade": "EXECUTION_RISK", "fillPrice": 101.2, "slippageBps": 120.0, "isSimulated": True, "executionSource": "SIMULATED_FILL"},
    ]

    report = Phase26ValidationEngine.evaluate_execution_friction(records)

    # Real cohort must contain ONLY the 2 broker fills
    real_ev = report["realEvidence"]
    assert real_ev["sampleCount"] == 2
    assert real_ev["cohorts"]["HIGH_TRADING_LIQUIDITY"]["sampleSize"] == 2
    assert real_ev["cohorts"]["EXECUTION_RISK"]["sampleSize"] == 0

    # Simulated cohort must contain ONLY the 2 simulated fills
    sim_ev = report["simulatedEvidence"]
    assert sim_ev["sampleCount"] == 2
    assert sim_ev["cohorts"]["HIGH_TRADING_LIQUIDITY"]["sampleSize"] == 0
    assert sim_ev["cohorts"]["EXECUTION_RISK"]["sampleSize"] == 2

    # Formal promotion evidence must equal realEvidence
    assert report["formalPromotionEvidence"] == real_ev
    assert report["formalPromotionEvidence"]["pairwiseComparison"]["status"] == "INSUFFICIENT_SAMPLE_FOR_INFERENCE"
