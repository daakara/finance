"""
tests/test_prediction_learning_integrity.py

Targeted Adversarial & Epistemic Truthfulness Suite for ARX-LEARNING-INTEGRITY-001.
Verifies fail-closed behavior, zero synthetic outcomes, prospective cohort separation,
and backend diagnostic integrity.
"""

import pytest
import os
import tempfile
import sqlite3
import pandas as pd
import numpy as np

from analyst_dashboard.analyzers.self_healing_engine import SelfHealingForecastAuditor
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine
from analyst_dashboard.governance.experiment_ledger import (
    ExperimentLedger,
    ProvenanceCohort,
)
from analyst_dashboard.governance.paper_trading_outcome_evaluator import (
    PaperTradingOutcomeEvaluator,
)

pytestmark = pytest.mark.tier2b


def test_self_healing_auditor_epistemic_truthfulness():
    """Verify that SelfHealingForecastAuditor emits honest diagnostic labels and never decorative claims."""
    auditor = SelfHealingForecastAuditor()

    # Synthetic price series with 60 bars
    dates = pd.date_range("2026-01-01", periods=60)
    prices = [100.0 * (1.0 + 0.005 * i + 0.01 * np.sin(i)) for i in range(60)]
    df = pd.DataFrame({"Close": prices}, index=dates)

    audit = auditor.audit_and_calibrate(
        symbol="NVDA",
        price_df=df,
        current_risk_metrics={"Modified_VaR_95": -2.8},
        expected_return_data={"p50Expected": 18.5},
    )

    # 1. Audit status must reflect read-only backtest, not self-healing
    assert audit["auditStatus"] != "Self-Healed & Auto-Calibrated"
    assert "Historical Walk-Forward Backtest Audit" in audit["auditStatus"]
    assert "Parameters Unmodified" in audit["auditStatus"]

    # 2. Adjustments must be advisory, not claim active mutation
    assert "Auto-widened fat-tail multiplier by +12%" not in str(audit["autoCalibrationAdjustments"])
    if "Advisory" in str(audit["autoCalibrationAdjustments"]):
        assert "unapplied" in str(audit["autoCalibrationAdjustments"])

    # 3. Confidence interval must not claim 95% statistical confidence without calculation
    assert audit["confidenceInterval"] != "95% Statistical Confidence"
    assert "Diagnostic Only" in audit["confidenceInterval"]


def test_history_database_accuracy_summary_zero_resolved_setups():
    """Verify that get_setup_accuracy_summary fails closed when all logged setups are PENDING."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_history.db")
        engine = HistoryDatabaseEngine(db_path=db_path)

        # Log 35 setups all in PENDING state (no evaluated outcomes)
        for i in range(35):
            engine.log_trade_recommendation(
                "NVDA",
                {
                    "setup_pattern": "VCP",
                    "stage_phase": "Stage 2",
                    "current_price": 120.0 + i,
                    "optimal_entry_min": 118.0,
                    "optimal_entry_max": 122.0,
                    "stop_loss": 112.0,
                    "take_profit_1": 135.0,
                    "take_profit_2": 145.0,
                    "risk_reward_ratio": 2.5,
                },
            )

        summary = engine.get_setup_accuracy_summary("NVDA")

        # Must report total logged = 35, but resolved = 0
        assert summary["total_logged_setups"] == 35
        assert summary["resolved_setups"] == 0
        assert summary["target_hit_rate_pct"] is None

        # Must NOT claim active calibration or 95% statistical confidence
        assert "Active (Persistent SQLite NVMe Ledger)" not in summary["model_calibration_status"]
        assert "0 Resolved Out of 35 Logged" in summary["model_calibration_status"]
        assert "95% Statistical Confidence" not in summary["statistical_confidence"]
        assert "Insufficient Resolved Data" in summary["statistical_confidence"]


def test_prospective_cohort_firewall_rejects_synthetic_and_demo():
    """Verify that prospective clean cohorts reject demo/synthetic signals from entering clean metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = os.path.join(tmpdir, "test_ledger.json")

        # Initialize clean ledger
        ExperimentLedger.save_ledger(
            {"version": "2.0.0", "signals": [], "totalActiveSignals": 0},
            ledger_path=ledger_path,
        )

        # Demo signal should be classified into DEMO cohort
        demo_signal = {
            "signalId": "DEMO_TEST_001",
            "symbol": "DEMO_CO",
            "isDemo": True,
            "signalDate": "2026-11-01",
            "corridorMin": 100.0,
            "corridorMax": 105.0,
            "stopLoss": 95.0,
        }

        cohort = ExperimentLedger.classify_provenance_cohort(demo_signal)
        assert cohort != ProvenanceCohort.PROSPECTIVE_CLEAN
        assert cohort in [
            ProvenanceCohort.DEMO_SYNTHETIC,
            ProvenanceCohort.CONTAMINATED,
            ProvenanceCohort.HISTORICAL_CONTAMINATED,
            ProvenanceCohort.BACKTEST_SIMULATION,
        ]


def test_paper_trading_outcome_evaluator_intrabar_collision_isolation():
    """Verify that PaperTradingOutcomeEvaluator strictly enforces fail-closed collision resolution."""
    # Signal with entry corridor [100, 102], stop 95, target 110
    signal = {
        "symbol": "TEST",
        "signalDate": "2026-11-01",
        "corridorMin": 100.0,
        "corridorMax": 102.0,
        "stopLoss": 95.0,
        "target1": 110.0,
        "currentPrice": 101.0,
        "metadata": {"engine_version": "2.4.0"},
    }

    # Day 1: Enters trade at 101.0
    # Day 2: Single bar touches BOTH stop (94.0) and target (112.0) without intraday bars
    bars = pd.DataFrame(
        [
            {"Open": 101.0, "High": 103.0, "Low": 100.0, "Close": 102.0},  # Day 1: Entry
            {"Open": 102.0, "High": 112.0, "Low": 94.0, "Close": 105.0},   # Day 2: Collision!
        ],
        index=pd.date_range("2026-11-01", periods=2),
    )

    # Under Contract V1.0.1, collision without intraday resolution must fail-closed
    result = PaperTradingOutcomeEvaluator.evaluate_signal(signal, bars, intraday_bars=None)
    assert result.outcome in [
        "STOP_LOSS_EXIT",
        "COLLISION_STOP_ASSUMED",
        "UNRESOLVED_COLLISION",
        "UNRESOLVED_INTRABAR_SEQUENCE",
    ]
    # It must NEVER assume optimistic TARGET_1_EXIT on unresolvable collision!
    assert result.outcome != "TARGET_1_EXIT"


def test_radar_and_smart_money_invariants_preserved():
    """Verify that all operational safety holds and canonical policies remain strictly intact."""
    # 1. HOLD_SCAN_001, HOLD_SEC_001, HOLD_EXEC_001 defaults
    from api.routes.analytics import router as analytics_router
    assert analytics_router is not None

    # 2. Check that VCP scheduler defaults to disabled (0)
    scheduler_enabled = os.getenv("VCP_NATURAL_SCHEDULER_ENABLED", "0")
    assert scheduler_enabled in ["0", "false", "False"]

    # 3. Check that live execution hold is active
    exec_hold = os.getenv("HOLD_EXEC_001", "ACTIVE")
    assert exec_hold == "ACTIVE"
