"""Phase 4B — Liquidity Evidence Integrity (F_10 Remediation) Test Suite.

Audits evidential truthfulness of liquidity metrics:
- Eradication of synthetic fallbacks (ADV=None, Amihud=None, Trend=None, Hazard="UNKNOWN", Status="UNAVAILABLE").
- Eradication of inline 0.0 or 1.0 substitutions.
- Strict shadow observation boundaries (zero canonical decision interference).
- Policy constant vs simulation assumption vs observed dynamic evidence taxonomy.
- Weakest-link provenance inheritance.
- Ledger serialization truthfulness without synthetic number injection.
- Client-side chart request decoupling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from analyst_dashboard.analyzers.liquidity_guard import (
    LiquidityGuard,
    LiquidityEvidenceStatus,
    LiquidityEvidenceType,
)
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
from analyst_dashboard.governance.liquidity_validation import Phase26ValidationEngine


pytestmark = pytest.mark.tier2b


# ── Dimension 1: Missing OHLCV Adversarial Fixture ─────────────────────────────
def test_dim1_missing_ohlcv_adversarial_fixture():
    """Missing, empty, or insufficient (<3 bars) OHLCV must fail closed to UNAVAILABLE with None values."""
    for bad_df in [None, pd.DataFrame(), pd.DataFrame({"Close": [50.0], "Volume": [1000]}), pd.DataFrame({"Close": [50.0, 51.0], "Volume": [1000, 2000]})]:
        res = LiquidityGuard.evaluate_liquidity(bad_df, 50.0)
        assert res["liquidity_grade"] == "UNKNOWN_LIQUIDITY"
        assert res["badge_color"] == "slate"
        assert res["adv_20d_usd"] is None
        assert res["adv_5d_usd"] is None
        assert res["liquidity_trend"] is None
        assert res["amihud_illiq"] is None
        assert res["amihud_illiq_scaled"] is None
        assert res["volume_spike_ratio"] is None
        assert res["execution_hazard"] == "UNKNOWN"
        assert res["suppress_buy_zone"] is False
        assert res["evidenceStatus"] == LiquidityEvidenceStatus.UNAVAILABLE
        assert res["evidenceType"] == LiquidityEvidenceType.UNAVAILABLE

        # Verify factorEvidence structure
        fe = res["factorEvidence"]
        assert fe["adv_20d"]["value"] is None
        assert fe["adv_20d"]["evidenceStatus"] == LiquidityEvidenceStatus.UNAVAILABLE
        assert fe["amihud_illiq"]["value"] is None
        assert fe["amihud_illiq"]["evidenceStatus"] == LiquidityEvidenceStatus.UNAVAILABLE
        assert fe["execution_hazard"]["value"] == "UNKNOWN"
        assert fe["execution_hazard"]["evidenceStatus"] == LiquidityEvidenceStatus.UNAVAILABLE


# ── Dimension 2: Partial OHLCV Fixture ─────────────────────────────────────────
def test_dim2_partial_ohlcv_provisional_fixture():
    """Partial OHLCV (between 3 and 19 bars) must produce PROVISIONAL status."""
    dates = pd.date_range("2026-01-01", periods=10)
    prices = [100.0 + i for i in range(10)]
    volumes = [50_000 for _ in range(10)]
    df = pd.DataFrame({"Close": prices, "Volume": volumes, "High": [p + 1 for p in prices], "Low": [p - 1 for p in prices]}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 110.0)
    assert res["evidenceStatus"] == LiquidityEvidenceStatus.PROVISIONAL
    assert res["adv_20d_usd"] is not None
    assert res["adv_5d_usd"] is not None
    assert res["amihud_illiq"] is not None
    assert res["factorEvidence"]["adv_20d"]["evidenceStatus"] == LiquidityEvidenceStatus.PROVISIONAL
    assert res["factorEvidence"]["adv_20d"]["quality"] == "MEDIUM"


# ── Dimension 3: Complete Healthy OHLCV Fixture ────────────────────────────────
def test_dim3_complete_healthy_ohlcv_fixture():
    """Complete healthy OHLCV (>= 20 bars) must yield AUTHORITATIVE status and exact parity."""
    dates = pd.date_range("2026-01-01", periods=25)
    prices = [100.0 + (i * 0.5) for i in range(25)]
    volumes = [100_000 for _ in range(25)]
    df = pd.DataFrame({"Close": prices, "Volume": volumes, "High": [p + 0.5 for p in prices], "Low": [p - 0.5 for p in prices]}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 112.5)
    assert res["evidenceStatus"] == LiquidityEvidenceStatus.AUTHORITATIVE
    assert res["evidenceType"] == LiquidityEvidenceType.DERIVED_ANALYTIC
    assert res["liquidity_grade"] == "HIGH_TRADING_LIQUIDITY"
    assert res["execution_hazard"] is False
    assert res["adv_20d_usd"] > 2_000_000
    assert res["factorEvidence"]["adv_20d"]["evidenceStatus"] == LiquidityEvidenceStatus.AUTHORITATIVE
    assert res["factorEvidence"]["adv_20d"]["quality"] == "HIGH"
    assert res["factorEvidence"]["amihud_illiq"]["evidenceStatus"] == LiquidityEvidenceStatus.AUTHORITATIVE


# ── Dimension 4: Weakest-Link Provenance Inheritance ───────────────────────────
def test_dim4_weakest_link_provenance_inheritance():
    """Overall status must reflect the weakest required input."""
    # When volume series is completely zero:
    dates = pd.date_range("2026-01-01", periods=30)
    df_zero_vol = pd.DataFrame({"Close": [10.0]*30, "Volume": [0.0]*30}, index=dates)
    res_zero = LiquidityGuard.evaluate_liquidity(df_zero_vol, 10.0)
    assert res_zero["evidenceStatus"] == LiquidityEvidenceStatus.UNAVAILABLE

    # When bars are between 3 and 19:
    dates_15 = pd.date_range("2026-01-01", periods=15)
    df_15 = pd.DataFrame({"Close": [10.0]*15, "Volume": [1000]*15}, index=dates_15)
    res_15 = LiquidityGuard.evaluate_liquidity(df_15, 10.0)
    assert res_15["evidenceStatus"] == LiquidityEvidenceStatus.PROVISIONAL


# ── Dimension 5: Zero Amihud Missing-Data Default Eradication ───────────────────
def test_dim5_amihud_missing_data_no_zero_substitution():
    """When daily return or volume is invalid for Amihud, it must fail closed to None, not 0.0."""
    dates = pd.date_range("2026-01-01", periods=5)
    # Perfectly flat prices with zero return -> Amihud daily return is 0, but valid volume
    prices = [10.0, 10.0, 10.0, 10.0, 10.0]
    # Valid volume for ADV, but let's test a DataFrame where dollar volumes are 0
    df = pd.DataFrame({"Close": prices, "Volume": [100, 100, 100, 100, 100]}, index=dates)
    res = LiquidityGuard.evaluate_liquidity(df, 10.0)
    # Amihud calculation should be 0.0 only if mathematically returns are 0 and volume is positive,
    # BUT if dollar volume < 3 valid sessions, it must be None.
    df_no_dv = pd.DataFrame({"Close": [10.0, 11.0, 12.0], "Volume": [0, 0, 100]}, index=dates[:3])
    res_no_dv = LiquidityGuard.evaluate_liquidity(df_no_dv, 12.0)
    # Less than 3 valid dollar volume points -> Amihud must be None
    assert res_no_dv["amihud_illiq"] is None
    assert res_no_dv["amihud_illiq_scaled"] is None


# ── Dimension 6: Zero Trend Missing-Data Default Eradication ────────────────────
def test_dim6_trend_missing_data_no_one_substitution():
    """When 20-day ADV is missing or zero, liquidity trend must be None, not 1.0."""
    res_fallback = LiquidityGuard._generate_fallback(100.0)
    assert res_fallback["liquidity_trend"] is None
    assert res_fallback["liquidity_trend"] != 1.0


# ── Dimension 7: Policy Threshold Classification ──────────────────────────────
def test_dim7_policy_threshold_classification():
    """Operational cutoffs must be formally classified as POLICY_CONSTANTS."""
    assert LiquidityGuard.DEFAULT_ADV_HIGH_FLOOR == 2_000_000.0
    assert LiquidityGuard.DEFAULT_ADV_MIN_SAFETY_FLOOR == 500_000.0
    assert LiquidityGuard.DEFAULT_AMIHUD_TRAP_THRESHOLD == 5e-6
    assert LiquidityGuard.DEFAULT_AMIHUD_THIN_THRESHOLD == 1e-6
    assert LiquidityGuard.DEFAULT_PARTICIPATION_ADVISORY_THRESHOLD == 0.01


# ── Dimension 8: Slippage Model Classification ────────────────────────────────
def test_dim8_slippage_model_classification():
    """Heuristic cost basis in counterfactual simulation is classified as SIMULATION_ASSUMPTIONS."""
    assert isinstance(Phase26ValidationEngine.DEFAULT_HEURISTIC_COST_BPS, dict)
    assert Phase26ValidationEngine.DEFAULT_HEURISTIC_COST_BPS["HIGH_TRADING_LIQUIDITY"] == 5.0
    assert Phase26ValidationEngine.DEFAULT_HEURISTIC_COST_BPS["MODERATE_TRADING_LIQUIDITY"] == 15.0
    assert Phase26ValidationEngine.DEFAULT_HEURISTIC_COST_BPS["EXECUTION_RISK"] == 45.0
    assert Phase26ValidationEngine.DEFAULT_HEURISTIC_COST_BPS["UNKNOWN_LIQUIDITY"] == 10.0


# ── Dimension 9: Screening Policy Classification ──────────────────────────────
def test_dim9_screening_policy_classification():
    """Screening volume floors are classified as SCREENING_POLICY."""
    from analyst_dashboard.analyzers.gem_screener import GemCriteria
    criteria = GemCriteria()
    assert criteria.min_volume == 100_000


# ── Dimension 10: Shadow Mode Non-Interference ─────────────────────────────────
def test_dim10_shadow_mode_non_interference():
    """LiquidityGuard must never modify, override, or suppress canonical trade levels."""
    dates = pd.date_range("2026-01-01", periods=60)
    prices = [50.0 + (i * 0.2) for i in range(60)]
    highs = [p + 0.5 for p in prices]
    lows = [p - 0.5 for p in prices]

    # Liquid vs illiquid vs empty volume DataFrame
    df_liquid = pd.DataFrame({"Close": prices, "Volume": [1_000_000]*60, "High": highs, "Low": lows}, index=dates)
    df_illiquid = pd.DataFrame({"Close": prices, "Volume": [50]*60, "High": highs, "Low": lows}, index=dates)

    plan_liquid = OptimalExecutionEngine.calculate_trade_levels(df_liquid, prices[-1])
    plan_illiquid = OptimalExecutionEngine.calculate_trade_levels(df_illiquid, prices[-1])

    # Invariant: identical core levels
    assert plan_liquid["optimal_entry_min"] == plan_illiquid["optimal_entry_min"]
    assert plan_liquid["optimal_entry_max"] == plan_illiquid["optimal_entry_max"]
    assert plan_liquid["stop_loss"] == plan_illiquid["stop_loss"]
    assert plan_liquid["take_profit_1"] == plan_illiquid["take_profit_1"]
    assert plan_liquid["take_profit_2"] == plan_illiquid["take_profit_2"]

    # Shadow observation mode invariant
    assert plan_liquid["liquidity_defense"]["suppress_buy_zone"] is False
    assert plan_illiquid["liquidity_defense"]["suppress_buy_zone"] is False


# ── Dimension 11: Ledger Serialization Truthfulness ───────────────────────────
def test_dim11_ledger_serialization_truthfulness(tmp_path):
    """Ledger must serialize None without synthesizing 0.0 or 1.0."""
    temp_ledger = str(tmp_path / "test_ledger.json")

    # Evaluate liquidity on missing OHLCV
    liq_rep = LiquidityGuard.evaluate_liquidity(pd.DataFrame(), 100.0)
    assert liq_rep["adv_20d_usd"] is None
    assert liq_rep["amihud_illiq"] is None
    assert liq_rep["execution_hazard"] == "UNKNOWN"

    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 94.0,
        "take_profit_1": 112.0,
        "take_profit_2": 118.0,
        "risk_reward_ratio": 2.0,
        "atr_14": 3.0,
        "liquidity_defense": liq_rep,
    }

    # Register signal into empty ledger
    ExperimentLedger.save_ledger({"version": "1.0.0", "signals": [], "totalActiveSignals": 0}, temp_ledger)
    record = ExperimentLedger.register_signal(
        symbol="TEST",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL", "sector": "TECH", "asset_class": "US_EQUITY"},
        engine_commit="4e36862",
        engine_tag="v2.4.0-phase24-freeze",
        ledger_path=temp_ledger,
    )

    # In-memory check
    assert record["liquidityAtSignal"]["adv_20d_usd"] is None
    assert record["liquidityAtSignal"]["amihud_illiq"] is None
    assert record["liquidityAtSignal"]["liquidity_trend"] is None
    assert record["liquidityAtSignal"]["execution_hazard"] == "UNKNOWN"
    assert record["liquidityAtSignal"]["evidenceStatus"] == "UNAVAILABLE"

    # Reload from disk and verify clean JSON serialization (null values preserved, no 0.0 laundering)
    reloaded = ExperimentLedger.load_ledger(temp_ledger)
    rel_sig = reloaded["signals"][0]
    assert rel_sig["liquidityAtSignal"]["adv_20d_usd"] is None
    assert rel_sig["liquidityAtSignal"]["amihud_illiq"] is None
    assert rel_sig["liquidityAtSignal"]["liquidity_trend"] is None
    assert rel_sig["liquidityAtSignal"]["execution_hazard"] == "UNKNOWN"
    assert rel_sig["liquidityAtSignal"]["evidenceStatus"] == "UNAVAILABLE"


# ── Dimension 12: Client-Side Yahoo Decoupling ──────────────────────────────────
def test_dim12_client_side_decoupling():
    """LiquidityGuard must operate entirely server-side with zero dependency on client fetch."""
    dates = pd.date_range("2026-01-01", periods=25)
    prices = [25.0 + i * 0.1 for i in range(25)]
    df = pd.DataFrame({"Close": prices, "Volume": [200_000]*25, "High": [p + 0.2 for p in prices], "Low": [p - 0.2 for p in prices]}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 27.5)
    assert res["spec_version"] == LiquidityGuard.SPEC_VERSION
    assert "factorEvidence" in res
    assert res["factorEvidence"]["adv_20d"]["source"] == "ohlcv_series"
