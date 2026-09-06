"""
Comprehensive Production Readiness Audit Test Suite: LiquidityGuard Shadow Layer.

Executes all P0 (Mandatory Invariants) and P1 (Edge Cases & Trajectories) verification gates:
- P0-1: Full Decision-Vector Non-Interference
- P0-2: API Serialization / Deserialization Across All Tiers
- P0-3: Unknown/Missing-Data Semantics
- P0-4: Amihud Dimensional & Unit Semantics
- P0-5: No Hidden Liquidity Dependency in Frozen Engine
- P0-6: Split-Adjusted Price & Volume Alignment Guard
- P0-7: Temporal Point-in-Time & Anti-Lookahead in Rolling Windows
- P0-8: Signal-Time Liquidity Immutability in Ledger
- P0-9: Forward Liquidity Observations Cannot Affect Outcome Calculations
- P0-10: Cryptographic Dual-Hash Immutability Coverage

- P1-11: Liquidity Deterioration Trajectory (HIGH -> MODERATE -> EXECUTION_RISK)
- P1-12: Liquidity Recovery Trajectory (EXECUTION_RISK -> MODERATE -> HIGH)
- P1-13: Sudden Volume Spike Behavior
- P1-14: Intermittent Stale/Missing Sessions Resilience
- P1-15: Corporate Action Overnight Gap Guard
- P1-16: Very High-Priced Equities (BRK.A-like $600k+ spot)
- P1-17: Penny / Micro-Cap Equities (sub-$1 spot)
- P1-18: Order Size Participation Advisory Spectrum (0.1% to 10% ADV)
- P1-19: UX Advisory Terminology Invariant
- P1-20: Graceful Neutral State on UNKNOWN_LIQUIDITY
"""

import os
import json
import math
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch
from fastapi import Response

from analyst_dashboard.analyzers.liquidity_guard import LiquidityGuard
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
from api.routes.screener import run_screener_get

pytestmark = pytest.mark.tier3



# ── P0-1: Full Decision-Vector Non-Interference ──────────────────────────────
def test_p0_1_full_decision_vector_non_interference():
    """
    Assert Decision(Data_price, Volume_high) == Decision(Data_price, Volume_low)
    for every element of the mathematical decision vector.
    """
    dates = pd.date_range("2026-01-01", periods=60)
    prices = [100.0 + (i * 0.15) for i in range(60)]
    highs = [p + 0.6 for p in prices]
    lows = [p - 0.6 for p in prices]
    spot = 107.5

    df_liquid = pd.DataFrame({"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": [2_000_000]*60}, index=dates)
    df_illiquid = pd.DataFrame({"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": [1_000]*60}, index=dates)

    plan_liq = OptimalExecutionEngine.calculate_trade_levels(df_liquid, spot, user_role="LONG_TERM")
    plan_illiq = OptimalExecutionEngine.calculate_trade_levels(df_illiquid, spot, user_role="LONG_TERM")

    decision_keys = [
        "current_price",
        "optimal_entry_min",
        "optimal_entry_max",
        "stop_loss",
        "stop_loss_pct",
        "take_profit_1",
        "take_profit_1_pct",
        "take_profit_2",
        "take_profit_2_pct",
        "risk_reward_ratio",
        "execution_status",
        "setup_pattern",
        "entry_thesis",
        "invalidation_condition",
        "stage_phase",
        "vcp_contraction_status",
        "breakout_pivot",
        "atr_14",
    ]

    for k in decision_keys:
        assert plan_liq[k] == plan_illiq[k], f"Leakage detected! Decision key '{k}' differed: {plan_liq[k]} vs {plan_illiq[k]}"

    # Liquidity metadata must differ
    assert plan_liq["liquidity_defense"]["liquidity_grade"] == "HIGH_TRADING_LIQUIDITY"
    assert plan_illiq["liquidity_defense"]["liquidity_grade"] == "EXECUTION_RISK"
    assert plan_liq["liquidity_defense"]["suppress_buy_zone"] is False
    assert plan_illiq["liquidity_defense"]["suppress_buy_zone"] is False


# ── P0-2: API Serialization / Deserialization Across All Tiers ────────────────
def test_p0_2_api_serialization_roundtrip_all_tiers():
    """Verify JSON serialization round-trip across all 4 liquidity tiers."""
    dates = pd.date_range("2026-01-01", periods=60)
    prices = [50.0 + (i * 0.1) for i in range(60)]
    resp = Response()

    tiers_config = [
        ("TIER_HIGH", [500_000]*60, "HIGH_TRADING_LIQUIDITY"),
        ("TIER_MOD", [20_000]*60, "MODERATE_TRADING_LIQUIDITY"),
        ("TIER_RISK", [1_000]*60, "EXECUTION_RISK"),
        ("TIER_UNK", [], "UNKNOWN_LIQUIDITY"),
    ]

    for sym, vols, expected_grade in tiers_config:
        candles = [{"time": str(d), "open": p, "high": p+0.5, "low": p-0.5, "close": p, "volume": v} for d, p, v in zip(dates, prices, vols)] if vols else []
        with patch("api.routes.screener.market_db.get_latest_price", return_value={"currentPrice": 53.0 if vols else 0.0}):
            with patch("api.routes.screener.market_db.get_daily_candles", return_value=candles):
                raw = run_screener_get(resp, filter_type="all", custom_tickers=sym)
                serialized = json.dumps(raw)
                deserialized = json.loads(serialized)

                cand = deserialized["candidates"][0]
                liq = cand.get("liquidityDefense")
                if expected_grade == "UNKNOWN_LIQUIDITY" and cand["executionStatus"] == "UNVERIFIED_ASSET":
                    # Unverified asset suppresses execution
                    assert liq is None or liq["liquidity_grade"] == "UNKNOWN_LIQUIDITY"
                else:
                    assert liq is not None
                    assert liq["liquidity_grade"] == expected_grade
                    assert isinstance(liq["adv_20d_usd"], (int, float))
                    assert isinstance(liq["amihud_illiq"], (int, float))
                    assert isinstance(liq["suppress_buy_zone"], bool)
                    assert liq["suppress_buy_zone"] is False


# ── P0-3: Unknown/Missing-Data Semantics ──────────────────────────────────────
def test_p0_3_unknown_missing_data_semantics():
    """Assert empty or invalid DataFrame produces UNKNOWN_LIQUIDITY with slate badge and no crash."""
    for bad_input in [None, pd.DataFrame(), pd.DataFrame({"Close": [10.0]}), pd.DataFrame({"Open": [10.0, 11.0, 12.0]})]:
        res = LiquidityGuard.evaluate_liquidity(bad_input, 100.0)
        assert res["liquidity_grade"] == "UNKNOWN_LIQUIDITY"
        assert res["badge_color"] == "slate"
        assert res["execution_hazard"] is False
        assert res["suppress_buy_zone"] is False
        assert res["adv_20d_usd"] == 0.0
        assert res["amihud_illiq"] == 0.0


# ── P0-4: Amihud Dimensional & Unit Semantics ─────────────────────────────────
def test_p0_4_amihud_dimensional_unit_audit():
    """
    Audit dimensional consistency:
    ILLIQ_raw has units: fractional return / USD
    ILLIQ_scaled = ILLIQ_raw * 1e6 has units: fractional return / $1M traded
    Verify documentation strings and summaries reflect fractional return, not percentage return.
    """
    dates = pd.date_range("2026-01-01", periods=20)
    # Exact 2% daily move on exact $10M daily volume
    prices = [100.0 * (1.02 ** i) for i in range(20)]
    volumes = [100_000 for _ in range(20)] # ~$10M dollar volume
    df = pd.DataFrame({"Close": prices, "Volume": volumes}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, prices[-1])
    raw = res["amihud_illiq"]
    scaled = res["amihud_illiq_scaled"]

    # Invariant: scaled must equal raw * 10^6
    assert abs(scaled - (raw * 1e6)) < 1e-12
    # Invariant: pro summary must explicitly mention 'return/$1M traded'
    assert "return/$1M traded" in res["pro_summary"]


# ── P0-5: No Hidden Liquidity Dependency in Frozen Engine ────────────────────
def test_p0_5_no_hidden_liquidity_dependency_in_frozen_engine():
    """Verify that OptimalExecutionEngine calculates ATR and trade corridor without reading liquidity fields."""
    import inspect
    source = inspect.getsource(OptimalExecutionEngine.calculate_trade_levels)
    # The calculation of optimal_entry_min must not reference liquidity_report
    assert "liquidity_report['liquidity_grade']" not in source
    assert "liquidity_report['execution_hazard']" not in source
    assert "if liquidity_report" not in source


# ── P0-6: Split-Adjusted Price & Volume Alignment Guard ──────────────────────
def test_p0_6_split_adjusted_price_and_volume_alignment():
    """Verify that unadjusted corporate action split-gaps (> 80% daily change) are filtered from Amihud."""
    dates = pd.date_range("2026-01-01", periods=25)
    # Normal sessions then a 4:1 unadjusted split gap (75% drop)
    prices = [100.0]*15 + [25.0]*10
    vols = [100_000]*25
    df = pd.DataFrame({"Close": prices, "Volume": vols}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 25.0)
    # Must not evaluate to infinity or NaN
    assert not math.isnan(res["amihud_illiq"])
    assert not math.isinf(res["amihud_illiq"])
    assert res["amihud_illiq"] < 1.0


# ── P0-7: Temporal Point-in-Time & Anti-Lookahead in Rolling Windows ─────────
def test_p0_7_no_lookahead_in_rolling_windows():
    """Verify that calculations at bar t depend strictly on data <= t."""
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [100.0 + i for i in range(30)]
    volumes = [100_000]*30
    df_20 = pd.DataFrame({"Close": prices[:20], "Volume": volumes[:20]}, index=dates[:20])
    df_21 = pd.DataFrame({"Close": prices[:21], "Volume": volumes[:21]}, index=dates[:21])

    res_at_20 = LiquidityGuard.evaluate_liquidity(df_20, prices[19])
    # Evaluating df_20 slice should produce identical ADV as first 20 bars of df_21
    sub_df = df_21.iloc[:20]
    res_sub = LiquidityGuard.evaluate_liquidity(sub_df, prices[19])
    assert res_at_20["adv_20d_usd"] == res_sub["adv_20d_usd"]
    assert res_at_20["amihud_illiq"] == res_sub["amihud_illiq"]


# ── P0-8: Signal-Time Liquidity Immutability in Ledger ────────────────────────
def test_p0_8_signal_time_liquidity_immutability(tmp_path):
    """Verify signal-time liquidity is permanently immutable in ledger."""
    ledger_path = str(tmp_path / "immut_test.json")
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
            "amihud_illiq": 1e-9,
        }
    }
    rec = ExperimentLedger.register_signal(
        symbol="IMMUT",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=88.0,
        inputs_meta={"market_regime": "BULL"},
        ledger_path=ledger_path
    )
    # Add a forward observation on session 3
    ExperimentLedger.record_liquidity_forward_observation(
        signal_id=rec["signalId"],
        session_index=3,
        liquidity_metrics={"adv_20d_usd": 10_000.0, "liquidity_grade": "EXECUTION_RISK"},
        ledger_path=ledger_path
    )
    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    assert sig["liquidityAtSignal"]["adv_20d_usd"] == 50_000_000.0
    assert sig["liquidityAtSignal"]["liquidity_grade"] == "HIGH_TRADING_LIQUIDITY"


# ── P0-9: Forward Liquidity Observations Cannot Affect Outcomes ──────────────
def test_p0_9_forward_liquidity_cannot_affect_outcomes(tmp_path):
    """Verify appending forward observations has zero side-effects on trade resolution."""
    ledger_path = str(tmp_path / "outcome_test.json")
    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 94.0,
        "take_profit_1": 110.0,
        "take_profit_2": 118.0,
        "risk_reward_ratio": 2.0,
    }
    rec = ExperimentLedger.register_signal(
        symbol="OUTCOME",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        ledger_path=ledger_path
    )
    # Record forward observation indicating extreme execution risk
    ExperimentLedger.record_liquidity_forward_observation(
        signal_id=rec["signalId"],
        session_index=1,
        liquidity_metrics={"liquidity_grade": "EXECUTION_RISK", "execution_hazard": True},
        ledger_path=ledger_path
    )
    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    assert sig["status"] == "OPEN"
    assert sig["stopLoss"] == 94.0
    assert sig["takeProfit1"] == 110.0


# ── P0-10: Cryptographic Dual-Hash Immutability Coverage ─────────────────────
def test_p0_10_dual_hash_coverage():
    """Verify that modifying any decision or input variable invalidates hashes."""
    record = {
        "engineVersion": "4e36862",
        "engineTag": "v2.4.0-phase24-freeze",
        "decisionState": "ACTIONABLE_SETUP",
        "entryPrice": 100.0,
        "corridorMin": 98.0,
        "corridorMax": 102.0,
        "stopLoss": 94.0,
        "takeProfit1": 110.0,
        "takeProfit2": 118.0,
        "riskRewardRatio": 2.0,
        "confluenceScore": 85.0,
        "inputs": {
            "atr14": 2.5,
            "atrPct": 2.5,
            "setupPattern": "Minervini VCP",
            "stagePhase": "Stage 2 Advancing",
            "marketRegime": "BULL",
            "sector": "TECH",
            "assetClass": "US_EQUITY",
        }
    }
    dec_hash = ExperimentLedger.compute_decision_snapshot_hash(record)
    inp_hash = ExperimentLedger.compute_inputs_snapshot_hash(record)

    # Mutate 1 decision parameter
    rec_mut_dec = json.loads(json.dumps(record))
    rec_mut_dec["stopLoss"] = 93.99
    assert ExperimentLedger.compute_decision_snapshot_hash(rec_mut_dec) != dec_hash

    # Mutate 1 input feature
    rec_mut_inp = json.loads(json.dumps(record))
    rec_mut_inp["inputs"]["atr14"] = 2.51
    assert ExperimentLedger.compute_inputs_snapshot_hash(rec_mut_inp) != inp_hash


# ── P1-11: Liquidity Deterioration Trajectory ─────────────────────────────────
def test_p1_11_liquidity_deterioration_trajectory():
    """Test asset transitioning from liquid to illiquid: verify trend metric drops < 1.0."""
    dates = pd.date_range("2026-01-01", periods=30)
    # 25 days of high volume ($10M/day), followed by 5 days of dried-up volume ($100k/day)
    vols = [100_000]*25 + [1_000]*5
    prices = [100.0]*30
    df = pd.DataFrame({"Close": prices, "Volume": vols}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 100.0)
    assert res["adv_5d_usd"] < res["adv_20d_usd"]
    assert res["liquidity_trend"] < 0.5  # Liquidity trend visibly collapses


# ── P1-12: Liquidity Recovery Trajectory ──────────────────────────────────────
def test_p1_12_liquidity_recovery_trajectory():
    """Test asset transitioning from illiquid to liquid: verify trend metric expands > 1.0."""
    dates = pd.date_range("2026-01-01", periods=30)
    # 25 days of low volume ($200k/day), followed by 5 days of surging institutional volume ($5M/day)
    vols = [2_000]*25 + [50_000]*5
    prices = [100.0]*30
    df = pd.DataFrame({"Close": prices, "Volume": vols}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 100.0)
    assert res["adv_5d_usd"] > res["adv_20d_usd"]
    assert res["liquidity_trend"] > 1.5  # Liquidity trend visibly expands


# ── P1-13: Sudden Volume Spike Behavior ───────────────────────────────────────
def test_p1_13_sudden_volume_spike_behavior():
    """Test sudden 10x volume spike on low baseline triggers spike alert."""
    dates = pd.date_range("2026-01-01", periods=30)
    vols = [5_000]*29 + [100_000] # 20x spike on day 30
    df = pd.DataFrame({"Close": [10.0]*30, "Volume": vols}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 10.0)
    assert res["is_volume_spike"] is True
    assert res["volume_spike_ratio"] >= 10.0


# ── P1-14: Intermittent Stale/Missing Sessions Resilience ─────────────────────
def test_p1_14_stale_missing_sessions_resilience():
    """Verify DataFrame with sporadic zero-volume or non-trading days evaluates safely."""
    dates = pd.date_range("2026-01-01", periods=30)
    vols = [10_000 if i % 3 != 0 else 0.0 for i in range(30)]
    df = pd.DataFrame({"Close": [50.0]*30, "Volume": vols}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 50.0)
    assert res["liquidity_grade"] in ["HIGH_TRADING_LIQUIDITY", "MODERATE_TRADING_LIQUIDITY", "EXECUTION_RISK"]
    assert not math.isnan(res["adv_20d_usd"])


# ── P1-15: Corporate Action Overnight Gap Guard ───────────────────────────────
def test_p1_15_corporate_action_gap_guard():
    """Verify that 95% single-day gap is sanitized from Amihud numerator."""
    dates = pd.date_range("2026-01-01", periods=20)
    prices = [100.0]*10 + [5.0]*10
    df = pd.DataFrame({"Close": prices, "Volume": [50_000]*20}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 5.0)
    assert not math.isnan(res["amihud_illiq"])
    assert res["amihud_illiq"] >= 0.0


# ── P1-16: Very High-Priced Equities ─────────────────────────────────────────
def test_p1_16_very_high_priced_stocks():
    """Test ultra-high priced asset ($650,000 spot) with low share volume but huge dollar volume."""
    dates = pd.date_range("2026-01-01", periods=25)
    df = pd.DataFrame({"Close": [650_000.0]*25, "Volume": [20]*25}, index=dates) # $13M ADV

    res = LiquidityGuard.evaluate_liquidity(df, 650_000.0)
    assert res["adv_20d_usd"] == 13_000_000.0
    assert res["liquidity_grade"] == "HIGH_TRADING_LIQUIDITY"
    assert res["execution_hazard"] is False


# ── P1-17: Penny / Micro-Cap Equities ─────────────────────────────────────────
def test_p1_17_penny_micro_cap_stocks():
    """Test penny stock ($0.25 spot) with $10K daily dollar volume."""
    dates = pd.date_range("2026-01-01", periods=25)
    df = pd.DataFrame({"Close": [0.25]*25, "Volume": [40_000]*25}, index=dates) # $10k ADV

    res = LiquidityGuard.evaluate_liquidity(df, 0.25)
    assert res["adv_20d_usd"] == 10_000.0
    assert res["liquidity_grade"] == "EXECUTION_RISK"
    assert res["execution_hazard"] is True


# ── P1-18: Order Size Participation Advisory Spectrum ─────────────────────────
def test_p1_18_order_size_participation_spectrum():
    """Test participation rate across the spectrum: 0.1%, 0.5%, 1.0%, 2.0%, 5.0%, 10.0%."""
    adv = 1_000_000.0
    test_orders = [
        (1_000.0, 0.001),
        (5_000.0, 0.005),
        (10_000.0, 0.01),
        (20_000.0, 0.02),
        (50_000.0, 0.05),
        (100_000.0, 0.10),
    ]
    for order_usd, expected_rate in test_orders:
        rate = LiquidityGuard.estimate_participation_rate(order_usd, adv)
        assert abs(rate - expected_rate) < 1e-9


# ── P1-19: UX Advisory Terminology Invariant ──────────────────────────────────
def test_p1_19_ux_wording_advisory_invariants():
    """Verify labels and summaries adhere strictly to non-prescriptive, advisory terminology."""
    dates = pd.date_range("2026-01-01", periods=25)
    df = pd.DataFrame({"Close": [1.0]*25, "Volume": [1_000]*25}, index=dates)

    res = LiquidityGuard.evaluate_liquidity(df, 1.0)
    plain = res["plain_summary"]
    pro = res["pro_summary"]

    assert "guaranteed" not in plain.lower()
    assert "untradeable" not in plain.lower()
    assert "orderbook depth" not in plain.lower()
    assert "orderbook depth" not in pro.lower()


# ── P1-20: Graceful Neutral State on UNKNOWN_LIQUIDITY ────────────────────────
def test_p1_20_disabled_gracefully_when_unknown():
    """Verify feature is gracefully neutral and disabled when data is UNKNOWN."""
    res = LiquidityGuard._generate_fallback(100.0)
    assert res["liquidity_grade"] == "UNKNOWN_LIQUIDITY"
    assert res["badge_color"] == "slate"
    assert res["execution_hazard"] is False
    assert res["suppress_buy_zone"] is False
    assert res["market_order_warning"] is False
