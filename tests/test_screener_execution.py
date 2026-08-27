"""Industrial-grade Differential and Invariant QA Test Suite for Screener Execution Scanner."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from api.routes.screener import run_screener_get, DEFAULT_CANDIDATES
from fastapi import Response

try:
    import pandas as pd
except ImportError:
    pd = None


def test_optimal_execution_levels():
    """Verify ATR-derived trade levels maintain monotonic ladder integrity."""
    engine = OptimalExecutionEngine()
    
    current_price = 106.80
    if pd is not None:
        dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "Open": [100.0 + (i * 0.2) for i in range(30)],
            "High": [102.0 + (i * 0.2) for i in range(30)],
            "Low": [99.0 + (i * 0.2) for i in range(30)],
            "Close": [101.0 + (i * 0.2) for i in range(30)],
            "Volume": [1000000 for _ in range(30)],
        }, index=dates)
    else:
        df = []

    levels = engine.calculate_trade_levels(df, current_price, user_role="LONG_TERM")

    assert "optimal_entry_min" in levels
    assert "optimal_entry_max" in levels
    assert "stop_loss" in levels
    assert "take_profit_1" in levels
    assert "take_profit_2" in levels
    assert "risk_reward_ratio" in levels

    # Invariant: Stop Loss < Entry Min <= Entry Max < Take Profit 1 < Take Profit 2
    assert levels["stop_loss"] < levels["optimal_entry_min"], "Stop loss must be strictly below Entry Min"
    assert levels["optimal_entry_min"] <= levels["optimal_entry_max"], "Entry Min must be <= Entry Max"
    assert levels["optimal_entry_max"] < levels["take_profit_1"], "Entry Max must be below Take Profit 1"
    assert levels["take_profit_1"] < levels["take_profit_2"], "Take Profit 1 must be below Take Profit 2"
    assert levels["risk_reward_ratio"] >= 1.0, "Risk-Reward ratio must be at least 1.0"


def test_screener_differential_subsets():
    """Differential Test: Verify that every filter returns a distinct, non-empty, mathematically valid subset."""
    resp = Response()
    
    all_data = run_screener_get(resp, filter_type="all")
    all_candidates = all_data.get("candidates", [])
    all_syms = set(c["symbol"] for c in all_candidates)
    assert len(all_candidates) == len(DEFAULT_CANDIDATES), "All filter must return the full universe"

    filters = [
        ("high_confluence", lambda c: c["confluenceScore"] >= 80.0),
        ("in_buy_zone", lambda c: c["executionStatus"] == "IN_BUY_ZONE"),
        ("approaching_target", lambda c: c["executionStatus"] == "APPROACHING_TARGET"),
        ("high_rr", lambda c: c["riskRewardRatio"] >= 2.0),
        ("lynch", lambda c: "Lynch" in c["expertArchetype"]),
        ("greenblatt", lambda c: "Greenblatt" in c["expertArchetype"] or "Magic" in c["expertArchetype"]),
        ("rule_breakers", lambda c: "Rule Breakers" in c["expertArchetype"] or "Disruptive" in c["expertArchetype"]),
    ]

    for filter_name, validator in filters:
        data = run_screener_get(resp, filter_type=filter_name)
        candidates = data.get("candidates", [])
        syms = set(c["symbol"] for c in candidates)

        # 1. Non-Zero Distribution Assertion: No filter should return 0 results
        assert len(candidates) > 0, f"Filter '{filter_name}' returned 0 results; must return non-zero candidates"
        
        # 2. Strict Subset Assertion: Filtered set must be a proper subset of total universe
        assert len(candidates) <= len(all_candidates), f"Filter '{filter_name}' count cannot exceed total universe"
        assert syms.issubset(all_syms), f"Filter '{filter_name}' contained unknown symbols not in universe"

        # 3. Filter Criterion Compliance: Every returned asset must satisfy the filter rule
        for cand in candidates:
            assert validator(cand), f"Candidate {cand['symbol']} failed validation rule for filter '{filter_name}'"


def test_screener_disjoint_execution_states():
    """Verify that execution state categories are mutually exclusive (disjoint)."""
    resp = Response()
    
    buy_zone_data = run_screener_get(resp, filter_type="in_buy_zone")
    tp_target_data = run_screener_get(resp, filter_type="approaching_target")
    
    buy_zone_syms = set(c["symbol"] for c in buy_zone_data.get("candidates", []))
    tp_target_syms = set(c["symbol"] for c in tp_target_data.get("candidates", []))

    # Assert Disjoint Sets: No asset can be in Buy Zone AND Approaching Target simultaneously
    overlap = buy_zone_syms.intersection(tp_target_syms)
    assert len(overlap) == 0, f"Detected state overlap between IN_BUY_ZONE and APPROACHING_TARGET: {overlap}"


def test_screener_candidate_data_integrity():
    """Verify all candidate fields meet strict institutional boundary invariants."""
    resp = Response()
    all_data = run_screener_get(resp, filter_type="all")
    
    for c in all_data.get("candidates", []):
        sym = c["symbol"]
        assert c["currentPrice"] > 0, f"{sym}: Price must be positive"
        assert c["gemScore"] >= 0 and c["gemScore"] <= 100, f"{sym}: GemScore out of bounds"
        assert c["confluenceScore"] >= 0 and c["confluenceScore"] <= 100, f"{sym}: ConfluenceScore out of bounds"
        assert c["stopLoss"] < c["optimalEntryMin"] <= c["optimalEntryMax"] < c["takeProfit1"], f"{sym}: Invalid level ladder"
        assert c["executionStatus"] in ["IN_BUY_ZONE", "APPROACHING_TARGET", "WAITING_PULLBACK", "STOPPED_OUT"]


if __name__ == "__main__":
    test_optimal_execution_levels()
    test_screener_differential_subsets()
    test_screener_disjoint_execution_states()
    test_screener_candidate_data_integrity()
    print("[PASS] ALL SCREENER EXECUTION & DIFFERENTIAL TESTS PASSED SUCCESSFULLY")

