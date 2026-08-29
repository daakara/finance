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
    
    all_data = run_screener_get(filter_type="all")
    all_candidates = all_data.get("candidates", [])
    all_syms = set(c["symbol"] for c in all_candidates)
    assert len(all_candidates) == len(DEFAULT_CANDIDATES), "All filter must return the full universe"

    filters = [
        ("high_confluence", lambda c: c["confluenceScore"] >= 80.0),
        ("in_buy_zone", lambda c: c["executionStatus"] == "IN_BUY_ZONE"),
        ("approaching_target", lambda c: c["executionStatus"] == "APPROACHING_TARGET"),
        ("high_rr", lambda c: c["riskRewardRatio"] >= 2.0 and (c["executionStatus"] == "IN_BUY_ZONE" or c["currentPrice"] <= c.get("optimalEntryMax", c["currentPrice"]) * 1.02)),
        ("lynch", lambda c: (0 < float(c["pegRatio"]) <= 1.05) or "Lynch" in c["expertArchetype"] or "GARP" in c["expertArchetype"]),
        ("greenblatt", lambda c: float(c["roic"].replace("%", "")) >= 28.0 or "Greenblatt" in c["expertArchetype"] or "Magic" in c["expertArchetype"]),
        ("rule_breakers", lambda c: float(c["grossMargin"].replace("%", "")) >= 65.0 or "Rule Breakers" in c["expertArchetype"] or "Disruptive" in c["expertArchetype"]),
    ]

    for filter_name, validator in filters:
        data = run_screener_get(filter_type=filter_name)
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
    buy_zone_data = run_screener_get(filter_type="in_buy_zone")
    tp_target_data = run_screener_get(filter_type="approaching_target")
    
    buy_zone_syms = set(c["symbol"] for c in buy_zone_data.get("candidates", []))
    tp_target_syms = set(c["symbol"] for c in tp_target_data.get("candidates", []))

    # Assert Disjoint Sets: No asset can be in Buy Zone AND Approaching Target simultaneously
    overlap = buy_zone_syms.intersection(tp_target_syms)
    assert len(overlap) == 0, f"Detected state overlap between IN_BUY_ZONE and APPROACHING_TARGET: {overlap}"


def test_screener_candidate_data_integrity():
    """Verify all candidate fields meet strict institutional boundary invariants."""
    resp = Response()
    all_data = run_screener_get(filter_type="all")
    
    for c in all_data.get("candidates", []):
        sym = c["symbol"]
        assert c["currentPrice"] > 0, f"{sym}: Price must be positive"
        assert c["gemScore"] >= 0 and c["gemScore"] <= 100, f"{sym}: GemScore out of bounds"
        assert c["confluenceScore"] >= 0 and c["confluenceScore"] <= 100, f"{sym}: ConfluenceScore out of bounds"
        assert c["stopLoss"] < c["optimalEntryMin"] <= c["optimalEntryMax"] < c["takeProfit1"], f"{sym}: Invalid level ladder"
def test_screener_dual_horizon_distinct_universes():
    """Verify that Day Trader mode and Long-Term mode return distinct asset universes tailored to their horizon."""
    resp = Response()

    day_data = run_screener_get(resp, filter_type="all", user_role="DAY_TRADER")
    long_data = run_screener_get(resp, filter_type="all", user_role="LONG_TERM")

    day_syms = set(c["symbol"] for c in day_data.get("candidates", []))
    long_syms = set(c["symbol"] for c in long_data.get("candidates", []))

    assert len(day_syms) > 0, "Day trader universe must not be empty"
    assert len(long_syms) > 0, "Long term universe must not be empty"
    
    # Assert distinct sets tailored to each profile
    assert day_syms != long_syms, "Day trader and Long term modes must recommend distinct universe profiles"
    assert "NVDA" in day_syms or "TSLA" in day_syms, "Day trader universe must contain high-beta volatility leaders"
    assert "LNTH" in long_syms or "CPRX" in long_syms, "Long term universe must contain high-ROIC compounders"


def test_screener_custom_tickers_on_demand():
    """Verify on-demand custom ticker list evaluation."""
    resp = Response()
    custom_list = "AAPL, AMD, META, AVGO, CRWD"
    data = run_screener_get(resp, filter_type="all", user_role="DAY_TRADER", custom_tickers=custom_list)
    candidates = data.get("candidates", [])
    syms = [c["symbol"] for c in candidates]

    assert len(syms) == 5, "Must evaluate all 5 custom tickers"
    assert "AAPL" in syms and "AMD" in syms and "META" in syms, "Custom symbols must match parsed input"
    for c in candidates:
        assert c["currentPrice"] > 0
        assert c["stopLoss"] < c["optimalEntryMin"] <= c["optimalEntryMax"] < c["takeProfit1"]


def test_screener_cloudflare_cache_control_headers():
    """Verify that screener endpoints output Cloudflare edge SWR and stale-if-error headers."""
    resp = Response()
    run_screener_get(resp, filter_type="all")
    assert "Cache-Control" in resp.headers
    assert "stale-while-revalidate=86400" in resp.headers["Cache-Control"]
    assert "stale-if-error=86400" in resp.headers["Cache-Control"]
    assert resp.headers.get("Cloudflare-CDN-Cache-Control") == "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"


import unittest

class TestScreenerExecution(unittest.TestCase):
    def test_optimal_execution_levels(self):
        test_optimal_execution_levels()

    def test_screener_differential_subsets(self):
        test_screener_differential_subsets()

    def test_screener_disjoint_execution_states(self):
        test_screener_disjoint_execution_states()

    def test_screener_candidate_data_integrity(self):
        test_screener_candidate_data_integrity()

    def test_screener_dual_horizon_distinct_universes(self):
        test_screener_dual_horizon_distinct_universes()

    def test_screener_custom_tickers_on_demand(self):
        test_screener_custom_tickers_on_demand()

    def test_screener_cloudflare_cache_control_headers(self):
        test_screener_cloudflare_cache_control_headers()


if __name__ == "__main__":
    unittest.main()


