"""ARX Decision Parity & State Consensus Test Suite (Phase 21).

Verifies the non-negotiable decision parity across the 6 canonical golden assets:
  1. NVDA: Strong institutional compounder outside buy zone -> VALID_SETUP, isActionable: False.
  2. FIX: Stage 4 distribution -> VALID_SETUP, isActionable: False, sizing blocked.
  3. CPRX: Stale historical tape (> 4 days) -> STALE_DATA, triggers suspended.
  4. SPY: Broad-market ETF outside corridor -> VALID_SETUP, isActionable: False.
  5. GLX: Insufficient history (< 50 sessions) -> INSUFFICIENT_DATA, trade levels suppressed.
  6. TEST_AAA: Unknown uncataloged asset -> UNVERIFIED (Precedence 1) / 404 Fail-Closed.
"""

import pytest
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine
from tests.fixtures.golden_universe import GOLDEN_ASSETS


def test_decision_trace_contract_completeness():
    """Verify that DecisionTraceEngine produces a complete, strongly-typed trace payload."""
    trace = DecisionTraceEngine.build_decision_trace(
        symbol="NVDA",
        current_price=228.45,
        candles=[{"open": 220, "high": 230, "low": 218, "close": 228.45}] * 252,
        freshness={"status": "LIVE", "providerSource": "yfinance", "stalenessDays": 1, "candleCount": 252},
        technicals={"sma_50": 230.42, "ema_20": 225.10},
        confluence={"confluenceScore": 74.6, "verdict": "Valid Setup", "pillars": []},
        factor_scores={"compositeFactorScore": 74.6},
        optimal_execution={
            "optimal_entry_min": 215.00,
            "optimal_entry_max": 221.83,
            "stop_loss": 208.73,
            "take_profit_1": 283.32,
            "risk_reward_ratio": 2.85,
            "execution_status": "APPROACHING_TARGET",
        },
        smart_money={"has_congress_buy": False, "has_insider_buy": False, "optionsFlow": []},
        macro_difficulty={"rating": 45, "regime": "NEUTRAL"},
    )

    assert trace["symbol"] == "NVDA"
    assert trace["decisionState"] in [s.value for s in DecisionState]
    assert "decisionStateLabel" in trace and trace["decisionStateLabel"] is not None
    assert isinstance(trace["isActionable"], bool)
    assert isinstance(trace["canSizeTrade"], bool)
    assert isinstance(trace["allowedActions"], list)
    assert "disqualificationReason" in trace
    assert "trace" in trace
    assert "executionPlan" in trace["trace"]


def test_golden_parity_nvda_awaiting_pullback():
    """NVDA: Current price $228.45 is above entry ceiling $221.83 -> VALID_SETUP, not actionable."""
    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="NVDA",
        current_price=228.45,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=74.6,
        stage_phase=2,
        is_in_buy_zone=False,
        risk_reward_ratio=2.85,
        is_cataloged=True,
    )

    assert state["state"] == DecisionState.VALID_SETUP.value
    assert state["isActionable"] is False
    assert state["canSizeTrade"] is False
    assert "outside the optimal entry corridor" in state["disqualificationReason"]


def test_golden_parity_fix_stage4_distribution():
    """FIX: Price below 50 SMA in Stage 4 distribution -> VALID_SETUP, isActionable: False, sizing blocked."""
    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="FIX",
        current_price=1580.19,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=56.6,
        stage_phase=4,
        is_in_buy_zone=False,
        risk_reward_ratio=1.15,
        is_cataloged=True,
    )

    assert state["state"] == DecisionState.VALID_SETUP.value
    assert state["isActionable"] is False
    assert state["canSizeTrade"] is False
    assert "Stage 4 distribution" in state["disqualificationReason"]


def test_golden_parity_cprx_stale_data_precedence():
    """CPRX: 45-day stale historical tape -> resolves STALE_DATA (Precedence 3), triggers suspended."""
    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="CPRX",
        current_price=31.49,
        candle_count=105,
        freshness_status="STALE_HISTORICAL",
        has_fundamentals=True,
        confluence_score=41.2,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_cataloged=True,
    )

    assert state["state"] == DecisionState.STALE_DATA.value
    assert state["isActionable"] is False
    assert state["canSizeTrade"] is False
    assert "stale" in state["disqualificationReason"].lower()


def test_golden_parity_spy_etf_corridor():
    """SPY: Broad-market ETF outside entry corridor -> VALID_SETUP, isActionable: False."""
    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="SPY",
        current_price=773.17,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=74.3,
        stage_phase=2,
        is_in_buy_zone=False,
        risk_reward_ratio=1.82,
        is_cataloged=True,
    )

    assert state["state"] == DecisionState.VALID_SETUP.value
    assert state["isActionable"] is False
    assert state["canSizeTrade"] is False


def test_golden_parity_glx_insufficient_history():
    """GLX: 47 sessions (< 50) -> resolves INSUFFICIENT_DATA (Precedence 2), trade sizing strictly blocked."""
    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="GLX",
        current_price=17.53,
        candle_count=47,
        freshness_status="LIVE",
        has_fundamentals=False,
        confluence_score=46.2,
        stage_phase=None,
        is_in_buy_zone=False,
        risk_reward_ratio=None,
        is_cataloged=True,
    )

    assert state["state"] == DecisionState.INSUFFICIENT_DATA.value
    assert state["isActionable"] is False
    assert state["canSizeTrade"] is False
    assert "47 provided" in state["disqualificationReason"]


def test_golden_parity_test_aaa_unknown_unverified():
    """TEST_AAA: Unknown ticker without exchange tape -> UNVERIFIED (Precedence 1)."""
    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST_AAA",
        current_price=0.0,
        candle_count=0,
        freshness_status="UNAVAILABLE",
        has_fundamentals=False,
        confluence_score=0.0,
        is_cataloged=False,
    )

    assert state["state"] == DecisionState.UNVERIFIED.value
    assert state["isActionable"] is False
    assert state["canSizeTrade"] is False


def test_decision_hierarchy_exhaustive_precedence():
    """Exhaustive test verifying that higher precedence states always trump lower precedence conditions."""
    # Precedence 1 (UNVERIFIED) trumps all, even if candles >= 50 and score == 99
    p1 = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST_P1",
        current_price=0.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=95.0,
        is_in_buy_zone=True,
        risk_reward_ratio=3.0,
        is_cataloged=False,
    )
    assert p1["state"] == DecisionState.UNVERIFIED.value

    # Precedence 2 (INSUFFICIENT_DATA) trumps STALE_DATA and EVIDENCE_INCOMPLETE
    p2 = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST_P2",
        current_price=50.0,
        candle_count=30,
        freshness_status="STALE_HISTORICAL",
        has_fundamentals=False,
        confluence_score=80.0,
        is_in_buy_zone=True,
        risk_reward_ratio=3.0,
        is_cataloged=True,
    )
    assert p2["state"] == DecisionState.INSUFFICIENT_DATA.value

    # Precedence 3 (STALE_DATA) trumps EVIDENCE_INCOMPLETE and ACTIONABLE
    p3 = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST_P3",
        current_price=50.0,
        candle_count=100,
        freshness_status="STALE_HISTORICAL",
        has_fundamentals=False,
        confluence_score=90.0,
        is_in_buy_zone=True,
        risk_reward_ratio=3.0,
        is_cataloged=True,
    )
    assert p3["state"] == DecisionState.STALE_DATA.value

    # Precedence 4 (EVIDENCE_INCOMPLETE) trumps ACTIONABLE
    p4 = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST_P4",
        current_price=50.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=False,
        confluence_score=85.0,
        is_in_buy_zone=True,
        risk_reward_ratio=3.0,
        is_cataloged=True,
    )
    assert p4["state"] == DecisionState.EVIDENCE_INCOMPLETE.value
