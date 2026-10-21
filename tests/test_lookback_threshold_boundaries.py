"""
tests/test_lookback_threshold_boundaries.py

Sprint 2 Track C: Lookback Threshold Reconciliation Boundary Tests (Phase 4).
Reconciles the three-tier historical bar threshold hierarchy:
1. Technical Validation Minimum Floor: 50 daily bars (MarketDataQualityEvidence threshold).
2. Operational Builder Readiness Contract: 60 daily bars (UniverseBuilder / Universe publication data readiness).
3. Strategy Benchmark Readiness: 260 daily bars (52-week lookback for Minervini Trend Template).

Boundary tests executed at exactly: 49, 50, 59, 60, 259, 260 daily bars.
"""

import pytest

from analyst_dashboard.universe.isolated_publication import (
    IsolatedUniverseCandidate,
    IsolatedPublicationEngine,
)
from analyst_dashboard.universe.publication_state_machine import (
    PublicationState,
    PublicationStateMachine,
    MarketDataQualityEvidence,
    StrategyAdmissionEvidence,
    PolicyViolationError,
)


def _make_candidate(candle_count: int, symbol: str = "AAPL") -> IsolatedUniverseCandidate:
    return IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol=symbol,
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        security_type="COMMON_STOCK",
        asset_class="EQUITY",
        listing_status="ACTIVE",
        currency="USD",
        primary_listing=True,
        alpaca_symbol=symbol,
        candle_count=candle_count,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )


def test_boundary_49_bars_fails_technical_floor():
    """
    49 bars: strictly below 50-bar technical validation minimum floor.
    Must fail closed in MarketDataQualityEvidence. State remains PROVIDER_RESOLVABLE.
    """
    # 1. State machine level: MarketDataQualityEvidence raises PolicyViolationError
    ev = MarketDataQualityEvidence(
        symbol="AAPL",
        candle_count=49,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
    )
    with pytest.raises(PolicyViolationError, match="Insufficient historical coverage.*builder floor.*50"):
        ev.validate()

    with pytest.raises(PolicyViolationError):
        PublicationStateMachine.transition(
            PublicationState.PROVIDER_RESOLVABLE,
            PublicationState.DATA_QUALIFIED,
            ev,
        )

    # 2. Engine level: evaluate_candidate sets DATA_UNAVAILABLE and state remains PROVIDER_RESOLVABLE
    engine = IsolatedPublicationEngine()
    cand = _make_candidate(49)
    res = engine.evaluate_candidate(cand, "2026-10-19")
    assert res.current_publication_state == PublicationState.PROVIDER_RESOLVABLE
    assert res.data_readiness_decision == "DATA_UNAVAILABLE"
    assert "DATA_QUALIFICATION_FAILED" in res.eligibility_reason

    # 3. Dry-run cannot publish 49-bar candidate
    dry_run = engine.execute_dry_run([cand], "2026-10-19")
    assert dry_run["published_symbols"] == []
    assert dry_run["accounting"]["data_ready_count"] == 0


def test_boundary_50_bars_passes_technical_floor_but_fails_builder_contract():
    """
    50 bars: meets 50-bar technical validation floor, entering DATA_QUALIFIED,
    but fails 60-bar operational builder readiness contract for universe inclusion.
    """
    # 1. State machine level: passes MarketDataQualityEvidence
    ev = MarketDataQualityEvidence(
        symbol="AAPL",
        candle_count=50,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
    )
    ev.validate()
    state = PublicationStateMachine.transition(
        PublicationState.PROVIDER_RESOLVABLE,
        PublicationState.DATA_QUALIFIED,
        ev,
    )
    assert state == PublicationState.DATA_QUALIFIED

    # 2. Engine level: passes to PUBLICATION_ELIGIBLE but data_readiness is DATA_UNAVAILABLE (< 60)
    engine = IsolatedPublicationEngine()
    cand = _make_candidate(50)
    res = engine.evaluate_candidate(cand, "2026-10-19")
    assert res.current_publication_state == PublicationState.PUBLICATION_ELIGIBLE
    assert res.eligibility_decision == "ELIGIBLE"
    assert res.data_readiness_decision == "DATA_UNAVAILABLE"
    assert "INSUFFICIENT_BUILDER_READINESS_HISTORY: 50 bars < 60 bars operational contract" in res.data_readiness_reason

    # 3. Dry-run cannot publish because data_readiness is not DATA_READY
    dry_run = engine.execute_dry_run([cand], "2026-10-19")
    assert dry_run["published_symbols"] == []
    assert dry_run["accounting"]["data_ready_count"] == 0
    assert dry_run["accounting"]["data_unavailable_count"] == 1


def test_boundary_59_bars_fails_builder_contract():
    """
    59 bars: passes technical floor (50 bars), but strictly below 60-bar builder contract.
    """
    engine = IsolatedPublicationEngine()
    cand = _make_candidate(59)
    res = engine.evaluate_candidate(cand, "2026-10-19")
    assert res.current_publication_state == PublicationState.PUBLICATION_ELIGIBLE
    assert res.eligibility_decision == "ELIGIBLE"
    assert res.data_readiness_decision == "DATA_UNAVAILABLE"
    assert "INSUFFICIENT_BUILDER_READINESS_HISTORY: 59 bars < 60 bars operational contract" in res.data_readiness_reason

    dry_run = engine.execute_dry_run([cand], "2026-10-19")
    assert dry_run["published_symbols"] == []
    assert dry_run["accounting"]["data_ready_count"] == 0


def test_boundary_60_bars_satisfies_builder_contract_published():
    """
    60 bars: meets operational builder readiness contract.
    Qualifies as DATA_READY and is published to the universe pointer,
    but fails 52-week Strategy Benchmark readiness (260 bars).
    """
    # 1. Engine level: DATA_READY and PUBLISHED
    engine = IsolatedPublicationEngine()
    cand = _make_candidate(60)
    res = engine.evaluate_candidate(cand, "2026-10-19")
    assert res.current_publication_state == PublicationState.PUBLICATION_ELIGIBLE
    assert res.eligibility_decision == "ELIGIBLE"
    assert res.data_readiness_decision == "DATA_READY"
    assert res.data_readiness_reason == "PASSED_MARKET_DATA_QUALITY"

    dry_run = engine.execute_dry_run([cand], "2026-10-19")
    assert dry_run["published_symbols"] == ["AAPL"]
    assert dry_run["accounting"]["data_ready_count"] == 1
    assert dry_run["accounting"]["scannable_count"] == 1

    # 2. Strategy benchmark check: 60 bars fails 260-bar strategy benchmark
    strat_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="SMART_MONEY",
        strategy_type="SMART_MONEY",
        min_price_floor=10.0,
        min_history_bars=cand.candle_count,  # 60 bars
        admission_approved=True,
        approved_by="SMART_MONEY_LEAD",
    )
    with pytest.raises(PolicyViolationError, match="Strategy benchmark violation.*60 bars < 260 bars"):
        strat_ev.validate()


def test_boundary_259_bars_published_fails_strategy_benchmark():
    """
    259 bars: satisfies builder readiness (published), but strictly below
    260-bar Strategy Benchmark readiness.
    """
    engine = IsolatedPublicationEngine()
    cand = _make_candidate(259)
    res = engine.evaluate_candidate(cand, "2026-10-19")
    assert res.data_readiness_decision == "DATA_READY"

    dry_run = engine.execute_dry_run([cand], "2026-10-19")
    assert dry_run["published_symbols"] == ["AAPL"]

    # Fails 260-bar strategy benchmark
    strat_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        strategy_type="VCP",
        g_vcp_012_certified=True,
        min_price_floor=10.0,
        min_history_bars=259,
        admission_approved=True,
        approved_by="VCP_LEAD",
    )
    with pytest.raises(PolicyViolationError, match="Strategy benchmark violation.*259 bars < 260 bars"):
        strat_ev.validate()


def test_boundary_260_bars_satisfies_all_three_thresholds():
    """
    260 bars: satisfies technical floor (50), builder readiness (60),
    and 52-week Strategy Benchmark readiness (260).
    Can transition through State 5 to State 6 (STRATEGY_ADMITTED).
    """
    engine = IsolatedPublicationEngine()
    cand = _make_candidate(260)
    res = engine.evaluate_candidate(cand, "2026-10-19")
    assert res.data_readiness_decision == "DATA_READY"

    dry_run = engine.execute_dry_run([cand], "2026-10-19")
    assert dry_run["published_symbols"] == ["AAPL"]

    # Strategy admission passes
    strat_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="SMART_MONEY",
        strategy_type="SMART_MONEY",
        strategy_certified=True,
        min_price_floor=10.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="SMART_MONEY_LEAD",
    )
    strat_ev.validate()

    next_state = PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        strat_ev,
    )
    assert next_state == PublicationState.STRATEGY_ADMITTED
