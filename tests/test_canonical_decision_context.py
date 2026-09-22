"""Unit tests for ARX Canonical Decision Context and Evidence Contracts (Phase 0/1)."""

import json
import os
import pytest
from analyst_dashboard.governance.decision_context import (
    ARXDecision,
    ARXDecisionAuthority,
    ARXDecisionContext,
    ARXEvidenceItem,
    DataCompleteness,
    DecisionState,
    DecisionVerdict,
    EvidenceDomain,
    EvidenceQualityState,
    ExecutionLevels,
    FundamentalEvidenceContract,
    LiquidityEvidenceContract,
    MacroEvidenceContract,
    MarketEvidenceContract,
    PointInTimeStatus,
    TimeHorizon,
    UserRole,
    can_quality_contribute_evidence,
    can_quality_create_actionability,
    create_degraded_decision_context,
    evaluate_actionability,
    is_status_actionable,
)


def test_status_actionability():
    """Verify individual execution status actionability."""
    assert is_status_actionable("IN_BUY_ZONE") is True
    assert is_status_actionable("READY_TO_BUY") is True
    assert is_status_actionable("UNVERIFIED_ASSET") is False
    assert is_status_actionable("INSUFFICIENT_HISTORY") is False
    assert is_status_actionable("WAITING_PULLBACK") is False
    assert is_status_actionable("STOPPED_OUT") is False
    assert is_status_actionable(None) is False
    assert is_status_actionable("") is False


def test_joint_decision_actionability_fail_closed():
    """Verify strict fail-closed joint evaluation between DecisionState and ExecutionStatus."""
    # Only ACTIONABLE_SETUP + (IN_BUY_ZONE | READY_TO_BUY) is actionable
    assert evaluate_actionability(DecisionState.ACTIONABLE_SETUP, "IN_BUY_ZONE") is True
    assert evaluate_actionability("ACTIONABLE_SETUP", "READY_TO_BUY") is True

    # Any other combination fails closed
    assert evaluate_actionability(DecisionState.VALID_SETUP, "IN_BUY_ZONE") is False
    assert evaluate_actionability(DecisionState.EVIDENCE_INCOMPLETE, "IN_BUY_ZONE") is False
    assert evaluate_actionability(DecisionState.UNVERIFIED, "IN_BUY_ZONE") is False
    assert evaluate_actionability(DecisionState.STALE_DATA, "IN_BUY_ZONE") is False
    assert evaluate_actionability(DecisionState.INSUFFICIENT_DATA, "IN_BUY_ZONE") is False
    assert evaluate_actionability(DecisionState.ACTIONABLE_SETUP, "UNVERIFIED_ASSET") is False
    assert evaluate_actionability(DecisionState.ACTIONABLE_SETUP, "WAITING_PULLBACK") is False

    # Missing or None state fails closed
    assert evaluate_actionability(None, "IN_BUY_ZONE") is False
    assert evaluate_actionability("", "IN_BUY_ZONE") is False
    assert evaluate_actionability(DecisionState.ACTIONABLE_SETUP, None) is False


def test_evidence_quality_governance_rules():
    """Verify decision eligibility semantics for every EvidenceQualityState."""
    # Only AUTHORITATIVE can create canonical actionability
    assert can_quality_create_actionability(EvidenceQualityState.AUTHORITATIVE) is True
    assert can_quality_create_actionability(EvidenceQualityState.PROVISIONAL) is False
    assert can_quality_create_actionability(EvidenceQualityState.FALLBACK) is False
    assert can_quality_create_actionability(EvidenceQualityState.UNAVAILABLE) is False
    assert can_quality_create_actionability(EvidenceQualityState.STALE) is False

    # AUTHORITATIVE, PROVISIONAL, and FALLBACK may contribute evidence; UNAVAILABLE and STALE may not
    assert can_quality_contribute_evidence(EvidenceQualityState.AUTHORITATIVE) is True
    assert can_quality_contribute_evidence(EvidenceQualityState.PROVISIONAL) is True
    assert can_quality_contribute_evidence(EvidenceQualityState.FALLBACK) is True
    assert can_quality_contribute_evidence(EvidenceQualityState.UNAVAILABLE) is False
    assert can_quality_contribute_evidence(EvidenceQualityState.STALE) is False


def test_fundamental_point_in_time_contract():
    """Verify FundamentalEvidenceContract enforces PIT vs CURRENT_ONLY distinction."""
    pit_fund = FundamentalEvidenceContract(
        pe_ratio=25.0,
        market_cap=1000000000.0,
        sector="Technology",
        source="sec_edgar",
        fetched_at="2026-08-28T20:00:00Z",
        as_of="2026-06-30",
        filing_date="2026-08-28",
        available_from="2026-08-28T21:00:00Z",
        point_in_time_status=PointInTimeStatus.POINT_IN_TIME,
        quality=EvidenceQualityState.AUTHORITATIVE,
    )
    assert pit_fund.point_in_time_status == PointInTimeStatus.POINT_IN_TIME
    assert pit_fund.quality == EvidenceQualityState.AUTHORITATIVE

    # CURRENT_ONLY cannot masquerade as historical PIT
    current_fund = FundamentalEvidenceContract(
        pe_ratio=25.0,
        source="current_quote_aggregator",
        point_in_time_status=PointInTimeStatus.CURRENT_ONLY,
        quality=EvidenceQualityState.PROVISIONAL,
    )
    assert current_fund.point_in_time_status != PointInTimeStatus.POINT_IN_TIME


def test_regime_semantic_separation():
    """Verify distinct regime concepts are preserved separately in MacroEvidenceContract."""
    macro = MacroEvidenceContract(
        tactical_equity_regime="BULL_TRENDING",
        structural_macro_regime="EXPANSION",
        macro_risk_friction="STABLE",
        regime_label="Bull Trending / Macro Expansion",
        yield_spread_10y_2y=0.15,
        inflation_rate=2.6,
        fred_observation_date="2026-09-18",
        source="fred",
    )
    assert macro.tactical_equity_regime == "BULL_TRENDING"
    assert macro.structural_macro_regime == "EXPANSION"
    assert macro.macro_risk_friction == "STABLE"
    assert macro.regime_label != macro.tactical_equity_regime


def test_create_degraded_decision_context():
    """Verify degraded context factory creates fail-closed non-actionable structure."""
    context = create_degraded_decision_context(
        symbol="NVDA",
        horizon=TimeHorizon.SWING,
        user_role=UserRole.DAY_TRADER,
        market_data={
            "candles_count": 80,
            "last_close": 115.5,
            "provider": "yahoo_finance_direct",
            "as_of": "2026-09-22T06:00:00Z",
        },
    )

    assert context.symbol == "NVDA"
    assert context.horizon == TimeHorizon.SWING
    assert context.user_role == UserRole.DAY_TRADER
    assert context.is_degraded is True
    assert context.evidence_completeness == DataCompleteness.DEGRADED

    # Market evidence has fallback quality
    assert context.market_evidence.domain == EvidenceDomain.MARKET_DATA
    assert context.market_evidence.quality == EvidenceQualityState.FALLBACK
    assert context.market_evidence.payload["last_close"] == 115.5
    assert context.market_evidence.is_stale is False

    # Non-market domains are UNAVAILABLE and marked stale/unavailable
    assert context.fundamental_evidence.domain == EvidenceDomain.FUNDAMENTALS
    assert context.fundamental_evidence.quality == EvidenceQualityState.UNAVAILABLE
    assert context.fundamental_evidence.is_stale is True

    assert context.macro_evidence.domain == EvidenceDomain.MACRO
    assert context.macro_evidence.quality == EvidenceQualityState.UNAVAILABLE
    assert context.macro_evidence.is_stale is True

    assert context.liquidity_evidence.domain == EvidenceDomain.LIQUIDITY
    assert context.liquidity_evidence.quality == EvidenceQualityState.UNAVAILABLE
    assert context.liquidity_evidence.payload["liquidity_gate_passed"] is False

    # Test serialization
    as_dict = context.to_dict()
    assert as_dict["symbol"] == "NVDA"
    assert as_dict["isDegraded"] is True
    assert as_dict["evidenceCompleteness"] == "DEGRADED"
    assert as_dict["marketEvidence"]["payload"]["last_close"] == 115.5


def test_cross_language_schema_parity_with_fixture():
    """Verify Python models faithfully parse and match the canonical cross-language fixture."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "canonical_decision_fixture.json"
    )
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture_data = json.load(f)

    full = fixture_data["full_decision"]
    degraded = fixture_data["degraded_decision"]

    # Validate full decision structure
    assert full["authority"] == ARXDecisionAuthority.BACKEND_CANONICAL.value
    assert full["verdict"]["isActionable"] is True
    assert full["verdict"]["decisionState"] == DecisionState.ACTIONABLE_SETUP.value
    assert full["context"]["marketEvidence"]["quality"] == EvidenceQualityState.AUTHORITATIVE.value
    assert full["context"]["fundamentalEvidence"]["payload"]["pointInTimeStatus"] == PointInTimeStatus.POINT_IN_TIME.value
    assert full["context"]["macroEvidence"]["payload"]["tacticalEquityRegime"] == "BULL_TRENDING"
    assert full["context"]["macroEvidence"]["payload"]["structuralMacroRegime"] == "EXPANSION"

    # Validate degraded decision structure
    assert degraded["authority"] == ARXDecisionAuthority.DISPLAY_ONLY_MARKET_DATA.value
    assert degraded["verdict"]["isActionable"] is False
    assert degraded["verdict"]["decisionState"] == DecisionState.UNVERIFIED.value
    assert degraded["verdict"]["levels"]["entryMin"] is None
    assert degraded["verdict"]["levels"]["stopLoss"] is None
    assert degraded["context"]["isDegraded"] is True
    assert degraded["context"]["evidenceCompleteness"] == DataCompleteness.DEGRADED.value
