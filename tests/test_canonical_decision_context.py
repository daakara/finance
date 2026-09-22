"""Unit tests for ARX Canonical Decision Context and Evidence Contracts (Phase 0/1)."""

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
    TimeHorizon,
    UserRole,
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
    assert as_dict["is_degraded"] is True
    assert as_dict["evidence_completeness"] == "DEGRADED"
    assert as_dict["market_evidence"]["payload"]["last_close"] == 115.5


def test_full_decision_contract_structure():
    """Verify full ARXDecision structure with verdict and model trace."""
    context = create_degraded_decision_context("AAPL")
    verdict = DecisionVerdict(
        symbol="AAPL",
        horizon=TimeHorizon.SWING,
        user_role=UserRole.LONG_TERM,
        is_actionable=False,
        can_size_trade=False,
        decision_state=DecisionState.UNVERIFIED,
        execution_status="UNVERIFIED_ASSET",
        verdict_label="Degraded Display Only",
        disqualification_reason="Direct client tape fallback",
        confluence_score=0.0,
        observation_date="2026-09-22",
        levels=ExecutionLevels(
            entry_min=None,
            entry_max=None,
            stop_loss=None,
            stop_loss_pct=0.0,
            target_1=None,
            target_1_pct=0.0,
            target_2=None,
            target_2_pct=0.0,
            risk_reward_ratio=None,
        ),
        data_completeness=DataCompleteness.DEGRADED,
    )

    decision = ARXDecision(
        context=context,
        verdict=verdict,
        confluence_score=0.0,
        model_trace={
            "model_name": "client_fallback",
            "version": "phase_0_1",
            "generated_at": "2026-09-22T06:00:00Z",
            "passed_gates": [],
            "failed_gates": ["BACKEND_DECISION_ENGINE_UNREACHABLE"],
        },
        authority=ARXDecisionAuthority.DISPLAY_ONLY_MARKET_DATA,
    )

    dec_dict = decision.to_dict()
    assert dec_dict["authority"] == "DISPLAY_ONLY_MARKET_DATA"
    assert dec_dict["verdict"]["is_actionable"] is False
    assert dec_dict["verdict"]["execution_status"] == "UNVERIFIED_ASSET"
    assert dec_dict["verdict"]["levels"]["stop_loss"] is None
    assert dec_dict["verdict"]["levels"]["entry_min"] is None
