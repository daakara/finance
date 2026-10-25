"""
tests/test_strategy_governance_adversarial.py

Sprint 2 Track C: Strategy Governance & Adversarial Validation Tests (Phase 5).
Verifies:
1. VCP admission without G-VCP-012 certification -> FAIL CLOSED (DENIED).
2. VCP admission with invalid or missing certification -> FAIL CLOSED.
3. Non-VCP admission with valid strategy-specific certification -> ADMITTED to strategy evaluation only (State 6).
4. Non-VCP admission without certification -> FAIL CLOSED.
5. Adversarial input with invalid strategy_type -> FAIL CLOSED.
6. Adversarial input with contradictory strategy_type and strategy_id -> FAIL CLOSED.
7. Candidate published to universe -> ZERO scanner enrollment (State 5 firewall).
8. Candidate admitted to strategy evaluation without G-S2-ACT -> ZERO scanner enrollment.
9. Candidate admitted to strategy evaluation with valid G-S2-ACT -> scanner enrollment permitted ONLY for authorized strategy.
"""

import pytest

from analyst_dashboard.universe.isolated_publication import IsolatedPublicationEngine
from analyst_dashboard.universe.publication_state_machine import (
    PublicationState,
    PublicationStateMachine,
    StrategyAdmissionEvidence,
    ScannerDispatchEvidence,
    PolicyViolationError,
)


def test_01_vcp_without_g_vcp_012_fails_closed():
    """VCP admission without G-VCP-012 certification must fail closed (DENIED)."""
    ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        strategy_type="VCP",
        g_vcp_012_certified=False,  # Absent G-VCP-012
        min_price_floor=10.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="VCP_CHAIR",
    )
    with pytest.raises(PolicyViolationError, match="G-VCP-012 certification absent.*VCP strategy"):
        ev.validate()

    with pytest.raises(PolicyViolationError):
        PublicationStateMachine.transition(
            PublicationState.PUBLISHED_UNIVERSE,
            PublicationState.STRATEGY_ADMITTED,
            ev,
        )


def test_02_vcp_with_missing_certification_fails_closed():
    """VCP admission with None/missing certification fails closed."""
    ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VCP_BREAKOUT",
        g_vcp_012_certified=None,  # Missing
        min_price_floor=15.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="VCP_CHAIR",
    )
    with pytest.raises(PolicyViolationError, match="G-VCP-012 certification absent"):
        ev.validate()


def test_03_non_vcp_with_valid_strategy_certification_admitted():
    """Non-VCP strategy with valid strategy-specific certification is ADMITTED to State 6."""
    ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="SMART_MONEY",
        strategy_type="SMART_MONEY",
        strategy_certified=True,
        min_price_floor=10.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="SMART_MONEY_LEAD",
    )
    ev.validate()
    assert ev.effective_governance_authority == "G-STRAT-SMART_MONEY"

    next_state = PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        ev,
    )
    assert next_state == PublicationState.STRATEGY_ADMITTED


def test_04_non_vcp_without_certification_fails_closed():
    """Non-VCP strategy without strategy-specific certification fails closed."""
    ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VALUE_GARP",
        strategy_type="VALUE_GARP",
        strategy_certified=False,  # Uncertified
        min_price_floor=10.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="VALUE_LEAD",
    )
    with pytest.raises(PolicyViolationError, match="Strategy certification absent for non-VCP strategy"):
        ev.validate()


def test_05_adversarial_invalid_strategy_type_fails_closed():
    """Adversarial input providing invalid or unknown strategy_type fails closed."""
    ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="SMART_MONEY",
        strategy_type="ARBITRARY_BYPASS_ATTACK",
        strategy_certified=True,
        min_price_floor=10.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="ATTACKER",
    )
    with pytest.raises(PolicyViolationError, match="Invalid strategy_type 'ARBITRARY_BYPASS_ATTACK'"):
        ev.validate()


def test_06_adversarial_contradictory_strategy_id_and_type_fails_closed():
    """
    Adversarial input attempting to disguise a VCP strategy as non-VCP (to bypass G-VCP-012)
    or non-VCP as VCP fails closed due to contradictory parameters.
    """
    # 1. Strategy ID is MINERVINI_VCP but attacker asserts strategy_type="SMART_MONEY"
    ev1 = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        strategy_type="SMART_MONEY",
        strategy_certified=True,
        min_price_floor=10.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="ATTACKER",
    )
    with pytest.raises(PolicyViolationError, match="Contradictory strategy specification"):
        ev1.validate()

    # 2. Strategy ID is SMART_MONEY but strategy_type="VCP"
    ev2 = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="SMART_MONEY",
        strategy_type="VCP",
        g_vcp_012_certified=True,
        min_price_floor=10.0,
        min_history_bars=260,
        admission_approved=True,
        approved_by="ATTACKER",
    )
    with pytest.raises(PolicyViolationError, match="Contradictory strategy specification"):
        ev2.validate()


def test_07_published_candidate_has_zero_scanner_enrollment():
    """State 5 firewall: published securities have ZERO active scanner enrollment."""
    published_symbols = ["AAPL", "ABBV", "GOOG", "GOOGL"]
    active_scanner_symbols = set()

    # Verification passes when no symbols are enrolled in scanners
    IsolatedPublicationEngine.assert_no_scanner_enrollment(published_symbols, active_scanner_symbols)

    # If any published symbol is enrolled in active scanner without State 7 qualifying evidence, fail closed
    leaked_scanner_symbols = {"AAPL"}
    with pytest.raises(PolicyViolationError, match="Actionability leakage detected: Symbols \\['AAPL'\\]"):
        IsolatedPublicationEngine.assert_no_scanner_enrollment(published_symbols, leaked_scanner_symbols)


def test_08_strategy_admitted_candidate_without_g_s2_act_fails_scanner_dispatch():
    """Candidate admitted to strategy evaluation (State 6) without G-S2-ACT fails scanner enrollment."""
    ev = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="SMART_MONEY_SCANNER",
        g_s2_act_authorized=False,  # Missing G-S2-ACT
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    with pytest.raises(PolicyViolationError, match="G-S2-ACT absent for AAPL"):
        ev.validate()

    with pytest.raises(PolicyViolationError):
        PublicationStateMachine.transition(
            PublicationState.STRATEGY_ADMITTED,
            PublicationState.ACTIVELY_SCANNED,
            ev,
        )


def test_09_scanner_enrollment_permitted_only_for_authorized_strategy():
    """
    With valid G-S2-ACT, scanner enrollment is permitted ONLY for the authorized strategy.
    Mismatched scanner dispatch fails closed.
    """
    # 1. Valid dispatch for authorized strategy SMART_MONEY
    valid_ev = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="SMART_MONEY_SCANNER",
        authorized_strategy_id="SMART_MONEY",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    valid_ev.validate()
    state = PublicationStateMachine.transition(
        PublicationState.STRATEGY_ADMITTED,
        PublicationState.ACTIVELY_SCANNED,
        valid_ev,
    )
    assert state == PublicationState.ACTIVELY_SCANNED

    # 2. Attempted dispatch to MOMENTUM scanner when only SMART_MONEY was authorized fails closed
    mismatched_ev = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="MOMENTUM_SCANNER",
        authorized_strategy_id="SMART_MONEY",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    with pytest.raises(PolicyViolationError, match="Scanner enrollment unauthorized: scanner 'MOMENTUM_SCANNER' does not match authorized strategy 'SMART_MONEY'"):
        mismatched_ev.validate()
