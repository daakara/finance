"""
Tests for Liquidity Hazard & Canonical Actionability Precedence (GAP-WS41-003).

Directive ID: ARX_S4_WS41_LIQUIDITY_ACTIONABILITY_IMPLEMENTATION_005
Validates:
1. INV-LIQ-001: Severe liquidity hazard prohibits ACTIONABLE_SETUP, sets isActionable=False, canSizeTrade=False.
2. INV-LIQ-002: STOPPED_OUT structural precedence over liquidity hazard; both traceable in reason.
3. INV-LIQ-003: Missing liquidity evidence (UNAVAILABLE / UNKNOWN) defaults to non-actionable unverified status.
4. INV-LIQ-004: Position sizer zeroes executable shares for blocked setups while retaining hypothetical shares.
5. INV-LIQ-005: Preserves advisory thresholds without unratified 2% caps or $250k floors.
6. INV-LIQ-006: DecisionTraceEngine and ConfluenceEngine coherence across backend surfaces.
"""

import math
import pytest
import pandas as pd

from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine


def test_case_01_high_technical_quality_severe_liquidity_hazard():
    """Case 1: High technical quality setup with severe liquidity hazard must be non-actionable."""
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="ILLIQ",
        current_price=50.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=88.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.6,
        is_cataloged=True,
        is_confirmed=True,
        execution_status="IN_BUY_ZONE",
        execution_hazard=True,
        adv_20d=45000.0,
    )

    assert dec["state"] == DecisionState.VALID_SETUP.value
    assert dec["isActionable"] is False
    assert dec["canSizeTrade"] is False
    assert dec["label"] == "Valid Setup — Liquidity Restricted"
    assert "severe liquidity risk detected" in dec["disqualificationReason"].lower()
    assert "SIZE_TRADE" not in dec["allowedActions"]


def test_case_02_high_confluence_low_authoritative_liquidity():
    """Case 2: High confluence with liquidity defense reporting severe hazard via dict."""
    liq_defense = {
        "execution_hazard": True,
        "evidence_status": "AUTHENTIC",
        "adv_20d_usd": 85000.0,
        "reason": "ADV below safety threshold",
    }
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TINY",
        current_price=25.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=92.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=3.1,
        is_cataloged=True,
        is_confirmed=True,
        execution_status="IN_BUY_ZONE",
        liquidity_defense=liq_defense,
    )

    assert dec["state"] == DecisionState.VALID_SETUP.value
    assert dec["isActionable"] is False
    assert dec["label"] == "Valid Setup — Liquidity Restricted"
    assert "$85,000/day" in dec["disqualificationReason"]


def test_case_03_valid_execution_levels_confirmed_liquidity_hazard():
    """Case 3: Valid execution levels in DecisionTraceEngine with confirmed hazard."""
    optimal_execution = {
        "setup_pattern": "VCP Breakout",
        "stage_phase": 2,
        "entry_pivot": 100.0,
        "stop_loss": 95.0,
        "take_profit_1": 112.0,
        "risk_reward_ratio": 2.4,
        "execution_status": "IN_BUY_ZONE",
        "execution_hazard": True,
        "liquidity_defense": {
            "execution_hazard": True,
            "evidence_status": "AUTHENTIC",
            "adv_20d_usd": 120000.0,
        },
    }

    trace = DecisionTraceEngine.build_decision_trace(
        symbol="MEDHAZ",
        current_price=100.0,
        candles=[{"close": 100.0}] * 70,
        freshness={"status": "LIVE"},
        technicals={"rsi_14": 55.0},
        confluence={"confluenceScore": 82.0},
        factor_scores={"qualityScore": 80.0},
        optimal_execution=optimal_execution,
    )

    assert trace["isActionable"] is False
    assert trace["canSizeTrade"] is False
    assert trace["decisionStateLabel"] == "Valid Setup — Liquidity Restricted"


def test_case_04_stopped_out_healthy_liquidity():
    """Case 4: STOPPED_OUT with healthy liquidity displays stopped out invalidation."""
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="STOPPED",
        current_price=90.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=80.0,
        stage_phase=2,
        is_in_buy_zone=False,
        risk_reward_ratio=2.2,
        is_cataloged=True,
        is_confirmed=False,
        execution_status="STOPPED_OUT",
        execution_hazard=False,
        adv_20d=5_000_000.0,
    )

    assert dec["state"] == DecisionState.VALID_SETUP.value
    assert dec["isActionable"] is False
    assert dec["label"] == "Setup Invalidated — Stopped Out"
    assert "breached stop-loss threshold" in dec["disqualificationReason"]
    assert "liquidity" not in dec["disqualificationReason"].lower()


def test_case_05_stopped_out_severe_liquidity_hazard():
    """Case 5: STOPPED_OUT takes structural precedence over liquidity hazard; both traceable in reason."""
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="STOPHAZ",
        current_price=90.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=80.0,
        stage_phase=2,
        is_in_buy_zone=False,
        risk_reward_ratio=2.2,
        is_cataloged=True,
        is_confirmed=False,
        execution_status="STOPPED_OUT",
        execution_hazard=True,
        adv_20d=50_000.0,
    )

    assert dec["state"] == DecisionState.VALID_SETUP.value
    assert dec["isActionable"] is False
    assert dec["label"] == "Setup Invalidated — Stopped Out"
    reason = dec["disqualificationReason"]
    assert "breached stop-loss threshold" in reason
    assert "severe execution hazard" in reason


def test_case_06_missing_adv_liquidity_unverified():
    """Case 6: Missing ADV telemetry must yield unverified liquidity and block actionability."""
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="NOADV",
        current_price=50.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_cataloged=True,
        is_confirmed=True,
        execution_status="IN_BUY_ZONE",
        execution_hazard="UNKNOWN",
        liquidity_status="UNAVAILABLE",
    )

    assert dec["state"] == DecisionState.VALID_SETUP.value
    assert dec["isActionable"] is False
    assert dec["canSizeTrade"] is False
    assert dec["label"] == "Valid Setup — Liquidity Unverified"
    assert "insufficient historical volume" in dec["disqualificationReason"].lower()


def test_case_07_stale_volume_evidence_stale_data():
    """Case 7: Stale historical data triggers Precedence 2 STALE_DATA regardless of liquidity."""
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="STALE",
        current_price=50.0,
        candle_count=100,
        freshness_status="STALE_HISTORICAL",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_cataloged=True,
        is_confirmed=True,
        execution_status="IN_BUY_ZONE",
        execution_hazard=False,
    )

    assert dec["state"] == DecisionState.STALE_DATA.value
    assert dec["isActionable"] is False
    assert "stale" in dec["disqualificationReason"].lower()


def test_case_08_zero_or_malformed_price_sizing_protection():
    """Case 8: Zero, negative, or NaN price returns 0 shares and isExecutable=False."""
    # Zero entry price
    res_zero = ConfluenceEngine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,
        entry_price=0.0,
        stop_loss=95.0,
        take_profit_1=105.0,
    )
    assert res_zero["shares"] == 0
    assert res_zero["executableShares"] == 0
    assert res_zero["isExecutable"] is False

    # NaN entry price
    res_nan = ConfluenceEngine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,
        entry_price=float("nan"),
        stop_loss=95.0,
        take_profit_1=105.0,
    )
    assert res_nan["shares"] == 0
    assert res_nan["executableShares"] == 0
    assert res_nan["isExecutable"] is False

    # Stop loss >= Entry price
    res_inverted = ConfluenceEngine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit_1=115.0,
    )
    assert res_inverted["shares"] == 0
    assert res_inverted["executableShares"] == 0
    assert res_inverted["isExecutable"] is False


def test_case_09_healthy_liquidity_valid_setup_is_actionable():
    """Case 9: Healthy liquidity + valid setup produces ACTIONABLE_SETUP."""
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="HEALTHY",
        current_price=100.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_cataloged=True,
        is_confirmed=True,
        execution_status="IN_BUY_ZONE",
        execution_hazard=False,
        adv_20d=25_000_000.0,
    )

    assert dec["state"] == DecisionState.ACTIONABLE_SETUP.value
    assert dec["isActionable"] is True
    assert dec["canSizeTrade"] is True
    assert "Actionable Setup" in dec["label"]


def test_case_10_cross_surface_backend_precedence():
    """Case 10: Backend canonical decision state overrides any divergent UI or execution claims."""
    # Setup passes technicals, but backend confirms severe liquidity hazard
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="OVERRIDE",
        current_price=100.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_cataloged=True,
        is_confirmed=True,
        execution_status="IN_BUY_ZONE",
        execution_hazard=True,
        adv_20d=80_000.0,
    )

    # Regardless of is_confirmed=True and IN_BUY_ZONE, actionability is denied
    assert dec["isActionable"] is False
    assert dec["canSizeTrade"] is False
    assert dec["state"] != DecisionState.ACTIONABLE_SETUP.value


def test_case_11_technical_only_radar_discovery():
    """Case 11: Technical-only Radar discovery without execution confirmation is not actionable."""
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="RADAR1",
        current_price=100.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=82.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_cataloged=True,
        is_confirmed=False,  # Radar structural discovery only
        execution_status="IN_BUY_ZONE",
        execution_hazard=False,
        adv_20d=5_000_000.0,
    )

    assert dec["state"] == DecisionState.VALID_SETUP.value
    assert dec["isActionable"] is False
    assert dec["label"] == "Valid Setup — Awaiting Trigger"


def test_case_12_blocked_actionability_position_sizing_consistency():
    """Case 12: Blocked actionability results in shares=0, executableShares=0, hypotheticalShares>0."""
    res = ConfluenceEngine.calculate_position_size(
        account_equity=100_000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=115.0,
        is_actionable=False,
        can_size_trade=False,
        disqualification_reason="Liquidity hazard: low volume",
        execution_hazard=True,
        adv_20d=50_000.0,
    )

    # Executable shares must be strictly 0
    assert res["shares"] == 0
    assert res["executableShares"] == 0
    assert res["isExecutable"] is False
    assert res["positionValue"] == 0.0
    assert res["actualDollarRisk"] == 0.0

    # Hypothetical calculation preserved for analysis
    assert res["hypotheticalShares"] == 200  # $1,000 / $5 = 200 shares
    assert res["hypotheticalPositionValue"] == 20000.0
    assert res["hypotheticalActualDollarRisk"] == 1000.0
    assert res["isHypotheticalOnly"] is True
