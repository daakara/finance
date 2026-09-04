"""ARX Golden Regression Universe Test Suite (Phase 20E).

Tests the 6 canonical archetype assets to ensure future improvements
never weaken the decision quality or epistemic model:
  1. NVDA (Strong Compounder)
  2. FIX (Weak Distribution)
  3. CPRX (Catalyst Bio-Pharma)
  4. SPY (Broad-Market ETF)
  5. GLX (Insufficient History)
  6. TEST_AAA (Unknown / Uncataloged)
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.data.market_db import MarketDatabaseEngine, DB_PATH
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from tests.fixtures.golden_universe import GOLDEN_ASSETS

client = TestClient(app)
db = MarketDatabaseEngine(db_path=DB_PATH)


def test_golden_nvda_strong_compounder():
    """NVDA: Strong verified compounder in Stage 2 markup with valid trade levels and confluence >= 75."""
    fixture = GOLDEN_ASSETS["NVDA"]

    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="NVDA",
        current_price=128.50,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=82.5,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=3.1,
        is_cataloged=True,
    )

    assert state["state"] in fixture["expected_state"]
    assert state["isActionable"] == fixture["is_actionable_expected"]
    assert state["canSizeTrade"] == fixture["can_size_trade_expected"]
    assert state["disqualificationReason"] is None


def test_golden_fix_weak_distribution():
    """FIX: Weak / Stage 4 distribution setup must never emit actionable buy recommendation."""
    fixture = GOLDEN_ASSETS["FIX"]

    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="FIX",
        current_price=35.20,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=42.0,
        stage_phase=4,  # Stage 4 Distribution
        is_in_buy_zone=False,
        risk_reward_ratio=1.1,
        is_cataloged=True,
    )

    assert state["state"] in fixture["expected_state"]
    assert state["isActionable"] == fixture["is_actionable_expected"]
    assert state["canSizeTrade"] == fixture["can_size_trade_expected"]
    assert "Stage 4" in state["disqualificationReason"]


def test_golden_cprx_catalyst_driven():
    """CPRX: Bio-pharma catalyst asset resolves valid setup with registered clinical pipeline."""
    fixture = GOLDEN_ASSETS["CPRX"]

    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="CPRX",
        current_price=18.40,
        candle_count=180,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=78.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.6,
        is_cataloged=True,
    )

    assert state["state"] in fixture["expected_state"]
    assert state["isActionable"] is True


def test_golden_spy_broad_market_etf():
    """SPY: Broad-market ETF verified tape, individual insider checks bypassed, macro active."""
    fixture = GOLDEN_ASSETS["SPY"]

    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="SPY",
        current_price=540.20,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,  # Macro ETFs mapped as having valid fund background
        confluence_score=72.0,
        stage_phase=2,
        is_in_buy_zone=False,  # Consolidating near highs
        risk_reward_ratio=1.8,
        is_cataloged=True,
    )

    assert state["state"] in fixture["expected_state"]
    # Outside buy zone -> valid setup but not actionable
    assert state["state"] == "VALID_SETUP"
    assert state["isActionable"] is False


def test_golden_glx_insufficient_history():
    """GLX: Unseasoned ticker (< 50 sessions) must resolve INSUFFICIENT_DATA and suppress execution."""
    fixture = GOLDEN_ASSETS["GLX"]

    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="GLX",
        current_price=17.50,
        candle_count=47,  # 47 of 50 sessions
        freshness_status="LIVE",
        has_fundamentals=False,
        confluence_score=46.0,
        stage_phase=None,
        is_in_buy_zone=False,
        risk_reward_ratio=None,
        is_cataloged=True,
    )

    assert state["state"] in fixture["expected_state"]
    assert state["state"] == DecisionState.INSUFFICIENT_DATA.value
    assert state["isActionable"] == fixture["is_actionable_expected"]
    assert state["canSizeTrade"] == fixture["can_size_trade_expected"]
    assert "Insufficient History" in state["label"]


def test_golden_test_aaa_unknown_unverified():
    """TEST_AAA: Completely unknown ticker fails closed with 404 or UNVERIFIED state."""
    fixture = GOLDEN_ASSETS["TEST_AAA"]

    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST_AAA",
        current_price=0.0,
        candle_count=0,
        freshness_status="UNAVAILABLE",
        has_fundamentals=False,
        confluence_score=0.0,
        is_cataloged=False,
    )

    assert state["state"] in fixture["expected_state"]
    assert state["state"] == DecisionState.UNVERIFIED.value
    assert state["isActionable"] is False
    assert state["canSizeTrade"] is False
