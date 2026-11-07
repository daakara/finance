"""
ARX-PREFLIGHT-001 Phase 3 Formal Acceptance Test Suite (Backend)

Verifies:
1. market_db.py save_daily_candles atomic replacement clears asset_ohlcv_provenance
   to prevent contradictory candle provenance from acquiring live-data authority.
2. DecisionHierarchyEngine enforces canSizeTrade=False and isActionable=False
   when R:R < 2.0 or when setup trigger is unconfirmed.
3. MarketDatabase get_daily_candles_with_evidence falls back cleanly to UNKNOWN
   provenance after legacy replacement rather than returning obsolete provenance.
"""

import os
import sys
import tempfile
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analyst_dashboard.data.market_db import MarketDatabase
from analyst_dashboard.data.market_evidence import (
    MarketProvenance,
    ObservationPrecision,
    StructuralQuality,
)
from analyst_dashboard.analyzers.decision_hierarchy import (
    DecisionHierarchyEngine,
    DecisionState,
)


def test_market_db_provenance_atomicity_and_isolation():
    """
    PREFLIGHT-001 Target 5 Acceptance:
    Verify that replacing candle data via save_daily_candles completely removes
    any stale provenance entries from asset_ohlcv_provenance.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = MarketDatabase(db_path=db_path)
        symbol = "AAPL"
        trade_date = "2026-11-06"

        # Step 1: Save verified candle with evidence
        prov = MarketProvenance(
            provider="ALPACA_IEX",
            ingestion_source="ALPACA_HISTORICAL_REST_V2",
            observed_at="2026-11-06T20:00:00Z",
            observed_date=trade_date,
            observation_precision=ObservationPrecision.TIMESTAMP,
            observation_source="EXCHANGE_OFFICIAL_FEED",
            ingested_at="2026-11-06T20:01:00Z",
            structural_quality=StructuralQuality.COMPLETE,
        )
        candles = [
            {
                "time": trade_date,
                "open": 220.0,
                "high": 225.0,
                "low": 218.0,
                "close": 224.0,
                "volume": 1000000,
            }
        ]
        db.save_daily_candles_with_evidence(symbol, candles, prov)

        # Verify initial provenance
        res_candles, res_ev = db.get_daily_candles_with_evidence(symbol, limit=5)
        assert len(res_candles) == 1
        assert res_candles[0]["close"] == 224.0
        assert res_ev.provenance.provider == "ALPACA_IEX"
        assert res_ev.provenance.observed_at == "2026-11-06T20:00:00Z"

        # Step 2: Replace candle using save_daily_candles (e.g. unverified legacy script or fallback)
        legacy_candles = [
            {
                "time": trade_date,
                "open": 221.0,
                "high": 226.0,
                "low": 219.0,
                "close": 225.0,
                "volume": 1200000,
            }
        ]
        db.save_daily_candles(symbol, legacy_candles)

        # Step 3: Verify that contradictory provenance was atomically deleted
        res_candles_after, res_ev_after = db.get_daily_candles_with_evidence(symbol, limit=5)
        assert len(res_candles_after) == 1
        assert res_candles_after[0]["close"] == 225.0
        # Must fall back cleanly to UNKNOWN rather than claiming ALPACA_IEX with obsolete timestamps
        assert res_ev_after.provenance.provider == "UNKNOWN"
        assert res_ev_after.provenance.observed_at is None
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_decision_hierarchy_rr_floor_and_sizing_denial():
    """
    PREFLIGHT-001 Target 3 Acceptance:
    Verify that DecisionHierarchyEngine enforces R:R >= 2.0 and sets canSizeTrade=False
    when R:R < 2.0 or when setup trigger is unconfirmed.
    """
    # Case A: Sub-threshold R:R (1.85:1 < 2.0 institutional floor)
    dec_sub_rr = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST",
        current_price=100.0,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase="Stage 2",
        is_in_buy_zone=True,
        is_confirmed=True,
        risk_reward_ratio=1.85,  # Sub-threshold
    )
    assert dec_sub_rr["state"] != DecisionState.ACTIONABLE_SETUP.value
    assert dec_sub_rr["isActionable"] is False
    assert dec_sub_rr["canSizeTrade"] is False

    # Case B: Unconfirmed trigger with valid R:R (2.5:1)
    dec_unconfirmed = DecisionHierarchyEngine.resolve_decision_state(
        symbol="NVO",
        current_price=100.0,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=80.0,
        stage_phase="Stage 2",
        is_in_buy_zone=True,
        is_confirmed=False,  # Unconfirmed
        risk_reward_ratio=2.5,
    )
    assert dec_unconfirmed["state"] == DecisionState.VALID_SETUP.value
    assert dec_unconfirmed["isActionable"] is False
    assert dec_unconfirmed["canSizeTrade"] is False

    # Case C: Fully confirmed actionable setup with R:R >= 2.0
    dec_actionable = DecisionHierarchyEngine.resolve_decision_state(
        symbol="AAPL",
        current_price=220.0,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase="Stage 2",
        is_in_buy_zone=True,
        is_confirmed=True,
        risk_reward_ratio=2.5,
    )
    assert dec_actionable["state"] == DecisionState.ACTIONABLE_SETUP.value
    assert dec_actionable["isActionable"] is True
    assert dec_actionable["canSizeTrade"] is True
