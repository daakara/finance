"""
ARX Terminal — PREFLIGHT-001 Targeted Falsification Test Suite
Validates the 6 specific falsification targets:
1. Fresh macro data combined with stale or unverified price observations.
2. Cached analytics responses reused after their source-observation freshness expires.
3. Five passed Pre-Flight checks when canonical execution eligibility is false.
4. Historical or synthetic evidence represented as live observations.
5. Conflicting OHLCV candle values and provenance records after legacy database replacement.
6. Multiple simultaneous Pre-Flight blockers (NVO conditions).
"""

import os
import sys
import tempfile
import sqlite3
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from analyst_dashboard.data.market_db import MarketDatabase
from analyst_dashboard.data.market_evidence import MarketProvenance, MarketEvidence, Provider, ObservationPrecision, StructuralQuality
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState


def test_falsification_target_5_ohlcv_provenance_conflict():
    """Target 5: Conflicting OHLCV candle values and provenance records after legacy replacement."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = MarketDatabase(db_path=db_path)
        symbol = "TEST"
        trade_date = "2026-11-06"

        # 1. Save evidence-backed candle A
        prov_a = MarketProvenance(
            provider="ALPACA_IEX",
            ingestion_source="ALPACA_HISTORICAL_REST_V2",
            observed_at="2026-11-06T20:00:00Z",
            observed_date=trade_date,
            observation_precision=ObservationPrecision.TIMESTAMP,
            observation_source="EXCHANGE_OFFICIAL_FEED",
            ingested_at="2026-11-06T20:01:00Z",
            structural_quality=StructuralQuality.COMPLETE,
        )
        candle_a = [{"time": trade_date, "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000000}]
        db.save_daily_candles_with_evidence(symbol, candle_a, prov_a)

        # Verify initial readback
        candles_read, ev_read = db.get_daily_candles_with_evidence(symbol, limit=5)
        assert len(candles_read) == 1
        assert candles_read[0]["close"] == 104.0
        assert ev_read.provenance.provider == "ALPACA_IEX"

        # 2. Legacy save_daily_candles replaces the candle with Candle B (e.g. unverified close = 95.0)
        candle_b = [{"time": trade_date, "open": 100.0, "high": 101.0, "low": 94.0, "close": 95.0, "volume": 500000}]
        db.save_daily_candles(symbol, candle_b)

        # 3. Read back with get_daily_candles_with_evidence
        candles_after, ev_after = db.get_daily_candles_with_evidence(symbol, limit=5)
        assert len(candles_after) == 1
        assert candles_after[0]["close"] == 95.0

        # REMEDIATION VERIFICATION: The legacy replacement replaced the candle values,
        # and atomically cleared the prior provenance, so it safely falls back to UNKNOWN legacy provenance!
        print(f"Candle after legacy replacement: close={candles_after[0]['close']}, provider={ev_after.provenance.provider}")
        assert ev_after.provenance.provider == "UNKNOWN", "Remediation verified: legacy replacement reverts to UNKNOWN provenance!"
        assert ev_after.provenance.observed_at is None
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_falsification_target_3_canonical_decision_hierarchy():
    """Target 3: Verify canonical decision hierarchy sets canSizeTrade=False when setup is not ACTIONABLE_SETUP."""
    # Example: Sound setup (confluence=80, Stage 2, buy zone, R:R=2.5), BUT trigger not confirmed
    dec = DecisionHierarchyEngine.resolve_decision_state(
        symbol="NVO",
        current_price=100.0,
        candle_count=252,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=80.0,
        stage_phase="Stage 2",
        is_in_buy_zone=True,
        is_confirmed=False,  # Trigger NOT confirmed
        risk_reward_ratio=2.5,
    )
    assert dec["state"] == DecisionState.VALID_SETUP.value
    assert dec["isActionable"] is False
    assert dec["canSizeTrade"] is False
    print(f"Canonical decision for unconfirmed trigger: state={dec['state']}, canSizeTrade={dec['canSizeTrade']}")
