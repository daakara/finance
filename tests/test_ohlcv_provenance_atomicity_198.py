"""
ARX Terminal — TD-02 OHLCV Provenance Atomicity Acceptance Suite
Tracker: Issue #198 (TD-02 / ARX-EVIDENCE-INTEGRITY-032)

Epistemic & Atomicity Invariants:
1. Evidence-Backed Candle Ingestion: Candles written with MarketEvidence retain exact provider,
   observed timestamps, observation precision, and structural quality.
2. Contradiction Prevention on Legacy Overwrite: When legacy save_daily_candles updates or replaces
   a daily candle, prior provenance records for that symbol and trade_date are atomically deleted
   within the same transaction.
3. Safe Legacy Provenance Fallback: When reading back candles where provenance was cleared or unmigrated,
   get_daily_candles_with_evidence derives factual legacy provenance (provider="UNKNOWN",
   observed_at=None) without crashing or fabricating observation times.
4. Clean Re-enrichment: When save_daily_candles_with_evidence subsequently re-writes a candle,
   the authoritative provenance is atomically re-established.
5. Scoped Range Replacement Integrity: Replacing a sub-range (D2..D4) atomically purges old records
   within [scope_start, scope_end] while strictly preserving surrounding dates (D1, D5).
"""

import os
import sys
import tempfile
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from analyst_dashboard.data.market_db import MarketDatabase
from analyst_dashboard.data.market_evidence import (
    MarketProvenance,
    MarketEvidence,
    Provider,
    ObservationPrecision,
    ObservationSource,
    StructuralQuality,
)


def test_ohlcv_provenance_atomicity_and_legacy_fallback():
    """Verify candle replacement A -> B atomically clears provenance and falls back to UNKNOWN."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = MarketDatabase(db_path=db_path)
        symbol = "TEST_ATOM"
        trade_date = "2026-11-10"

        # 1. Write evidence-backed candle A
        prov_a = MarketProvenance(
            provider="ALPACA_IEX",
            ingestion_source="ALPACA_HISTORICAL_REST_V2",
            observed_at="2026-11-10T21:00:00Z",
            observed_date=trade_date,
            observation_precision=ObservationPrecision.TIMESTAMP,
            observation_source=ObservationSource.EXCHANGE_TIMESTAMP,
            ingested_at="2026-11-10T21:05:00Z",
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        candle_a = [{
            "time": trade_date,
            "open": 150.0,
            "high": 155.0,
            "low": 149.0,
            "close": 154.0,
            "volume": 2500000,
        }]
        db.save_daily_candles_with_evidence(symbol, candle_a, prov_a)

        # Read back and verify A
        candles_a, ev_a = db.get_daily_candles_with_evidence(symbol, limit=10)
        assert len(candles_a) == 1
        assert candles_a[0]["close"] == 154.0
        assert ev_a.provenance.provider == "ALPACA_IEX"
        assert ev_a.provenance.observed_at == "2026-11-10T21:00:00Z"
        assert ev_a.provenance.fallback_status is False

        # 2. Legacy save_daily_candles replaces Candle A with Candle B (no evidence supplied)
        candle_b = [{
            "time": trade_date,
            "open": 150.0,
            "high": 152.0,
            "low": 145.0,
            "close": 146.0,
            "volume": 1200000,
        }]
        db.save_daily_candles(symbol, candle_b)

        # Read back and verify B + fallback to UNKNOWN
        candles_b, ev_b = db.get_daily_candles_with_evidence(symbol, limit=10)
        assert len(candles_b) == 1
        assert candles_b[0]["close"] == 146.0
        assert ev_b.provenance.provider == "UNKNOWN", "Legacy overwrite must atomically clear provenance to UNKNOWN"
        assert ev_b.provenance.observed_at is None, "Legacy overwrite must clear misleading observation timestamp"
        assert ev_b.provenance.observation_source == ObservationSource.DERIVED_FROM_TRADE_DATE

        # 3. Re-enrich with authoritative Candle C
        prov_c = MarketProvenance(
            provider="POLYGON",
            ingestion_source="POLYGON_STOCKS_V2",
            observed_at="2026-11-10T22:00:00Z",
            observed_date=trade_date,
            observation_precision=ObservationPrecision.TIMESTAMP,
            observation_source=ObservationSource.EXCHANGE_TIMESTAMP,
            ingested_at="2026-11-10T22:02:00Z",
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        candle_c = [{
            "time": trade_date,
            "open": 150.0,
            "high": 153.0,
            "low": 147.0,
            "close": 148.5,
            "volume": 1800000,
        }]
        db.save_daily_candles_with_evidence(symbol, candle_c, prov_c)

        # Read back and verify C
        candles_c, ev_c = db.get_daily_candles_with_evidence(symbol, limit=10)
        assert len(candles_c) == 1
        assert candles_c[0]["close"] == 148.5
        assert ev_c.provenance.provider == "POLYGON"
        assert ev_c.provenance.observed_at == "2026-11-10T22:00:00Z"
        assert ev_c.provenance.fallback_status is False

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_scoped_range_replacement_isolation():
    """Verify that replacing a sub-range (D2..D4) preserves surrounding dates (D1, D5)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = MarketDatabase(db_path=db_path)
        symbol = "RANGE_TEST"

        dates = ["2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"]
        initial_candles = [
            {"time": d, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0 + i, "volume": 100000}
            for i, d in enumerate(dates)
        ]
        prov_initial = MarketProvenance(
            provider="BATCH_PROVIDER",
            ingestion_source="INITIAL_FEED",
            observed_at="2026-11-05T20:00:00Z",
            observed_date="2026-11-05",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-11-05T20:01:00Z",
            structural_quality=StructuralQuality.COMPLETE,
        )
        db.save_daily_candles_with_evidence(symbol, initial_candles, prov_initial)

        # Replace only D2..D4 with new provider
        replacement_dates = ["2026-11-02", "2026-11-03", "2026-11-04"]
        replacement_candles = [
            {"time": d, "open": 200.0, "high": 205.0, "low": 199.0, "close": 202.0, "volume": 500000}
            for d in replacement_dates
        ]
        prov_replacement = MarketProvenance(
            provider="REPLACEMENT_PROVIDER",
            ingestion_source="HIGH_RES_FEED",
            observed_at="2026-11-06T10:00:00Z",
            observed_date="2026-11-04",
            observation_precision=ObservationPrecision.TIMESTAMP,
            observation_source=ObservationSource.EXCHANGE_TIMESTAMP,
            ingested_at="2026-11-06T10:01:00Z",
            structural_quality=StructuralQuality.COMPLETE,
        )
        db.save_daily_candles_with_evidence(
            symbol,
            replacement_candles,
            prov_replacement,
            replace_existing_scope=True,
            scope_start_date="2026-11-02",
            scope_end_date="2026-11-04",
        )

        # Query all candles
        all_candles, latest_ev = db.get_daily_candles_with_evidence(symbol, limit=10)
        assert len(all_candles) == 5

        # Query direct SQLite provenance rows to verify per-date provider isolation
        conn = db._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT trade_date, provider FROM asset_ohlcv_provenance WHERE symbol = ? ORDER BY trade_date",
                (symbol,)
            )
            rows = dict(cursor.fetchall())
            assert rows["2026-11-01"] == "BATCH_PROVIDER", "D1 must retain original provider"
            assert rows["2026-11-02"] == "REPLACEMENT_PROVIDER", "D2 must have replacement provider"
            assert rows["2026-11-03"] == "REPLACEMENT_PROVIDER", "D3 must have replacement provider"
            assert rows["2026-11-04"] == "REPLACEMENT_PROVIDER", "D4 must have replacement provider"
            assert rows["2026-11-05"] == "BATCH_PROVIDER", "D5 must retain original provider"
        finally:
            conn.close()

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
