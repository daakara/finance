"""F_13A Comprehensive Unit Test Suite.

Verifies:
1. Direct provider provenance
2. Fallback provider provenance
3. Legacy cache without sidecar row
4. New cache with sidecar row
5. Observation TIMESTAMP precision
6. Observation DATE precision
7. Unknown observation
8. Unknown provider
9. Adjustment UNKNOWN
10. Structural COMPLETE, PARTIAL, CORRUPT assessment
11. Structural classification does not alter data / filter rows
12. Atomic write rollback (failure on sidecar rolls back candles)
13. Legacy API parity (signatures, return values unchanged)
14. Sidecar reopen persistence
15. Serving source vs ingestion source distinction
"""

import os
import sqlite3
import pytest
import pandas as pd
from unittest.mock import patch

from analyst_dashboard.data.market_evidence import (
    MarketProvenance,
    MarketEvidence,
    Provider,
    IngestionSource,
    ServingSource,
    CacheOrigin,
    ObservationPrecision,
    ObservationSource,
    AdjustmentState,
    StructuralQuality,
    assess_structural_quality,
    create_legacy_provenance,
    create_direct_evidence,
    create_cached_evidence,
)
from analyst_dashboard.data.market_db import MarketDatabaseEngine

pytestmark = pytest.mark.tier1


# ── 1. Dataclass Creation & Serialization Tests ──────────────────────────────

def test_direct_provider_provenance_creation():
    """Verify direct provider provenance construction and serialization."""
    candles = [
        {"time": "2026-09-20", "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000000}
    ]
    evidence = create_direct_evidence(
        provider=Provider.YFINANCE,
        candles=candles,
        observed_date="2026-09-20",
        fallback_status=False,
    )
    assert evidence.provenance.provider == Provider.YFINANCE
    assert evidence.provenance.ingestion_source == IngestionSource.DIRECT_PROVIDER
    assert evidence.provenance.fallback_status is False
    assert evidence.serving_source == ServingSource.DIRECT_PROVIDER
    assert evidence.cache_origin == CacheOrigin.NONE
    assert evidence.provenance.structural_quality == StructuralQuality.COMPLETE
    assert evidence.candle_count == 1

    d = evidence.to_dict()
    assert d["provenance"]["provider"] == "YFINANCE"
    assert d["serving_source"] == "DIRECT_PROVIDER"
    assert "served_at" in d


def test_fallback_provider_provenance():
    """Verify fallback provider provenance properly flags fallback status."""
    evidence = create_direct_evidence(
        provider=Provider.YFINANCE,
        candles=[],
        observed_date="2026-09-20",
        fallback_status=True,
    )
    assert evidence.provenance.ingestion_source == IngestionSource.FALLBACK_PROVIDER
    assert evidence.provenance.fallback_status is True
    assert evidence.serving_source == ServingSource.FALLBACK_PROVIDER


def test_observation_precision_timestamp_vs_date():
    """Verify TIMESTAMP vs DATE precision differentiation."""
    # Timestamp
    prov_ts = MarketProvenance(
        provider=Provider.YFINANCE,
        observed_at="2026-09-20T19:55:00Z",
        observed_date="2026-09-20",
        observation_precision=ObservationPrecision.TIMESTAMP,
        observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
    )
    assert prov_ts.observation_precision == ObservationPrecision.TIMESTAMP
    assert prov_ts.observed_at == "2026-09-20T19:55:00Z"

    # Date
    prov_date = MarketProvenance(
        provider=Provider.YFINANCE,
        observed_at=None,
        observed_date="2026-09-20",
        observation_precision=ObservationPrecision.DATE,
        observation_source=ObservationSource.DERIVED_FROM_TRADE_DATE,
    )
    assert prov_date.observation_precision == ObservationPrecision.DATE
    assert prov_date.observed_at is None
    assert prov_date.observed_date == "2026-09-20"


def test_unknown_observation_and_provider():
    """Verify unknown observation and provider defaults."""
    prov = MarketProvenance()
    assert prov.provider == Provider.UNKNOWN
    assert prov.observation_precision == ObservationPrecision.UNKNOWN
    assert prov.observation_source == ObservationSource.UNKNOWN
    assert prov.adjustment_state == AdjustmentState.UNKNOWN
    assert prov.observed_at is None
    assert prov.observed_date is None


# ── 2. Structural Quality Assessment ──────────────────────────────────────────

def test_structural_quality_complete():
    """Valid OHLCV data yields COMPLETE."""
    valid_candles = [
        {"open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 1000},
        {"open": 102.0, "high": 106.0, "low": 101.0, "close": 105.0, "volume": 2000},
    ]
    assert assess_structural_quality(valid_candles) == StructuralQuality.COMPLETE

    df = pd.DataFrame(valid_candles)
    assert assess_structural_quality(df) == StructuralQuality.COMPLETE


def test_structural_quality_partial():
    """Missing required OHLC field yields PARTIAL."""
    partial_candles = [{"open": 100.0, "close": 102.0}]
    assert assess_structural_quality(partial_candles) == StructuralQuality.PARTIAL

    df = pd.DataFrame(partial_candles)
    assert assess_structural_quality(df) == StructuralQuality.PARTIAL


def test_structural_quality_corrupt():
    """NaN, inverted geometry (High < Low) or negative price yields CORRUPT."""
    corrupt_nan = [{"open": 100.0, "high": float("nan"), "low": 98.0, "close": 102.0}]
    assert assess_structural_quality(corrupt_nan) == StructuralQuality.CORRUPT

    corrupt_inverted = [{"open": 100.0, "high": 90.0, "low": 98.0, "close": 102.0}]
    assert assess_structural_quality(corrupt_inverted) == StructuralQuality.CORRUPT

    corrupt_negative = [{"open": -10.0, "high": 10.0, "low": -20.0, "close": 5.0}]
    assert assess_structural_quality(corrupt_negative) == StructuralQuality.CORRUPT


def test_structural_quality_does_not_alter_data():
    """Verify that assessing structural quality is read-only and never filters or mutates rows."""
    corrupt_candles = [
        {"open": 100.0, "high": 90.0, "low": 98.0, "close": 102.0},
        {"open": 102.0, "high": 106.0, "low": 101.0, "close": 105.0},
    ]
    orig_len = len(corrupt_candles)
    quality = assess_structural_quality(corrupt_candles)
    assert quality == StructuralQuality.CORRUPT
    assert len(corrupt_candles) == orig_len  # Zero rows filtered


# ── 3. Database Engine & Sidecar Isolation Tests ─────────────────────────────

def test_legacy_cache_read_without_sidecar(tmp_path):
    """Verify that candles stored without sidecar return factual legacy UNKNOWN provenance."""
    db_file = str(tmp_path / "test_market.db")
    engine = MarketDatabaseEngine(db_path=db_file)

    # Save via legacy API (no provenance written)
    candles = [
        {"time": "2026-09-18", "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 500000},
        {"time": "2026-09-19", "open": 103.0, "high": 106.0, "low": 102.0, "close": 105.0, "volume": 600000},
    ]
    engine.save_daily_candles("LEGACY_SYM", candles)

    # Legacy read returns exact candles
    legacy_read = engine.get_daily_candles("LEGACY_SYM")
    assert len(legacy_read) == 2
    assert legacy_read[0]["close"] == 103.0

    # Evidence read on unmigrated rows derives factual legacy provenance
    read_candles, evidence = engine.get_daily_candles_with_evidence("LEGACY_SYM")
    assert len(read_candles) == 2
    assert read_candles == legacy_read
    assert evidence.provenance.provider == Provider.UNKNOWN
    assert evidence.provenance.ingestion_source == IngestionSource.UNKNOWN
    assert evidence.provenance.observation_source == ObservationSource.DERIVED_FROM_TRADE_DATE
    assert evidence.serving_source == ServingSource.LOCAL_CACHE
    assert evidence.cache_origin == CacheOrigin.SQLITE_MARKET_STORE


def test_new_cache_read_with_sidecar(tmp_path):
    """Verify that candles saved with provenance retrieve authentic stored metadata."""
    db_file = str(tmp_path / "test_market.db")
    engine = MarketDatabaseEngine(db_path=db_file)

    prov = MarketProvenance(
        provider=Provider.YFINANCE,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        observed_at=None,
        observed_date="2026-09-20",
        observation_precision=ObservationPrecision.DATE,
        observation_source=ObservationSource.DERIVED_FROM_TRADE_DATE,
        ingested_at="2026-09-20T21:00:00Z",
        adjustment_state=AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED,
        structural_quality=StructuralQuality.COMPLETE,
        fallback_status=False,
    )

    candles = [
        {"time": "2026-09-20", "open": 150.0, "high": 155.0, "low": 149.0, "close": 154.0, "volume": 1200000}
    ]
    engine.save_daily_candles_with_evidence("NEW_SYM", candles, prov)

    # Verify both legacy read and evidence read
    legacy_read = engine.get_daily_candles("NEW_SYM")
    assert len(legacy_read) == 1
    assert legacy_read[0]["close"] == 154.0

    read_candles, evidence = engine.get_daily_candles_with_evidence("NEW_SYM")
    assert len(read_candles) == 1
    assert evidence.provenance.provider == Provider.YFINANCE
    assert evidence.provenance.ingestion_source == IngestionSource.DIRECT_PROVIDER
    assert evidence.provenance.ingested_at == "2026-09-20T21:00:00Z"
    assert evidence.serving_source == ServingSource.LOCAL_CACHE
    assert evidence.cache_origin == CacheOrigin.SQLITE_MARKET_STORE


def test_sidecar_reopen_persistence(tmp_path):
    """Verify that persisted sidecar data survives closing and re-opening MarketDatabaseEngine."""
    db_file = str(tmp_path / "test_market.db")
    engine1 = MarketDatabaseEngine(db_path=db_file)

    prov = MarketProvenance(
        provider=Provider.EODHD,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        observed_date="2026-09-21",
        observation_precision=ObservationPrecision.DATE,
        observation_source=ObservationSource.DERIVED_FROM_TRADE_DATE,
        ingested_at="2026-09-21T18:30:00Z",
        adjustment_state=AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED,
        structural_quality=StructuralQuality.COMPLETE,
    )
    candles = [{"time": "2026-09-21", "open": 50.0, "high": 52.0, "low": 49.5, "close": 51.5, "volume": 300000}]
    engine1.save_daily_candles_with_evidence("PERSIST_SYM", candles, prov)

    # Reopen in new instance
    engine2 = MarketDatabaseEngine(db_path=db_file)
    read_candles, evidence = engine2.get_daily_candles_with_evidence("PERSIST_SYM")
    assert len(read_candles) == 1
    assert evidence.provenance.provider == Provider.EODHD
    assert evidence.provenance.ingested_at == "2026-09-21T18:30:00Z"


def test_atomic_candle_and_provenance_rollback_on_failure(tmp_path):
    """Force failure during sidecar provenance insert and verify candle insert rolls back."""
    db_file = str(tmp_path / "test_market.db")
    engine = MarketDatabaseEngine(db_path=db_file)

    prov = MarketProvenance(
        provider=Provider.YFINANCE,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        observed_date="2026-09-20",
    )
    candles = [{"time": "2026-09-20", "open": 200.0, "high": 205.0, "low": 198.0, "close": 204.0, "volume": 800000}]

    # Corrupt table asset_ohlcv_provenance by dropping it or adding a trigger that raises an error
    conn = engine._get_connection()
    with conn:
        conn.execute("DROP TABLE asset_ohlcv_provenance")
    conn.close()

    # Attempt save_daily_candles_with_evidence - should raise OperationalError because sidecar table is missing
    with pytest.raises(sqlite3.OperationalError):
        engine.save_daily_candles_with_evidence("FAIL_SYM", candles, prov)

    # Reconnect and verify asset_ohlcv_daily has ZERO rows for FAIL_SYM
    conn2 = engine._get_connection()
    cursor = conn2.cursor()
    cursor.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'FAIL_SYM'")
    count = cursor.fetchone()[0]
    conn2.close()

    assert count == 0, "ATOMICITY_VIOLATION: Candle row was persisted despite provenance insert failure!"


def test_serving_source_vs_ingestion_source_distinction():
    """Verify that serving_source reflects retrieval channel while ingestion_source reflects original fetch."""
    prov = MarketProvenance(
        provider=Provider.YFINANCE,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        fallback_status=False,
    )
    # When served from cache:
    cached_evidence = create_cached_evidence(prov, cache_origin=CacheOrigin.SQLITE_MARKET_STORE, candle_count=10)
    assert cached_evidence.provenance.ingestion_source == IngestionSource.DIRECT_PROVIDER
    assert cached_evidence.serving_source == ServingSource.LOCAL_CACHE
    assert cached_evidence.cache_origin == CacheOrigin.SQLITE_MARKET_STORE
