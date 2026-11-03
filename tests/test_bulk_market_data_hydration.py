"""
tests/test_bulk_market_data_hydration.py

Authoritative Test Suite for Resumable Bulk Market Data Hydration Engine.
Validates:
1. Resumable Checkpoint Store (persistence, updates, status grouping).
2. Pre-flight WAL-checkpointed SQLite backup creation.
3. Automatic Delisted / Acquired security exclusion (CPRX lifecycle guard).
4. Numeric OHLCV bar validation and sidecar evidence persistence.
5. Gate 12 readiness calculation (threshold >= 95.0%).
6. Candidate full-universe manifest generation and deterministic SHA-256 hashing.
7. Non-destructive invariant: active production build UB-ARX-LAUNCH-133-20261102 preserved.
"""

from datetime import date, datetime, timedelta
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
import io
import json
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch
import pytest

from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.data.hydrate_market_data_bulk import (
    BulkHydrationCheckpointStore,
    BulkMarketDataHydrationEngine,
    CheckpointRecord,
    build_candidate_universe_hashes,
    main,
    read_universe_file,
)


class TestBulkHydrationCheckpointStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_checkpoint.db"
        self.store = BulkHydrationCheckpointStore(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_record_and_get_progress(self):
        self.store.record_progress(
            symbol="AAPL",
            status="COMPLETED",
            bars_saved=250,
            earliest_date="2025-11-03",
            latest_date="2026-10-30",
            latency_ms=120.5,
        )

        rec = self.store.get_record("AAPL")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.symbol, "AAPL")
        self.assertEqual(rec.status, "COMPLETED")
        self.assertEqual(rec.bars_saved, 250)
        self.assertEqual(rec.earliest_date, "2025-11-03")
        self.assertEqual(rec.latest_date, "2026-10-30")
        self.assertAlmostEqual(rec.latency_ms, 120.5)

    def test_summary_aggregation(self):
        self.store.record_progress("AAPL", "COMPLETED", bars_saved=250)
        self.store.record_progress("MSFT", "COMPLETED", bars_saved=250)
        self.store.record_progress("CPRX", "EXCLUDED_DELISTED", error_message="Acquired")
        self.store.record_progress("BADSYM", "FAILED", error_message="HTTP 404")

        summary = self.store.get_summary()
        self.assertEqual(summary.get("COMPLETED"), 2)
        self.assertEqual(summary.get("EXCLUDED_DELISTED"), 1)
        self.assertEqual(summary.get("FAILED"), 1)


class TestBulkHydrationEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.market_db_path = Path(self.temp_dir.name) / "market.db"
        # Initialize market DB schema
        self.market_engine = MarketDatabaseEngine(db_path=str(self.market_db_path))

        self.mock_fetcher = MagicMock()
        self.engine = BulkMarketDataHydrationEngine(
            target_db_path=self.market_db_path,
            checkpoint_dir=Path(self.temp_dir.name),
            fetcher=self.mock_fetcher,
            rate_limit_per_min=6000,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preflight_backup_creation(self):
        # Insert a sample candle to ensure DB has data
        conn = sqlite3.connect(str(self.market_db_path))
        conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('TEST', '2026-10-16', 10.0, 11.0, 9.0, 10.0, 1000)")
        conn.commit()
        conn.close()

        backup = self.engine.create_preflight_backup()
        self.assertTrue(backup.exists())
        self.assertGreater(backup.stat().st_size, 0)

        # Verify backup contains the inserted candle
        b_conn = sqlite3.connect(str(backup))
        count = b_conn.execute("SELECT count(*) FROM asset_ohlcv_daily WHERE symbol='TEST'").fetchone()[0]
        b_conn.close()
        self.assertEqual(count, 1)

    def test_cprx_delisted_exclusion(self):
        rec = self.engine.hydrate_symbol("CPRX", dry_run=False)
        self.assertEqual(rec.status, "EXCLUDED_DELISTED")
        self.assertIn("LIFECYCLE_EXCLUSION", rec.error_message)
        # Mock fetcher should never have been called
        self.mock_fetcher.fetch_historical_bars.assert_not_called()

    def test_successful_hydration_and_persistence(self):
        valid_bars = [
            {"t": f"{(datetime(2025, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')}T04:00:00Z", "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000000}
            for i in range(220)
        ]
        self.mock_fetcher.fetch_historical_bars.return_value = {
            "success": True,
            "status_code": 200,
            "symbol": "AAPL",
            "bars": valid_bars,
        }

        rec = self.engine.hydrate_symbol("AAPL", dry_run=False)
        self.assertEqual(rec.status, "COMPLETED")
        self.assertEqual(rec.bars_saved, 220)

        # Verify persisted in market DB
        candles = self.market_engine.get_daily_candles("AAPL", limit=300)
        self.assertGreaterEqual(len(candles), 200)

    def test_read_only_dry_run_engine_creates_no_local_database_or_checkpoint(self):
        absent_db = Path(self.temp_dir.name) / "absent-market.db"
        absent_checkpoint = Path(self.temp_dir.name) / "market_hydration_checkpoint.db"
        before = absent_checkpoint.read_bytes()
        ro = BulkMarketDataHydrationEngine(
            target_db_path=absent_db, checkpoint_dir=Path(self.temp_dir.name),
            fetcher=self.mock_fetcher, read_only=True,
        )
        self.assertFalse(absent_db.exists())
        self.assertEqual(absent_checkpoint.read_bytes(), before)
        with self.assertRaisesRegex(RuntimeError, "READ_ONLY_ENGINE"):
            ro.hydrate_symbol("AAPL", dry_run=False)

    def test_dry_run_never_marks_checkpoint_completed_or_mutates_market_store(self):
        valid_bars = [
            {"t": f"{(datetime(2025, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')}T04:00:00Z",
             "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000000}
            for i in range(220)
        ]
        self.mock_fetcher.fetch_historical_bars.return_value = {"success": True, "bars": valid_bars}

        rec = self.engine.hydrate_symbol("AAPL", dry_run=True)
        self.assertEqual(rec.status, "DRY_RUN_VALIDATED")
        self.assertEqual(rec.bars_saved, 220)
        self.assertIsNone(self.engine.checkpoint_store.get_record("AAPL"))
        self.assertEqual(self.market_engine.get_daily_candles("AAPL", limit=300), [])

    def test_dry_run_failure_does_not_change_existing_checkpoint(self):
        self.engine.checkpoint_store.record_progress("AAPL", "COMPLETED", bars_saved=220)
        self.mock_fetcher.fetch_historical_bars.return_value = {"success": False, "error": "PROVIDER_UNAVAILABLE"}
        rec = self.engine.hydrate_symbol("AAPL", dry_run=True)
        self.assertEqual(rec.status, "FAILED")
        saved = self.engine.checkpoint_store.get_record("AAPL")
        self.assertEqual(saved.status, "COMPLETED")
        self.assertEqual(saved.bars_saved, 220)

    def test_gate_12_readiness_evaluation(self):
        # Populate market DB: AAPL has 220 bars, MSFT has 220 bars, GLX has 30 bars, NOMKT has 0
        valid_bars = [
            {"time": (datetime(2025, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d"), "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000}
            for i in range(220)
        ]
        self.market_engine.save_daily_candles("AAPL", valid_bars)
        self.market_engine.save_daily_candles("MSFT", valid_bars)
        self.market_engine.save_daily_candles("GLX", valid_bars[:30])

        candidates = ["AAPL", "MSFT", "GLX", "CPRX", "NOMKT"]
        res = self.engine.evaluate_gate_12_readiness(candidates, min_required_bars=200)

        self.assertEqual(res["eligible_population_count"], 5)
        self.assertEqual(res["scannable_count"], 2)  # AAPL, MSFT
        self.assertEqual(res["insufficient_bars_count"], 1)  # GLX (30 bars < 200)
        self.assertEqual(res["delisted_excluded_count"], 1)  # CPRX
        self.assertEqual(res["missing_data_count"], 1)  # NOMKT
        self.assertEqual(res["readiness_coverage_pct"], 40.0)
        self.assertFalse(res["gate_12_passed"])  # 40% < 95%


class TestCandidateUniverseHashGeneration(unittest.TestCase):
    def test_deterministic_hash_generation(self):
        eligible = ["AAPL", "MSFT", "NVDA"]
        scannable = ["AAPL", "MSFT", "NVDA"]
        unavailable = []

        m1 = build_candidate_universe_hashes(eligible, scannable, unavailable, source_population_count=3)
        m2 = build_candidate_universe_hashes(list(reversed(eligible)), scannable, unavailable, source_population_count=3)

        self.assertEqual(m1["eligible_membership_hash"], m2["eligible_membership_hash"])
        self.assertEqual(m1["scannable_membership_hash"], m2["scannable_membership_hash"])
        self.assertFalse(m1["production_published"])
        self.assertEqual(m1["active_production_universe"], "UB-ARX-LAUNCH-133-20261102")


class TestGate12ReadOnlyCLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "market.db"
        self.market_engine = MarketDatabaseEngine(db_path=str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_only_cli_with_explicit_universe_file(self):
        symbols_file = Path(self.temp_dir.name) / "universe.txt"
        symbols_file.write_text("AAPL\nMSFT\n", encoding="utf-8")
        checkpoint_file = Path(self.temp_dir.name) / "market_hydration_checkpoint.db"
        self.assertFalse(checkpoint_file.exists())
        output = io.StringIO()
        with patch("sys.argv", ["radar-hydrate", "--db-path", str(self.db_path),
                                "--qualify-gate-12", "--universe-file", str(symbols_file)]):
            with redirect_stdout(output):
                main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["assessment_scope"], "UNIVERSE_FILE")
        self.assertEqual(result["eligible_population_count"], 2)
        self.assertEqual(result["missing_data_count"], 2)
        self.assertFalse(result["production_publication_authorized"])
        self.assertFalse(checkpoint_file.exists())

    def test_gate12_cli_requires_explicit_membership(self):
        with patch("sys.argv", ["radar-hydrate", "--db-path", str(self.db_path), "--qualify-gate-12"]):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)

    def test_universe_file_rejects_duplicate_symbols(self):
        symbols_file = Path(self.temp_dir.name) / "universe.txt"
        symbols_file.write_text("AAPL\nAAPL\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_universe_file(symbols_file)
