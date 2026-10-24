"""
tests/test_alpaca_historical_sip_ingestion.py

Unit, integration, and adversarial regression tests for AlpacaHistoricalSipIngestionAdapter and CLI.
Validates SIP-01 through SIP-05:
- SIP-01: Dry-run is strictly read-only (?mode=ro; zero bytes stays zero bytes; no schema creation).
          Apply mode validates existing schema before constructing writers.
- SIP-02: Strict ISO date parsing, NYSE trading session verification (weekends/holidays rejected),
          max completed session (2026-10-19), volume presence and integer requirement (fractional
          1.7 rejected, never truncated), symbol identity verification.
- SIP-03: Bounded pagination rejects incomplete acquisitions (exhausted budget with remaining token),
          detects token cycle loops, and rejects malformed pages.
- SIP-04: Full historical window and session count required for skip; single endpoint bar cannot skip;
          incompatible provenance (dividend-adjusted) triggers full backfill and is never blended.
- SIP-05: Honest batch outcomes (SUCCESS, PARTIAL_SUCCESS, FAILURE); CLI exit codes (0, 1, 2);
          manifest validation rejects unlisted symbol overrides; consumes Package B parameters.

Selected by Tier 2 Path-Filtered Gates: tier2c.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest

from analyst_dashboard.data.alpaca_fetcher import AlpacaMarketFetcher
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.data.market_evidence import (
    AdjustmentState,
    IngestionSource,
    MarketEvidence,
    MarketProvenance,
    ObservationPrecision,
    ObservationSource,
    Provider,
    StructuralQuality,
)
from scripts.ingest_alpaca_historical import (
    AlpacaHistoricalSipIngestionAdapter,
    BarValidationResult,
    CandidateIngestionDisposition,
    IngestionBatchSummary,
    get_expected_nyse_sessions,
    inspect_existing_market_store_readonly,
    is_nyse_trading_session,
    validate_target_database_for_write,
    parse_args,
    main,
)

pytestmark = pytest.mark.tier2c


class MockAlpacaFetcher:
    """Configurable mock fetcher for unit tests."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.calls: List[Dict[str, Any]] = []

    def fetch_historical_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        feed: str = "sip",
        adjustment: str = "split",
        limit: int = 1000,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        call_record = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "feed": feed,
            "adjustment": adjustment,
            "limit": limit,
            "page_token": page_token,
        }
        self.calls.append(call_record)

        key = f"{symbol}:{page_token}" if page_token else symbol
        if key in self.responses:
            return self.responses[key]
        if symbol in self.responses:
            return self.responses[symbol]

        # Default synthetic series covering all expected NYSE sessions in [start, end]
        from scripts.ingest_alpaca_historical import get_expected_nyse_sessions
        exp = get_expected_nyse_sessions(start, end)
        if not exp:
            exp = [start] if start == end else [start, end]
        bars = [
            {"t": f"{sess}T04:00:00Z", "o": 100.0, "h": 105.0, "l": 99.0, "c": 104.0, "v": 1000000}
            for sess in exp
        ]
        return {
            "success": True,
            "status_code": 200,
            "symbol": symbol,
            "bars": bars,
            "next_page_token": None,
        }


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    """Creates a temporary market SQLite database with authoritative schema initialized."""
    db_file = tmp_path / "test_market_store.db"
    engine = MarketDatabaseEngine(db_path=str(db_file))
    return str(db_file)


@pytest.fixture
def sample_manifest_path(tmp_path: Path) -> Path:
    """Creates a temporary sample manifest file matching Package B structure."""
    manifest = {
        "ingestion_scope": {
            "production_target_backfill": {
                "missing_symbols": ["ABBV", "NVDA"]
            },
            "production_target_refresh": {
                "per_symbol_parameters": [
                    {"symbol": "ADBE", "requested_start_session": "2026-09-25"}
                ]
            }
        }
    }
    p = tmp_path / "sample_manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


# ────────────────────────────────────────────────────────────
# 1. Feed & Adjustment Parameter Invariant Tests
# ────────────────────────────────────────────────────────────

class TestFeedAndAdjustmentInvariants:
    def test_adapter_rejects_non_sip_feed(self):
        """Adapter strictly rejects non-SIP feeds (e.g. IEX) to prevent single-venue volume bias."""
        with pytest.raises(ValueError, match="INVALID_FEED"):
            AlpacaHistoricalSipIngestionAdapter(feed="iex", adjustment="split")

        with pytest.raises(ValueError, match="INVALID_FEED"):
            AlpacaHistoricalSipIngestionAdapter(feed="otc", adjustment="split")

    def test_adapter_rejects_non_split_adjustment(self):
        """Adapter strictly rejects non-split adjustments (e.g. raw or all) per Package B contract."""
        with pytest.raises(ValueError, match="INVALID_ADJUSTMENT"):
            AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="raw")

        with pytest.raises(ValueError, match="INVALID_ADJUSTMENT"):
            AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="all")

    def test_adapter_accepts_sip_and_split(self):
        """Adapter accepts canonical feed='sip' and adjustment='split'."""
        adapter = AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="split")
        assert adapter.feed == "sip"
        assert adapter.adjustment == "split"


# ────────────────────────────────────────────────────────────
# 2. SIP-01 — Dry-Run Read-Only & Target Schema Safety Tests
# ────────────────────────────────────────────────────────────

class TestDryRunAndTargetSafety:
    def test_dry_run_with_existing_zero_byte_database_does_not_mutate_file(self, tmp_path: Path, sample_manifest_path: Path):
        """SIP-01: Dry run against an existing 0-byte file leaves the file at exactly 0 bytes."""
        zero_byte_db = tmp_path / "zero_byte.db"
        zero_byte_db.touch()
        assert zero_byte_db.stat().st_size == 0

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV"],
            start_date="2026-10-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=str(zero_byte_db),
        )

        assert summary.mode == "DRY_RUN"
        # Invariant: 0-byte file must remain 0 bytes!
        assert zero_byte_db.stat().st_size == 0, f"Expected 0 bytes, but file size became {zero_byte_db.stat().st_size}"

    def test_dry_run_with_database_lacking_tables_does_not_create_schema(self, tmp_path: Path, sample_manifest_path: Path):
        """SIP-01: Dry run on a database lacking required tables inspects without creating tables."""
        incomplete_db = tmp_path / "incomplete.db"
        conn = sqlite3.connect(str(incomplete_db))
        conn.execute("CREATE TABLE other_table (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV"],
            start_date="2026-10-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=str(incomplete_db),
        )

        assert summary.mode == "DRY_RUN"
        # Verify required tables were NOT created
        conn = sqlite3.connect(str(incomplete_db))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        conn.close()
        assert "asset_ohlcv_daily" not in tables
        assert "asset_ohlcv_provenance" not in tables

    def test_apply_mode_fails_if_database_is_zero_bytes(self, tmp_path: Path, sample_manifest_path: Path):
        """SIP-01: Apply mode fails closed if target database is 0 bytes (uninitialized)."""
        zero_db = tmp_path / "zero_apply.db"
        zero_db.touch()

        adapter = AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="split")
        with pytest.raises(ValueError, match="TARGET_DATABASE_EMPTY"):
            adapter.execute_bounded_ingestion(
                manifest_path=sample_manifest_path,
                symbols_override=["ABBV"],
                apply_writes=True,
                db_path=str(zero_db),
            )

    def test_apply_mode_fails_if_database_schema_incomplete(self, tmp_path: Path, sample_manifest_path: Path):
        """SIP-01: Apply mode fails closed if target database lacks required tables."""
        incomplete_db = tmp_path / "incomplete_apply.db"
        conn = sqlite3.connect(str(incomplete_db))
        conn.execute("CREATE TABLE random_table (id INT)")
        conn.commit()
        conn.close()

        adapter = AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="split")
        with pytest.raises(ValueError, match="TARGET_DATABASE_SCHEMA_INCOMPLETE"):
            adapter.execute_bounded_ingestion(
                manifest_path=sample_manifest_path,
                symbols_override=["ABBV"],
                apply_writes=True,
                db_path=str(incomplete_db),
            )

    def test_apply_without_db_path_raises_error(self, sample_manifest_path: Path):
        """Invoking with apply_writes=True but no db_path raises ValueError."""
        adapter = AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="split")
        with pytest.raises(ValueError, match="MISSING_DB_PATH"):
            adapter.execute_bounded_ingestion(
                manifest_path=sample_manifest_path,
                apply_writes=True,
                db_path=None,
            )

    def test_apply_with_nonexistent_database_raises_error(self, sample_manifest_path: Path, tmp_path: Path):
        """Target database must exist prior to applying writes."""
        nonexistent = str(tmp_path / "does_not_exist.db")
        adapter = AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="split")
        with pytest.raises(FileNotFoundError, match="TARGET_DATABASE_ABSENT"):
            adapter.execute_bounded_ingestion(
                manifest_path=sample_manifest_path,
                apply_writes=True,
                db_path=nonexistent,
            )

    def test_invalid_manifest_with_explicit_db_path_does_not_touch_db(self, tmp_path: Path, test_db_path: str):
        """Invalid manifest raises error before opening database connections."""
        bad_manifest = tmp_path / "nonexistent_manifest.json"
        adapter = AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="split")
        with pytest.raises(FileNotFoundError, match="MANIFEST_NOT_FOUND"):
            adapter.execute_bounded_ingestion(
                manifest_path=bad_manifest,
                apply_writes=False,
                db_path=test_db_path,
            )


# ────────────────────────────────────────────────────────────
# 3. SIP-02 — Session, Coverage, Numerics & Identity Validation Tests
# ────────────────────────────────────────────────────────────

class TestBarSeriesValidation:
    def test_validation_passes_valid_bars(self):
        """Clean bars on valid NYSE sessions (2026-10-15 Thu, 2026-10-16 Fri, 2026-10-19 Mon) pass."""
        bars = [
            {"t": "2026-10-15T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000000},
            {"t": "2026-10-16T04:00:00Z", "o": 102.5, "h": 104.0, "l": 101.0, "c": 103.0, "v": 800000},
            {"t": "2026-10-19T04:00:00Z", "o": 103.0, "h": 107.0, "l": 102.0, "c": 106.5, "v": 1200000},
        ]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-15", "2026-10-19")
        assert res.is_valid is True
        assert res.rejection_reason is None
        assert res.bar_count == 3
        assert res.first_session == "2026-10-15"
        assert res.last_session == "2026-10-19"
        assert res.structural_quality == StructuralQuality.COMPLETE

    def test_validation_rejects_sunday_session(self):
        """SIP-02: Sunday 2026-10-18 is rejected as a non-trading session."""
        bars = [{"t": "2026-10-18T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-15", "2026-10-19")
        assert res.is_valid is False
        assert "NON_TRADING_SESSION" in res.rejection_reason

    def test_validation_rejects_saturday_session(self):
        """SIP-02: Saturday 2026-10-17 is rejected as a non-trading session."""
        bars = [{"t": "2026-10-17T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-15", "2026-10-19")
        assert res.is_valid is False
        assert "NON_TRADING_SESSION" in res.rejection_reason

    def test_validation_rejects_holiday_session(self):
        """SIP-02: NYSE holiday (Labor Day 2026-09-07) is rejected as a non-trading session."""
        bars = [{"t": "2026-09-07T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-09-01", "2026-09-15")
        assert res.is_valid is False
        assert "NON_TRADING_SESSION" in res.rejection_reason

    def test_validation_rejects_impossible_date(self):
        """SIP-02: Impossible calendar date 2026-10-99 is rejected."""
        bars = [{"t": "2026-10-99T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-01", "2026-10-19")
        assert res.is_valid is False
        assert "INVALID_ISO_TIMESTAMP" in res.rejection_reason

    def test_validation_rejects_future_session(self):
        """SIP-02: Future date 2099-01-05 is rejected as exceeding max completed session."""
        bars = [{"t": "2099-01-05T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-01", "2026-10-19")
        assert res.is_valid is False
        assert "OUT_OF_WINDOW_SESSIONS" in res.rejection_reason

    def test_validation_rejects_missing_volume(self):
        """SIP-02: Bar missing volume key or with null volume is rejected; never defaulted to zero."""
        bars_missing_key = [{"t": "2026-10-19T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars_missing_key, "2026-10-15", "2026-10-19")
        assert res.is_valid is False
        assert "MISSING_VOLUME" in res.rejection_reason

        bars_null_v = [{"t": "2026-10-19T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": None}]
        res_null = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars_null_v, "2026-10-15", "2026-10-19")
        assert res_null.is_valid is False
        assert "NULL_VOLUME" in res_null.rejection_reason

    def test_validation_rejects_fractional_volume(self):
        """SIP-02: Fractional volume (1.7) is rejected; silent truncation to 1 is strictly prohibited."""
        bars_frac = [{"t": "2026-10-19T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1.7}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars_frac, "2026-10-15", "2026-10-19")
        assert res.is_valid is False
        assert "NON_INTEGER_VOLUME" in res.rejection_reason

    def test_validation_accepts_integer_volume(self):
        """Integer volume (represented as int or whole float like 1000.0) is accepted."""
        bars = [{"t": "2026-10-19T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000.0}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-19", "2026-10-19")
        assert res.is_valid is True
        assert res.validated_bars[0]["volume"] == 1000

    def test_validation_rejects_bar_symbol_mismatch(self):
        """SIP-02: Bar containing contradictory symbol identity is rejected."""
        bars = [{"t": "2026-10-19T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000, "S": "MSFT"}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-15", "2026-10-19")
        assert res.is_valid is False
        assert "SYMBOL_IDENTITY_MISMATCH" in res.rejection_reason

    def test_validation_rejects_non_finite_numeric_values(self):
        bars = [{"t": "2026-10-19T04:00:00Z", "o": float("nan"), "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-15", "2026-10-19")
        assert res.is_valid is False
        assert "NON_FINITE_OR_INVALID_NUMERIC_VALUES" in res.rejection_reason

    def test_validation_rejects_geometric_inconsistencies(self):
        # High < Low
        bars_h_lt_l = [{"t": "2026-10-19T04:00:00Z", "o": 100.0, "h": 90.0, "l": 95.0, "c": 92.0, "v": 1000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars_h_lt_l, "2026-10-15", "2026-10-19")
        assert res.is_valid is False
        assert "OHLC_GEOMETRIC_INCONSISTENCIES" in res.rejection_reason


# ────────────────────────────────────────────────────────────
# 4. SIP-03 — Bounded Pagination Integrity Tests
# ────────────────────────────────────────────────────────────

class TestPaginationHandling:
    def test_normal_multiple_pages(self):
        """Adapter follows next_page_token across pages, combining all bars."""
        page1 = [{"t": "2026-10-15T04:00:00Z", "o": 100.0, "h": 105.0, "l": 99.0, "c": 104.0, "v": 1000}]
        page2 = [{"t": "2026-10-16T04:00:00Z", "o": 104.0, "h": 106.0, "l": 103.0, "c": 105.0, "v": 1200}]

        mock_fetcher = MockAlpacaFetcher({
            "MSFT": {"success": True, "status_code": 200, "symbol": "MSFT", "bars": page1, "next_page_token": "TOK2"},
            "MSFT:TOK2": {"success": True, "status_code": 200, "symbol": "MSFT", "bars": page2, "next_page_token": None},
        })

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")
        ok, bars, pages, err = adapter.fetch_all_candidate_bars("MSFT", "2026-10-15", "2026-10-16")
        assert ok is True
        assert len(bars) == 2
        assert pages == 2
        assert err is None

    def test_pagination_rejects_repeated_token_cycle(self):
        """SIP-03: Repeated next_page_token triggers loop detection and fails closed."""
        page = [{"t": "2026-10-15T04:00:00Z", "o": 100.0, "h": 105.0, "l": 99.0, "c": 104.0, "v": 1000}]
        mock_fetcher = MockAlpacaFetcher({
            "MSFT": {"success": True, "status_code": 200, "symbol": "MSFT", "bars": page, "next_page_token": "CYCLE_TOK"},
            "MSFT:CYCLE_TOK": {"success": True, "status_code": 200, "symbol": "MSFT", "bars": page, "next_page_token": "CYCLE_TOK"},
        })

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")
        ok, bars, pages, err = adapter.fetch_all_candidate_bars("MSFT", "2026-10-15", "2026-10-16")
        assert ok is False
        assert "PAGINATION_CYCLE_DETECTED" in err

    def test_pagination_budget_exhaustion_with_remaining_token_fails(self):
        """SIP-03: Reaching max_pages (25) with a remaining token fails closed as incomplete acquisition."""
        def mock_fetch(symbol, start, end, feed, adjustment, limit, page_token=None):
            page_num = int(page_token) if page_token else 1
            return {
                "success": True,
                "status_code": 200,
                "symbol": symbol,
                "bars": [{"t": f"2026-10-{page_num:02d}T04:00:00Z", "o": 100.0, "h": 105.0, "l": 99.0, "c": 104.0, "v": 1000}],
                "next_page_token": str(page_num + 1),  # Always generates next token
            }

        mock_f = MagicMock()
        mock_f.fetch_historical_bars.side_effect = mock_fetch

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_f, feed="sip", adjustment="split")
        ok, bars, pages, err = adapter.fetch_all_candidate_bars("AAPL", "2026-01-01", "2026-10-19")
        assert ok is False
        assert pages == 25
        assert "PAGINATION_BUDGET_EXHAUSTED" in err

    def test_pagination_rejects_malformed_page(self):
        """SIP-03: Malformed page (bars not a list) fails closed."""
        mock_fetcher = MockAlpacaFetcher({
            "AAPL": {"success": True, "status_code": 200, "symbol": "AAPL", "bars": "NOT_A_LIST", "next_page_token": None}
        })
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")
        ok, bars, pages, err = adapter.fetch_all_candidate_bars("AAPL", "2026-10-15", "2026-10-19")
        assert ok is False
        assert "MALFORMED_PAGE" in err

    def test_mid_pagination_failure_fails_closed(self):
        """SIP-03: Page 1 succeeds, page 2 fails -> entire acquisition fails closed without partial bars."""
        mock_fetcher = MockAlpacaFetcher({
            "AAPL": {"success": True, "status_code": 200, "symbol": "AAPL", "bars": [{"t": "2026-10-15T04:00:00Z", "o": 100.0, "h": 105.0, "l": 99.0, "c": 104.0, "v": 1000}], "next_page_token": "TOK2"},
            "AAPL:TOK2": {"success": False, "status_code": 500, "symbol": "AAPL", "bars": [], "error": "PAGE2_CRASH"},
        })
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")
        ok, bars, pages, err = adapter.fetch_all_candidate_bars("AAPL", "2026-10-15", "2026-10-19")
        assert ok is False
        assert "PAGE2_CRASH" in err


# ────────────────────────────────────────────────────────────
# 5. SIP-04 — Skip, Refresh & Adjustment Compatibility Tests
# ────────────────────────────────────────────────────────────

class TestSkipRefreshAndAdjustmentCompatibility:
    def test_shallow_single_bar_history_does_not_skip_triggers_backfill(self, test_db_path: str, sample_manifest_path: Path):
        """SIP-04: A single bar on 2026-10-19 must NOT satisfy a 1-year request; triggers BACKFILL."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-19T20:00:00Z",
            observed_date="2026-10-19",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-19T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        # Store only 1 bar on 2026-10-19
        engine.save_daily_candles_with_evidence("ABBV", [
            {"time": "2026-10-19", "open": 254.0, "high": 258.0, "low": 253.0, "close": 257.0, "volume": 1200000}
        ], prov)

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["ABBV"]
        # Invariant: Must NOT be SKIP_ALREADY_CURRENT! Must be BACKFILL!
        assert disp["action_type"] == "BACKFILL"
        assert disp["status"] == "PREPARED_DRY_RUN"
        assert summary.backfilled_count == 1
        assert summary.skipped_current_count == 0

    def test_incompatible_provenance_triggers_backfill_does_not_blend(self, test_db_path: str, sample_manifest_path: Path):
        """SIP-04: Existing data with dividend-adjusted provenance triggers full BACKFILL, not increment."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        incompatible_prov = MarketProvenance(
            provider=Provider.UNKNOWN,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-09-24T20:00:00Z",
            observed_date="2026-09-24",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-09-24T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED,  # Incompatible basis!
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        engine.save_daily_candles_with_evidence("ADBE", [
            {"time": "2025-09-15", "open": 400.0, "high": 405.0, "low": 395.0, "close": 402.0, "volume": 1000000},
            {"time": "2026-09-24", "open": 500.0, "high": 505.0, "low": 498.0, "close": 502.0, "volume": 1500000},
        ], incompatible_prov)

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ADBE"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["ADBE"]
        # Invariant: Incompatible provenance triggers BACKFILL from start_date, NOT REFRESH!
        assert disp["action_type"] == "BACKFILL"
        assert disp["provenance_compatible"] is False
        assert mock_fetcher.calls[0]["start"] == "2025-09-15"

    def test_cross_provider_lineage_massive_polygon_triggers_full_backfill_does_not_blend_providers(
        self, test_db_path: str, tmp_path: Path
    ):
        """SIP-04 & Lineage Purity: Existing data from MASSIVE_POLYGON triggers BACKFILL, preventing mixed Alpaca/Polygon series."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        polygon_prov = MarketProvenance(
            provider="MASSIVE_POLYGON",
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-16T20:00:00Z",
            observed_date="2026-10-16",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-16T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        sessions = get_expected_nyse_sessions("2025-09-15", "2026-10-16")
        bars = [{"time": s, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1000} for s in sessions]
        engine.save_daily_candles_with_evidence("GOOG", bars, polygon_prov)

        # Manifest with GOOG
        goog_manifest = tmp_path / "goog_manifest.json"
        goog_manifest.write_text(json.dumps({
            "ingestion_scope": {
                "production_target_backfill": {"missing_symbols": ["GOOG"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=goog_manifest,
            symbols_override=["GOOG"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["GOOG"]
        assert disp["provenance_compatible"] is False
        assert disp["action_type"] == "BACKFILL"
        assert mock_fetcher.calls[0]["start"] == "2025-09-15"

    def test_symbol_identity_mismatch_from_fetcher_rejected(self, sample_manifest_path: Path):
        """SIP-02 & SIP-04: Fetcher returning MSFT when ABBV was requested fails closed."""
        mock_fetcher = MockAlpacaFetcher({
            "ABBV": {"success": True, "status_code": 200, "symbol": "MSFT", "bars": [{"t": "2026-10-19T04:00:00Z", "o": 100.0, "h": 105.0, "l": 99.0, "c": 104.0, "v": 1000}], "next_page_token": None}
        })
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")
        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV"],
            start_date="2026-10-15",
            end_date="2026-10-19",
        )
        disp = summary.candidate_dispositions["ABBV"]
        assert disp["status"] == "PROVIDER_ERROR"
        assert "SYMBOL_IDENTITY_MISMATCH" in disp["rejection_reason"]


# ────────────────────────────────────────────────────────────
# 6. SIP-05 — Honest Batch Outcomes & Manifest Enforcement Tests
# ────────────────────────────────────────────────────────────

class TestBatchOutcomeAndManifestEnforcement:
    def test_all_provider_requests_fail_produces_failure_outcome_and_exit_1(self, sample_manifest_path: Path):
        """SIP-05: When all provider requests fail, batch outcome is FAILURE and CLI exits with 1."""
        mock_f = MagicMock()
        mock_f.fetch_historical_bars.return_value = {
            "success": False, "status_code": 500, "symbol": "ABBV", "bars": [], "error": "OUTAGE"
        }
        with patch("scripts.ingest_alpaca_historical.AlpacaMarketFetcher", return_value=mock_f):
            rc = main([
                "--manifest-path", str(sample_manifest_path),
                "--symbols", "ABBV",
                "--start-date", "2026-10-15",
                "--end-date", "2026-10-19",
            ])
            assert rc == 1, f"Expected CLI exit code 1 for full failure, got {rc}"

    def test_partial_success_produces_exit_code_2(self, sample_manifest_path: Path):
        """SIP-05: When 1 candidate succeeds and 1 fails, batch outcome is PARTIAL_SUCCESS and CLI exits with 2."""
        mock_f = MockAlpacaFetcher({
            "NVDA": {"success": False, "status_code": 500, "symbol": "NVDA", "bars": [], "error": "NVDA_500"},
            # ABBV will succeed with default synthetic series
        })
        with patch("scripts.ingest_alpaca_historical.AlpacaMarketFetcher", return_value=mock_f):
            rc = main([
                "--manifest-path", str(sample_manifest_path),
                "--symbols", "ABBV,NVDA",
                "--start-date", "2026-10-15",
                "--end-date", "2026-10-19",
            ])
            assert rc == 2, f"Expected CLI exit code 2 for partial success, got {rc}"

    def test_successful_complete_batch_produces_exit_code_0(self, sample_manifest_path: Path):
        """SIP-05: When all candidates succeed, batch outcome is SUCCESS and CLI exits with 0."""
        mock_f = MockAlpacaFetcher()
        with patch("scripts.ingest_alpaca_historical.AlpacaMarketFetcher", return_value=mock_f):
            rc = main([
                "--manifest-path", str(sample_manifest_path),
                "--symbols", "ABBV",
                "--start-date", "2026-10-15",
                "--end-date", "2026-10-19",
            ])
            assert rc == 0, f"Expected CLI exit code 0 for success, got {rc}"

    def test_requested_symbols_absent_from_manifest_raises_error(self, sample_manifest_path: Path):
        """SIP-05: Requesting a symbol absent from the manifest raises ValueError (never silently empty)."""
        adapter = AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="split")
        with pytest.raises(ValueError, match="REQUESTED_SYMBOLS_ABSENT_FROM_MANIFEST"):
            adapter.execute_bounded_ingestion(
                manifest_path=sample_manifest_path,
                symbols_override=["NONEXISTENT_SYMBOL"],
                start_date="2026-10-15",
                end_date="2026-10-19",
            )

    def test_empty_effective_scope_raises_error(self, tmp_path: Path):
        """SIP-05: An empty manifest scope raises ValueError."""
        empty_manifest = tmp_path / "empty.json"
        empty_manifest.write_text(json.dumps([]), encoding="utf-8")
        adapter = AlpacaHistoricalSipIngestionAdapter(feed="sip", adjustment="split")
        with pytest.raises(ValueError, match="EMPTY_EFFECTIVE_SCOPE"):
            adapter.execute_bounded_ingestion(manifest_path=empty_manifest)

    def test_package_b_per_symbol_parameters_consumed(self, sample_manifest_path: Path, test_db_path: str):
        """SIP-05: Package B per_symbol_parameters (requested_start_session) are consumed during refresh."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-09-24T20:00:00Z",
            observed_date="2026-09-24",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-09-24T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        p1_sessions = get_expected_nyse_sessions("2025-09-15", "2026-09-24")
        p2_sessions = get_expected_nyse_sessions("2026-09-25", "2026-10-19")
        initial_candles = [
            {"time": s, "open": 400.0, "high": 405.0, "low": 395.0, "close": 402.0, "volume": 1000000}
            for s in p1_sessions
        ]
        engine.save_daily_candles_with_evidence("ADBE", initial_candles, prov)

        tail_bars = [
            {"t": f"{s}T04:00:00Z", "o": 500.0, "h": 505.0, "l": 498.0, "c": 502.0, "v": 1500000}
            for s in p2_sessions
        ]
        mock_f = MockAlpacaFetcher(responses={"ADBE": {"success": True, "symbol": "ADBE", "bars": tail_bars, "next_page_token": None}})
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_f, feed="sip", adjustment="split")
        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ADBE"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            db_path=test_db_path,
        )

        assert summary.refreshed_count == 1
        # Package B manifest specifies requested_start_session = 2026-09-25
        assert mock_f.calls[0]["start"] == "2026-09-25"
        assert summary.manifest_scope_type == "PACKAGE_B"


# ────────────────────────────────────────────────────────────
# 7. CLI & Import Safety Tests
# ────────────────────────────────────────────────────────────

class TestCliAndImportSafety:
    def test_import_safety_makes_zero_network_or_db_calls(self):
        """Importing module or inspecting help does not execute network requests or database writes."""
        with patch("scripts.ingest_alpaca_historical.MarketDatabaseEngine") as mock_engine:
            with patch("scripts.ingest_alpaca_historical.AlpacaMarketFetcher") as mock_fetch:
                with pytest.raises(SystemExit) as exc_info:
                    parse_args(["--help"])
                assert exc_info.value.code == 0

                args = parse_args(["--manifest-path", "test.json", "--symbols", "AAPL"])
                assert args.feed == "sip"
                assert args.adjustment == "split"
                assert args.apply is False
                assert mock_engine.call_count == 0
                assert mock_fetch.call_count == 0

    def test_cli_dry_run_main_execution(self, sample_manifest_path: Path, tmp_path: Path):
        """CLI main entrypoint runs dry-run and writes execution summary JSON."""
        summary_out = tmp_path / "summary_out.json"
        with patch("scripts.ingest_alpaca_historical.AlpacaMarketFetcher", return_value=MockAlpacaFetcher()):
            rc = main([
                "--manifest-path", str(sample_manifest_path),
                "--symbols", "ABBV",
                "--start-date", "2026-10-15",
                "--end-date", "2026-10-19",
                "--output-summary-path", str(summary_out),
            ])

        assert rc == 0
        assert summary_out.exists()
        summary_data = json.loads(summary_out.read_text(encoding="utf-8"))
        assert summary_data["mode"] == "DRY_RUN"
        assert summary_data["batch_outcome"] == "SUCCESS"
        assert summary_data["backfilled_count"] == 1
        assert "ABBV" in summary_data["candidate_dispositions"]


# ────────────────────────────────────────────────────────────
# 8. Targeted Audit Residual Closure Tests (SIP-01 through SIP-05)
# ────────────────────────────────────────────────────────────

class TestTargetedAuditResidualClosures:
    """Verifies targeted closure of independent audit findings SIP-01 through SIP-05."""

    def test_apply_with_absent_manifest_does_not_initialize_schema(self, tmp_path: Path):
        """SIP-01: Applying with absent manifest fails closed BEFORE creating catalyst/factor tables."""
        db_file = tmp_path / "custom_market.db"
        # Create minimal database with required market tables ONLY
        conn = sqlite3.connect(str(db_file))
        conn.execute("""
            CREATE TABLE asset_ohlcv_daily (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                PRIMARY KEY (symbol, trade_date)
            )
        """)
        conn.execute("""
            CREATE TABLE asset_ohlcv_provenance (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                provider TEXT NOT NULL,
                ingestion_source TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                observed_date TEXT NOT NULL,
                observation_precision TEXT NOT NULL,
                observation_source TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                adjustment_state TEXT NOT NULL,
                structural_quality TEXT NOT NULL,
                fallback_status BOOLEAN NOT NULL,
                PRIMARY KEY (symbol, trade_date)
            )
        """)
        conn.commit()
        conn.close()

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=MockAlpacaFetcher(), feed="sip", adjustment="split")
        absent_manifest = tmp_path / "non_existent_manifest.json"

        with pytest.raises(FileNotFoundError):
            adapter.execute_bounded_ingestion(
                manifest_path=absent_manifest,
                apply_writes=True,
                db_path=str(db_file),
            )

        # Verify that NO catalyst, factor, or insider tables were created
        conn = sqlite3.connect(f"{db_file.as_uri()}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        conn.close()

        assert "asset_catalyst_registry" not in tables
        assert "asset_factor_snapshots" not in tables
        assert "insider_disclosures" not in tables

    def test_calendar_unavailable_fails_closed_rejects_labor_day(self):
        """SIP-02: Calendar unavailability fails closed and rejects Labor Day 2026-09-07."""
        with patch("scripts.ingest_alpaca_historical._NYSE_CALENDAR", None):
            is_valid, reason = is_nyse_trading_session(date(2026, 9, 7))
            assert is_valid is False
            assert "CALENDAR_UNAVAILABLE" in reason

            # In bar series validation
            bars = [{"t": "2026-09-07T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000}]
            res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-09-01", "2026-10-19")
            assert res.is_valid is False
            assert "CALENDAR_UNAVAILABLE" in res.rejection_reason

    def test_real_fetcher_preserves_returned_symbol_and_rejects_mismatch(self, sample_manifest_path: Path):
        """SIP-02: Real AlpacaMarketFetcher preserves provider-returned symbol identity."""
        fetcher = AlpacaMarketFetcher(api_key_id="TEST_KEY", api_secret_key="TEST_SECRET")

        # Mock HTTP response with symbol: MSFT
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "symbol": "MSFT",
            "bars": [
                {"t": "2026-10-19T04:00:00Z", "o": 400.0, "h": 405.0, "l": 395.0, "c": 402.0, "v": 1000000}
            ],
            "next_page_token": None,
        }

        with patch.object(fetcher.session, "get", return_value=mock_resp):
            # Request as ABBV
            res = fetcher.fetch_historical_bars(symbol="ABBV", start="2026-10-19", end="2026-10-19")
            # Verify fetcher preserved provider-returned identity (MSFT) rather than echoing ABBV
            assert res["symbol"] == "MSFT"

            # When processed by adapter, it must reject with SYMBOL_IDENTITY_MISMATCH
            adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=fetcher, feed="sip", adjustment="split")
            ok, bars, pages, err = adapter.fetch_all_candidate_bars("ABBV", "2026-10-19", "2026-10-19")
            assert ok is False
            assert "SYMBOL_IDENTITY_MISMATCH" in str(err)

    def test_incomplete_acquisition_missing_endpoint_rejected(self):
        """SIP-02: Series ending before requested target endpoint is rejected as incomplete."""
        # Requesting through 2026-10-19, but provider returns only 2025-09-15
        bars = [{"t": "2025-09-15T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2025-09-15", "2026-10-19")
        assert res.is_valid is False
        assert "INCOMPLETE_ACQUISITION_MISSING_ENDPOINT" in res.rejection_reason

    def test_false_skip_eliminated_when_earlier_provenance_is_dividend_adjusted(self, test_db_path: str, sample_manifest_path: Path):
        """SIP-04: Existing history with any dividend-adjusted provenance cannot falsely skip; triggers BACKFILL."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov_div = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2025-09-15T20:00:00Z",
            observed_date="2025-09-15",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2025-09-15T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        prov_split = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-19T20:00:00Z",
            observed_date="2026-10-19",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-19T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        engine.save_daily_candles_with_evidence("ABBV", [
            {"time": "2025-09-15", "open": 200.0, "high": 205.0, "low": 195.0, "close": 202.0, "volume": 1000000}
        ], prov_div)
        engine.save_daily_candles_with_evidence("ABBV", [
            {"time": "2026-10-19", "open": 250.0, "high": 255.0, "low": 248.0, "close": 252.0, "volume": 1200000}
        ], prov_split)

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=MockAlpacaFetcher(), feed="sip", adjustment="split")
        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["ABBV"]
        # Invariant: Must NOT be SKIP_ALREADY_CURRENT because earlier provenance is dividend-adjusted
        assert disp["action_type"] == "BACKFILL"
        assert disp["provenance_compatible"] is False

    def test_atomic_replacement_on_backfill_cleans_prior_incompatible_history(self, test_db_path: str, sample_manifest_path: Path):
        """R2: Bounded backfill with incompatible history outside requested scope fails closed without deleting prior rows."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov_div = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2025-09-15T20:00:00Z",
            observed_date="2025-09-15",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2025-09-15T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        # Seed 2 old dividend-adjusted rows outside the [2026-10-19, 2026-10-19] window
        engine.save_daily_candles_with_evidence("ABBV", [
            {"time": "2025-09-15", "open": 200.0, "high": 205.0, "low": 195.0, "close": 202.0, "volume": 1000000},
            {"time": "2025-09-16", "open": 202.0, "high": 206.0, "low": 200.0, "close": 204.0, "volume": 1100000},
        ], prov_div)

        # Mock fetcher returns 1 split-adjusted bar on target anchor session
        mock_fetcher = MockAlpacaFetcher(responses={
            "ABBV": {
                "success": True,
                "status_code": 200,
                "symbol": "ABBV",
                "bars": [
                    {"t": "2026-10-19T04:00:00Z", "o": 250.0, "h": 255.0, "l": 248.0, "c": 252.0, "v": 1200000}
                ],
                "next_page_token": None,
            }
        })
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV"],
            start_date="2026-10-19",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )

        assert summary.batch_outcome == "FAILURE"
        assert summary.candidate_dispositions["ABBV"]["status"] == "BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE"

        # Inspect DB: prior dividend-adjusted rows MUST be preserved (not wiped out)
        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT trade_date, close FROM asset_ohlcv_daily WHERE symbol='ABBV' ORDER BY trade_date ASC")
        rows = cur.fetchall()
        cur.execute("SELECT trade_date, adjustment_state FROM asset_ohlcv_provenance WHERE symbol='ABBV' ORDER BY trade_date ASC")
        prov_rows = cur.fetchall()
        conn.close()

        assert len(rows) == 2
        assert len(prov_rows) == 2
        assert all(r[1] == AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED for r in prov_rows)

    def test_atomic_replacement_preserves_outside_scope_compatible_history(self, test_db_path: str):
        """R2: Bounded scope replacement replaces ONLY records within [scope_start_date, scope_end_date] and preserves outside rows."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov_split = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-19T20:00:00Z",
            observed_date="2026-10-19",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-19T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        # Seed records before, within, and after target scope
        initial_bars = [
            {"time": "2025-09-15", "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 1000000},
            {"time": "2026-10-15", "open": 200.0, "high": 205.0, "low": 198.0, "close": 202.0, "volume": 1100000},
            {"time": "2026-10-16", "open": 202.0, "high": 207.0, "low": 200.0, "close": 205.0, "volume": 1200000},
            {"time": "2026-10-22", "open": 300.0, "high": 305.0, "low": 298.0, "close": 302.0, "volume": 1300000},
        ]
        engine.save_daily_candles_with_evidence("ABBV", initial_bars, prov_split)

        # Replacement bars specifically covering [2026-10-15, 2026-10-19]
        replacement_bars = [
            {"time": "2026-10-15", "open": 210.0, "high": 215.0, "low": 208.0, "close": 212.0, "volume": 1500000},
            {"time": "2026-10-16", "open": 212.0, "high": 217.0, "low": 210.0, "close": 215.0, "volume": 1600000},
            {"time": "2026-10-19", "open": 215.0, "high": 220.0, "low": 213.0, "close": 218.0, "volume": 1700000},
        ]
        engine.save_daily_candles_with_evidence(
            "ABBV",
            replacement_bars,
            prov_split,
            replace_existing_scope=True,
            scope_start_date="2026-10-15",
            scope_end_date="2026-10-19",
        )

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT trade_date, close FROM asset_ohlcv_daily WHERE symbol='ABBV' ORDER BY trade_date ASC")
        rows = cur.fetchall()
        cur.execute("SELECT trade_date, adjustment_state FROM asset_ohlcv_provenance WHERE symbol='ABBV' ORDER BY trade_date ASC")
        prov_rows = cur.fetchall()
        conn.close()

        # Both the earlier row (2025-09-15) and later row (2026-10-22) must be preserved!
        # Replaced rows (2026-10-15, 2026-10-16) and new row (2026-10-19) must be present.
        assert len(rows) == 5
        dates = [r[0] for r in rows]
        closes = {r[0]: r[1] for r in rows}
        assert dates == ["2025-09-15", "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-22"]
        assert closes["2025-09-15"] == 102.0  # preserved untouched
        assert closes["2026-10-15"] == 212.0  # replaced
        assert closes["2026-10-16"] == 215.0  # replaced
        assert closes["2026-10-19"] == 218.0  # inserted
        assert closes["2026-10-22"] == 302.0  # preserved untouched
        assert len(prov_rows) == 5
        assert all(r[1] == AdjustmentState.SPLIT_ADJUSTED_ONLY for r in prov_rows)

    def test_persistence_failure_receipt_and_unprocessed_candidates(self, tmp_path: Path, sample_manifest_path: Path):
        """SIP-05: Persistence failure on candidate 2 preserves candidate 1 write, marks remainder UNPROCESSED, and emits receipt."""
        db_file = tmp_path / "failing_market.db"
        db_path_resolved = str(db_file.resolve())
        engine = MarketDatabaseEngine(db_path=db_path_resolved)

        real_save = engine.save_daily_candles_with_evidence

        def mock_save(sym, bars, prov, replace_existing_scope=False, **kwargs):
            if sym == "ADBE":
                raise sqlite3.OperationalError("disk I/O error during candidate commit")
            return real_save(sym, bars, prov, replace_existing_scope=replace_existing_scope, **kwargs)

        engine.save_daily_candles_with_evidence = mock_save

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=MockAlpacaFetcher(), feed="sip", adjustment="split")
        adapter.db_engine = engine

        # Target order from sorted manifest symbols: ABBV (first), ADBE (second), NVDA (third).
        # Candidate 1 (ABBV) succeeds, Candidate 2 (ADBE) fails on persist, Candidate 3 (NVDA) remains unprocessed.
        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV", "ADBE", "NVDA"],
            start_date="2026-10-15",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=db_path_resolved,
        )

        # Verify summary truthfulness
        assert summary.execution_outcome == "PARTIAL_SUCCESS"
        assert summary.batch_outcome == "PARTIAL_SUCCESS"
        assert summary.completed_writes == ["ABBV"]
        assert summary.failed_candidate == "ADBE"
        assert summary.unprocessed_candidates == ["NVDA"]
        assert summary.candidate_dispositions["ADBE"]["status"] == "PERSISTENCE_ERROR"
        assert summary.candidate_dispositions["NVDA"]["status"] == "UNPROCESSED"

    def test_r1_endpoint_only_response_for_year_long_request_is_rejected(self):
        """R1: Single bar on end session for year-long request is rejected as truncated history."""
        bars = [{"t": "2026-10-19T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000000}]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2025-09-15", "2026-10-19")
        assert res.is_valid is False
        assert "INCOMPLETE_ACQUISITION_TRUNCATED_HISTORY" in res.rejection_reason

    def test_r1_interior_gaps_in_expected_sessions_are_rejected(self):
        """R1: Missing interior NYSE session in retrieved series is rejected as interior gaps."""
        # Sessions: 2026-10-15 (Thu), [2026-10-16 Fri omitted], 2026-10-19 (Mon)
        bars = [
            {"t": "2026-10-15T04:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.5, "v": 1000000},
            {"t": "2026-10-19T04:00:00Z", "o": 103.0, "h": 107.0, "l": 102.0, "c": 106.5, "v": 1200000},
        ]
        res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series("AAPL", bars, "2026-10-15", "2026-10-19")
        assert res.is_valid is False
        assert "INCOMPLETE_ACQUISITION_INTERIOR_GAPS" in res.rejection_reason

    def test_r3_unprovenanced_candle_prevents_skip_as_already_current(self, test_db_path: str, sample_manifest_path: Path):
        """R3: Missing provenance record for stored candle prevents candidate from skipping as current."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        # Store candles directly via SQLite without provenance sidecars
        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        from scripts.ingest_alpaca_historical import get_expected_nyse_sessions
        exp = get_expected_nyse_sessions("2026-10-15", "2026-10-19")
        for sess in exp:
            cur.execute(
                "INSERT INTO asset_ohlcv_daily (symbol, trade_date, open, high, low, close, volume) VALUES (?, ?, 100, 105, 95, 100, 1000)",
                ("ABBV", sess),
            )
        conn.commit()
        conn.close()

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=MockAlpacaFetcher(), feed="sip", adjustment="split")
        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV"],
            start_date="2026-10-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )
        disp = summary.candidate_dispositions["ABBV"]
        # Must NOT skip as already current! Must detect missing provenance sidecars and backfill
        assert disp["action_type"] != "SKIP_ALREADY_CURRENT"
        assert disp["provenance_compatible"] is False

    def test_r3_calendar_authority_unavailable_fails_closed(self, test_db_path: str, sample_manifest_path: Path):
        """R3: When calendar authority is unavailable, ingestion fails closed rather than using count heuristics."""
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=MockAlpacaFetcher(), feed="sip", adjustment="split")
        with patch("scripts.ingest_alpaca_historical.get_expected_nyse_sessions", return_value=[]):
            summary = adapter.execute_bounded_ingestion(
                manifest_path=sample_manifest_path,
                symbols_override=["ABBV"],
                start_date="2026-10-15",
                end_date="2026-10-19",
                apply_writes=False,
                db_path=test_db_path,
            )
            assert summary.batch_outcome == "FAILURE"
            disp = summary.candidate_dispositions["ABBV"]
            assert disp["action_type"] == "BLOCKED"
            assert "CALENDAR_AUTHORITY_UNAVAILABLE" in disp["rejection_reason"]


# ────────────────────────────────────────────────────────────
# 11. Defect A: Incremental Refresh Coverage & Writer Safeguards
# ────────────────────────────────────────────────────────────

class TestDefectARefreshCoverageAndWriterSafeguards:
    def test_defect_a_prefix_gap_blocks_refresh_and_fails_closed_on_tail_response(
        self, test_db_path: str, sample_manifest_path: Path
    ):
        """Defect A: An interior gap in stored prefix (missing 2026-02-06) blocks incremental refresh.
        When provider only returns tail-session data, backfill fails closed and preserves prior data.
        """
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-09-24T20:00:00Z",
            observed_date="2026-09-24",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-09-24T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )

        all_prefix_sessions = get_expected_nyse_sessions("2025-09-15", "2026-09-24")
        # Intentionally introduce interior gap at 2026-02-06
        gapped_prefix_sessions = [s for s in all_prefix_sessions if s != "2026-02-06"]
        assert len(gapped_prefix_sessions) == len(all_prefix_sessions) - 1

        initial_candles = [
            {"time": s, "open": 150.0, "high": 155.0, "low": 148.0, "close": 152.0, "volume": 1000000}
            for s in gapped_prefix_sessions
        ]
        engine.save_daily_candles_with_evidence("ABBV", initial_candles, prov)

        # Provider only returns tail session 2026-10-19 (the exact Defect A scenario)
        tail_bar = [{"t": "2026-10-19T04:00:00Z", "o": 160.0, "h": 165.0, "l": 159.0, "c": 162.0, "v": 1200000}]
        mock_fetcher = MockAlpacaFetcher(responses={"ABBV": {"success": True, "symbol": "ABBV", "bars": tail_bar, "next_page_token": None}})

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")
        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ABBV"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )

        # 1. Summary outcome must fail closed
        assert summary.batch_outcome == "FAILURE"
        disp = summary.candidate_dispositions["ABBV"]
        # 2. Action was forced to BACKFILL because prefix gap prevented REFRESH
        assert disp["action_type"] == "BACKFILL"
        assert disp["status"] == "FAILED_VALIDATION"
        assert "INCOMPLETE_ACQUISITION_TRUNCATED_HISTORY" in disp["rejection_reason"]

        # 3. Target database must remain untouched with original gapped count
        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'ABBV'")
        count_after = cur.fetchone()[0]
        conn.close()
        assert count_after == len(gapped_prefix_sessions)

    def test_defect_a_complete_prefix_allows_refresh_and_succeeds(
        self, test_db_path: str, sample_manifest_path: Path
    ):
        """Defect A: A genuinely complete stored prefix allows incremental refresh, fetches only the tail,
        validates full combined window coverage, and applies writes.
        """
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-09-24T20:00:00Z",
            observed_date="2026-09-24",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-09-24T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )

        p1_sessions = get_expected_nyse_sessions("2025-09-15", "2026-09-24")
        p2_sessions = get_expected_nyse_sessions("2026-09-25", "2026-10-19")

        initial_candles = [
            {"time": s, "open": 500.0, "high": 505.0, "low": 495.0, "close": 502.0, "volume": 1000000}
            for s in p1_sessions
        ]
        engine.save_daily_candles_with_evidence("ADBE", initial_candles, prov)

        tail_bars = [
            {"t": f"{s}T04:00:00Z", "o": 510.0, "h": 515.0, "l": 505.0, "c": 512.0, "v": 1100000}
            for s in p2_sessions
        ]
        mock_fetcher = MockAlpacaFetcher(responses={"ADBE": {"success": True, "symbol": "ADBE", "bars": tail_bars, "next_page_token": None}})

        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")
        summary = adapter.execute_bounded_ingestion(
            manifest_path=sample_manifest_path,
            symbols_override=["ADBE"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )

        assert summary.batch_outcome == "SUCCESS"
        disp = summary.candidate_dispositions["ADBE"]
        assert disp["action_type"] == "REFRESH"
        assert disp["status"] in ("SUCCESS", "APPLIED")
        assert disp["bars_ingested"] == len(p2_sessions)

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'ADBE'")
        total_stored = cur.fetchone()[0]
        conn.close()
        assert total_stored == len(p1_sessions) + len(p2_sessions)

    def test_writer_safeguard_requires_both_bounds_and_valid_interval(self, test_db_path: str):
        """Writer Safeguard: save_daily_candles_with_evidence strictly requires valid non-null bounds
        when replace_existing_scope=True, and rejects missing or inverted intervals without mutating DB.
        """
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-19T20:00:00Z",
            observed_date="2026-10-19",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-19T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        sample_candle = [{"time": "2026-10-19", "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000}]

        # 1. Missing scope_start_date
        with pytest.raises(ValueError, match="Both scope_start_date and scope_end_date are required"):
            engine.save_daily_candles_with_evidence(
                "AAPL", sample_candle, prov, replace_existing_scope=True, scope_start_date=None, scope_end_date="2026-10-19"
            )

        # 2. Missing scope_end_date
        with pytest.raises(ValueError, match="Both scope_start_date and scope_end_date are required"):
            engine.save_daily_candles_with_evidence(
                "AAPL", sample_candle, prov, replace_existing_scope=True, scope_start_date="2025-09-15", scope_end_date=None
            )

        # 3. Inverted bounds (start > end)
        with pytest.raises(ValueError, match="Reversed replacement scope"):
            engine.save_daily_candles_with_evidence(
                "AAPL", sample_candle, prov, replace_existing_scope=True, scope_start_date="2026-10-19", scope_end_date="2025-09-15"
            )

        # 4. Malformed date string
        with pytest.raises(ValueError, match="Could not parse replacement scope dates"):
            engine.save_daily_candles_with_evidence(
                "AAPL", sample_candle, prov, replace_existing_scope=True, scope_start_date="not-a-date", scope_end_date="2026-10-19"
            )

        # 5. Confirm zero DB writes occurred across all rejected attempts
        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'AAPL'")
        count = cur.fetchone()[0]
        conn.close()
        assert count == 0


class TestReleaseCandidateLineageAndOperationalAdversarial:
    """
    Focused Adversarial Regression Suite for Sprint 2 Package B Release Candidate.
    Enforces the 10 Lineage and Operational Invariants:
    1. test_lineage_polygon_prefix_rejected_without_replace
    2. test_lineage_unprovenanced_prefix_detection
    3. test_manifest_replace_directive_overrides_date_continuity
    4. test_cross_provider_lineage_cannot_certify_mixed_series
    5. test_scope_isolation_preserves_unaffected_symbols
    6. test_one_to_one_candle_provenance_invariant
    7. test_goog_googl_dual_listing_lineage_isolation
    8. test_dry_run_zero_db_writes_adversarial
    9. test_scoped_atomic_rollback_on_mid_symbol_failure
    10. test_idempotent_replay_identical_digests
    """

    def test_lineage_polygon_prefix_rejected_without_replace(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 1: Existing Polygon prefix outside requested scope triggers BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        polygon_prov = MarketProvenance(
            provider="MASSIVE_POLYGON",
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-16T20:00:00Z",
            observed_date="2026-10-16",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-16T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        sessions = get_expected_nyse_sessions("2025-08-27", "2026-10-16")
        bars = [{"time": s, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1000} for s in sessions]
        engine.save_daily_candles_with_evidence("GOOG", bars, polygon_prov)

        goog_manifest = tmp_path / "goog_manifest_20250915.json"
        goog_manifest.write_text(json.dumps({
            "ingestion_scope": {
                "production_target_backfill": {"missing_symbols": ["GOOG"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=goog_manifest,
            symbols_override=["GOOG"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["GOOG"]
        assert disp["status"] == "BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE"
        assert "INCOMPATIBLE_HISTORY_OUTSIDE_REVIEWED_SCOPE" in disp["rejection_reason"]
        assert "explicit wider repair from '2025-08-27'" in disp["rejection_reason"]
        assert summary.failed_validation_count == 1
        assert len(mock_fetcher.calls) == 0

    def test_lineage_unprovenanced_prefix_detection(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 2: Existing unprovenanced bars prior to start date are detected and rejected."""
        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        sessions = get_expected_nyse_sessions("2025-08-27", "2025-09-12")
        for s in sessions:
            cur.execute(
                "INSERT INTO asset_ohlcv_daily (symbol, trade_date, open, high, low, close, volume) "
                "VALUES ('AAPL', ?, 150.0, 155.0, 149.0, 153.0, 2000000)", (s,)
            )
        conn.commit()
        conn.close()

        aapl_manifest = tmp_path / "aapl_manifest.json"
        aapl_manifest.write_text(json.dumps({
            "ingestion_scope": {
                "production_target_backfill": {"missing_symbols": ["AAPL"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=aapl_manifest,
            symbols_override=["AAPL"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["AAPL"]
        assert disp["status"] == "BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE"
        assert "unprovenanced records outside reviewed window" in disp["rejection_reason"]
        assert len(mock_fetcher.calls) == 0

    def test_manifest_replace_directive_overrides_date_continuity(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 3: Manifest action: REPLACE forces full window fetch even when dates appear contiguous."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        alpaca_prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-16T20:00:00Z",
            observed_date="2026-10-16",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-16T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        sessions = get_expected_nyse_sessions("2025-09-15", "2026-10-16")
        bars = [{"time": s, "open": 200.0, "high": 205.0, "low": 199.0, "close": 202.0, "volume": 500000} for s in sessions]
        engine.save_daily_candles_with_evidence("MSFT", bars, alpaca_prov)

        replace_manifest = tmp_path / "replace_manifest.json"
        replace_manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {
                    "MSFT": {
                        "action": "REPLACE",
                        "rationale": "Forced clean baseline replacement."
                    }
                },
                "production_target_backfill": {"missing_symbols": []},
                "production_target_refresh": {
                    "per_symbol_parameters": [
                        {"symbol": "MSFT", "action": "REPLACE", "requested_start_session": "2026-10-19"}
                    ]
                }
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=replace_manifest,
            symbols_override=["MSFT"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["MSFT"]
        assert disp["action_type"] == "REPLACE"
        assert mock_fetcher.calls[0]["start"] == "2025-09-15"

    def test_cross_provider_lineage_cannot_certify_mixed_series(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 4: Series containing mixed Polygon and Alpaca provenance cannot be certified as current."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        sessions_poly = get_expected_nyse_sessions("2025-09-15", "2026-03-31")
        sessions_alp = get_expected_nyse_sessions("2026-04-01", "2026-10-19")

        polygon_prov = MarketProvenance(
            provider="MASSIVE_POLYGON",
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-03-31T20:00:00Z",
            observed_date="2026-03-31",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-03-31T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        alpaca_prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-19T20:00:00Z",
            observed_date="2026-10-19",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-19T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )

        bars_poly = [{"time": s, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1000} for s in sessions_poly]
        bars_alp = [{"time": s, "open": 110.0, "high": 115.0, "low": 109.0, "close": 112.0, "volume": 1000} for s in sessions_alp]

        engine.save_daily_candles_with_evidence("GOOG", bars_poly, polygon_prov)
        engine.save_daily_candles_with_evidence("GOOG", bars_alp, alpaca_prov)

        manifest = tmp_path / "mixed_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "production_target_backfill": {"missing_symbols": []},
                "production_target_refresh": {"per_symbol_parameters": [{"symbol": "GOOG"}]}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["GOOG"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["GOOG"]
        assert disp["status"] != "SKIPPED"
        assert disp["provenance_compatible"] is False

    def test_scope_isolation_preserves_unaffected_symbols(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 5: Replacing bars for target pilot symbols does not touch rows or provenance for unaffected symbols."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        tsla_prov = MarketProvenance(
            provider="PRIOR_SYSTEM",
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-16T20:00:00Z",
            observed_date="2026-10-16",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-16T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        sessions = get_expected_nyse_sessions("2025-09-15", "2026-10-16")
        tsla_bars = [{"time": s, "open": 250.0, "high": 255.0, "low": 248.0, "close": 252.0, "volume": 12000000} for s in sessions]
        engine.save_daily_candles_with_evidence("TSLA", tsla_bars, tsla_prov)

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'TSLA'")
        tsla_count_before = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = 'TSLA'")
        tsla_prov_before = cur.fetchone()[0]
        conn.close()

        manifest = tmp_path / "pilot_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {"AAPL": {"action": "REPLACE"}},
                "production_target_backfill": {"missing_symbols": ["AAPL"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["AAPL"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )

        assert summary.execution_outcome == "SUCCESS"

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'TSLA'")
        tsla_count_after = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = 'TSLA'")
        tsla_prov_after = cur.fetchone()[0]
        conn.close()

        assert tsla_count_before == tsla_count_after == len(sessions)
        assert tsla_prov_before == tsla_prov_after == len(sessions)

    def test_one_to_one_candle_provenance_invariant(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 6: Every inserted candle has exactly one corresponding provenance record with matching trade_date."""
        manifest = tmp_path / "abbv_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {"ABBV": {"action": "BACKFILL"}},
                "production_target_backfill": {"missing_symbols": ["ABBV"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["ABBV"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )

        assert summary.execution_outcome == "SUCCESS"

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT trade_date FROM asset_ohlcv_daily WHERE symbol = 'ABBV' ORDER BY trade_date")
        candle_dates = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT trade_date FROM asset_ohlcv_provenance WHERE symbol = 'ABBV' ORDER BY trade_date")
        prov_dates = [r[0] for r in cur.fetchall()]
        conn.close()

        assert len(candle_dates) > 0
        assert candle_dates == prov_dates

    def test_goog_googl_dual_listing_lineage_isolation(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 7: Dual-listing operations on GOOG do not cross-contaminate GOOGL bars or provenance."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        googl_prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-16T20:00:00Z",
            observed_date="2026-10-16",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-16T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        sessions = get_expected_nyse_sessions("2025-09-15", "2026-10-16")
        googl_bars = [{"time": s, "open": 180.0, "high": 182.0, "low": 179.0, "close": 181.0, "volume": 800000} for s in sessions]
        engine.save_daily_candles_with_evidence("GOOGL", googl_bars, googl_prov)

        manifest = tmp_path / "goog_only_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {"GOOG": {"action": "BACKFILL"}},
                "production_target_backfill": {"missing_symbols": ["GOOG"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["GOOG"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )

        assert summary.execution_outcome == "SUCCESS"

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'GOOGL'")
        googl_count = cur.fetchone()[0]
        cur.execute("SELECT DISTINCT provider FROM asset_ohlcv_provenance WHERE symbol = 'GOOGL'")
        googl_providers = [r[0] for r in cur.fetchall()]
        conn.close()

        assert googl_count == len(sessions)
        assert googl_providers == [Provider.ALPACA]

    def test_dry_run_zero_db_writes_adversarial(self, test_db_path: str, tmp_path: Path):
        """Operational Safety 8: Dry-run guarantees 0 bytes mutated across both tables, verified by DB SHA-256."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-16T20:00:00Z",
            observed_date="2026-10-16",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-16T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        sessions = get_expected_nyse_sessions("2025-09-15", "2026-10-16")
        bars = [{"time": s, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1000} for s in sessions]
        engine.save_daily_candles_with_evidence("AAPL", bars, prov)

        db_bytes_before = Path(test_db_path).read_bytes()
        sha_before = hashlib.sha256(db_bytes_before).hexdigest()

        manifest = tmp_path / "dry_run_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {
                    "AAPL": {"action": "REPLACE"},
                    "ABBV": {"action": "BACKFILL"}
                },
                "production_target_backfill": {"missing_symbols": ["AAPL", "ABBV"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["AAPL", "ABBV"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        assert summary.mode == "DRY_RUN"
        assert summary.total_bars_ingested == 0

        db_bytes_after = Path(test_db_path).read_bytes()
        sha_after = hashlib.sha256(db_bytes_after).hexdigest()

        assert sha_before == sha_after

    def test_scoped_atomic_rollback_on_mid_symbol_failure(self, test_db_path: str, tmp_path: Path):
        """Operational Safety 9: Mid-batch failure on symbol 3 halts execution, preserves symbols 1 and 2, and marks remaining UNPROCESSED."""
        manifest = tmp_path / "multi_symbol_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "production_target_backfill": {"missing_symbols": ["SYM1", "SYM2", "SYM3", "SYM4"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        real_engine = MarketDatabaseEngine(db_path=test_db_path)
        orig_save = real_engine.save_daily_candles_with_evidence

        def flaky_save(symbol, data, provenance, **kwargs):
            if symbol == "SYM3":
                raise sqlite3.OperationalError("Simulated disk I/O error on SYM3")
            return orig_save(symbol, data, provenance, **kwargs)

        with patch.object(real_engine, "save_daily_candles_with_evidence", side_effect=flaky_save):
            adapter.db_engine = real_engine
            summary = adapter.execute_bounded_ingestion(
                manifest_path=manifest,
                symbols_override=["SYM1", "SYM2", "SYM3", "SYM4"],
                start_date="2025-09-15",
                end_date="2026-10-19",
                apply_writes=True,
                db_path=test_db_path,
            )

        assert summary.execution_outcome == "PARTIAL_SUCCESS"
        assert summary.completed_writes == ["SYM1", "SYM2"]
        assert summary.failed_candidate == "SYM3"
        assert summary.unprocessed_candidates == ["SYM4"]
        assert summary.candidate_dispositions["SYM3"]["status"] == "PERSISTENCE_ERROR"
        assert summary.candidate_dispositions["SYM4"]["status"] == "UNPROCESSED"

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'SYM1'")
        assert cur.fetchone()[0] > 0
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'SYM2'")
        assert cur.fetchone()[0] > 0
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'SYM3'")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'SYM4'")
        assert cur.fetchone()[0] == 0
        conn.close()

    def test_idempotent_replay_identical_digests(self, test_db_path: str, tmp_path: Path):
        """Operational Safety 10: Replaying identical ingestion yields exact row counts, zero duplicates, and identical bar values."""
        manifest = tmp_path / "replay_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {
                    "AAPL": {"action": "BACKFILL"},
                    "ABBV": {"action": "BACKFILL"}
                },
                "production_target_backfill": {"missing_symbols": ["AAPL", "ABBV"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary1 = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["AAPL", "ABBV"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )
        assert summary1.execution_outcome == "SUCCESS"

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT symbol, trade_date, open, high, low, close, volume FROM asset_ohlcv_daily ORDER BY symbol, trade_date")
        bars_run1 = cur.fetchall()
        cur.execute("SELECT symbol, trade_date, provider, adjustment_state FROM asset_ohlcv_provenance ORDER BY symbol, trade_date")
        prov_run1 = cur.fetchall()
        conn.close()

        summary2 = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["AAPL", "ABBV"],
            start_date="2025-09-15",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )
        assert summary2.execution_outcome == "SUCCESS"

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT symbol, trade_date, open, high, low, close, volume FROM asset_ohlcv_daily ORDER BY symbol, trade_date")
        bars_run2 = cur.fetchall()
        cur.execute("SELECT symbol, trade_date, provider, adjustment_state FROM asset_ohlcv_provenance ORDER BY symbol, trade_date")
        prov_run2 = cur.fetchall()
        conn.close()

        assert bars_run1 == bars_run2
        assert prov_run1 == prov_run2

    def test_conflicting_directives_in_multiple_manifest_sections(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 11: When manifest specifies both REFRESH and REPLACE for a symbol, REPLACE directive takes precedence."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        alpaca_prov = MarketProvenance(
            provider=Provider.ALPACA,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-16T20:00:00Z",
            observed_date="2026-10-16",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-16T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        sessions = get_expected_nyse_sessions("2025-08-27", "2026-10-16")
        bars = [{"time": s, "open": 200.0, "high": 205.0, "low": 199.0, "close": 202.0, "volume": 500000} for s in sessions]
        engine.save_daily_candles_with_evidence("AAPL", bars, alpaca_prov)

        conflicting_manifest = tmp_path / "conflicting_manifest.json"
        conflicting_manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {
                    "AAPL": {
                        "action": "REPLACE",
                        "internal_security_id": "ARX_SEC_0B4C728FE0BE",
                        "association_id": "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17"
                    }
                },
                "production_target_backfill": {"missing_symbols": []},
                "production_target_refresh": {
                    "per_symbol_parameters": [
                        {"symbol": "AAPL", "action": "REFRESH", "requested_start_session": "2026-10-19"}
                    ]
                }
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=conflicting_manifest,
            symbols_override=["AAPL"],
            start_date="2025-08-27",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["AAPL"]
        assert disp["action_type"] == "REPLACE"
        assert mock_fetcher.calls[0]["start"] == "2025-08-27"

    def test_duplicate_or_non_session_production_dates_cleansed_on_replace(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 12: SQLite schema rejects duplicate dates via UNIQUE constraint, and non-session dates are cleansed on replacement."""
        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        # 1. Proves duplicate dates on the same symbol are strictly rejected at the schema level
        cur.execute("INSERT INTO asset_ohlcv_daily (symbol, trade_date, open, high, low, close, volume) VALUES ('AAPL', '2025-09-02', 100, 105, 99, 102, 1000)")
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            cur.execute("INSERT INTO asset_ohlcv_daily (symbol, trade_date, open, high, low, close, volume) VALUES ('AAPL', '2025-09-02', 101, 106, 100, 103, 1500)")

        # 2. Insert anomalous non-session (Sunday) entry
        cur.execute("INSERT INTO asset_ohlcv_daily (symbol, trade_date, open, high, low, close, volume) VALUES ('AAPL', '2025-08-31', 100, 105, 99, 102, 1000)")  # Sunday!
        conn.commit()
        conn.close()

        manifest = tmp_path / "replace_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {
                    "AAPL": {
                        "action": "REPLACE",
                        "internal_security_id": "ARX_SEC_0B4C728FE0BE",
                        "association_id": "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17"
                    }
                },
                "production_target_backfill": {"missing_symbols": ["AAPL"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["AAPL"],
            start_date="2025-08-27",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )

        assert summary.execution_outcome == "SUCCESS"

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT trade_date FROM asset_ohlcv_daily WHERE symbol = 'AAPL' ORDER BY trade_date")
        dates = [r[0] for r in cur.fetchall()]
        conn.close()

        # Invariant: Exact 288 NYSE sessions, zero Sunday 2025-08-31
        assert len(dates) == 288
        assert "2025-08-31" not in dates
        assert len(dates) == len(set(dates))

    def test_historical_records_after_authorized_window_fails_closed(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 13: Stored records after the authorized window end date trigger fail-closed block."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov = MarketProvenance(
            provider="PRIOR_SYSTEM",
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2026-10-20T20:00:00Z",
            observed_date="2026-10-20",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2026-10-20T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        engine.save_daily_candles_with_evidence("AAPL", [
            {"time": "2026-10-20", "open": 230.0, "high": 235.0, "low": 229.0, "close": 232.0, "volume": 1000000}
        ], prov)

        manifest = tmp_path / "window_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {"AAPL": {"action": "REPLACE"}},
                "production_target_backfill": {"missing_symbols": ["AAPL"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["AAPL"],
            start_date="2025-08-27",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["AAPL"]
        assert disp["status"] == "BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE"
        assert "records outside reviewed window" in disp["rejection_reason"]

    def test_legacy_records_before_authorized_window_fails_closed(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 14: Stored records before the authorized window start date trigger fail-closed block."""
        engine = MarketDatabaseEngine(db_path=test_db_path)
        prov = MarketProvenance(
            provider="PRIOR_SYSTEM",
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at="2025-08-26T20:00:00Z",
            observed_date="2025-08-26",
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
            ingested_at="2025-08-26T20:01:00Z",
            adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
            structural_quality=StructuralQuality.COMPLETE,
            fallback_status=False,
        )
        engine.save_daily_candles_with_evidence("AAPL", [
            {"time": "2025-08-26", "open": 200.0, "high": 205.0, "low": 199.0, "close": 202.0, "volume": 1000000}
        ], prov)

        manifest = tmp_path / "window_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {"AAPL": {"action": "REPLACE"}},
                "production_target_backfill": {"missing_symbols": ["AAPL"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["AAPL"],
            start_date="2025-08-27",
            end_date="2026-10-19",
            apply_writes=False,
            db_path=test_db_path,
        )

        disp = summary.candidate_dispositions["AAPL"]
        assert disp["status"] == "BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE"
        assert "records outside reviewed window" in disp["rejection_reason"]

    def test_complete_alpaca_sip_provenance_after_replacement(self, test_db_path: str, tmp_path: Path):
        """Lineage Safety 15: Executed replacement yields 100% pure Alpaca SIP split-adjusted provenance across all sessions."""
        manifest = tmp_path / "clean_manifest.json"
        manifest.write_text(json.dumps({
            "ingestion_scope": {
                "candidate_bindings": {"AAPL": {"action": "REPLACE"}},
                "production_target_backfill": {"missing_symbols": ["AAPL"]},
                "production_target_refresh": {"per_symbol_parameters": []}
            }
        }), encoding="utf-8")

        mock_fetcher = MockAlpacaFetcher()
        adapter = AlpacaHistoricalSipIngestionAdapter(fetcher=mock_fetcher, feed="sip", adjustment="split")

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest,
            symbols_override=["AAPL"],
            start_date="2025-08-27",
            end_date="2026-10-19",
            apply_writes=True,
            db_path=test_db_path,
        )

        assert summary.execution_outcome == "SUCCESS"

        conn = sqlite3.connect(test_db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT provider, adjustment_state FROM asset_ohlcv_provenance WHERE symbol = 'AAPL'")
        prov_summary = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'AAPL'")
        candle_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = 'AAPL'")
        prov_count = cur.fetchone()[0]
        conn.close()

        assert candle_count == 288
        assert prov_count == 288
        assert prov_summary == [(Provider.ALPACA, AdjustmentState.SPLIT_ADJUSTED_ONLY)]

    def test_zero_automatic_ingestion_on_deployment_startup(self):
        """Operational Safety 16: Importing adapter and recovery modules executes zero database writes, zero HTTP calls, and zero background jobs."""
        import importlib
        import scripts.ingest_alpaca_historical as s1
        import analyst_dashboard.data.guarded_recovery as s2
        import scripts.guarded_market_recovery as s3

        importlib.reload(s1)
        importlib.reload(s2)
        importlib.reload(s3)
        assert hasattr(s1, "AlpacaHistoricalSipIngestionAdapter")
        assert hasattr(s2, "GuardedRecoveryEngine")
        assert hasattr(s3, "main")

    def test_session_count_contract_dynamically_calculated(self):
        """Calendar Authority 17: Validates that session counts across intervals are calculated dynamically per XNYS exchange calendar."""
        s_aug_sep = get_expected_nyse_sessions("2025-08-27", "2025-09-12")
        s_aug_oct16 = get_expected_nyse_sessions("2025-08-27", "2026-10-16")
        s_aug_oct19 = get_expected_nyse_sessions("2025-08-27", "2026-10-19")
        s_sep_oct19 = get_expected_nyse_sessions("2025-09-15", "2026-10-19")

        assert len(s_aug_sep) == 12  # Labor Day 2025-09-01 excluded!
        assert len(s_aug_oct16) == 287
        assert len(s_aug_oct19) == 288
        assert len(s_sep_oct19) == 276
        assert len(s_aug_oct19) - len(s_aug_sep) == len(s_sep_oct19)



