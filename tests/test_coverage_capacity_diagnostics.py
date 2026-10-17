"""
tests/test_coverage_capacity_diagnostics.py

Authoritative verification for Sprint 2 coverage and capacity diagnostics.

Marked tier2c (Provenance and Data Gate).

Coverage:
1. Pure read-only market store safety:
   - Missing path never creates a file (FileNotFoundError).
   - Read-only connection (?mode=ro) strictly rejects write/DDL attempts.
   - Missing table validation.
2. Complete 7-tier disjoint primary outcome precedence:
   - INSPECTION_FAILED > AMBIGUOUS_IDENTITY > MISSING_MARKET_DATA >
     INSUFFICIENT_HISTORY > INVALID_OHLCV > STALE_QUOTE > DATA_READY.
3. Dual rule evaluation:
   - Builder rule: has_price and candles >= 50.
   - Session guard rule: target_session date match, candles >= 50, valid OHLCV.
4. Input observations vs distinct lookup keys vs resolved identities:
   - Multi-feed duplicate handling.
   - Preserved locators and hashes.
   - Venue conflict detection.
   - Explicit NOT_ESTABLISHED security identity status.
5. Strict accounting closure and zero-population edge case handling.
6. Target session date context sensitivity.
7. Bounded capacity benchmarking side-effect-free execution.
8. Offline raw feed parsing into DiagnosticInputRecords.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from analyst_dashboard.universe.coverage_diagnostics import (
    MIN_REQUIRED_CANDLES,
    BuildReadinessOutcome,
    CapacityBenchmarker,
    CoverageAccountingSummary,
    CoverageDiagnosticEngine,
    DiagnosticInputRecord,
    ObservationDiagnosticResult,
    PrimaryDiagnosticOutcome,
    ReadOnlyMarketStore,
    SessionReadinessOutcome,
    SymbolDiagnosticResult,
    load_acquisition_artifacts_to_records,
)

pytestmark = pytest.mark.tier2c


# ─── Test Fixtures & Synthetic Store Setup ─────────────────────────────────────

@pytest.fixture
def synthetic_market_db(tmp_path) -> str:
    """Creates a temporary SQLite database populated with deterministic test data."""
    db_file = tmp_path / "test_market_store.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute("""
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
    cursor.execute("CREATE TABLE asset_factor_snapshots (symbol TEXT, factor_date TEXT)")

    from datetime import date, timedelta

    target_d = date(2026, 10, 16)

    # 1. READY_SYM: 60 candles ending on 2026-10-16, positive OHLCV, positive volume
    ready_dates = [(target_d - timedelta(days=59 - i)).isoformat() for i in range(60)]
    for i, dt_str in enumerate(ready_dates):
        cursor.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("READY_SYM", dt_str, 100.0 + i, 105.0 + i, 99.0 + i, 102.0 + i, 1000000 + i * 1000),
        )

    # 2. STALE_SYM: 60 candles ending on 2026-10-10 (stale for 2026-10-16)
    stale_d = date(2026, 10, 10)
    stale_dates = [(stale_d - timedelta(days=59 - i)).isoformat() for i in range(60)]
    for i, dt_str in enumerate(stale_dates):
        cursor.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("STALE_SYM", dt_str, 50.0 + i, 52.0 + i, 49.0 + i, 51.0 + i, 500000),
        )

    # 3. INSUFF_SYM: 20 candles ending on 2026-10-16 (< 50 MIN_REQUIRED_CANDLES)
    insuff_dates = [(target_d - timedelta(days=19 - i)).isoformat() for i in range(20)]
    for i, dt_str in enumerate(insuff_dates):
        cursor.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("INSUFF_SYM", dt_str, 20.0 + i, 21.0 + i, 19.5 + i, 20.5 + i, 200000),
        )

    # 4. BAD_OHLC_SYM: 60 candles ending on 2026-10-16, but latest close is negative
    bad_ohlc_dates = [(target_d - timedelta(days=59 - i)).isoformat() for i in range(60)]
    for i, dt_str in enumerate(bad_ohlc_dates):
        close_val = -10.0 if i == 59 else 30.0
        cursor.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("BAD_OHLC_SYM", dt_str, 30.0, 31.0, 29.0, close_val, 100000),
        )

    # 5. BAD_VOL_SYM: 60 candles ending on 2026-10-16, but latest volume is negative
    bad_vol_dates = [(target_d - timedelta(days=59 - i)).isoformat() for i in range(60)]
    for i, dt_str in enumerate(bad_vol_dates):
        vol_val = -500 if i == 59 else 50000
        cursor.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("BAD_VOL_SYM", dt_str, 40.0, 42.0, 39.0, 41.0, vol_val),
        )

    conn.commit()
    conn.close()
    return str(db_file)


# ─── 1. Read-Only Store Invariants & Safety ────────────────────────────────────

def test_read_only_store_missing_file_fails_closed(tmp_path):
    """ReadOnlyMarketStore must raise FileNotFoundError and NEVER create a missing file."""
    non_existent = tmp_path / "does_not_exist.db"
    assert not non_existent.exists()

    with pytest.raises(FileNotFoundError, match="Store path does not exist"):
        ReadOnlyMarketStore(str(non_existent))

    # Invariant: File must NOT have been created
    assert not non_existent.exists()


def test_read_only_store_immutability(synthetic_market_db):
    """ReadOnlyMarketStore operates strictly in mode=ro and rejects all write operations."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    uri = store._get_uri()
    assert "mode=ro" in uri

    # Verify that a write attempt on the URI connection fails closed
    conn = sqlite3.connect(uri, uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO asset_ohlcv_daily VALUES ('ILLEGAL', '2026-10-16', 10, 11, 9, 10, 100)"
            )
            conn.commit()
    finally:
        conn.close()


def test_read_only_store_missing_table_fails_validation(tmp_path):
    """ReadOnlyMarketStore must reject databases lacking the required schema."""
    empty_db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(empty_db))
    conn.execute("CREATE TABLE some_other_table (id INT)")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="missing 'asset_ohlcv_daily' table"):
        ReadOnlyMarketStore(str(empty_db))


def test_read_only_store_getters_and_metadata(synthetic_market_db):
    """Verify price, candle, symbol listing, and metadata retrieval."""
    store = ReadOnlyMarketStore(synthetic_market_db)

    # get_latest_price
    price_info = store.get_latest_price("READY_SYM")
    assert price_info is not None
    assert price_info["symbol"] == "READY_SYM"
    assert price_info["date"] == "2026-10-16"
    assert price_info["currentPrice"] > 0

    # get_daily_candles
    candles = store.get_daily_candles("READY_SYM", limit=60)
    assert len(candles) == 60
    assert candles[-1]["time"] == "2026-10-16"
    assert candles[-1]["close"] > 0

    # get_all_stored_symbols
    symbols = store.get_all_stored_symbols()
    assert set(symbols) == {"READY_SYM", "STALE_SYM", "INSUFF_SYM", "BAD_OHLC_SYM", "BAD_VOL_SYM"}

    # get_store_metadata
    meta = store.get_store_metadata()
    assert meta["distinct_symbols_count"] == 5
    assert meta["total_candles"] == 60 + 60 + 20 + 60 + 60


# ─── 2. Outcome Precedence & Disjoint Rules ────────────────────────────────────

def test_outcome_data_ready(synthetic_market_db):
    """Symbol with target session date, >= 50 candles, and valid OHLCV is DATA_READY."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    primary, b_read, s_read, details = engine.evaluate_symbol("READY_SYM")
    assert primary == PrimaryDiagnosticOutcome.DATA_READY
    assert b_read == BuildReadinessOutcome.DATA_READY
    assert s_read == SessionReadinessOutcome.SESSION_READY
    assert details["candle_count"] == 60
    assert details["has_valid_ohlcv"] is True


def test_outcome_stale_quote_precedence(synthetic_market_db):
    """
    Symbol with sufficient history and valid OHLCV but outdated price date is STALE_QUOTE.
    Evaluates:
      - Universe Builder considers it DATA_READY (builder has no session date rule).
      - Session Guard considers it STALE_SESSION.
    """
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    primary, b_read, s_read, details = engine.evaluate_symbol("STALE_SYM")
    assert primary == PrimaryDiagnosticOutcome.STALE_QUOTE
    assert b_read == BuildReadinessOutcome.DATA_READY
    assert s_read == SessionReadinessOutcome.STALE_SESSION
    assert details["price_date"] == "2026-10-10"


def test_outcome_insufficient_history(synthetic_market_db):
    """Symbol with < 50 candles is INSUFFICIENT_HISTORY across all evaluations."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    primary, b_read, s_read, details = engine.evaluate_symbol("INSUFF_SYM")
    assert primary == PrimaryDiagnosticOutcome.INSUFFICIENT_HISTORY
    assert b_read == BuildReadinessOutcome.DATA_UNAVAILABLE
    assert s_read == SessionReadinessOutcome.INSUFFICIENT_HISTORY
    assert details["candle_count"] == 20
    assert details["candle_count"] < MIN_REQUIRED_CANDLES


def test_outcome_invalid_ohlcv(synthetic_market_db):
    """Symbol with invalid prices or volume triggers INVALID_OHLCV."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    # Negative close: triggers INVALID_OHLCV for primary; Session Guard marks MISSING_PRICE (due to price <= 0)
    primary, _, s_read, _ = engine.evaluate_symbol("BAD_OHLC_SYM")
    assert primary == PrimaryDiagnosticOutcome.INVALID_OHLCV
    assert s_read == SessionReadinessOutcome.MISSING_PRICE

    # Negative volume: price is valid positive, but OHLCV integrity fails -> INVALID_OHLCV for both
    primary_v, _, s_read_v, _ = engine.evaluate_symbol("BAD_VOL_SYM")
    assert primary_v == PrimaryDiagnosticOutcome.INVALID_OHLCV
    assert s_read_v == SessionReadinessOutcome.INVALID_OHLCV


def test_outcome_missing_market_data(synthetic_market_db):
    """Symbol not found in store produces MISSING_MARKET_DATA."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    primary, b_read, s_read, details = engine.evaluate_symbol("UNSTORED_TICKER")
    assert primary == PrimaryDiagnosticOutcome.MISSING_MARKET_DATA
    assert b_read == BuildReadinessOutcome.DATA_UNAVAILABLE
    assert s_read == SessionReadinessOutcome.MISSING_PRICE
    assert details["candle_count"] == 0


def test_outcome_ambiguous_identity(synthetic_market_db):
    """Empty or whitespace symbol produces AMBIGUOUS_IDENTITY."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    for empty_sym in ["", "   ", "\t"]:
        primary, b_read, s_read, _ = engine.evaluate_symbol(empty_sym)
        assert primary == PrimaryDiagnosticOutcome.AMBIGUOUS_IDENTITY
        assert b_read == BuildReadinessOutcome.DATA_UNRESOLVED
        assert s_read == SessionReadinessOutcome.UNAVAILABLE


def test_outcome_inspection_failed_on_exception(synthetic_market_db):
    """Store query exception produces INSPECTION_FAILED."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    with patch.object(store, "get_latest_price", side_effect=sqlite3.DatabaseError("Disk I/O error")):
        primary, b_read, s_read, details = engine.evaluate_symbol("READY_SYM")
        assert primary == PrimaryDiagnosticOutcome.INSPECTION_FAILED
        assert b_read == BuildReadinessOutcome.DATA_UNRESOLVED
        assert s_read == SessionReadinessOutcome.UNAVAILABLE
        assert "Store query failed" in details["detail"]


# ─── 3. Target Session Sensitivity ─────────────────────────────────────────────

def test_target_session_date_sensitivity(synthetic_market_db):
    """The same stored state evaluates to DATA_READY for matched session, STALE_QUOTE for mismatched."""
    store = ReadOnlyMarketStore(synthetic_market_db)

    # Matched session
    engine_today = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")
    p_today, _, s_today, _ = engine_today.evaluate_symbol("READY_SYM")
    assert p_today == PrimaryDiagnosticOutcome.DATA_READY
    assert s_today == SessionReadinessOutcome.SESSION_READY

    # Next session
    engine_tomorrow = CoverageDiagnosticEngine(store, target_session_date="2026-10-17")
    p_tomorrow, _, s_tomorrow, _ = engine_tomorrow.evaluate_symbol("READY_SYM")
    assert p_tomorrow == PrimaryDiagnosticOutcome.STALE_QUOTE
    assert s_tomorrow == SessionReadinessOutcome.STALE_SESSION


# ─── 4. Full Diagnostics Run & Accounting Closure ─────────────────────────────

def test_run_diagnostics_accounting_closure(synthetic_market_db):
    """
    Evaluates a heterogeneous population of observations and verifies:
    1. Complete accounting closure: disjoint outcomes sum to distinct lookup keys.
    2. Builder readiness partition sums to distinct lookup keys.
    3. Proper mapping from observations to symbols.
    4. Explicit NOT_ESTABLISHED security identity status.
    """
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    records = [
        # Observation 1: READY_SYM from SEC
        DiagnosticInputRecord(
            record_locator="sec#1",
            original_symbol="READY_SYM",
            venue_context="NASDAQ",
            source_artifact="sec.json",
        ),
        # Observation 2: READY_SYM from NASDAQ (duplicate ticker, same venue)
        DiagnosticInputRecord(
            record_locator="nasdaq#1",
            original_symbol="READY_SYM",
            venue_context="NASDAQ",
            source_artifact="nasdaq.txt",
        ),
        # Observation 3: STALE_SYM from SEC (venue conflict test)
        DiagnosticInputRecord(
            record_locator="sec#2",
            original_symbol="STALE_SYM",
            venue_context="NYSE",
            source_artifact="sec.json",
        ),
        # Observation 4: STALE_SYM from Other (venue conflict test)
        DiagnosticInputRecord(
            record_locator="other#1",
            original_symbol="STALE_SYM",
            venue_context="ARCA",
            source_artifact="other.txt",
        ),
        # Observation 5: INSUFF_SYM
        DiagnosticInputRecord(
            record_locator="sec#3",
            original_symbol="INSUFF_SYM",
            venue_context="NASDAQ",
            source_artifact="sec.json",
        ),
        # Observation 6: BAD_OHLC_SYM
        DiagnosticInputRecord(
            record_locator="sec#4",
            original_symbol="BAD_OHLC_SYM",
            venue_context="NASDAQ",
            source_artifact="sec.json",
        ),
        # Observation 7: BAD_VOL_SYM
        DiagnosticInputRecord(
            record_locator="sec#5",
            original_symbol="BAD_VOL_SYM",
            venue_context="NASDAQ",
            source_artifact="sec.json",
        ),
        # Observation 8: MISSING_SYM (not in store)
        DiagnosticInputRecord(
            record_locator="sec#6",
            original_symbol="MISSING_SYM",
            venue_context="NASDAQ",
            source_artifact="sec.json",
        ),
        # Observation 9: Empty symbol
        DiagnosticInputRecord(
            record_locator="sec#7",
            original_symbol="   ",
            venue_context="UNKNOWN",
            source_artifact="sec.json",
        ),
    ]

    obs_results, symbol_results, summary = engine.run_diagnostics(records)

    # Observation count vs distinct keys
    assert len(obs_results) == 9
    assert len(symbol_results) == 7  # READY_SYM and STALE_SYM duplicated
    assert summary.total_input_observations == 9
    assert summary.distinct_lookup_keys == 7

    # Closure
    assert summary.accounting_closed is True
    assert "Accounting closure VERIFIED" in summary.reconciliation_message

    # Verified counts across distinct keys:
    # 1 DATA_READY (READY_SYM)
    # 1 STALE_QUOTE (STALE_SYM)
    # 1 INSUFFICIENT_HISTORY (INSUFF_SYM)
    # 2 INVALID_OHLCV (BAD_OHLC_SYM, BAD_VOL_SYM)
    # 1 MISSING_MARKET_DATA (MISSING_SYM)
    # 1 AMBIGUOUS_IDENTITY ("")
    # 0 INSPECTION_FAILED
    # Total = 1 + 1 + 1 + 2 + 1 + 1 + 0 = 7
    assert summary.data_ready_count == 1
    assert summary.stale_quote_count == 1
    assert summary.insufficient_history_count == 1
    assert summary.invalid_ohlcv_count == 2
    assert summary.missing_market_data_count == 1
    assert summary.ambiguous_identity_count == 1
    assert summary.inspection_failed_count == 0

    # Successful lookups = distinct (7) - missing (1) - ambig (1) - failed (0) = 5
    assert summary.successful_lookups == 5
    assert summary.coverage_ratio == round(5 / 7, 6)

    # Builder readiness:
    # Build ready: READY_SYM, STALE_SYM, BAD_VOL_SYM = 3
    # Build unavailable: BAD_OHLC_SYM, INSUFF_SYM, MISSING_SYM = 3
    # Build unresolved: "" (ambiguous) = 1
    assert summary.build_ready_count == 3
    assert summary.build_unavailable_count == 3
    assert summary.build_unresolved_count == 1
    assert summary.build_ready_count + summary.build_unavailable_count + summary.build_unresolved_count == 7

    # Identity boundary assertion: NOT_ESTABLISHED
    assert summary.resolved_securities_count == 0
    assert summary.resolved_securities_status == "NOT_ESTABLISHED"

    # Venue conflict assertion on STALE_SYM
    stale_res = symbol_results["STALE_SYM"]
    assert stale_res.has_venue_conflict is True
    assert set(stale_res.venues) == {"NYSE", "ARCA"}

    # No venue conflict on READY_SYM
    ready_res = symbol_results["READY_SYM"]
    assert ready_res.has_venue_conflict is False
    assert ready_res.venues == ["NASDAQ"]


def test_run_diagnostics_empty_population(synthetic_market_db):
    """Zero input records gracefully yields 0% coverage and closed accounting."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")

    obs_results, symbol_results, summary = engine.run_diagnostics([])

    assert len(obs_results) == 0
    assert len(symbol_results) == 0
    assert summary.total_input_observations == 0
    assert summary.distinct_lookup_keys == 0
    assert summary.coverage_ratio == 0.0
    assert summary.ready_ratio == 0.0
    assert summary.accounting_closed is True


# ─── 5. Bounded Capacity Benchmarking ──────────────────────────────────────────

def test_capacity_benchmarker_bounded_execution(synthetic_market_db):
    """CapacityBenchmarker runs side-effect-free measurements and produces report."""
    store = ReadOnlyMarketStore(synthetic_market_db)
    benchmarker = CapacityBenchmarker(store, target_session_date="2026-10-16")

    report = benchmarker.run_benchmarks(sample_symbols=["READY_SYM", "STALE_SYM"])

    assert report.target_session_date == "2026-10-16"
    assert len(report.measurements) == 3

    # Store measurements (60 and 252 bars)
    m_store_60 = report.measurements[0]
    assert m_store_60.operation_name == "STORE_QUERY_PRICE_AND_CANDLES_60"
    assert m_store_60.throughput_ops_per_second > 0
    assert m_store_60.mean_latency_ms >= 0

    m_store_252 = report.measurements[1]
    assert m_store_252.operation_name == "STORE_QUERY_PRICE_AND_CANDLES_252"
    assert m_store_252.throughput_ops_per_second > 0
    assert m_store_252.mean_latency_ms >= 0

    # Rule evaluation measurement
    m_eval = report.measurements[2]
    assert m_eval.operation_name == "IN_MEMORY_READINESS_RULE_CHECK"
    assert m_eval.throughput_ops_per_second > 0

    # Explicit disclosure of unmeasured assessment throughput
    assert report.assessment_throughput_status == "NOT_MEASURED"

    # Hardware and omitted disclosures
    assert "platform" in report.hardware_runtime_context
    assert len(report.omitted_costs_and_limitations) >= 5

    # Serializable to dict/JSON
    rep_dict = report.to_dict()
    assert json.dumps(rep_dict) is not None


# ─── 6. Raw Feed Artifact Loading ──────────────────────────────────────────────

def test_load_acquisition_artifacts_to_records_offline(tmp_path):
    """Offline parsing loads raw artifacts into DiagnosticInputRecords without synthetic IDs."""
    # Create synthetic SEC file
    sec_data = {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [
            [320193, "Apple Inc.", "AAPL", "Nasdaq"],
            [789019, "Microsoft Corp", "MSFT", "Nasdaq"],
        ]
    }
    sec_file = tmp_path / "sec_company_tickers_exchange.json"
    sec_file.write_text(json.dumps(sec_data), encoding="utf-8")

    # Create synthetic Nasdaq Listed file
    nasdaq_txt = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
        "File Creation Time: 1016202607:00|||||||\n"
    )
    nasdaq_file = tmp_path / "nasdaq_listed_directory.txt"
    nasdaq_file.write_text(nasdaq_txt, encoding="utf-8")

    # Create synthetic Other Listed file
    other_txt = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "SPY|SPDR S&P 500 ETF Trust|A|SPY|Y|100|N|SPY\n"
        "File Creation Time: 1016202607:00|||||||\n"
    )
    other_file = tmp_path / "other_listed_directory.txt"
    other_file.write_text(other_txt, encoding="utf-8")

    records, meta = load_acquisition_artifacts_to_records(str(tmp_path))

    # Total rows: 2 (SEC) + 1 (Nasdaq) + 1 (Other) = 4
    assert len(records) == 4
    assert "sec" in meta
    assert "nasdaq" in meta
    assert "other_listed" in meta

    symbols = [r.original_symbol for r in records]
    assert symbols == ["AAPL", "MSFT", "AAPL", "SPY"]

    # Verify no synthetic IDs were synthesized
    for r in records:
        assert r.record_locator != ""
        assert r.source_hash != ""
        assert "security_id" not in r.supplied_identifiers
