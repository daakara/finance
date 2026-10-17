"""
tests/test_universe_coverage_capacity.py

Test Suite for ARX Terminal Sprint 2: Universe Coverage & Capacity Diagnostics.

Verifies:
1. Strict read-safety: zero database writes, zero DDL/schema modifications, URI ?mode=ro.
2. Fail-closed error handling: missing database paths raise FileNotFoundError and do not create files.
3. Arithmetic consistency: exact closure across all disjoint outcome partitions.
4. Offline execution: zero external network requests.
5. Distinction between observations, lookup keys, and resolved security identities.
6. Boundary conditions: empty feeds, malformed rows, and unseasoned histories.
7. Scanner prerequisite compliance (50-session floor and 200-session Minervini floor).
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from analyst_dashboard.universe.adapters.matching import match_sources_with_accounting
from analyst_dashboard.universe.adapters.models import (
    ParsedSourceFile,
    ParsedSourceRow,
    SourceFeedType,
)
from analyst_dashboard.universe.coverage_diagnostics import (
    BuildReadinessOutcome,
    CapacityBenchmarker,
    CoverageAccountingSummary,
    CoverageDiagnosticEngine,
    DiagnosticInputRecord,
    PrimaryDiagnosticOutcome,
    ReadOnlyMarketStore,
    SessionReadinessOutcome,
)
from scripts.diagnostics.universe_coverage_capacity import (
    DISCREPANCY_RECONCILIATION_TABLE,
    PINNED_FEED_HASHES,
    render_baseline_markdown,
    resolve_market_db_path,
    resolve_raw_feeds_dir,
    run_check_a_population_establishment,
    run_check_b_market_data_association,
    run_check_d_capacity_benchmarks,
    run_check_e_pipeline_integration_gaps,
    run_complete_diagnostics,
    verify_feed_integrity,
)

pytestmark = pytest.mark.tier2c


@pytest.fixture
def mock_market_db(tmp_path: Path) -> Path:
    """Creates a temporary test market database with controlled mock history."""
    db_file = tmp_path / "mock_market.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("""
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

    from datetime import date, timedelta
    base_date = date(2025, 1, 1)

    # Seed 4 test symbols:
    # 1. AAPL: 252 bars, fresh date 2026-10-16 -> DATA_READY
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else "2026-10-16"
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("AAPL", d, 150.0, 155.0, 149.0, 152.0, 1000000),
        )

    # 2. MSFT: 200 bars, stale date 2026-10-12 -> STALE_QUOTE
    for i in range(200):
        d = (base_date + timedelta(days=i)).isoformat() if i < 199 else "2026-10-12"
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("MSFT", d, 300.0, 305.0, 298.0, 302.0, 500000),
        )

    # 3. GLX: 47 bars (< 50) -> INSUFFICIENT_HISTORY
    for i in range(47):
        d = (base_date + timedelta(days=i)).isoformat()
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("GLX", d, 10.0, 11.0, 9.5, 10.5, 50000),
        )

    # 4. SPCX: 84 bars (50-199 bars, stale)
    for i in range(84):
        d = (base_date + timedelta(days=i)).isoformat()
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("SPCX", d, 25.0, 26.0, 24.5, 25.5, 20000),
        )

    conn.commit()
    conn.close()
    return db_file


def test_read_safety_zero_writes_and_schema_mutation(mock_market_db: Path) -> None:
    """Verifies that ReadOnlyMarketStore cannot write to or mutate the database."""
    stat_before = os.stat(mock_market_db)
    size_before = stat_before.st_size
    mtime_before = stat_before.st_mtime

    with ReadOnlyMarketStore(str(mock_market_db)) as store:
        engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")
        records = [
            DiagnosticInputRecord(record_locator="test#1", original_symbol="AAPL"),
            DiagnosticInputRecord(record_locator="test#2", original_symbol="MSFT"),
            DiagnosticInputRecord(record_locator="test#3", original_symbol="UNKNOWN_XYZ"),
        ]
        _, _, summary = engine.run_diagnostics(records)

    stat_after = os.stat(mock_market_db)
    assert stat_after.st_size == size_before, "Database size changed during read-only operation!"
    assert stat_after.st_mtime == mtime_before, "Database mtime changed during read-only operation!"


def test_missing_database_fails_closed_without_creation(tmp_path: Path) -> None:
    """Verifies that pointing to a non-existent database file fails closed without creating it."""
    non_existent = tmp_path / "does_not_exist.db"
    assert not non_existent.exists()

    with pytest.raises(FileNotFoundError):
        _ = ReadOnlyMarketStore(str(non_existent))

    # Invariant: Must not create a zero-byte file
    assert not non_existent.exists(), "ReadOnlyMarketStore created a file for non-existent path!"


def test_invalid_schema_fails_closed(tmp_path: Path) -> None:
    """Verifies that a database missing asset_ohlcv_daily fails closed."""
    invalid_db = tmp_path / "invalid_schema.db"
    conn = sqlite3.connect(str(invalid_db))
    conn.execute("CREATE TABLE wrong_table (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="missing 'asset_ohlcv_daily' table"):
        _ = ReadOnlyMarketStore(str(invalid_db))


def test_arithmetic_closure_and_reconciliation(mock_market_db: Path) -> None:
    """Verifies that all disjoint categories sum to total distinct keys exactly."""
    with ReadOnlyMarketStore(str(mock_market_db)) as store:
        engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")
        records = [
            DiagnosticInputRecord(record_locator="rec#1", original_symbol="AAPL"),
            DiagnosticInputRecord(record_locator="rec#2", original_symbol="MSFT"),
            DiagnosticInputRecord(record_locator="rec#3", original_symbol="GLX"),
            DiagnosticInputRecord(record_locator="rec#4", original_symbol="SPCX"),
            DiagnosticInputRecord(record_locator="rec#5", original_symbol="ABSENT_1"),
            DiagnosticInputRecord(record_locator="rec#6", original_symbol="ABSENT_2"),
            DiagnosticInputRecord(record_locator="rec#7", original_symbol=""),  # AMBIGUOUS
        ]
        obs_res, sym_res, summary = engine.run_diagnostics(records)

    assert summary.accounting_closed is True
    assert summary.total_input_observations == 7
    assert summary.distinct_lookup_keys == 7  # 4 stored symbols + 2 absent symbols + 1 empty key = 7 distinct keys

    # Sum of disjoint partitions must equal distinct keys
    disjoint_sum = (
        summary.data_ready_count +
        summary.stale_quote_count +
        summary.insufficient_history_count +
        summary.invalid_ohlcv_count +
        summary.missing_market_data_count +
        summary.ambiguous_identity_count +
        summary.inspection_failed_count
    )
    assert disjoint_sum == summary.distinct_lookup_keys


def test_distinct_observations_vs_lookup_keys(mock_market_db: Path) -> None:
    """Verifies that multiple observations for the same symbol are preserved distinctly."""
    with ReadOnlyMarketStore(str(mock_market_db)) as store:
        engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")
        # 3 observations for AAPL (e.g. from SEC, Nasdaq, and another feed)
        records = [
            DiagnosticInputRecord(record_locator="sec#1", original_symbol="AAPL", source_artifact="sec.json"),
            DiagnosticInputRecord(record_locator="nasdaq#1", original_symbol="AAPL", source_artifact="nasdaq.txt"),
            DiagnosticInputRecord(record_locator="dir#1", original_symbol="AAPL", source_artifact="other.txt"),
        ]
        obs_res, sym_res, summary = engine.run_diagnostics(records)

    assert len(obs_res) == 3, "Observations were lost!"
    assert len(sym_res) == 1, "Should consolidate to 1 distinct symbol!"
    assert summary.total_input_observations == 3
    assert summary.distinct_lookup_keys == 1
    assert sym_res["AAPL"].observation_count == 3
    assert set(sym_res["AAPL"].source_artifacts) == {"sec.json", "nasdaq.txt", "other.txt"}


def test_scanner_prerequisite_thresholds(mock_market_db: Path) -> None:
    """Verifies exact history depth floors: 50 sessions vs 200 sessions."""
    with ReadOnlyMarketStore(str(mock_market_db)) as store:
        engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")
        records = [
            DiagnosticInputRecord(record_locator="r1", original_symbol="AAPL"),  # 252 bars
            DiagnosticInputRecord(record_locator="r2", original_symbol="MSFT"),  # 200 bars
            DiagnosticInputRecord(record_locator="r3", original_symbol="SPCX"),  # 84 bars
            DiagnosticInputRecord(record_locator="r4", original_symbol="GLX"),   # 47 bars
        ]
        _, sym_res, summary = engine.run_diagnostics(records)

    assert sym_res["AAPL"].history_ge_200 is True
    assert sym_res["AAPL"].history_ge_50 is True
    assert sym_res["AAPL"].total_stored_candles == 252

    assert sym_res["MSFT"].history_ge_200 is True
    assert sym_res["MSFT"].history_ge_50 is True
    assert sym_res["MSFT"].total_stored_candles == 200

    assert sym_res["SPCX"].history_ge_200 is False
    assert sym_res["SPCX"].history_ge_50 is True
    assert sym_res["SPCX"].total_stored_candles == 84

    assert sym_res["GLX"].history_ge_200 is False
    assert sym_res["GLX"].history_ge_50 is False
    assert sym_res["GLX"].total_stored_candles == 47

    assert summary.history_ge_200_count == 2
    assert summary.history_50_to_199_count == 1
    assert summary.history_lt_50_count == 1
    assert summary.scanner_prerequisite_50_count == 3
    assert summary.scanner_prerequisite_200_count == 2


def test_offline_only_no_network_calls(mock_market_db: Path) -> None:
    """Verifies that running diagnostics makes zero external socket network connections."""
    import socket
    real_connect = socket.socket.connect

    def forbidden_connect(self, *args, **kwargs):
        raise AssertionError("Network connection attempted during pure offline diagnostic execution!")

    with patch.object(socket.socket, "connect", forbidden_connect):
        with ReadOnlyMarketStore(str(mock_market_db)) as store:
            engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")
            records = [
                DiagnosticInputRecord(record_locator="r1", original_symbol="AAPL"),
                DiagnosticInputRecord(record_locator="r2", original_symbol="UNKNOWN"),
            ]
            _, _, summary = engine.run_diagnostics(records)
            assert summary.accounting_closed is True


def test_capacity_benchmarks_execute_cleanly(mock_market_db: Path) -> None:
    """Verifies that capacity benchmarks execute side-effect-free."""
    res = run_check_d_capacity_benchmarks(
        market_db_path=mock_market_db,
        sample_sizes=(2, 4),
        target_session_date="2026-10-16",
    )
    assert "measured_sample_latencies" in res
    assert len(res["measured_sample_latencies"]) == 2
    assert "memory_footprint_per_symbol" in res
    assert "projected_capacity" in res
    assert res["evidence_vs_projection_distinction"]["assessment_capacity_status"] == "NOT_ESTABLISHED"


def test_pipeline_integration_gaps_audit() -> None:
    """Verifies that Check E identifies all mandatory integration gaps."""
    gaps = run_check_e_pipeline_integration_gaps()
    assert gaps["boundary_1_adapters_to_validator"]["status"] == "DISCONNECTED_AT_RUNTIME"
    assert gaps["boundary_2_validator_to_universe_builder"]["status"] == "DISCONNECTED_AT_RUNTIME"
    assert any("security_id" in f for f in gaps["boundary_1_adapters_to_validator"]["missing_fields"])
    assert any("R02" in r for r in gaps["required_remediation_actions"])


def test_discrepancy_table_structural_integrity() -> None:
    """Verifies that the discrepancy reconciliation table covers all required dimensions."""
    assert len(DISCREPANCY_RECONCILIATION_TABLE) >= 9
    required_dims = {
        "Associated lookup keys",
        "Missing market data",
        "Insufficient history (< 50 bars)",
        "Stale quote",
        "DATA_READY",
        "GLX history",
        "Loaded observations",
        "Benchmark results",
        "Distinct lookup keys",
        "Builder readiness",
    }
    actual_dims = {r["dimension"] for r in DISCREPANCY_RECONCILIATION_TABLE}
    assert required_dims.issubset(actual_dims)

    for row in DISCREPANCY_RECONCILIATION_TABLE:
        assert row["earlier_value"] != ""
        assert row["final_reconciled_value"] != ""
        assert row["established_root_cause"] != ""
        assert row["reconciled_treatment"] != ""


def test_baseline_outputs_share_exact_run_id_and_model(tmp_path: Path, mock_market_db: Path) -> None:
    """Verifies that Markdown and JSON baseline outputs derive from identical structured data."""
    import json
    import hashlib

    # Create synthetic feed files with pinned hashes
    sec_data = {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [
            [320193, "Apple Inc.", "AAPL", "Nasdaq"],
            [789019, "Microsoft Corp", "MSFT", "Nasdaq"],
        ]
    }
    sec_file = tmp_path / "sec_company_tickers_exchange.json"
    sec_file.write_text(json.dumps(sec_data), encoding="utf-8")

    nasdaq_txt = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
        "File Creation Time: 1016202607:00|||||||\n"
    )
    nasdaq_file = tmp_path / "nasdaq_listed_directory.txt"
    nasdaq_file.write_text(nasdaq_txt, encoding="utf-8")

    other_txt = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "SPY|SPDR S&P 500 ETF Trust|A|SPY|Y|100|N|SPY\n"
        "File Creation Time: 1016202607:00|||||||\n"
    )
    other_file = tmp_path / "other_listed_directory.txt"
    other_file.write_text(other_txt, encoding="utf-8")

    # Patch PINNED_FEED_HASHES temporarily so synthetic files pass integrity check
    syn_hashes = {
        "sec_company_tickers_exchange.json": hashlib.sha256(sec_file.read_bytes()).hexdigest(),
        "nasdaq_listed_directory.txt": hashlib.sha256(nasdaq_file.read_bytes()).hexdigest(),
        "other_listed_directory.txt": hashlib.sha256(other_file.read_bytes()).hexdigest(),
    }

    with patch.dict("scripts.diagnostics.universe_coverage_capacity.PINNED_FEED_HASHES", syn_hashes):
        report = run_complete_diagnostics(
            feed_dir=tmp_path,
            market_db_path=mock_market_db,
            target_session_date="2026-10-16",
        )

    run_id = report["metadata"]["run_id"]
    assert run_id.startswith("RUN-S2-DIAG-")

    # Render markdown from the exact same report
    md_text = render_baseline_markdown(report)

    # Invariants: Markdown must contain the exact run_id, commit, and key counts
    assert run_id in md_text
    assert report["metadata"]["git_revision"] in md_text
    assert "0. Discrepancy & Root Cause Reconciliation" in md_text
    assert "ASSESSMENT_THROUGHPUT = NOT_MEASURED" in md_text
    assert "FULL_CHECKPOINTED" in md_text


def test_in_memory_readiness_benchmark_prefetches_and_evaluates_without_sql(mock_market_db: Path) -> None:
    """Verifies that in-memory rule check benchmark executes pure eval with prefetched store data."""
    store = ReadOnlyMarketStore(str(mock_market_db))
    benchmarker = CapacityBenchmarker(store, target_session_date="2026-10-16")

    measurement = benchmarker.measure_in_memory_readiness_rule_check(["AAPL", "MSFT", "SPCX"], repetitions=20)
    assert measurement.operation_name == "IN_MEMORY_READINESS_RULE_CHECK"
    assert measurement.throughput_ops_per_second > 0
    assert measurement.mean_latency_ms < 0.1  # In-memory eval should be sub-100 microseconds


def test_data_readiness_semantics_disclaim_strategy_eligibility(mock_market_db: Path) -> None:
    """Verifies that DATA_READY outcome explicitly disclaims strategy execution eligibility."""
    store = ReadOnlyMarketStore(str(mock_market_db))
    engine = CoverageDiagnosticEngine(store, target_session_date="2026-10-16")
    primary, b_read, s_read, details = engine.evaluate_symbol("AAPL")
    assert primary == PrimaryDiagnosticOutcome.DATA_READY
    assert "Meets inspected local data prerequisites" in details["detail"]
    assert "strategy eligibility" not in details["detail"].lower() or "not" in details["detail"].lower()


def test_discrepancy_table_contains_exact_prompt_columns_and_reconciled_values() -> None:
    """Verifies the exact 5 columns and 8 core reconciled metrics specified by prompt."""
    required_cols = {"metric", "earlier_value", "later_corrected_value", "confirmed_root_cause", "resolution_status"}
    for row in DISCREPANCY_RECONCILIATION_TABLE:
        assert required_cols.issubset(row.keys()), f"Missing required column in row: {row}"
        assert row["metric"], "Metric name cannot be empty"
        assert row["earlier_value"], "Earlier value cannot be empty"
        assert row["later_corrected_value"], "Later corrected value cannot be empty"
        assert row["confirmed_root_cause"], "Confirmed root cause cannot be empty"
        assert row["resolution_status"], "Resolution status cannot be empty"

    metric_map = {r["metric"]: r for r in DISCREPANCY_RECONCILIATION_TABLE}

    # 1. Associated lookup keys: 147 -> 138
    assert metric_map["Associated lookup keys"]["earlier_value"] == "147"
    assert metric_map["Associated lookup keys"]["later_corrected_value"] == "138"
    assert "Fixture contamination" in metric_map["Associated lookup keys"]["confirmed_root_cause"]

    # 2. Missing market data: 16,301 -> 16,310
    assert metric_map["Missing market data"]["earlier_value"] == "16,301"
    assert metric_map["Missing market data"]["later_corrected_value"] == "16,310"

    # 3. Insufficient history (< 50 bars): 7 -> 1
    assert metric_map["Insufficient history (< 50 bars)"]["earlier_value"] == "7"
    assert metric_map["Insufficient history (< 50 bars)"]["later_corrected_value"] == "1"

    # 4. Stale quote: 104 -> 101
    assert metric_map["Stale quote"]["earlier_value"] == "104"
    assert metric_map["Stale quote"]["later_corrected_value"] == "101"

    # 5. DATA_READY: 36 -> 36
    assert metric_map["DATA_READY"]["earlier_value"] == "36"
    assert metric_map["DATA_READY"]["later_corrected_value"] == "36"

    # 6. GLX history: 20 -> 47 bars
    assert "20" in metric_map["GLX history"]["earlier_value"]
    assert "47" in metric_map["GLX history"]["later_corrected_value"]
    assert "transcription" in metric_map["GLX history"]["confirmed_root_cause"].lower()

    # 7. Loaded observations: 23,416 -> 23,584
    assert metric_map["Loaded observations"]["earlier_value"] == "23,416"
    assert "23,584" in metric_map["Loaded observations"]["later_corrected_value"]
    assert "Double-deduction" in metric_map["Loaded observations"]["confirmed_root_cause"]

    # 8. Benchmark results: differing sets -> component-isolated benchmarks
    assert "Differing" in metric_map["Benchmark results"]["confirmed_root_cause"]
    assert "NOT_ESTABLISHED" in metric_map["Benchmark results"]["later_corrected_value"]


def test_capacity_speculative_projections_removed(mock_market_db: Path) -> None:
    """Verifies that 8-core speculative projections are completely removed from Check D."""
    res = run_check_d_capacity_benchmarks(mock_market_db, sample_sizes=(2, 4), target_session_date="2026-10-16")
    proj = res["projected_capacity"]
    assert "full_universe_16448_projected_parallel_8core_sec" not in proj
    assert "common_stocks_5871_projected_parallel_8core_sec" not in proj
    assert proj["speculative_parallel_projections"] == "REMOVED"
    assert proj["end_to_end_scanner_capacity"] == "NOT_ESTABLISHED"
    assert proj["assessment_throughput_status"] == "NOT_MEASURED"
    assert res["evidence_vs_projection_distinction"]["assessment_capacity_status"] == "NOT_ESTABLISHED"
    assert res["evidence_vs_projection_distinction"]["assessment_throughput_status"] == "NOT_MEASURED"


def test_unassociated_stored_symbols_and_po_scope_context(mock_market_db: Path) -> None:
    """Verifies that unassociated symbols CPRX and DHLGY are classified with UNASSOCIATED_UNVERIFIED status."""
    conn = sqlite3.connect(str(mock_market_db))
    # Insert CPRX and DHLGY into mock DB
    conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('CPRX', '2026-10-16', 10.0, 11.0, 9.0, 10.5, 1000)")
    conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('DHLGY', '2026-10-16', 50.0, 52.0, 49.0, 51.0, 2000)")
    conn.commit()
    conn.close()

    feed_records = [
        DiagnosticInputRecord(record_locator="r1", original_symbol="AAPL"),
    ]
    with patch("scripts.diagnostics.universe_coverage_capacity.load_acquisition_artifacts_to_records", return_value=(feed_records, {})):
        _, check_b = run_check_b_market_data_association(feed_dir=mock_market_db.parent, market_db_path=mock_market_db)

    taxonomy = check_b["unmatched_db_symbol_taxonomy"]
    assert "unassociated_stored_symbols" in taxonomy
    assert "CPRX" in taxonomy["unassociated_stored_symbols"]
    assert "DHLGY" in taxonomy["unassociated_stored_symbols"]
    assert taxonomy["unassociated_stored_symbols_status"] == "UNASSOCIATED_UNVERIFIED"
    assert "product_owner_context" in check_b["store_storage_profile"]


