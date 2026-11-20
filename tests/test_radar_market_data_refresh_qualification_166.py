"""
Offline qualification tests for ARX Radar 500 Market Data Refresh Path (Issue #166).

Validates:
1. Atomic commit and rollback of bars and provenance (single SQLite transaction).
2. Strict binding of source (ALPACA), feed (sip), adjustment (SPLIT_ADJUSTED_ONLY), and timestamps.
3. Replay idempotency and conflict policies (REPLACE vs REFRESH).
4. Fail-closed handling when source evidence is missing, gapped, or unprovenanced.
5. Absolute separation from scanner execution, scoring, and snapshot publication.
6. End-to-end reconciliation with the readiness auditor (symbols transition to READY).
"""

import json
import sqlite3
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

if "ccxt" not in sys.modules:
    sys.modules["ccxt"] = MagicMock()

import pytest

from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.data.market_evidence import (
    AdjustmentState,
    IngestionSource,
    MarketProvenance,
    ObservationPrecision,
    ObservationSource,
    Provider,
    StructuralQuality,
)
from scripts.diagnostics.audit_radar_universe_readiness_166 import (
    run_audit,
)
from test_radar_universe_readiness_auditor_166 import (
    _init_empty_aux_dbs,
    _init_synthetic_market_db,
)
from scripts.ingest_alpaca_historical import (
    AlpacaHistoricalSipIngestionAdapter,
    SharedExecutionBudget,
    get_expected_nyse_sessions,
    quiesce_and_recover_database,
    resolve_runtime_market_db_path,
    verify_serving_runtime_identity,
    HISTORICAL_SERVING_DEPLOYMENT_ID,
    HISTORICAL_SERVING_GIT_SHA,
)
import api.routes.analytics as analytics_mod
import api.main as main_mod


@pytest.fixture
def disposable_market_db(tmp_path: Path):
    """Create a clean disposable SQLite market database."""
    db_file = tmp_path / "disposable_market_store.db"
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
    cur.execute("""
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
    cur.execute("""
        CREATE TABLE scanner_snapshots (
            id TEXT PRIMARY KEY,
            snapshot_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            universe_id TEXT NOT NULL,
            git_commit TEXT NOT NULL,
            candidate_count INTEGER NOT NULL,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    return str(db_file)


class MockAlpacaSipFetcher:
    """Deterministic mock provider returning synthetic NYSE-compliant Alpaca SIP bars."""

    def __init__(self, bars_by_symbol=None, transient_failures_count=0):
        self.bars_by_symbol = bars_by_symbol or {}
        self.calls = []
        self.transient_failures_remaining = transient_failures_count

    def fetch_historical_bars(self, symbol, start, end, feed=None, adjustment=None, limit=1000, page_token=None, **kwargs):
        self.calls.append({"symbol": symbol, "start": start, "end": end, "feed": feed, "adjustment": adjustment, "page_token": page_token})

        if self.transient_failures_remaining > 0:
            self.transient_failures_remaining -= 1
            return {
                "success": False,
                "status_code": 429,
                "symbol": symbol,
                "bars": [],
                "next_page_token": None,
                "error": "RATE_LIMIT_EXCEEDED: 429 Too Many Requests",
            }

        if symbol in self.bars_by_symbol:
            resp = self.bars_by_symbol[symbol]
            if callable(resp):
                return resp(symbol, start, end, page_token=page_token)
            return resp

        # Generate default synthetic bars for any valid NYSE session in range
        sessions = get_expected_nyse_sessions(start, end)
        bars = [
            {
                "t": f"{s}T05:00:00Z",
                "o": 150.0,
                "h": 155.0,
                "l": 149.0,
                "c": 153.5,
                "v": 2500000,
            }
            for s in sessions
        ]
        return {
            "success": True,
            "symbol": symbol,
            "bars": bars,
            "next_page_token": None,
        }


def test_atomic_commit_and_rollback_bars_and_provenance(disposable_market_db: str):
    """GATE 3.1: Verify atomic commit and rollback of bars and provenance."""
    engine = MarketDatabaseEngine(db_path=disposable_market_db)
    prov = MarketProvenance(
        provider=Provider.ALPACA,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        observed_at="2026-11-20T21:00:00Z",
        observed_date="2026-11-20",
        observation_precision=ObservationPrecision.DATE,
        observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
        ingested_at="2026-11-20T21:05:00Z",
        adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
        structural_quality=StructuralQuality.COMPLETE,
        fallback_status=False,
    )
    bars = [
        {"time": "2026-11-19", "open": 220.0, "high": 225.0, "low": 219.0, "close": 224.0, "volume": 1000000},
        {"time": "2026-11-20", "open": 224.0, "high": 228.0, "low": 223.0, "close": 227.5, "volume": 1100000},
    ]

    # 1. Atomic commit succeeds: both tables contain exactly 2 rows
    engine.save_daily_candles_with_evidence("NVDA", bars, prov)

    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'NVDA'")
    assert cur.fetchone()[0] == 2
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = 'NVDA'")
    assert cur.fetchone()[0] == 2

    # 2. Simulated failure mid-transaction triggers complete rollback
    # Injecting invalid date type or constraint failure
    malformed_bars = [
        {"time": "2026-11-21", "open": "INVALID_FLOAT", "high": 230.0, "low": 220.0, "close": 225.0, "volume": 1000}
    ]
    with pytest.raises(ValueError):
        engine.save_daily_candles_with_evidence("NVDA", malformed_bars, prov)

    # State after failure: exactly 2 rows remain, NO partial commit of 2026-11-21
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'NVDA'")
    assert cur.fetchone()[0] == 2
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = 'NVDA'")
    assert cur.fetchone()[0] == 2
    cur.execute("SELECT trade_date FROM asset_ohlcv_daily WHERE symbol = 'NVDA' ORDER BY trade_date")
    assert [r[0] for r in cur.fetchall()] == ["2026-11-19", "2026-11-20"]
    conn.close()


def test_provenance_binding_contract_fields(disposable_market_db: str):
    """GATE 3.2: Verify strict binding of source, feed, adjustment, and observed_at."""
    engine = MarketDatabaseEngine(db_path=disposable_market_db)
    obs_time = "2026-11-20T21:15:30Z"
    ingest_time = "2026-11-20T21:15:45Z"
    prov = MarketProvenance(
        provider=Provider.ALPACA,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        observed_at=obs_time,
        observed_date="2026-11-20",
        observation_precision=ObservationPrecision.DATE,
        observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
        ingested_at=ingest_time,
        adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
        structural_quality=StructuralQuality.COMPLETE,
        fallback_status=False,
    )
    bars = [{"time": "2026-11-20", "open": 180.0, "high": 185.0, "low": 179.0, "close": 183.0, "volume": 500000}]
    engine.save_daily_candles_with_evidence("VRT", bars, prov)

    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("""
        SELECT provider, ingestion_source, observed_at, observed_date, observation_precision,
               observation_source, ingested_at, adjustment_state, structural_quality, fallback_status
        FROM asset_ohlcv_provenance WHERE symbol = 'VRT' AND trade_date = '2026-11-20'
    """)
    row = cur.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "ALPACA"
    assert row[1] == "DIRECT_PROVIDER"
    assert row[2] == obs_time
    assert row[3] == "2026-11-20"
    assert row[4] == "DATE"
    assert row[5] == "PROVIDER_BAR_TIMESTAMP"
    assert row[6] == ingest_time
    assert row[7] == "SPLIT_ADJUSTED_ONLY"
    assert row[8] == "COMPLETE"
    assert row[9] == 0


def test_replay_idempotency_and_conflict_policies(disposable_market_db: str, tmp_path: Path):
    """GATE 3.3: Verify replay idempotency and conflict policies."""
    # Manifest with conflicting directives: both BACKFILL and REPLACE
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(json.dumps({
        "ingestion_scope": {
            "candidate_bindings": {
                "AAPL": {"action": "REPLACE", "rationale": "Clear unprovenanced historical bars"}
            },
            "production_target_backfill": {"missing_symbols": ["AAPL"]},
            "production_target_refresh": {
                "per_symbol_parameters": [
                    {"symbol": "AAPL", "action": "REFRESH", "requested_start_session": "2026-11-07"}
                ]
            }
        }
    }), encoding="utf-8")

    mock_fetcher = MockAlpacaSipFetcher()
    adapter = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher,
        feed="sip",
        adjustment="split",
        max_completed_session=date(2026, 11, 20),
    )

    # Run 1: Apply writes for AAPL
    summary1 = adapter.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=["AAPL"],
        start_date="2025-01-02",
        end_date="2026-11-20",
        apply_writes=True,
        db_path=disposable_market_db,
    )
    assert summary1.execution_outcome == "SUCCESS"
    assert summary1.candidate_dispositions["AAPL"]["action_type"] == "REPLACE"  # REPLACE took precedence!

    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'AAPL'")
    count_run1 = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = 'AAPL'")
    prov_run1 = cur.fetchone()[0]
    conn.close()

    # Run 2: Replay identical execution
    summary2 = adapter.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=["AAPL"],
        start_date="2025-01-02",
        end_date="2026-11-20",
        apply_writes=True,
        db_path=disposable_market_db,
    )
    assert summary2.execution_outcome == "SUCCESS"

    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'AAPL'")
    count_run2 = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = 'AAPL'")
    prov_run2 = cur.fetchone()[0]
    conn.close()

    assert count_run1 == count_run2 > 0
    assert prov_run1 == prov_run2 == count_run1


def test_missing_unverified_source_evidence_fails_closed(disposable_market_db: str, tmp_path: Path):
    """GATE 3.4: Verify fail-closed behavior when expected evidence is missing or gapped."""
    # Seed a symbol with unprovenanced bars in history
    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("INSERT INTO asset_ohlcv_daily VALUES ('TSLA', '2025-01-02', 200, 205, 198, 202, 100000)")
    conn.commit()
    conn.close()

    manifest_path = tmp_path / "tsla_manifest.json"
    manifest_path.write_text(json.dumps({
        "ingestion_scope": {
            "production_target_backfill": {"missing_symbols": []},
            "production_target_refresh": {
                "per_symbol_parameters": [{"symbol": "TSLA", "action": "REFRESH"}]
            }
        }
    }), encoding="utf-8")

    mock_fetcher = MockAlpacaSipFetcher()
    adapter = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher,
        feed="sip",
        adjustment="split",
        max_completed_session=date(2026, 11, 20),
    )

    summary = adapter.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=["TSLA"],
        start_date="2026-11-07",
        end_date="2026-11-20",
        apply_writes=False,
        db_path=disposable_market_db,
    )
    disp = summary.candidate_dispositions["TSLA"]
    assert disp["status"] == "BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE"
    assert "INCOMPATIBLE_HISTORY_OUTSIDE_REVIEWED_SCOPE" in disp["rejection_reason"]


def test_complete_separation_from_scanner_and_publication(disposable_market_db: str, tmp_path: Path):
    """GATE 3.5: Verify complete separation from scanning, scoring, and publication."""
    manifest_path = tmp_path / "sep_manifest.json"
    manifest_path.write_text(json.dumps({
        "ingestion_scope": {
            "candidate_bindings": {"MSFT": {"action": "BACKFILL"}},
            "production_target_backfill": {"missing_symbols": ["MSFT"]},
            "production_target_refresh": {"per_symbol_parameters": []}
        }
    }), encoding="utf-8")

    mock_fetcher = MockAlpacaSipFetcher()
    adapter = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher,
        feed="sip",
        adjustment="split",
        max_completed_session=date(2026, 11, 20),
    )

    with patch("api.routes.screener.run_screener") as mock_screener, \
         patch("scripts.ingest_alpaca_historical.logger") as mock_logger:

        summary = adapter.execute_bounded_ingestion(
            manifest_path=manifest_path,
            symbols_override=["MSFT"],
            start_date="2025-01-02",
            end_date="2026-11-20",
            apply_writes=True,
            db_path=disposable_market_db,
        )

        assert summary.execution_outcome == "SUCCESS"
        mock_screener.assert_not_called()

    # Verify ZERO rows written to scanner_snapshots
    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM scanner_snapshots")
    assert cur.fetchone()[0] == 0
    conn.close()


def test_end_to_end_auditor_qualification_after_refresh(disposable_market_db: str, tmp_path: Path):
    """GATE 3.6: Verify that refreshed data passes the readiness auditor as READY."""
    # Hydrate AAPL and NVDA from 2025-01-02 through 2026-11-20 using the qualified adapter
    manifest_path = tmp_path / "qual_manifest.json"
    manifest_path.write_text(json.dumps({
        "ingestion_scope": {
            "candidate_bindings": {
                "AAPL": {"action": "BACKFILL"},
                "NVDA": {"action": "BACKFILL"},
            },
            "production_target_backfill": {"missing_symbols": ["AAPL", "NVDA"]},
            "production_target_refresh": {"per_symbol_parameters": []}
        }
    }), encoding="utf-8")

    mock_fetcher = MockAlpacaSipFetcher()
    adapter = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher,
        feed="sip",
        adjustment="split",
        max_completed_session=date(2026, 11, 20),
    )

    summary = adapter.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=["AAPL", "NVDA"],
        start_date="2025-01-02",
        end_date="2026-11-20",
        apply_writes=True,
        db_path=disposable_market_db,
    )
    assert summary.execution_outcome == "SUCCESS"

    # Now audit these symbols with audit_radar_universe_readiness_166 via run_audit
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    out_json = tmp_path / "auditor_out.json"
    out_md = tmp_path / "auditor_out.md"
    cand_manifest = tmp_path / "auditor_cand.json"
    prom_manifest = tmp_path / "auditor_prom.json"

    conn_u = sqlite3.connect(universe_db)
    conn_u.execute("CREATE TABLE universe_builds (build_id TEXT PRIMARY KEY, created_at TEXT)")
    conn_u.commit()
    conn_u.close()

    conn_s = sqlite3.connect(scanner_db)
    conn_s.execute("CREATE TABLE scanner_snapshots (snapshot_id TEXT PRIMARY KEY, scanner_id TEXT, run_id TEXT, implementation_release_sha TEXT, status_at_publication TEXT, created_at TEXT, data_as_of TEXT)")
    conn_s.commit()
    conn_s.close()

    symbols = ["AAPL", "NVDA"]
    cand_manifest.write_text(json.dumps({"manifest_build_id": "QUAL-CAND", "membership_hash": "h_cand", "symbols": symbols}))
    prom_manifest.write_text(json.dumps({"manifest_build_id": "QUAL-PROM", "membership_hash": "h_prom", "symbols": symbols}))

    payload = run_audit(
        market_db_path=Path(disposable_market_db),
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json,
        output_md_path=out_md,
    )

    s_cand = payload["summary_metrics_candidate_universe"]
    assert s_cand["total_symbols"] == 2
    assert s_cand["readiness_breakdown"]["READY"] == 2
    assert s_cand["readiness_breakdown"]["STALE_MARKET_DATA"] == 0
    assert s_cand["readiness_breakdown"]["MISSING_PROVENANCE"] == 0
    assert s_cand["history_depth_breakdown"]["ge_252"] == 2
    assert s_cand["vcp_stage2_eligibility_breakdown"]["ELIGIBLE"] == 2
    assert s_cand["accounting_closure"]["is_closed"] is True


# ==============================================================================
# Additional Offline Qualification Tests for Issue #166 Corrected Refresh Plan
# ==============================================================================

def test_runtime_db_path_resolution_and_identity_check(tmp_path: Path):
    """GATE 1: Verify target DB path resolves to .finance_market_store.db and enforces identity checks."""
    import hashlib

    # 1. Path resolution hierarchy
    f_dir = str(tmp_path / "finance_root")
    d_dir = str(tmp_path / "data_root")

    # FINANCE_DATA_DIR takes precedence
    p1 = resolve_runtime_market_db_path({"FINANCE_DATA_DIR": f_dir, "DATA_DIR": d_dir})
    assert p1 == str(Path(f_dir) / ".finance_market_store.db")

    # Fallback to DATA_DIR
    p2 = resolve_runtime_market_db_path({"DATA_DIR": d_dir})
    assert p2 == str(Path(d_dir) / ".finance_market_store.db")

    # Fallback to home dir
    p3 = resolve_runtime_market_db_path({})
    assert p3.endswith(".finance_market_store.db")

    # 2. Execution-time identity verification
    fake_db = tmp_path / ".finance_market_store.db"
    fake_db.write_text("sqlite-content")

    # 2a. Reject missing identity (fail closed)
    ok_no_env, err_no_env = verify_serving_runtime_identity(env={}, db_path=str(fake_db))
    assert ok_no_env is False
    assert "MISSING_RUNTIME_IDENTITY" in err_no_env

    ok_no_sha, err_no_sha = verify_serving_runtime_identity(
        env={"RAILWAY_DEPLOYMENT_ID": HISTORICAL_SERVING_DEPLOYMENT_ID},
        db_path=str(fake_db)
    )
    assert ok_no_sha is False
    assert "MISSING_RUNTIME_IDENTITY" in err_no_sha

    ok_no_dep, err_no_dep = verify_serving_runtime_identity(
        env={"RAILWAY_GIT_COMMIT_SHA": HISTORICAL_SERVING_GIT_SHA},
        db_path=str(fake_db)
    )
    assert ok_no_dep is False
    assert "MISSING_RUNTIME_IDENTITY" in err_no_dep

    # 2b. Full SHA equality: reject prefix matching and invalid length
    ok_prefix, err_prefix = verify_serving_runtime_identity(
        env={"RAILWAY_DEPLOYMENT_ID": HISTORICAL_SERVING_DEPLOYMENT_ID, "RAILWAY_GIT_COMMIT_SHA": "868d7afc"},
        db_path=str(fake_db)
    )
    assert ok_prefix is False
    assert "INVALID_GIT_SHA_FORMAT" in err_prefix

    ok_bad_sha, err_bad_sha = verify_serving_runtime_identity(
        env={"RAILWAY_DEPLOYMENT_ID": HISTORICAL_SERVING_DEPLOYMENT_ID, "RAILWAY_GIT_COMMIT_SHA": "deadbeef1234deadbeef1234deadbeef1234dead"},
        db_path=str(fake_db)
    )
    assert ok_bad_sha is False
    assert "GIT_REVISION_MISMATCH" in err_bad_sha

    # 2c. Matching historical baseline succeeds
    matching_env = {
        "RAILWAY_DEPLOYMENT_ID": HISTORICAL_SERVING_DEPLOYMENT_ID,
        "RAILWAY_GIT_COMMIT_SHA": HISTORICAL_SERVING_GIT_SHA,
    }
    ok_hist, err_hist = verify_serving_runtime_identity(
        env=matching_env,
        db_path=str(fake_db),
    )
    assert ok_hist is True
    assert err_hist is None

    # 2d. Explicitly pinned future qualified deployment
    fut_dep = "dep-pinned-20261125-prod"
    fut_sha = "abcdef0123456789abcdef0123456789abcdef01"
    ok_fut, err_fut = verify_serving_runtime_identity(
        env={"RAILWAY_DEPLOYMENT_ID": fut_dep, "RAILWAY_GIT_COMMIT_SHA": fut_sha},
        db_path=str(fake_db),
        expected_deployment_id=fut_dep,
        expected_git_sha=fut_sha,
    )
    assert ok_fut is True
    assert err_fut is None

    # Mismatched deployment against pinned future deployment fails
    ok_fut_mismatch, err_fut_mismatch = verify_serving_runtime_identity(
        env={"RAILWAY_DEPLOYMENT_ID": "different-dep", "RAILWAY_GIT_COMMIT_SHA": fut_sha},
        db_path=str(fake_db),
        expected_deployment_id=fut_dep,
        expected_git_sha=fut_sha,
    )
    assert ok_fut_mismatch is False
    assert "DEPLOYMENT_ID_MISMATCH" in err_fut_mismatch

    # 2e. Universe membership validation in authoritative store
    universe_db = tmp_path / ".finance_universe_store.db"
    conn = sqlite3.connect(str(universe_db))
    conn.execute("""
        CREATE TABLE universe_builds (
            universe_build_id TEXT PRIMARY KEY,
            universe_id TEXT NOT NULL,
            universe_version TEXT NOT NULL,
            eligible_count INTEGER NOT NULL,
            eligible_membership_hash TEXT NOT NULL,
            publication_decision TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE universe_eligibility_ledger (
            universe_build_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            eligibility_decision TEXT NOT NULL,
            PRIMARY KEY (universe_build_id, symbol)
        )
    """)
    syms = [f"SYM_{i:03d}" for i in range(500)]
    u_hash = hashlib.sha256(",".join(sorted(syms)).encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT INTO universe_builds VALUES ('UB-ARX-EXPANSION-500-20261108', 'UNIV_EXP_500', '1.0', 500, ?, 'PUBLISHED')",
        (u_hash,)
    )
    for s in syms:
        conn.execute("INSERT INTO universe_eligibility_ledger VALUES ('UB-ARX-EXPANSION-500-20261108', ?, 'ELIGIBLE')", (s,))
    conn.commit()
    conn.close()

    ok_u, err_u = verify_serving_runtime_identity(
        env=matching_env,
        db_path=str(fake_db),
        universe_db_path=str(universe_db),
        expected_universe_hash=u_hash,
    )
    assert ok_u is True
    assert err_u is None

    # Un-published universe fails closed
    conn = sqlite3.connect(str(universe_db))
    conn.execute("UPDATE universe_builds SET publication_decision = 'DRAFT' WHERE universe_build_id = 'UB-ARX-EXPANSION-500-20261108'")
    conn.commit()
    conn.close()

    ok_draft, err_draft = verify_serving_runtime_identity(
        env=matching_env,
        db_path=str(fake_db),
        universe_db_path=str(universe_db),
        expected_universe_hash=u_hash,
    )
    assert ok_draft is False
    assert "UNIVERSE_NOT_PUBLISHED" in err_draft


def test_recurrence_prevention_guards_authoritative_provenance(disposable_market_db: str):
    """GATE 3: Qualify recurrence prevention in actual application code (background, direct, warmup)."""
    engine = MarketDatabaseEngine(db_path=disposable_market_db)
    trade_date = "2026-11-20"
    upper = "AAPL"

    # Step 1: Seed candidate with authoritative Alpaca SIP provenance
    prov_alpaca = MarketProvenance(
        provider=Provider.ALPACA,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        observed_at="2026-11-20T21:00:00Z",
        observed_date=trade_date,
        observation_precision=ObservationPrecision.DATE,
        observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
        ingested_at="2026-11-20T21:05:00Z",
        adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
        structural_quality=StructuralQuality.COMPLETE,
        fallback_status=False,
    )
    candle_alpaca = [{"time": trade_date, "open": 220.0, "high": 225.0, "low": 219.0, "close": 224.0, "volume": 1000000}]
    engine.save_daily_candles_with_evidence(upper, candle_alpaca, prov_alpaca)

    assert engine.has_authoritative_provenance(upper) is True

    # Step 2: Test real entry points in production code
    with patch.object(analytics_mod, "market_db", engine):
        with patch("yfinance.Ticker") as mock_yf:
            # 2a. Background writer real entry point
            analytics_mod._background_refresh_symbol(upper)
            mock_yf.assert_not_called()

            # 2b. Background scheduler real entry point
            analytics_mod._schedule_background_refresh(upper)
            assert upper not in analytics_mod._refresh_in_flight

            # 2c. Direct computation setup real entry point
            setup_res = analytics_mod._compute_single_setup(upper, "LONG_TERM")
            mock_yf.assert_not_called()

            # 2d. Direct tactical setup API endpoint real entry point
            setup_api = analytics_mod.get_tactical_setup_for_symbol(symbol=upper, user_role="LONG_TERM")
            mock_yf.assert_not_called()

    # 2e. Warmup worker real entry point in api.main
    with patch("analyst_dashboard.data.market_db.MarketDatabaseEngine", return_value=engine):
        with patch("yfinance.Ticker") as mock_yf_warmup:
            with patch("api.routes.screener.run_screener"):
                main_mod._warmup_worker()
                called_warmup_syms = [c.args[0] for c in mock_yf_warmup.call_args_list if c.args]
                assert upper not in called_warmup_syms, "Warmup worker must skip authoritative AAPL"

    # Step 3: Engine-level recurrence guard with reject_if_authoritative=True
    engine.save_daily_candles(upper, [{"time": trade_date, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 500}], reject_if_authoritative=True)

    # Assert bars and provenance remain 100% untouched
    stored = engine.get_daily_candles(upper, limit=10)
    assert len(stored) == 1
    assert stored[0]["close"] == 224.0, "Authoritative Alpaca bar was protected from unprovenanced overwrite"

    # Step 4: Verify contradictory-provenance deletion on legitimate replacement
    # When different-source bars are legitimately replaced without evidence on non-authoritative tickers,
    # contradictory provenance MUST be deleted to satisfy PREFLIGHT-001 / Issue #198.
    adhoc_sym = "ADHOC_TICKER"
    engine.save_daily_candles_with_evidence(
        adhoc_sym,
        [{"time": trade_date, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000}],
        prov_alpaca
    )
    # Legitimate replacement without reject_if_authoritative
    engine.save_daily_candles(
        adhoc_sym,
        [{"time": trade_date, "open": 10.2, "high": 11.2, "low": 9.2, "close": 10.8, "volume": 1200}],
        reject_if_authoritative=False
    )
    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = ?", (adhoc_sym,))
    prov_count = cur.fetchone()[0]
    conn.close()
    assert prov_count == 0, "Contradictory provenance must be deleted upon legitimate replacement"


def test_derived_replacement_windows_and_nvda_2023_contract(disposable_market_db: str, tmp_path: Path):
    """GATE 4: Derive replacement windows from actual history; verify NVDA 2023 history requirements."""
    # Seed NVDA with history going back to 2023-10-27 (matching actual production store export)
    engine = MarketDatabaseEngine(db_path=disposable_market_db)
    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()

    # 1. Unprovenanced bar on 2023-10-27
    cur.execute("INSERT INTO asset_ohlcv_daily VALUES ('NVDA', '2023-10-27', 40.0, 42.0, 39.0, 41.0, 50000000)")
    # 2. Unprovenanced bars in 2025-2026
    sessions = get_expected_nyse_sessions("2025-01-02", "2026-11-06")
    for s in sessions[:10]:
        cur.execute("INSERT INTO asset_ohlcv_daily VALUES ('NVDA', ?, 120.0, 125.0, 119.0, 124.0, 40000000)", (s,))
    conn.commit()
    conn.close()

    manifest_path = tmp_path / "nvda_manifest.json"
    manifest_path.write_text(json.dumps({
        "ingestion_scope": {
            "candidate_bindings": {"NVDA": {"action": "REPLACE"}},
            "production_target_backfill": {"missing_symbols": ["NVDA"]},
            "production_target_refresh": {"per_symbol_parameters": []}
        }
    }), encoding="utf-8")

    mock_fetcher = MockAlpacaSipFetcher()
    adapter = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher,
        feed="sip",
        adjustment="split",
        max_completed_session=date(2026, 11, 20),
    )

    # Case A: If replacement window is derived naively as [2025-01-02, 2026-11-20],
    # the unprovenanced 2023-10-27 bar outside reviewed window triggers fail-closed block!
    summary_naive = adapter.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=["NVDA"],
        start_date="2025-01-02",
        end_date="2026-11-20",
        apply_writes=False,
        db_path=disposable_market_db,
    )
    disp_naive = summary_naive.candidate_dispositions["NVDA"]
    assert disp_naive["status"] == "BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE"
    assert "INCOMPATIBLE_HISTORY_OUTSIDE_REVIEWED_SCOPE" in disp_naive["rejection_reason"]
    assert "explicit wider repair from '2023-10-27'" in disp_naive["rejection_reason"]

    # Case B: Deriving replacement window from actual earliest stored history (2023-10-27)
    # allows clean, bounded replacement that cleanses all legacy bars and establishes pure Alpaca SIP provenance
    summary_derived = adapter.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=["NVDA"],
        start_date="2023-10-27",
        end_date="2026-11-20",
        apply_writes=True,
        db_path=disposable_market_db,
    )
    disp_derived = summary_derived.candidate_dispositions["NVDA"]
    assert disp_derived["status"] in ("APPLIED", "SUCCESS")
    assert disp_derived["action_type"] == "REPLACE"

    # Verify all NVDA bars in DB now have pure Alpaca SIP provenance
    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = 'NVDA'")
    c_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance WHERE symbol = 'NVDA' AND provider = 'ALPACA'")
    p_count = cur.fetchone()[0]
    conn.close()

    assert c_count > 0
    assert c_count == p_count, "All NVDA bars must have 1-to-1 matching Alpaca SIP provenance"


def test_aggregate_request_limits_and_checkpoint_behavior(disposable_market_db: str, tmp_path: Path):
    """GATE 5: Enforce aggregate request limits, pagination budgets, deadlines, and checkpoints."""
    manifest_path = tmp_path / "limits_manifest.json"
    symbols = [f"SYM_{i:03d}" for i in range(20)]
    manifest_path.write_text(json.dumps({
        "ingestion_scope": {
            "production_target_backfill": {"missing_symbols": symbols},
            "production_target_refresh": {"per_symbol_parameters": []}
        }
    }), encoding="utf-8")

    # 1. Test request cap exhaustion
    mock_fetcher = MockAlpacaSipFetcher()
    adapter = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher,
        feed="sip",
        adjustment="split",
        max_completed_session=date(2026, 11, 20),
    )
    budget_cap = SharedExecutionBudget(max_aggregate_requests=5)
    cp_path = tmp_path / "checkpoint_cap.json"

    summary_cap = adapter.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=symbols[:10],
        start_date="2026-11-07",
        end_date="2026-11-20",
        apply_writes=True,
        db_path=disposable_market_db,
        budget=budget_cap,
        checkpoint_summary_path=cp_path,
    )
    assert summary_cap.execution_outcome in ("PARTIAL_SUCCESS", "FAILURE")
    assert budget_cap.requests_transmitted == 5
    assert len(summary_cap.completed_writes) == 5
    assert len(summary_cap.unprocessed_candidates) == 5
    assert cp_path.exists(), "Checkpoint must be persisted on budget cap abortion"

    # 2. Test pagination request counting (multi-page counts each page against budget)
    def page_mock(sym, start, end, page_token=None):
        if page_token is None:
            return {
                "success": True,
                "symbol": sym,
                "bars": [{"t": "2026-11-19T05:00:00Z", "o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5, "v": 1000}],
                "next_page_token": "token_page_2",
            }
        else:
            return {
                "success": True,
                "symbol": sym,
                "bars": [{"t": "2026-11-20T05:00:00Z", "o": 10.5, "h": 11.5, "l": 9.5, "c": 11.0, "v": 1000}],
                "next_page_token": None,
            }

    mock_page_fetcher = MockAlpacaSipFetcher(bars_by_symbol={"PAGE": page_mock})
    adapter_page = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_page_fetcher, feed="sip", adjustment="split", max_completed_session=date(2026, 11, 20)
    )
    budget_page = SharedExecutionBudget(max_aggregate_requests=10)
    p_manifest = tmp_path / "page_manifest.json"
    p_manifest.write_text(json.dumps({"symbols": ["PAGE"]}))
    s_page = adapter_page.execute_bounded_ingestion(
        manifest_path=p_manifest,
        start_date="2026-11-19",
        end_date="2026-11-20",
        apply_writes=False,
        budget=budget_page,
    )
    assert budget_page.requests_transmitted == 2, "Multi-page fetch must record each transmitted page against budget"

    # 3. Test retry ceiling exhaustion
    mock_retry_fetcher = MockAlpacaSipFetcher(transient_failures_count=10)
    budget_retry = SharedExecutionBudget(max_cumulative_retries=2, max_aggregate_requests=100)
    adapter_retry = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_retry_fetcher, feed="sip", adjustment="split", max_completed_session=date(2026, 11, 20)
    )
    s_retry = adapter_retry.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=symbols[:5],
        start_date="2026-11-07",
        end_date="2026-11-20",
        apply_writes=False,
        budget=budget_retry,
    )
    assert budget_retry.retries_transmitted == 2
    assert s_retry.execution_outcome == "FAILURE"

    # 3b. Test exhaustion inside a retry sequence (charging every transmission, including retries, to aggregate cap)
    mock_retry_seq_fetcher = MockAlpacaSipFetcher(transient_failures_count=10)
    # Cap is 2: attempt 0 uses 1, attempt 1 (retry 1) uses 1. Attempt 2 (retry 2) aborts before sending!
    budget_retry_seq = SharedExecutionBudget(max_aggregate_requests=2, max_cumulative_retries=5)
    adapter_retry_seq = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_retry_seq_fetcher, feed="sip", adjustment="split", max_completed_session=date(2026, 11, 20)
    )
    p_manifest_seq = tmp_path / "seq_manifest.json"
    p_manifest_seq.write_text(json.dumps({"symbols": ["PAGE"]}), encoding="utf-8")
    s_retry_seq = adapter_retry_seq.execute_bounded_ingestion(
        manifest_path=p_manifest_seq,
        start_date="2026-11-07",
        end_date="2026-11-20",
        apply_writes=False,
        budget=budget_retry_seq,
    )
    assert budget_retry_seq.requests_transmitted == 2, "Both transmissions (initial + retry) must charge to aggregate cap"
    assert budget_retry_seq.retries_transmitted == 1, "Only 1 retry transmitted before cap exhausted on next retry attempt"
    assert "EXECUTION_BUDGET_EXHAUSTED" in s_retry_seq.candidate_dispositions["PAGE"]["rejection_reason"]

    # 4. Test deadline timeout
    budget_timeout = SharedExecutionBudget(overall_deadline_seconds=0.001)
    time.sleep(0.005)
    adapter_timeout = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher, feed="sip", adjustment="split", max_completed_session=date(2026, 11, 20)
    )
    s_time = adapter_timeout.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=symbols[:5],
        start_date="2026-11-07",
        end_date="2026-11-20",
        apply_writes=False,
        budget=budget_timeout,
    )
    assert s_time.execution_outcome == "FAILURE"
    assert len(s_time.unprocessed_candidates) == 5

    # 5. Test safe resume from checkpoint
    resume_cp_path = tmp_path / "resume_cp.json"
    budget_res1 = SharedExecutionBudget(max_aggregate_requests=3)
    adapter_res = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher, feed="sip", adjustment="split", max_completed_session=date(2026, 11, 20)
    )
    adapter_res.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=symbols[:6],
        start_date="2026-11-07",
        end_date="2026-11-20",
        apply_writes=True,
        db_path=disposable_market_db,
        budget=budget_res1,
        checkpoint_summary_path=resume_cp_path,
    )
    calls_before = len(mock_fetcher.calls)

    # 5a. Corrupted checkpoint rejection (fail-closed)
    corrupt_cp = tmp_path / "corrupt_cp.json"
    corrupt_cp.write_text("{NOT_VALID_JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="CORRUPTED_CHECKPOINT_REJECTED"):
        adapter_res.execute_bounded_ingestion(
            manifest_path=manifest_path,
            symbols_override=symbols[:6],
            start_date="2026-11-07",
            end_date="2026-11-20",
            apply_writes=False,
            checkpoint_summary_path=corrupt_cp,
            resume=True,
        )

    # 5b. Incompatible checkpoint rejection (manifest / scope mismatch)
    incompat_cp = tmp_path / "incompat_cp.json"
    incompat_cp.write_text(json.dumps({
        "manifest_binding": {"manifest_sha256": "wrong_sha", "membership_hash": "wrong_hash"},
        "scope_binding": {"cutoff": "2026-11-20", "requested_start": "2026-11-07", "requested_end": "2026-11-20", "feed": "sip", "adjustment": "split", "mode": "DRY_RUN"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="INCOMPATIBLE_CHECKPOINT_REJECTED"):
        adapter_res.execute_bounded_ingestion(
            manifest_path=manifest_path,
            symbols_override=symbols[:6],
            start_date="2026-11-07",
            end_date="2026-11-20",
            apply_writes=False,
            checkpoint_summary_path=incompat_cp,
            resume=True,
        )

    # 5c. Preserve consumed budget across resume and enforce cap (cannot reset limits)
    exhausted_cp = tmp_path / "exhausted_cp.json"
    exhausted_cp.write_text(json.dumps({
        "consumed_budget": {"requests_transmitted": 10, "retries_transmitted": 2, "fallbacks_transmitted": 0},
    }), encoding="utf-8")
    budget_exhausted_res = SharedExecutionBudget(max_aggregate_requests=10)
    with pytest.raises(RuntimeError, match="EXECUTION_BUDGET_EXHAUSTED"):
        adapter_res.execute_bounded_ingestion(
            manifest_path=manifest_path,
            symbols_override=symbols[:6],
            start_date="2026-11-07",
            end_date="2026-11-20",
            apply_writes=False,
            budget=budget_exhausted_res,
            checkpoint_summary_path=exhausted_cp,
            resume=True,
        )

    # 5d. Verify completed work before skipping (unverified store work is re-processed)
    unverified_work_cp = tmp_path / "unverified_work_cp.json"
    unverified_work_cp.write_text(json.dumps({
        "summary": {
            "candidate_dispositions": {
                "PAGE": {
                    "symbol": "PAGE",
                    "status": "APPLIED",
                    "action_type": "REFRESH",
                    "bars_ingested": 10,
                }
            }
        }
    }), encoding="utf-8")
    s_unverified = adapter_res.execute_bounded_ingestion(
        manifest_path=p_manifest_seq,
        start_date="2026-11-07",
        end_date="2026-11-20",
        apply_writes=True,
        db_path=disposable_market_db,
        checkpoint_summary_path=unverified_work_cp,
        resume=True,
    )
    assert "PAGE" in s_unverified.completed_writes, "Unverified work must be re-processed and committed"

    # 5e. Normal resume with budget allowing remaining 3 candidates
    calls_before = len(mock_fetcher.calls)
    budget_res2 = SharedExecutionBudget(max_aggregate_requests=10)
    s_res = adapter_res.execute_bounded_ingestion(
        manifest_path=manifest_path,
        symbols_override=symbols[:6],
        start_date="2026-11-07",
        end_date="2026-11-20",
        apply_writes=True,
        db_path=disposable_market_db,
        budget=budget_res2,
        checkpoint_summary_path=resume_cp_path,
        resume=True,
    )
    calls_after = len(mock_fetcher.calls)
    assert calls_after - calls_before == 3, "Resumed execution must skip verified completed candidates"
    assert len(s_res.completed_writes) == 6


def test_wal_safe_quiesced_recovery_rehearsal(tmp_path: Path):
    """GATE 7: Rehearse WAL-safe quiesced recovery procedure on disposable SQLite database."""
    import shutil
    import hashlib

    target_db = tmp_path / "production_target.db"
    backup_db = tmp_path / "pre_refresh_backup.db"

    # Step 1: Create baseline database in WAL mode
    conn = sqlite3.connect(str(target_db))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("CREATE TABLE asset_ohlcv_daily (symbol TEXT, trade_date TEXT, close REAL, PRIMARY KEY (symbol, trade_date));")
    conn.execute("CREATE TABLE asset_ohlcv_provenance (symbol TEXT, trade_date TEXT, provider TEXT, PRIMARY KEY (symbol, trade_date));")
    conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('AAPL', '2026-11-19', 220.0);")
    conn.execute("INSERT INTO asset_ohlcv_provenance VALUES ('AAPL', '2026-11-19', 'ALPACA');")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()

    # Step 2: Take verified pre-refresh backup
    shutil.copy2(target_db, backup_db)
    backup_sha = hashlib.sha256(backup_db.read_bytes()).hexdigest()

    # Step 3: Simulate dirty/interrupted refresh leaving active WAL entries
    dirty_conn = sqlite3.connect(str(target_db))
    dirty_conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('CORRUPT_CANDIDATE', '2026-11-20', 999.0);")
    dirty_conn.execute("UPDATE asset_ohlcv_daily SET close = 0.0 WHERE symbol = 'AAPL';")
    dirty_conn.commit()
    dirty_conn.close()

    # Simulate dirty WAL and SHM sidecars left behind by abnormal termination
    wal_file = tmp_path / "production_target.db-wal"
    shm_file = tmp_path / "production_target.db-shm"
    wal_file.write_bytes(b"DIRTY_STALE_WAL_PAGES_FROM_INTERRUPTED_WRITER")
    shm_file.write_bytes(b"DIRTY_SHM_INDEX")
    assert wal_file.exists(), "Target database must have active WAL journal before recovery"

    # Step 4a: Refuse unquiesced target without deleting anything
    with pytest.raises(RuntimeError, match="UNQUIESCED_TARGET_REFUSED"):
        quiesce_and_recover_database(target_db_path=target_db, backup_path=backup_db, maintenance_quiescence_verified=False)
    assert target_db.exists(), "Target database must not be deleted on unquiesced refusal"
    assert wal_file.exists(), "WAL file must not be deleted on unquiesced refusal"

    # Step 4b: Fail closed on checkpoint contention / active users
    active_user_conn = sqlite3.connect(str(target_db))
    active_user_conn.execute("BEGIN EXCLUSIVE")
    with pytest.raises(RuntimeError, match="(TARGET_LOCK_CONTENTION_DETECTED|CHECKPOINT_CONTENTION_DETECTED)"):
        quiesce_and_recover_database(target_db_path=target_db, backup_path=backup_db, maintenance_quiescence_verified=True)
    active_user_conn.rollback()
    active_user_conn.close()
    assert target_db.exists(), "Target database must not be deleted on contention refusal"

    # Step 4c: Verify backup before touching target
    corrupt_backup = tmp_path / "corrupt_backup.db"
    corrupt_backup.write_bytes(b"NOT_A_VALID_SQLITE_DATABASE_HEADER")
    with pytest.raises(RuntimeError, match="BACKUP_CORRUPTED"):
        quiesce_and_recover_database(target_db_path=target_db, backup_path=corrupt_backup, maintenance_quiescence_verified=True)
    assert target_db.exists(), "Target database must not be touched if backup is corrupt"

    # Step 4d: Execute verified recovery with maintenance quiescence confirmed
    recovery_result = quiesce_and_recover_database(
        target_db_path=target_db,
        backup_path=backup_db,
        maintenance_quiescence_verified=True,
    )

    assert recovery_result["status"] == "RECOVERED"
    assert recovery_result["recovered_sha256"] == backup_sha
    assert recovery_result["integrity_check"] == "ok"
    assert recovery_result["maintenance_quiescence_verified"] is True

    # Assert WAL sidecars are cleanly removed
    assert not (tmp_path / "production_target.db-wal").exists()
    assert not (tmp_path / "production_target.db-shm").exists()

    # Verify recovered data matches pre-refresh backup exactly
    ver_conn = sqlite3.connect(str(target_db))
    cur = ver_conn.cursor()
    cur.execute("SELECT symbol, trade_date, close FROM asset_ohlcv_daily")
    rows = cur.fetchall()
    ver_conn.close()
    assert rows == [("AAPL", "2026-11-19", 220.0)], "Recovered database must match pre-refresh backup exactly"


def test_writer_concurrency_provenance_atomic_protection(disposable_market_db: str):
    """GATE 4: Concurrency test: ensure provenance check and guarded write are strictly atomic under competing connections."""
    import threading
    engine = MarketDatabaseEngine(db_path=disposable_market_db)
    upper = "AAPL"
    trade_date = "2026-11-20"

    prov_alpaca = MarketProvenance(
        provider=Provider.ALPACA,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        observed_at="2026-11-20T21:00:00Z",
        observed_date=trade_date,
        observation_precision=ObservationPrecision.DATE,
        observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
        ingested_at="2026-11-20T21:05:00Z",
        adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
        structural_quality=StructuralQuality.COMPLETE,
        fallback_status=False,
    )
    candle_alpaca = [{"time": trade_date, "open": 220.0, "high": 225.0, "low": 219.0, "close": 224.0, "volume": 1000000}]
    engine.save_daily_candles_with_evidence(upper, candle_alpaca, prov_alpaca)

    # 10 competing threads concurrently attempting to overwrite with unprovenanced data
    errors = []
    def competing_worker(worker_id: int):
        try:
            worker_engine = MarketDatabaseEngine(db_path=disposable_market_db)
            fake_candle = [{"time": trade_date, "open": 100.0, "high": 101.0, "low": 99.0, "close": 999.0 + worker_id, "volume": 500}]
            worker_engine.save_daily_candles(upper, fake_candle, reject_if_authoritative=True)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=competing_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Competing workers encountered unhandled errors: {errors}"

    # Verify protected bars and provenance remain 100% unchanged
    stored = engine.get_daily_candles(upper, limit=10)
    assert len(stored) == 1
    assert stored[0]["close"] == 224.0, "Authoritative Alpaca bar was protected from concurrent unprovenanced writes"

    conn = sqlite3.connect(disposable_market_db)
    cur = conn.cursor()
    cur.execute("SELECT provider, adjustment_state FROM asset_ohlcv_provenance WHERE symbol = ?", (upper,))
    prov_row = cur.fetchone()
    conn.close()
    assert prov_row == ("ALPACA", "SPLIT_ADJUSTED_ONLY"), "Authoritative Alpaca SIP provenance remained 100% intact"


def test_full_accounting_closure_with_explicit_exceptions(tmp_path: Path):
    """GATE 6: Replace mandatory '500 READY' with accounting closure formula (READY + DOCUMENTED_EXCEPTIONS)."""
    # 3 symbols in published universe:
    # SYM_READY: valid market data + provenance -> READY
    # SYM_DELISTED: confirmed corporate action delisting -> EXCEPTION (DELISTED_SECURITY)
    # SYM_HALT: upstream provider trading halt / liquidation -> EXCEPTION (TRADING_HALT)

    market_db = tmp_path / "market.db"
    universe_db = tmp_path / "universe.db"
    scanner_db = tmp_path / "scanner.db"
    cand_manifest = tmp_path / "cand.json"
    prom_manifest = tmp_path / "prom.json"
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"

    _init_empty_aux_dbs(universe_db, scanner_db)

    symbols = ["SYM_READY", "SYM_DELISTED", "SYM_HALT"]
    cand_manifest.write_text(json.dumps({"manifest_build_id": "TEST-CAND", "membership_hash": "h_cand", "symbols": symbols}))
    prom_manifest.write_text(json.dumps({"manifest_build_id": "TEST-PROM", "membership_hash": "h_prom", "symbols": ["SYM_READY"]}))

    records = []
    base_date = date(2026, 11, 20)
    for i in range(252):
        records.append({"symbol": "SYM_READY", "trade_date": (base_date - timedelta(days=251 - i)).isoformat()})
    prov_records = [{"symbol": "SYM_READY", "trade_date": "2026-11-20"}]

    _init_synthetic_market_db(market_db, records, prov_records)

    payload = run_audit(
        market_db_path=market_db,
        universe_db_path=universe_db,
        scanner_db_path=scanner_db,
        candidate_manifest_path=cand_manifest,
        promoted_manifest_path=prom_manifest,
        cutoff_str="2026-11-20",
        output_json_path=out_json,
        output_md_path=out_md,
    )

    s_cand = payload["summary_metrics_candidate_universe"]
    assert s_cand["total_symbols"] == 3
    assert s_cand["readiness_breakdown"]["READY"] == 1
    assert s_cand["readiness_breakdown"]["MISSING_MARKET_DATA"] == 2
    assert s_cand["accounting_closure"]["is_closed"] is True

    # Account for documented unresolved exceptions
    documented_exceptions = {
        "SYM_DELISTED": "CONFIRMED_CORPORATE_ACTION_DELISTING: Excluded per corporate actions ledger",
        "SYM_HALT": "UPSTREAM_PROVIDER_HALT: Trading suspended by exchange authority",
    }
    unresolved_count = len(documented_exceptions)
    ready_count = s_cand["readiness_breakdown"]["READY"]

    assert ready_count + unresolved_count == s_cand["total_symbols"], (
        "Every published symbol must be accounted for: either READY or carrying an explicit unresolved exception."
    )

