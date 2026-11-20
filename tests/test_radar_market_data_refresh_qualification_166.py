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
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from scripts.ingest_alpaca_historical import (
    AlpacaHistoricalSipIngestionAdapter,
    get_expected_nyse_sessions,
)


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

    def __init__(self, bars_by_symbol=None):
        self.bars_by_symbol = bars_by_symbol or {}
        self.calls = []

    def fetch_historical_bars(self, symbol, start, end, feed=None, adjustment=None, limit=1000, page_token=None, **kwargs):
        self.calls.append({"symbol": symbol, "start": start, "end": end, "feed": feed, "adjustment": adjustment, "page_token": page_token})
        if symbol in self.bars_by_symbol:
            return self.bars_by_symbol[symbol]

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
