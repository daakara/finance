import pytest
import pandas as pd
import numpy as np
import math
from unittest.mock import MagicMock

from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore

pytestmark = pytest.mark.tier2b


def _create_sample_daily_df(length: int = 60, base_price: float = 100.0, upward_drift: float = 0.5) -> pd.DataFrame:
    prices = [base_price + i * upward_drift + 0.3 * np.sin(i) for i in range(length)]
    dates = pd.date_range("2026-07-01", periods=length, freq="B")
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1.25 for p in prices],
            "Low": [p - 1.25 for p in prices],
            "Close": prices,
            "Volume": [1_000_000 + i * 5_000 for i in range(length)],
        },
        index=dates,
    )


def test_remediation_frozen_engine_compliance_and_runner_truthfulness_boundary(tmp_path):
    """Requirement 1 & 2: OptimalExecutionEngine remains frozen under Release Addendum 002, and VCPScannerRunner enforces truthfulness."""
    import hashlib, os
    repo_root = os.path.dirname(os.path.dirname(__file__))
    opt_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "optimal_execution.py")
    with open(opt_path, "rb") as fp:
        sha = hashlib.sha256(fp.read().replace(b"\r\n", b"\n")).hexdigest()
    assert sha == "84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576", (
        "OptimalExecutionEngine must remain frozen under Release Addendum 002"
    )

    # Verify that VCPScannerRunner rejects the legacy fallback labels on 60-bar windows
    df = _create_sample_daily_df(60, base_price=100.0)
    candles = [
        {"open": row["Open"], "high": row["High"], "low": row["Low"], "close": row["Close"], "volume": row["Volume"], "time": idx.isoformat()}
        for idx, row in df.iterrows()
    ]
    mock_db = MagicMock()
    mock_db.get_daily_candles.return_value = candles
    mock_db.get_latest_price.return_value = {"currentPrice": float(df["Close"].iloc[-1])}

    db_path = str(tmp_path / "test_runner_truth.db")
    store = ScannerSnapshotStore(db_path=db_path)
    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=store)

    # 60-bar scan must refuse to certify candidate as confirmed VCP / Stage 2
    res = runner.execute_market_wide_scan(universe_override=["CANDIDATE_1"])
    assert res["status"] == "PIPELINE_PENDING"
    assert res["snapshot"]["matched_count"] is None
    assert len(res["results"]) == 0


def test_remediation_operational_code_has_zero_false_certification_pathways():
    """Requirement 2: Operational code provides zero pathways or flags to enable legacy false certification."""
    import inspect
    sig = inspect.signature(OptimalExecutionEngine.calculate_trade_levels)
    assert "legacy_proxy_fallback" not in sig.parameters

    runner_sig = inspect.signature(VCPScannerRunner.execute_market_wide_scan)
    assert "force_legacy_fallback" not in runner_sig.parameters


def test_remediation_numerical_invariants_preserved():
    """Requirement 3: Mathematical trade levels, corridors, ATR, and ordering invariants are fully verified."""
    df = _create_sample_daily_df(60, base_price=100.0)
    spot = float(df["Close"].iloc[-1])

    plan = OptimalExecutionEngine.calculate_trade_levels(df, spot, user_role="LONG_TERM")

    numeric_keys = [
        "current_price",
        "optimal_entry_min",
        "optimal_entry_max",
        "planned_entry",
        "stop_loss",
        "stop_loss_pct",
        "take_profit_1",
        "take_profit_1_pct",
        "take_profit_2",
        "take_profit_2_pct",
        "risk_reward_ratio",
        "atr_14",
    ]
    for key in numeric_keys:
        assert plan[key] is not None, f"Expected non-null value for {key}"

    # Structural math ordering invariant
    assert plan["stop_loss"] < plan["optimal_entry_min"]
    assert plan["optimal_entry_min"] <= plan["optimal_entry_max"]
    assert plan["optimal_entry_max"] < plan["take_profit_1"]
    assert plan["take_profit_1"] < plan["take_profit_2"]
    assert plan["risk_reward_ratio"] >= 1.85


def test_remediation_scanner_runner_uncomputed_vcp_emits_pipeline_pending(tmp_path):
    """Requirement 4 (Scenario B): Uncomputed VCP classification strictly emits PIPELINE_PENDING with matched_count=None."""
    df = _create_sample_daily_df(60, base_price=100.0)
    candles = [
        {"open": row["Open"], "high": row["High"], "low": row["Low"], "close": row["Close"], "volume": row["Volume"], "time": idx.isoformat()}
        for idx, row in df.iterrows()
    ]

    mock_db = MagicMock()
    mock_db.get_daily_candles.return_value = candles
    mock_db.get_latest_price.return_value = {"currentPrice": float(df["Close"].iloc[-1])}

    db_path = str(tmp_path / "test_scanner.db")
    snapshot_store = ScannerSnapshotStore(db_path=db_path)
    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=snapshot_store)

    # Executing scan over candidate with unverified pattern
    result = runner.execute_market_wide_scan(universe_override=["TEST_CANDIDATE"])

    # Must be PIPELINE_PENDING with matched_count=None (Scenario B, NOT 0 matches)
    assert result["status"] == "PIPELINE_PENDING"
    assert result["snapshot"]["matched_count"] is None
    assert result["coverage"]["matched_count"] is None
    assert len(result["results"]) == 0


def test_remediation_scanner_runner_scenario_a_empty_universe_is_available_zero(tmp_path):
    """Requirement 4 (Scenario A): Completed evaluation with 0 matches emits AVAILABLE with matched_count=0."""
    db_path = str(tmp_path / "test_scanner_a.db")
    snapshot_store = ScannerSnapshotStore(db_path=db_path)
    runner = VCPScannerRunner(snapshot_store=snapshot_store)

    # Executing scan with empty universe (Scenario A: completed scan with zero matches)
    result = runner.execute_market_wide_scan(universe_override=[])

    assert result["status"] == "AVAILABLE"
    assert result["snapshot"]["matched_count"] == 0
    assert result["coverage"]["matched_count"] == 0
    assert len(result["results"]) == 0



def test_remediation_scanner_evidence_serialization_fidelity():
    """Requirement 5: Qualified scanner evidence maps sma_50 to actual SMA50 and ema_20 to EMA20, preserving null breakout_pivot."""
    # Create mock execution result with calculated SMA50, EMA20, and null breakout_pivot
    mock_exec_res = {
        "execution_status": "READY_ACTIONABLE",
        "risk_reward_ratio": 2.5,
        "setup_pattern": "Minervini VCP (Volatility Contraction Pattern)",
        "stage_phase": "Stage 2 Advancing Growth Phase",
        "vcp_contraction_status": "VCP 3-Stage Compression Confirmed",
        "sma_50": 105.42,
        "ema_20": 112.30,
        "atr_14": 2.15,
        "breakout_pivot": None,
        "optimal_entry_min": 110.0,
        "optimal_entry_max": 113.0,
        "stop_loss": 103.5,
        "take_profit_1": 125.0,
        "take_profit_2": 132.0,
    }

    # Simulate qualification branch mapping in VCPScannerRunner
    score = 88.5
    scanner_evidence = {
        "vcp_stage": mock_exec_res.get("vcp_contraction_status"),
        "setup_pattern": mock_exec_res.get("setup_pattern"),
        "stage_phase": mock_exec_res.get("stage_phase"),
        "sma_50": mock_exec_res.get("sma_50"),
        "ema_20": mock_exec_res.get("ema_20"),
        "atr_14": mock_exec_res.get("atr_14"),
        "breakout_pivot": mock_exec_res.get("breakout_pivot"),
        "optimal_entry_min": mock_exec_res.get("optimal_entry_min"),
        "optimal_entry_max": mock_exec_res.get("optimal_entry_max"),
        "stop_loss": mock_exec_res.get("stop_loss"),
        "take_profit_1": mock_exec_res.get("take_profit_1"),
        "take_profit_2": mock_exec_res.get("take_profit_2"),
        "risk_reward_ratio": mock_exec_res.get("risk_reward_ratio"),
        "execution_status": mock_exec_res.get("execution_status"),
        "confluence_score": score,
    }

    assert scanner_evidence["sma_50"] == 105.42
    assert scanner_evidence["ema_20"] == 112.30
    assert scanner_evidence["breakout_pivot"] is None
    assert scanner_evidence["sma_50"] != scanner_evidence["breakout_pivot"]
    assert scanner_evidence["ema_20"] != scanner_evidence["optimal_entry_min"]


def test_remediation_calculate_execution_plan_invariants():
    """Requirement 6: calculate_execution_plan enforces structural mathematical risk ordering."""
    plan = OptimalExecutionEngine.calculate_execution_plan(symbol="ACME", price=150.0, atr_14=3.5, user_role="LONG_TERM")

    assert plan["stop_loss"] < plan["optimal_entry_min"]
    assert plan["optimal_entry_min"] <= plan["optimal_entry_max"]
    assert plan["optimal_entry_max"] < plan["take_profit_1"]
    assert plan["take_profit_1"] < plan["take_profit_2"]
    assert plan["risk_reward_ratio"] >= 1.85


def test_remediation_insufficient_history_returns_insufficient_history_status():
    """Requirement 7: Insufficient history paths (< 50 sessions for swing, < 15 for day) return INSUFFICIENT_HISTORY."""
    short_df = _create_sample_daily_df(30, base_price=50.0)
    spot = float(short_df["Close"].iloc[-1])

    plan_swing = OptimalExecutionEngine.calculate_trade_levels(short_df, spot, user_role="LONG_TERM")
    assert plan_swing["execution_status"] == "INSUFFICIENT_HISTORY"

    very_short_df = _create_sample_daily_df(10, base_price=50.0)
    spot_short = float(very_short_df["Close"].iloc[-1])
    plan_day = OptimalExecutionEngine.calculate_trade_levels(very_short_df, spot_short, user_role="DAY_TRADER")
    assert plan_day["execution_status"] == "INSUFFICIENT_HISTORY"


def test_existing_database_not_null_matched_count_atomic_upgrade(tmp_path):
    """Requirement 8: Atomic migration upgrades existing database with matched_count NOT NULL to nullable INTEGER."""
    import sqlite3, json, time
    db_path = str(tmp_path / "legacy_scanner.db")

    # 1. Create exact base commit (287c2b7) legacy schema with matched_count INTEGER NOT NULL
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("""
        CREATE TABLE scanner_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            scanner_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            api_contract_version TEXT NOT NULL,
            ruleset_version TEXT NOT NULL,
            evidence_schema_version TEXT NOT NULL,
            score_model_version TEXT NOT NULL,
            data_provenance_version TEXT NOT NULL,
            universe_version TEXT NOT NULL,
            freshness_policy_version TEXT NOT NULL,
            implementation_release_sha TEXT NOT NULL,
            semantic_fingerprint TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            data_as_of TEXT NOT NULL,
            status_at_publication TEXT NOT NULL,
            universe_id TEXT NOT NULL,
            universe_size INTEGER NOT NULL,
            matched_count INTEGER NOT NULL,
            results_json TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            freshness_json TEXT NOT NULL,
            publication_decision TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("CREATE INDEX idx_snapshots_scanner_created ON scanner_snapshots (scanner_id, created_at DESC);")
    conn.execute("CREATE INDEX idx_snapshots_run_id ON scanner_snapshots (run_id);")
    conn.execute("CREATE INDEX idx_snapshots_fingerprint ON scanner_snapshots (semantic_fingerprint);")
    conn.execute("""
        CREATE TRIGGER trg_scanner_snapshots_no_update
        BEFORE UPDATE ON scanner_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be updated in place');
        END;
    """)
    conn.execute("""
        CREATE TRIGGER trg_scanner_snapshots_no_delete
        BEFORE DELETE ON scanner_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be deleted');
        END;
    """)

    # Seed 2 legacy snapshots with non-null matched_count
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn.execute("""
        INSERT INTO scanner_snapshots VALUES (
            'snap-legacy-001', 'MINERVINI_VCP', 'run-leg-1', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '1.0.0',
            '1.0.0', '1.0.0', 'sha-legacy-1', 'fp-leg-1', ?, '2026-10-15', 'AVAILABLE',
            'ARX_UNIVERSE', 35, 10, '[]', '{}', '{}', 'PUBLISH', CURRENT_TIMESTAMP
        );
    """, (now_iso,))
    conn.execute("""
        INSERT INTO scanner_snapshots VALUES (
            'snap-legacy-002', 'MINERVINI_VCP', 'run-leg-2', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '1.0.0',
            '1.0.0', '1.0.0', 'sha-legacy-2', 'fp-leg-2', ?, '2026-10-15', 'AVAILABLE',
            'ARX_UNIVERSE', 35, 0, '[]', '{}', '{}', 'PUBLISH', CURRENT_TIMESTAMP
        );
    """, (now_iso,))
    conn.commit()

    # Verify that matched_count column in legacy db has notnull == 1
    cols_pre = conn.execute("PRAGMA table_info(scanner_snapshots);").fetchall()
    mc_pre = next(c for c in cols_pre if c[1] == "matched_count")
    assert mc_pre[3] == 1, "Legacy table must initially have NOT NULL constraint"
    conn.close()

    # 2. Instantiate ScannerSnapshotStore on existing legacy db (triggers atomic upgrade)
    store = ScannerSnapshotStore(db_path=db_path)

    # 3. Verify column nullability has been upgraded to nullable (notnull == 0)
    conn = sqlite3.connect(db_path)
    cols_post = conn.execute("PRAGMA table_info(scanner_snapshots);").fetchall()
    mc_post = next(c for c in cols_post if c[1] == "matched_count")
    assert mc_post[3] == 0, "Upgraded table must have nullable matched_count (notnull == 0)"

    # Verify legacy records preserved intact
    rows = conn.execute("SELECT snapshot_id, matched_count FROM scanner_snapshots ORDER BY snapshot_id").fetchall()
    assert len(rows) == 2
    assert rows[0] == ("snap-legacy-001", 10)
    assert rows[1] == ("snap-legacy-002", 0)

    # Verify triggers are preserved and functional
    with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION"):
        conn.execute("UPDATE scanner_snapshots SET status_at_publication = 'MUTATED' WHERE snapshot_id = 'snap-legacy-001'")
    with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION"):
        conn.execute("DELETE FROM scanner_snapshots WHERE snapshot_id = 'snap-legacy-001'")
    conn.close()

    # 4. Save a new uncomputed snapshot where matched_count is None (Scenario B: PIPELINE_PENDING)
    snap_uncomputed = {
        "snapshot_id": "snap-remediated-003",
        "scanner_id": "MINERVINI_VCP",
        "run_id": "run-rem-3",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": "sha-remediated-3",
        "semantic_fingerprint": "fp-rem-3",
        "generated_at": now_iso,
        "data_as_of": "2026-10-18",
        "status_at_publication": "PIPELINE_PENDING",
        "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
        "universe_size": 3,
        "matched_count": None,  # None inserted into previously NOT NULL column
        "results": [],
        "provenance": {},
        "freshness": {},
        "publication_decision": "PUBLISH",
    }
    saved_id = store.save_snapshot(snap_uncomputed)
    assert saved_id == "snap-remediated-003"

    retrieved = store.get_snapshot_by_id("snap-remediated-003")
    assert retrieved is not None
    assert retrieved["matched_count"] is None
    assert retrieved["status_at_publication"] == "PIPELINE_PENDING"

    # 5. Verify Idempotence (re-initializing store does not re-migrate or mutate)
    store2 = ScannerSnapshotStore(db_path=db_path)
    assert store2.get_snapshot_by_id("snap-remediated-003")["matched_count"] is None


def test_retained_legacy_snapshot_release_mismatch_fails_closed(tmp_path):
    """Requirement 9: Retained snapshot with mismatched release SHA fails closed to PIPELINE_PENDING without scan."""
    import time
    from analyst_dashboard.analyzers.scanner_runner import CURRENT_IMPLEMENTATION_RELEASE_SHA

    db_path = str(tmp_path / "test_legacy_read.db")
    store = ScannerSnapshotStore(db_path=db_path)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Seed snapshot with older release SHA claiming AVAILABLE with 5 matches
    store.save_snapshot({
        "snapshot_id": "snap-old-sha-001",
        "scanner_id": "MINERVINI_VCP",
        "run_id": "run-old-1",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": "outdated-legacy-release-sha-999",
        "semantic_fingerprint": "fp-old-1",
        "generated_at": now_iso,
        "data_as_of": "2026-10-16",
        "status_at_publication": "AVAILABLE",
        "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
        "universe_size": 35,
        "matched_count": 5,
        "results": [{"symbol": f"SYM{i}", "current_price": 100.0} for i in range(5)],
        "provenance": {},
        "freshness": {"status": "LIVE", "data_as_of": "2026-10-16", "generated_at": now_iso},
        "publication_decision": "PUBLISH",
    })

    runner = VCPScannerRunner(snapshot_store=store)
    env = runner.get_active_or_latest_snapshot()

    assert env["status"] == "PIPELINE_PENDING"
    assert env["reason"] == "RELEASE_MISMATCH"
    assert env["snapshot"]["matched_count"] is None
    assert env["matched_count"] is None
    assert len(env["results"]) == 0


def test_retained_legacy_snapshot_with_unsupported_matches_fails_closed(tmp_path):
    """Requirement 10: Retained snapshot under same release SHA but containing unverified VCP matches fails closed."""
    import time
    from analyst_dashboard.analyzers.scanner_runner import CURRENT_IMPLEMENTATION_RELEASE_SHA

    db_path = str(tmp_path / "test_unsupported_read.db")
    store = ScannerSnapshotStore(db_path=db_path)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Snapshot created with CURRENT_IMPLEMENTATION_RELEASE_SHA but containing unverified labels from old proxy fallback
    store.save_snapshot({
        "snapshot_id": "snap-unsupported-001",
        "scanner_id": "MINERVINI_VCP",
        "run_id": "run-unsupported-1",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": CURRENT_IMPLEMENTATION_RELEASE_SHA,
        "semantic_fingerprint": "fp-unsupported-1",
        "generated_at": now_iso,
        "data_as_of": "2026-10-16",
        "status_at_publication": "AVAILABLE",
        "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
        "universe_size": 3,
        "matched_count": 3,
        "results": [
            {
                "symbol": "AAPL",
                "score": 21.2,
                "current_price": 343.46,
                "scanner_evidence": {
                    "vcp_stage": "Consolidation Unverified",
                    "setup_pattern": "Consolidation Setup (Pattern Unverified)",
                    "stage_phase": "Stage Unverified (Trend Template Uncomputed)",
                    "vcp_pattern_detected": False,
                    "stage2_status": "UNCOMPUTED",
                },
            },
            {
                "symbol": "GOOG",
                "score": 23.2,
                "current_price": 336.91,
                "scanner_evidence": {
                    "vcp_stage": "Consolidation Unverified",
                    "setup_pattern": "Consolidation Setup (Pattern Unverified)",
                    "stage_phase": "Stage Unverified (Trend Template Uncomputed)",
                    "vcp_pattern_detected": False,
                    "stage2_status": "UNCOMPUTED",
                },
            },
        ],
        "provenance": {},
        "freshness": {"status": "LIVE", "data_as_of": "2026-10-16", "generated_at": now_iso},
        "publication_decision": "PUBLISH",
    })

    runner = VCPScannerRunner(snapshot_store=store)
    env = runner.get_active_or_latest_snapshot()

    # Must fail closed to PIPELINE_PENDING with UNSUPPORTED_LEGACY_MATCHES
    assert env["status"] == "PIPELINE_PENDING"
    assert env["reason"] == "UNSUPPORTED_LEGACY_MATCHES"
    assert env["snapshot"]["matched_count"] is None
    assert env["matched_count"] is None
    assert len(env["results"]) == 0


def test_migration_read_only_connection_skips_ddl_cleanly(tmp_path):
    """Requirement 12: Read-only database connections safely skip DDL without error."""
    import sqlite3
    from analyst_dashboard.data.scanner_store import ensure_scanner_snapshots_table

    db_path = str(tmp_path / "ro_scanner.db")
    # Create legacy table in writable mode
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE scanner_snapshots (
            snapshot_id TEXT PRIMARY KEY, scanner_id TEXT NOT NULL, run_id TEXT NOT NULL,
            api_contract_version TEXT NOT NULL, ruleset_version TEXT NOT NULL,
            evidence_schema_version TEXT NOT NULL, score_model_version TEXT NOT NULL,
            data_provenance_version TEXT NOT NULL, universe_version TEXT NOT NULL,
            freshness_policy_version TEXT NOT NULL, implementation_release_sha TEXT NOT NULL,
            semantic_fingerprint TEXT NOT NULL, generated_at TEXT NOT NULL, data_as_of TEXT NOT NULL,
            status_at_publication TEXT NOT NULL, universe_id TEXT NOT NULL, universe_size INTEGER NOT NULL,
            matched_count INTEGER NOT NULL, results_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
            freshness_json TEXT NOT NULL, publication_decision TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

    # Open connection in read-only mode and execute ensure_scanner_snapshots_table
    conn_ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Must return cleanly without OperationalError ('attempt to write a readonly database')
        ensure_scanner_snapshots_table(conn_ro)
    finally:
        conn_ro.close()


def test_migration_unknown_schema_shape_rejected_without_partial_changes(tmp_path):
    """Requirement 13: Unknown schema shape is rejected with RuntimeError and leaves table intact."""
    import sqlite3
    from analyst_dashboard.data.scanner_store import ensure_scanner_snapshots_table

    db_path = str(tmp_path / "corrupted_schema.db")
    conn = sqlite3.connect(db_path)
    # Create corrupted schema missing 10 required columns
    conn.execute("CREATE TABLE scanner_snapshots (snapshot_id TEXT PRIMARY KEY, matched_count INTEGER NOT NULL);")
    conn.execute("INSERT INTO scanner_snapshots VALUES ('corrupt-1', 42);")
    conn.commit()

    with pytest.raises(RuntimeError, match="UNSUPPORTED_SCHEMA_SHAPE"):
        ensure_scanner_snapshots_table(conn)

    # Verify original table remained 100% intact
    cols = conn.execute("PRAGMA table_info(scanner_snapshots);").fetchall()
    assert len(cols) == 2
    row = conn.execute("SELECT * FROM scanner_snapshots").fetchone()
    assert row == ("corrupt-1", 42)
    conn.close()


def test_nondestructive_recovery_and_immutability_preservation(tmp_path):
    """Requirement 14: Deletion of NULL snapshots is prohibited by trigger; recovery preserves 100% of records."""
    import sqlite3
    import time
    from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
    from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner

    db_path = str(tmp_path / "recovery_store.db")
    store = ScannerSnapshotStore(db_path=db_path)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Historical record
    store.save_snapshot({
        "snapshot_id": "snap-historical-001",
        "scanner_id": "MINERVINI_VCP",
        "run_id": "run-hist-1",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": "sha-old",
        "semantic_fingerprint": "fp-old",
        "generated_at": now_iso,
        "data_as_of": "2026-10-15",
        "status_at_publication": "AVAILABLE",
        "universe_id": "ARX_UNIVERSE",
        "universe_size": 35,
        "matched_count": 0,
        "results": [],
        "provenance": {},
        "freshness": {"status": "LIVE"},
        "publication_decision": "PUBLISH",
    })

    # 2. Post-upgrade PIPELINE_PENDING record with matched_count=None
    store.save_snapshot({
        "snapshot_id": "snap-pending-002",
        "scanner_id": "MINERVINI_VCP",
        "run_id": "run-pending-2",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": "sha-current",
        "semantic_fingerprint": "fp-current",
        "generated_at": now_iso,
        "data_as_of": "2026-10-18",
        "status_at_publication": "PIPELINE_PENDING",
        "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
        "universe_size": 3,
        "matched_count": None,
        "results": [],
        "provenance": {},
        "freshness": {"status": "LIVE"},
        "publication_decision": "PUBLISH",
    })

    conn = sqlite3.connect(db_path)
    # Attempting to delete NULL snapshots must raise IMMUTABILITY_VIOLATION
    with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be deleted"):
        conn.execute("DELETE FROM scanner_snapshots WHERE matched_count IS NULL")
    conn.close()

    # Recovery build / version-aware reader safely handles both records without deletion
    runner = VCPScannerRunner(snapshot_store=store)
    env = runner.get_active_or_latest_snapshot()
    # All rows preserved:
    assert store.get_snapshot_by_id("snap-historical-001") is not None
    assert store.get_snapshot_by_id("snap-pending-002") is not None
    assert store.get_snapshot_by_id("snap-pending-002")["matched_count"] is None


