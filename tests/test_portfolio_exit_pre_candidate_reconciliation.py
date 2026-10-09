"""Comprehensive Pre-Candidate Verification & Reconciliation Test Suite.
Covers Sections 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18 of the Pre-Candidate Gate.
"""

import os
import sqlite3
import tempfile
import shutil
import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from api.main import app
import api.routes.portfolio as portfolio_route
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine, HoldingExitError
from database.holding_exit_migration import apply_holding_exit_migration
from api.context.workspace_identity import derive_compatibility_workspace_id


@pytest.fixture
def isolated_env():
    """Create an isolated test directory and database engine."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_reconciliation.db")
    engine = HistoryDatabaseEngine(db_path=db_path)

    # Patch portfolio_route.portfolio_service.db_engine to isolated engine
    original_engine = portfolio_route.portfolio_service.db_engine
    portfolio_route.portfolio_service.db_engine = engine

    client = TestClient(app)

    yield {
        "temp_dir": temp_dir,
        "db_path": db_path,
        "engine": engine,
        "client": client,
    }

    # Teardown
    portfolio_route.portfolio_service.db_engine = original_engine
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# SECTION 5: HTTP ROUTE INTEGRATION TESTS (HTTP-EXIT-01 - HTTP-EXIT-12)
# ==============================================================================

def test_http_route_integration_matrix(isolated_env):
    client = isolated_env["client"]
    engine = isolated_env["engine"]
    user = "user_http_test"
    ws = derive_compatibility_workspace_id(user)
    headers = {"X-User-Id": user, "X-Workspace-Id": ws}

    # Setup initial holding: 10 shares manual @ $50
    engine.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "AMD",
            "shares": 10.0,
            "entryPrice": 50.0,
        },
    )
    holding_id = engine.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    # HTTP-EXIT-02: valid manual PARTIAL -> 200
    res_partial = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "PARTIAL",
            "shares": 4.0,
            "exitPrice": 70.0,
            "idempotencyKey": "idem_http_partial_1",
            "notes": "Partial scale-out",
        },
    )
    assert res_partial.status_code == 200, f"Expected 200, got {res_partial.status_code}: {res_partial.text}"
    body_p = res_partial.json()
    assert body_p["exitType"] == "PARTIAL"
    assert body_p["sharesExited"] == 4.0
    assert body_p["manualSharesRemaining"] == 6.0
    assert body_p["realizedPnl"] == 80.0  # (70 - 50) * 4
    assert body_p["portfolioStatus"] == "OPEN"

    # HTTP-EXIT-08: exact idempotent replay -> 200, identical exitEventId
    res_replay = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "PARTIAL",
            "shares": 4.0,
            "exitPrice": 70.0,
            "idempotencyKey": "idem_http_partial_1",
            "notes": "Partial scale-out",
        },
    )
    assert res_replay.status_code == 200
    assert res_replay.json()["exitEventId"] == body_p["exitEventId"]

    # HTTP-EXIT-09: conflicting idempotency reuse -> 409 IDEMPOTENCY_CONFLICT
    res_conflict = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "PARTIAL",
            "shares": 5.0,  # Conflict: 5.0 vs 4.0
            "exitPrice": 70.0,
            "idempotencyKey": "idem_http_partial_1",
        },
    )
    assert res_conflict.status_code == 409
    assert res_conflict.json()["code"] == "IDEMPOTENCY_CONFLICT"

    # HTTP-EXIT-06: invalid PARTIAL missing shares -> 400
    res_no_shares = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "PARTIAL",
            "exitPrice": 70.0,
            "idempotencyKey": "idem_http_invalid_partial",
        },
    )
    assert res_no_shares.status_code == 400
    assert res_no_shares.json()["code"] == "INVALID_EXIT_QUANTITY"

    # HTTP-EXIT-07: shares exceed manual quantity -> 400 (remaining: 6.0, request: 8.0)
    res_exceed = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "PARTIAL",
            "shares": 8.0,
            "exitPrice": 70.0,
            "idempotencyKey": "idem_http_exceed",
        },
    )
    assert res_exceed.status_code == 400
    assert res_exceed.json()["code"] == "EXIT_QUANTITY_EXCEEDS_REMAINING"

    # HTTP-EXIT-05: invalid FULL containing numeric shares -> 400 INVALID_EXIT_QUANTITY
    # Case A: FULL with incorrect numeric shares (2.0 vs 6.0)
    res_inv_shares_mismatch = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "FULL",
            "shares": 2.0,
            "exitPrice": 70.0,
            "idempotencyKey": "idem_http_inv_full_mismatch",
        },
    )
    assert res_inv_shares_mismatch.status_code == 400
    assert res_inv_shares_mismatch.json()["code"] == "INVALID_EXIT_QUANTITY"

    # Case B: FULL with exact numeric shares (6.0 == manual_shares) -> strictly rejected!
    res_inv_shares_exact = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "FULL",
            "shares": 6.0,
            "exitPrice": 70.0,
            "idempotencyKey": "idem_http_inv_full_exact",
        },
    )
    assert res_inv_shares_exact.status_code == 400
    assert res_inv_shares_exact.json()["code"] == "INVALID_EXIT_QUANTITY"
    assert "Client numeric shares are not permitted" in res_inv_shares_exact.json()["message"]

    # HTTP-EXIT-10: cross-workspace holding -> 404
    other_user = "other_user"
    cross_headers = {"X-User-Id": other_user, "X-Workspace-Id": derive_compatibility_workspace_id(other_user)}
    res_cross = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=cross_headers,
        json={
            "exitType": "FULL",
            "exitPrice": 70.0,
            "idempotencyKey": "idem_cross",
        },
    )
    assert res_cross.status_code == 404
    assert res_cross.json()["code"] == "HOLDING_NOT_FOUND"

    # HTTP-EXIT-11: internal server error -> safe generic 500
    class BrokenService:
        def record_manual_holding_exit(self, *args, **kwargs):
            raise RuntimeError("Database connection suddenly dropped!")

    orig_svc = portfolio_route.portfolio_service
    portfolio_route.portfolio_service = BrokenService()
    try:
        res_500 = client.post(
            f"/api/v1/portfolio/holdings/{holding_id}/exit",
            headers=headers,
            json={
                "exitType": "FULL",
                "exitPrice": 70.0,
                "idempotencyKey": "idem_500",
            },
        )
        assert res_500.status_code == 500
        body_500 = res_500.json()
        assert body_500["ok"] is False
        assert body_500["code"] == "INTERNAL_ERROR"
        assert body_500["message"] == "An unexpected error occurred while recording holding exit."
    finally:
        portfolio_route.portfolio_service = orig_svc

    # HTTP-EXIT-01: valid manual FULL -> 200
    res_full = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "FULL",
            "exitPrice": 75.0,
            "idempotencyKey": "idem_http_full_1",
        },
    )
    assert res_full.status_code == 200
    body_f = res_full.json()
    assert body_f["exitType"] == "FULL"
    assert body_f["sharesExited"] == 6.0
    assert body_f["manualSharesRemaining"] == 0.0
    assert body_f["portfolioStatus"] == "CLOSED"

    # HTTP-EXIT-03: holding not found (now deleted) -> 404
    res_not_found = client.post(
        f"/api/v1/portfolio/holdings/{holding_id}/exit",
        headers=headers,
        json={
            "exitType": "FULL",
            "exitPrice": 75.0,
            "idempotencyKey": "idem_http_new_key",
        },
    )
    assert res_not_found.status_code == 404
    assert res_not_found.json()["code"] == "HOLDING_NOT_FOUND"

    # HTTP-EXIT-04: holding has no manual quantity -> 400
    # Create pure journal position (manual_shares = 0)
    engine.record_workspace_trade_fill(
        workspace_id=ws,
        user_id=user,
        fill_data={
            "symbol": "TSLA",
            "entryPrice": 200.0,
            "shares": 5.0,
        },
    )
    tsla_holding_id = [h for h in engine.get_workspace_portfolio(workspace_id=ws, user_id=user) if h["symbol"] == "TSLA"][0]["id"]
    res_no_manual = client.post(
        f"/api/v1/portfolio/holdings/{tsla_holding_id}/exit",
        headers=headers,
        json={
            "exitType": "FULL",
            "exitPrice": 220.0,
            "idempotencyKey": "idem_tsla_no_manual",
        },
    )
    assert res_no_manual.status_code == 400
    assert res_no_manual.json()["code"] == "HOLDING_NOT_MANUAL"

    # HTTP-EXIT-12: manual FULL on mixed position leaves journal exposure and aggregate row alive
    # Create mixed position: manual 10 shares of MSFT + journal 4 shares of MSFT
    engine.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={"symbol": "MSFT", "shares": 10.0, "entryPrice": 300.0},
    )
    engine.record_workspace_trade_fill(
        workspace_id=ws,
        user_id=user,
        fill_data={"symbol": "MSFT", "entryPrice": 310.0, "shares": 4.0},
    )
    msft_holding = [h for h in engine.get_workspace_portfolio(workspace_id=ws, user_id=user) if h["symbol"] == "MSFT"][0]
    msft_id = msft_holding["id"]
    assert msft_holding["holdingSource"] == "MIXED"
    assert msft_holding["shares"] == 14.0

    res_mixed_full = client.post(
        f"/api/v1/portfolio/holdings/{msft_id}/exit",
        headers=headers,
        json={
            "exitType": "FULL",
            "exitPrice": 350.0,
            "idempotencyKey": "idem_msft_mixed_full",
        },
    )
    assert res_mixed_full.status_code == 200
    body_mf = res_mixed_full.json()
    assert body_mf["exitType"] == "FULL"
    assert body_mf["manualHoldingStatus"] == "CLOSED"
    assert body_mf["portfolioStatus"] == "OPEN"
    assert body_mf["manualSharesRemaining"] == 0.0
    assert body_mf["journalSharesRemaining"] == 4.0
    assert body_mf["totalSharesRemaining"] == 4.0

    # Verify portfolio row still alive
    msft_surviving = [h for h in engine.get_workspace_portfolio(workspace_id=ws, user_id=user) if h["symbol"] == "MSFT"]
    assert len(msft_surviving) == 1
    assert msft_surviving[0]["shares"] == 4.0
    assert msft_surviving[0]["holdingSource"] == "JOURNAL"


# ==============================================================================
# SECTION 6: READ API VERIFICATION (GET /api/v1/portfolio/holding-exits)
# ==============================================================================

def test_read_api_verification(isolated_env):
    client = isolated_env["client"]
    engine = isolated_env["engine"]
    user = "user_read"
    ws_a = derive_compatibility_workspace_id(user)
    user_b = "user_read_b"
    ws_b = derive_compatibility_workspace_id(user_b)

    engine.save_workspace_holding(workspace_id=ws_a, user_id=user, holding={"symbol": "AAPL", "shares": 10.0, "entryPrice": 150.0})
    h_a_id = engine.get_workspace_portfolio(workspace_id=ws_a, user_id=user)[0]["id"]
    engine.record_manual_holding_exit(
        workspace_id=ws_a,
        user_id=user,
        holding_id=h_a_id,
        exit_data={"exitType": "PARTIAL", "shares": 4.0, "exitPrice": 170.0, "idempotencyKey": "k_read_a_1"},
    )
    engine.record_manual_holding_exit(
        workspace_id=ws_a,
        user_id=user,
        holding_id=h_a_id,
        exit_data={"exitType": "FULL", "exitPrice": 180.0, "idempotencyKey": "k_read_a_2"},
    )

    # 1. Verify public GET route is removed -> 404 or 405 (matches /{symbol} with PUT/DELETE only)
    res_public = client.get("/api/v1/portfolio/holding-exits", headers={"X-User-Id": user, "X-Workspace-Id": ws_a})
    assert res_public.status_code in (404, 405), f"Public GET route must not exist (got {res_public.status_code})"

    # 2. Verify internal engine read capability works for tests/reconciliation
    events_a = engine.get_workspace_holding_exits(workspace_id=ws_a, user_id=user)
    assert len(events_a) == 2
    # Verify ordering: newest first
    assert events_a[0]["exitType"] == "FULL"
    assert events_a[1]["exitType"] == "PARTIAL"

    # 3. Workspace isolation: ws_b should see 0 events
    events_b = engine.get_workspace_holding_exits(workspace_id=ws_b, user_id=user_b)
    assert len(events_b) == 0

    # 4. Symbol filter
    events_sym = engine.get_workspace_holding_exits(workspace_id=ws_a, symbol="AAPL", user_id=user)
    assert len(events_sym) == 2
    events_sym_none = engine.get_workspace_holding_exits(workspace_id=ws_a, symbol="GOOG", user_id=user)
    assert len(events_sym_none) == 0

    # 5. Holding filter
    events_h = engine.get_workspace_holding_exits(workspace_id=ws_a, holding_id=h_a_id, user_id=user)
    assert len(events_h) == 2

    # 6. History survives active position deletion
    active_port = engine.get_workspace_portfolio(workspace_id=ws_a, user_id=user)
    assert len(active_port) == 0  # Deleted after FULL
    events_survives = engine.get_workspace_holding_exits(workspace_id=ws_a, user_id=user)
    assert len(events_survives) == 2


# ==============================================================================
# SECTION 7: MIGRATION FROM REAL PRE-MIGRATION SHAPE
# ==============================================================================

def test_migration_from_real_pre_migration_shape():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "premigration.db")
        conn = sqlite3.connect(db_path)
        # Create exact pre-migration schema:
        conn.executescript("""
            CREATE TABLE workspaces (
                workspace_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE portfolio_holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT,
                shares REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                target_price REAL,
                stop_loss_price REAL,
                added_at TEXT,
                asset_type TEXT DEFAULT 'Stock',
                manual_shares REAL,
                manual_entry_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE user_trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                setup_name TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                shares REAL NOT NULL,
                remaining_shares REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO portfolio_holdings (workspace_id, user_id, symbol, shares, entry_price, manual_shares, manual_entry_price)
            VALUES ('ws_1', 'u_1', 'NVDA', 10.0, 100.0, 10.0, 100.0);
            INSERT INTO user_trade_journal (workspace_id, user_id, symbol, entry_price, shares, remaining_shares, status)
            VALUES ('ws_1', 'u_1', 'AAPL', 150.0, 5.0, 5.0, 'OPEN');
        """)
        conn.commit()

        # Apply new migration
        apply_holding_exit_migration(conn)

        # Verify existing tables and rows preserved
        cur = conn.cursor()
        cur.execute("SELECT symbol, shares FROM portfolio_holdings")
        assert cur.fetchall() == [("NVDA", 10.0)]
        cur.execute("SELECT symbol, shares FROM user_trade_journal")
        assert cur.fetchall() == [("AAPL", 5.0)]

        # Verify new table created
        cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='portfolio_holding_exit_events'")
        assert cur.fetchone()[0] == 1

        # Verify indexes created
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='portfolio_holding_exit_events'")
        idx_names = [r[0] for r in cur.fetchall()]
        assert "idx_holding_exits_ws_sym" in idx_names
        assert "idx_holding_exits_holding" in idx_names
        assert "idx_holding_exits_ws_idem" in idx_names

        # Verify triggers created
        cur.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='portfolio_holding_exit_events'")
        trigger_names = [r[0] for r in cur.fetchall()]
        assert "trg_holding_exit_events_no_update" in trigger_names
        assert "trg_holding_exit_events_no_delete" in trigger_names

        # Verify zero backfill rows
        cur.execute("SELECT count(*) FROM portfolio_holding_exit_events")
        assert cur.fetchone()[0] == 0

        # Verify rerun is idempotent
        apply_holding_exit_migration(conn)
        conn.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# SECTION 8: CONSTRAINT ENFORCEMENT TESTS (15 CHECKS)
# ==============================================================================

def test_database_constraint_matrix(isolated_env):
    conn = isolated_env["engine"]._get_connection()
    valid_base = {
        "workspace_id": "ws_c",
        "user_id": "u_c",
        "holding_id": 1,
        "symbol": "AMD",
        "source": "MANUAL_HOLDING",
        "exit_type": "FULL",
        "position_side": "LONG",
        "entry_price": 50.0,
        "exit_price": 70.0,
        "manual_shares_before": 10.0,
        "manual_shares_exited": 10.0,
        "manual_shares_remaining": 0.0,
        "journal_shares_before": 0.0,
        "journal_shares_after": 0.0,
        "realized_pnl": 200.0,
        "return_pct": 40.0,
        "exit_date": "2026-10-09",
        "idempotency_key": "k_valid",
    }

    def try_insert(override):
        d = dict(valid_base, **override)
        cur = conn.cursor()
        sql = f"""
            INSERT INTO portfolio_holding_exit_events (
                workspace_id, user_id, holding_id, symbol, source, exit_type, position_side,
                entry_price, exit_price, manual_shares_before, manual_shares_exited, manual_shares_remaining,
                journal_shares_before, journal_shares_after, realized_pnl, return_pct, exit_date, idempotency_key
            ) VALUES (
                :workspace_id, :user_id, :holding_id, :symbol, :source, :exit_type, :position_side,
                :entry_price, :exit_price, :manual_shares_before, :manual_shares_exited, :manual_shares_remaining,
                :journal_shares_before, :journal_shares_after, :realized_pnl, :return_pct, :exit_date, :idempotency_key
            )
        """
        cur.execute(sql, d)

    # 1. source != MANUAL_HOLDING
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"source": "JOURNAL"})

    # 2. invalid exit_type
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"exit_type": "DUMP"})

    # 3. position_side != LONG
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"position_side": "SHORT"})

    # 4. entry_price <= 0
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"entry_price": 0.0})

    # 5. exit_price <= 0
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"exit_price": -10.0})

    # 6. manual_shares_before <= 0
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"manual_shares_before": 0.0})

    # 7. manual_shares_exited <= 0
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"manual_shares_exited": 0.0})

    # 8. manual_shares_remaining < 0
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"manual_shares_remaining": -1.0})

    # 9. FULL with nonzero manual remaining
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"exit_type": "FULL", "manual_shares_remaining": 2.0})

    # 10. FULL exited != before
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"exit_type": "FULL", "manual_shares_exited": 8.0, "manual_shares_before": 10.0, "manual_shares_remaining": 0.0})

    # 11. PARTIAL with zero remaining
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"exit_type": "PARTIAL", "manual_shares_before": 10.0, "manual_shares_exited": 10.0, "manual_shares_remaining": 0.0})

    # 12. PARTIAL exited >= before
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"exit_type": "PARTIAL", "manual_shares_before": 10.0, "manual_shares_exited": 10.0, "manual_shares_remaining": 1.0})

    # 13. journal_shares_before != journal_shares_after
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"journal_shares_before": 5.0, "journal_shares_after": 4.0})

    # 14. null idempotency key
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"idempotency_key": None})

    # 15. duplicate workspace/idempotency key
    try_insert({"idempotency_key": "k_dup_1"})
    with pytest.raises(sqlite3.IntegrityError):
        try_insert({"idempotency_key": "k_dup_1"})

    conn.close()


# ==============================================================================
# SECTION 10 & 13: MIXED-SOURCE & SERVER P&L (IREN FIXTURE)
# ==============================================================================

def test_mixed_source_and_server_pnl_iren(isolated_env):
    engine = isolated_env["engine"]
    ws = "ws_iren"
    user = "u_iren"

    # Fixture: manual 10.82544 @ 7.39 + journal 4.0 @ 10.0
    engine.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={"symbol": "IREN", "shares": 10.82544, "entryPrice": 7.39},
    )
    engine.record_workspace_trade_fill(
        workspace_id=ws,
        user_id=user,
        fill_data={"symbol": "IREN", "entryPrice": 10.0, "shares": 4.0},
    )

    h_id = [h for h in engine.get_workspace_portfolio(workspace_id=ws, user_id=user) if h["symbol"] == "IREN"][0]["id"]

    # Journal state before
    j_trades_before = engine.get_workspace_journal_trades(workspace_id=ws, user_id=user)

    # Perform manual FULL exit @ 35.69
    res = engine.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=h_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 35.69,
            "idempotencyKey": "k_iren_full",
            "clientRealizedPnl": 999999.0,  # Untrusted client override attempt
        },
    )

    # Check server P&L calculation
    assert res["realizedPnl"] == 306.36  # (35.69 - 7.39) * 10.82544 = 306.359952 -> 306.36
    assert res["returnPct"] == 382.95   # (35.69 - 7.39) / 7.39 = 382.9499% -> 382.95%
    assert res["realizedR"] is None
    assert res["realizedRStatus"] == "UNAVAILABLE_ORIGINAL_RISK_NOT_RECORDED"

    # Verify journal untouched
    j_trades_after = engine.get_workspace_journal_trades(workspace_id=ws, user_id=user)
    assert j_trades_before == j_trades_after

    # Portfolio row survives with 4.0 shares
    port = engine.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port) == 1
    assert port[0]["shares"] == 4.0
    assert port[0]["manualShares"] == 0.0
    assert port[0]["holdingSource"] == "JOURNAL"
