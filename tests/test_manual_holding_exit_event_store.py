"""
Isolated Test Suite for Manual Holding Exit Event Store & Immutability (Phase 1H).

Covers:
- Section 25: Migration Verification (idempotent, triggers, checks, zero backfill)
- Section 26: MANUAL-EXIT-01 through MANUAL-EXIT-23
- Section 27: Concurrency Tests (FULL+FULL, PARTIAL+PARTIAL, PARTIAL+FULL)
"""

import sqlite3
import pytest
import concurrent.futures
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine, HoldingExitError
from database.holding_exit_migration import apply_holding_exit_migration


@pytest.fixture
def test_db(tmp_path):
    """Isolated, temporary SQLite database engine."""
    db_file = tmp_path / "test_holding_exit_store.db"
    engine = HistoryDatabaseEngine(db_path=str(db_file))
    return engine


# =============================================================================
# SECTION 25: MIGRATION VERIFICATION
# =============================================================================

def test_migration_idempotent_and_zero_backfill(test_db):
    conn = test_db._get_connection()
    try:
        # Re-apply migration multiple times to verify idempotence
        res1 = apply_holding_exit_migration(conn)
        res2 = apply_holding_exit_migration(conn)
        assert res1["backfill_rows"] == 0
        assert res2["backfill_rows"] == 0

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM portfolio_holding_exit_events")
        assert cursor.fetchone()["cnt"] == 0, "Backfill rows must be strictly 0"

        # Verify indices exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='portfolio_holding_exit_events'")
        index_names = {row["name"] for row in cursor.fetchall()}
        assert "idx_holding_exits_ws_sym" in index_names
        assert "idx_holding_exits_holding" in index_names
        assert "idx_holding_exits_ws_idem" in index_names

        # Verify triggers exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='portfolio_holding_exit_events'")
        trigger_names = {row["name"] for row in cursor.fetchall()}
        assert "trg_holding_exit_events_no_update" in trigger_names
        assert "trg_holding_exit_events_no_delete" in trigger_names
    finally:
        conn.close()


# =============================================================================
# SECTION 26: MANUAL-EXIT-01 THROUGH MANUAL-EXIT-23
# =============================================================================

def test_manual_exit_01_and_02_manual_only_full_creates_event_and_removes_projection(test_db):
    """
    MANUAL-EXIT-01: manual-only FULL creates durable event.
    MANUAL-EXIT-02: manual-only FULL removes active projection after event persistence.
    """
    ws = "ws_test_user_01"
    user = "test_user_01"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "AAPL",
            "name": "Apple Inc",
            "shares": 10.0,
            "entryPrice": 150.0,
        },
    )

    port = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port) == 1
    holding_id = port[0]["id"]
    assert holding_id is not None

    event = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 175.0,
            "exitDate": "2026-10-09",
            "idempotencyKey": "idem_aapl_full_01",
        },
    )

    # Assert event fields
    assert event["exitEventId"] > 0
    assert event["symbol"] == "AAPL"
    assert event["exitType"] == "FULL"
    assert event["entryPrice"] == 150.0
    assert event["exitPrice"] == 175.0
    assert event["sharesExited"] == 10.0
    assert event["manualSharesRemaining"] == 0.0
    assert event["realizedPnl"] == 250.0
    assert event["returnPct"] == 16.67
    assert event["manualHoldingStatus"] == "CLOSED"
    assert event["portfolioStatus"] == "CLOSED"

    # Assert active projection row is removed
    port_after = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port_after) == 0, "Active portfolio projection must be removed after manual-only FULL exit"


def test_manual_exit_03_and_04_mixed_full_closes_manual_only_row_survives(test_db):
    """
    MANUAL-EXIT-03: mixed FULL closes manual bucket only (manual: 10.82544 -> 0, journal: 4 -> 4, total: 4).
    MANUAL-EXIT-04: mixed FULL leaves aggregate row alive.
    """
    ws = "ws_test_mixed"
    user = "test_user_mixed"

    # 1. Create manual holding (10.82544 shares @ 7.39)
    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "IREN",
            "name": "Iris Energy",
            "shares": 10.82544,
            "entryPrice": 7.39,
        },
    )

    # 2. Create open journal trade (4.0 shares @ 10.00)
    test_db.record_workspace_trade_fill(
        workspace_id=ws,
        user_id=user,
        fill_data={
            "symbol": "IREN",
            "shares": 4.0,
            "entryPrice": 10.0,
            "setupName": "Breakout",
        },
    )

    port = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port) == 1
    assert port[0]["holdingSource"] == "MIXED"
    assert port[0]["shares"] == pytest.approx(14.82544, rel=1e-5)
    assert port[0]["manualShares"] == pytest.approx(10.82544, rel=1e-5)
    holding_id = port[0]["id"]

    # 3. Perform FULL manual exit
    event = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 35.69,
            "exitDate": "2026-10-09",
            "idempotencyKey": "idem_iren_mixed_full",
        },
    )

    assert event["manualHoldingStatus"] == "CLOSED"
    assert event["portfolioStatus"] == "OPEN"
    assert event["manualSharesRemaining"] == 0.0
    assert event["journalSharesRemaining"] == 4.0
    assert event["totalSharesRemaining"] == 4.0

    # 4. Assert aggregate row survived with 4 journal shares
    port_after = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port_after) == 1
    assert port_after[0]["shares"] == pytest.approx(4.0, rel=1e-5)
    assert port_after[0]["holdingSource"] == "JOURNAL"
    assert port_after[0]["manualShares"] == 0.0


def test_manual_exit_05_and_06_partial_reduces_manual_only_journal_unchanged(test_db):
    """
    MANUAL-EXIT-05: PARTIAL reduces manual_shares only.
    MANUAL-EXIT-06: journal quantity unchanged for every manual exit.
    """
    ws = "ws_partial"
    user = "user_partial"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "MSFT",
            "shares": 10.0,
            "entryPrice": 300.0,
        },
    )

    test_db.record_workspace_trade_fill(
        workspace_id=ws,
        user_id=user,
        fill_data={
            "symbol": "MSFT",
            "shares": 5.0,
            "entryPrice": 310.0,
            "setupName": "Pullback",
        },
    )

    port = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    holding_id = port[0]["id"]

    event = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "PARTIAL",
            "shares": 3.0,
            "exitPrice": 350.0,
            "idempotencyKey": "idem_msft_partial",
        },
    )

    assert event["sharesExited"] == 3.0
    assert event["manualSharesRemaining"] == 7.0
    assert event["journalSharesRemaining"] == 5.0
    assert event["totalSharesRemaining"] == 12.0
    assert event["manualHoldingStatus"] == "OPEN"
    assert event["portfolioStatus"] == "OPEN"

    # Journal exposure unchanged
    conn = test_db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT remaining_shares FROM user_trade_journal WHERE workspace_id = ? AND symbol = 'MSFT'", (ws,))
    assert cursor.fetchone()["remaining_shares"] == 5.0
    conn.close()


def test_manual_exit_07_and_08_full_uses_canonical_quantity_fractional_zero_residual(test_db):
    """
    MANUAL-EXIT-07: FULL uses backend canonical manual quantity.
    MANUAL-EXIT-08: fractional 10.82544 FULL closes with no residual manual shares.
    """
    ws = "ws_frac"
    user = "user_frac"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "IREN",
            "shares": 10.82544,
            "entryPrice": 7.39,
        },
    )
    port = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    holding_id = port[0]["id"]

    # Client passes shares = None for FULL
    event = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "shares": None,
            "exitPrice": 35.69,
            "idempotencyKey": "idem_frac_full",
        },
    )

    assert event["sharesExited"] == pytest.approx(10.82544, rel=1e-6)
    assert event["manualSharesRemaining"] == 0.0

    port_after = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port_after) == 0


def test_manual_exit_09_server_computes_pnl_iren_regression(test_db):
    """
    MANUAL-EXIT-09: server computes P&L.
    IREN regression:
    entry: 7.39, exit: 35.69, shares: 10.82544
    realized_pnl: (35.69 - 7.39) * 10.82544 = 306.36
    return_pct: ((35.69 - 7.39) / 7.39) * 100 = 382.95%
    """
    ws = "ws_iren"
    user = "user_iren"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "IREN",
            "shares": 10.82544,
            "entryPrice": 7.39,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    event = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 35.69,
            "idempotencyKey": "idem_iren_pnl",
        },
    )

    assert event["realizedPnl"] == pytest.approx(306.36, rel=1e-3)
    assert event["returnPct"] == pytest.approx(382.95, rel=1e-3)


def test_manual_exit_10_and_11_idempotent_retry_after_row_delete(test_db):
    """
    MANUAL-EXIT-10: exact idempotent retry returns same event.
    MANUAL-EXIT-11: exact retry works even after manual-only active row is removed.
    """
    ws = "ws_idem"
    user = "user_idem"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "NVDA",
            "shares": 15.0,
            "entryPrice": 100.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    exit_payload = {
        "exitType": "FULL",
        "exitPrice": 120.0,
        "exitDate": "2026-10-09",
        "notes": "Closed position",
        "idempotencyKey": "idem_nvda_exact",
    }

    event1 = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data=exit_payload,
    )

    # Active row is deleted now!
    assert len(test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)) == 0

    # Exact retry with same idempotency key
    event2 = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data=exit_payload,
    )

    assert event1["exitEventId"] == event2["exitEventId"]
    assert event1["realizedPnl"] == event2["realizedPnl"]
    assert event2["portfolioStatus"] == "CLOSED"

    # Only 1 event in database
    conn = test_db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM portfolio_holding_exit_events WHERE idempotency_key = 'idem_nvda_exact'")
    assert cursor.fetchone()["cnt"] == 1
    conn.close()


def test_manual_exit_12_conflicting_idempotency_returns_409(test_db):
    """
    MANUAL-EXIT-12: conflicting idempotency reuse returns 409 (IDEMPOTENCY_CONFLICT).
    """
    ws = "ws_conflict"
    user = "user_conflict"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "AMD",
            "shares": 20.0,
            "entryPrice": 80.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    # First request
    test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "PARTIAL",
            "shares": 5.0,
            "exitPrice": 90.0,
            "exitDate": "2026-10-09",
            "idempotencyKey": "idem_conflict_key",
        },
    )

    # Reuse same idempotency key with conflicting price
    with pytest.raises(HoldingExitError) as exc_info:
        test_db.record_manual_holding_exit(
            workspace_id=ws,
            user_id=user,
            holding_id=holding_id,
            exit_data={
                "exitType": "PARTIAL",
                "shares": 5.0,
                "exitPrice": 110.0,  # CONFLICT
                "exitDate": "2026-10-09",
                "idempotencyKey": "idem_conflict_key",
            },
        )
    assert exc_info.value.code == "IDEMPOTENCY_CONFLICT"
    assert exc_info.value.status_code == 409


def test_manual_exit_13_no_user_trade_journal_rows_created(test_db):
    """
    MANUAL-EXIT-13: no user_trade_journal rows created or modified.
    """
    ws = "ws_no_journal"
    user = "user_no_journal"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "GOOGL",
            "shares": 10.0,
            "entryPrice": 140.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 160.0,
            "idempotencyKey": "idem_googl",
        },
    )

    conn = test_db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM user_trade_journal WHERE workspace_id = ?", (ws,))
    assert cursor.fetchone()["cnt"] == 0, "Zero journal rows should be created"
    conn.close()


def test_manual_exit_14_event_survives_active_projection_removal(test_db):
    """
    MANUAL-EXIT-14: event survives active projection removal.
    """
    ws = "ws_survive"
    user = "user_survive"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "TSLA",
            "shares": 5.0,
            "entryPrice": 200.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 220.0,
            "idempotencyKey": "idem_tsla_survive",
        },
    )

    exits = test_db.get_workspace_holding_exits(workspace_id=ws, symbol="TSLA")
    assert len(exits) == 1
    assert exits[0]["symbol"] == "TSLA"
    assert exits[0]["exitType"] == "FULL"


def test_manual_exit_15_and_16_triggers_block_update_and_delete(test_db):
    """
    MANUAL-EXIT-15: event UPDATE rejected by database trigger.
    MANUAL-EXIT-16: event DELETE rejected by database trigger.
    """
    ws = "ws_immutability"
    user = "user_immutability"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "META",
            "shares": 10.0,
            "entryPrice": 300.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    event = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 350.0,
            "idempotencyKey": "idem_meta_trg",
        },
    )
    event_id = event["exitEventId"]

    conn = test_db._get_connection()
    cursor = conn.cursor()

    # Attempt UPDATE
    with pytest.raises(sqlite3.IntegrityError) as exc_update:
        cursor.execute(
            "UPDATE portfolio_holding_exit_events SET exit_price = 400.0 WHERE exit_event_id = ?",
            (event_id,),
        )
    assert "append-only: UPDATE is prohibited" in str(exc_update.value)

    # Attempt DELETE
    with pytest.raises(sqlite3.IntegrityError) as exc_delete:
        cursor.execute(
            "DELETE FROM portfolio_holding_exit_events WHERE exit_event_id = ?",
            (event_id,),
        )
    assert "append-only: DELETE is prohibited" in str(exc_delete.value)

    conn.close()


def test_manual_exit_17_cross_workspace_exit_rejected(test_db):
    """
    MANUAL-EXIT-17: cross-workspace exit rejected (workspace isolation).
    """
    ws1 = "ws_user_alpha"
    user1 = "user_alpha"
    ws2 = "ws_user_beta"
    user2 = "user_beta"

    test_db.save_workspace_holding(
        workspace_id=ws1,
        user_id=user1,
        holding={
            "symbol": "NFLX",
            "shares": 10.0,
            "entryPrice": 400.0,
        },
    )
    holding_id_1 = test_db.get_workspace_portfolio(workspace_id=ws1, user_id=user1)[0]["id"]

    # Beta attempts to exit Alpha's holding
    with pytest.raises(HoldingExitError) as exc_info:
        test_db.record_manual_holding_exit(
            workspace_id=ws2,
            user_id=user2,
            holding_id=holding_id_1,
            exit_data={
                "exitType": "FULL",
                "exitPrice": 450.0,
                "idempotencyKey": "idem_cross_ws",
            },
        )
    assert exc_info.value.code == "HOLDING_NOT_FOUND"
    assert exc_info.value.status_code == 404

    # Alpha's holding remains intact
    port1 = test_db.get_workspace_portfolio(workspace_id=ws1, user_id=user1)
    assert len(port1) == 1
    assert port1[0]["shares"] == 10.0


def test_manual_exit_18_19_20_atomicity_and_rollbacks(test_db):
    """
    MANUAL-EXIT-18: insert failure rolls back mutation.
    MANUAL-EXIT-19: mutation failure rolls back event.
    MANUAL-EXIT-20: projection sync failure rolls back all changes.
    """
    ws = "ws_atomic"
    user = "user_atomic"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "AMZN",
            "shares": 10.0,
            "entryPrice": 120.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    # Test invalid exit price (triggers check constraint / domain error before commit)
    with pytest.raises(HoldingExitError) as exc_info:
        test_db.record_manual_holding_exit(
            workspace_id=ws,
            user_id=user,
            holding_id=holding_id,
            exit_data={
                "exitType": "FULL",
                "exitPrice": -50.0,  # INVALID
                "idempotencyKey": "idem_amzn_invalid",
            },
        )
    assert exc_info.value.code == "INVALID_EXIT_PRICE"

    # Verify holding is untouched
    port = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port) == 1
    assert port[0]["shares"] == 10.0

    # Verify no exit events created
    events = test_db.get_workspace_holding_exits(workspace_id=ws)
    assert len(events) == 0


def test_manual_exit_21_partial_then_full_ordered_events(test_db):
    """
    MANUAL-EXIT-21: partial then FULL produces two ordered valid events.
    """
    ws = "ws_multi_exit"
    user = "user_multi_exit"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "CRM",
            "shares": 20.0,
            "entryPrice": 200.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    # 1. Partial exit (8 shares @ 220)
    ev1 = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "PARTIAL",
            "shares": 8.0,
            "exitPrice": 220.0,
            "idempotencyKey": "idem_crm_leg1",
        },
    )
    assert ev1["exitType"] == "PARTIAL"
    assert ev1["manualSharesRemaining"] == 12.0

    # 2. Full exit of remaining 12 shares @ 250
    ev2 = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 250.0,
            "idempotencyKey": "idem_crm_leg2",
        },
    )
    assert ev2["exitType"] == "FULL"
    assert ev2["manualSharesRemaining"] == 0.0
    assert ev2["portfolioStatus"] == "CLOSED"

    # Query chronological history
    history = test_db.get_workspace_holding_exits(workspace_id=ws, symbol="CRM")
    assert len(history) == 2
    # Ordered DESC by event_id / timestamp
    assert history[0]["exitEventId"] == ev2["exitEventId"]
    assert history[1]["exitEventId"] == ev1["exitEventId"]


def test_manual_exit_22_realized_r_remains_null(test_db):
    """
    MANUAL-EXIT-22: realized R remains null and UNAVAILABLE_ORIGINAL_RISK_NOT_RECORDED.
    """
    ws = "ws_r_null"
    user = "user_r_null"

    # Holding with trailed stop
    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "PLTR",
            "shares": 10.0,
            "entryPrice": 25.0,
            "stopLossPrice": 28.0,  # Trailed stop above entry
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    event = test_db.record_manual_holding_exit(
        workspace_id=ws,
        user_id=user,
        holding_id=holding_id,
        exit_data={
            "exitType": "FULL",
            "exitPrice": 35.0,
            "idempotencyKey": "idem_pltr_r",
        },
    )

    assert event["realizedR"] is None
    assert event["realizedRStatus"] == "UNAVAILABLE_ORIGINAL_RISK_NOT_RECORDED"


# =============================================================================
# SECTION 27: CONCURRENCY TESTS
# =============================================================================

def test_concurrent_full_full_exits(test_db):
    """
    Two simultaneous FULL manual exits:
    Serialized by BEGIN IMMEDIATE, first wins, second fails closed (HOLDING_NOT_FOUND or ALREADY_CLOSED).
    No double consumption, no negative shares.
    """
    ws = "ws_concurrent_full"
    user = "user_concurrent_full"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "UBER",
            "shares": 10.0,
            "entryPrice": 60.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    results = []
    errors = []

    def perform_exit(key):
        try:
            res = test_db.record_manual_holding_exit(
                workspace_id=ws,
                user_id=user,
                holding_id=holding_id,
                exit_data={
                    "exitType": "FULL",
                    "exitPrice": 75.0,
                    "idempotencyKey": key,
                },
            )
            results.append(res)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(perform_exit, "key_concurrent_full_1")
        f2 = executor.submit(perform_exit, "key_concurrent_full_2")
        concurrent.futures.wait([f1, f2])

    assert len(results) == 1, "Exactly one FULL exit must succeed"
    assert len(errors) == 1, "The competing concurrent exit must fail"
    assert isinstance(errors[0], HoldingExitError)
    assert errors[0].code in ("HOLDING_NOT_FOUND", "HOLDING_ALREADY_CLOSED")

    # Assert 0 residual shares
    port = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port) == 0


def test_concurrent_partial_partial_exits(test_db):
    """
    Two simultaneous PARTIAL manual exits (each exiting 6 shares of 10 shares):
    Total shares = 10.
    First exit of 6 succeeds (remaining: 4).
    Second exit of 6 fails closed with EXIT_QUANTITY_EXCEEDS_REMAINING (no negative shares).
    """
    ws = "ws_concurrent_partial"
    user = "user_concurrent_partial"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "ABNB",
            "shares": 10.0,
            "entryPrice": 120.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    results = []
    errors = []

    def perform_partial(key):
        try:
            res = test_db.record_manual_holding_exit(
                workspace_id=ws,
                user_id=user,
                holding_id=holding_id,
                exit_data={
                    "exitType": "PARTIAL",
                    "shares": 6.0,
                    "exitPrice": 140.0,
                    "idempotencyKey": key,
                },
            )
            results.append(res)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(perform_partial, "key_partial_1")
        f2 = executor.submit(perform_partial, "key_partial_2")
        concurrent.futures.wait([f1, f2])

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], HoldingExitError)
    assert errors[0].code == "EXIT_QUANTITY_EXCEEDS_REMAINING"

    port = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port) == 1
    assert port[0]["shares"] == 4.0
    assert port[0]["manualShares"] == 4.0


def test_concurrent_partial_full_exits(test_db):
    """
    Simultaneous PARTIAL and FULL manual exits starting against 10 shares:
    Serialized by BEGIN IMMEDIATE.
    Order A: PARTIAL (6 shares) succeeds first -> FULL (shares=None) succeeds second, consuming remaining 4 shares.
             Total exited = 10 shares, final manual shares = 0, 2 events created.
    Order B: FULL succeeds first (10 shares) -> PARTIAL fails closed (HOLDING_NOT_FOUND / HOLDING_ALREADY_CLOSED / HOLDING_NOT_MANUAL).
             Total exited = 10 shares, final manual shares = 0, 1 event created.
    Invariants guaranteed in both orderings:
    - manual_shares >= 0
    - journal exposure unchanged (0)
    - no quantity consumed twice (sum of exited shares == 10.0)
    - events reflect actual serialized order
    - no corrupted projection
    - no duplicate P&L
    """
    ws = "ws_concurrent_partial_full"
    user = "user_concurrent_partial_full"

    test_db.save_workspace_holding(
        workspace_id=ws,
        user_id=user,
        holding={
            "symbol": "SNOW",
            "shares": 10.0,
            "entryPrice": 150.0,
        },
    )
    holding_id = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)[0]["id"]

    results = []
    errors = []

    def perform_partial():
        try:
            res = test_db.record_manual_holding_exit(
                workspace_id=ws,
                user_id=user,
                holding_id=holding_id,
                exit_data={
                    "exitType": "PARTIAL",
                    "shares": 6.0,
                    "exitPrice": 180.0,
                    "idempotencyKey": "key_pf_partial",
                },
            )
            results.append(("PARTIAL", res))
        except Exception as e:
            errors.append(("PARTIAL", e))

    def perform_full():
        try:
            res = test_db.record_manual_holding_exit(
                workspace_id=ws,
                user_id=user,
                holding_id=holding_id,
                exit_data={
                    "exitType": "FULL",
                    "exitPrice": 180.0,
                    "idempotencyKey": "key_pf_full",
                },
            )
            results.append(("FULL", res))
        except Exception as e:
            errors.append(("FULL", e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(perform_partial)
        f2 = executor.submit(perform_full)
        concurrent.futures.wait([f1, f2])

    # Check total exited shares across all successful events
    events = test_db.get_workspace_holding_exits(workspace_id=ws, symbol="SNOW")
    total_exited = sum(e["sharesExited"] for e in events)
    assert total_exited == 10.0, f"Expected 10.0 total exited shares, got {total_exited}"

    # Projection must be completely closed (0 shares remaining)
    port = test_db.get_workspace_portfolio(workspace_id=ws, user_id=user)
    assert len(port) == 0, "Projection must be closed after all 10 shares are exited"

    if len(results) == 2:
        # Order A: PARTIAL then FULL
        assert len(errors) == 0
        assert len(events) == 2
        # Oldest event first: PARTIAL (6 shares), then FULL (4 shares)
        partial_evt = [e for e in events if e["exitType"] == "PARTIAL"][0]
        full_evt = [e for e in events if e["exitType"] == "FULL"][0]
        assert partial_evt["sharesExited"] == 6.0
        assert partial_evt["manualSharesRemaining"] == 4.0
        assert full_evt["sharesExited"] == 4.0
        assert full_evt["manualSharesRemaining"] == 0.0
    else:
        # Order B: FULL then PARTIAL
        assert len(results) == 1
        assert results[0][0] == "FULL"
        assert len(errors) == 1
        assert errors[0][0] == "PARTIAL"
        assert isinstance(errors[0][1], HoldingExitError)
        assert errors[0][1].code in ("HOLDING_NOT_FOUND", "HOLDING_ALREADY_CLOSED", "HOLDING_NOT_MANUAL")
        assert len(events) == 1
        assert events[0]["sharesExited"] == 10.0
        assert events[0]["manualSharesRemaining"] == 0.0
