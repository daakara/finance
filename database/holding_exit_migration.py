"""
ARX Portfolio Lifecycle: Manual Holding Exit Event Store Migration Engine.

Applies schema definition and immutability triggers for portfolio_holding_exit_events:
1. Creates table if not exists with all domain CHECK constraints.
2. Creates performance & audit indices.
3. Establishes append-only database triggers (prohibiting UPDATE and DELETE).
4. Strictly ZERO historical backfill.
"""

import sqlite3
import logging
from typing import Dict, Any

logger = logging.getLogger("database.holding_exit_migration")


def apply_holding_exit_migration(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Apply portfolio_holding_exit_events migration to SQLite database.
    Fully idempotent: safe to execute multiple times against any database state.
    """
    results: Dict[str, Any] = {
        "exit_events_table_created": False,
        "indexes_created": [],
        "triggers_created": [],
        "backfill_rows": 0,
    }

    cursor = conn.cursor()

    # 1. Create Event Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_holding_exit_events (
            exit_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            holding_id INTEGER,
            symbol TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'MANUAL_HOLDING',
            exit_type TEXT NOT NULL,
            position_side TEXT NOT NULL DEFAULT 'LONG',
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            manual_shares_before REAL NOT NULL,
            manual_shares_exited REAL NOT NULL,
            manual_shares_remaining REAL NOT NULL,
            journal_shares_before REAL NOT NULL,
            journal_shares_after REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            return_pct REAL NOT NULL,
            realized_r REAL,
            realized_r_status TEXT NOT NULL DEFAULT 'UNAVAILABLE_ORIGINAL_RISK_NOT_RECORDED',
            exit_date TEXT NOT NULL,
            notes TEXT,
            idempotency_key TEXT NOT NULL,
            created_at_utc TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (source = 'MANUAL_HOLDING'),
            CHECK (exit_type IN ('FULL', 'PARTIAL')),
            CHECK (position_side = 'LONG'),
            CHECK (entry_price > 0),
            CHECK (exit_price > 0),
            CHECK (manual_shares_before > 0),
            CHECK (manual_shares_exited > 0),
            CHECK (manual_shares_remaining >= 0),
            CHECK (journal_shares_before >= 0),
            CHECK (journal_shares_after >= 0),
            CHECK (journal_shares_before = journal_shares_after),
            CHECK (
                (
                    exit_type = 'FULL'
                    AND manual_shares_remaining = 0
                    AND ROUND(manual_shares_exited, 6) = ROUND(manual_shares_before, 6)
                )
                OR
                (
                    exit_type = 'PARTIAL'
                    AND manual_shares_remaining > 0
                    AND manual_shares_exited < manual_shares_before
                    AND ROUND(manual_shares_remaining + manual_shares_exited, 6) = ROUND(manual_shares_before, 6)
                )
            ),
            CONSTRAINT uq_holding_exit_idempotency UNIQUE (workspace_id, idempotency_key)
        );
    """)
    results["exit_events_table_created"] = True

    # 2. Performance & Audit Indices
    indices = [
        ("idx_holding_exits_ws_sym", "portfolio_holding_exit_events", "(workspace_id, symbol, created_at_utc DESC)"),
        ("idx_holding_exits_holding", "portfolio_holding_exit_events", "(holding_id)"),
        ("idx_holding_exits_ws_idem", "portfolio_holding_exit_events", "(workspace_id, idempotency_key)"),
    ]
    for idx_name, tbl, cols in indices:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl} {cols};")
        results["indexes_created"].append(idx_name)

    # 3. Immutability Triggers (Append-Only Enforcement)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_holding_exit_events_no_update
        BEFORE UPDATE ON portfolio_holding_exit_events
        BEGIN
            SELECT RAISE(ABORT, 'portfolio_holding_exit_events is append-only: UPDATE is prohibited.');
        END;
    """)
    results["triggers_created"].append("trg_holding_exit_events_no_update")

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_holding_exit_events_no_delete
        BEFORE DELETE ON portfolio_holding_exit_events
        BEGIN
            SELECT RAISE(ABORT, 'portfolio_holding_exit_events is append-only: DELETE is prohibited.');
        END;
    """)
    results["triggers_created"].append("trg_holding_exit_events_no_delete")

    return results
