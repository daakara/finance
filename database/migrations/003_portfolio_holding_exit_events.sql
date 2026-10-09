-- Migration 003: ARX Manual Holding Exit Event Store & Immutability Triggers
-- Target Engines: SQLite (Local/Test) & PostgreSQL (Railway Production)
-- Invariants:
-- 1. portfolio_holding_exit_events is append-only for MANUAL_HOLDING lifecycle events.
-- 2. holding_id is an informational snapshot reference; zero cascading foreign keys.
-- 3. Database triggers enforce append-only immutability (prohibit UPDATE and DELETE).
-- 4. Unique constraint on (workspace_id, idempotency_key) enforces idempotency.
-- 5. Zero historical backfill.

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

-- Performance & Audit Query Indices
CREATE INDEX IF NOT EXISTS idx_holding_exits_ws_sym ON portfolio_holding_exit_events (workspace_id, symbol, created_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_holding_exits_holding ON portfolio_holding_exit_events (holding_id);
CREATE INDEX IF NOT EXISTS idx_holding_exits_ws_idem ON portfolio_holding_exit_events (workspace_id, idempotency_key);

-- Database-Level Immutability Triggers (Append-Only Enforcement)
CREATE TRIGGER IF NOT EXISTS trg_holding_exit_events_no_update
BEFORE UPDATE ON portfolio_holding_exit_events
BEGIN
    SELECT RAISE(ABORT, 'portfolio_holding_exit_events is append-only: UPDATE is prohibited.');
END;

CREATE TRIGGER IF NOT EXISTS trg_holding_exit_events_no_delete
BEFORE DELETE ON portfolio_holding_exit_events
BEGIN
    SELECT RAISE(ABORT, 'portfolio_holding_exit_events is append-only: DELETE is prohibited.');
END;
