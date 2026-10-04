"""
ARX SaaS Foundation Phase 1G: Workspace Tenancy Migration & Backfill Engine.

Implements the Expand phase of the Expand-Contract Migration Strategy:
1. Creates 'workspaces' and 'workspace_memberships' tables.
2. Additively adds nullable 'workspace_id' column to WORKSPACE_OWNED tables:
   - portfolio_holdings
   - user_trade_journal
   - user_cockpit_actions
3. Creates dedicated workspace-scoped indexes.
4. Preserves user_profiles as ACTOR_PROFILE without workspace_id.
5. Performs deterministic, idempotent backfill using derive_compatibility_workspace_id (INV-SAAS-06).
"""

import sqlite3
import logging
from typing import Dict, Any, List, Optional

from api.context.workspace_identity import derive_compatibility_workspace_id

logger = logging.getLogger("database.workspace_migration")

WORKSPACE_OWNED_TABLES = [
    "portfolio_holdings",
    "user_trade_journal",
    "user_cockpit_actions",
]


def _get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    """Retrieve list of column names for a given table."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]


def _table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if table exists in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cursor.fetchone() is not None


def apply_workspace_tenancy_migration(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Apply Phase 1G schema expansion to SQLite database.
    Fully idempotent: safe to execute multiple times against any database state.
    """
    results: Dict[str, Any] = {
        "workspaces_table_created": False,
        "workspace_memberships_table_created": False,
        "columns_added": {},
        "indexes_created": [],
    }

    cursor = conn.cursor()

    # 1. Create Core Workspaces Table
    if not _table_exists(cursor, "workspaces"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        results["workspaces_table_created"] = True

    # 2. Create Workspace Memberships Table
    if not _table_exists(cursor, "workspace_memberships"):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_memberships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'owner',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, user_id),
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            );
        """)
        results["workspace_memberships_table_created"] = True

    # 3. Additive Columns for WORKSPACE_OWNED Tables
    for table in WORKSPACE_OWNED_TABLES:
        if _table_exists(cursor, table):
            existing_cols = _get_table_columns(cursor, table)
            if "workspace_id" not in existing_cols:
                # Add column as NULLABLE (Expand phase requirement)
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN workspace_id TEXT;")
                results["columns_added"][table] = "workspace_id"
                logger.info(f"Added nullable workspace_id column to {table}")

    # 4. Create Performance & Isolation Indexes
    indexes_to_create = [
        ("idx_workspace_memberships_user", "workspace_memberships", "(user_id)"),
        ("idx_workspace_memberships_ws", "workspace_memberships", "(workspace_id)"),
        ("idx_portfolio_holdings_ws", "portfolio_holdings", "(workspace_id)"),
        ("idx_portfolio_holdings_ws_sym", "portfolio_holdings", "(workspace_id, symbol)"),
        ("idx_user_trade_journal_ws", "user_trade_journal", "(workspace_id)"),
        ("idx_user_cockpit_actions_ws", "user_cockpit_actions", "(workspace_id)"),
    ]

    for idx_name, table, cols in indexes_to_create:
        if _table_exists(cursor, table):
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} {cols};")
            results["indexes_created"].append(idx_name)

    conn.commit()
    return results


def backfill_workspace_tenancy(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Perform deterministic, idempotent backfill of workspace_id on WORKSPACE_OWNED tables.
    Uses canonical derive_compatibility_workspace_id authority (INV-SAAS-06).
    Zero synthetic or guessed assignments: records without valid user_id remain NULL.
    """
    ledger: Dict[str, Any] = {
        "tables": {},
        "workspaces_created": 0,
        "memberships_created": 0,
    }

    cursor = conn.cursor()

    for table in WORKSPACE_OWNED_TABLES:
        if not _table_exists(cursor, table):
            continue

        cols = _get_table_columns(cursor, table)
        if "workspace_id" not in cols or "user_id" not in cols:
            continue

        # Measure row counts before
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        row_count_before = cursor.fetchone()[0]

        # Find rows needing backfill
        cursor.execute(f"SELECT id, user_id FROM {table} WHERE workspace_id IS NULL;")
        pending_rows = cursor.fetchall()

        rows_eligible = len(pending_rows)
        rows_backfilled = 0
        rows_unresolved = 0
        rows_conflicting = 0

        # Group by user_id for batch updates
        user_ids_to_update: Dict[str, List[Any]] = {}
        for r_id, u_id in pending_rows:
            if not u_id or not str(u_id).strip():
                rows_unresolved += 1
                continue
            cleaned_uid = str(u_id).strip()
            user_ids_to_update.setdefault(cleaned_uid, []).append(r_id)

        for user_id, row_ids in user_ids_to_update.items():
            ws_id = derive_compatibility_workspace_id(user_id)

            # Ensure workspace entity exists
            cursor.execute(
                "INSERT OR IGNORE INTO workspaces (workspace_id, name) VALUES (?, ?);",
                (ws_id, f"Workspace {user_id}"),
            )
            if cursor.rowcount > 0:
                ledger["workspaces_created"] += 1

            # Ensure membership entity exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO workspace_memberships (workspace_id, user_id, role)
                VALUES (?, ?, 'owner');
                """,
                (ws_id, user_id),
            )
            if cursor.rowcount > 0:
                ledger["memberships_created"] += 1

            # Execute backfill update for this user's records
            cursor.execute(
                f"UPDATE {table} SET workspace_id = ? WHERE user_id = ? AND workspace_id IS NULL;",
                (ws_id, user_id),
            )
            rows_backfilled += cursor.rowcount

        # Measure row counts after to verify data preservation
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        row_count_after = cursor.fetchone()[0]

        ledger["tables"][table] = {
            "row_count_before": row_count_before,
            "rows_eligible": rows_eligible,
            "rows_backfilled": rows_backfilled,
            "rows_unresolved": rows_unresolved,
            "rows_conflicting": rows_conflicting,
            "row_count_after": row_count_after,
        }

    conn.commit()
    return ledger
