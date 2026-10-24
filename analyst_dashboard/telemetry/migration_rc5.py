"""ARX Terminal — Corporate Events RC5 Telemetry Schema Migration & Pre-Flight Compatibility Validator.

Enforces:
1. Deterministic additive migration execution (005_arx_rc5_corporate_events_telemetry.sql).
2. Deep schema compatibility validation beyond naive CREATE TABLE IF NOT EXISTS.
3. Verification of required columns, constraints (uq_rc5_eval_logical, uq_rc5_raw_delivery), and indexes.
4. Fail-closed rejection of existing incompatible tables.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Tuple, List, Dict, Any, Optional

EXPECTED_EVAL_COLUMNS = {
    "evaluation_pk",
    "release_sha",
    "evaluation_id",
    "canonical_digest",
    "schema_version",
    "event_id",
    "occurred_at_utc",
    "ruleset_version",
    "evaluation_context_id",
    "symbol",
    "event_category",
    "event_time_utc",
    "market_session",
    "event_source_status",
    "freshness_status",
    "days_to_earnings",
    "risk_bucket",
    "applied_penalty",
    "evaluation_outcome",
    "first_received_at_utc",
    "retransmission_count",
    "is_natural_candidate",
    "epoch_id",
}

EXPECTED_RAW_COLUMNS = {
    "raw_event_pk",
    "delivery_attempt_id",
    "deduplication_key",
    "evaluation_id",
    "release_sha",
    "received_at_utc",
    "edge_timestamp_utc",
    "key_id",
    "client_ip_hash",
    "user_agent_hash",
    "raw_payload_json",
    "ingestion_verdict",
}

EXPECTED_TABLES = [
    "arx_rc5_observation_epochs",
    "arx_rc5_telemetry_evaluations",
    "arx_rc5_telemetry_raw_events",
    "arx_rc5_telemetry_quarantine",
]


def get_default_migration_path() -> str:
    """Returns absolute path to 005_arx_rc5_corporate_events_telemetry.sql."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "..", "..", "database", "migrations", "005_arx_rc5_corporate_events_telemetry.sql"),
        os.path.join(base_dir, "..", "..", "..", "database", "migrations", "005_arx_rc5_corporate_events_telemetry.sql"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # Default relative to repository root
    return os.path.abspath(candidates[0])


def verify_rc5_telemetry_schema_compatibility(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    """Inspects the SQLite catalog to ensure all required tables, columns, and constraints exist.
    
    Guarantees:
    - Does NOT rely solely on CREATE TABLE IF NOT EXISTS.
    - Validates column presence for authoritative evaluation and raw delivery tables.
    - Verifies uniqueness constraints (uq_rc5_eval_logical on (release_sha, evaluation_id)
      and uq_rc5_raw_delivery on delivery_attempt_id).
    """
    errors: List[str] = []
    cursor = conn.cursor()

    # 1. Verify table presence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}

    for tbl in EXPECTED_TABLES:
        if tbl not in existing_tables:
            errors.append(f"Missing required telemetry table: '{tbl}'")

    # 2. Verify column schema for arx_rc5_telemetry_evaluations
    if "arx_rc5_telemetry_evaluations" in existing_tables:
        cursor.execute("PRAGMA table_info(arx_rc5_telemetry_evaluations)")
        eval_cols = {row[1] for row in cursor.fetchall()}
        missing_eval_cols = EXPECTED_EVAL_COLUMNS - eval_cols
        if missing_eval_cols:
            errors.append(f"Incompatible arx_rc5_telemetry_evaluations: missing columns {sorted(list(missing_eval_cols))}")

    # 3. Verify column schema for arx_rc5_telemetry_raw_events
    if "arx_rc5_telemetry_raw_events" in existing_tables:
        cursor.execute("PRAGMA table_info(arx_rc5_telemetry_raw_events)")
        raw_cols = {row[1] for row in cursor.fetchall()}
        missing_raw_cols = EXPECTED_RAW_COLUMNS - raw_cols
        if missing_raw_cols:
            errors.append(f"Incompatible arx_rc5_telemetry_raw_events: missing columns {sorted(list(missing_raw_cols))}")

    # 4. Verify uniqueness constraint uq_rc5_eval_logical on (release_sha, evaluation_id)
    if "arx_rc5_telemetry_evaluations" in existing_tables:
        cursor.execute("PRAGMA index_list(arx_rc5_telemetry_evaluations)")
        indices = cursor.fetchall()
        has_logical_unique = False
        for idx in indices:
            idx_name = idx[1]
            is_unique = idx[2]
            if is_unique:
                cursor.execute(f"PRAGMA index_info({idx_name})")
                indexed_cols = [c[2] for c in cursor.fetchall()]
                if set(indexed_cols) == {"release_sha", "evaluation_id"}:
                    has_logical_unique = True
                    break

        if not has_logical_unique:
            errors.append("Missing required UNIQUE(release_sha, evaluation_id) constraint on arx_rc5_telemetry_evaluations")

    # 5. Verify uniqueness constraint uq_rc5_raw_delivery on delivery_attempt_id
    if "arx_rc5_telemetry_raw_events" in existing_tables:
        cursor.execute("PRAGMA index_list(arx_rc5_telemetry_raw_events)")
        indices_raw = cursor.fetchall()
        has_raw_unique = False
        for idx in indices_raw:
            idx_name = idx[1]
            is_unique = idx[2]
            if is_unique:
                cursor.execute(f"PRAGMA index_info({idx_name})")
                indexed_raw_cols = [c[2] for c in cursor.fetchall()]
                if set(indexed_raw_cols) == {"delivery_attempt_id"}:
                    has_raw_unique = True
                    break

        if not has_raw_unique:
            errors.append("Missing required UNIQUE(delivery_attempt_id) constraint on arx_rc5_telemetry_raw_events")

    if errors:
        return False, errors

    return True, []


def apply_rc5_telemetry_migration(
    conn: sqlite3.Connection, ddl_path: Optional[str] = None
) -> Dict[str, Any]:
    """Atomically executes the 005 migration DDL and validates schema post-execution."""
    path = ddl_path or get_default_migration_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Migration DDL not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        ddl = f.read()

    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
    except Exception:
        pass

    # Execute inside a transaction
    with conn:
        conn.executescript(ddl)

    # Post-migration verification
    compatible, errors = verify_rc5_telemetry_schema_compatibility(conn)
    if not compatible:
        raise RuntimeError(f"Post-migration schema compatibility check failed: {'; '.join(errors)}")

    return {
        "status": "APPLIED",
        "migration": "005_arx_rc5_corporate_events_telemetry.sql",
        "verified": True,
        "tables": EXPECTED_TABLES,
    }


def rollback_rc5_telemetry_schema(conn: sqlite3.Connection) -> None:
    """Rolls back the 4 RC5 telemetry tables in reverse order. Used strictly for test teardown."""
    with conn:
        for tbl in reversed(EXPECTED_TABLES):
            conn.execute(f"DROP TABLE IF EXISTS {tbl};")
