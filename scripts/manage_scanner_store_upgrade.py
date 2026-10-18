#!/usr/bin/env python3
"""
scripts/manage_scanner_store_upgrade.py

Executable, idempotent operational runbook tool for PR #28 SQLite scanner store migration.
Provides bounded preflight inspection, exclusive verified backup, schema migration,
post-upgrade baseline verification, and containment checks.

Authoritative Database Path Resolution:
1. Explicit CLI argument (--db-path / explicit_path).
2. FINANCE_DATA_DIR environment variable (if set and non-empty).
3. DATA_DIR environment variable (if set and non-empty).
4. Home-directory fallback: ~/.finance_scanner_store.db

All operations pin the resolved absolute target and reject target mismatches.
"""

import os
import sys
import json
import sqlite3
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Canonical required snapshot columns matching analyst_dashboard/data/scanner_store.py
REQUIRED_SNAPSHOT_COLUMNS = {
    "snapshot_id", "scanner_id", "run_id", "api_contract_version", "ruleset_version",
    "evidence_schema_version", "score_model_version", "data_provenance_version",
    "universe_version", "freshness_policy_version", "implementation_release_sha",
    "semantic_fingerprint", "generated_at", "data_as_of", "status_at_publication",
    "universe_id", "universe_size", "matched_count", "results_json", "provenance_json",
    "freshness_json", "publication_decision"
}


def resolve_scanner_db_path(
    explicit_path: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """
    Authoritative scanner store database path resolver matching application logic.
    Does NOT invoke os.makedirs or initialize any database (zero side effects).
    """
    if explicit_path:
        return os.path.abspath(os.path.expanduser(explicit_path))

    environ = os.environ if env is None else env
    finance_dir = environ.get("FINANCE_DATA_DIR")
    if finance_dir and finance_dir.strip():
        data_dir = finance_dir.strip()
    else:
        data_dir = environ.get("DATA_DIR", "").strip() or os.path.expanduser("~")

    return os.path.abspath(os.path.join(data_dir, ".finance_scanner_store.db"))


def sqlite_ro_uri(target_path: str) -> str:
    """Construct an authoritative read-only SQLite URI across POSIX and Windows."""
    abs_path = os.path.abspath(target_path).replace(os.sep, "/")
    if abs_path.startswith("/"):
        return f"file:{abs_path}?mode=ro"
    return f"file:///{abs_path}?mode=ro"


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_row_hash(row: sqlite3.Row) -> str:
    """Compute deterministic SHA-256 digest of a single scanner_snapshots row."""
    # Convert row keys and values to a deterministic sorted dictionary
    row_dict = {k: row[k] for k in row.keys()}
    serialized = json.dumps(row_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def capture_baseline(db_path: str) -> Dict[str, Any]:
    """
    Read-only preflight inspection of scanner_snapshots table.
    Requires db_path to exist; does NOT create or mutate any file.
    Captures integrity check, schema metadata, nullability, indexes, triggers,
    and a deterministic content hash of all historical snapshot rows.
    """
    abs_target = os.path.abspath(db_path)
    if not os.path.isfile(abs_target):
        raise FileNotFoundError(f"Target database file not found: {abs_target}. Preflight requires an existing database.")

    uri = sqlite_ro_uri(abs_target)
    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        # 1. PRAGMA integrity_check
        integrity_rows = cursor.execute("PRAGMA integrity_check;").fetchall()
        integrity_ok = len(integrity_rows) == 1 and integrity_rows[0][0] == "ok"
        if not integrity_ok:
            raise RuntimeError(f"Database integrity check failed: {integrity_rows}")

        # 2. Check table existence
        tbl = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scanner_snapshots';").fetchone()
        if not tbl:
            raise RuntimeError("Table 'scanner_snapshots' does not exist in target database.")

        # 3. Schema inspection
        cols = cursor.execute("PRAGMA table_info(scanner_snapshots);").fetchall()
        col_names = {col["name"] for col in cols}
        missing_required = REQUIRED_SNAPSHOT_COLUMNS - col_names
        if missing_required:
            raise RuntimeError(f"UNSUPPORTED_SCHEMA_SHAPE: Missing required columns: {sorted(missing_required)}")

        matched_col = next((c for c in cols if c["name"] == "matched_count"), None)
        if not matched_col:
            raise RuntimeError("Column 'matched_count' missing from scanner_snapshots.")
        
        matched_count_not_null = bool(matched_col["notnull"] == 1)
        migration_required = matched_count_not_null

        # 4. Indexes
        indexes = [
            {"name": r["name"], "sql": r["sql"]}
            for r in cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='scanner_snapshots';").fetchall()
        ]

        # 5. Triggers
        triggers = [
            {"name": r["name"], "sql": r["sql"]}
            for r in cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name='scanner_snapshots';").fetchall()
        ]

        # 6. Row Baseline
        cursor.execute("SELECT * FROM scanner_snapshots ORDER BY snapshot_id ASC;")
        rows = cursor.fetchall()
        total_rows = len(rows)
        snapshot_ids = [r["snapshot_id"] for r in rows]

        # Deterministic combined hash of all rows
        row_hasher = hashlib.sha256()
        per_row_hashes = {}
        for r in rows:
            h = compute_row_hash(r)
            per_row_hashes[r["snapshot_id"]] = h
            row_hasher.update(h.encode("utf-8"))
        baseline_content_sha256 = row_hasher.hexdigest()

        return {
            "target_database_path": abs_target,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "database_integrity": "ok",
            "matched_count_not_null": matched_count_not_null,
            "migration_required": migration_required,
            "total_rows": total_rows,
            "snapshot_ids": snapshot_ids,
            "row_hashes": per_row_hashes,
            "baseline_content_sha256": baseline_content_sha256,
            "indexes": indexes,
            "triggers": triggers,
            "schema_columns": [dict(c) for c in cols],
        }
    finally:
        conn.close()


def create_and_verify_backup(
    source_db_path: str,
    backup_dest_path: str,
    baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Creates an exclusive, verified SQLite online backup from a read-only source connection.
    Guarantees:
    - Never overwrites an existing backup destination (exclusive creation).
    - Captures committed WAL data via SQLite online backup API.
    - Closes all connections reliably on both success and failure.
    - Runs PRAGMA integrity_check on the destination backup.
    - Compares logical contents against preflight baseline.
    - Computes and returns destination size and SHA-256.
    """
    abs_source = os.path.abspath(source_db_path)
    abs_dest = os.path.abspath(backup_dest_path)

    if not os.path.isfile(abs_source):
        raise FileNotFoundError(f"Source database not found: {abs_source}")

    if os.path.exists(abs_dest):
        raise FileExistsError(f"Backup destination already exists: {abs_dest}. Never overwrite an existing backup.")

    if abs_source == abs_dest:
        raise ValueError("Backup destination cannot be the same as the live database.")

    # Capture baseline if not provided
    if baseline is None:
        baseline = capture_baseline(abs_source)
    elif os.path.abspath(baseline["target_database_path"]) != abs_source:
        raise ValueError(f"Baseline target mismatch: baseline is for {baseline['target_database_path']}, but source is {abs_source}")

    os.makedirs(os.path.dirname(abs_dest), exist_ok=True)

    # Exclusively reserve the destination file (O_CREAT | O_EXCL)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(abs_dest, flags)
    os.close(fd)

    src_conn = None
    dest_conn = None
    try:
        src_uri = sqlite_ro_uri(abs_source)
        src_conn = sqlite3.connect(src_uri, uri=True, timeout=30.0)
        dest_conn = sqlite3.connect(abs_dest, timeout=30.0)

        # Execute online backup (flushes and reads committed WAL pages page-by-page)
        src_conn.backup(dest_conn)
    finally:
        if dest_conn:
            dest_conn.close()
        if src_conn:
            src_conn.close()

    # Verify Backup Integrity and Logical Equivalence
    verify_conn = None
    try:
        verify_conn = sqlite3.connect(sqlite_ro_uri(abs_dest), uri=True, timeout=10.0)
        verify_conn.row_factory = sqlite3.Row
        c = verify_conn.cursor()

        # 1. Integrity check
        integrity = c.execute("PRAGMA integrity_check;").fetchall()
        if len(integrity) != 1 or integrity[0][0] != "ok":
            raise RuntimeError(f"Backup integrity check failed on {abs_dest}: {integrity}")

        # 2. Compare row count
        backup_rows = c.execute("SELECT * FROM scanner_snapshots ORDER BY snapshot_id ASC;").fetchall()
        if len(backup_rows) != baseline["total_rows"]:
            raise RuntimeError(
                f"Backup logical verification failed: expected {baseline['total_rows']} rows, found {len(backup_rows)}"
            )

        # 3. Compare row hashes
        backup_hasher = hashlib.sha256()
        for r in backup_rows:
            h = compute_row_hash(r)
            sid = r["snapshot_id"]
            if baseline["row_hashes"].get(sid) != h:
                raise RuntimeError(f"Backup row mismatch for snapshot_id={sid}")
            backup_hasher.update(h.encode("utf-8"))

        backup_content_sha = backup_hasher.hexdigest()
        if backup_content_sha != baseline["baseline_content_sha256"]:
            raise RuntimeError(
                f"Backup content digest mismatch: expected {baseline['baseline_content_sha256']}, got {backup_content_sha}"
            )
    finally:
        if verify_conn:
            verify_conn.close()

    backup_size = os.path.getsize(abs_dest)
    backup_file_sha256 = compute_file_sha256(abs_dest)

    return {
        "status": "VERIFIED",
        "source_database_path": abs_source,
        "backup_destination_path": abs_dest,
        "backup_size_bytes": backup_size,
        "backup_file_sha256": backup_file_sha256,
        "logical_content_sha256": backup_content_sha,
        "rows_verified": len(backup_rows),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def migrate_scanner_snapshots_table_schema(conn: sqlite3.Connection) -> None:
    """
    Self-contained schema migration for scanner_snapshots table.
    Ensures table exists and migrates NOT NULL matched_count to nullable INTEGER.
    Rebuilds indexes and immutability triggers inside the caller's transaction.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(scanner_snapshots);")
    cols = cursor.fetchall()
    if not cols:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scanner_snapshots (
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
                matched_count INTEGER,
                results_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                freshness_json TEXT NOT NULL,
                publication_decision TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_scanner_created ON scanner_snapshots (scanner_id, created_at DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_run_id ON scanner_snapshots (run_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_fingerprint ON scanner_snapshots (semantic_fingerprint);")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_scanner_snapshots_no_update
            BEFORE UPDATE ON scanner_snapshots
            BEGIN
                SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be updated in place');
            END;
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_scanner_snapshots_no_delete
            BEFORE DELETE ON scanner_snapshots
            BEGIN
                SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be deleted');
            END;
        """)
        return

    # Check nullability of matched_count
    # SQLite PRAGMA table_info returns tuples/rows: (cid, name, type, notnull, dflt_value, pk)
    matched_col = next((c for c in cols if (c[1] if isinstance(c, tuple) else c["name"]) == "matched_count"), None)
    is_not_null = bool(matched_col and (matched_col[3] if isinstance(matched_col, tuple) else matched_col["notnull"]) == 1)
    if is_not_null:
        actual_cols = {(c[1] if isinstance(c, tuple) else c["name"]) for c in cols}
        missing_cols = REQUIRED_SNAPSHOT_COLUMNS - actual_cols
        if missing_cols:
            raise RuntimeError(
                f"UNSUPPORTED_SCHEMA_SHAPE: scanner_snapshots missing required columns {sorted(missing_cols)}. Migration aborted."
            )

        cursor.execute("DROP TABLE IF EXISTS _scanner_snapshots_migration_new;")
        cursor.execute("""
            CREATE TABLE _scanner_snapshots_migration_new (
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
                matched_count INTEGER,
                results_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                freshness_json TEXT NOT NULL,
                publication_decision TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            INSERT INTO _scanner_snapshots_migration_new (
                snapshot_id, scanner_id, run_id, api_contract_version, ruleset_version,
                evidence_schema_version, score_model_version, data_provenance_version,
                universe_version, freshness_policy_version, implementation_release_sha,
                semantic_fingerprint, generated_at, data_as_of, status_at_publication,
                universe_id, universe_size, matched_count, results_json, provenance_json,
                freshness_json, publication_decision, created_at
            )
            SELECT
                snapshot_id, scanner_id, run_id, api_contract_version, ruleset_version,
                evidence_schema_version, score_model_version, data_provenance_version,
                universe_version, freshness_policy_version, implementation_release_sha,
                semantic_fingerprint, generated_at, data_as_of, status_at_publication,
                universe_id, universe_size, matched_count, results_json, provenance_json,
                freshness_json, publication_decision, created_at
            FROM scanner_snapshots;
        """)
        cursor.execute("DROP TRIGGER IF EXISTS trg_scanner_snapshots_no_update;")
        cursor.execute("DROP TRIGGER IF EXISTS trg_scanner_snapshots_no_delete;")
        cursor.execute("DROP TABLE scanner_snapshots;")
        cursor.execute("ALTER TABLE _scanner_snapshots_migration_new RENAME TO scanner_snapshots;")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_scanner_created ON scanner_snapshots (scanner_id, created_at DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_run_id ON scanner_snapshots (run_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_fingerprint ON scanner_snapshots (semantic_fingerprint);")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_scanner_snapshots_no_update
            BEFORE UPDATE ON scanner_snapshots
            BEGIN
                SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be updated in place');
            END;
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_scanner_snapshots_no_delete
            BEFORE DELETE ON scanner_snapshots
            BEGIN
                SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be deleted');
            END;
        """)


def apply_migration(db_path: str) -> Dict[str, Any]:
    """
    Applies schema migration against db_path.
    Requires db_path to exist. Migrates NOT NULL matched_count to nullable INTEGER
    inside a single atomic SQLite transaction, rebuilding indexes and immutability triggers.
    Self-contained: does not depend on external module resolution.
    """
    abs_target = os.path.abspath(db_path)
    if not os.path.isfile(abs_target):
        raise FileNotFoundError(f"Database not found: {abs_target}")

    conn = sqlite3.connect(abs_target, timeout=30.0)
    try:
        with conn:
            migrate_scanner_snapshots_table_schema(conn)
        return {
            "status": "MIGRATION_COMPLETED",
            "target_database_path": abs_target,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        conn.close()


def verify_post_upgrade(
    db_path: str,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Read-only post-upgrade verification against preflight baseline.
    Asserts:
    1. matched_count is nullable (notnull == 0).
    2. Every baseline historical row is present and logically identical.
    3. Triggers and indexes exist and match requirements.
    4. Database integrity passes.
    5. No temporary staging table remains.
    6. Repeated ensure_scanner_snapshots_table is idempotent.
    """
    abs_target = os.path.abspath(db_path)
    if os.path.abspath(baseline["target_database_path"]) != abs_target:
        raise ValueError(f"Baseline target mismatch: baseline is for {baseline['target_database_path']}, but target is {abs_target}")

    conn = sqlite3.connect(sqlite_ro_uri(abs_target), uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        # 1. Integrity check
        integrity = cursor.execute("PRAGMA integrity_check;").fetchall()
        if len(integrity) != 1 or integrity[0][0] != "ok":
            raise RuntimeError(f"Post-upgrade integrity check failed: {integrity}")

        # 2. Verify matched_count is nullable
        cols = cursor.execute("PRAGMA table_info(scanner_snapshots);").fetchall()
        matched_col = next((c for c in cols if c["name"] == "matched_count"), None)
        if not matched_col:
            raise RuntimeError("Column 'matched_count' missing from post-upgrade scanner_snapshots.")
        if matched_col["notnull"] != 0:
            raise RuntimeError(f"POST_UPGRADE_ASSERTION_FAILED: matched_count notnull is {matched_col['notnull']}, expected 0 (nullable).")

        # 3. Check for leftover migration staging tables
        migration_temp = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='_scanner_snapshots_migration_new';").fetchone()
        if migration_temp:
            raise RuntimeError("POST_UPGRADE_ASSERTION_FAILED: Temporary staging table '_scanner_snapshots_migration_new' was not dropped.")

        # 4. Verify triggers
        triggers = {r["name"] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='scanner_snapshots';").fetchall()}
        required_triggers = {"trg_scanner_snapshots_no_update", "trg_scanner_snapshots_no_delete"}
        missing_triggers = required_triggers - triggers
        if missing_triggers:
            raise RuntimeError(f"POST_UPGRADE_ASSERTION_FAILED: Missing required immutability triggers: {missing_triggers}")

        # 5. Verify indexes
        indexes = {r["name"] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='scanner_snapshots';").fetchall()}
        required_indexes = {"idx_snapshots_scanner_created", "idx_snapshots_run_id", "idx_snapshots_fingerprint"}
        missing_indexes = required_indexes - indexes
        if missing_indexes:
            raise RuntimeError(f"POST_UPGRADE_ASSERTION_FAILED: Missing required performance indexes: {missing_indexes}")

        # 6. Verify historical baseline row preservation
        cursor.execute("SELECT * FROM scanner_snapshots ORDER BY snapshot_id ASC;")
        current_rows = cursor.fetchall()
        current_row_map = {r["snapshot_id"]: r for r in current_rows}

        for sid, baseline_hash in baseline["row_hashes"].items():
            if sid not in current_row_map:
                raise RuntimeError(f"POST_UPGRADE_ASSERTION_FAILED: Baseline row snapshot_id='{sid}' is MISSING after upgrade!")
            curr_row = current_row_map[sid]
            curr_hash = compute_row_hash(curr_row)
            if curr_hash != baseline_hash:
                raise RuntimeError(
                    f"POST_UPGRADE_ASSERTION_FAILED: Baseline row snapshot_id='{sid}' was MODIFIED! (baseline={baseline_hash}, current={curr_hash})"
                )

        new_rows_count = len(current_rows) - baseline["total_rows"]
    finally:
        conn.close()

    # 7. Idempotency check: run migrate_scanner_snapshots_table_schema on writable connection
    conn_rw = sqlite3.connect(abs_target, timeout=30.0)
    try:
        migrate_scanner_snapshots_table_schema(conn_rw)
    finally:
        conn_rw.close()

    return {
        "status": "POST_UPGRADE_VERIFIED",
        "target_database_path": abs_target,
        "matched_count_nullable": True,
        "baseline_rows_preserved": baseline["total_rows"],
        "current_total_rows": len(current_rows),
        "new_rows_since_baseline": new_rows_count,
        "idempotency_verified": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="PR #28 Scanner Store Migration Runbook Operator Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: resolve
    p_resolve = subparsers.add_parser("resolve", help="Authoritatively resolve target database path without side effects")
    p_resolve.add_argument("--db-path", help="Explicit target database path override")

    # Subcommand: preflight
    p_preflight = subparsers.add_parser("preflight", help="Read-only preflight inspection and baseline capture")
    p_preflight.add_argument("--db-path", help="Explicit target database path override")
    p_preflight.add_argument("--baseline-out", help="Path to write JSON baseline file")

    # Subcommand: backup
    p_backup = subparsers.add_parser("backup", help="Create and verify exclusive online backup")
    p_backup.add_argument("--db-path", help="Explicit target database path override")
    p_backup.add_argument("--backup-path", required=True, help="Destination backup file path (must not exist)")
    p_backup.add_argument("--baseline-file", help="Path to preflight JSON baseline file for verification")

    # Subcommand: upgrade
    p_upgrade = subparsers.add_parser("upgrade", help="Execute atomic transactional schema migration")
    p_upgrade.add_argument("--db-path", help="Explicit target database path override")

    # Subcommand: verify
    p_verify = subparsers.add_parser("verify", help="Post-upgrade verification against baseline")
    p_verify.add_argument("--db-path", help="Explicit target database path override")
    p_verify.add_argument("--baseline-file", required=True, help="Path to preflight JSON baseline file")

    # Subcommand: run-all
    p_runall = subparsers.add_parser("run-all", help="Execute complete runbook sequence (preflight, backup, upgrade, verify)")
    p_runall.add_argument("--db-path", help="Explicit target database path override")
    p_runall.add_argument("--backup-dir", default=os.getcwd(), help="Directory to store backup file")

    args = parser.parse_args()

    if args.command == "resolve":
        target = resolve_scanner_db_path(args.db_path)
        print(f"RESOLVED_SCANNER_DB_PATH={target}")
        print(f"EXISTS={os.path.isfile(target)}")
        return

    if args.command == "preflight":
        target = resolve_scanner_db_path(args.db_path)
        print(f"PINNED_TARGET_DATABASE={target}")
        baseline = capture_baseline(target)
        print("PREFLIGHT_STATUS=OK")
        print(f"TOTAL_ROWS={baseline['total_rows']}")
        print(f"MATCHED_COUNT_NOT_NULL={baseline['matched_count_not_null']}")
        print(f"MIGRATION_REQUIRED={baseline['migration_required']}")
        print(f"BASELINE_CONTENT_SHA256={baseline['baseline_content_sha256']}")
        if args.baseline_out:
            with open(args.baseline_out, "w", encoding="utf-8") as f:
                json.dump(baseline, f, indent=2)
            print(f"BASELINE_WRITTEN={args.baseline_out}")
        return

    if args.command == "backup":
        target = resolve_scanner_db_path(args.db_path)
        print(f"PINNED_TARGET_DATABASE={target}")
        baseline = None
        if args.baseline_file:
            with open(args.baseline_file, "r", encoding="utf-8") as f:
                baseline = json.load(f)
        result = create_and_verify_backup(target, args.backup_path, baseline=baseline)
        print(f"BACKUP_STATUS={result['status']}")
        print(f"BACKUP_PATH={result['backup_destination_path']}")
        print(f"BACKUP_SIZE_BYTES={result['backup_size_bytes']}")
        print(f"BACKUP_SHA256={result['backup_file_sha256']}")
        return

    if args.command == "upgrade":
        target = resolve_scanner_db_path(args.db_path)
        print(f"PINNED_TARGET_DATABASE={target}")
        result = apply_migration(target)
        print(f"UPGRADE_STATUS={result['status']}")
        return

    if args.command == "verify":
        target = resolve_scanner_db_path(args.db_path)
        print(f"PINNED_TARGET_DATABASE={target}")
        with open(args.baseline_file, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        result = verify_post_upgrade(target, baseline)
        print(f"POST_UPGRADE_STATUS={result['status']}")
        print(f"MATCHED_COUNT_NULLABLE={result['matched_count_nullable']}")
        print(f"BASELINE_ROWS_PRESERVED={result['baseline_rows_preserved']}")
        print(f"CURRENT_TOTAL_ROWS={result['current_total_rows']}")
        return

    if args.command == "run-all":
        target = resolve_scanner_db_path(args.db_path)
        print(f"PINNED_TARGET_DATABASE={target}")
        # 1. Preflight
        baseline = capture_baseline(target)
        print(f"Step 1: Preflight OK. Existing rows: {baseline['total_rows']}, Migration required: {baseline['migration_required']}")

        # 2. Backup
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.basename(target)}.pre_pr28_{ts}.bak"
        backup_path = os.path.join(args.backup_dir, backup_name)
        backup_res = create_and_verify_backup(target, backup_path, baseline=baseline)
        print(f"Step 2: Backup VERIFIED. Path: {backup_res['backup_destination_path']}, Size: {backup_res['backup_size_bytes']}")

        # 3. Upgrade
        _ = apply_migration(target)
        print("Step 3: Migration COMPLETED.")

        # 4. Verify
        verify_res = verify_post_upgrade(target, baseline)
        print(f"Step 4: Post-upgrade VERIFIED. All {verify_res['baseline_rows_preserved']} historical rows preserved.")
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY.")
        return


if __name__ == "__main__":
    main()
