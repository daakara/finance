"""
ARX Terminal — Smart Money Automated Restore & Parity Verifier.

Executes automated end-to-end restore validation against an isolated PostgreSQL 16 target:
1. Downloads artifacts and validates SHA-256 digests;
2. Restores cluster globals via psql;
3. Restores database schema and data via pg_restore;
4. Verifies canonical table presence and enabled immutability triggers;
5. Records restored row counts (source-to-target parity is not established);
6. Validates active snapshot pointer and application read contract;
7. Emits machine-readable validation receipt.
"""

import os
import json
import time
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional, List
from ipaddress import ip_address

try:
    import psycopg2
    from psycopg2.extensions import parse_dsn
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    PSYCOPG2_AVAILABLE = False

from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    BackupStorageAdapter,
    get_backup_storage_adapter,
    compute_sha256,
)
from analyst_dashboard.smart_money.storage.backup.backup_engine import find_pg_binary
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.storage.config import get_database_url

logger = logging.getLogger(__name__)

CANONICAL_REQUIRED_TABLES = [
    "scanner_active_snapshots",
    "scanner_snapshots",
    "form4_raw_filings",
    "sm_run_requests",
    "form4_parsed_transactions",
    "governed_issuers",
    "sm_amendments",
    "sm_audit_events",
    "sm_reconciliation_receipts",
    "sm_schema_migrations",
    "sm_source_availability_ledger",
    "sm_source_gaps",
]

REQUIRED_IMMUTABILITY_TRIGGERS = {
    "trg_raw_filings_immutable": "form4_raw_filings",
    "trg_receipts_append_only": "sm_reconciliation_receipts",
    "trg_snapshots_append_only": "scanner_snapshots",
    "trg_audit_append_only": "sm_audit_events",
}


def _target_identity(dsn):
    """Ignore credentials and URL spelling when comparing database destinations."""
    parts = parse_dsn(dsn)
    host = parts.get("hostaddr") or parts.get("host", "")
    if host in ("localhost", "127.0.0.1", "::1"):
        host = "loopback"
    return (host.rstrip(".").lower(), parts.get("port", "5432"),
            parts.get("dbname") or parts.get("user", ""))


def _check_destination(target_dsn, primary_dsn, override):
    if override:
        return
    if primary_dsn and _target_identity(target_dsn) == _target_identity(primary_dsn):
        raise RuntimeError("DESTINATION_SAFETY_VIOLATION: Target matches primary database.")
    parts = parse_dsn(target_dsn)
    host = parts.get("host", "")
    # Only explicit local rehearsal targets are accepted without an override.
    # Do not inherit PGHOST, service files, or multi-host failover destinations.
    hosts = [host]
    if parts.get("hostaddr"):
        hosts.append(parts["hostaddr"])
    for endpoint in hosts:
        local = endpoint == "localhost" or endpoint.startswith("/")
        try:
            local = local or ip_address(endpoint).is_loopback
        except ValueError:
            pass
        if not local or "," in endpoint or "service" in parts:
            raise RuntimeError("DESTINATION_SAFETY_VIOLATION: Explicit local target required.")



class BackupRestoreVerifier:
    """Automated restore verification workflow."""

    def __init__(self, storage_adapter: Optional[BackupStorageAdapter] = None):
        self.storage_adapter = storage_adapter or get_backup_storage_adapter()

    def verify_backup(
        self,
        backup_id: str,
        target_dsn: str,
        allow_production_override: bool = False,
        required_tables: Optional[List[str]] = None,
        require_active_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes complete verification of a backup against target_dsn.
        Enforces destination safety fencing, cryptographic integrity,
        and post-restore structural and application-read checks.
        Returns machine-readable verification receipt.
        """
        start_time = time.time()

        # Check before any download or destructive PostgreSQL command.
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("CANONICAL_POSTGRES_DRIVER_UNAVAILABLE: psycopg2 is required.")
        _check_destination(target_dsn, get_database_url(), allow_production_override)

        with tempfile.TemporaryDirectory(prefix="arx_restore_verify_") as tmp_dir:
            # 1. Download manifest
            manifest_name = f"manifest_{backup_id}.json"
            local_manifest = os.path.join(tmp_dir, manifest_name)
            self.storage_adapter.download_file(manifest_name, local_manifest)

            with open(local_manifest, "r", encoding="utf-8") as f:
                try:
                    manifest = json.load(f)
                except Exception as e:
                    raise ValueError(f"MANIFEST_CORRUPT: Failed to parse JSON manifest: {e}")

            if not isinstance(manifest, dict) or "backup_id" not in manifest:
                raise ValueError("MANIFEST_INVALID: Manifest missing required 'backup_id' field.")

            if manifest["backup_id"] != backup_id:
                raise ValueError("MANIFEST_INVALID: Backup identity mismatch.")

            # 2. Download dump (and globals if present)
            globals_name = manifest.get("globals_filename") or manifest.get("globals_artifact")
            dump_name = manifest.get("dump_filename") or manifest.get("database_artifact")
            if not dump_name:
                raise ValueError("MANIFEST_INVALID: Missing dump artifact filename in manifest.")

            local_dump = os.path.join(tmp_dir, dump_name)
            self.storage_adapter.download_file(dump_name, local_dump)

            sha_start = time.time()
            dl_dump_sha = compute_sha256(local_dump)
            sha_duration_sec = time.time() - sha_start

            expected_dump_sha = manifest.get("artifact_sha256") or manifest.get("database_sha256")
            if dl_dump_sha != expected_dump_sha:
                raise ValueError(
                    f"DIGEST_MISMATCH_DUMP: Expected {expected_dump_sha}, got {dl_dump_sha}"
                )

            local_globals = None
            if globals_name:
                local_globals = os.path.join(tmp_dir, globals_name)
                self.storage_adapter.download_file(globals_name, local_globals)
                dl_globals_sha = compute_sha256(local_globals)
                expected_globals_sha = manifest.get("globals_artifact_sha256") or manifest.get("globals_sha256")
                if dl_globals_sha != expected_globals_sha:
                    raise ValueError(
                        f"DIGEST_MISMATCH_GLOBALS: Expected {expected_globals_sha}, got {dl_globals_sha}"
                    )

            env = os.environ.copy()

            # Gate: Target Connection & Version Verification Pre-check
            if not PSYCOPG2_AVAILABLE:
                raise RuntimeError(
                    "CANONICAL_POSTGRES_DRIVER_UNAVAILABLE: "
                    "psycopg2 is required for restore verification, but is not installed."
                )

            try:
                test_conn = psycopg2.connect(target_dsn)
                with test_conn.cursor() as cur:
                    cur.execute("SHOW server_version_num;")
                    target_ver_num = int(cur.fetchone()[0])
                    target_major = target_ver_num // 10000
                    manifest_major = manifest.get("postgres_major_version")
                    if manifest_major and target_major != int(manifest_major):
                        raise RuntimeError(
                            f"POSTGRES_VERSION_MISMATCH: Manifest specifies major version {manifest_major}, "
                            f"but target database is running version {target_major}."
                        )
                test_conn.close()
            except psycopg2.OperationalError as e:
                raise ConnectionError(f"DATABASE_CONNECTION_FAILED: Unable to connect to target database: {e}")

            # 3. Restore Globals via psql (only if present)
            if local_globals:
                psql_bin = find_pg_binary("psql")
                cmd_globals = [psql_bin, "-X", "-v", "ON_ERROR_STOP=1",
                               "-d", target_dsn, "-f", local_globals]
                proc_globals = subprocess.run(cmd_globals, env=env, capture_output=True, text=True)
                if proc_globals.returncode != 0:
                    raise RuntimeError("GLOBALS_RESTORE_FAILED: psql returned nonzero status.")

            # 4. Restore Database via pg_restore
            pg_restore_bin = find_pg_binary("pg_restore")
            cmd_restore = [pg_restore_bin, "-d", target_dsn, "--clean", "--if-exists",
                           "--no-owner", "--no-privileges", "--exit-on-error", local_dump]
            proc_restore = subprocess.run(cmd_restore, env=env, capture_output=True, text=True)
            if proc_restore.returncode != 0:
                raise RuntimeError("PG_RESTORE_FAILED: pg_restore returned nonzero status.")

            # 5. Connect and Verify Structural and Semantic Parity
            conn = psycopg2.connect(target_dsn)
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Count tables
                    cur.execute("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
                    """)
                    present_tables = {row["table_name"] for row in cur.fetchall()}
                    table_count = len(present_tables)

                    # Verify required tables
                    required = list(dict.fromkeys(CANONICAL_REQUIRED_TABLES + (required_tables or [])))
                    missing = [t for t in required if t not in present_tables]
                    if missing:
                        raise RuntimeError(
                            f"PARTIAL_RESTORE_DETECTED: Restored database missing required tables: {missing}"
                        )

                    # Validate named triggers on the correct public tables, enabled for
                    # normal writes, BEFORE UPDATE and DELETE, using the expected function.
                    cur.execute("""
                        SELECT t.tgname, c.relname, t.tgenabled, t.tgtype, p.proname,
                               pn.nspname AS function_schema
                        FROM pg_trigger t
                        JOIN pg_class c ON c.oid = t.tgrelid
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        JOIN pg_proc p ON p.oid = t.tgfoid
                        JOIN pg_namespace pn ON pn.oid = p.pronamespace
                        WHERE n.nspname = 'public' AND NOT t.tgisinternal;
                    """)
                    triggers = {row["tgname"]: row for row in cur.fetchall()}
                    for name, table in REQUIRED_IMMUTABILITY_TRIGGERS.items():
                        row = triggers.get(name)
                        if (not row or row["relname"] != table
                                or row["tgenabled"] not in ("O", "A")
                                or row["tgtype"] != 27
                                or row["proname"] != "abort_immutable_record"
                                or row["function_schema"] != "public"):
                            raise RuntimeError("IMMUTABILITY_TRIGGER_INVALID: " + name)
                    trigger_count = len(REQUIRED_IMMUTABILITY_TRIGGERS)

                    # Count rows per core table
                    row_counts = {}
                    for t in required:
                        if t in present_tables:
                            cur.execute(sql.SQL("SELECT count(*) FROM public.{};").format(sql.Identifier(t)))
                            row_counts[t] = cur.fetchone()["count"]

                    # Active pointer
                    cur.execute("SELECT * FROM scanner_active_snapshots WHERE scan_type = 'SMART_MONEY';")
                    active_ptr = cur.fetchone()
                    active_snap_id = active_ptr["active_snapshot_id"] if active_ptr else None

                    if require_active_snapshot and not active_snap_id:
                        raise RuntimeError(
                            "PARTIAL_RESTORE_DETECTED: Restored database missing active snapshot pointer in scanner_active_snapshots."
                        )

                    # Verify application read contract via repo
                    repo = PostgresSmartMoneyRepository(dsn=target_dsn)
                    active_snap = repo.get_latest_active_snapshot("SMART_MONEY")

                    if require_active_snapshot and (
                        active_snap is None or active_snap.get("snapshot_id") != active_snap_id
                    ):
                        raise RuntimeError("APPLICATION_READ_FAILED: Active snapshot cannot be read consistently.")

                    receipt = {
                        "verification_scope": "ISOLATED_RESTORE_STRUCTURE_AND_APPLICATION_READ",
                        "source_data_parity_verified": False,
                        "roles_and_grants_verified": False,
                        "trigger_behavior_verified": False,
                        "backup_id": backup_id,
                        "status": "VERIFIED_RECOVERABLE",
                        "tables_count": table_count,
                        "present_tables": sorted(list(present_tables)),
                        "row_counts": row_counts,
                        "triggers_count": trigger_count,
                        "active_snapshot_id": active_snap_id,
                        "globals_restored": bool(local_globals),
                        "application_read_successful": bool(active_snap is not None),
                        "sha256_overhead_sec": round(sha_duration_sec, 4),
                        "verification_duration_sec": round(time.time() - start_time, 3),
                        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    logger.info(f"Backup {backup_id} verified successfully: {receipt}")
                    return receipt
            finally:
                conn.close()
