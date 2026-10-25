"""
ARX Terminal — Smart Money PostgreSQL Automated Backup Engine.

Executes transactionally consistent logical backups using native PostgreSQL tools:
1. pg_dumpall --globals-only (roles, authentication, cluster grants);
2. pg_dump -Fc (custom compressed format with internal table of contents);
3. Mutual exclusion via LOCK_BACKUP_EXECUTION advisory lock;
4. Off-volume / off-host storage upload with SHA-256 integrity verification;
5. Machine-readable manifest and audit event logging.
"""

import os
import sys
import json
import uuid
import shutil
import hashlib
import logging
import tempfile
import subprocess
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, Optional

try:
    import psycopg2
    from psycopg2.extras import Json
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    def Json(val):  # type: ignore[no-redef]
        return json.dumps(val)
    PSYCOPG2_AVAILABLE = False

from analyst_dashboard.smart_money.storage.config import (
    get_database_url,
    SMART_MONEY_POSTGRES_MAJOR,
    ACTIVE_SCHEMA_VERSION,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    acquire_backup_lock,
    release_backup_lock,
)
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    BackupStorageAdapter,
    get_backup_storage_adapter,
    compute_sha256,
)

logger = logging.getLogger(__name__)


def find_pg_binary(binary_name: str) -> str:
    """Locates a PostgreSQL client binary in PATH or pgserver distribution."""
    # 1. Check system PATH
    found = shutil.which(binary_name)
    if found:
        return found

    # 2. Check if pgserver package has binaries (for local test environments)
    try:
        import pgserver
        base_dir = os.path.dirname(pgserver.__file__)
        exe_suffix = ".exe" if os.name == "nt" else ""
        candidate = os.path.join(base_dir, "pginstall", "bin", f"{binary_name}{exe_suffix}")
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass

    raise RuntimeError(
        f"POSTGRES_BINARY_NOT_FOUND: Could not find required binary '{binary_name}' in PATH or pgserver."
    )


class SmartMoneyBackupEngine:
    """Automated logical backup engine with off-volume / off-host upload."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        storage_adapter: Optional[BackupStorageAdapter] = None,
    ):
        self.dsn = dsn or get_database_url()
        if not self.dsn:
            raise ValueError("DATABASE_URL_REQUIRED: A valid PostgreSQL connection URL is required.")
        self.storage_adapter = storage_adapter or get_backup_storage_adapter()

    def _parse_dsn(self) -> Dict[str, Any]:
        """Parses DSN into connection components."""
        parsed = urlparse(self.dsn)
        return {
            "user": parsed.username or "postgres",
            "password": parsed.password or "",
            "host": parsed.hostname or "127.0.0.1",
            "port": str(parsed.port or 5432),
            "dbname": parsed.path.lstrip("/") or "postgres",
        }

    def create_backup(
        self,
        custom_backup_id: Optional[str] = None,
        include_globals: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes a complete, consistent backup cycle.
        For routine scheduled backups, include_globals=False (pg_dump -Fc only),
        which eliminates any superuser requirement and protects cluster credentials.
        Cluster globals (pg_dumpall --globals-only) are reserved for exceptional
        administrative exports when explicitly requested (include_globals=True).
        Returns the verified backup manifest.
        """
        started_at = datetime.utcnow().isoformat() + "Z"
        conn_info = self._parse_dsn()
        backup_id = custom_backup_id or f"sm-backup-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

        # 1. Establish session connection and acquire mutual exclusion lock
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError(
                "CANONICAL_POSTGRES_DRIVER_UNAVAILABLE: "
                "psycopg2 is required for SmartMoneyBackupEngine, but is not installed."
            )
        lock_conn = psycopg2.connect(self.dsn)
        try:
            if not acquire_backup_lock(lock_conn):
                raise RuntimeError(
                    "CONCURRENT_BACKUP_BLOCKED: Another backup execution is active. "
                    "Failing closed to prevent concurrent resource contention."
                )

            # Query database server version
            with lock_conn.cursor() as cur:
                cur.execute("SHOW server_version;")
                pg_version_str = str(cur.fetchone()[0])

            # 2. Setup isolated temporary directory for raw capture
            with tempfile.TemporaryDirectory(prefix="arx_backup_tmp_") as tmp_dir:
                globals_filename = f"globals_{backup_id}.sql" if include_globals else None
                globals_local_path = os.path.join(tmp_dir, globals_filename) if globals_filename else None

                dump_filename = f"smart_money_{backup_id}.dump"
                dump_local_path = os.path.join(tmp_dir, dump_filename)

                manifest_filename = f"manifest_{backup_id}.json"
                manifest_local_path = os.path.join(tmp_dir, manifest_filename)

                # Set environment for CLI tools
                env = os.environ.copy()
                if conn_info["password"]:
                    env["PGPASSWORD"] = conn_info["password"]

                # 3. Capture Cluster Globals via pg_dumpall --globals-only (only if requested)
                globals_sha = None
                globals_size = 0
                receipt_globals = None
                if include_globals:
                    pg_dumpall_bin = find_pg_binary("pg_dumpall")
                    cmd_globals = [
                        pg_dumpall_bin,
                        "-h", conn_info["host"],
                        "-p", conn_info["port"],
                        "-U", conn_info["user"],
                        "--globals-only",
                        "-f", globals_local_path,
                    ]
                    proc_globals = subprocess.run(cmd_globals, env=env, capture_output=True, text=True)
                    if proc_globals.returncode != 0:
                        raise RuntimeError(f"PG_DUMPALL_FAILED: {proc_globals.stderr}")

                    globals_sha = compute_sha256(globals_local_path)
                    globals_size = os.path.getsize(globals_local_path)
                    receipt_globals = self.storage_adapter.upload_file(globals_local_path, globals_filename)

                # 4. Capture Database Custom Dump via pg_dump -Fc
                pg_dump_bin = find_pg_binary("pg_dump")
                cmd_dump = [
                    pg_dump_bin,
                    "-h", conn_info["host"],
                    "-p", conn_info["port"],
                    "-U", conn_info["user"],
                    "-d", conn_info["dbname"],
                    "-Fc",
                    "--no-owner",
                    "--no-privileges",
                    "-f", dump_local_path,
                ]
                proc_dump = subprocess.run(cmd_dump, env=env, capture_output=True, text=True)
                if proc_dump.returncode != 0:
                    raise RuntimeError(f"PG_DUMP_FAILED: {proc_dump.stderr}")

                dump_sha = compute_sha256(dump_local_path)
                dump_size = os.path.getsize(dump_local_path)

                # 5. Upload database dump to storage adapter
                receipt_dump = self.storage_adapter.upload_file(dump_local_path, dump_filename)

                completed_at = datetime.utcnow().isoformat() + "Z"

                # Verification check: dump must match, and globals (if included) must match
                is_upload_verified = bool(receipt_dump.get("sha256") == dump_sha)
                if include_globals and receipt_globals:
                    is_upload_verified = bool(is_upload_verified and receipt_globals.get("sha256") == globals_sha)

                # 6. Build Manifest
                manifest = {
                    "backup_id": backup_id,
                    "source_environment": os.environ.get("ARX_ENVIRONMENT", "rehearsal"),
                    "source_database": conn_info["dbname"],
                    "source_release_sha": os.environ.get("ARX_RELEASE_SHA", "c8cc1cda0be5a2a62816d3024b6499515e97101a"),
                    "postgres_major_version": 16,
                    "postgres_version": pg_version_str,
                    "schema_version": ACTIVE_SCHEMA_VERSION,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "database_artifact": dump_filename,
                    "globals_artifact": globals_filename,
                    "database_sha256": dump_sha,
                    "globals_sha256": globals_sha,
                    "artifact_size_bytes": dump_size,
                    "artifact_sha256": dump_sha,
                    "globals_artifact_sha256": globals_sha,
                    "globals_size_bytes": globals_size,
                    "remote_storage_provider": getattr(self.storage_adapter, "provider_name", "local_off_volume"),
                    "remote_bucket": getattr(self.storage_adapter, "bucket", "local"),
                    "remote_object_prefix": getattr(self.storage_adapter, "prefix", ""),
                    "storage_destination": self.storage_adapter.get_storage_destination(),
                    "dump_filename": dump_filename,
                    "globals_filename": globals_filename,
                    "include_globals": include_globals,
                    "upload_verified": is_upload_verified,
                    "restore_verified": False,
                    "is_external": self.storage_adapter.is_external(),
                    "restore_validation_status": "PENDING_VERIFICATION",
                }

                with open(manifest_local_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                self.storage_adapter.upload_file(manifest_local_path, manifest_filename)

                # 7. Write audit event to database
                with lock_conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sm_audit_events (
                            event_id, event_type, subject_id, payload, event_hash
                        ) VALUES (%s, %s, %s, %s, %s);
                    """, (
                        str(uuid.uuid4()),
                        "BACKUP_COMPLETED",
                        backup_id,
                        Json({
                            "backup_id": backup_id,
                            "dump_size": dump_size,
                            "dump_sha": dump_sha,
                            "globals_sha": globals_sha,
                            "include_globals": include_globals,
                            "destination": manifest["storage_destination"],
                            "is_external": manifest["is_external"],
                        }),
                        hashlib.sha256(f"BACKUP_{backup_id}_{dump_sha}".encode()).hexdigest(),
                    ))
                lock_conn.commit()

                logger.info(f"Backup {backup_id} completed successfully (dump={dump_size} bytes, sha={dump_sha[:12]}..., globals={include_globals}).")
                return manifest

        finally:
            try:
                release_backup_lock(lock_conn)
            except Exception as e:
                logger.warning(f"Error releasing backup advisory lock: {e}")
            finally:
                lock_conn.close()
