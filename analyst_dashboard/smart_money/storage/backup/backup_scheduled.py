"""Dedicated scheduled Smart Money production backup entry point.

Enforces:
1. Explicit scheduled authorization (ARX_BACKUP_SCHEDULED_AUTHORIZED=true);
2. Distinct from one-shot authorization (rejects ARX_BACKUP_ONESHOT_AUTHORIZED=true);
3. Authenticated PostgreSQL session_user and current_user must both be 'smart_money_backup';
4. Production environment, approved release identity, and Cloudflare R2 custody;
5. Mutual exclusion via PostgreSQL session advisory lock (LOCK_BACKUP_EXECUTION);
6. Single backup execution with verified upload and manifest integrity;
7. Never invokes retention pruning, SEC ingestion, scanning, trading or restore;
8. Fail-closed without credential leakage;
9. Clean exit code after single execution.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, Optional

import psycopg2

from analyst_dashboard.smart_money.storage.config import get_database_url
from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
from analyst_dashboard.smart_money.storage.backup.storage_adapter import get_backup_storage_adapter


SHA40 = re.compile(r"[0-9a-f]{40}\Z")


def preflight(expected_release_sha: Optional[str], env: Dict[str, str]) -> str:
    """Validate all environment and authorization preconditions before touching database or storage.

    Returns the verified 40-character release SHA.
    """
    if env.get("ARX_BACKUP_SCHEDULED_AUTHORIZED") != "true":
        raise RuntimeError("BACKUP_SCHEDULED_AUTHORIZATION_MISSING")

    if env.get("ARX_BACKUP_ONESHOT_AUTHORIZED") == "true":
        raise RuntimeError("BACKUP_SCHEDULED_REJECTS_ONESHOT_AUTHORIZATION")

    if env.get("ARX_ENVIRONMENT") != "production":
        raise RuntimeError("BACKUP_SCHEDULED_REQUIRES_PRODUCTION_ENVIRONMENT")

    sec_role = env.get("ARX_SEC_EGRESS_ROLE", "").strip().upper()
    if sec_role not in ("DISABLED", "STANDBY"):
        raise RuntimeError("BACKUP_SCHEDULED_REQUIRES_SEC_EGRESS_DISABLED")

    raw_sha = (expected_release_sha or env.get("ARX_RELEASE_SHA") or env.get("RAILWAY_GIT_COMMIT_SHA") or "").strip()
    if not SHA40.fullmatch(raw_sha):
        raise RuntimeError("BACKUP_SCHEDULED_INVALID_RELEASE_IDENTITY")
    release_sha = raw_sha

    configured_sha = env.get("ARX_RELEASE_SHA", "").strip()
    if expected_release_sha and configured_sha and configured_sha != expected_release_sha.strip():
        raise RuntimeError("BACKUP_SCHEDULED_RELEASE_IDENTITY_MISMATCH")

    if env.get("SMART_MONEY_DATABASE_ROLE") != "smart_money_backup":
        raise RuntimeError("BACKUP_SCHEDULED_REQUIRES_BACKUP_DATABASE_ROLE")

    if env.get("ARX_BACKUP_PROVIDER") != "cloudflare_r2":
        raise RuntimeError("BACKUP_SCHEDULED_REQUIRES_CLOUDFLARE_R2")

    if env.get("ARX_BACKUP_BUCKET") != "arx-smart-money-backups":
        raise RuntimeError("BACKUP_SCHEDULED_WRONG_BUCKET")

    if env.get("ARX_BACKUP_PREFIX", "").rstrip("/") != "production":
        raise RuntimeError("BACKUP_SCHEDULED_WRONG_PREFIX")

    return release_sha


def verify_backup_database_identity() -> None:
    """Read-only attestation of the authenticated PostgreSQL session_user and current_user."""
    dsn = get_database_url()
    if not dsn or not dsn.startswith(("postgres://", "postgresql://")):
        raise RuntimeError("BACKUP_SCHEDULED_INVALID_DATABASE_URL")
    conn = None
    try:
        conn = psycopg2.connect(dsn, connect_timeout=5)
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SELECT session_user, current_user")
            row = cur.fetchone()
        if row != ("smart_money_backup", "smart_money_backup"):
            raise RuntimeError("BACKUP_SCHEDULED_DATABASE_PRINCIPAL_MISMATCH")
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc) == "BACKUP_SCHEDULED_DATABASE_PRINCIPAL_MISMATCH":
            raise
        raise RuntimeError("BACKUP_SCHEDULED_DATABASE_PRINCIPAL_UNVERIFIABLE") from None
    finally:
        if conn is not None:
            conn.close()


def run_scheduled_backup(expected_release_sha: Optional[str] = None) -> Dict[str, Any]:
    """Executes a single scheduled backup pass with strict preflights and manifest verification.

    Never performs retention pruning, SEC ingestion, scanning, trading, or restoration.
    """
    env = dict(os.environ)
    release_sha = preflight(expected_release_sha, env)
    verify_backup_database_identity()

    adapter = get_backup_storage_adapter()
    if (
        not adapter.is_external()
        or getattr(adapter, "provider_name", None) != "cloudflare_r2"
        or getattr(adapter, "bucket", None) != "arx-smart-money-backups"
        or getattr(adapter, "prefix", "").rstrip("/") != "production"
    ):
        raise RuntimeError("BACKUP_SCHEDULED_EXTERNAL_CUSTODY_NOT_VERIFIED")

    engine = SmartMoneyBackupEngine(storage_adapter=adapter)
    manifest = engine.create_backup(include_globals=False)

    if (
        manifest.get("status") != "SUCCESS"
        or not manifest.get("upload_verified")
        or not manifest.get("is_external")
        or manifest.get("remote_bucket") != "arx-smart-money-backups"
        or manifest.get("remote_object_prefix", "").rstrip("/") != "production"
        or manifest.get("source_release_sha") != release_sha
        or not manifest.get("artifact_sha256")
        or manifest.get("artifact_sha256") != manifest.get("database_sha256")
        or not manifest.get("backup_id")
    ):
        raise RuntimeError("BACKUP_SCHEDULED_UNVERIFIED_UPLOAD")

    return {
        "backup_id": manifest["backup_id"],
        "upload_verified": True,
        "artifact_sha256": manifest["artifact_sha256"],
        "artifact_size_bytes": manifest.get("artifact_size_bytes"),
        "restore_verified": bool(manifest.get("restore_verified", False)),
        "restore_validation_status": manifest.get("restore_validation_status", "PENDING_VERIFICATION"),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dedicated scheduled Smart Money R2 backup entry point")
    parser.add_argument("--expected-release-sha", default=None, help="Approved 40-character release SHA (optional; defaults to ARX_RELEASE_SHA)")
    args = parser.parse_args(argv)

    try:
        result = run_scheduled_backup(expected_release_sha=args.expected_release_sha)
    except Exception as exc:
        error_name = type(exc).__name__
        reason = str(exc) if isinstance(exc, RuntimeError) else error_name
        # Sanitize against credential or DSN leakage
        if any(token in reason.lower() for token in ("postgres://", "postgresql://", "password", "secret")):
            reason = error_name
        print(f"BACKUP_SCHEDULED_FAILED: {reason}", file=sys.stderr)
        return 1

    print(f"BACKUP_SCHEDULED_SUCCESS: {result['backup_id']} sha256={result['artifact_sha256']}")
    if not result["restore_verified"]:
        print("BACKUP_SCHEDULED_RESTORE_VERIFICATION_PENDING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
