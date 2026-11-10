"""Explicit one-shot Smart Money production backup entry point.

Never starts SEC ingestion, scans, scheduling, retention pruning or restores.
No production service should invoke this module without a separate backup-only GO.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
from analyst_dashboard.smart_money.storage.backup.storage_adapter import get_backup_storage_adapter


SHA40 = re.compile(r"[0-9a-f]{40}\Z")


def preflight(expected_release_sha: str, env: dict[str, str]) -> None:
    """Require a scoped explicit authorization before even connecting to storage."""
    if env.get("ARX_BACKUP_ONESHOT_AUTHORIZED") != "true":
        raise RuntimeError("BACKUP_ONLY_AUTHORIZATION_MISSING")
    if env.get("ARX_ENVIRONMENT") != "production":
        raise RuntimeError("BACKUP_ONLY_REQUIRES_PRODUCTION_ENVIRONMENT")
    if env.get("ARX_SEC_EGRESS_ROLE", "").strip().upper() not in ("DISABLED", "STANDBY"):
        raise RuntimeError("BACKUP_ONLY_REQUIRES_SEC_EGRESS_DISABLED")
    if not SHA40.fullmatch(expected_release_sha or ""):
        raise RuntimeError("BACKUP_ONLY_INVALID_EXPECTED_RELEASE")
    if env.get("ARX_RELEASE_SHA") != expected_release_sha:
        raise RuntimeError("BACKUP_ONLY_RELEASE_IDENTITY_MISMATCH")
    if env.get("SMART_MONEY_DATABASE_ROLE") != "smart_money_backup":
        raise RuntimeError("BACKUP_ONLY_REQUIRES_BACKUP_DATABASE_ROLE")
    if env.get("ARX_BACKUP_PROVIDER") != "cloudflare_r2":
        raise RuntimeError("BACKUP_ONLY_REQUIRES_CLOUDFLARE_R2")
    if env.get("ARX_BACKUP_BUCKET") != "arx-smart-money-backups":
        raise RuntimeError("BACKUP_ONLY_WRONG_BUCKET")
    if env.get("ARX_BACKUP_PREFIX") != "production":
        raise RuntimeError("BACKUP_ONLY_WRONG_PREFIX")


def run_backup_once(expected_release_sha: str) -> dict:
    preflight(expected_release_sha, dict(os.environ))
    adapter = get_backup_storage_adapter()
    if (not adapter.is_external()
            or getattr(adapter, "provider_name", None) != "cloudflare_r2"
            or getattr(adapter, "bucket", None) != "arx-smart-money-backups"
            or getattr(adapter, "prefix", None) != "production"):
        raise RuntimeError("BACKUP_ONLY_EXTERNAL_CUSTODY_NOT_VERIFIED")
    manifest = SmartMoneyBackupEngine(storage_adapter=adapter).create_backup(include_globals=False)
    if (manifest.get("status") != "SUCCESS"
            or not manifest.get("upload_verified")
            or not manifest.get("is_external")
            or manifest.get("remote_bucket") != "arx-smart-money-backups"
            or manifest.get("remote_object_prefix") != "production"
            or manifest.get("source_release_sha") != expected_release_sha):
        raise RuntimeError("BACKUP_ONLY_UNVERIFIED_UPLOAD")
    return {
        "backup_id": manifest["backup_id"],
        "upload_verified": True,
        "restore_verified": bool(manifest.get("restore_verified", False)),
        "restore_validation_status": manifest.get("restore_validation_status"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gated, unscheduled, one-shot Smart Money R2 backup")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable one backup only")
    parser.add_argument("--expected-release-sha", required=True, help="Approved 40-character release SHA")
    args = parser.parse_args(argv)
    if not args.execute:
        print("BACKUP_ONLY_NOT_EXECUTED: --execute required", file=sys.stderr)
        return 2
    try:
        result = run_backup_once(args.expected_release_sha)
    except Exception as exc:
        # Do not print database URLs, adapter exceptions, or credentials.
        print(f"BACKUP_ONLY_FAILED: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"BACKUP_ONLY_UPLOAD_VERIFIED: {result['backup_id']}")
    if not result["restore_verified"]:
        print("BACKUP_ONLY_RESTORE_VERIFICATION_PENDING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
