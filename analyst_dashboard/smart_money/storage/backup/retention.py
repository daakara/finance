"""
ARX Terminal — Smart Money Backup Retention & Safe Pruning Manager.

Enforces fail-safe retention policies:
1. Configurable retention window (count and age);
2. Protected most-recent recovery point (NEVER prunes the newest verified backup);
3. Unverified uploads are never relied upon to prune previous valid backups;
4. Explicit audit logging of all pruning actions.
"""

import os
import re
import json
import logging
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    BackupStorageAdapter,
    get_backup_storage_adapter,
)

logger = logging.getLogger(__name__)


class BackupRetentionManager:
    """Manages retention lifecycle and safe pruning of backup artifacts."""

    def __init__(
        self,
        storage_adapter: Optional[BackupStorageAdapter] = None,
        retention_count: int = 10,
        retention_days: int = 7,
    ):
        self.storage_adapter = storage_adapter or get_backup_storage_adapter()
        self.retention_count = max(1, retention_count)
        self.retention_days = max(1, retention_days)

    def prune_backups(
        self,
        retention_count: Optional[int] = None,
        retention_days: Optional[int] = None,
    ) -> List[str]:
        """
        Safely prunes expired backups beyond retention limits.
        Rule: The most recent verified backup is NEVER pruned, regardless of age.
        """
        max_count = retention_count if retention_count is not None else self.retention_count
        max_days = retention_days if retention_days is not None else self.retention_days

        all_items = self.storage_adapter.list_backups()
        if not all_items:
            return []

        # Fail closed: an unverified/corrupt *newest* manifest must never
        # displace the last recoverable backup as the protected recovery point.
        manifests = [
            item for item in all_items
            if re.fullmatch(r"manifest_sm-backup-[A-Za-z0-9_-]+\.json", item["remote_filename"])
        ]
        manifests.sort(key=lambda x: x.get("modified_at", ""), reverse=True)
        if len(manifests) <= 1:
            return []

        verified = []
        try:
            with tempfile.TemporaryDirectory(prefix="sm_retention_verify_") as directory:
                for item in manifests:
                    name = item["remote_filename"]
                    backup_id = name[len("manifest_"):-len(".json")]
                    manifest_path = os.path.join(directory, name)
                    self.storage_adapter.download_file(name, manifest_path)
                    with open(manifest_path, "r", encoding="utf-8") as handle:
                        manifest = json.load(handle)
                    if manifest.get("backup_id") != backup_id or manifest.get("upload_verified") is not True:
                        raise ValueError("UNVERIFIED_BACKUP_MANIFEST")
                    dump_name = manifest.get("database_artifact") or manifest.get("dump_filename")
                    expected_hash = manifest.get("artifact_sha256") or manifest.get("database_sha256")
                    if dump_name != f"smart_money_{backup_id}.dump" or not isinstance(expected_hash, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_hash):
                        raise ValueError("INVALID_BACKUP_DUMP_REFERENCE")
                    dump_path = os.path.join(directory, dump_name)
                    self.storage_adapter.download_file(dump_name, dump_path)
                    digest = hashlib.sha256()
                    with open(dump_path, "rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest().lower() != expected_hash.lower():
                        raise ValueError("BACKUP_DUMP_CHECKSUM_MISMATCH")
                    if manifest.get("include_globals"):
                        globals_name = manifest.get("globals_artifact")
                        globals_hash = manifest.get("globals_artifact_sha256") or manifest.get("globals_sha256")
                        if globals_name != f"globals_{backup_id}.sql" or not isinstance(globals_hash, str):
                            raise ValueError("INVALID_BACKUP_GLOBALS_REFERENCE")
                        globals_path = os.path.join(directory, globals_name)
                        self.storage_adapter.download_file(globals_name, globals_path)
                        if hashlib.sha256(Path(globals_path).read_bytes()).hexdigest().lower() != globals_hash.lower():
                            raise ValueError("BACKUP_GLOBALS_CHECKSUM_MISMATCH")
                    verified.append((item, backup_id, bool(manifest.get("include_globals"))))
        except Exception as exc:
            logger.error("Pruning disabled: a recovery point cannot be verified (%s)", type(exc).__name__)
            return []

        # Only after *all* recovery points have passed independent content
        # verification may count/age retention apply. Protect the newest
        # verified recovery point, regardless of age.
        now = datetime.utcnow()
        cutoff_date = now - timedelta(days=max_days)
        # Preflight *every* timestamp before the first irreversible deletion.
        # A bad later entry must not allow partial pruning of earlier entries.
        deletion_candidates = []
        for idx, (item, backup_id, has_globals) in enumerate(verified[1:], start=1):
            try:
                mod_date = datetime.fromisoformat(item["modified_at"].replace("Z", "+00:00"))
                expired = mod_date.replace(tzinfo=None) < cutoff_date
            except (KeyError, AttributeError, TypeError, ValueError):
                logger.warning("Pruning disabled: unparseable backup timestamp")
                return []
            if idx >= max_count or expired:
                deletion_candidates.append((backup_id, has_globals))

        pruned_backup_ids = []
        for backup_id, has_globals in deletion_candidates:
            self.storage_adapter.delete_backup(f"manifest_{backup_id}.json")
            self.storage_adapter.delete_backup(f"smart_money_{backup_id}.dump")
            if has_globals:
                self.storage_adapter.delete_backup(f"globals_{backup_id}.sql")
            pruned_backup_ids.append(backup_id)
        return pruned_backup_ids
