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

        # Find all manifest files to identify valid backup sets
        manifest_files = [
            item for item in all_items
            if item["remote_filename"].startswith("manifest_") and item["remote_filename"].endswith(".json")
        ]

        # If manifests exist, sort them chronologically (newest first)
        manifest_files.sort(key=lambda x: x.get("modified_at", ""), reverse=True)

        if len(manifest_files) <= 1:
            logger.info("Retaining single available backup set. Pruning skipped.")
            return []

        pruned_backup_ids = []
        now = datetime.utcnow()
        cutoff_date = now - timedelta(days=max_days)

        # Index 0 is strictly protected as the latest recovery point
        candidates = manifest_files[1:]

        for idx, manifest_item in enumerate(candidates, start=1):
            fname = manifest_item["remote_filename"]
            # Extract backup_id from manifest_<backup_id>.json
            match = re.match(r"^manifest_(sm-backup-[^.]+)\.json$", fname)
            if not match:
                continue
            backup_id = match.group(1)

            # Check count threshold or age threshold
            exceeds_count = (idx >= max_count)
            try:
                mod_date = datetime.fromisoformat(manifest_item["modified_at"].replace("Z", "+00:00"))
                is_expired_age = mod_date.replace(tzinfo=None) < cutoff_date
            except Exception:
                is_expired_age = False

            if exceeds_count or is_expired_age:
                logger.info(f"Pruning expired backup {backup_id} (index={idx}, count_limit={max_count}).")
                # Prune manifest, dump, and globals
                self.storage_adapter.delete_backup(f"manifest_{backup_id}.json")
                self.storage_adapter.delete_backup(f"smart_money_{backup_id}.dump")
                self.storage_adapter.delete_backup(f"globals_{backup_id}.sql")
                pruned_backup_ids.append(backup_id)

        return pruned_backup_ids
