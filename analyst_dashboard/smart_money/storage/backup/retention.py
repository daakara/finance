"""
ARX Terminal — Smart Money Backup Retention & Safe Pruning Manager.

Enforces fail-safe retention policies:
1. Configurable retention window (count and age);
2. Protected most-recent recovery point (NEVER prunes the newest verified backup);
3. Unverified uploads are never relied upon to prune previous valid backups;
4. Explicit audit logging of all pruning actions;
5. Bounded execution: respects monotonic cycle deadline and network attempt budgets,
   deferring pruning rather than causing cycle timeout;
6. Strict legacy manifest contract: verifies exact artifact and dump checksum binding
   without weakening verification standards or deletion safety.
"""

import os
import re
import json
import time
import logging
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, NamedTuple

from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    BackupStorageAdapter,
    get_backup_storage_adapter,
)

logger = logging.getLogger(__name__)


class RetentionOutcome:
    SUCCESS = "SUCCESS"
    PARTIAL_DEFERRED = "PARTIAL_DEFERRED"
    DEFERRED_DEADLINE_EXCEEDED = "DEFERRED_DEADLINE_EXCEEDED"
    DEFERRED_ATTEMPTS_EXCEEDED = "DEFERRED_ATTEMPTS_EXCEEDED"
    DISABLED_UNVERIFIED_RECOVERY_POINT = "DISABLED_UNVERIFIED_RECOVERY_POINT"
    DISABLED_UNPARSEABLE_TIMESTAMP = "DISABLED_UNPARSEABLE_TIMESTAMP"
    NO_ACTION_REQUIRED = "NO_ACTION_REQUIRED"


class RetentionPlan(NamedTuple):
    outcome: str
    protected_backup_id: Optional[str]
    protected_window_count: int
    deletion_candidates: List[str]
    items_verified: int
    total_manifests: int


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
        self.last_outcome: str = RetentionOutcome.NO_ACTION_REQUIRED
        self.last_details: Dict[str, Any] = {}

    def _adapter_download_file(
        self, remote_filename: str, local_dest_path: str, deadline_monotonic: Optional[float] = None
    ) -> str:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"RETENTION_DEADLINE_EXCEEDED: download_file {remote_filename}")
        try:
            return self.storage_adapter.download_file(
                remote_filename, local_dest_path, deadline_monotonic=deadline_monotonic
            )
        except TypeError as te:
            if "deadline_monotonic" in str(te):
                if deadline_monotonic is not None:
                    raise TimeoutError(
                        f"RETENTION_DEADLINE_UNSUPPORTED: storage adapter {type(self.storage_adapter).__name__}.download_file "
                        "does not accept deadline_monotonic; cannot fall back to unbounded call"
                    )
                return self.storage_adapter.download_file(remote_filename, local_dest_path)
            raise

    def _adapter_list_backups(self, deadline_monotonic: Optional[float] = None) -> List[Dict[str, Any]]:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("RETENTION_DEADLINE_EXCEEDED: list_backups")
        try:
            return self.storage_adapter.list_backups(deadline_monotonic=deadline_monotonic)
        except TypeError as te:
            if "deadline_monotonic" in str(te):
                if deadline_monotonic is not None:
                    raise TimeoutError(
                        f"RETENTION_DEADLINE_UNSUPPORTED: storage adapter {type(self.storage_adapter).__name__}.list_backups "
                        "does not accept deadline_monotonic; cannot fall back to unbounded call"
                    )
                return self.storage_adapter.list_backups()
            raise

    def _adapter_delete_backup(self, remote_filename: str, deadline_monotonic: Optional[float] = None) -> bool:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"RETENTION_DEADLINE_EXCEEDED: delete_backup {remote_filename}")
        try:
            return self.storage_adapter.delete_backup(remote_filename, deadline_monotonic=deadline_monotonic)
        except TypeError as te:
            if "deadline_monotonic" in str(te):
                if deadline_monotonic is not None:
                    raise TimeoutError(
                        f"RETENTION_DEADLINE_UNSUPPORTED: storage adapter {type(self.storage_adapter).__name__}.delete_backup "
                        "does not accept deadline_monotonic; cannot fall back to unbounded call"
                    )
                return self.storage_adapter.delete_backup(remote_filename)
            raise

    def _verify_single_recovery_point(
        self,
        item: Dict[str, Any],
        directory: str,
        network_attempts: List[int],
        deadline_monotonic: Optional[float],
        safety_margin_sec: float,
        max_network_attempts: Optional[int],
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Independently verifies a single recovery point against strict schema & checksum invariants.
        Fails closed on any corruption, mismatch, or unverified flag.
        """
        def _check_budget(action: str):
            if deadline_monotonic is not None and time.monotonic() >= (deadline_monotonic - safety_margin_sec):
                raise TimeoutError(f"RETENTION_DEADLINE_EXCEEDED: {action}")
            if max_network_attempts is not None and network_attempts[0] >= max_network_attempts:
                raise TimeoutError(f"MAX_NETWORK_ATTEMPTS_EXCEEDED: {action}")

        name = item["remote_filename"]
        backup_id = name[len("manifest_"):-len(".json")]

        _check_budget(f"download_manifest:{name}")
        manifest_path = os.path.join(directory, name)
        self._adapter_download_file(name, manifest_path, deadline_monotonic=deadline_monotonic)
        network_attempts[0] += 1

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("backup_id") != backup_id:
            raise ValueError("UNVERIFIED_BACKUP_MANIFEST")

        # Schema verification: distinguish modern vs legacy manifests
        if "upload_verified" in manifest:
            upload_verified = manifest.get("upload_verified")
            if upload_verified is not True:
                # Explicit False or non-boolean invalid value fails closed
                raise ValueError("UNVERIFIED_BACKUP_MANIFEST")
            dump_name = manifest.get("database_artifact") or manifest.get("dump_filename")
            expected_hash = manifest.get("artifact_sha256") or manifest.get("database_sha256")
        else:
            # Pre-SM-ACT-020 legacy schema contract:
            # 1. upload_verified key is absent (None).
            # 2. dump_filename or database_artifact must be explicitly specified and match smart_money_{backup_id}.dump.
            # 3. dump_sha256 must be present, 64-character valid hex.
            # 4. A legacy checksum alone DOES NOT imply verification; actual dump must be downloaded & checksummed.
            dump_name = manifest.get("dump_filename") or manifest.get("database_artifact")
            expected_hash = manifest.get("dump_sha256") or manifest.get("artifact_sha256") or manifest.get("database_sha256")
            if not dump_name or not expected_hash:
                raise ValueError("MALFORMED_LEGACY_BACKUP_MANIFEST")

        if dump_name != f"smart_money_{backup_id}.dump" or not isinstance(expected_hash, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_hash):
            raise ValueError("INVALID_BACKUP_DUMP_REFERENCE")

        _check_budget(f"download_dump:{dump_name}")
        dump_path = os.path.join(directory, dump_name)
        self._adapter_download_file(dump_name, dump_path, deadline_monotonic=deadline_monotonic)
        network_attempts[0] += 1

        # Check dump size if specified in legacy schema
        expected_size = manifest.get("dump_size_bytes")
        if isinstance(expected_size, int) and expected_size > 0:
            if os.path.getsize(dump_path) != expected_size:
                raise ValueError("BACKUP_DUMP_SIZE_MISMATCH")

        digest = hashlib.sha256()
        with open(dump_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected_hash.lower():
            raise ValueError("BACKUP_DUMP_CHECKSUM_MISMATCH")

        has_globals = bool(manifest.get("include_globals"))
        if has_globals:
            globals_name = manifest.get("globals_artifact")
            globals_hash = manifest.get("globals_artifact_sha256") or manifest.get("globals_sha256")
            if globals_name != f"globals_{backup_id}.sql" or not isinstance(globals_hash, str):
                raise ValueError("INVALID_BACKUP_GLOBALS_REFERENCE")
            _check_budget(f"download_globals:{globals_name}")
            globals_path = os.path.join(directory, globals_name)
            self._adapter_download_file(globals_name, globals_path, deadline_monotonic=deadline_monotonic)
            network_attempts[0] += 1
            if hashlib.sha256(Path(globals_path).read_bytes()).hexdigest().lower() != globals_hash.lower():
                raise ValueError("BACKUP_GLOBALS_CHECKSUM_MISMATCH")

        return backup_id, has_globals, manifest

    def prune_backups(
        self,
        retention_count: Optional[int] = None,
        retention_days: Optional[int] = None,
        deadline_monotonic: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
        max_verification_items: Optional[int] = None,
        max_network_attempts: Optional[int] = None,
        safety_margin_sec: float = 2.0,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Safely prunes expired backups beyond retention limits.
        Rule: The most recent verified backup is NEVER pruned, regardless of age.
        Bounds work by monotonic deadline and network attempt budgets.
        """
        max_count = retention_count if retention_count is not None else self.retention_count
        max_days = retention_days if retention_days is not None else self.retention_days

        effective_deadline = deadline_monotonic
        if effective_deadline is None and max_duration_sec is not None:
            effective_deadline = time.monotonic() + max_duration_sec

        # Preflight deadline budget
        if effective_deadline is not None and time.monotonic() >= (effective_deadline - safety_margin_sec):
            logger.warning("Pruning deferred: monotonic cycle deadline exceeded before starting.")
            self.last_outcome = RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
            return []

        try:
            all_items = self._adapter_list_backups(deadline_monotonic=effective_deadline)
        except TimeoutError as timeout_err:
            logger.warning("Pruning deferred: %s during listing. 0 files deleted.", timeout_err)
            self.last_outcome = RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
            return []
        except Exception as exc:
            logger.error("Pruning disabled: error during listing backups (%s)", type(exc).__name__)
            self.last_outcome = RetentionOutcome.DISABLED_UNVERIFIED_RECOVERY_POINT
            return []
        network_attempts = [1]
        if not all_items:
            self.last_outcome = RetentionOutcome.NO_ACTION_REQUIRED
            return []

        manifests = [
            item for item in all_items
            if re.fullmatch(r"manifest_sm-backup-[A-Za-z0-9_-]+\.json", item["remote_filename"])
        ]
        manifests.sort(key=lambda x: x.get("modified_at", ""), reverse=True)
        if len(manifests) <= 1:
            self.last_outcome = RetentionOutcome.NO_ACTION_REQUIRED
            return []

        # Preflight *every* timestamp before downloading or verifying any files.
        # A bad entry anywhere must fail closed immediately.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff_date = now - timedelta(days=max_days)
        for item in manifests[1:]:
            try:
                mod_date = datetime.fromisoformat(item["modified_at"].replace("Z", "+00:00"))
            except (KeyError, AttributeError, TypeError, ValueError):
                logger.warning("Pruning disabled: unparseable backup timestamp")
                self.last_outcome = RetentionOutcome.DISABLED_UNPARSEABLE_TIMESTAMP
                return []

        verified = []
        try:
            with tempfile.TemporaryDirectory(prefix="sm_retention_verify_") as directory:
                for item in manifests:
                    if max_verification_items is not None and len(verified) >= max_verification_items:
                        logger.warning(
                            "Pruning deferred: max verification items limit (%d) reached.",
                            max_verification_items,
                        )
                        self.last_outcome = RetentionOutcome.DEFERRED_ATTEMPTS_EXCEEDED
                        return []

                    backup_id, has_globals, _ = self._verify_single_recovery_point(
                        item=item,
                        directory=directory,
                        network_attempts=network_attempts,
                        deadline_monotonic=effective_deadline,
                        safety_margin_sec=safety_margin_sec,
                        max_network_attempts=max_network_attempts,
                    )
                    verified.append((item, backup_id, has_globals))
        except TimeoutError as timeout_err:
            logger.warning("Pruning deferred: %s. Safe verification incomplete; 0 files deleted.", timeout_err)
            self.last_outcome = RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
            return []
        except Exception as exc:
            logger.error("Pruning disabled: a recovery point cannot be verified (%s)", type(exc).__name__)
            self.last_outcome = RetentionOutcome.DISABLED_UNVERIFIED_RECOVERY_POINT
            return []

        if len(verified) < len(manifests):
            self.last_outcome = RetentionOutcome.DEFERRED_ATTEMPTS_EXCEEDED
            return []

        deletion_candidates = []
        for idx, (item, backup_id, has_globals) in enumerate(verified[1:], start=1):
            mod_date = datetime.fromisoformat(item["modified_at"].replace("Z", "+00:00"))
            expired = mod_date.replace(tzinfo=None) < cutoff_date
            if idx >= max_count or expired:
                deletion_candidates.append((backup_id, has_globals))

        if not deletion_candidates:
            self.last_outcome = RetentionOutcome.NO_ACTION_REQUIRED
            return []

        if dry_run:
            self.last_outcome = RetentionOutcome.SUCCESS
            self.last_details = {
                "protected_newest": verified[0][1],
                "verified_count": len(verified),
                "candidates_to_prune": [b_id for b_id, _ in deletion_candidates],
            }
            return [b_id for b_id, _ in deletion_candidates]

        attempted_candidates: List[str] = []
        fully_pruned_candidates: List[str] = []
        partial_candidates: List[str] = []
        completed_file_deletions: List[str] = []
        failed_file_deletions: List[str] = []
        interrupted_by_deadline: bool = False

        for backup_id, has_globals in deletion_candidates:
            if effective_deadline is not None and time.monotonic() >= (effective_deadline - safety_margin_sec):
                interrupted_by_deadline = True
                logger.warning(
                    "Pruning interrupted: deadline reached before processing candidate %s. "
                    "Safely pruned %d candidates.",
                    backup_id,
                    len(fully_pruned_candidates),
                )
                break

            attempted_candidates.append(backup_id)
            candidate_files = [f"manifest_{backup_id}.json", f"smart_money_{backup_id}.dump"]
            if has_globals:
                candidate_files.append(f"globals_{backup_id}.sql")

            candidate_ok = True
            for f_name in candidate_files:
                if effective_deadline is not None and time.monotonic() >= (effective_deadline - safety_margin_sec):
                    interrupted_by_deadline = True
                    candidate_ok = False
                    failed_file_deletions.append(f_name)
                    logger.warning(
                        "Pruning interrupted: deadline reached during deletion of %s for candidate %s.",
                        f_name,
                        backup_id,
                    )
                    break

                try:
                    deleted = self._adapter_delete_backup(f_name, deadline_monotonic=effective_deadline)
                    if deleted:
                        completed_file_deletions.append(f_name)
                    else:
                        candidate_ok = False
                        failed_file_deletions.append(f_name)
                        logger.error("Storage adapter reported failure deleting %s", f_name)
                except Exception as del_err:
                    candidate_ok = False
                    failed_file_deletions.append(f_name)
                    logger.error("Exception deleting %s: %s", f_name, del_err)

            if candidate_ok:
                fully_pruned_candidates.append(backup_id)
            else:
                partial_candidates.append(backup_id)

            if interrupted_by_deadline:
                break

        # Explicit deletion accounting: never claim SUCCESS on partial/interrupted deletions
        if interrupted_by_deadline:
            if fully_pruned_candidates or completed_file_deletions:
                self.last_outcome = RetentionOutcome.PARTIAL_DEFERRED
            else:
                self.last_outcome = RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
        elif failed_file_deletions or partial_candidates:
            self.last_outcome = RetentionOutcome.PARTIAL_DEFERRED
        elif len(fully_pruned_candidates) == len(deletion_candidates):
            self.last_outcome = RetentionOutcome.SUCCESS
        else:
            self.last_outcome = RetentionOutcome.PARTIAL_DEFERRED

        remaining_candidates = [b_id for b_id, _ in deletion_candidates if b_id not in fully_pruned_candidates]
        self.last_details = {
            "total_candidates": len(deletion_candidates),
            "attempted_candidates": attempted_candidates,
            "fully_pruned_candidates": fully_pruned_candidates,
            "partial_candidates": partial_candidates,
            "remaining_candidates": remaining_candidates,
            "completed_file_deletions": completed_file_deletions,
            "failed_file_deletions": failed_file_deletions,
            "interrupted_by_deadline": interrupted_by_deadline,
            "protected_newest": verified[0][1],
        }

        return fully_pruned_candidates
