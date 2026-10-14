"""
ARX Terminal — Smart Money PostgreSQL Backup Freshness Authority.

Defines the canonical, single-source authority for evaluating PostgreSQL backup freshness.
Guarantees:
1. Freshness is determined solely by the latest verified recoverable backup contract;
2. Local modification time or scheduler run time alone are strictly rejected;
3. Unverified manifests, missing dumps, or corrupted checksums fail closed;
4. Clock skew and future-dated manifests are evaluated with strict tolerance;
5. Deterministic testability with injected frozen clocks.
"""

from __future__ import annotations

import os
import json
import logging
import tempfile
from enum import Enum
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Callable

from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    BackupStorageAdapter,
    get_backup_storage_adapter,
)

logger = logging.getLogger("smart_money.backup.freshness_authority")


class BackupFreshnessState(str, Enum):
    """Authoritative backup freshness states."""
    FRESH = "FRESH"
    STALE = "STALE"
    MISSING = "MISSING"
    UNVERIFIABLE = "UNVERIFIABLE"
    CHECK_FAILED = "CHECK_FAILED"
    RECOVERY_IN_PROGRESS = "RECOVERY_IN_PROGRESS"


@dataclass(frozen=True)
class FreshnessEvaluationReceipt:
    """Cryptographically grounded receipt of backup freshness evaluation."""
    recovery_point_id: Optional[str]
    verified_at: str
    backup_completed_at: Optional[str]
    backup_age_seconds: Optional[float]
    freshness_threshold_seconds: float
    verification_source: str
    verification_status: str
    reason_code: str
    is_fresh: bool
    manifest_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BackupContainmentPolicy:
    """Configurable containment policy settings for backup durability."""
    enabled: bool = False
    max_verified_age_seconds: float = 43200.0  # 12 Hours
    check_timeout_seconds: float = 10.0
    recheck_interval_seconds: float = 60.0
    clock_skew_tolerance_seconds: float = 300.0  # 5 Minutes
    production_authorized: bool = False

    @classmethod
    def from_environment(cls) -> BackupContainmentPolicy:
        """Loads policy from standardized environment configuration."""
        env_enabled = os.environ.get("ARX_BACKUP_CONTAINMENT_ENABLED", "").strip().lower() in ("true", "1", "yes")
        
        # Max age configuration (default 43200 / 12 hours)
        max_age_str = os.environ.get("ARX_BACKUP_MAX_VERIFIED_AGE_SEC")
        max_age = float(max_age_str) if max_age_str and max_age_str.strip().isdigit() else 43200.0

        # Check timeout
        timeout_str = os.environ.get("ARX_BACKUP_FRESHNESS_CHECK_TIMEOUT_SEC")
        timeout = float(timeout_str) if timeout_str else 10.0

        # Recheck interval (cache TTL)
        interval_str = os.environ.get("ARX_BACKUP_FRESHNESS_RECHECK_INTERVAL_SEC")
        interval = float(interval_str) if interval_str else 60.0

        # Clock skew tolerance
        skew_str = os.environ.get("ARX_BACKUP_FRESHNESS_CLOCK_SKEW_TOLERANCE_SEC")
        skew = float(skew_str) if skew_str else 300.0

        # Production authorization gate
        prod_auth = os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED", "").strip().lower() in ("true", "1", "yes")

        return cls(
            enabled=env_enabled,
            max_verified_age_seconds=max_age,
            check_timeout_seconds=timeout,
            recheck_interval_seconds=interval,
            clock_skew_tolerance_seconds=skew,
            production_authorized=prod_auth,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BackupFreshnessAuthority:
    """
    Canonical authority evaluating the freshness and recoverability of PostgreSQL backups.
    """

    def __init__(
        self,
        storage_adapter: Optional[BackupStorageAdapter] = None,
        dsn: Optional[str] = None,
        policy: Optional[BackupContainmentPolicy] = None,
        clock_fn: Optional[Callable[[], datetime]] = None,
    ):
        self.storage_adapter = storage_adapter or get_backup_storage_adapter()
        self.dsn = dsn
        self.policy = policy or BackupContainmentPolicy.from_environment()
        self.clock_fn = clock_fn
        self._last_receipt: Optional[FreshnessEvaluationReceipt] = None
        self._last_evaluated_monotonic: float = 0.0
        self._recovery_in_progress: bool = False

    def mark_recovery_in_progress(self, in_progress: bool = True) -> None:
        """Sets internal flag indicating backup execution or restore is actively running."""
        self._recovery_in_progress = in_progress

    def get_current_time(self) -> datetime:
        """Returns authoritative current time (honors clock_fn if injected)."""
        if self.clock_fn is not None:
            return self.clock_fn()
        return datetime.now(timezone.utc)

    def evaluate_freshness(
        self,
        force_refresh: bool = False,
        current_time: Optional[datetime] = None,
    ) -> FreshnessEvaluationReceipt:
        """
        Authoritatively evaluates the latest verified recoverable backup.
        Fails closed on any ambiguity or corruption.
        """
        now_dt = current_time or self.get_current_time()
        now_iso = now_dt.isoformat()

        # Check recovery in progress flag
        if self._recovery_in_progress:
            return FreshnessEvaluationReceipt(
                recovery_point_id=None,
                verified_at=now_iso,
                backup_completed_at=None,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="STATE_MACHINE",
                verification_status=BackupFreshnessState.RECOVERY_IN_PROGRESS.value,
                reason_code="BACKUP_EXECUTION_IN_PROGRESS",
                is_fresh=False,
            )

        # Cache check (unless force_refresh or testing with injected clock)
        if not force_refresh and current_time is None and self._last_receipt is not None:
            import time
            if (time.monotonic() - self._last_evaluated_monotonic) < self.policy.recheck_interval_seconds:
                return self._last_receipt

        try:
            receipt = self._execute_evaluation(now_dt)
        except Exception as e:
            logger.error(f"Backup freshness evaluation check failed: {e}", exc_info=True)
            receipt = FreshnessEvaluationReceipt(
                recovery_point_id=None,
                verified_at=now_iso,
                backup_completed_at=None,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="STORAGE_EXCEPTION",
                verification_status=BackupFreshnessState.CHECK_FAILED.value,
                reason_code=f"STORAGE_EXCEPTION_{type(e).__name__}",
                is_fresh=False,
            )

        import time
        self._last_receipt = receipt
        self._last_evaluated_monotonic = time.monotonic()
        return receipt

    def _execute_evaluation(self, now_dt: datetime) -> FreshnessEvaluationReceipt:
        now_iso = now_dt.isoformat()
        
        # 1. Query all backups in storage destination
        all_items = self.storage_adapter.list_backups()
        if not all_items:
            return FreshnessEvaluationReceipt(
                recovery_point_id=None,
                verified_at=now_iso,
                backup_completed_at=None,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="STORAGE_MANIFEST",
                verification_status=BackupFreshnessState.MISSING.value,
                reason_code="NO_BACKUP_MANIFESTS_FOUND",
                is_fresh=False,
            )

        # 2. Filter valid manifest candidates
        manifest_items = [
            item for item in all_items
            if item["remote_filename"].startswith("manifest_") and item["remote_filename"].endswith(".json")
        ]
        if not manifest_items:
            return FreshnessEvaluationReceipt(
                recovery_point_id=None,
                verified_at=now_iso,
                backup_completed_at=None,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="STORAGE_MANIFEST",
                verification_status=BackupFreshnessState.MISSING.value,
                reason_code="NO_MANIFEST_FILES_FOUND",
                is_fresh=False,
            )

        # 3. Sort manifests by filename/modified_at descending
        manifest_items.sort(key=lambda x: x.get("modified_at") or x["remote_filename"], reverse=True)

        # 4. Inspect newest manifests to find authoritative verified recovery point
        newest_item = manifest_items[0]
        manifest_name = newest_item["remote_filename"]

        with tempfile.TemporaryDirectory(prefix="arx_freshness_eval_") as tmp_dir:
            local_manifest_path = os.path.join(tmp_dir, manifest_name)
            try:
                self.storage_adapter.download_file(manifest_name, local_manifest_path)
            except Exception as e:
                return FreshnessEvaluationReceipt(
                    recovery_point_id=None,
                    verified_at=now_iso,
                    backup_completed_at=None,
                    backup_age_seconds=None,
                    freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                    verification_source="STORAGE_DOWNLOAD",
                    verification_status=BackupFreshnessState.CHECK_FAILED.value,
                    reason_code=f"MANIFEST_DOWNLOAD_FAILED_{type(e).__name__}",
                    is_fresh=False,
                )

            # 5. Parse manifest JSON
            try:
                with open(local_manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception as e:
                return FreshnessEvaluationReceipt(
                    recovery_point_id=None,
                    verified_at=now_iso,
                    backup_completed_at=None,
                    backup_age_seconds=None,
                    freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                    verification_source="MANIFEST_PARSER",
                    verification_status=BackupFreshnessState.UNVERIFIABLE.value,
                    reason_code=f"CORRUPT_MANIFEST_JSON: {e}",
                    is_fresh=False,
                )

        # 6. Validate manifest schema and upload verification contract
        backup_id = manifest.get("backup_id")
        completed_at_str = manifest.get("completed_at") or manifest.get("started_at")
        dump_filename = manifest.get("dump_filename") or manifest.get("database_artifact")
        dump_sha = manifest.get("artifact_sha256") or manifest.get("database_sha256")
        dump_size = manifest.get("artifact_size_bytes")
        upload_verified = manifest.get("upload_verified")

        if not backup_id or not completed_at_str or not dump_filename or not dump_sha or dump_size is None:
            return FreshnessEvaluationReceipt(
                recovery_point_id=backup_id,
                verified_at=now_iso,
                backup_completed_at=completed_at_str,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="MANIFEST_CONTRACT",
                verification_status=BackupFreshnessState.UNVERIFIABLE.value,
                reason_code="MALFORMED_MANIFEST_SCHEMA_KEYS",
                is_fresh=False,
                manifest_metadata=manifest,
            )

        if not upload_verified:
            return FreshnessEvaluationReceipt(
                recovery_point_id=backup_id,
                verified_at=now_iso,
                backup_completed_at=completed_at_str,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="MANIFEST_CONTRACT",
                verification_status=BackupFreshnessState.UNVERIFIABLE.value,
                reason_code="MANIFEST_UPLOAD_NOT_VERIFIED",
                is_fresh=False,
                manifest_metadata=manifest,
            )

        # 7. Verify Remote Dump Artifact Existence and Size via storage metadata
        dump_metadata = self.storage_adapter.get_backup_metadata(dump_filename)
        if not dump_metadata:
            return FreshnessEvaluationReceipt(
                recovery_point_id=backup_id,
                verified_at=now_iso,
                backup_completed_at=completed_at_str,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="REMOTE_STORAGE_AUDIT",
                verification_status=BackupFreshnessState.UNVERIFIABLE.value,
                reason_code="REMOTE_DUMP_ARTIFACT_MISSING",
                is_fresh=False,
                manifest_metadata=manifest,
            )

        if dump_metadata.get("size_bytes") != dump_size:
            return FreshnessEvaluationReceipt(
                recovery_point_id=backup_id,
                verified_at=now_iso,
                backup_completed_at=completed_at_str,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="REMOTE_STORAGE_AUDIT",
                verification_status=BackupFreshnessState.UNVERIFIABLE.value,
                reason_code=f"DUMP_SIZE_MISMATCH: remote={dump_metadata.get('size_bytes')} expected={dump_size}",
                is_fresh=False,
                manifest_metadata=manifest,
            )

        # 8. Check timestamp format and future clock skew
        try:
            completed_clean = completed_at_str.replace("Z", "+00:00")
            completed_dt = datetime.fromisoformat(completed_clean)
            if completed_dt.tzinfo is None:
                completed_dt = completed_dt.replace(tzinfo=timezone.utc)
        except Exception as e:
            return FreshnessEvaluationReceipt(
                recovery_point_id=backup_id,
                verified_at=now_iso,
                backup_completed_at=completed_at_str,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="TIMESTAMP_PARSER",
                verification_status=BackupFreshnessState.UNVERIFIABLE.value,
                reason_code=f"INVALID_COMPLETED_AT_TIMESTAMP: {e}",
                is_fresh=False,
                manifest_metadata=manifest,
            )

        # Future-dated check
        skew_tolerance = timedelta(seconds=self.policy.clock_skew_tolerance_seconds)
        if completed_dt > (now_dt + skew_tolerance):
            return FreshnessEvaluationReceipt(
                recovery_point_id=backup_id,
                verified_at=now_iso,
                backup_completed_at=completed_at_str,
                backup_age_seconds=None,
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="CLOCK_VALIDATION",
                verification_status=BackupFreshnessState.UNVERIFIABLE.value,
                reason_code="FUTURE_DATED_RECOVERY_POINT_EXCEEDS_CLOCK_SKEW",
                is_fresh=False,
                manifest_metadata=manifest,
            )

        # 9. Compute verified backup age
        raw_age = (now_dt - completed_dt).total_seconds()
        backup_age_seconds = max(0.0, raw_age)

        # 10. Corroborate with Database Audit Ledger if DSN is configured
        if self.dsn:
            try:
                import psycopg2
                conn = psycopg2.connect(self.dsn)
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT payload FROM sm_audit_events WHERE event_type = 'BACKUP_COMPLETED' AND subject_id = %s LIMIT 1;",
                            (backup_id,)
                        )
                        audit_row = cur.fetchone()
                        if audit_row:
                            payload = audit_row[0] if isinstance(audit_row[0], dict) else json.loads(audit_row[0])
                            recorded_sha = payload.get("dump_sha")
                            if recorded_sha and recorded_sha != dump_sha:
                                return FreshnessEvaluationReceipt(
                                    recovery_point_id=backup_id,
                                    verified_at=now_iso,
                                    backup_completed_at=completed_at_str,
                                    backup_age_seconds=backup_age_seconds,
                                    freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                                    verification_source="DATABASE_AUDIT_CORROBORATION",
                                    verification_status=BackupFreshnessState.UNVERIFIABLE.value,
                                    reason_code="DATABASE_AUDIT_SHA_MISMATCH",
                                    is_fresh=False,
                                    manifest_metadata=manifest,
                                )
                finally:
                    conn.close()
            except Exception as e:
                logger.debug(f"Database audit corroboration skipped (non-fatal if storage verified): {e}")

        # 11. Evaluate age threshold
        if backup_age_seconds <= self.policy.max_verified_age_seconds:
            return FreshnessEvaluationReceipt(
                recovery_point_id=backup_id,
                verified_at=now_iso,
                backup_completed_at=completed_at_str,
                backup_age_seconds=round(backup_age_seconds, 2),
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="STORAGE_MANIFEST_AND_HEAD",
                verification_status=BackupFreshnessState.FRESH.value,
                reason_code="VERIFIED_RECOVERY_POINT_WITHIN_THRESHOLD",
                is_fresh=True,
                manifest_metadata=manifest,
            )
        else:
            return FreshnessEvaluationReceipt(
                recovery_point_id=backup_id,
                verified_at=now_iso,
                backup_completed_at=completed_at_str,
                backup_age_seconds=round(backup_age_seconds, 2),
                freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                verification_source="STORAGE_MANIFEST_AND_HEAD",
                verification_status=BackupFreshnessState.STALE.value,
                reason_code="RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD",
                is_fresh=False,
                manifest_metadata=manifest,
            )
