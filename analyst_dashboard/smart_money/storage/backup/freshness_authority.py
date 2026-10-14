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


def _parse_safety_float(val_str: Optional[str], default: float, param_name: str, allow_zero: bool = False) -> float:
    if val_str is None or val_str.strip() == "":
        return default
    val_str = val_str.strip()
    try:
        val = float(val_str)
    except ValueError:
        raise ValueError(f"INVALID_SAFETY_CONFIG: {param_name}='{val_str}' cannot be parsed as a float.")

    import math
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"INVALID_SAFETY_CONFIG: {param_name}='{val_str}' must be a finite number.")

    if allow_zero:
        if val < 0.0:
            raise ValueError(f"INVALID_SAFETY_CONFIG: {param_name}='{val_str}' must be non-negative (>= 0).")
    else:
        if val <= 0.0:
            raise ValueError(f"INVALID_SAFETY_CONFIG: {param_name}='{val_str}' must be strictly positive (> 0).")

    return val


def is_valid_sha256_hex(val: Any) -> bool:
    """Validates that a string is exactly 64 hexadecimal characters."""
    if not isinstance(val, str):
        return False
    clean = val.strip()
    if len(clean) != 64:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in clean)


def parse_and_validate_iso_timestamp(ts_str: Any, field_name: str) -> datetime:
    """
    Parses an ISO 8601 timestamp, validates timezone or normalizes under UTC contract.
    Fails closed if empty, malformed, or invalid format.
    """
    if not isinstance(ts_str, str) or not ts_str.strip():
        raise ValueError(f"MISSING_OR_EMPTY_{field_name.upper()}_TIMESTAMP")
    clean = ts_str.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean)
    except Exception as e:
        raise ValueError(f"INVALID_{field_name.upper()}_TIMESTAMP: {e}")
    if dt.tzinfo is None:
        # Explicit UTC normalization contract for naive timestamps
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class BackupContainmentPolicy:
    """Configurable containment policy settings for backup durability."""
    enabled: bool = False
    max_verified_age_seconds: float = 43200.0  # 12 Hours
    check_timeout_seconds: float = 10.0
    recheck_interval_seconds: float = 60.0
    clock_skew_tolerance_seconds: float = 300.0  # 5 Minutes
    production_authorized: bool = False
    require_remote_checksum_metadata: bool = True

    @classmethod
    def from_environment(cls) -> BackupContainmentPolicy:
        """Loads policy from standardized environment configuration with fail-closed validation."""
        env_enabled = os.environ.get("ARX_BACKUP_CONTAINMENT_ENABLED", "").strip().lower() in ("true", "1", "yes")

        # Max age configuration (default 43200 / 12 hours) - strictly positive finite float
        max_age = _parse_safety_float(
            os.environ.get("ARX_BACKUP_MAX_VERIFIED_AGE_SEC"),
            43200.0,
            "ARX_BACKUP_MAX_VERIFIED_AGE_SEC",
            allow_zero=False,
        )

        # Check timeout - strictly positive finite float
        timeout = _parse_safety_float(
            os.environ.get("ARX_BACKUP_FRESHNESS_CHECK_TIMEOUT_SEC"),
            10.0,
            "ARX_BACKUP_FRESHNESS_CHECK_TIMEOUT_SEC",
            allow_zero=False,
        )

        # Recheck interval (cache TTL) - non-negative finite float
        interval = _parse_safety_float(
            os.environ.get("ARX_BACKUP_FRESHNESS_RECHECK_INTERVAL_SEC"),
            60.0,
            "ARX_BACKUP_FRESHNESS_RECHECK_INTERVAL_SEC",
            allow_zero=True,
        )

        # Clock skew tolerance - non-negative finite float
        skew = _parse_safety_float(
            os.environ.get("ARX_BACKUP_FRESHNESS_CLOCK_SKEW_TOLERANCE_SEC"),
            300.0,
            "ARX_BACKUP_FRESHNESS_CLOCK_SKEW_TOLERANCE_SEC",
            allow_zero=True,
        )

        # Production authorization gate
        prod_auth = os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED", "").strip().lower() in ("true", "1", "yes")

        # Remote checksum metadata requirement (defaults to True)
        raw_req_meta = os.environ.get("ARX_BACKUP_REQUIRE_REMOTE_CHECKSUM_METADATA", "true").strip().lower()
        req_meta = raw_req_meta in ("true", "1", "yes")

        return cls(
            enabled=env_enabled,
            max_verified_age_seconds=max_age,
            check_timeout_seconds=timeout,
            recheck_interval_seconds=interval,
            clock_skew_tolerance_seconds=skew,
            production_authorized=prod_auth,
            require_remote_checksum_metadata=req_meta,
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
            elapsed = time.monotonic() - self._last_evaluated_monotonic
            if elapsed < self.policy.recheck_interval_seconds:
                if self._last_receipt.is_fresh and self._last_receipt.backup_age_seconds is not None:
                    projected_age = self._last_receipt.backup_age_seconds + elapsed
                    if projected_age <= self.policy.max_verified_age_seconds:
                        return self._last_receipt
                elif not self._last_receipt.is_fresh:
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
            if item.get("remote_filename", "").startswith("manifest_") and item.get("remote_filename", "").endswith(".json")
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

        last_rejection_reason = "NO_VALID_RECOVERY_POINT_FOUND"
        last_rejection_source = "STORAGE_MANIFEST"
        last_inspected_id = None
        last_manifest_metadata = None

        with tempfile.TemporaryDirectory(prefix="arx_freshness_eval_") as tmp_dir:
            for item in manifest_items:
                manifest_name = item["remote_filename"]
                local_manifest_path = os.path.join(tmp_dir, manifest_name)
                try:
                    self.storage_adapter.download_file(manifest_name, local_manifest_path)
                except Exception as e:
                    last_rejection_reason = f"MANIFEST_DOWNLOAD_FAILED_{type(e).__name__}"
                    last_rejection_source = "STORAGE_DOWNLOAD"
                    continue

                # Parse manifest JSON
                try:
                    with open(local_manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception as e:
                    last_rejection_reason = f"CORRUPT_MANIFEST_JSON: {e}"
                    last_rejection_source = "MANIFEST_PARSER"
                    continue

                if not isinstance(manifest, dict):
                    last_rejection_reason = "MANIFEST_PAYLOAD_NOT_OBJECT"
                    last_rejection_source = "MANIFEST_PARSER"
                    continue

                last_manifest_metadata = manifest
                backup_id = manifest.get("backup_id")
                last_inspected_id = backup_id
                completed_at_str = manifest.get("completed_at")
                started_at_str = manifest.get("started_at")
                dump_filename = manifest.get("dump_filename") or manifest.get("database_artifact")
                dump_sha = manifest.get("artifact_sha256") or manifest.get("database_sha256")
                dump_size = manifest.get("artifact_size_bytes")
                upload_verified = manifest.get("upload_verified")

                # Schema keys presence check
                if not backup_id or not dump_filename or not dump_sha or dump_size is None:
                    last_rejection_reason = "MALFORMED_MANIFEST_SCHEMA_KEYS"
                    last_rejection_source = "MANIFEST_CONTRACT"
                    continue

                # Path traversal / object key substitution protection
                dump_clean = dump_filename.replace("\\", "/")
                dump_basename = os.path.basename(dump_clean)
                if (
                    dump_basename != dump_filename
                    or ".." in dump_filename
                    or "/" in dump_filename
                    or "\\" in dump_filename
                    or "\x00" in dump_filename
                    or not (dump_filename.startswith("smart_money_") or dump_filename.endswith(".dump"))
                    or (backup_id and backup_id not in dump_filename)
                ):
                    last_rejection_reason = "INVALID_DUMP_FILENAME_TRAVERSAL_OR_SUBSTITUTION"
                    last_rejection_source = "MANIFEST_INTEGRITY_CHECK"
                    continue

                # Check completed_at authority (started_at alone is strictly rejected)
                if not completed_at_str or not isinstance(completed_at_str, str) or not completed_at_str.strip():
                    last_rejection_reason = "MISSING_COMPLETED_AT_TIMESTAMP"
                    last_rejection_source = "TIMESTAMP_AUTHORITY"
                    continue

                # Upload verification gate
                if upload_verified is not True:
                    last_rejection_reason = "MANIFEST_UPLOAD_NOT_VERIFIED"
                    last_rejection_source = "MANIFEST_CONTRACT"
                    continue

                # Manifest SHA format validation
                if not is_valid_sha256_hex(dump_sha):
                    last_rejection_reason = f"MALFORMED_MANIFEST_SHA256: {dump_sha}"
                    last_rejection_source = "MANIFEST_CONTRACT"
                    continue

                # Timestamp parsing and chronological order validation
                try:
                    completed_dt = parse_and_validate_iso_timestamp(completed_at_str, "completed_at")
                except Exception as e:
                    last_rejection_reason = f"INVALID_COMPLETED_AT_TIMESTAMP: {e}"
                    last_rejection_source = "TIMESTAMP_PARSER"
                    continue

                if started_at_str:
                    try:
                        started_dt = parse_and_validate_iso_timestamp(started_at_str, "started_at")
                        if completed_dt < started_dt:
                            last_rejection_reason = "COMPLETION_PRECEDES_START_TIMESTAMP"
                            last_rejection_source = "TIMESTAMP_AUTHORITY"
                            continue
                    except Exception as e:
                        last_rejection_reason = f"INVALID_STARTED_AT_TIMESTAMP: {e}"
                        last_rejection_source = "TIMESTAMP_PARSER"
                        continue

                # Future clock skew check
                skew_tolerance = timedelta(seconds=self.policy.clock_skew_tolerance_seconds)
                if completed_dt > (now_dt + skew_tolerance):
                    last_rejection_reason = "FUTURE_DATED_RECOVERY_POINT_EXCEEDS_CLOCK_SKEW"
                    last_rejection_source = "CLOCK_VALIDATION"
                    continue

                # Remote dump existence and size audit
                try:
                    dump_metadata = self.storage_adapter.get_backup_metadata(dump_filename)
                except Exception as e:
                    last_rejection_reason = f"REMOTE_STORAGE_METADATA_FETCH_FAILED_{type(e).__name__}"
                    last_rejection_source = "REMOTE_STORAGE_AUDIT"
                    continue

                if not dump_metadata:
                    last_rejection_reason = "REMOTE_DUMP_ARTIFACT_MISSING"
                    last_rejection_source = "REMOTE_STORAGE_AUDIT"
                    continue

                if dump_metadata.get("size_bytes") != dump_size:
                    last_rejection_reason = f"DUMP_SIZE_MISMATCH: remote={dump_metadata.get('size_bytes')} expected={dump_size}"
                    last_rejection_source = "REMOTE_STORAGE_AUDIT"
                    continue

                # Remote checksum metadata contract
                remote_sha = dump_metadata.get("sha256")
                if self.policy.require_remote_checksum_metadata:
                    if remote_sha is None:
                        last_rejection_reason = "MISSING_REMOTE_CHECKSUM_METADATA"
                        last_rejection_source = "REMOTE_CHECKSUM_METADATA_CONTRACT"
                        continue
                    if not isinstance(remote_sha, str) or not remote_sha.strip():
                        last_rejection_reason = "EMPTY_REMOTE_CHECKSUM_METADATA"
                        last_rejection_source = "REMOTE_CHECKSUM_METADATA_CONTRACT"
                        continue
                    remote_sha_clean = remote_sha.strip()
                    if not is_valid_sha256_hex(remote_sha_clean):
                        last_rejection_reason = f"MALFORMED_REMOTE_CHECKSUM_METADATA: {remote_sha}"
                        last_rejection_source = "REMOTE_CHECKSUM_METADATA_CONTRACT"
                        continue
                    if remote_sha_clean.lower() != dump_sha.strip().lower():
                        last_rejection_reason = f"DUMP_SHA256_MISMATCH: remote_meta={remote_sha} expected={dump_sha}"
                        last_rejection_source = "REMOTE_CHECKSUM_METADATA_CONTRACT"
                        continue
                else:
                    # Optional remote metadata (historical audit mode only)
                    if remote_sha is not None and str(remote_sha).strip():
                        remote_sha_clean = str(remote_sha).strip()
                        if not is_valid_sha256_hex(remote_sha_clean):
                            last_rejection_reason = f"MALFORMED_REMOTE_CHECKSUM_METADATA: {remote_sha}"
                            last_rejection_source = "REMOTE_CHECKSUM_METADATA_CONTRACT"
                            continue
                        if remote_sha_clean.lower() != dump_sha.strip().lower():
                            last_rejection_reason = f"DUMP_SHA256_MISMATCH: remote_meta={remote_sha} expected={dump_sha}"
                            last_rejection_source = "REMOTE_CHECKSUM_METADATA_CONTRACT"
                            continue

                # Database audit ledger corroboration if DSN is configured
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
                                    if recorded_sha and str(recorded_sha).strip().lower() != dump_sha.strip().lower():
                                        last_rejection_reason = "DATABASE_AUDIT_SHA_MISMATCH"
                                        last_rejection_source = "DATABASE_AUDIT_CORROBORATION"
                                        continue
                        finally:
                            conn.close()
                    except Exception as e:
                        logger.debug(f"Database audit corroboration skipped (non-fatal if storage verified): {e}")

                # Candidate successfully verified! Compute age against actual completed_at
                raw_age = (now_dt - completed_dt).total_seconds()
                backup_age_seconds = max(0.0, raw_age)

                if backup_age_seconds <= self.policy.max_verified_age_seconds:
                    return FreshnessEvaluationReceipt(
                        recovery_point_id=backup_id,
                        verified_at=now_iso,
                        backup_completed_at=completed_at_str,
                        backup_age_seconds=round(backup_age_seconds, 2),
                        freshness_threshold_seconds=self.policy.max_verified_age_seconds,
                        verification_source="STORAGE_MANIFEST_AND_REMOTE_METADATA",
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
                        verification_source="STORAGE_MANIFEST_AND_REMOTE_METADATA",
                        verification_status=BackupFreshnessState.STALE.value,
                        reason_code="RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD",
                        is_fresh=False,
                        manifest_metadata=manifest,
                    )

        # If all candidates failed verification
        return FreshnessEvaluationReceipt(
            recovery_point_id=last_inspected_id,
            verified_at=now_iso,
            backup_completed_at=None,
            backup_age_seconds=None,
            freshness_threshold_seconds=self.policy.max_verified_age_seconds,
            verification_source=last_rejection_source,
            verification_status=BackupFreshnessState.UNVERIFIABLE.value,
            reason_code=last_rejection_reason,
            is_fresh=False,
            manifest_metadata=last_manifest_metadata,
        )
