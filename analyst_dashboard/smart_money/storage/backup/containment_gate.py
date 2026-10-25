"""
ARX Terminal — Smart Money Backup Age Containment Gate & State Machine.

Enforces deterministic fail-closed protection for SEC Form 4 ingestion and snapshot publication:
1. Ingestion is halted immediately when backup age exceeds maximum threshold;
2. Snapshot publication is blocked if state depends on unverified or stale recovery points;
3. State machine governs clean transitions:
   HEALTHY -> FRESHNESS_WARNING -> INGESTION_SUSPENDED -> BACKUP_RECOVERY_PENDING -> RECOVERY_POINT_VERIFIED -> RESUMPTION_ELIGIBLE -> HEALTHY;
4. Emits structured alerts and telemetry envelopes for observability.
"""

from __future__ import annotations

import os
import uuid
import logging
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupFreshnessAuthority,
    BackupFreshnessState,
    BackupContainmentPolicy,
    FreshnessEvaluationReceipt,
)

logger = logging.getLogger("smart_money.backup.containment_gate")


class ContainmentState(str, Enum):
    """Lifecycle states of the backup age containment state machine."""
    HEALTHY = "HEALTHY"
    FRESHNESS_WARNING = "FRESHNESS_WARNING"
    INGESTION_SUSPENDED = "INGESTION_SUSPENDED"
    BACKUP_RECOVERY_PENDING = "BACKUP_RECOVERY_PENDING"
    RECOVERY_POINT_VERIFIED = "RECOVERY_POINT_VERIFIED"
    RESUMPTION_ELIGIBLE = "RESUMPTION_ELIGIBLE"


class BackupAgeContainmentViolationException(RuntimeError):
    """Raised when an operation attempts to write ingestion data while containment is breached."""
    pass


class SnapshotPublicationBlockedContainmentException(RuntimeError):
    """Raised when snapshot publication is attempted while backup freshness is unverified."""
    pass


class BackupAgeContainmentGate:
    """
    Deterministic gate controlling ingestion and snapshot publication eligibility.
    """

    def __init__(
        self,
        freshness_authority: Optional[BackupFreshnessAuthority] = None,
        policy: Optional[BackupContainmentPolicy] = None,
        warning_threshold_ratio: float = 0.8,
    ):
        self.policy = policy or BackupContainmentPolicy.from_environment()
        self.freshness_authority = freshness_authority or BackupFreshnessAuthority(policy=self.policy)
        self.warning_threshold_ratio = warning_threshold_ratio
        
        self.current_state: ContainmentState = ContainmentState.HEALTHY
        self.last_suspend_at: Optional[str] = None
        self.last_resume_at: Optional[str] = None
        self.last_error_code: Optional[str] = None
        self.suspension_count: int = 0
        self.resumption_count: int = 0

    def check_ingestion_eligibility(
        self,
        force_refresh: bool = False,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Evaluates whether new ingestion writes and worker processing are permitted.
        If containment is disabled, returns True.
        If containment is breached, transitions state to INGESTION_SUSPENDED and returns False.
        """
        if not self.policy.enabled:
            return True

        receipt = self.freshness_authority.evaluate_freshness(
            force_refresh=force_refresh, current_time=current_time
        )

        if receipt.is_fresh:
            # Check if recovering from suspension
            if self.current_state in (ContainmentState.INGESTION_SUSPENDED, ContainmentState.BACKUP_RECOVERY_PENDING):
                self._transition_to(ContainmentState.RECOVERY_POINT_VERIFIED, receipt)
                self._transition_to(ContainmentState.RESUMPTION_ELIGIBLE, receipt)
                self._transition_to(ContainmentState.HEALTHY, receipt)
                self.last_resume_at = receipt.verified_at
                self.last_error_code = None
                self.resumption_count += 1
                logger.info(
                    f"INGESTION_RESUMED: Verified recovery point {receipt.recovery_point_id} "
                    f"(age={receipt.backup_age_seconds}s). Ingestion eligibility restored."
                )
            else:
                # Warning evaluation
                if (
                    receipt.backup_age_seconds is not None
                    and receipt.backup_age_seconds >= (self.policy.max_verified_age_seconds * self.warning_threshold_ratio)
                ):
                    if self.current_state != ContainmentState.FRESHNESS_WARNING:
                        self._transition_to(ContainmentState.FRESHNESS_WARNING, receipt)
                        logger.warning(
                            f"BACKUP_FRESHNESS_WARNING: Backup age {receipt.backup_age_seconds}s approaches "
                            f"threshold {self.policy.max_verified_age_seconds}s."
                        )
                else:
                    if self.current_state != ContainmentState.HEALTHY:
                        self._transition_to(ContainmentState.HEALTHY, receipt)
            return True
        else:
            # Freshness breached -> suspend ingestion
            if self.current_state != ContainmentState.INGESTION_SUSPENDED and not self.freshness_authority._recovery_in_progress:
                self._transition_to(ContainmentState.INGESTION_SUSPENDED, receipt)
                self.last_suspend_at = receipt.verified_at
                self.last_error_code = receipt.reason_code
                self.suspension_count += 1
                correlation_id = str(uuid.uuid4())
                logger.critical(
                    f"INGESTION_SUSPENDED [cid={correlation_id}]: Containment gate closed. "
                    f"Status={receipt.verification_status}, Reason={receipt.reason_code}, "
                    f"Age={receipt.backup_age_seconds}s, Threshold={self.policy.max_verified_age_seconds}s."
                )
            return False

    def enforce_ingestion_gate(self, current_time: Optional[datetime] = None) -> None:
        """
        Enforces ingestion containment fail-closed.
        Raises BackupAgeContainmentViolationException if containment is breached.
        """
        if not self.check_ingestion_eligibility(current_time=current_time):
            raise BackupAgeContainmentViolationException(
                f"INGESTION_CONTAINMENT_BREACH: Ingestion suspended due to backup staleness/unverifiability: {self.last_error_code}"
            )

    def enforce_snapshot_publication_gate(self, current_time: Optional[datetime] = None) -> None:
        """
        Enforces snapshot publication containment.
        Requires force-refresh of freshness authority.
        Raises SnapshotPublicationBlockedContainmentException if containment is breached.
        """
        if not self.check_ingestion_eligibility(force_refresh=True, current_time=current_time):
            raise SnapshotPublicationBlockedContainmentException(
                f"SNAPSHOT_PUBLICATION_BLOCKED: Cannot publish active snapshot while backup freshness is compromised: {self.last_error_code}"
            )

    def mark_backup_in_progress(self) -> None:
        """Transitions state to BACKUP_RECOVERY_PENDING during active backup execution."""
        self.freshness_authority.mark_recovery_in_progress(True)
        if self.current_state == ContainmentState.INGESTION_SUSPENDED:
            self.current_state = ContainmentState.BACKUP_RECOVERY_PENDING
            logger.info("Containment state transitioned to BACKUP_RECOVERY_PENDING.")

    def mark_backup_completed(self) -> None:
        """Clears backup execution in progress flag."""
        self.freshness_authority.mark_recovery_in_progress(False)

    def _transition_to(self, new_state: ContainmentState, receipt: FreshnessEvaluationReceipt) -> None:
        old_state = self.current_state
        self.current_state = new_state
        logger.debug(f"Containment state transition: {old_state.value} -> {new_state.value} ({receipt.reason_code})")

    def get_observability_envelope(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Returns standardized observability envelope.
        Zero secrets or connection strings exposed.
        """
        receipt = self.freshness_authority.evaluate_freshness(current_time=current_time)
        return {
            "containment_enabled": self.policy.enabled,
            "containment_effective": self.policy.enabled and (not (os.environ.get("ARX_ENVIRONMENT") == "production" and not self.policy.production_authorized)),
            "current_state": self.current_state.value,
            "latest_verified_recovery_point": receipt.recovery_point_id,
            "latest_verified_backup_age_seconds": receipt.backup_age_seconds,
            "configured_max_age_seconds": self.policy.max_verified_age_seconds,
            "last_freshness_check_at": receipt.verified_at,
            "last_freshness_check_result": receipt.verification_status,
            "reason_code": receipt.reason_code,
            "ingestion_permitted": (not self.policy.enabled) or receipt.is_fresh,
            "snapshot_publication_permitted": (not self.policy.enabled) or receipt.is_fresh,
            "last_suspend_at": self.last_suspend_at,
            "last_resume_at": self.last_resume_at,
            "last_error_code": self.last_error_code,
            "suspension_count": self.suspension_count,
            "resumption_count": self.resumption_count,
        }
