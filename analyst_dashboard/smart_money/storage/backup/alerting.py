"""
ARX Terminal — Smart Money Backup Observability & Alerting Interface.

Defines passive, non-mutating observability structures and external alerting interfaces
for the background backup scheduler and freshness containment subsystems:
1. Passive alert evaluation (no state mutation, no domain side effects);
2. Multi-tier severity classification (INFO, WARNING, CRITICAL);
3. Threshold-based alert generation (thread death, consecutive failures, 8h warning, 12h suspension);
4. Anti-storm alert debouncing to prevent flood during persistent failure states;
5. Explicit recovery event generation upon state restoration;
6. Zero transmission to external or paid services without explicit webhook configuration;
7. Zero secrets or database credentials emitted in alert payloads.
"""

import os
import time
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)


class BackupAlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class BackupAlertEvent:
    event_type: str
    severity: BackupAlertSeverity
    message: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class BackupAlertDispatcher:
    """
    Passive dispatcher and rule evaluator for Smart Money backup alerts.
    Does not transmit data externally unless an authorized webhook URL is provided.
    Implements anti-storm debouncing and explicit recovery event generation.
    """

    def __init__(
        self,
        warning_age_threshold_sec: float = 28800.0,  # 8 hours
        critical_age_threshold_sec: float = 43200.0, # 12 hours
        debounce_window_sec: float = 300.0,          # 5 minutes
        webhook_url: Optional[str] = None,
    ):
        self.warning_age_threshold_sec = warning_age_threshold_sec
        self.critical_age_threshold_sec = critical_age_threshold_sec
        self.debounce_window_sec = debounce_window_sec
        self.webhook_url = webhook_url or os.environ.get("ARX_BACKUP_ALERT_WEBHOOK_URL")
        self._handlers: List[Callable[[BackupAlertEvent], None]] = []
        self._history: List[BackupAlertEvent] = []
        self._last_dispatched_at: Dict[str, float] = {}
        self._was_degraded: bool = False

    def register_handler(self, handler: Callable[[BackupAlertEvent], None]) -> None:
        """Registers an in-process callback handler for alert events."""
        self._handlers.append(handler)

    def dispatch(self, event: BackupAlertEvent, enforce_debounce: bool = True) -> bool:
        """
        Dispatches an alert event to registered handlers and logs appropriately.
        Enforces anti-storm debouncing for repeated warnings and critical alerts.
        Returns True if dispatched, False if suppressed by debouncer.
        """
        now = time.time()
        fingerprint = f"{event.event_type}:{event.severity.value}:{event.message}"

        if enforce_debounce and event.severity in (BackupAlertSeverity.WARNING, BackupAlertSeverity.CRITICAL):
            last_sent = self._last_dispatched_at.get(fingerprint, 0.0)
            if (now - last_sent) < self.debounce_window_sec:
                logger.debug(
                    f"Alert {event.event_type} suppressed by debouncer (last sent {now - last_sent:.1f}s ago)."
                )
                return False

        self._last_dispatched_at[fingerprint] = now
        self._history.append(event)

        log_level = logging.INFO
        if event.severity == BackupAlertSeverity.WARNING:
            log_level = logging.WARNING
        elif event.severity == BackupAlertSeverity.CRITICAL:
            log_level = logging.ERROR

        logger.log(
            log_level,
            f"[BACKUP_ALERT:{event.severity.value}] {event.event_type}: {event.message} (metadata={event.metadata})"
        )

        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Error in alert handler callback: {e}")

        # Passive webhook dispatch (only if explicitly configured and permitted)
        if self.webhook_url:
            self._send_webhook(event)

        return True

    def _send_webhook(self, event: BackupAlertEvent) -> None:
        """Transmits alert payload to configured webhook without printing secrets."""
        try:
            import urllib.request
            import json
            payload = json.dumps(event.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "ARX-BackupAlertDispatcher/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                logger.info(f"Webhook alert dispatched successfully (status={resp.status})")
        except Exception as e:
            logger.warning(f"Failed to dispatch alert webhook: {e}")

    def evaluate_scheduler_health(
        self,
        scheduler_status: Optional[Dict[str, Any]],
        containment_status: Optional[Dict[str, Any]] = None,
    ) -> List[BackupAlertEvent]:
        """
        Passively evaluates status envelopes and generates alert events without modifying state.
        Detects thread liveness, failures, stall, overdue passes, and recovery.
        """
        events: List[BackupAlertEvent] = []

        if not scheduler_status:
            return events

        has_degradation = False

        # Rule 1: Scheduler Thread Liveness
        is_running = scheduler_status.get("is_running", scheduler_status.get("thread_running", False))
        is_alive = scheduler_status.get("is_alive", False)
        if not is_alive and is_running:
            has_degradation = True
            events.append(
                BackupAlertEvent(
                    event_type="SCHEDULER_THREAD_TERMINATED",
                    severity=BackupAlertSeverity.CRITICAL,
                    message="SmartMoneyBackupSchedulerThread terminated unexpectedly while worker is running.",
                    metadata={
                        "failed_runs": scheduler_status.get("failed_runs", 0),
                        "last_error": scheduler_status.get("last_error"),
                    },
                )
            )

        # Rule 2: Startup Failure
        if scheduler_status.get("startup_error"):
            has_degradation = True
            events.append(
                BackupAlertEvent(
                    event_type="SCHEDULER_STARTUP_FAILED",
                    severity=BackupAlertSeverity.CRITICAL,
                    message=f"Backup scheduler initial pass failed: {scheduler_status.get('startup_error')}",
                    metadata={"error": scheduler_status.get("startup_error")},
                )
            )

        # Rule 3: Execution Overdue / Hung Pass
        if scheduler_status.get("is_execution_overdue"):
            has_degradation = True
            events.append(
                BackupAlertEvent(
                    event_type="BACKUP_EXECUTION_OVERDUE",
                    severity=BackupAlertSeverity.CRITICAL,
                    message="Backup execution pass exceeded maximum expected duration.",
                    metadata={
                        "pass_duration_sec": scheduler_status.get("pass_duration_sec", 0.0),
                        "max_execution_duration_sec": scheduler_status.get("max_execution_duration_sec", 1800.0),
                    },
                )
            )

        # Rule 4: Scheduler Stalled
        if scheduler_status.get("is_stalled"):
            has_degradation = True
            events.append(
                BackupAlertEvent(
                    event_type="SCHEDULER_STALLED_NO_PROGRESS",
                    severity=BackupAlertSeverity.CRITICAL,
                    message="Backup scheduler thread alive but made no execution progress for over 1.5x scheduled interval.",
                    metadata={
                        "interval_sec": scheduler_status.get("interval_sec"),
                        "last_attempt_timestamp": scheduler_status.get("last_attempt_timestamp"),
                    },
                )
            )

        # Rule 5: Consecutive Failures
        consecutive = scheduler_status.get("consecutive_failures", 0)
        if consecutive >= 3:
            has_degradation = True
            events.append(
                BackupAlertEvent(
                    event_type="CONSECUTIVE_BACKUP_FAILURES_CRITICAL",
                    severity=BackupAlertSeverity.CRITICAL,
                    message=f"Backup scheduler failed {consecutive} consecutive passes.",
                    metadata={
                        "consecutive_failures": consecutive,
                        "last_error": scheduler_status.get("last_error"),
                    },
                )
            )
        elif consecutive >= 1:
            has_degradation = True
            events.append(
                BackupAlertEvent(
                    event_type="CONSECUTIVE_BACKUP_FAILURES_WARNING",
                    severity=BackupAlertSeverity.WARNING,
                    message=f"Backup scheduler pass failed ({consecutive} consecutive).",
                    metadata={
                        "consecutive_failures": consecutive,
                        "last_error": scheduler_status.get("last_error"),
                    },
                )
            )

        # Rule 6: Backup Age Warnings & Containment Gate
        if containment_status:
            age_sec = containment_status.get("latest_verified_backup_age_seconds")
            state = containment_status.get("current_state")

            if state == "INGESTION_SUSPENDED" or (age_sec is not None and age_sec >= self.critical_age_threshold_sec):
                has_degradation = True
                events.append(
                    BackupAlertEvent(
                        event_type="BACKUP_AGE_CONTAINMENT_SUSPENSION",
                        severity=BackupAlertSeverity.CRITICAL,
                        message=f"Ingestion suspended: Backup age ({age_sec}s) breached critical threshold ({self.critical_age_threshold_sec}s).",
                        metadata={
                            "backup_age_seconds": age_sec,
                            "critical_threshold_seconds": self.critical_age_threshold_sec,
                            "recovery_point": containment_status.get("latest_verified_recovery_point"),
                        },
                    )
                )
            elif age_sec is not None and age_sec >= self.warning_age_threshold_sec:
                has_degradation = True
                events.append(
                    BackupAlertEvent(
                        event_type="BACKUP_AGE_WARNING",
                        severity=BackupAlertSeverity.WARNING,
                        message=f"Backup age warning: Backup age ({age_sec}s) exceeds warning threshold ({self.warning_age_threshold_sec}s).",
                        metadata={
                            "backup_age_seconds": age_sec,
                            "warning_threshold_seconds": self.warning_age_threshold_sec,
                        },
                    )
                )

        # Rule 7: Recovery Event Generation
        if not has_degradation and self._was_degraded:
            events.append(
                BackupAlertEvent(
                    event_type="BACKUP_RECOVERY_SUCCESS",
                    severity=BackupAlertSeverity.INFO,
                    message="Backup subsystem has fully recovered from prior degradation.",
                    metadata={
                        "last_backup_id": scheduler_status.get("last_backup_id"),
                        "last_successful_timestamp": scheduler_status.get("last_successful_timestamp"),
                        "consecutive_failures": 0,
                    },
                )
            )
            self._was_degraded = False
        elif has_degradation:
            self._was_degraded = True

        return events

    def get_alert_history(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._history]
