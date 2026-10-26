"""
ARX Terminal — Smart Money Solo-Maintainer Alert Dispatcher & Routing Engine.

Enforces:
1. Prioritized solo-maintainer notification routing (Webhook / Sentry / In-process handlers);
2. Anti-storm debouncing to prevent alert fatigue during persistent degradation;
3. Failure escalation on repeated consecutive incidents;
4. Deterministic recovery event generation upon incident resolution;
5. Fail-safe delivery: Telemetry outages never crash the operational pipeline.
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Callable

from analyst_dashboard.smart_money.observability.events import (
    SmartMoneyEvent,
    SmartMoneyEventType,
    EventSeverity,
    PipelineStage,
)

logger = logging.getLogger("smart_money.observability")


class SmartMoneyAlertDispatcher:
    """
    Solo-maintainer alert dispatcher with debouncing, escalation,
    and automatic recovery notifications.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        debounce_window_sec: float = 300.0,
        escalation_threshold: int = 3,
        timeout_sec: float = 5.0,
    ):
        self.webhook_url = webhook_url or os.environ.get("ARX_ALERT_WEBHOOK_URL")
        self.debounce_window_sec = debounce_window_sec
        self.escalation_threshold = escalation_threshold
        self.timeout_sec = timeout_sec

        self._handlers: List[Callable[[SmartMoneyEvent], None]] = []
        self._history: List[Dict[str, Any]] = []
        self._last_dispatched_at: Dict[str, float] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._active_degradations: Dict[str, SmartMoneyEvent] = {}

        self.stats = {
            "dispatched_count": 0,
            "suppressed_count": 0,
            "delivery_errors_count": 0,
            "recoveries_emitted": 0,
        }

    def register_handler(self, handler: Callable[[SmartMoneyEvent], None]) -> None:
        """Registers an in-process callback listener."""
        self._handlers.append(handler)

    def dispatch(self, event: SmartMoneyEvent, enforce_debounce: bool = True) -> bool:
        """
        Processes and dispatches an event.
        Returns True if dispatched to handlers/webhook, False if debounced/suppressed.
        """
        now = time.time()
        event_dict = event.to_dict()
        category = event.event_type.value
        fingerprint = f"{category}:{event.severity.value}:{event.message}"

        # 1. Update Failure Tracking & Escalation
        if event.severity in (EventSeverity.WARNING, EventSeverity.CRITICAL):
            count = self._consecutive_failures.get(category, 0) + 1
            self._consecutive_failures[category] = count
            self._active_degradations[category] = event

            # Escalate severity if threshold reached
            if count >= self.escalation_threshold and event.severity == EventSeverity.WARNING:
                event.severity = EventSeverity.CRITICAL
                event.action_required = True
                event.message = f"[ESCALATED after {count} consecutive failures] {event.message}"
                event_dict = event.to_dict()
        else:
            # INFO event clears consecutive failure count for category
            if category in self._active_degradations:
                # Issue recovery event
                self._emit_recovery(category, event)
            self._consecutive_failures.pop(category, None)

        # 2. Debouncing check for WARNING and CRITICAL
        if enforce_debounce and event.severity in (EventSeverity.WARNING, EventSeverity.CRITICAL):
            last_sent = self._last_dispatched_at.get(fingerprint, 0.0)
            if (now - last_sent) < self.debounce_window_sec:
                self.stats["suppressed_count"] += 1
                logger.debug(f"Event {category} suppressed by debounce window ({now - last_sent:.1f}s ago).")
                return False

        # 3. Record dispatch
        self._last_dispatched_at[fingerprint] = now
        self._history.append(event_dict)
        if len(self._history) > 1000:
            self._history.pop(0)

        self.stats["dispatched_count"] += 1

        # 4. In-process handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Error in registered alert handler: {e}")

        # 5. External Webhook Delivery (fail-safe)
        if self.webhook_url:
            self._transmit_webhook(event_dict)

        return True

    def _emit_recovery(self, category: str, resolution_event: SmartMoneyEvent) -> None:
        """Emits an explicit recovery event when an active degradation is resolved."""
        prior = self._active_degradations.pop(category, None)
        if not prior:
            return

        recovery_event = SmartMoneyEvent(
            event_type=SmartMoneyEventType.RECOVERY_ACTION_TRIGGERED,
            severity=EventSeverity.INFO,
            message=f"RECOVERY: Degradation in {category} resolved. Resolution: {resolution_event.message}",
            pipeline_stage=PipelineStage.RECOVERY,
            action_required=False,
            correlation_id=prior.event_id,
            metadata={
                "resolved_category": category,
                "prior_incident_severity": prior.severity.value,
                "consecutive_failures_before_recovery": self._consecutive_failures.get(category, 0),
            },
        )
        self.stats["recoveries_emitted"] += 1
        self._history.append(recovery_event.to_dict())

        for handler in self._handlers:
            try:
                handler(recovery_event)
            except Exception as e:
                logger.warning(f"Error in handler during recovery dispatch: {e}")

        if self.webhook_url:
            self._transmit_webhook(recovery_event.to_dict())

    def _transmit_webhook(self, payload: Dict[str, Any]) -> bool:
        """
        Transmits JSON payload to external webhook.
        Fail-safe: Returns False on any network/HTTP failure without raising exceptions.
        """
        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ARX-SmartMoneyObservability/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                if 200 <= resp.status < 300:
                    return True
                else:
                    self.stats["delivery_errors_count"] += 1
                    logger.warning(f"Webhook returned non-success HTTP status: {resp.status}")
                    return False
        except Exception as e:
            self.stats["delivery_errors_count"] += 1
            logger.warning(f"Webhook delivery failure (fail-safe suppressed): {e}")
            return False

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)
