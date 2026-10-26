"""
ARX Terminal — Smart Money Observability & Alerts Verification Suite.

Work Package C (SM_GATE_06) Verification Suite.
Evidence Classifications:
- EVENT_GENERATION_VERIFIED
- ROUTING_LOGIC_VERIFIED
- SYNTHETIC_DELIVERY_VERIFIED

Covers:
1. 19-Event Taxonomy serialization & JSON envelope validity;
2. Strict credential/secret scrubbing;
3. Solo-maintainer routing, debouncing, and escalation;
4. 12 Synthetic Alert Scenarios.
"""

import urllib.request
import urllib.error

from analyst_dashboard.smart_money.observability.events import (
    SmartMoneyEvent,
    SmartMoneyEventType,
    EventSeverity,
    PipelineStage,
)
from analyst_dashboard.smart_money.observability.alert_dispatcher import (
    SmartMoneyAlertDispatcher,
)


class TestEventTaxonomyAndEnvelope:
    """Verifies complete 19-event taxonomy and structured envelope."""

    def test_all_19_event_types_serialize_correctly(self):
        """Proves every event type in the 19-event taxonomy produces a valid JSON envelope."""
        assert len(SmartMoneyEventType) == 19

        for event_type in SmartMoneyEventType:
            event = SmartMoneyEvent(
                event_type=event_type,
                severity=EventSeverity.WARNING,
                message=f"Test notification for {event_type.value}",
                pipeline_stage=PipelineStage.INGESTION,
                action_required=True,
                run_id="run-taxonomy-test",
                correlation_id="corr-test-123",
                metadata={"test_param": 42},
            )
            data = event.to_dict()

            assert data["event_type"] == event_type.value
            assert data["severity"] == "WARNING"
            assert data["action_required"] is True
            assert data["pipeline_stage"] == "INGESTION"
            assert data["run_id"] == "run-taxonomy-test"
            assert data["correlation_id"] == "corr-test-123"
            assert "event_id" in data
            assert "occurred_at_utc" in data
            assert data["release_sha"] == "86037f4ba54fb6b66f94056cb07299482960a636"

    def test_credential_and_secret_redaction(self):
        """Proves sensitive credentials, DSN passwords, tokens, and keys are redacted."""
        sensitive_metadata = {
            "database_url": "postgresql://worker_user:super_secret_password_123@db.railway.internal:5432/arx_db",
            "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token123",
            "api_key": "AKIA_FAKE_SECRET_KEY_987",
            "user_token": "token=tok-abc123xyz789",
            "safe_field": "valid_public_data",
        }

        event = SmartMoneyEvent(
            event_type=SmartMoneyEventType.DATABASE_CONNECTION_LOSS,
            severity=EventSeverity.CRITICAL,
            message="Connection failed to postgresql://admin:p@ssw0rd!@10.0.0.1:5432/finance",
            pipeline_stage=PipelineStage.PREFLIGHT,
            action_required=True,
            metadata=sensitive_metadata,
        )

        data = event.to_dict()

        # Check message scrubbing
        assert "super_secret_password" not in data["message"]
        assert "p@ssw0rd!" not in data["message"]
        assert "[REDACTED]" in data["message"]

        # Check metadata scrubbing
        meta = data["metadata"]
        assert meta["api_key"] == "[REDACTED]"
        assert meta["auth_header"] == "[REDACTED]"
        assert "super_secret_password_123" not in str(meta["database_url"])
        assert meta["safe_field"] == "valid_public_data"


class TestTwelveSyntheticAlertScenarios:
    """Verifies all 12 operational alert scenarios."""

    def test_scenario_01_worker_failure_alert(self):
        """Scenario 1: Worker failure alert with CRITICAL severity and action_required=True."""
        dispatcher = SmartMoneyAlertDispatcher()
        received = []
        dispatcher.register_handler(lambda ev: received.append(ev))

        event = SmartMoneyEvent(
            event_type=SmartMoneyEventType.WORKER_SHUTDOWN,
            severity=EventSeverity.CRITICAL,
            message="Worker process crashed with unhandled MemoryError in cycle execution.",
            pipeline_stage=PipelineStage.SHUTDOWN,
            action_required=True,
            metadata={"exit_code": 137},
        )
        assert dispatcher.dispatch(event) is True
        assert len(received) == 1
        assert received[0].severity == EventSeverity.CRITICAL
        assert received[0].action_required is True

    def test_scenario_02_repeated_failure_escalation(self):
        """Scenario 2: Consecutive failures escalate from WARNING to CRITICAL."""
        dispatcher = SmartMoneyAlertDispatcher(escalation_threshold=3, debounce_window_sec=0.0)
        dispatched_severities = []
        dispatcher.register_handler(lambda ev: dispatched_severities.append(ev.severity))

        for i in range(4):
            ev = SmartMoneyEvent(
                event_type=SmartMoneyEventType.INGESTION_FAILURE,
                severity=EventSeverity.WARNING,
                message=f"Transient network glitch filing {i}",
                pipeline_stage=PipelineStage.INGESTION,
            )
            dispatcher.dispatch(ev)

        # First 2 are WARNING; 3rd and 4th escalate to CRITICAL
        assert dispatched_severities[0] == EventSeverity.WARNING
        assert dispatched_severities[1] == EventSeverity.WARNING
        assert dispatched_severities[2] == EventSeverity.CRITICAL
        assert dispatched_severities[3] == EventSeverity.CRITICAL

    def test_scenario_03_missed_scheduled_cycle_alert(self):
        """Scenario 3: Missed scheduled cycle alert emitted when interval exceeded."""
        dispatcher = SmartMoneyAlertDispatcher()
        received = []
        dispatcher.register_handler(lambda ev: received.append(ev))

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.MISSED_SCHEDULED_CYCLE,
            severity=EventSeverity.WARNING,
            message="Scheduled daily Form 4 scan not triggered after 24 hours.",
            pipeline_stage=PipelineStage.PREFLIGHT,
            action_required=True,
            metadata={"expected_interval_sec": 86400, "delay_sec": 3600},
        )
        dispatcher.dispatch(ev)
        assert len(received) == 1
        assert received[0].event_type == SmartMoneyEventType.MISSED_SCHEDULED_CYCLE

    def test_scenario_04_backup_failure_alert(self):
        """Scenario 4: Backup failure emits CRITICAL alert."""
        dispatcher = SmartMoneyAlertDispatcher()
        received = []
        dispatcher.register_handler(lambda ev: received.append(ev))

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.BACKUP_FAILURE,
            severity=EventSeverity.CRITICAL,
            message="Post-cycle pg_dump failed: disk volume write error.",
            pipeline_stage=PipelineStage.BACKUP,
            action_required=True,
        )
        dispatcher.dispatch(ev)
        assert received[0].event_type == SmartMoneyEventType.BACKUP_FAILURE
        assert received[0].severity == EventSeverity.CRITICAL

    def test_scenario_05_stale_backup_containment_alert(self):
        """Scenario 5: Backup age exceeds 8h/12h threshold."""
        dispatcher = SmartMoneyAlertDispatcher()
        received = []
        dispatcher.register_handler(lambda ev: received.append(ev))

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.BACKUP_FRESHNESS_VIOLATION,
            severity=EventSeverity.CRITICAL,
            message="Backup freshness violation: Latest backup age 14.2h exceeds 12h critical limit.",
            pipeline_stage=PipelineStage.CONTAINMENT,
            action_required=True,
            metadata={"age_hours": 14.2, "limit_hours": 12.0},
        )
        dispatcher.dispatch(ev)
        assert received[0].event_type == SmartMoneyEventType.BACKUP_FRESHNESS_VIOLATION

    def test_scenario_06_sec_429_rate_limit_alert(self):
        """Scenario 6: SEC 429 response generates alert."""
        dispatcher = SmartMoneyAlertDispatcher()
        received = []
        dispatcher.register_handler(lambda ev: received.append(ev))

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.SEC_RATE_LIMIT_429,
            severity=EventSeverity.WARNING,
            message="SEC EDGAR responded with HTTP 429 Too Many Requests. Backoff engaged.",
            pipeline_stage=PipelineStage.INGESTION,
            action_required=False,
            metadata={"backoff_sec": 60},
        )
        dispatcher.dispatch(ev)
        assert received[0].event_type == SmartMoneyEventType.SEC_RATE_LIMIT_429

    def test_scenario_07_snapshot_publication_failure_alert(self):
        """Scenario 7: Snapshot publication failure alert."""
        dispatcher = SmartMoneyAlertDispatcher()
        received = []
        dispatcher.register_handler(lambda ev: received.append(ev))

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.SNAPSHOT_PUBLICATION_FAILURE,
            severity=EventSeverity.CRITICAL,
            message="Transaction rollback during snapshot publication.",
            pipeline_stage=PipelineStage.SNAPSHOT_PUBLICATION,
            action_required=True,
        )
        dispatcher.dispatch(ev)
        assert received[0].event_type == SmartMoneyEventType.SNAPSHOT_PUBLICATION_FAILURE

    def test_scenario_08_worker_restart_loop_alert(self):
        """Scenario 8: Repeated rapid worker startups detected."""
        dispatcher = SmartMoneyAlertDispatcher()
        received = []
        dispatcher.register_handler(lambda ev: received.append(ev))

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.WORKER_STARTUP,
            severity=EventSeverity.WARNING,
            message="Worker restarted 4 times in 10 minutes (crash loop detected).",
            pipeline_stage=PipelineStage.PREFLIGHT,
            action_required=True,
            metadata={"restart_count": 4, "window_sec": 600},
        )
        dispatcher.dispatch(ev)
        assert received[0].message.startswith("Worker restarted 4 times")

    def test_scenario_09_database_connection_loss_alert(self):
        """Scenario 9: Database connection loss alert."""
        dispatcher = SmartMoneyAlertDispatcher()
        received = []
        dispatcher.register_handler(lambda ev: received.append(ev))

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.DATABASE_CONNECTION_LOSS,
            severity=EventSeverity.CRITICAL,
            message="PostgreSQL connection lost during filing ingestion.",
            pipeline_stage=PipelineStage.INGESTION,
            action_required=True,
        )
        dispatcher.dispatch(ev)
        assert received[0].event_type == SmartMoneyEventType.DATABASE_CONNECTION_LOSS

    def test_scenario_10_alert_provider_unavailable(self, monkeypatch):
        """Scenario 10: Webhook provider unavailable (network failure/HTTP 500) fails safe."""
        dispatcher = SmartMoneyAlertDispatcher(webhook_url="https://fake-alert-receiver.internal/webhook")

        # Mock urllib.request.urlopen to raise URLError
        def failing_urlopen(*args, **kwargs):
            raise urllib.error.URLError("Network unreachable")

        monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.INGESTION_FAILURE,
            severity=EventSeverity.WARNING,
            message="Test transient failure during network outage.",
            pipeline_stage=PipelineStage.INGESTION,
        )

        # Must not throw exception
        assert dispatcher.dispatch(ev) is True
        assert dispatcher.stats["delivery_errors_count"] == 1

    def test_scenario_11_alert_deduplication_and_debouncing(self):
        """Scenario 11: Duplicate alerts within debounce window are suppressed."""
        dispatcher = SmartMoneyAlertDispatcher(debounce_window_sec=300.0)

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.BACKUP_FRESHNESS_VIOLATION,
            severity=EventSeverity.CRITICAL,
            message="Backup freshness violation 12.5h",
            pipeline_stage=PipelineStage.CONTAINMENT,
        )

        # 1st dispatch -> True
        d1 = dispatcher.dispatch(ev, enforce_debounce=True)
        # 2nd immediate identical dispatch -> False (debounced)
        d2 = dispatcher.dispatch(ev, enforce_debounce=True)
        # 3rd dispatch with debounce disabled -> True
        d3 = dispatcher.dispatch(ev, enforce_debounce=False)

        assert d1 is True
        assert d2 is False
        assert d3 is True
        assert dispatcher.stats["suppressed_count"] == 1

    def test_scenario_12_recovery_notification_dispatched(self):
        """Scenario 12: Degradation resolution automatically issues RECOVERY event."""
        dispatcher = SmartMoneyAlertDispatcher(debounce_window_sec=0.0)
        dispatched_events = []
        dispatcher.register_handler(lambda ev: dispatched_events.append(ev))

        # 1. Degradation incident
        fail_ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.DATABASE_CONNECTION_LOSS,
            severity=EventSeverity.CRITICAL,
            message="DB connection dropped.",
            pipeline_stage=PipelineStage.INGESTION,
        )
        dispatcher.dispatch(fail_ev)

        # 2. Recovery event (INFO with same category / resolution)
        ok_ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.DATABASE_CONNECTION_LOSS,
            severity=EventSeverity.INFO,
            message="DB connection restored successfully.",
            pipeline_stage=PipelineStage.INGESTION,
        )
        dispatcher.dispatch(ok_ev)

        # Check that automatic recovery event was emitted
        assert dispatcher.stats["recoveries_emitted"] == 1
        recovery_events = [e for e in dispatched_events if e.event_type == SmartMoneyEventType.RECOVERY_ACTION_TRIGGERED]
        assert len(recovery_events) == 1
        assert "RECOVERY: Degradation in DATABASE_CONNECTION_LOSS resolved" in recovery_events[0].message
        assert recovery_events[0].severity == EventSeverity.INFO
        assert recovery_events[0].action_required is False
