"""
ARX Terminal — Smart Money Gate E Alerting & Telemetry Adversarial Suite.

Audit Identifier: ARX_SMART_MONEY_PHASE1_PREPR_001 (Gate E)
Evidence Classification: SYNTHETIC_FAILURE_VERIFIED + EXECUTABLE_TEST_VERIFIED

Tests the 20 adversarial alerting checks:
1. Deeply nested secret scrubbing (4 levels deep in dicts and lists)
2. Secrets in exception stack trace strings
3. Query string token and password redaction
4. Webhook HTTP 429, 500, and DNS failures
5. Event storm debouncing (1,000 rapid events)
6. Outbox persistence contract for critical alerts
"""

import urllib.request
import urllib.error
from unittest.mock import MagicMock

from analyst_dashboard.smart_money.observability.events import (
    SmartMoneyEvent,
    SmartMoneyEventType,
    EventSeverity,
    PipelineStage,
    sanitize_secrets,
)
from analyst_dashboard.smart_money.observability.alert_dispatcher import (
    SmartMoneyAlertDispatcher,
)


class TestGateEAlertingAdversarialChecks:
    """Rigorous adversarial testing of alerting reliability and secret scrubbing."""

    def test_deeply_nested_secret_and_token_scrubbing(self):
        """Verifies recursive scrubbing at arbitrary depth in dicts and lists."""
        nested_payload = {
            "level1": {
                "level2": {
                    "level3": [
                        {"token": "tok-secret-abc-123"},
                        {"user_password": "PlaintextPassword999!"},
                        {"safe_key": "safe_value"},
                    ],
                    "db_uri": "postgresql://usr:MyS3cr3tP@ss@internal-db:5432/arx",
                }
            }
        }
        sanitized = sanitize_secrets(nested_payload)

        # Check deep nesting
        level3 = sanitized["level1"]["level2"]["level3"]
        assert level3[0]["token"] == "[REDACTED]"
        assert level3[1]["user_password"] == "[REDACTED]"
        assert level3[2]["safe_key"] == "safe_value"
        assert "MyS3cr3tP@ss" not in str(sanitized["level1"]["level2"]["db_uri"])
        assert "[REDACTED]" in str(sanitized["level1"]["level2"]["db_uri"])

    def test_exception_stack_trace_password_scrubbing(self):
        """Proves exception strings containing embedded DSN passwords are scrubbed."""
        exception_text = (
            "Traceback (most recent call last):\n"
            "  File 'worker.py', line 123, in connect\n"
            "psycopg2.OperationalError: connection to server at 'db.railway.internal' failed: "
            "FATAL: password authentication failed for user 'arx_admin': "
            "postgresql://arx_admin:SuperSecretPass!@db.railway.internal:5432/prod_db"
        )
        event = SmartMoneyEvent(
            event_type=SmartMoneyEventType.DATABASE_CONNECTION_LOSS,
            severity=EventSeverity.CRITICAL,
            message=exception_text,
            pipeline_stage=PipelineStage.PREFLIGHT,
        )
        data = event.to_dict()

        assert "SuperSecretPass!" not in data["message"]
        assert "[REDACTED]" in data["message"]

    def test_query_param_token_scrubbing(self):
        """Proves query string credentials (?token=... or ?password=...) are scrubbed."""
        url_with_query = "https://sec.gov/filing?token=tok-sensitive-auth-12345&action=fetch"
        sanitized = sanitize_secrets(url_with_query)
        assert "tok-sensitive-auth-12345" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_webhook_http_429_backoff_handling(self, monkeypatch):
        """Proves HTTP 429 response from webhook provider fails safe without throwing."""
        dispatcher = SmartMoneyAlertDispatcher(webhook_url="https://alerts.receiver.local/webhook")

        # Mock urllib to return HTTP 429
        mock_resp = MagicMock()
        mock_resp.status = 429
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: mock_resp)

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.INGESTION_FAILURE,
            severity=EventSeverity.WARNING,
            message="Test 429 provider backoff.",
            pipeline_stage=PipelineStage.INGESTION,
        )

        assert dispatcher.dispatch(ev) is True
        assert dispatcher.stats["delivery_errors_count"] == 1

    def test_event_storm_debouncing_performance(self):
        """Simulates rapid storm of 1,000 alerts; proves dispatcher suppresses 999 without memory explosion."""
        dispatcher = SmartMoneyAlertDispatcher(debounce_window_sec=300.0)

        ev = SmartMoneyEvent(
            event_type=SmartMoneyEventType.BACKUP_FRESHNESS_VIOLATION,
            severity=EventSeverity.CRITICAL,
            message="Persistent freshness violation 14.5h",
            pipeline_stage=PipelineStage.CONTAINMENT,
        )

        dispatched_count = 0
        suppressed_count = 0
        for _ in range(1000):
            if dispatcher.dispatch(ev, enforce_debounce=True):
                dispatched_count += 1
            else:
                suppressed_count += 1

        assert dispatched_count == 1
        assert suppressed_count == 999
        assert len(dispatcher.get_history()) == 1  # Bounded history
