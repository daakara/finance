"""
Unit and regression tests for ARX Smart Money Worker Sentry Telemetry & Probe.
DIRECTIVE: ARX-SM-ACT-020-SENTRY-ACCEPTANCE-005

Enforces:
1. Safe fail-open initialization (missing DSN, invalid DSN, SDK init failure).
2. Proper tag attribution (subsystem=smart_money, service=arx-smart-money-worker, release, environment).
3. Sensitive value scrubbing & data minimization.
4. Nonfatal diagnostic telemetry probe contracts (zero SEC requests, zero DB mutations, zero locks).
5. Robustness under timeouts, errors, STANDBY, and emergency stop modes.
"""

import os
import sys
import uuid
import pytest
from unittest.mock import patch, MagicMock

from analyst_dashboard.smart_money.worker import (
    init_worker_sentry,
    capture_worker_exception,
    emit_worker_telemetry_probe,
    _sanitize_worker_event,
    SmartMoneyIngestionWorker,
    WorkerHealthState,
)


class TestWorkerSentryInitialization:
    """Test suite for Sentry initialization and failure-injection."""

    def test_init_missing_dsn(self, monkeypatch):
        """Missing SENTRY_DSN safely returns False without raising."""
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        assert init_worker_sentry() is False

    def test_init_whitespace_dsn(self, monkeypatch):
        """Whitespace-only SENTRY_DSN safely returns False."""
        monkeypatch.setenv("SENTRY_DSN", "   \t  \n ")
        assert init_worker_sentry() is False

    def test_init_sdk_exception_fail_open(self, monkeypatch):
        """SDK initialization failure logs a warning and fails open."""
        monkeypatch.setenv("SENTRY_DSN", "https://key@o1234.ingest.sentry.io/5678")
        with patch("sentry_sdk.init", side_effect=RuntimeError("Sentry init boom")):
            assert init_worker_sentry() is False

    def test_init_success_with_canonical_tags(self, monkeypatch):
        """Successful initialization sets service, subsystem, environment, release."""
        monkeypatch.setenv("SENTRY_DSN", "https://key@o1234.ingest.sentry.io/5678")
        monkeypatch.setenv("ARX_ENVIRONMENT", "production")
        monkeypatch.setenv("ARX_RELEASE_SHA", "e62f16e2a642433574ae84ab69b47dd33217079d")

        mock_scope = MagicMock()
        with patch("sentry_sdk.init") as mock_init, \
             patch("sentry_sdk.get_global_scope", return_value=mock_scope):
            assert init_worker_sentry() is True
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["dsn"] == "https://key@o1234.ingest.sentry.io/5678"
            assert kwargs["environment"] == "production"
            assert kwargs["release"] == "e62f16e2a642433574ae84ab69b47dd33217079d"
            assert kwargs["send_default_pii"] is False
            assert kwargs["traces_sample_rate"] == 0.0

            # Verify global scope tags
            mock_scope.set_tag.assert_any_call("service", "arx-smart-money-worker")
            mock_scope.set_tag("subsystem", "smart_money")


class TestWorkerExceptionCapture:
    """Test suite for worker exception capture and tag isolation."""

    def test_capture_exception_success(self, monkeypatch):
        """Captures exception with mandatory service and subsystem tags."""
        monkeypatch.setenv("SENTRY_DSN", "https://key@o1234.ingest.sentry.io/5678")
        monkeypatch.setenv("ARX_RELEASE_SHA", "test_sha_123")
        monkeypatch.setenv("ARX_ENVIRONMENT", "production")

        mock_scope = MagicMock()
        mock_scope_context = MagicMock()
        mock_scope_context.__enter__.return_value = mock_scope
        mock_scope_context.__exit__.return_value = False

        with patch("analyst_dashboard.smart_money.worker.init_worker_sentry", return_value=True), \
             patch("sentry_sdk.push_scope", return_value=mock_scope_context), \
             patch("sentry_sdk.capture_exception", return_value="evt_999"):
            event_id = capture_worker_exception(ValueError("Worker failure"), tags={"custom_k": "custom_v"})
            assert event_id == "evt_999"

            mock_scope.set_tag.assert_any_call("service", "arx-smart-money-worker")
            mock_scope.set_tag.assert_any_call("subsystem", "smart_money")
            mock_scope.set_tag.assert_any_call("environment", "production")
            mock_scope.set_tag.assert_any_call("release", "test_sha_123")
            mock_scope.set_tag.assert_any_call("custom_k", "custom_v")

    def test_capture_exception_fail_open_on_sentry_error(self, monkeypatch):
        """If Sentry SDK capture fails, capture_worker_exception returns None without raising."""
        monkeypatch.setenv("SENTRY_DSN", "https://key@o1234.ingest.sentry.io/5678")
        with patch("analyst_dashboard.smart_money.worker.init_worker_sentry", return_value=True), \
             patch("sentry_sdk.capture_exception", side_effect=Exception("Transport dropped")):
            event_id = capture_worker_exception(RuntimeError("fatal error"))
            assert event_id is None


class TestWorkerTelemetryProbe:
    """Test suite for one-shot safe nonfatal diagnostic telemetry probe."""

    def test_probe_missing_dsn_fails_safely(self, monkeypatch):
        """When SENTRY_DSN is missing, probe returns SKIPPED_NO_DSN receipt."""
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        receipt = emit_worker_telemetry_probe(directive_id="ARX-TEST-001")
        assert receipt["status"] == "SKIPPED_NO_DSN"
        assert receipt["sec_traffic_generated"] is False
        assert receipt["database_mutated"] is False
        assert receipt["advisory_lock_held"] is False

    def test_probe_success_receipt(self, monkeypatch):
        """Emits probe carrying required metadata and returns SENT status."""
        monkeypatch.setenv("SENTRY_DSN", "https://key@o1234.ingest.sentry.io/5678")
        monkeypatch.setenv("ARX_ENVIRONMENT", "production")
        monkeypatch.setenv("ARX_RELEASE_SHA", "e62f16e2a642433574ae84ab69b47dd33217079d")

        mock_scope = MagicMock()
        mock_scope_context = MagicMock()
        mock_scope_context.__enter__.return_value = mock_scope
        mock_scope_context.__exit__.return_value = False

        probe_id = "test-probe-uuid-001"
        with patch("sentry_sdk.init"), \
             patch("sentry_sdk.push_scope", return_value=mock_scope_context), \
             patch("sentry_sdk.capture_message", return_value="event_abc_123") as mock_cap, \
             patch("sentry_sdk.flush") as mock_flush:

            receipt = emit_worker_telemetry_probe(
                directive_id="ARX-SM-ACT-020-SENTRY-ACCEPTANCE-005",
                probe_uuid=probe_id,
            )

            assert receipt["status"] == "SENT"
            assert receipt["event_id"] == "event_abc_123"
            assert receipt["probe_uuid"] == probe_id
            assert receipt["directive_id"] == "ARX-SM-ACT-020-SENTRY-ACCEPTANCE-005"
            assert receipt["subsystem"] == "smart_money"
            assert receipt["service"] == "arx-smart-money-worker"
            assert receipt["environment"] == "production"
            assert receipt["release_sha"] == "e62f16e2a642433574ae84ab69b47dd33217079d"
            assert receipt["synthetic"] is True
            assert receipt["sec_traffic_generated"] is False
            assert receipt["database_mutated"] is False
            assert receipt["advisory_lock_held"] is False

            # Verify tags on scope
            mock_scope.set_tag.assert_any_call("subsystem", "smart_money")
            mock_scope.set_tag.assert_any_call("service", "arx-smart-money-worker")
            mock_scope.set_tag.assert_any_call("synthetic", "true")
            mock_scope.set_tag.assert_any_call("probe_uuid", probe_id)
            mock_scope.set_tag.assert_any_call("probe_type", "worker_telemetry_acceptance")

            # Verify flush was called
            mock_flush.assert_called_once_with(timeout=5.0)

    def test_probe_network_timeout_or_rejection(self, monkeypatch):
        """Network timeout or exception during capture/flush returns FAILED without crashing."""
        monkeypatch.setenv("SENTRY_DSN", "https://key@o1234.ingest.sentry.io/5678")
        with patch("sentry_sdk.init"), \
             patch("sentry_sdk.capture_message", side_effect=TimeoutError("Sentry network timeout")):
            receipt = emit_worker_telemetry_probe(directive_id="ARX-TIMEOUT-TEST")
            assert receipt["status"] == "FAILED"
            assert "timeout" in receipt["error"].lower()
            assert receipt["sec_traffic_generated"] is False
            assert receipt["database_mutated"] is False

    def test_probe_unique_uuid_generation(self, monkeypatch):
        """Consecutive calls without explicit UUID generate unique IDs."""
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        r1 = emit_worker_telemetry_probe()
        r2 = emit_worker_telemetry_probe()
        assert r1["probe_uuid"] != r2["probe_uuid"]


class TestWorkerTelemetrySecurityAndInvariants:
    """Test suite for zero-disclosure and operational invariance."""

    def test_sanitize_worker_event_redacts_credentials(self):
        """Sanitizer scrubs password, secret, token, and auth strings."""
        dirty_event = {
            "message": "Error connecting to postgres://user:SuperSecretPassword123@host:5432/db",
            "extra": {
                "auth_token": "bearer_xyz_secret_999",
                "database_url": "postgresql://admin:TopSecret@host/prod",
                "safe_field": "public_info",
            },
        }
        cleaned = _sanitize_worker_event(dirty_event, {})
        assert cleaned["extra"]["auth_token"] == "[REDACTED]"
        assert cleaned["extra"]["database_url"] == "[REDACTED]"
        assert cleaned["extra"]["safe_field"] == "public_info"
        assert "SuperSecretPassword123" not in cleaned["message"]
        assert "postgres://user:[REDACTED]@host:5432/db" in cleaned["message"]

    def test_telemetry_never_activates_ingestion_or_acquires_lock(self, monkeypatch):
        """Sentry operations never set worker to OWNER or acquire SEC egress lock."""
        monkeypatch.setenv("SENTRY_DSN", "https://key@o1234.ingest.sentry.io/5678")
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "STANDBY")

        with patch("analyst_dashboard.smart_money.worker.get_database_url", return_value="postgresql://mock"), \
             patch("analyst_dashboard.smart_money.worker.BackupFreshnessAuthority"), \
             patch("analyst_dashboard.smart_money.worker.PostgresSmartMoneyRepository"), \
             patch("analyst_dashboard.smart_money.worker.BackupAgeContainmentGate"), \
             patch("analyst_dashboard.smart_money.worker.init_worker_sentry", return_value=True), \
             patch("sentry_sdk.capture_message", return_value="evt_mock_123"), \
             patch("sentry_sdk.flush"):
            worker = SmartMoneyIngestionWorker()
            assert worker.state == WorkerHealthState.STARTING

            # Emit probe
            receipt = emit_worker_telemetry_probe()
            assert receipt["advisory_lock_held"] is False
            assert receipt["status"] == "SENT"

            # Verify worker state is unchanged
            assert worker.state == WorkerHealthState.STARTING

    def test_telemetry_in_emergency_stop_mode(self, monkeypatch):
        """Probe functions cleanly when ARX_EMERGENCY_STOP_ACTIVE=true without altering flag."""
        monkeypatch.setenv("ARX_EMERGENCY_STOP_ACTIVE", "true")
        monkeypatch.delenv("SENTRY_DSN", raising=False)

        receipt = emit_worker_telemetry_probe()
        assert receipt["status"] == "SKIPPED_NO_DSN"
        assert os.environ.get("ARX_EMERGENCY_STOP_ACTIVE") == "true"
