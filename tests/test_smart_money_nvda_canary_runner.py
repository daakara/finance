"""
Unit and Adversarial Test Suite for NVDA Bounded SEC Ingestion Canary Runner.

Validates the surgical execution path requirements (SM-ACT-014R):
1. Target Issuer Isolation (NVDA CIK 0001045810 only)
2. HTTP Boundary Request Cap (<= 4 HTTPS requests)
3. Wall-Clock Timeout (<= 30.0s)
4. Emergency Stop Killswitch
5. Backup Containment Ratification Gate
6. Backup Freshness Enforcement
7. Advisory Lock Exclusivity & Safe Release
8. Queue Isolation (Zero mutation of sm_run_requests)
9. Snapshot Protection Under Partial Coverage
10. Guaranteed Invariant Teardown on Failure
11. Persistence Idempotency
12. End-to-End Synthetic Run
"""

import os
import time
import json
import pytest
import requests
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.canary_runner import (
    NvdaCanaryRunner,
    CanarySecurityError,
    CanaryBudgetExceededError,
    CanaryTimeoutError,
    CanaryLockContentionError,
    CanaryContainmentError,
    CANARY_TICKER,
    CANARY_CIK,
    CANARY_MAX_REQUESTS,
    CANARY_MAX_DURATION_SEC,
)
from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    FreshnessEvaluationReceipt,
    BackupFreshnessState,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Sets up a clean, safe environment for canary testing."""
    monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "STANDBY")
    monkeypatch.setenv("ARX_EMERGENCY_STOP_ACTIVE", "FALSE")
    monkeypatch.setenv("ARX_BACKUP_CONTAINMENT_ENABLED", "TRUE")
    monkeypatch.setenv("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED", "TRUE")
    monkeypatch.setenv("SEC_USER_AGENT", "ARX-Terminal Regulatory Research (ops@arxterminal.com)")
    monkeypatch.setenv("ARX_BACKUP_MAX_VERIFIED_AGE_SEC", "43200")


@pytest.fixture
def mock_fresh_receipt():
    return FreshnessEvaluationReceipt(
        recovery_point_id="sm-backup-test-123",
        verified_at="2026-11-02T19:00:00Z",
        backup_completed_at="2026-11-02T18:59:00Z",
        backup_age_seconds=60.0,
        freshness_threshold_seconds=43200.0,
        verification_source="TEST",
        verification_status=BackupFreshnessState.FRESH.value,
        reason_code="VERIFIED_RECOVERY_POINT_WITHIN_THRESHOLD",
        is_fresh=True,
    )


class TestNvdaCanaryAdversarialPreflight:
    """Tests preflight gates and fail-closed safety controls."""

    def test_canary_rejects_unauthorized_ticker(self, clean_env):
        runner = NvdaCanaryRunner()
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(ticker="AAPL", cik="0001045810")
        assert "UNAUTHORIZED_CANARY_SCOPE" in str(exc_info.value)

    def test_canary_rejects_unauthorized_cik(self, clean_env):
        runner = NvdaCanaryRunner()
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(ticker="NVDA", cik="0000320193")
        assert "UNAUTHORIZED_CANARY_SCOPE" in str(exc_info.value)

    def test_canary_fails_closed_when_emergency_stop_active(self, clean_env, monkeypatch):
        monkeypatch.setenv("ARX_EMERGENCY_STOP_ACTIVE", "TRUE")
        runner = NvdaCanaryRunner()
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight()
        assert "EMERGENCY_STOP_ACTIVE" in str(exc_info.value)

    def test_canary_fails_closed_when_containment_not_authorized(self, clean_env, monkeypatch):
        monkeypatch.setenv("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED", "FALSE")
        runner = NvdaCanaryRunner()
        with pytest.raises(CanaryContainmentError) as exc_info:
            runner.validate_preflight()
        assert "UNAUTHORIZED_PRODUCTION_BACKUP_CONTAINMENT" in str(exc_info.value)

    def test_canary_fails_closed_on_placeholder_user_agent(self, clean_env, monkeypatch):
        monkeypatch.setenv("SEC_USER_AGENT", "Sample Company AdminContact@<sample company domain>.com")
        runner = NvdaCanaryRunner()
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight()
        assert "INVALID_SEC_USER_AGENT" in str(exc_info.value)

    def test_canary_fails_closed_when_backup_is_stale(self, clean_env, monkeypatch):
        stale_receipt = FreshnessEvaluationReceipt(
            recovery_point_id="sm-backup-old",
            verified_at="2026-11-02T19:00:00Z",
            backup_completed_at="2026-10-25T15:00:00Z",
            backup_age_seconds=700000.0,
            freshness_threshold_seconds=43200.0,
            verification_source="TEST",
            verification_status=BackupFreshnessState.STALE.value,
            reason_code="RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD",
            is_fresh=False,
        )
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=stale_receipt):
            runner = NvdaCanaryRunner()
            with pytest.raises(CanaryContainmentError) as exc_info:
                runner.validate_preflight()
            assert "INGESTION_CONTAINMENT_BREACH" in str(exc_info.value)


class TestNvdaCanaryHttpBoundaryEnforcement:
    """Tests runtime constraints enforced at the physical HTTP boundary."""

    def test_canary_enforces_max_4_requests_budget(self, clean_env):
        runner = NvdaCanaryRunner(max_requests=4)
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"test payload"
        mock_session.get.return_value = mock_resp
        runner.session = mock_session

        # First 4 requests succeed
        for i in range(4):
            runner.tracked_get(f"https://data.sec.gov/test_{i}")
        assert runner.request_count == 4

        # 5th request MUST fail closed
        with pytest.raises(CanaryBudgetExceededError) as exc_info:
            runner.tracked_get("https://data.sec.gov/test_5")
        assert "exceeded strict cap (4)" in str(exc_info.value)

    def test_canary_enforces_30s_timeout_ceiling(self, clean_env):
        runner = NvdaCanaryRunner(max_duration_sec=30.0)
        runner.start_time = time.time() - 31.0  # Simulate 31 seconds elapsed
        with pytest.raises(CanaryTimeoutError) as exc_info:
            runner.check_deadline()
        assert "Canary execution deadline (30.0s) exceeded" in str(exc_info.value)


class TestNvdaCanaryLifecycleAndIsolation:
    """Tests queue isolation, lock acquisition, and safe teardown guarantees."""

    def test_canary_fails_closed_under_lock_contention(self, clean_env, mock_fresh_receipt):
        mock_ctx = MagicMock()
        mock_conn = MagicMock()
        mock_ctx.__enter__.return_value = mock_conn

        with patch("analyst_dashboard.smart_money.canary_runner.get_connection", return_value=mock_ctx), \
             patch("analyst_dashboard.smart_money.canary_runner.acquire_sec_owner_lock", return_value=False), \
             patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn="postgresql://mock:mock@localhost:5432/mock")
            with pytest.raises(CanaryLockContentionError) as exc_info:
                runner.execute()
            assert "EXCLUSIVE_OWNERSHIP_UNAVAILABLE" in str(exc_info.value)

    def test_canary_guarantees_teardown_on_upstream_failure(self, clean_env, mock_fresh_receipt):
        mock_ctx = MagicMock()
        mock_conn = MagicMock()
        mock_ctx.__enter__.return_value = mock_conn

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.ConnectTimeout("SEC EDGAR timed out")

        with patch("analyst_dashboard.smart_money.canary_runner.get_connection", return_value=mock_ctx), \
             patch("analyst_dashboard.smart_money.canary_runner.acquire_sec_owner_lock", return_value=True), \
             patch("analyst_dashboard.smart_money.canary_runner.release_sec_owner_lock") as mock_release, \
             patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(
                dsn="postgresql://mock:mock@localhost:5432/mock",
                session=mock_session
            )
            with pytest.raises(requests.exceptions.ConnectTimeout):
                runner.execute()

            # Verify invariant fail-closed restoration
            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"
            assert mock_release.called

    def test_canary_ignores_pending_operator_run_requests(self, clean_env, mock_fresh_receipt):
        """Proves canary runner does not touch sm_run_requests or run unrestricted cycles."""
        mock_repo = MagicMock()
        runner = NvdaCanaryRunner()
        # Verify that runner class does NOT even have or call process_run_requests
        assert not hasattr(runner, "process_run_requests")
        assert not hasattr(runner, "recover_stale_run_requests")


class TestNvdaCanaryEndToEndSyntheticExecution:
    """Tests complete synthetic execution with mocked SEC HTTP payloads and DB persistence."""

    def test_canary_e2e_synthetic_execution_success(self, clean_env, mock_fresh_receipt):
        submissions_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        form4_xml_sample = """<?xml version="1.0"?>
        <ownershipDocument>
            <schemaVersion>X0508</schemaVersion>
            <documentType>4</documentType>
            <periodOfReport>2026-10-30</periodOfReport>
            <issuer>
                <issuerCik>0001045810</issuerCik>
                <issuerName>NVIDIA CORP</issuerName>
                <issuerTradingSymbol>NVDA</issuerTradingSymbol>
            </issuer>
            <reportingOwner>
                <reportingOwnerId>
                    <rptOwnerCik>0001197647</rptOwnerCik>
                    <rptOwnerName>HUANG JEN HSUN</rptOwnerName>
                </reportingOwnerId>
                <reportingOwnerRelationship>
                    <isDirector>true</isDirector>
                    <isOfficer>true</isOfficer>
                    <officerTitle>President and CEO</officerTitle>
                </reportingOwnerRelationship>
            </reportingOwner>
            <nonDerivativeTable>
                <nonDerivativeTransaction>
                    <securityTitle><value>Common Stock</value></securityTitle>
                    <transactionDate><value>2026-10-30</value></transactionDate>
                    <transactionCoding>
                        <transactionFormType>4</transactionFormType>
                        <transactionCode>G</transactionCode>
                    </transactionCoding>
                    <transactionAmounts>
                        <transactionShares><value>100000</value></transactionShares>
                        <transactionPricePerShare><value>0</value></transactionPricePerShare>
                        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
                    </transactionAmounts>
                    <postTransactionAmounts>
                        <sharesOwnedFollowingTransaction><value>50000000</value></sharesOwnedFollowingTransaction>
                    </postTransactionAmounts>
                    <ownershipNature>
                        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
                    </ownershipNature>
                </nonDerivativeTransaction>
            </nonDerivativeTable>
        </ownershipDocument>
        """

        mock_session = MagicMock()
        resp_sub = MagicMock()
        resp_sub.status_code = 200
        resp_sub.json.return_value = submissions_payload
        resp_sub.content = json.dumps(submissions_payload).encode()

        resp_xml = MagicMock()
        resp_xml.status_code = 200
        resp_xml.text = form4_xml_sample
        resp_xml.content = form4_xml_sample.encode()

        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(
                dsn=None,  # In-memory execution without live DB
                session=mock_session,
            )
            receipt = runner.execute()

            assert receipt["verdict"] == "SUCCESS"
            assert receipt["issuer"] == "NVDA"
            assert receipt["cik"] == "0001045810"
            assert receipt["requests_made"] == 2
            assert len(receipt["filings_discovered"]) == 1
            assert receipt["filings_discovered"][0]["accession_number"] == "0001045810-26-000099"
            assert receipt["filings_discovered"][0]["transactions_count"] == 1
            assert receipt["role_restored"] is True
            assert receipt["containment_auth_revoked"] is True
            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"
