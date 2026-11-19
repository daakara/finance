"""
Unit and Adversarial Test Suite for NVDA Bounded SEC Ingestion Canary Runner.

Validates the surgical execution path requirements (SM-ACT-014R / PR #305):
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

Acceptance Gates 1-5 (PR #305 Verification):
- Gate 1: Disposable PostgreSQL rehearsal demonstrates one cycle only
- Gate 2: Fail-closed constraints (wrong issuer, expired auth, concurrent lock, active queue, excess budget)
- Gate 3: Injected failures (HTTP 429, missing availability, parse failure, ambiguous amendment, timeout) leave 0 partial writes
- Gate 4: Existing baseline records and active snapshot remain unchanged
- Gate 5: Coverage receipt verifies coverage_status = 'PARTIAL'; narrow cycle cannot certify COMPLETE
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import gzip
import zlib
import subprocess
import http.server
import threading
import urllib.parse
from typing import Optional, Dict, Any, List
import pytest
import requests
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.canary_runner import (
    NvdaCanaryRunner,
    CanarySecurityError,
    CanaryBudgetExceededError,
    CanaryTimeoutError,
    CanaryLockContentionError,
    CanaryContainmentError,
    CanaryQueueActiveError,
    CanaryLockLostError,
    CanaryRateLimitError,
    CanaryDataIntegrityError,
    CANARY_TICKER,
    CANARY_CIK,
    CANARY_MAX_REQUESTS,
    CANARY_MAX_DURATION_SEC,
    CANARY_MAX_RPS,
    CANARY_MAX_UNSEEN_DOCS,
)
from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    FreshnessEvaluationReceipt,
    BackupFreshnessState,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    acquire_sec_owner_lock,
    release_sec_owner_lock,
    try_advisory_lock,
    LOCK_SEC_EGRESS_OWNER,
    LOCK_SEC_EGRESS_OWNER_CLASSID,
    LOCK_SEC_EGRESS_OWNER_OBJID,
)
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository


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


def make_valid_auth(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Generates an explicit, valid 10-minute UTC authorization window."""
    current = now or datetime.now(timezone.utc)
    return {
        "auth_issued_at": current - timedelta(seconds=5),
        "auth_expires_at": current + timedelta(minutes=10),
    }


@pytest.fixture
def valid_auth():
    return make_valid_auth()


@pytest.fixture(scope="module")
def pg_server_instance(tmp_path_factory):
    """Module-scoped ephemeral PostgreSQL server via pgserver."""
    pgserver = pytest.importorskip("pgserver")
    tmp_dir = tmp_path_factory.mktemp("pg_canary_suite")
    try:
        server = pgserver.get_server(tmp_dir)
    except PermissionError:
        if os.getenv("ARX_REQUIRE_LOCAL_POSTGRES_TESTS") == "true":
            raise
        pytest.skip("Runtime denies local sockets; required PostgreSQL acceptance runs in CI")
    yield server
    server.cleanup()


@pytest.fixture
def disposable_db(pg_server_instance):
    """Yields clean disposable DB connection and DSN with fresh schema."""
    psycopg2 = pytest.importorskip("psycopg2")
    from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema

    dsn = pg_server_instance.get_uri()
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    initialize_postgres_schema(conn)
    conn.autocommit = False
    try:
        yield conn, dsn
    finally:
        conn.close()


def make_sample_nvda_form4_xml(
    accession: str = "0001045810-26-000099",
    date: str = "2026-11-17",
    ticker: str = "NVDA",
    cik: str = "0001045810",
    is_amendment: bool = False,
    amendment_of: str = "",
    original_date: str = "",
    remarks: str = "",
) -> str:
    doc_type = "4/A" if is_amendment else "4"
    amend_node = f"<amendmentOfAccessionNumber>{amendment_of}</amendmentOfAccessionNumber>" if amendment_of else ""
    orig_sub_node = f"<dateOfOriginalSubmission>{original_date}</dateOfOriginalSubmission>" if original_date else ""
    remarks_node = f"<remarks>{remarks}</remarks>" if remarks else ""
    return f"""<?xml version="1.0"?>
    <ownershipDocument>
        <schemaVersion>X0508</schemaVersion>
        <documentType>{doc_type}</documentType>
        <periodOfReport>{date}</periodOfReport>
        {orig_sub_node}
        {amend_node}
        <issuer>
            <issuerCik>{cik}</issuerCik>
            <issuerName>NVIDIA CORP</issuerName>
            <issuerTradingSymbol>{ticker}</issuerTradingSymbol>
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
                <transactionDate><value>{date}</value></transactionDate>
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
        {remarks_node}
    </ownershipDocument>"""


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
        runner = NvdaCanaryRunner(max_requests=4, **make_valid_auth())
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
        runner = NvdaCanaryRunner(max_duration_sec=30.0, **make_valid_auth())
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
            runner = NvdaCanaryRunner(dsn="postgresql://mock:mock@localhost:5432/mock", **make_valid_auth())
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
                session=mock_session,
                **make_valid_auth(),
            )
            with pytest.raises(requests.exceptions.ConnectTimeout):
                runner.execute()

            # Verify invariant fail-closed restoration
            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"
            assert mock_release.called

    def test_canary_ignores_pending_operator_run_requests(self, clean_env, mock_fresh_receipt):
        """Proves canary runner does not touch sm_run_requests or run unrestricted cycles."""
        runner = NvdaCanaryRunner()
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
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        form4_xml_sample = make_sample_nvda_form4_xml(accession="0001045810-26-000099", date="2026-10-30")

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
                **make_valid_auth(),
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


class TestNvdaCanaryGate1OneCycleRehearsal:
    """Acceptance Gate 1: Disposable PostgreSQL rehearsal demonstrates one cycle only."""

    def test_gate1_disposable_postgres_rehearsal_single_cycle(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        xml_sample = make_sample_nvda_form4_xml()

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_sample, content=xml_sample.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            receipt = runner.execute()

            # Verify single-cycle execution receipt
            assert receipt["verdict"] == "SUCCESS"
            assert receipt["requests_made"] == 2
            assert receipt["coverage_status"] == "PARTIAL"
            assert receipt["coverage_receipt_id"] is not None

            # Verify PostgreSQL database state
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings WHERE ticker = 'NVDA';")
                assert cur.fetchone()[0] == 1

                cur.execute("SELECT count(*) FROM sm_source_availability_ledger;")
                assert cur.fetchone()[0] == 1

                cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
                assert cur.fetchone()[0] == 1

                cur.execute("SELECT count(*) FROM sm_reconciliation_receipts WHERE coverage_status = 'PARTIAL';")
                assert cur.fetchone()[0] == 1

                cur.execute("SELECT count(*) FROM sm_audit_events WHERE event_type = 'CANARY_INGESTION_COMPLETED';")
                assert cur.fetchone()[0] == 1

                # Verify advisory lock was released
                cur.execute("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory';")
                assert cur.fetchone()[0] == 0

            # Verify role restoration and containment revocation
            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"


class TestNvdaCanaryGate2FailClosedConstraints:
    """Acceptance Gate 2: Wrong issuer, expired authorization, concurrent execution, active queue, and excess budgets fail closed."""

    def test_gate2_wrong_issuer_ticker_fails_closed(self, clean_env):
        runner = NvdaCanaryRunner(**make_valid_auth())
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(ticker="AAPL")
        assert "UNAUTHORIZED_CANARY_SCOPE" in str(exc_info.value)

    def test_gate2_wrong_issuer_cik_fails_closed(self, clean_env):
        runner = NvdaCanaryRunner(**make_valid_auth())
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(cik="0000320193")
        assert "UNAUTHORIZED_CANARY_SCOPE" in str(exc_info.value)

    def test_gate2_xml_issuer_mismatch_fails_closed(self, clean_env, mock_fresh_receipt):
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        # XML payload contains MSFT instead of NVDA
        xml_mismatch = make_sample_nvda_form4_xml(ticker="MSFT", cik="0000789019")

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_mismatch, content=xml_mismatch.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=None, session=mock_session, **make_valid_auth())
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.execute()
            assert "SEC_FORM4_ISSUER_MISMATCH" in str(exc_info.value)

    def test_gate2_expired_authorization_fails_closed(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner()
            now = datetime.now(timezone.utc)
            past_issued = now - timedelta(minutes=20)
            past_expiry = now - timedelta(minutes=10)
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.validate_preflight(auth_issued_at=past_issued, auth_expires_at=past_expiry)
            assert "AUTHORIZATION_EXPIRED" in str(exc_info.value)

    def test_gate2_future_authorization_issued_fails_closed(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner()
            now = datetime.now(timezone.utc)
            future_issued = now + timedelta(minutes=10)
            future_expires = now + timedelta(minutes=20)
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.validate_preflight(auth_issued_at=future_issued, auth_expires_at=future_expires)
            assert "AUTHORIZATION_NOT_YET_VALID" in str(exc_info.value)

    def test_gate2_inverted_interval_fails_closed(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner()
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.validate_preflight(
                    interval_start="2026-11-20T00:00:00Z",
                    interval_end="2026-11-10T00:00:00Z",
                    **make_valid_auth(),
                )
            assert "INVALID_INTERVAL" in str(exc_info.value)

    def test_gate2_cutoff_before_interval_end_fails_closed(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner()
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.validate_preflight(
                    interval_end="2026-11-20T00:00:00Z",
                    public_availability_cutoff="2026-11-19T00:00:00Z",
                    **make_valid_auth(),
                )
            assert "INVALID_CUTOFF" in str(exc_info.value)

    def test_gate2_prospective_post_baseline_interval_success(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(
                mode="prospective",
                interval_start="2026-11-17T00:00:00Z",
                interval_end="2026-11-18T23:59:59Z",
                public_availability_cutoff="2026-11-19T00:00:00Z",
                **make_valid_auth(),
            )
            assert runner.validate_preflight() is True

    def test_gate2_prospective_mode_rejects_silent_frozen_baseline_substitution(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(
                mode="prospective",
                interval_start="2026-08-18T00:00:00Z",
                interval_end="2026-11-16T23:59:59Z",
                public_availability_cutoff="2026-11-17T00:00:00Z",
                **make_valid_auth(),
            )
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.validate_preflight()
            assert "PROSPECTIVE_INTERVAL_OVERLAPS_FROZEN_BASELINE" in str(exc_info.value)

    def test_gate2_historical_audit_mode_allows_frozen_baseline(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(
                mode="historical_audit",
                interval_start="2026-08-18T00:00:00Z",
                interval_end="2026-11-16T23:59:59Z",
                public_availability_cutoff="2026-11-17T00:00:00Z",
                **make_valid_auth(),
            )
            assert runner.validate_preflight() is True

    def test_gate2_historical_audit_mode_rejects_post_baseline_interval(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(
                mode="historical_audit",
                interval_start="2026-11-17T00:00:00Z",
                interval_end="2026-11-18T23:59:59Z",
                public_availability_cutoff="2026-11-19T00:00:00Z",
                **make_valid_auth(),
            )
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.validate_preflight()
            assert "HISTORICAL_AUDIT_OUT_OF_BOUNDS" in str(exc_info.value)

    def test_gate2_concurrent_execution_lock_contention_fails_closed(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        psycopg2 = pytest.importorskip("psycopg2")

        # Session 1 acquires the exclusive advisory lock
        conn_holder = psycopg2.connect(dsn)
        assert acquire_sec_owner_lock(conn_holder) is True

        try:
            with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
                runner = NvdaCanaryRunner(dsn=dsn, **make_valid_auth())
                with pytest.raises(CanaryLockContentionError) as exc_info:
                    runner.execute()
                assert "EXCLUSIVE_OWNERSHIP_UNAVAILABLE" in str(exc_info.value)
        finally:
            release_sec_owner_lock(conn_holder)
            conn_holder.close()

    @pytest.mark.parametrize("active_status", ["PENDING", "RUNNING", "RETRY_ELIGIBLE"])
    def test_gate2_active_queue_states_fail_closed(self, clean_env, mock_fresh_receipt, disposable_db, active_status):
        conn, dsn = disposable_db
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sm_run_requests (
                    request_id, request_type, requested_by_subject_hash, status,
                    requested_at, attempt_count, max_attempts
                ) VALUES (
                    %s, 'SCHEDULED_DISCOVERY', 'test_hash', %s, NOW(), 0, 3
                );
            """, (str(uuid.uuid4()), active_status))
        conn.commit()

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, **make_valid_auth())
            with pytest.raises(CanaryQueueActiveError) as exc_info:
                runner.validate_preflight(conn=conn, **make_valid_auth())
            assert "ACTIVE_QUEUE_REQUESTS_PRESENT" in str(exc_info.value)

    def test_gate2_unseen_docs_budget_exceeded_fails_closed(self, clean_env, mock_fresh_receipt):
        # SEC catalog advertises 4 unseen Form 4 filings (limit is 3)
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4", "4", "4", "4"],
                    "accessionNumber": [
                        "0001045810-26-000101",
                        "0001045810-26-000102",
                        "0001045810-26-000103",
                        "0001045810-26-000104",
                    ],
                    "filingDate": ["2026-11-17", "2026-11-17", "2026-11-17", "2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"] * 4,
                    "primaryDocument": ["doc1.xml", "doc2.xml", "doc3.xml", "doc4.xml"],
                }
            }
        }
        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        mock_session.get.return_value = resp_sub

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=None, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryBudgetExceededError) as exc_info:
                runner.execute()
            assert "SEC_NEW_FILINGS_EXCEED_ISSUER_BUDGET" in str(exc_info.value)


class TestNvdaCanaryGate3InjectedFailuresAndRollback:
    """Acceptance Gate 3: Injected failures, timeout, and interruption leave no partial writes or residual authorization."""

    def test_gate3_injected_http_429_fails_closed_and_rolls_back(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        mock_session = MagicMock()
        resp_429 = MagicMock(status_code=429, content=b"Too Many Requests")
        mock_session.get.return_value = resp_429

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryRateLimitError) as exc_info:
                runner.execute()
            assert "SEC_RATE_LIMIT_EXCEEDED" in str(exc_info.value)

            # Proves zero partial database writes
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings;")
                assert cur.fetchone()[0] == 0
                cur.execute("SELECT count(*) FROM sm_reconciliation_receipts;")
                assert cur.fetchone()[0] == 0
                cur.execute("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory';")
                assert cur.fetchone()[0] == 0

            # Proves no residual authorization
            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"

    def test_gate3_missing_acceptance_datetime_fails_closed_and_rolls_back(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": [""],  # Empty availability timestamp
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        mock_session.get.return_value = resp_sub

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryDataIntegrityError) as exc_info:
                runner.execute()
            assert "SEC_PUBLIC_AVAILABILITY_MISSING" in str(exc_info.value)

            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings;")
                assert cur.fetchone()[0] == 0

    def test_gate3_corrupted_xml_parse_failure_fails_closed_and_rolls_back(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        corrupted_xml = "<XML><ownershipDocument><bad>unclosed"

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=corrupted_xml, content=corrupted_xml.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryDataIntegrityError) as exc_info:
                runner.execute()
            assert "SEC_FORM4_PARSE_FAILED" in str(exc_info.value)

            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings;")
                assert cur.fetchone()[0] == 0

    def test_gate3_ambiguous_amendment_scope_fails_closed_and_rolls_back(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        # Pre-seed a base filing with 2 transactions
        base_acc = "0001045810-26-000080"
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, '2026-09-01', '2026-09-01T21:00:00Z', '<xml></xml>', 'base_hash', 'https://www.sec.gov/base.xml');
            """, (base_acc, CANARY_CIK, CANARY_TICKER, "NVIDIA CORP"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type
                ) VALUES ('base_tx1', %s, '2026-09-01', 'S', 100, 100, 10000, 'Test Insider', 'Director', 'D'),
                         ('base_tx2', %s, '2026-09-01', 'S', 200, 100, 20000, 'Test Insider', 'Director', 'D');
            """, (base_acc, base_acc))
        conn.commit()

        # Amendment has conflicting / ambiguous remarks ("corrects both line 1 and line 2")
        amend_acc = "0001045810-26-000085"
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4/A"],
                    "accessionNumber": [amend_acc],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_amend.xml"],
                }
            }
        }
        amend_xml = make_sample_nvda_form4_xml(
            accession=amend_acc,
            date="2026-11-17",
            is_amendment=True,
            amendment_of=base_acc,
            remarks="This amendment is filed to correct the first transaction and also correct the second transaction.",
        )

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=amend_xml, content=amend_xml.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryDataIntegrityError) as exc_info:
                runner.execute()
            assert "AMENDMENT_SCOPE_AMBIGUOUS" in str(exc_info.value)

            # Verify amendment was rolled back
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number = %s;", (amend_acc,))
                assert cur.fetchone()[0] == 0

    def test_gate3_lock_loss_before_commit_rolls_back(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        xml_sample = make_sample_nvda_form4_xml()

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_sample, content=xml_sample.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        # Simulate lock check returning True during initial claim, but False right before commit
        lock_checks = [True, False]

        def dynamic_lock_check(connection):
            if lock_checks:
                return lock_checks.pop(0)
            return False

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt), \
             patch.object(NvdaCanaryRunner, "_has_verified_owner_lock", side_effect=dynamic_lock_check):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryLockLostError) as exc_info:
                runner.execute()
            assert "ADVISORY_LOCK_LOST" in str(exc_info.value)

            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings;")
                assert cur.fetchone()[0] == 0


class TestNvdaCanaryGate4BaselineAndSnapshotPreservation:
    """Acceptance Gate 4: Existing baseline records and active snapshot remain unchanged."""

    def test_gate4_frozen_baseline_and_active_snapshot_preserved(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db

        # 1. Pre-seed 16 baseline raw filings
        with conn.cursor() as cur:
            for i in range(1, 17):
                acc = f"0001045810-26-{i:06d}"
                cur.execute("""
                    INSERT INTO form4_raw_filings (
                        accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url
                    ) VALUES (%s, %s, %s, %s, '2026-09-01', '2026-09-01T21:00:00Z', '<xml></xml>', %s, 'https://www.sec.gov/base.xml');
                """, (acc, CANARY_CIK, CANARY_TICKER, "NVIDIA CORP", f"hash_{i}"))

                cur.execute("""
                    INSERT INTO sm_source_availability_ledger (
                        accession_number, public_availability_at, source, source_reference
                    ) VALUES (%s, '2026-09-01T21:00:00Z', 'SEC_EDGAR_LIVE_PROD', 'test_url');
                """, (acc,))

            # 2. Pre-seed 36 baseline transactions across the 16 accessions
            for tx_i in range(1, 37):
                parent_acc = f"0001045810-26-{(tx_i % 16) + 1:06d}"
                cur.execute("""
                    INSERT INTO form4_parsed_transactions (
                        transaction_id, accession_number, transaction_date, transaction_code,
                        shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type
                    ) VALUES (%s, %s, '2026-09-01', 'S', 100, 100, 10000, 'Baseline Insider', 'Director', 'D');
                """, (f"tx_baseline_{tx_i}", parent_acc))

            # 3. Pre-seed Parker amendment linkage
            cur.execute("""
                INSERT INTO sm_amendments (
                    amendment_accession, base_accession, amendment_number, resolution_status, resolved_at
                ) VALUES ('0001045810-26-000085', '0001045810-26-000080', 1, 'PARTIAL_CORRECTION', NOW());
            """)

            # 4. Pre-seed active snapshot in scanner_active_snapshots
            cur.execute("""
                INSERT INTO scanner_snapshots VALUES (
                    'SNAP_SM_1792939989_48f30db6', 'SMART_MONEY', 'run-m1', '1.0', '1.0', '1.0', '1.0',
                    '1.0', '1.0', '1.0', '308f5392', 'fingerprint_sm', '2026-10-01T20:00:00Z',
                    '2026-10-01T20:00:00Z', 'AVAILABLE', 'UNIV-1', 10, 1,
                    '[{"symbol":"NVDA"}]', '{}', '{}', 'PUBLISH'
                );
            """)
            cur.execute("""
                INSERT INTO scanner_active_snapshots (
                    scan_type, active_snapshot_id, published_at, published_by
                ) VALUES ('SMART_MONEY', 'SNAP_SM_1792939989_48f30db6', NOW(), 'ORCHESTRATOR');
            """)
        conn.commit()

        # Execute runner on a new 17th accession
        new_acc = "0001045810-26-000099"
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": [new_acc],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        xml_sample = make_sample_nvda_form4_xml(accession=new_acc)

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_sample, content=xml_sample.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            receipt = runner.execute()
            assert receipt["verdict"] == "SUCCESS"

            # Verify that original 16 baseline accessions are completely intact
            with conn.cursor() as cur:
                for i in range(1, 17):
                    acc = f"0001045810-26-{i:06d}"
                    cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number = %s;", (acc,))
                    assert cur.fetchone()[0] == 1
                    cur.execute("SELECT count(*) FROM sm_source_availability_ledger WHERE accession_number = %s;", (acc,))
                    assert cur.fetchone()[0] == 1

                # Total is now 16 baseline + 1 new = 17
                cur.execute("SELECT count(*) FROM form4_raw_filings WHERE ticker = 'NVDA';")
                assert cur.fetchone()[0] == 17

                # 36 baseline transactions + 1 new = 37
                cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
                assert cur.fetchone()[0] == 37

                # Parker amendment unchanged
                cur.execute("SELECT resolution_status, base_accession FROM sm_amendments WHERE amendment_accession = '0001045810-26-000085';")
                p_row = cur.fetchone()
                assert p_row[0] == "PARTIAL_CORRECTION"
                assert p_row[1] == "0001045810-26-000080"

                # Active snapshot in scanner_active_snapshots is STRICTLY PRESERVED
                cur.execute("SELECT active_snapshot_id FROM scanner_active_snapshots WHERE scan_type = 'SMART_MONEY';")
                snap_row = cur.fetchone()
                assert snap_row[0] == "SNAP_SM_1792939989_48f30db6"
                cur.execute("SELECT count(*) FROM scanner_active_snapshots;")
                assert cur.fetchone()[0] == 1


class TestNvdaCanaryGate5CoverageCertification:
    """Acceptance Gate 5: Coverage receipt verifies coverage_status = 'PARTIAL'; narrow cycle cannot certify COMPLETE."""

    def test_gate5_coverage_receipt_certifies_partial_status(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        xml_sample = make_sample_nvda_form4_xml()

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_sample, content=xml_sample.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            receipt = runner.execute()

            assert receipt["coverage_status"] == "PARTIAL"
            r_id = receipt["coverage_receipt_id"]
            assert r_id is not None

            with conn.cursor() as cur:
                cur.execute("SELECT * FROM sm_reconciliation_receipts WHERE receipt_id = %s;", (r_id,))
                colnames = [desc[0] for desc in cur.description]
                rec = dict(zip(colnames, cur.fetchone()))

            assert rec is not None
            assert rec["coverage_status"] == "PARTIAL"
            assert rec["accounted_count"] == 1
            assert rec["material_gap_count"] == 1

            meta = rec["failure_reasons"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            assert meta["ticker"] == "NVDA"
            assert meta["issuer_cik"] == "0001045810"
            assert meta["accession_inventory"] == ["0001045810-26-000099"]
            assert len(meta["material_gaps"]) == 1
            assert meta["material_gaps"][0]["gap_type"] == "SINGLE_SHOT_CANARY_CYCLE"

    def test_gate5_narrow_cycle_cannot_certify_complete_coverage(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        repo = PostgresSmartMoneyRepository(dsn=dsn)

        # Calling record_issuer_coverage_receipt with COMPLETE status but empty inventory/catalog proof fails closed
        with pytest.raises(RuntimeError) as exc_info:
            repo.record_issuer_coverage_receipt(
                ticker="NVDA",
                issuer_cik="0001045810",
                interval_start="2026-08-18T00:00:00Z",
                interval_end="2026-11-16T23:59:59Z",
                certified_watermark="2026-11-17T00:00:00Z",
                accession_inventory=[],
                coverage_status="COMPLETE",
                conn=conn,
            )
        assert "ZERO_ACTIVITY_UNPROVED" in str(exc_info.value)


class TestNvdaCanaryAdvisoryLockDecompositionAndLifecycle:
    """Acceptance: Advisory lock constants decomposition and lifecycle testing."""

    def test_advisory_lock_exact_constants_and_decomposition(self):
        assert LOCK_SEC_EGRESS_OWNER == 401007311509093371
        assert LOCK_SEC_EGRESS_OWNER_CLASSID == 93366790
        assert LOCK_SEC_EGRESS_OWNER_OBJID == 1926593531
        # 64-bit bigint decomposes to 32-bit high and low unsigned parts
        assert (LOCK_SEC_EGRESS_OWNER >> 32) & 0xFFFFFFFF == 93366790
        assert LOCK_SEC_EGRESS_OWNER & 0xFFFFFFFF == 1926593531
        assert ((93366790 << 32) | 1926593531) == 401007311509093371

    def test_advisory_lock_acquisition_and_ownership_verification_in_pg_locks(self, disposable_db):
        conn, dsn = disposable_db
        runner = NvdaCanaryRunner(dsn=dsn, **make_valid_auth())

        # Initially no lock held
        assert runner._has_verified_owner_lock(conn) is False

        # Acquire lock on this session
        assert acquire_sec_owner_lock(conn) is True

        # Verify exact decomposition exists in pg_locks
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pid, classid, objid, objsubid, granted
                FROM pg_locks
                WHERE locktype = 'advisory'
                  AND pid = pg_backend_pid()
                  AND classid = %s
                  AND objid = %s
                  AND objsubid = 1;
            """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
            row = cur.fetchone()
            assert row is not None
            assert row[1] == 93366790
            assert row[2] == 1926593531
            assert row[3] == 1
            assert row[4] is True

        # Active ownership verification by runner
        assert runner._has_verified_owner_lock(conn) is True

        # Explicit release
        assert release_sec_owner_lock(conn) is True

        # Verify pg_locks no longer holds this entry
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) FROM pg_locks
                WHERE locktype = 'advisory'
                  AND classid = %s
                  AND objid = %s;
            """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
            assert cur.fetchone()[0] == 0

        # Active ownership verification now returns False
        assert runner._has_verified_owner_lock(conn) is False

    def test_advisory_lock_contention_using_exact_lock(self, disposable_db):
        psycopg2 = pytest.importorskip("psycopg2")
        conn1, dsn = disposable_db
        conn2 = psycopg2.connect(dsn)
        try:
            # conn1 acquires lock 401007311509093371
            assert acquire_sec_owner_lock(conn1) is True

            # conn2 attempts acquisition and fails nonblockingly
            assert acquire_sec_owner_lock(conn2) is False
            assert try_advisory_lock(conn2, 401007311509093371) is False

            # conn1 releases lock
            assert release_sec_owner_lock(conn1) is True

            # conn2 can now acquire the lock
            assert acquire_sec_owner_lock(conn2) is True
            assert release_sec_owner_lock(conn2) is True
        finally:
            conn2.close()

    def test_advisory_lock_loss_detection_using_exact_lock(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        xml_sample = make_sample_nvda_form4_xml()

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_sample, content=xml_sample.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        original_verify = NvdaCanaryRunner._has_verified_owner_lock
        verification_counts = []

        def drop_lock_on_second_check(self_runner, c):
            res = original_verify(self_runner, c)
            verification_counts.append(res)
            if len(verification_counts) == 2:
                # Drop advisory lock directly on session right before commit check
                release_sec_owner_lock(c)
                return original_verify(self_runner, c)
            return res

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt), \
             patch.object(NvdaCanaryRunner, "_has_verified_owner_lock", drop_lock_on_second_check):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryLockLostError) as exc_info:
                runner.execute()
            assert "ADVISORY_LOCK_LOST" in str(exc_info.value)

            # Proves complete rollback: 0 records inserted
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings;")
                assert cur.fetchone()[0] == 0

    def test_advisory_lock_cleanup_using_exact_lock(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        xml_sample = make_sample_nvda_form4_xml()

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_sample, content=xml_sample.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            receipt = runner.execute()
            assert receipt["verdict"] == "SUCCESS"
            assert receipt.get("lock_released") is True

            # Verify in pg_locks that lock is completely cleaned up
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT count(*) FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND classid = %s
                      AND objid = %s;
                """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
                assert cur.fetchone()[0] == 0


class TestNvdaCanaryInterruptionAndNetworkTimeout:
    """Acceptance: Demonstration of interruption cleanup and real wall-clock network timeouts."""

    def test_interruption_cleanup_on_keyboard_interrupt(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        # Trigger KeyboardInterrupt during the raw filing fetch
        mock_session.get.side_effect = [resp_sub, KeyboardInterrupt("Operator SIGINT interruption")]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(KeyboardInterrupt):
                runner.execute()

            # Guaranteed transactional rollback: 0 records
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings;")
                assert cur.fetchone()[0] == 0
                cur.execute("""
                    SELECT count(*) FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND classid = %s
                      AND objid = %s;
                """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
                assert cur.fetchone()[0] == 0

            # Invariant teardown: role reverted to STANDBY, auth revoked
            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"

    def test_real_wall_clock_timeout_blocked_network_call(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db

        def simulate_blocked_network_hang(*args, **kwargs):
            time.sleep(0.55)
            raise requests.exceptions.Timeout("Connection timed out after deadline")

        mock_session = MagicMock()
        mock_session.get.side_effect = simulate_blocked_network_hang

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, max_duration_sec=0.5, **make_valid_auth())
            with pytest.raises(CanaryTimeoutError) as exc_info:
                runner.execute()
            assert "TIMEOUT" in str(exc_info.value)

            # Full rollback and cleanup verified
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings;")
                assert cur.fetchone()[0] == 0
                cur.execute("""
                    SELECT count(*) FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND classid = %s
                      AND objid = %s;
                """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
                assert cur.fetchone()[0] == 0

            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"


class TestNvdaCanaryNonOverridableCeilingsAndAuth:
    """Tests non-overridable ceilings and explicit authorization timestamps."""

    @pytest.mark.parametrize("bad_requests", [5, 10, 0, -1, True])
    def test_rejects_invalid_or_excessive_max_requests(self, bad_requests):
        with pytest.raises(CanarySecurityError) as exc_info:
            NvdaCanaryRunner(max_requests=bad_requests, **make_valid_auth())
        assert "CANARY_CEILING_EXCEEDED" in str(exc_info.value)

    @pytest.mark.parametrize("bad_rps", [5.1, 10.0, 0.0, -1.0, float("nan"), float("inf"), True])
    def test_rejects_invalid_or_excessive_max_rps(self, bad_rps):
        with pytest.raises(CanarySecurityError) as exc_info:
            NvdaCanaryRunner(max_rps=bad_rps, **make_valid_auth())
        assert "CANARY_CEILING_EXCEEDED" in str(exc_info.value)

    @pytest.mark.parametrize("bad_unseen", [4, 10, 0, -1, True])
    def test_rejects_invalid_or_excessive_max_unseen_docs(self, bad_unseen):
        with pytest.raises(CanarySecurityError) as exc_info:
            NvdaCanaryRunner(max_unseen_docs=bad_unseen, **make_valid_auth())
        assert "CANARY_CEILING_EXCEEDED" in str(exc_info.value)

    @pytest.mark.parametrize("bad_dur", [30.1, 60.0, 0.0, -1.0, float("nan"), float("inf"), True])
    def test_rejects_invalid_or_excessive_max_duration(self, bad_dur):
        with pytest.raises(CanarySecurityError) as exc_info:
            NvdaCanaryRunner(max_duration_sec=bad_dur, **make_valid_auth())
        assert "CANARY_CEILING_EXCEEDED" in str(exc_info.value)

    def test_rejects_missing_authorization_timestamps(self, clean_env, mock_fresh_receipt):
        runner = NvdaCanaryRunner()
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.validate_preflight()
            assert "AUTHORIZATION_TIMESTAMPS_REQUIRED" in str(exc_info.value)

    def test_rejects_auth_window_exceeding_15_minutes(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(CanarySecurityError) as exc_info:
            NvdaCanaryRunner(
                auth_issued_at=now,
                auth_expires_at=now + timedelta(seconds=901),
            )
        assert "AUTHORIZATION_WINDOW_EXCEEDED" in str(exc_info.value)

    def test_rejects_inverted_auth_window(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(CanarySecurityError) as exc_info:
            NvdaCanaryRunner(
                auth_issued_at=now + timedelta(seconds=60),
                auth_expires_at=now,
            )
        assert "INVALID_AUTHORIZATION_WINDOW" in str(exc_info.value)

    def test_active_expiry_check_before_egress_and_commit(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        now = datetime.now(timezone.utc)
        runner = NvdaCanaryRunner(
            dsn=dsn,
            auth_issued_at=now - timedelta(minutes=5),
            auth_expires_at=now + timedelta(seconds=1),
        )
        runner.auth_expires_at = now - timedelta(seconds=1)
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.check_authorization_not_expired()
        assert "AUTHORIZATION_EXPIRED" in str(exc_info.value)


class TestNvdaCanaryGuaranteedCleanupAndQueueInterruption:
    """Tests guaranteed cleanup on queue check failure and immediate interruption after lock acquisition."""

    def test_queue_check_failure_under_lock_releases_lock_and_revokes_auth(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sm_run_requests (
                    request_id, request_type, requested_by_subject_hash, status,
                    requested_at, attempt_count, max_attempts
                ) VALUES (
                    %s, 'SCHEDULED_DISCOVERY', 'test_hash', 'PENDING', NOW(), 0, 3
                );
            """, (str(uuid.uuid4()),))
        conn.commit()

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, **make_valid_auth())
            with pytest.raises(CanaryQueueActiveError) as exc_info:
                runner.execute()
            assert "ACTIVE_QUEUE_REQUESTS_PRESENT" in str(exc_info.value)

            with conn.cursor() as cur:
                cur.execute("""
                    SELECT count(*) FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND classid = %s
                      AND objid = %s;
                """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
                assert cur.fetchone()[0] == 0

            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"

    def test_interruption_immediately_after_lock_acquisition(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db

        def interrupt_after_lock(conn_arg):
            raise KeyboardInterrupt("Simulated operator interrupt right after lock claim")

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt),              patch.object(NvdaCanaryRunner, "check_queue_isolation", side_effect=interrupt_after_lock):
            runner = NvdaCanaryRunner(dsn=dsn, **make_valid_auth())
            with pytest.raises(KeyboardInterrupt):
                runner.execute()

            with conn.cursor() as cur:
                cur.execute("""
                    SELECT count(*) FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND classid = %s
                      AND objid = %s;
                """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
                assert cur.fetchone()[0] == 0

            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"


class TestNvdaCanaryFilingWindowFilteringAndNoFallback:
    """Tests interval and cutoff filtering before unseen counts, and removal of newest-filing fallback."""

    def test_out_of_window_filings_excluded_and_never_downloaded(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000050"],
                    "filingDate": ["2026-11-10"],
                    "acceptanceDateTime": ["2026-11-10T21:00:00Z"],
                    "primaryDocument": ["form4_50.xml"],
                }
            }
        }
        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        mock_session.get.return_value = resp_sub

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryDataIntegrityError) as exc_info:
                runner.execute()
            assert "NO_UNSEEN_ELIGIBLE_FILINGS" in str(exc_info.value)
            assert mock_session.get.call_count == 1
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings;")
                assert cur.fetchone()[0] == 0

    def test_post_cutoff_filings_excluded_and_never_downloaded(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000099"],
                    "filingDate": ["2026-11-18"],
                    "acceptanceDateTime": ["2026-11-19T05:00:00Z"],  # Post cutoff
                    "primaryDocument": ["form4_99.xml"],
                }
            }
        }
        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        mock_session.get.return_value = resp_sub

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryDataIntegrityError) as exc_info:
                runner.execute()
            assert "NO_UNSEEN_ELIGIBLE_FILINGS" in str(exc_info.value)
            assert mock_session.get.call_count == 1

    def test_mixed_catalog_only_downloads_and_persists_in_window_filing(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4", "4", "4"],
                    "accessionNumber": [
                        "0001045810-26-000010",
                        "0001045810-26-000020",
                        "0001045810-26-000030",
                    ],
                    "filingDate": ["2026-11-10", "2026-11-17", "2026-11-18"],
                    "acceptanceDateTime": [
                        "2026-11-10T21:00:00Z",
                        "2026-11-17T21:00:00Z",
                        "2026-11-20T10:00:00Z",
                    ],
                    "primaryDocument": ["f10.xml", "f20.xml", "f30.xml"],
                }
            }
        }
        xml_in_window = make_sample_nvda_form4_xml(accession="0001045810-26-000020", date="2026-11-17")

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_in_window, content=xml_in_window.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            receipt = runner.execute()

            assert receipt["verdict"] == "SUCCESS"
            assert receipt["filings_discovered"][0]["accession_number"] == "0001045810-26-000020"

            with conn.cursor() as cur:
                cur.execute("SELECT accession_number FROM form4_raw_filings;")
                rows = cur.fetchall()
                assert len(rows) == 1
                assert rows[0][0] == "0001045810-26-000020"


class TestNvdaCanaryRepositoryContractsAndLineage:
    """Tests repository contract reuse: lineage, transitions, source-hash conflicts, and mapping persistence."""

    def test_amendment_base_resolution_via_repo_contract(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        base_acc = "0001045810-26-000080"
        amend_acc = "0001045810-26-000085"

        base_xml = make_sample_nvda_form4_xml(accession=base_acc, date="2026-11-16")
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, '2026-11-16', '2026-11-16T21:00:00Z', %s, 'base_hash', 'https://www.sec.gov/base.xml');
            """, (base_acc, CANARY_CIK, CANARY_TICKER, "NVIDIA CORP", base_xml))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type
                ) VALUES ('base_tx1', %s, '2026-11-16', 'S', 100, 100, 10000, 'HUANG JEN HSUN', 'President and CEO', 'D');
            """, (base_acc,))
        conn.commit()

        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4/A"],
                    "accessionNumber": [amend_acc],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_amend.xml"],
                }
            }
        }
        # Form 4/A without explicit amendment_of, testing repository resolution
        amend_xml = make_sample_nvda_form4_xml(
            accession=amend_acc,
            date="2026-11-16",
            is_amendment=True,
            amendment_of="",
            original_date="2026-11-16",
            remarks="This amendment is filed solely to correct the transaction price of the first transaction.",
        )

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=amend_xml, content=amend_xml.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            receipt = runner.execute()
            assert receipt["verdict"] == "SUCCESS"

            with conn.cursor() as cur:
                cur.execute("SELECT base_accession, resolution_status FROM sm_amendments WHERE amendment_accession = %s;", (amend_acc,))
                row = cur.fetchone()
                assert row is not None
                assert row[0] == base_acc
                assert row[1] == "PARTIAL_CORRECTION"

    def test_unresolvable_amendment_base_fails_closed(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        amend_acc = "0001045810-26-000085"
        sub_payload = {
            "filings": {
                "recent": {
                    "form": ["4/A"],
                    "accessionNumber": [amend_acc],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "primaryDocument": ["form4_amend.xml"],
                }
            }
        }
        amend_xml = make_sample_nvda_form4_xml(
            accession=amend_acc,
            date="2026-11-17",
            is_amendment=True,
            amendment_of="",
            remarks="Amendment with unresolvable base filing.",
        )

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=amend_xml, content=amend_xml.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, **make_valid_auth())
            with pytest.raises(CanaryDataIntegrityError) as exc_info:
                runner.execute()
            assert "AMENDMENT_BASE_UNRESOLVED" in str(exc_info.value)

            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number = %s;", (amend_acc,))
                assert cur.fetchone()[0] == 0

    def test_source_content_hash_conflict_triggers_rollback(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        acc = "0001045810-26-000099"
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, '2026-11-17', '2026-11-17T21:00:00Z', '<xml>prior</xml>', 'hash_orig_different', 'https://sec.gov');
            """, (acc, CANARY_CIK, CANARY_TICKER, "NVIDIA CORP"))
        conn.commit()

        xml_sample = make_sample_nvda_form4_xml(accession=acc, date="2026-11-17")
        repo = PostgresSmartMoneyRepository(dsn=dsn)
        with pytest.raises(RuntimeError) as exc_info:
            repo.save_raw_filing(
                {
                    "accession_number": acc,
                    "cik": CANARY_CIK,
                    "ticker": CANARY_TICKER,
                    "company_name": "NVIDIA CORP",
                    "filing_date": "2026-11-17",
                    "accepted_datetime": "2026-11-17T21:00:00Z",
                    "xml_content": xml_sample,
                    "content_hash": "hash_tampered_new",
                    "source_url": "https://sec.gov",
                },
                conn=conn,
            )
        assert "SOURCE_INTEGRITY_CONFLICT" in str(exc_info.value)
        conn.rollback()

    def test_persisted_amendment_mapping_readable_by_replay_path(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        base_acc = "0001045810-26-000080"
        amend_acc = "0001045810-26-000085"

        repo = PostgresSmartMoneyRepository(dsn=dsn)

        # 1. Base filing with 2 transactions: tx_A and tx_B
        repo.save_raw_filing(
            {
                "accession_number": base_acc,
                "cik": CANARY_CIK,
                "ticker": CANARY_TICKER,
                "company_name": "NVIDIA CORP",
                "filing_date": "2026-11-16",
                "accepted_datetime": "2026-11-16T21:00:00Z",
                "xml_content": "<xml></xml>",
                "content_hash": "base_hash_1",
                "source_url": "https://sec.gov",
            },
            conn=conn,
        )
        base_tx_A = {
            "transaction_id": "tx_A",
            "accession_number": base_acc,
            "transaction_date": "2026-11-16",
            "transaction_code": "S",
            "shares": 100,
            "price_per_share": 100.0,
            "total_value_usd": 10000.0,
            "reporting_owner_name": "HUANG JEN HSUN",
            "reporting_owner_role": "Director",
            "ownership_type": "D",
        }
        base_tx_B = {
            "transaction_id": "tx_B",
            "accession_number": base_acc,
            "transaction_date": "2026-11-16",
            "transaction_code": "S",
            "shares": 200,
            "price_per_share": 100.0,
            "total_value_usd": 20000.0,
            "reporting_owner_name": "HUANG JEN HSUN",
            "reporting_owner_role": "Director",
            "ownership_type": "D",
        }
        repo.save_parsed_transactions([base_tx_A, base_tx_B], conn=conn)

        # 2. Amendment correcting tx_B
        repo.save_raw_filing(
            {
                "accession_number": amend_acc,
                "cik": CANARY_CIK,
                "ticker": CANARY_TICKER,
                "company_name": "NVIDIA CORP",
                "filing_date": "2026-11-17",
                "accepted_datetime": "2026-11-17T21:00:00Z",
                "xml_content": "<xml>amend</xml>",
                "content_hash": "amend_hash_1",
                "source_url": "https://sec.gov",
            },
            conn=conn,
        )
        amend_tx_B = {
            "transaction_id": "tx_B_amended",
            "accession_number": amend_acc,
            "transaction_date": "2026-11-16",
            "transaction_code": "S",
            "shares": 200,
            "price_per_share": 120.0,
            "total_value_usd": 24000.0,
            "reporting_owner_name": "HUANG JEN HSUN",
            "reporting_owner_role": "Director",
            "ownership_type": "D",
        }
        repo.save_parsed_transactions([amend_tx_B], conn=conn)
        repo.save_amendment(
            amendment_accession=amend_acc,
            base_accession=base_acc,
            amendment_number=1,
            resolution_status="PARTIAL_CORRECTION",
            conn=conn,
        )
        repo.record_amendment_transaction_mappings(
            base_accession=base_acc,
            amendment_accession=amend_acc,
            mappings=[{
                "base_transaction_id": "tx_B",
                "amendment_transaction_id": "tx_B_amended",
                "effect_type": "REPLACE_TRANSACTION",
                "confidence": 1.0,
                "rule_matched": "VERIFIED_SOURCE_POSITION_MAPPING",
            }],
            conn=conn,
        )

        # Ledger entries
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (
                    accession_number, public_availability_at, source, source_reference
                ) VALUES
                    (%s, '2026-11-16T21:00:00Z', 'SEC_EDGAR_LIVE_PROD', 'test_base_url'),
                    (%s, '2026-11-17T21:00:00Z', 'SEC_EDGAR_LIVE_PROD', 'test_amend_url');
            """, (base_acc, amend_acc))
        conn.commit()

        # 3. Read back via production replay path: get_bulk_transactions_as_of
        as_of_before = "2026-11-16T23:59:59Z"
        res_before = repo.get_bulk_transactions_as_of(
            as_of_watermark=as_of_before,
            symbols=["NVDA"],
            min_date="2026-11-01",
            conn=conn,
        )
        txs_before = res_before.get("NVDA", [])
        tx_map_before = {t["transaction_id"]: t for t in txs_before}
        assert "tx_A" in tx_map_before
        assert "tx_B" in tx_map_before
        assert float(tx_map_before["tx_B"]["price_per_share"]) == 100.0

        # After amendment: tx_A remains intact, tx_B replaced by tx_B_amended
        as_of_after = "2026-11-17T23:59:59Z"
        res_after = repo.get_bulk_transactions_as_of(
            as_of_watermark=as_of_after,
            symbols=["NVDA"],
            min_date="2026-11-01",
            conn=conn,
        )
        txs_after = res_after.get("NVDA", [])
        tx_map_after = {t["transaction_id"]: t for t in txs_after}
        assert "tx_A" in tx_map_after
        assert float(tx_map_after["tx_A"]["price_per_share"]) == 100.0
        assert "tx_B" not in tx_map_after
        assert "tx_B_amended" in tx_map_after
        assert float(tx_map_after["tx_B_amended"]["price_per_share"]) == 120.0


class TestNvdaCanaryStalledHttpServerDeadline:
    """Acceptance: Wall-clock timeout enforcement against an actual stalled HTTP server over real TCP sockets."""

    def test_stalled_http_server_deadline_enforced_over_tcp(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db

        class StalledHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                # Stall longer than timeout (2.0s vs 0.5s timeout)
                time.sleep(2.0)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"delayed payload")

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), StalledHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        class LocalRedirectAdapter(requests.adapters.HTTPAdapter):
            def send(self, req, **kwargs):
                parsed = urllib.parse.urlparse(req.url)
                req.url = urllib.parse.urlunparse(parsed._replace(scheme="http", netloc=f"127.0.0.1:{port}"))
                return super().send(req, **kwargs)

        session = requests.Session()
        session.mount("https://", LocalRedirectAdapter())

        t0 = time.time()
        try:
            with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
                runner = NvdaCanaryRunner(
                    dsn=dsn,
                    session=session,
                    max_duration_sec=0.5,
                    **make_valid_auth(),
                )
                with pytest.raises(CanaryTimeoutError) as exc_info:
                    runner.execute()
                assert "TIMEOUT" in str(exc_info.value).upper() or "DEADLINE" in str(exc_info.value).upper()

                elapsed = time.time() - t0
                # Must abort near 0.5s, well before the 2.0s server sleep finishes
                assert elapsed < 1.8

                # Guaranteed transactional rollback: 0 records
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM form4_raw_filings;")
                    assert cur.fetchone()[0] == 0
                    cur.execute("""
                        SELECT count(*) FROM pg_locks
                        WHERE locktype = 'advisory'
                          AND classid = %s
                          AND objid = %s;
                    """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
                    assert cur.fetchone()[0] == 0

                assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
                assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"
        finally:
            server.shutdown()
            server.server_close()


class TestNvdaCanaryTricklingServerAndPreCommitDeadline:
    """Acceptance: Wall-clock timeout enforcement against continuously trickling response and pre-commit check."""

    def test_reproduction_trickling_server_rejected_within_deadline(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db

        class TricklingHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                # 22 chunks of 1 byte, sleeping 0.05s between chunks = 1.10s total
                for _ in range(22):
                    time.sleep(0.05)
                    self.wfile.write(b"x")
                    self.wfile.flush()

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), TricklingHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        class LocalRedirectAdapter(requests.adapters.HTTPAdapter):
            def send(self, req, **kwargs):
                parsed = urllib.parse.urlparse(req.url)
                req.url = urllib.parse.urlunparse(parsed._replace(scheme="http", netloc=f"127.0.0.1:{port}"))
                return super().send(req, **kwargs)

        session = requests.Session()
        session.mount("https://", LocalRedirectAdapter())

        t0 = time.time()
        try:
            with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
                runner = NvdaCanaryRunner(
                    dsn=dsn,
                    session=session,
                    max_duration_sec=0.35,
                    **make_valid_auth(),
                )
                with pytest.raises(CanaryTimeoutError) as exc_info:
                    runner.execute()
                assert "DEADLINE" in str(exc_info.value).upper() or "TIMEOUT" in str(exc_info.value).upper()

                elapsed = time.time() - t0
                # Must abort near 0.35s, well before the 1.10s trickling finishes
                assert elapsed < 0.75, f"Execution took {elapsed:.2f}s, did not enforce 0.35s deadline on trickling transfer!"

                # Guaranteed transactional rollback: 0 records
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM form4_raw_filings;")
                    assert cur.fetchone()[0] == 0
                    cur.execute("""
                        SELECT count(*) FROM pg_locks
                        WHERE locktype = 'advisory'
                          AND classid = %s
                          AND objid = %s;
                    """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
                    assert cur.fetchone()[0] == 0

                assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
                assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"
        finally:
            server.shutdown()
            server.server_close()

    def test_deadline_enforced_immediately_before_commit(self, clean_env, mock_fresh_receipt, disposable_db):
        conn, dsn = disposable_db
        acc = "0001045810-26-000099"
        sub_payload = {
            "filings": {
                "recent": {
                    "accessionNumber": [acc],
                    "filingDate": ["2026-11-17"],
                    "acceptanceDateTime": ["2026-11-17T21:00:00Z"],
                    "form": ["4"],
                    "primaryDocument": ["doc.xml"],
                }
            }
        }
        xml_content = make_sample_nvda_form4_xml(accession=acc, date="2026-11-17")

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_content, content=xml_content.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, max_duration_sec=30.0, **make_valid_auth())

            # Advance start_time right before pre-commit check by wrapping _has_verified_owner_lock
            orig_verify = runner._has_verified_owner_lock
            def advance_time_and_verify(c):
                runner.start_time = time.time() - 31.0
                return orig_verify(c)

            runner._has_verified_owner_lock = advance_time_and_verify

            with pytest.raises(CanaryTimeoutError) as exc_info:
                runner.execute()
            assert "DEADLINE" in str(exc_info.value).upper() or "TIMEOUT" in str(exc_info.value).upper()

            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number = %s;", (acc,))
                assert cur.fetchone()[0] == 0
                cur.execute("""
                    SELECT count(*) FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND classid = %s
                      AND objid = %s;
                """, (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
                assert cur.fetchone()[0] == 0

            assert os.environ.get("ARX_SEC_EGRESS_ROLE") == "STANDBY"
            assert os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED") == "false"


class TestNvdaCanaryFunctionalCLI:
    """Acceptance: Functional CLI entrypoint via subprocess tests."""

    def test_cli_dry_run_cannot_enter_live_execution(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        now = datetime.now(timezone.utc)

        cmd = [
            sys.executable,
            "-m",
            "analyst_dashboard.smart_money.canary_runner",
            "--dry-run",
            "--auth-issued-at", now.isoformat(),
            "--auth-expires-at", (now + timedelta(minutes=10)).isoformat(),
            "--dsn", dsn,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        env["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"
        env["ARX_EMERGENCY_STOP_ACTIVE"] = "false"
        env["ARX_BACKUP_CONTAINMENT_ENABLED"] = "false"
        env["SEC_USER_AGENT"] = "ARX-Terminal Regulatory Research (ops@arxterminal.com)"

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI exited with {res.returncode}: {res.stderr}"

        # Parse receipt from stdout
        lines = [l for l in res.stdout.split("\n") if l.strip() and not l.startswith("===") and not l.startswith("[") and not l.startswith("Canary Execution Receipt:")]
        receipt = json.loads("\n".join(lines))
        assert receipt["mode"] == "DRY_RUN"
        assert receipt["preflight_verdict"] == "PASSED"
        assert receipt["requests_made"] == 0
        assert receipt["transactions_persisted"] == 0
        assert receipt["verdict"] == "SUCCESS"

        # Verify zero database mutations
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM form4_raw_filings;")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM sm_audit_events;")
            assert cur.fetchone()[0] == 0

    def test_cli_supplied_boundaries_are_honored(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        now = datetime.now(timezone.utc)

        cmd = [
            sys.executable,
            "-m",
            "analyst_dashboard.smart_money.canary_runner",
            "--dry-run",
            "--interval-start", "2026-11-17T00:00:00Z",
            "--interval-end", "2026-11-18T23:59:59Z",
            "--public-availability-cutoff", "2026-11-19T00:00:00Z",
            "--auth-issued-at", now.isoformat(),
            "--auth-expires-at", (now + timedelta(minutes=10)).isoformat(),
            "--max-requests", "2",
            "--max-duration-sec", "15.0",
            "--dsn", dsn,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        env["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"
        env["ARX_EMERGENCY_STOP_ACTIVE"] = "false"
        env["ARX_BACKUP_CONTAINMENT_ENABLED"] = "false"
        env["SEC_USER_AGENT"] = "ARX-Terminal Regulatory Research (ops@arxterminal.com)"

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI exited with {res.returncode}: {res.stderr}"

        lines = [l for l in res.stdout.split("\n") if l.strip() and not l.startswith("===") and not l.startswith("[") and not l.startswith("Canary Execution Receipt:")]
        receipt = json.loads("\n".join(lines))
        assert receipt["interval_start"] == "2026-11-17T00:00:00+00:00"
        assert receipt["interval_end"] == "2026-11-18T23:59:59+00:00"
        assert receipt["public_availability_cutoff"] == "2026-11-19T00:00:00+00:00"
        assert receipt["max_requests"] == 2
        assert receipt["max_duration_sec"] == 15.0

    def test_cli_missing_authorization_fails(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        cmd = [
            sys.executable,
            "-m",
            "analyst_dashboard.smart_money.canary_runner",
            "--dry-run",
            "--dsn", dsn,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 1
        assert "Missing authorization timestamps" in res.stderr

    def test_cli_unknown_flags_are_rejected(self, clean_env):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        cmd = [
            sys.executable,
            "-m",
            "analyst_dashboard.smart_money.canary_runner",
            "--unknown-illegal-flag-12345",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 2
        assert "unrecognized arguments" in res.stderr

    def test_cli_dry_run_reports_individual_checks_and_lock_inspection(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        now = datetime.now(timezone.utc)

        cmd = [
            sys.executable,
            "-m",
            "analyst_dashboard.smart_money.canary_runner",
            "--dry-run",
            "--auth-issued-at", now.isoformat(),
            "--auth-expires-at", (now + timedelta(minutes=10)).isoformat(),
            "--dsn", dsn,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        env["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"
        env["ARX_EMERGENCY_STOP_ACTIVE"] = "false"
        env["ARX_BACKUP_CONTAINMENT_ENABLED"] = "false"
        env["SEC_USER_AGENT"] = "ARX-Terminal Regulatory Research (ops@arxterminal.com)"

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI exited with {res.returncode}: {res.stderr}"

        lines = [l for l in res.stdout.split("\n") if l.strip() and not l.startswith("===") and not l.startswith("[") and not l.startswith("Canary Execution Receipt:")]
        receipt = json.loads("\n".join(lines))
        assert receipt["mode"] == "DRY_RUN"
        assert receipt["preflight_verdict"] == "PASSED"
        assert receipt["verdict"] == "SUCCESS"

        checks = receipt["preflight_checks"]
        assert checks["scope_check"]["status"] == "PASS"
        assert checks["emergency_stop_check"]["status"] == "PASS"
        assert checks["containment_ratification_check"]["status"] == "PASS"
        assert checks["user_agent_check"]["status"] == "PASS"
        assert checks["backup_freshness_check"]["status"] == "PASS"
        assert checks["authorization_window_check"]["status"] == "PASS"
        assert checks["interval_and_cutoff_check"]["status"] == "PASS"
        assert checks["database_connectivity"]["status"] == "PASS"
        assert checks["queue_isolation_check"]["status"] == "PASS"
        assert checks["advisory_lock_availability"]["status"] == "PASS"

    def test_cli_dry_run_skipped_db_checks_reports_incomplete_and_blocks(self, clean_env):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        now = datetime.now(timezone.utc)

        cmd = [
            sys.executable,
            "-m",
            "analyst_dashboard.smart_money.canary_runner",
            "--dry-run",
            "--auth-issued-at", now.isoformat(),
            "--auth-expires-at", (now + timedelta(minutes=10)).isoformat(),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        env["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"
        env["ARX_EMERGENCY_STOP_ACTIVE"] = "false"
        env["ARX_BACKUP_CONTAINMENT_ENABLED"] = "false"
        env["SEC_USER_AGENT"] = "ARX-Terminal Regulatory Research (ops@arxterminal.com)"
        env.pop("SMART_MONEY_DATABASE_URL", None)
        env.pop("DATABASE_URL", None)

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 1, "Should exit 1 when required database checks were skipped"

        lines = [l for l in res.stdout.split("\n") if l.strip() and not l.startswith("===") and not l.startswith("[") and not l.startswith("Canary Execution Receipt:")]
        receipt = json.loads("\n".join(lines))
        assert receipt["preflight_verdict"] == "INCOMPLETE"
        assert receipt["verdict"] == "BLOCKED"
        assert receipt["preflight_checks"]["database_connectivity"]["status"] == "NOT_CHECKED"
        assert receipt["preflight_checks"]["queue_isolation_check"]["status"] == "NOT_CHECKED"
        assert receipt["preflight_checks"]["advisory_lock_availability"]["status"] == "NOT_CHECKED"

    def test_cli_dry_run_with_active_queue_reports_fail(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        now = datetime.now(timezone.utc)

        # Insert active request into sm_run_requests
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sm_run_requests (
                    request_id, request_type, requested_by_subject_hash, status,
                    requested_at, attempt_count, max_attempts
                ) VALUES (
                    %s, 'SCHEDULED_DISCOVERY', 'test_hash', 'PENDING', NOW(), 0, 3
                );
            """, (str(uuid.uuid4()),))
        conn.commit()

        cmd = [
            sys.executable,
            "-m",
            "analyst_dashboard.smart_money.canary_runner",
            "--dry-run",
            "--auth-issued-at", now.isoformat(),
            "--auth-expires-at", (now + timedelta(minutes=10)).isoformat(),
            "--dsn", dsn,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        env["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"
        env["ARX_EMERGENCY_STOP_ACTIVE"] = "false"
        env["ARX_BACKUP_CONTAINMENT_ENABLED"] = "false"
        env["SEC_USER_AGENT"] = "ARX-Terminal Regulatory Research (ops@arxterminal.com)"

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 1

        lines = [l for l in res.stdout.split("\n") if l.strip() and not l.startswith("===") and not l.startswith("[") and not l.startswith("Canary Execution Receipt:")]
        receipt = json.loads("\n".join(lines))
        assert receipt["preflight_verdict"] == "FAILED"
        assert receipt["verdict"] == "FAILED"
        assert receipt["preflight_checks"]["queue_isolation_check"]["status"] == "FAIL"

    def test_cli_dry_run_with_lock_held_reports_fail(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        now = datetime.now(timezone.utc)

        # Acquire lock in external session
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s);", (LOCK_SEC_EGRESS_OWNER,))
            assert cur.fetchone()[0] is True

        try:
            cmd = [
                sys.executable,
                "-m",
                "analyst_dashboard.smart_money.canary_runner",
                "--dry-run",
                "--auth-issued-at", now.isoformat(),
                "--auth-expires-at", (now + timedelta(minutes=10)).isoformat(),
                "--dsn", dsn,
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = project_root
            env["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"
            env["ARX_EMERGENCY_STOP_ACTIVE"] = "false"
            env["ARX_BACKUP_CONTAINMENT_ENABLED"] = "false"
            env["SEC_USER_AGENT"] = "ARX-Terminal Regulatory Research (ops@arxterminal.com)"

            res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
            assert res.returncode == 1

            lines = [l for l in res.stdout.split("\n") if l.strip() and not l.startswith("===") and not l.startswith("[") and not l.startswith("Canary Execution Receipt:")]
            receipt = json.loads("\n".join(lines))
            assert receipt["preflight_verdict"] == "FAILED"
            assert receipt["verdict"] == "FAILED"
            assert receipt["preflight_checks"]["advisory_lock_availability"]["status"] == "FAIL"
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s);", (LOCK_SEC_EGRESS_OWNER,))

    def test_script_cli_dry_run_cannot_enter_live_execution(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(project_root, "scripts", "execute_nvda_oneshot_cycle.py")
        now = datetime.now(timezone.utc)

        cmd = [
            sys.executable,
            script_path,
            "--dry-run",
            "--auth-issued-at", now.isoformat(),
            "--auth-expires-at", (now + timedelta(minutes=10)).isoformat(),
            "--dsn", dsn,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        env["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"
        env["ARX_EMERGENCY_STOP_ACTIVE"] = "false"
        env["ARX_BACKUP_CONTAINMENT_ENABLED"] = "false"
        env["SEC_USER_AGENT"] = "ARX-Terminal Regulatory Research (ops@arxterminal.com)"

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"Script CLI exited with {res.returncode}: {res.stderr}"

        lines = [l for l in res.stdout.split("\n") if l.strip() and not l.startswith("===") and not l.startswith("[") and not l.startswith("Canary Execution Receipt:")]
        receipt = json.loads("\n".join(lines))
        assert receipt["mode"] == "DRY_RUN"
        assert receipt["preflight_verdict"] == "PASSED"
        assert receipt["requests_made"] == 0
        assert receipt["transactions_persisted"] == 0
        assert receipt["verdict"] == "SUCCESS"

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM form4_raw_filings;")
            assert cur.fetchone()[0] == 0

    def test_script_cli_missing_authorization_fails(self, clean_env, disposable_db):
        conn, dsn = disposable_db
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(project_root, "scripts", "execute_nvda_oneshot_cycle.py")

        cmd = [
            sys.executable,
            script_path,
            "--dry-run",
            "--dsn", dsn,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 1
        assert "Missing authorization timestamps" in res.stderr

    def test_script_cli_unknown_flags_are_rejected(self, clean_env):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(project_root, "scripts", "execute_nvda_oneshot_cycle.py")

        cmd = [
            sys.executable,
            script_path,
            "--unknown-illegal-script-flag-999",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        res = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        assert res.returncode == 2
        assert "unrecognized arguments" in res.stderr


class TestNvdaCanaryContentEncodingAndDecompression:
    """Acceptance: Content-Encoding (gzip and deflate) decoding and trickling deadline enforcement."""

    def test_reproduction_gzip_json_response_decoded_and_parsed(self, clean_env):
        payload = b"{\"filings\": {\"recent\": {\"form\": [\"4\"], \"accessionNumber\": [\"0001045810-26-000100\"]}}}"
        compressed = gzip.compress(payload)

        class GzipHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                self.wfile.write(compressed)

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), GzipHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        session = requests.Session()
        runner = NvdaCanaryRunner(
            session=session,
            max_duration_sec=10.0,
            **make_valid_auth(),
        )

        try:
            resp = runner.tracked_get(f"http://127.0.0.1:{port}")
            assert resp.status_code == 200
            # Reproduction check: response.json() MUST decode cleanly without JSONDecodeError
            data = resp.json()
            assert data["filings"]["recent"]["form"] == ["4"]
            assert data["filings"]["recent"]["accessionNumber"] == ["0001045810-26-000100"]
        finally:
            server.shutdown()
            server.server_close()

    def test_deflate_zlib_json_response_decoded_and_parsed(self, clean_env):
        payload = b"{\"filings\": {\"recent\": {\"form\": [\"4/A\"], \"accessionNumber\": [\"0001045810-26-000101\"]}}}"
        compressed = zlib.compress(payload)

        class DeflateHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "deflate")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                self.wfile.write(compressed)

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), DeflateHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        session = requests.Session()
        runner = NvdaCanaryRunner(
            session=session,
            max_duration_sec=10.0,
            **make_valid_auth(),
        )

        try:
            resp = runner.tracked_get(f"http://127.0.0.1:{port}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["filings"]["recent"]["form"] == ["4/A"]
            assert data["filings"]["recent"]["accessionNumber"] == ["0001045810-26-000101"]
        finally:
            server.shutdown()
            server.server_close()

    def test_deflate_raw_json_response_decoded_and_parsed(self, clean_env):
        payload = b"{\"filings\": {\"recent\": {\"form\": [\"4\"], \"accessionNumber\": [\"0001045810-26-000102\"]}}}"
        comp = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -zlib.MAX_WBITS)
        compressed = comp.compress(payload) + comp.flush()

        class RawDeflateHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "deflate")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                self.wfile.write(compressed)

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), RawDeflateHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        session = requests.Session()
        runner = NvdaCanaryRunner(
            session=session,
            max_duration_sec=10.0,
            **make_valid_auth(),
        )

        try:
            resp = runner.tracked_get(f"http://127.0.0.1:{port}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["filings"]["recent"]["form"] == ["4"]
            assert data["filings"]["recent"]["accessionNumber"] == ["0001045810-26-000102"]
        finally:
            server.shutdown()
            server.server_close()

    def test_compressed_trickling_response_within_deadline(self, clean_env):
        payload = b"{\"status\": \"trickling_gzip_success\", \"count\": 123}"
        compressed = gzip.compress(payload)

        class FastTrickleHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                # Trickle 2 bytes every 0.005s (total ~0.15s, within 5.0s budget)
                for i in range(0, len(compressed), 2):
                    self.wfile.write(compressed[i:i+2])
                    self.wfile.flush()
                    time.sleep(0.005)

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), FastTrickleHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        session = requests.Session()
        runner = NvdaCanaryRunner(
            session=session,
            max_duration_sec=5.0,
            **make_valid_auth(),
        )

        try:
            resp = runner.tracked_get(f"http://127.0.0.1:{port}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "trickling_gzip_success"
            assert data["count"] == 123
        finally:
            server.shutdown()
            server.server_close()

    def test_compressed_trickling_response_exceeding_deadline_rejected(self, clean_env):
        payload = b"{\"status\": \"trickling_gzip_exceeding_deadline\"}"
        compressed = gzip.compress(payload)

        class SlowTrickleHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                # Trickle 1 byte every 0.05s (total ~2.5s, exceeds 0.35s limit)
                for b in compressed:
                    time.sleep(0.05)
                    self.wfile.write(bytes([b]))
                    self.wfile.flush()

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), SlowTrickleHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        session = requests.Session()
        runner = NvdaCanaryRunner(
            session=session,
            max_duration_sec=0.35,
            **make_valid_auth(),
        )

        try:
            with pytest.raises(CanaryTimeoutError) as exc_info:
                runner.tracked_get(f"http://127.0.0.1:{port}")
            assert "DEADLINE" in str(exc_info.value).upper() or "TIMEOUT" in str(exc_info.value).upper()
        finally:
            server.shutdown()
            server.server_close()

    def test_deflate_trickling_response_exceeding_deadline_rejected(self, clean_env):
        payload = b"{\"status\": \"trickling_deflate_exceeding_deadline\"}"
        compressed = zlib.compress(payload)

        class SlowDeflateHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "deflate")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                # Trickle 1 byte every 0.05s (total ~2.5s, exceeds 0.35s limit)
                for b in compressed:
                    time.sleep(0.05)
                    self.wfile.write(bytes([b]))
                    self.wfile.flush()

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), SlowDeflateHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        session = requests.Session()
        runner = NvdaCanaryRunner(
            session=session,
            max_duration_sec=0.35,
            **make_valid_auth(),
        )

        try:
            with pytest.raises(CanaryTimeoutError) as exc_info:
                runner.tracked_get(f"http://127.0.0.1:{port}")
            assert "DEADLINE" in str(exc_info.value).upper() or "TIMEOUT" in str(exc_info.value).upper()
        finally:
            server.shutdown()
            server.server_close()


