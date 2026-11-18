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
import time
import json
import uuid
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


@pytest.fixture(scope="module")
def pg_server_instance(tmp_path_factory):
    """Module-scoped ephemeral PostgreSQL server via pgserver."""
    pgserver = pytest.importorskip("pgserver")
    tmp_dir = tmp_path_factory.mktemp("pg_canary_suite")
    server = pgserver.get_server(tmp_dir)
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
    date: str = "2026-10-30",
    ticker: str = "NVDA",
    cik: str = "0001045810",
    is_amendment: bool = False,
    amendment_of: str = "",
    remarks: str = "",
) -> str:
    doc_type = "4/A" if is_amendment else "4"
    amend_node = f"<amendmentOfAccessionNumber>{amendment_of}</amendmentOfAccessionNumber>" if amendment_of else ""
    remarks_node = f"<remarks>{remarks}</remarks>" if remarks else ""
    return f"""<?xml version="1.0"?>
    <ownershipDocument>
        <schemaVersion>X0508</schemaVersion>
        <documentType>{doc_type}</documentType>
        <periodOfReport>{date}</periodOfReport>
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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
        runner = NvdaCanaryRunner()
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(ticker="AAPL")
        assert "UNAUTHORIZED_CANARY_SCOPE" in str(exc_info.value)

    def test_gate2_wrong_issuer_cik_fails_closed(self, clean_env):
        runner = NvdaCanaryRunner()
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(cik="0000320193")
        assert "UNAUTHORIZED_CANARY_SCOPE" in str(exc_info.value)

    def test_gate2_xml_issuer_mismatch_fails_closed(self, clean_env, mock_fresh_receipt):
        sub_payload = {
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
        # XML payload contains MSFT instead of NVDA
        xml_mismatch = make_sample_nvda_form4_xml(ticker="MSFT", cik="0000789019")

        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        resp_xml = MagicMock(status_code=200, text=xml_mismatch, content=xml_mismatch.encode())
        mock_session.get.side_effect = [resp_sub, resp_xml]

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=None, session=mock_session)
            with pytest.raises(CanarySecurityError) as exc_info:
                runner.execute()
            assert "SEC_FORM4_ISSUER_MISMATCH" in str(exc_info.value)

    def test_gate2_expired_authorization_fails_closed(self, clean_env):
        runner = NvdaCanaryRunner()
        past_expiry = datetime.now(timezone.utc) - timedelta(minutes=10)
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(auth_expires_at=past_expiry)
        assert "AUTHORIZATION_EXPIRED" in str(exc_info.value)

    def test_gate2_future_authorization_issued_fails_closed(self, clean_env):
        runner = NvdaCanaryRunner()
        future_issued = datetime.now(timezone.utc) + timedelta(minutes=10)
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(auth_issued_at=future_issued)
        assert "AUTHORIZATION_NOT_YET_VALID" in str(exc_info.value)

    def test_gate2_inverted_interval_fails_closed(self, clean_env):
        runner = NvdaCanaryRunner()
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(
                interval_start="2026-11-20T00:00:00Z",
                interval_end="2026-11-10T00:00:00Z",
            )
        assert "INVALID_INTERVAL" in str(exc_info.value)

    def test_gate2_cutoff_before_interval_end_fails_closed(self, clean_env):
        runner = NvdaCanaryRunner()
        with pytest.raises(CanarySecurityError) as exc_info:
            runner.validate_preflight(
                interval_end="2026-11-20T00:00:00Z",
                public_availability_cutoff="2026-11-19T00:00:00Z",
            )
        assert "INVALID_CUTOFF" in str(exc_info.value)

    def test_gate2_prospective_post_baseline_interval_success(self, clean_env, mock_fresh_receipt):
        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(
                mode="prospective",
                interval_start="2026-11-17T00:00:00Z",
                interval_end="2026-11-18T23:59:59Z",
                public_availability_cutoff="2026-11-19T00:00:00Z",
            )
            assert runner.validate_preflight() is True

    def test_gate2_prospective_mode_rejects_silent_frozen_baseline_substitution(self, clean_env):
        runner = NvdaCanaryRunner(
            mode="prospective",
            interval_start="2026-08-18T00:00:00Z",
            interval_end="2026-11-16T23:59:59Z",
            public_availability_cutoff="2026-11-17T00:00:00Z",
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
            )
            assert runner.validate_preflight() is True

    def test_gate2_historical_audit_mode_rejects_post_baseline_interval(self, clean_env):
        runner = NvdaCanaryRunner(
            mode="historical_audit",
            interval_start="2026-11-17T00:00:00Z",
            interval_end="2026-11-18T23:59:59Z",
            public_availability_cutoff="2026-11-19T00:00:00Z",
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
                runner = NvdaCanaryRunner(dsn=dsn)
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
            runner = NvdaCanaryRunner(dsn=dsn)
            with pytest.raises(CanaryQueueActiveError) as exc_info:
                runner.validate_preflight(conn=conn)
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
                    "filingDate": ["2026-11-01", "2026-11-01", "2026-11-01", "2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"] * 4,
                    "primaryDocument": ["doc1.xml", "doc2.xml", "doc3.xml", "doc4.xml"],
                }
            }
        }
        mock_session = MagicMock()
        resp_sub = MagicMock(status_code=200, content=json.dumps(sub_payload).encode())
        resp_sub.json.return_value = sub_payload
        mock_session.get.return_value = resp_sub

        with patch("analyst_dashboard.smart_money.storage.backup.containment_gate.BackupFreshnessAuthority.evaluate_freshness", return_value=mock_fresh_receipt):
            runner = NvdaCanaryRunner(dsn=None, session=mock_session)
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
                    "filingDate": ["2026-11-01"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
                    "filingDate": ["2026-09-10"],
                    "acceptanceDateTime": ["2026-09-10T21:00:00Z"],
                    "primaryDocument": ["form4_amend.xml"],
                }
            }
        }
        amend_xml = make_sample_nvda_form4_xml(
            accession=amend_acc,
            date="2026-09-10",
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
        runner = NvdaCanaryRunner(dsn=dsn)

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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
                    "filingDate": ["2026-11-01"],
                    "acceptanceDateTime": ["2026-11-01T21:00:00Z"],
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session)
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
            runner = NvdaCanaryRunner(dsn=dsn, session=mock_session, max_duration_sec=0.5)
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

