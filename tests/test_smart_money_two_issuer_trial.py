"""
ARX Terminal — Smart Money Two-Issuer Recurring Ingestion Trial Test Suite.

Verifies the 10 core requirements from GitHub Issue #165:
1. Explicit two-issuer authorization profile (RECURRING_2_ISSUER_FORM4_TRIAL).
2. Existing ten-issuer profile (DAILY_10_ISSUER_FORM4) independence.
3. Separation of ingestion completion from snapshot publication (active snapshot untouched).
4. Durable scheduling and progress independent of active snapshot.
5. Non-overridable ceilings: <= 8 requests/cycle, <= 4/issuer, <= 5 req/s.
6. Absolute cycle duration and deadline enforcement.
7. Safe SEC primary-document path resolution (accepts legitimate paths, rejects traversal/host substitution).
8. Fail-safe primary-document fetch without silent truncation.
9. Truthful evidence: parsing, amendment classification, availability ledger, and atomic persistence.
10. Historical coverage remains PARTIAL (completed cycle != COMPLETE; no-work != VERIFIED_ZERO).
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import hashlib
import re
import socket
import threading
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Any, Optional, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from analyst_dashboard.smart_money.continuous_authorization import (
    require_continuous_authorization,
    SCOPE_DAILY_10,
    SCOPE_TRIAL_2,
    _TRIAL_2_CIKS,
)
from analyst_dashboard.smart_money.continuous_pipeline import (
    SmartMoneyContinuousPipeline,
    ContinuousOutboundSession,
    ContinuousBudgetExceededError,
    TRIAL_2_GOVERNED_ISSUERS,
    DEFAULT_GOVERNED_ISSUERS,
    resolve_sec_primary_document_url,
    PROSPECTIVE_TRIAL_START_BOUNDARY,
    validate_cycle_duration,
    parse_sec_datetime_utc,
)
from analyst_dashboard.smart_money.smart_money_scheduler import (
    TwoIssuerTrialScheduler,
    TRIAL_DAILY_SLOTS,
    MAX_CYCLES_PER_DAY,
    MAX_CYCLES_PER_TRIAL,
    MAX_REQUESTS_PER_DAY,
    MAX_REQUESTS_PER_TRIAL,
)
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    StaleLeaseOwnershipLostError,
)
from analyst_dashboard.smart_money.worker import (
    SmartMoneyIngestionWorker,
    WorkerHealthState,
)
from analyst_dashboard.data.sec_edgar_fetcher import RateLimiter

UTC = timezone.utc
MOCK_SHA = "a" * 40
MOCK_GRANT_ID = "b" * 32
NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)

TRIAL_ISSUERS = [
    {"ticker": "NVDA", "cik": "0001045810", "company_name": "NVIDIA CORP"},
    {"ticker": "AAPL", "cik": "0000320193", "company_name": "APPLE INC"},
]


def trial_grant(**changes) -> Dict[str, str]:
    env = {
        "ARX_SEC_EGRESS_ROLE": "OWNER",
        "ARX_CONTINUOUS_INGESTION_AUTHORIZED": "true",
        "ARX_CONTINUOUS_AUTH_SCOPE": SCOPE_TRIAL_2,
        "ARX_CONTINUOUS_AUTH_UNIVERSE_ID": "UB-US-EQUITIES-20261025-001",
        "ARX_CONTINUOUS_AUTH_ID": MOCK_GRANT_ID,
        "ARX_RELEASE_SHA": MOCK_SHA,
        "ARX_CONTINUOUS_AUTH_RELEASE_SHA": MOCK_SHA,
        "ARX_CONTINUOUS_AUTH_ISSUED_AT": "2026-11-20T10:00:00Z",
        "ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-21T10:00:00Z",
    }
    env.update(changes)
    return env


def daily_10_grant(**changes) -> Dict[str, str]:
    env = {
        "ARX_SEC_EGRESS_ROLE": "OWNER",
        "ARX_CONTINUOUS_INGESTION_AUTHORIZED": "true",
        "ARX_CONTINUOUS_AUTH_SCOPE": SCOPE_DAILY_10,
        "ARX_CONTINUOUS_AUTH_UNIVERSE_ID": "UB-US-EQUITIES-20261025-001",
        "ARX_CONTINUOUS_AUTH_ID": MOCK_GRANT_ID,
        "ARX_RELEASE_SHA": MOCK_SHA,
        "ARX_CONTINUOUS_AUTH_RELEASE_SHA": MOCK_SHA,
        "ARX_CONTINUOUS_AUTH_ISSUED_AT": "2026-11-20T10:00:00Z",
        "ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-21T10:00:00Z",
    }
    env.update(changes)
    return env


def generate_form4_xml(
    ticker: str = "NVDA",
    cik: str = "0001045810",
    date: str = "2026-11-19",
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
            <issuerName>{'NVIDIA CORP' if ticker == 'NVDA' else 'APPLE INC'}</issuerName>
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
                    <transactionCode>S</transactionCode>
                </transactionCoding>
                <transactionAmounts>
                    <transactionShares><value>5000</value></transactionShares>
                    <transactionPricePerShare><value>145.50</value></transactionPricePerShare>
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


# ==============================================================================
# SECTION 1: TWO-ISSUER AUTHORIZATION PROFILE
# ==============================================================================

class TestTwoIssuerAuthorizationProfile:
    """Verifies that the two-issuer trial profile enforces exact constraints."""

    def test_valid_two_issuer_grant_accepted(self):
        require_continuous_authorization(
            env=trial_grant(),
            issuers=TRIAL_ISSUERS,
            now=NOW,
        )

    def test_two_issuer_reversed_order_accepted(self):
        reversed_issuers = list(reversed(TRIAL_ISSUERS))
        require_continuous_authorization(
            env=trial_grant(),
            issuers=reversed_issuers,
            now=NOW,
        )

    def test_extra_issuer_rejected(self):
        extra = TRIAL_ISSUERS + [{"ticker": "MSFT", "cik": "0000789019"}]
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_ISSUER_MISMATCH"):
            require_continuous_authorization(env=trial_grant(), issuers=extra, now=NOW)

    def test_missing_issuer_rejected(self):
        missing = [TRIAL_ISSUERS[0]]  # Only NVDA
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_ISSUER_MISMATCH"):
            require_continuous_authorization(env=trial_grant(), issuers=missing, now=NOW)

    def test_wrong_cik_rejected(self):
        bad_cik = [
            {"ticker": "NVDA", "cik": "0009999999"},
            {"ticker": "AAPL", "cik": "0000320193"},
        ]
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_ISSUER_MISMATCH"):
            require_continuous_authorization(env=trial_grant(), issuers=bad_cik, now=NOW)

    def test_expired_grant_rejected(self):
        expired_now = datetime(2026, 11, 21, 10, 0, 1, tzinfo=UTC)
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_EXPIRED"):
            require_continuous_authorization(env=trial_grant(), issuers=TRIAL_ISSUERS, now=expired_now)

    def test_future_grant_rejected(self):
        future_now = datetime(2026, 11, 20, 9, 59, 59, tzinfo=UTC)
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_NOT_YET_VALID"):
            require_continuous_authorization(env=trial_grant(), issuers=TRIAL_ISSUERS, now=future_now)

    def test_release_sha_mismatch_rejected(self):
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_RELEASE_MISMATCH"):
            require_continuous_authorization(
                env=trial_grant(ARX_CONTINUOUS_AUTH_RELEASE_SHA="c" * 40),
                issuers=TRIAL_ISSUERS,
                now=NOW,
            )

    def test_unknown_scope_rejected(self):
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_SCOPE_MISMATCH"):
            require_continuous_authorization(
                env=trial_grant(ARX_CONTINUOUS_AUTH_SCOPE="UNKNOWN_SCOPE"),
                issuers=TRIAL_ISSUERS,
                now=NOW,
            )

    def test_missing_authorization_rejected(self):
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_MISSING"):
            require_continuous_authorization(
                env=trial_grant(ARX_CONTINUOUS_INGESTION_AUTHORIZED="false"),
                issuers=TRIAL_ISSUERS,
                now=NOW,
            )

    def test_non_owner_role_rejected(self):
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_OWNER_REQUIRED"):
            require_continuous_authorization(
                env=trial_grant(ARX_SEC_EGRESS_ROLE="STANDBY"),
                issuers=TRIAL_ISSUERS,
                now=NOW,
            )

    def test_ten_issuer_profile_preserved(self):
        # Daily 10 grant with DEFAULT_GOVERNED_ISSUERS must succeed
        require_continuous_authorization(
            env=daily_10_grant(),
            issuers=DEFAULT_GOVERNED_ISSUERS,
            now=NOW,
        )
        # Daily 10 grant with 2 issuers must fail
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_ISSUER_MISMATCH"):
            require_continuous_authorization(
                env=daily_10_grant(),
                issuers=TRIAL_ISSUERS,
                now=NOW,
            )


# ==============================================================================
# SECTION 2: SAFE SEC PRIMARY DOCUMENT PATH RESOLUTION
# ==============================================================================

class TestSafeSecPathResolution:
    """Verifies safe resolution of SEC primary-document paths."""

    ARCHIVE_ROOT = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000001/"

    @pytest.mark.parametrize("valid_doc,expected_suffix", [
        ("form4.xml", "form4.xml"),
        ("xslF345X06/form4.xml", "xslF345X06/form4.xml"),
        ("edgar.xml", "edgar.xml"),
        ("primary_doc.xml", "primary_doc.xml"),
        ("sub_folder/doc_123.xml", "sub_folder/doc_123.xml"),
    ])
    def test_valid_sec_paths_resolved(self, valid_doc, expected_suffix):
        resolved = resolve_sec_primary_document_url(self.ARCHIVE_ROOT, valid_doc)
        assert resolved == f"{self.ARCHIVE_ROOT}{expected_suffix}"
        assert resolved.startswith(self.ARCHIVE_ROOT)

    @pytest.mark.parametrize("malicious_doc", [
        "../other.xml",
        "../../etc/passwd",
        "dir/../../secret.xml",
        "/absolute/path/form4.xml",
        "\\windows\\path.xml",
        "https://evil.com/form4.xml",
        "http://evil.com/form4.xml",
        "//evil.com/form4.xml",
        "form4.xml?query=1",
        "form4.xml#frag",
        "user:pass@form4.xml",
        "dir/./current.xml",
        "",
        "   ",
        "doc name with spaces.xml",
    ])
    def test_malicious_or_invalid_paths_rejected(self, malicious_doc):
        with pytest.raises(RuntimeError, match="SEC_PRIMARY_DOCUMENT_INVALID"):
            resolve_sec_primary_document_url(self.ARCHIVE_ROOT, malicious_doc)


# ==============================================================================
# SECTION 3: BOUNDED FETCH CEILINGS & DEADLINES
# ==============================================================================

class TestBoundedFetchCeilings:
    """Verifies non-overridable ceilings: <= 8 req/cycle, <= 4/issuer, <= 5 req/s."""

    def test_two_issuer_total_request_ceiling_enforced(self):
        """Enforces <= 8 total requests per cycle for 2-issuer trial profile."""
        session = ContinuousOutboundSession(
            allow_test_mode=True,
            max_total_requests=8,
            max_per_issuer_requests=4,
            issuers=TRIAL_ISSUERS,
        )
        assert session.max_total_requests == 8

        mock_resp = requests.Response()
        mock_resp.status_code = 200

        with patch.object(requests.Session, "send", return_value=mock_resp):
            req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0001045810.json").prepare()

            session.set_current_issuer("NVDA")
            for _ in range(4):
                session.send(req)

            session.set_current_issuer("AAPL")
            for _ in range(4):
                session.send(req)

            assert session.total_transmissions == 8

            # 9th request must fail closed
            with pytest.raises(ContinuousBudgetExceededError, match="CYCLE_BUDGET_EXCEEDED"):
                session.send(req)

    def test_env_var_cannot_raise_trial_ceiling_above_eight(self, monkeypatch):
        """Environment configuration may lower ceiling, but NEVER raise it above 8 for trial."""
        monkeypatch.setenv("ARX_SEC_MAX_TOTAL_REQUESTS_PER_CYCLE", "50")
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_SCOPE", SCOPE_TRIAL_2)

        session = ContinuousOutboundSession(
            allow_test_mode=True,
            max_total_requests=50,  # Caller asks for 50
            issuers=TRIAL_ISSUERS,
        )
        assert session.max_total_requests <= 8  # Clamped to 8

        mock_resp = requests.Response()
        mock_resp.status_code = 200

        with patch.object(requests.Session, "send", return_value=mock_resp):
            req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0001045810.json").prepare()
            session.set_current_issuer("NVDA")
            for _ in range(4):
                session.send(req)
            session.set_current_issuer("AAPL")
            for _ in range(4):
                session.send(req)

            # 9th request must fail closed despite env var setting 50
            with pytest.raises(ContinuousBudgetExceededError, match="CYCLE_BUDGET_EXCEEDED"):
                session.send(req)

    def test_per_issuer_ceiling_enforced(self):
        """Enforces <= 4 requests per issuer."""
        session = ContinuousOutboundSession(
            allow_test_mode=True,
            max_total_requests=8,
            max_per_issuer_requests=4,
            issuers=TRIAL_ISSUERS,
        )
        session.set_current_issuer("NVDA")

        mock_resp = requests.Response()
        mock_resp.status_code = 200

        with patch.object(requests.Session, "send", return_value=mock_resp):
            req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0001045810.json").prepare()
            for _ in range(4):
                session.send(req)

            # 5th request for NVDA must fail closed
            with pytest.raises(ContinuousBudgetExceededError, match="ISSUER_BUDGET_EXCEEDED"):
                session.send(req)

    def test_rate_limiter_clamped_to_5rps(self):
        """Rate limiter ceiling must never exceed 5.0 req/s."""
        limiter = MagicMock()
        limiter.max_per_second = 10.0
        session = ContinuousOutboundSession(
            rate_limiter=limiter,
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
        )
        assert session.rate_limiter.max_per_second <= 5.0

    def test_cycle_deadline_timeout_enforced(self):
        """Absolute cycle deadline blocks sends when exceeded."""
        session = ContinuousOutboundSession(
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            cycle_deadline=time.time() - 1.0,  # Already expired
        )
        req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0001045810.json").prepare()
        with pytest.raises(RuntimeError, match="CYCLE_TIMEOUT_EXCEEDED"):
            session.send(req)

    def test_new_filings_exceeding_budget_fails_closed_without_silent_truncation(self, monkeypatch):
        """If catalog advertises > 3 new filings, fail closed instead of silently truncating."""
        monkeypatch.setenv("SEC_USER_AGENT", "ARX Terminal Research (ops@arxterminal.com)")
        pipeline = SmartMoneyContinuousPipeline(
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
        )
        pipeline.trial_scheduler.start_date = date(2026, 11, 19)
        catalog_mock = MagicMock()
        catalog_mock.status_code = 200
        # 4 new Form 4 accessions
        catalog_mock.json.return_value = {
            "filings": {
                "recent": {
                    "form": ["4", "4", "4", "4"],
                    "accessionNumber": [
                        "0001045810-26-000001",
                        "0001045810-26-000002",
                        "0001045810-26-000003",
                        "0001045810-26-000004",
                    ],
                    "filingDate": ["2026-11-19"] * 4,
                    "acceptanceDateTime": ["2026-11-19T20:00:00Z"] * 4,
                    "primaryDocument": ["form4.xml"] * 4,
                }
            }
        }
        with patch.object(pipeline.outbound_session, "get", return_value=catalog_mock):
            with pytest.raises(RuntimeError, match="SEC_NEW_FILINGS_EXCEED_ISSUER_BUDGET"):
                pipeline._fetch_filings_from_sec("NVDA", "0001045810", "NVIDIA CORP")


# ==============================================================================
# SECTION 4: DISPOSABLE POSTGRESQL & FULL PIPELINE INTEGRATION
# ==============================================================================

@pytest.fixture(scope="module")
def pg_server_instance(tmp_path_factory):
    """Module-scoped ephemeral PostgreSQL server via pgserver."""
    pgserver = pytest.importorskip("pgserver")
    tmp_dir = tmp_path_factory.mktemp("pg_two_issuer_trial_suite")
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
    conn.close()
    return dsn


def make_mock_backup_engine():
    mock_engine = MagicMock()
    def fake_create_backup(custom_backup_id=None, include_globals=False):
        bid = custom_backup_id or f"mock-backup-{int(time.time())}"
        return {
            "backup_id": bid,
            "status": "VERIFIED",
            "upload_verified": True,
            "storage_destination": f"mock://storage/{bid}.dump",
            "artifact_sha256": "0" * 64,
            "artifact_size_bytes": 1024,
        }
    mock_engine.create_backup.side_effect = fake_create_backup
    return mock_engine


class TestTwoIssuerTrialPostgresIntegration:
    """Verifies trial pipeline behavior against disposable PostgreSQL."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        monkeypatch.setenv("SEC_USER_AGENT", "ARX Terminal Research (ops@arxterminal.com)")
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_SCOPE", SCOPE_TRIAL_2)
        monkeypatch.setenv("ARX_SMART_MONEY_INGESTION_ONLY", "true")
        monkeypatch.setenv("ARX_EMERGENCY_STOP_ACTIVE", "false")

    def test_ingestion_completion_without_snapshot_publication(self, disposable_db, monkeypatch):
        """Trial persists filing evidence and completes run request without touching active snapshot."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)

        # 1. Plant pre-existing active snapshot
        psycopg2 = pytest.importorskip("psycopg2")
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(disposable_db)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scanner_snapshots (
                    snapshot_id, scanner_id, run_id, api_contract_version, ruleset_version,
                    evidence_schema_version, score_model_version, data_provenance_version,
                    universe_version, freshness_policy_version, implementation_release_sha,
                    semantic_fingerprint, generated_at, data_as_of, status_at_publication,
                    universe_id, universe_size, matched_count, results_json, provenance_json,
                    freshness_json, publication_decision
                ) VALUES (
                    'SNAP_INITIAL_BASELINE', 'SMART_MONEY', 'RUN_INITIAL_001', '1.0.0', '1.0.0',
                    '1.0.0', '1.0.0', '1.0.0', '1.0.0', '1.0.0', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', NOW(), '2026-10-25T20:00:00Z', 'AVAILABLE',
                    'UB-US-EQUITIES-20261025-001', 10, 1, '[]'::jsonb, '{}'::jsonb,
                    '{}'::jsonb, 'PUBLISH'
                );
                INSERT INTO scanner_active_snapshots (scan_type, active_snapshot_id, published_at, published_by)
                VALUES ('SMART_MONEY', 'SNAP_INITIAL_BASELINE', NOW(), 'system');
            """)
            conn.commit()

        # 2. Plant an operator run request
        req_id = str(uuid.uuid4())
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO sm_run_requests (request_id, request_type, requested_by_subject_hash, status, lease_token, lease_expires_at)
                VALUES (%s, 'OPERATOR_INGESTION', 'hash123', 'RUNNING', 'lease-token-001', NOW() + interval '5 minutes')
                RETURNING request_id;
            """, (req_id,))
            conn.commit()
        conn.close()

        # 3. Initialize trial pipeline
        pipeline = SmartMoneyContinuousPipeline(
            dsn=disposable_db,
            repository=repo,
            backup_engine=make_mock_backup_engine(),
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )
        assert pipeline.ingestion_only is True

        # Mock SEC response for NVDA and AAPL
        nvda_xml = generate_form4_xml("NVDA", "0001045810", "2026-11-19")
        filing_data = [{
            "ticker": "NVDA",
            "cik": "0001045810",
            "company_name": "NVIDIA CORP",
            "accession_number": "0001045810-26-000001",
            "filing_date": "2026-11-19",
            "accepted_datetime": "2026-11-19T20:30:00Z",
            "document_type": "4",
            "xml_content": nvda_xml,
            "source_url": "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000001/form4.xml",
            "content_hash": hashlib.sha256(nvda_xml.encode("utf-8")).hexdigest(),
        }]

        with patch.object(pipeline, "_step6_ingest_form4_filings", return_value=filing_data):
            res = pipeline.execute_cycle(
                run_request_id=req_id,
                lease_token="lease-token-001",
                force=True,
            )

        # 4. Verify results
        assert res["status"] == "SUCCESS"
        assert res["snapshot_id"] is None  # NO snapshot published
        assert "STEP_8_INGESTION_ONLY_COMPLETED" in res["steps_completed"]
        assert "STEP_8_ATOMIC_SNAPSHOT_PUBLISHED" not in res["steps_completed"]

        # 5. Verify active snapshot pointer is completely untouched
        conn = psycopg2.connect(disposable_db)
        with conn.cursor() as cur:
            cur.execute("SELECT active_snapshot_id FROM scanner_active_snapshots WHERE scan_type = 'SMART_MONEY';")
            active_snap = cur.fetchone()[0]
            assert active_snap == "SNAP_INITIAL_BASELINE"  # UNCHANGED!

            # 6. Verify run request was marked COMPLETED under lease token
            cur.execute("SELECT status, result_generation_id FROM sm_run_requests WHERE request_id = %s;", (req_id,))
            req_row = cur.fetchone()
            assert req_row[0] == "COMPLETED"
            assert req_row[1] == res["receipt_id"]  # bound to receipt_id, not snapshot_id

            # 7. Verify filing evidence persisted
            cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number = '0001045810-26-000001';")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM form4_parsed_transactions WHERE accession_number = '0001045810-26-000001';")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM sm_source_availability_ledger WHERE accession_number = '0001045810-26-000001';")
            assert cur.fetchone()[0] == 1

            # 8. Verify reconciliation receipt has PARTIAL coverage status
            cur.execute("SELECT coverage_status, failure_reasons FROM sm_reconciliation_receipts WHERE receipt_id = %s;", (res["receipt_id"],))
            receipt_row = cur.fetchone()
            assert receipt_row[0] == "PARTIAL"
            assert "TRIAL_INTERVAL_PARTIAL_COVERAGE" in str(receipt_row[1])

        conn.close()

    def test_duplicate_delivery_protection_under_lock(self, disposable_db, monkeypatch):
        """Duplicate delivery of the same trial slot is skipped independently of active snapshot."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)

        # Plant reconciliation receipt with today's date in New York
        from analyst_dashboard.smart_money.freshness_authority import NY_TZ
        now_ny = datetime(2026, 11, 20, 9, 20, tzinfo=NY_TZ)
        today_iso = now_ny.strftime("%Y-%m-%dT22:30:00Z")
        target_slot = f"{now_ny.strftime('%Y-%m-%d')}:SLOT_1:0915"
        receipt_meta = {
            "receipt_id": str(uuid.uuid4()),
            "interval_start": "2026-11-17T00:00:00Z",
            "interval_end": today_iso,
            "certified_watermark": today_iso,
            "receipt_version": 1,
            "expected_count": 1,
            "accounted_count": 1,
            "missing_count": 0,
            "material_gap_count": 0,
            "coverage_status": "PARTIAL",
            "failure_reasons": ["TRIAL_INTERVAL_PARTIAL_COVERAGE", {"trial_id": "ARX-TRIAL-2026-NVDA-AAPL-001", "slot_id": target_slot}],
            "covered_ciks": ["0001045810", "0000320193"],
            "covered_tickers": ["NVDA", "AAPL"],
            "accession_inventory": ["0001045810-26-000001"],
            "source_hashes": {},
            "reconciliation_hash": "a" * 64,
            "release_sha": MOCK_SHA,
        }
        repo.save_reconciliation_receipt(receipt_meta)

        pipeline = SmartMoneyContinuousPipeline(
            dsn=disposable_db,
            repository=repo,
            backup_engine=make_mock_backup_engine(),
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )
        pipeline.trial_scheduler.start_date = now_ny.date()
        pipeline.trial_scheduler.end_date = now_ny.date() + timedelta(days=5)

        monkeypatch.setattr(pipeline.trial_scheduler, "get_ny_now", lambda now=None: now_ny)

        # Check slot eligibility under lock without force for today
        should_run, reason = pipeline._step3_check_durable_date_eligibility_under_lock(
            force=False, as_of=now_ny.strftime("%Y-%m-%d")
        )
        assert should_run is False
        assert "ALREADY_EXECUTED" in reason or "SLOT_ALREADY_EXECUTED" in reason

    def test_stale_lease_ownership_loss_fences_completion(self, disposable_db):
        """If worker loses lease ownership, run request completion is safely fenced."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)

        req_id = str(uuid.uuid4())
        psycopg2 = pytest.importorskip("psycopg2")
        conn = psycopg2.connect(disposable_db)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sm_run_requests (request_id, request_type, requested_by_subject_hash, status, lease_token, lease_expires_at)
                VALUES (%s, 'OPERATOR_INGESTION', 'hash123', 'RUNNING', 'valid-token', NOW() + interval '5 minutes');
            """, (req_id,))
            conn.commit()
        conn.close()

        # Attempt to complete with an outdated / wrong lease token
        with pytest.raises(StaleLeaseOwnershipLostError, match="STALE_LEASE_OWNERSHIP_LOST"):
            repo.complete_run_request(
                request_id=req_id,
                lease_token="stale-outdated-token",
                status="COMPLETED",
            )

    def test_stale_request_recovery(self, disposable_db):
        """Expired running requests are recovered to RETRY_ELIGIBLE."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)

        req_id = str(uuid.uuid4())
        psycopg2 = pytest.importorskip("psycopg2")
        conn = psycopg2.connect(disposable_db)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sm_run_requests (request_id, request_type, requested_by_subject_hash, status, lease_token, lease_expires_at)
                VALUES (%s, 'OPERATOR_INGESTION', 'hash123', 'RUNNING', 'tok-1', NOW() - interval '10 minutes');
            """, (req_id,))
            conn.commit()
        conn.close()

        recovered_count = repo.recover_stale_run_requests()
        assert recovered_count >= 1

        conn = psycopg2.connect(disposable_db)
        with conn.cursor() as cur:
            cur.execute("SELECT status, lease_token FROM sm_run_requests WHERE request_id = %s;", (req_id,))
            row = cur.fetchone()
            assert row[0] == "RETRY_ELIGIBLE"
            assert row[1] is None
        conn.close()

    def test_no_work_cycle_preserves_partial_coverage(self, disposable_db, monkeypatch):
        """Zero new filings produces successful cycle with PARTIAL coverage (no verified-zero claim)."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)

        pipeline = SmartMoneyContinuousPipeline(
            dsn=disposable_db,
            repository=repo,
            backup_engine=make_mock_backup_engine(),
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )

        with patch.object(pipeline, "_step6_ingest_form4_filings", return_value=[]):
            res = pipeline.execute_cycle(force=True)

        assert res["status"] == "SUCCESS"
        assert res["snapshot_id"] is None

        # Verify receipt has expected_count = 0 and coverage_status = "PARTIAL"
        psycopg2 = pytest.importorskip("psycopg2")
        conn = psycopg2.connect(disposable_db)
        with conn.cursor() as cur:
            cur.execute("SELECT expected_count, coverage_status FROM sm_reconciliation_receipts WHERE receipt_id = %s;", (res["receipt_id"],))
            row = cur.fetchone()
            assert row[0] == 0
            assert row[1] == "PARTIAL"
        conn.close()


# ==============================================================================
# SECTION 5: DURATION & INGESTION-ONLY NON-OVERRIDABILITY GOVERNANCE
# ==============================================================================

class TestCycleDurationAndIngestionOnlyGovernance:
    """Verifies duration validation and non-overridable ingestion-only mode under trial profile."""

    def test_validate_cycle_duration_rejects_invalid_types_and_ranges(self):
        # Reject booleans
        with pytest.raises(ValueError, match="Boolean value is not a valid duration"):
            validate_cycle_duration(True)
        with pytest.raises(ValueError, match="Boolean value is not a valid duration"):
            validate_cycle_duration(False)

        # Reject non-numeric strings
        with pytest.raises(ValueError, match="Non-numeric duration value"):
            validate_cycle_duration("invalid_sec")

        # Reject non-finite values
        with pytest.raises(ValueError, match="Non-finite duration value"):
            validate_cycle_duration(float("nan"))
        with pytest.raises(ValueError, match="Non-finite duration value"):
            validate_cycle_duration(float("inf"))
        with pytest.raises(ValueError, match="Non-finite duration value"):
            validate_cycle_duration(-float("inf"))

        # Reject non-positive values
        with pytest.raises(ValueError, match="Non-positive duration value"):
            validate_cycle_duration(0.0)
        with pytest.raises(ValueError, match="Non-positive duration value"):
            validate_cycle_duration(-5.0)

        # Reject excessive values (> 60.0s)
        with pytest.raises(ValueError, match="exceeds maximum allowed ceiling of 60.0s"):
            validate_cycle_duration(60.1)
        with pytest.raises(ValueError, match="exceeds maximum allowed ceiling of 60.0s"):
            validate_cycle_duration(120.0)

        # Accept valid values
        assert validate_cycle_duration(30.0) == 30.0
        assert validate_cycle_duration(60.0) == 60.0
        assert validate_cycle_duration("45.5") == 45.5

    def test_pipeline_rejects_ingestion_only_override_under_trial_scope(self, monkeypatch):
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_SCOPE", SCOPE_TRIAL_2)
        monkeypatch.setenv("ARX_SMART_MONEY_INGESTION_ONLY", "false")

        # Disabling ingestion-only via env var must raise fail-closed
        with pytest.raises(RuntimeError, match="TRIAL_INGESTION_ONLY_NON_OVERRIDABLE"):
            SmartMoneyContinuousPipeline(
                allow_test_mode=True,
                issuers=TRIAL_ISSUERS,
            )

        # Passing ingestion_only=False explicitly must raise fail-closed
        monkeypatch.setenv("ARX_SMART_MONEY_INGESTION_ONLY", "true")
        with pytest.raises(RuntimeError, match="TRIAL_INGESTION_ONLY_NON_OVERRIDABLE"):
            SmartMoneyContinuousPipeline(
                allow_test_mode=True,
                issuers=TRIAL_ISSUERS,
                ingestion_only=False,
            )

    def test_worker_rejects_ingestion_only_override_under_trial_scope(self, monkeypatch):
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_SCOPE", SCOPE_TRIAL_2)
        monkeypatch.setenv("ARX_SMART_MONEY_INGESTION_ONLY", "false")

        with pytest.raises(RuntimeError, match="TRIAL_INGESTION_ONLY_NON_OVERRIDABLE"):
            SmartMoneyIngestionWorker.__new__(SmartMoneyIngestionWorker).__init__()


# ==============================================================================
# SECTION 6: TWO-ISSUER TRIAL SCHEDULER & DISSEMINATION CALENDAR
# ==============================================================================

class TestTwoIssuerTrialScheduler:
    """Verifies timezone-aware 4-slot scheduling, quotas, calendar, and issuer validation."""

    def test_scheduler_slots_and_slot_id_generation(self):
        scheduler = TwoIssuerTrialScheduler(
            start_date=date(2026, 11, 23),
            end_date=date(2026, 11, 25),
        )

        from analyst_dashboard.smart_money.freshness_authority import NY_TZ

        # Slot 1: 09:15 NY
        t1 = datetime(2026, 11, 23, 9, 20, tzinfo=NY_TZ)
        idx1, slot1, _ = scheduler.get_slot_for_time(t1)
        assert idx1 == 1
        assert slot1 == "2026-11-23:SLOT_1:0915"

        # Slot 2: 13:00 NY
        t2 = datetime(2026, 11, 23, 13, 10, tzinfo=NY_TZ)
        idx2, slot2, _ = scheduler.get_slot_for_time(t2)
        assert idx2 == 2
        assert slot2 == "2026-11-23:SLOT_2:1300"

        # Slot 3: 16:30 NY
        t3 = datetime(2026, 11, 23, 16, 45, tzinfo=NY_TZ)
        idx3, slot3, _ = scheduler.get_slot_for_time(t3)
        assert idx3 == 3
        assert slot3 == "2026-11-23:SLOT_3:1630"

        # Slot 4: 20:30 NY
        t4 = datetime(2026, 11, 23, 20, 35, tzinfo=NY_TZ)
        idx4, slot4, _ = scheduler.get_slot_for_time(t4)
        assert idx4 == 4
        assert slot4 == "2026-11-23:SLOT_4:2030"

        # Outside window: 11:00 NY
        t_out = datetime(2026, 11, 23, 11, 0, tzinfo=NY_TZ)
        idx_out, slot_out, _ = scheduler.get_slot_for_time(t_out)
        assert idx_out is None
        assert slot_out is None

    def test_scheduler_timezone_aware_date_and_utc_handling(self):
        """20:30 NY on Nov 23 is 01:30 UTC on Nov 24; scheduler correctly attributes to Nov 23."""
        scheduler = TwoIssuerTrialScheduler(
            start_date=date(2026, 11, 23),
            end_date=date(2026, 11, 25),
        )

        utc_time = datetime(2026, 11, 24, 1, 35, tzinfo=timezone.utc)
        should_run, slot_id, reason = scheduler.evaluate_execution_eligibility(
            now=utc_time,
            issuers=TRIAL_ISSUERS,
        )
        assert should_run is True
        assert slot_id == "2026-11-23:SLOT_4:2030"

    def test_scheduler_skips_weekends_and_federal_holidays(self):
        scheduler = TwoIssuerTrialScheduler(
            start_date=date(2026, 11, 23),
            end_date=date(2026, 11, 29),
        )
        from analyst_dashboard.smart_money.freshness_authority import NY_TZ

        # Saturday
        sat = datetime(2026, 11, 28, 9, 20, tzinfo=NY_TZ)
        should_run, _, reason = scheduler.evaluate_execution_eligibility(sat, issuers=TRIAL_ISSUERS)
        assert should_run is False
        assert "WEEKEND_EXCLUDED" in reason

        # Sunday
        sun = datetime(2026, 11, 29, 9, 20, tzinfo=NY_TZ)
        should_run, _, reason = scheduler.evaluate_execution_eligibility(sun, issuers=TRIAL_ISSUERS)
        assert should_run is False
        assert "WEEKEND_EXCLUDED" in reason

        # Thanksgiving Thursday (Nov 26, 2026)
        thx = datetime(2026, 11, 26, 9, 20, tzinfo=NY_TZ)
        should_run, _, reason = scheduler.evaluate_execution_eligibility(thx, issuers=TRIAL_ISSUERS)
        assert should_run is False
        assert "DISSEMINATION_HOLIDAY_EXCLUDED" in reason

    def test_scheduler_hard_quotas_and_duplicate_slot_rejection(self):
        scheduler = TwoIssuerTrialScheduler(
            start_date=date(2026, 11, 23),
            end_date=date(2026, 11, 25),
        )
        from analyst_dashboard.smart_money.freshness_authority import NY_TZ
        t1 = datetime(2026, 11, 23, 9, 20, tzinfo=NY_TZ)

        # Duplicate slot rejection
        p_dup = {"executed_slots": {"2026-11-23:SLOT_1:0915"}}
        should_run, _, reason = scheduler.evaluate_execution_eligibility(t1, progress=p_dup, issuers=TRIAL_ISSUERS)
        assert should_run is False
        assert "SLOT_ALREADY_EXECUTED" in reason

        # Daily cycle quota: 4 cycles
        p_day_cycles = {"cycles_today": 4}
        should_run, _, reason = scheduler.evaluate_execution_eligibility(t1, progress=p_day_cycles, issuers=TRIAL_ISSUERS)
        assert should_run is False
        assert "DAILY_CYCLE_QUOTA_REACHED" in reason

        # Trial cycle quota: 12 cycles
        p_trial_cycles = {"cycles_trial": 12}
        should_run, _, reason = scheduler.evaluate_execution_eligibility(t1, progress=p_trial_cycles, issuers=TRIAL_ISSUERS)
        assert should_run is False
        assert "TRIAL_CYCLE_QUOTA_REACHED" in reason

        # Daily request quota: 32 requests
        p_day_reqs = {"requests_today": 32}
        should_run, _, reason = scheduler.evaluate_execution_eligibility(t1, progress=p_day_reqs, issuers=TRIAL_ISSUERS)
        assert should_run is False
        assert "DAILY_REQUEST_QUOTA_REACHED" in reason

        # Trial request quota: 96 requests
        p_trial_reqs = {"requests_trial": 96}
        should_run, _, reason = scheduler.evaluate_execution_eligibility(t1, progress=p_trial_reqs, issuers=TRIAL_ISSUERS)
        assert should_run is False
        assert "TRIAL_REQUEST_QUOTA_REACHED" in reason

    def test_scheduler_issuer_validation(self):
        scheduler = TwoIssuerTrialScheduler()

        # Valid NVDA and AAPL pass
        scheduler.validate_issuers(TRIAL_ISSUERS)

        # Missing CIK fails closed
        with pytest.raises(RuntimeError, match="Missing mandatory CIK"):
            scheduler.validate_issuers([{"ticker": "NVDA", "cik": ""}, {"ticker": "AAPL", "cik": "0000320193"}])

        # Wrong CIK fails closed
        with pytest.raises(RuntimeError, match="CIK mismatch"):
            scheduler.validate_issuers([{"ticker": "NVDA", "cik": "0000000001"}, {"ticker": "AAPL", "cik": "0000320193"}])

        # Unauthorized ticker fails closed
        with pytest.raises(RuntimeError, match="Unauthorized ticker"):
            scheduler.validate_issuers([{"ticker": "MSFT", "cik": "0000789019"}, {"ticker": "AAPL", "cik": "0000320193"}])

        # Incorrect count fails closed
        with pytest.raises(RuntimeError, match="strictly requires exactly 2 issuers"):
            scheduler.validate_issuers([{"ticker": "NVDA", "cik": "0001045810"}])


# ==============================================================================
# SECTION 7: QUOTAS, RECOVERY & PERSISTENT PROGRESS REGRESSIONS
# ==============================================================================

class TestTwoIssuerQuotasAndRecoveryRegressions:
    """Verifies quota persistence across restarts/token renewals, restart recovery, and fallback error handling."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        monkeypatch.setenv("SEC_USER_AGENT", "ARX Terminal Research (ops@arxterminal.com)")
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_SCOPE", SCOPE_TRIAL_2)
        monkeypatch.setenv("ARX_SMART_MONEY_INGESTION_ONLY", "true")
        monkeypatch.setenv("ARX_EMERGENCY_STOP_ACTIVE", "false")

    def test_quotas_persisted_across_restarts_and_auth_renewal(self, disposable_db, monkeypatch):
        """Plant 4 audit events today; verify get_trial_progress and scheduler block execution even after new auth grant."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)

        # Record 4 completed cycle events today
        from analyst_dashboard.smart_money.freshness_authority import NY_TZ
        ny_today = datetime.now(NY_TZ).date().isoformat()
        for idx in range(1, 5):
            repo.record_trial_cycle_audit(
                trial_id="ARX-TRIAL-2026-NVDA-AAPL-001",
                slot_id=f"{ny_today}:SLOT_{idx}:0915",
                ny_date=ny_today,
                slot_index=idx,
                status="SUCCESS",
                requests_transmitted=8,
            )

        # Set initial grant
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_ID", "initial-grant-1111")
        progress = repo.get_trial_progress("ARX-TRIAL-2026-NVDA-AAPL-001", ny_today)
        assert progress["cycles_today"] == 4
        assert progress["requests_today"] == 32

        # Simulate renewal with a NEW short-lived authorization grant
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_ID", "renewed-grant-2222")
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_ISSUED_AT", "2026-11-20T12:00:00Z")

        # Create new pipeline instance (restart simulation)
        pipeline = SmartMoneyContinuousPipeline(
            dsn=disposable_db,
            repository=repo,
            backup_engine=make_mock_backup_engine(),
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )

        # Under advisory lock, Step 3 must evaluate eligibility as False due to daily cycle quota
        pipeline._step2_claim_advisory_lock()
        try:
            should_run, reason = pipeline._step3_check_durable_date_eligibility_under_lock()
            assert should_run is False
            assert "DAILY_CYCLE_QUOTA_REACHED" in reason or "DAILY_REQUEST_QUOTA_REACHED" in reason
        finally:
            pipeline._cleanup_lock()

    def test_commit_before_job_completion_prevents_duplicate_slot(self, disposable_db):
        """Plant a committed receipt for slot 1; verify restart in same window skips re-execution."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)
        from analyst_dashboard.smart_money.freshness_authority import NY_TZ
        ny_today = datetime.now(NY_TZ).date().isoformat()
        target_slot = f"{ny_today}:SLOT_1:0915"

        receipt_meta = {
            "receipt_id": str(uuid.uuid4()),
            "interval_start": "2026-11-23T14:15:00Z",
            "interval_end": "2026-11-23T15:00:00Z",
            "certified_watermark": "2026-11-23T15:00:00Z",
            "receipt_version": 1,
            "expected_count": 1,
            "accounted_count": 1,
            "missing_count": 0,
            "material_gap_count": 0,
            "coverage_status": "PARTIAL",
            "failure_reasons": [{"trial_id": "ARX-TRIAL-2026-NVDA-AAPL-001", "slot_id": target_slot}],
            "covered_ciks": ["0001045810", "0000320193"],
            "covered_tickers": ["NVDA", "AAPL"],
            "accession_inventory": ["0001045810-26-000001"],
            "source_hashes": {},
            "reconciliation_hash": "hash123",
            "release_sha": MOCK_SHA,
        }
        repo.save_reconciliation_receipt(receipt_meta)

        progress = repo.get_trial_progress("ARX-TRIAL-2026-NVDA-AAPL-001", ny_today)
        assert target_slot in progress["executed_slots"]
        assert progress["cycles_today"] == 1
        assert progress["requests_today"] == 8

        scheduler = TwoIssuerTrialScheduler(start_date=datetime.now(NY_TZ).date(), end_date=datetime.now(NY_TZ).date())
        slot_time = datetime.now(NY_TZ).replace(hour=9, minute=20)
        should_run, slot_id, reason = scheduler.evaluate_execution_eligibility(
            now=slot_time,
            progress=progress,
            issuers=TRIAL_ISSUERS,
        )
        assert should_run is False
        assert "SLOT_ALREADY_EXECUTED" in reason

    def test_receipt_query_failure_fails_closed(self, disposable_db):
        """Database error during progress or receipt lookup must fail closed with DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)
        pipeline = SmartMoneyContinuousPipeline(
            dsn=disposable_db,
            repository=repo,
            backup_engine=make_mock_backup_engine(),
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )

        pipeline._step2_claim_advisory_lock()
        try:
            with patch.object(repo, "get_trial_progress", side_effect=RuntimeError("DB_CONNECTION_TIMEOUT")):
                with pytest.raises(RuntimeError, match="DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED"):
                    pipeline._step3_check_durable_date_eligibility_under_lock()
        finally:
            pipeline._cleanup_lock()

    def test_get_latest_reconciliation_receipt_no_fallback_to_unrelated(self, disposable_db):
        """When 10-issuer receipt exists, querying for 2-issuer cohort returns None, never rows[0]."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)

        ten_issuer_receipt = {
            "receipt_id": str(uuid.uuid4()),
            "interval_start": "2026-11-20T00:00:00Z",
            "interval_end": "2026-11-20T22:30:00Z",
            "certified_watermark": "2026-11-20T22:30:00Z",
            "receipt_version": 1,
            "expected_count": 10,
            "accounted_count": 10,
            "missing_count": 0,
            "material_gap_count": 0,
            "coverage_status": "COMPLETE",
            "failure_reasons": [],
            "covered_ciks": ["0001045810", "0000320193", "0000789019"],
            "covered_tickers": ["NVDA", "AAPL", "MSFT"],
            "accession_inventory": [],
            "source_hashes": {},
            "reconciliation_hash": "hash_ten",
            "release_sha": MOCK_SHA,
        }
        repo.save_reconciliation_receipt(ten_issuer_receipt)

        # Query for exactly NVDA and AAPL (2 issuers)
        receipt = repo.get_latest_reconciliation_receipt(covered_tickers=["NVDA", "AAPL"])
        assert receipt is None

    def test_prospective_filtering_before_budget_limit(self, monkeypatch):
        """Catalog containing historical filings outside prospective window filters them out before budget counting."""
        pipeline = SmartMoneyContinuousPipeline(
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )

        mock_catalog = {
            "filings": {
                "recent": {
                    "form": ["4", "4", "4", "4", "4", "4", "4"],
                    "accessionNumber": [
                        "0001045810-26-000001",
                        "0001045810-26-000002",
                        "0001045810-26-000003",
                        "0001045810-26-000004",
                        "0001045810-26-000005",
                        "0001045810-26-000006",
                        "0001045810-26-000007",
                    ],
                    "filingDate": ["2026-11-10"] * 5 + ["2026-11-23", "2026-11-23"],
                    "acceptanceDateTime": [
                        "2026-11-10T15:00:00.000Z",
                        "2026-11-10T16:00:00.000Z",
                        "2026-11-10T17:00:00.000Z",
                        "2026-11-10T18:00:00.000Z",
                        "2026-11-10T19:00:00.000Z",
                        "2026-11-23T14:15:00.000Z",
                        "2026-11-23T14:20:00.000Z",
                    ],
                    "primaryDocument": ["form4.xml"] * 7,
                }
            }
        }

        form4_xml = generate_form4_xml(ticker="NVDA", cik="0001045810", date="2026-11-23")

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "submissions" in url:
                resp.json.return_value = mock_catalog
            else:
                resp.text = form4_xml
            return resp

        monkeypatch.setattr(pipeline.outbound_session, "get", mock_get)

        pipeline.trial_scheduler.start_date = date(2026, 11, 23)
        filings = pipeline._fetch_filings_from_sec("NVDA", "0001045810", "NVIDIA CORP", as_of="2026-11-23")
        assert len(filings) == 2
        assert filings[0]["accession_number"] == "0001045810-26-000006"
        assert filings[1]["accession_number"] == "0001045810-26-000007"

    def test_prospective_unseen_filings_exceeding_budget_fails_closed(self, monkeypatch):
        """When more than 3 unseen filings appear strictly within the prospective window, fail closed."""
        pipeline = SmartMoneyContinuousPipeline(
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )

        mock_catalog = {
            "filings": {
                "recent": {
                    "form": ["4", "4", "4", "4"],
                    "accessionNumber": [
                        "0001045810-26-000001",
                        "0001045810-26-000002",
                        "0001045810-26-000003",
                        "0001045810-26-000004",
                    ],
                    "filingDate": ["2026-11-23"] * 4,
                    "acceptanceDateTime": [
                        "2026-11-23T14:15:00.000Z",
                        "2026-11-23T14:20:00.000Z",
                        "2026-11-23T14:25:00.000Z",
                        "2026-11-23T14:30:00.000Z",
                    ],
                    "primaryDocument": ["form4.xml"] * 4,
                }
            }
        }

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = mock_catalog
            return resp

        monkeypatch.setattr(pipeline.outbound_session, "get", mock_get)
        pipeline.trial_scheduler.start_date = date(2026, 11, 23)

        with pytest.raises(RuntimeError, match="SEC_NEW_FILINGS_EXCEED_ISSUER_BUDGET"):
            pipeline._fetch_filings_from_sec("NVDA", "0001045810", "NVIDIA CORP", as_of="2026-11-23")

    def test_fallback_429_immediately_aborts_without_swallowing(self, monkeypatch):
        """A 429 during fallback representation must abort cycle immediately without trying further fallbacks."""
        pipeline = SmartMoneyContinuousPipeline(
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )

        mock_catalog = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "accessionNumber": ["0001045810-26-000001"],
                    "filingDate": ["2026-11-23"],
                    "acceptanceDateTime": ["2026-11-23T14:15:00.000Z"],
                    "primaryDocument": ["sub/form4.xml"],
                }
            }
        }

        request_urls = []

        def mock_get(url, **kwargs):
            request_urls.append(url)
            resp = MagicMock()
            if "submissions" in url:
                resp.status_code = 200
                resp.json.return_value = mock_catalog
                return resp
            if "sub/form4.xml" in url:
                resp.status_code = 200
                resp.text = "<html><body>Not an XML document</body></html>"
                return resp
            if "form4.xml" in url:
                resp.status_code = 429
                return resp
            return resp

        monkeypatch.setattr(pipeline.outbound_session, "get", mock_get)
        pipeline.trial_scheduler.start_date = date(2026, 11, 23)

        with pytest.raises(RuntimeError, match="SEC_RATE_LIMITED_429"):
            pipeline._fetch_filings_from_sec("NVDA", "0001045810", "NVIDIA CORP", as_of="2026-11-23")

        # Confirm that after the 429 on fallback 1, NO .txt request was sent!
        assert not any(u.endswith(".txt") for u in request_urls)

    def test_local_tcp_trickling_server_deadline_timeout(self):
        """Real local TCP stalled server triggers CYCLE_TIMEOUT_EXCEEDED during chunked transfer."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def _serve():
            try:
                conn, _ = server.accept()
                conn.recv(1024)
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n")
                for _ in range(20):
                    time.sleep(0.1)
                    conn.sendall(b"4\r\ntest\r\n")
                conn.sendall(b"0\r\n\r\n")
                conn.close()
            except Exception:
                pass
            finally:
                server.close()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()

        class TrickleAdapter(requests.adapters.HTTPAdapter):
            def send(self, request, **kwargs):
                request.url = f"http://127.0.0.1:{port}/"
                return super().send(request, **kwargs)

        session = ContinuousOutboundSession(allow_test_mode=True)
        session.mount("https://data.sec.gov", TrickleAdapter(max_retries=0))
        session.deadline_monotonic = time.monotonic() + 0.25

        with pytest.raises(RuntimeError, match="CYCLE_TIMEOUT_EXCEEDED"):
            session.get("https://data.sec.gov/test")

    def test_zero_snapshot_publications_or_table_mutations(self, disposable_db):
        """A complete two-issuer cycle makes 0 snapshot method calls and leaves snapshot tables untouched."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)
        scanner = MagicMock()

        pipeline = SmartMoneyContinuousPipeline(
            dsn=disposable_db,
            repository=repo,
            backup_engine=make_mock_backup_engine(),
            scanner=scanner,
            allow_test_mode=True,
            issuers=TRIAL_ISSUERS,
            ingestion_only=True,
        )

        with patch.object(pipeline, "_step6_ingest_form4_filings", return_value=[]):
            res = pipeline.execute_cycle(force=True)

        assert res["status"] == "SUCCESS"
        assert res["snapshot_id"] is None
        # Assert publish_snapshot was never called
        scanner.publish_snapshot.assert_not_called()

        psycopg2 = pytest.importorskip("psycopg2")
        conn = psycopg2.connect(disposable_db)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM scanner_snapshots;")
            snap_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM scanner_active_snapshots;")
            active_count = cur.fetchone()[0]
            assert snap_count == 0
            assert active_count == 0
        conn.close()


class TestCrashSafeTrialAdmission:
    """Production SQL and durable admission regressions, with no SEC egress."""

    @pytest.fixture(autouse=True)
    def env(self, monkeypatch):
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        monkeypatch.setenv("ARX_CONTINUOUS_AUTH_SCOPE", SCOPE_TRIAL_2)
        monkeypatch.setenv("ARX_SMART_MONEY_INGESTION_ONLY", "true")
        monkeypatch.setenv("ARX_RELEASE_SHA", MOCK_SHA)
        monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)

    def pipeline(self, dsn):
        return SmartMoneyContinuousPipeline(
            dsn=dsn, repository=PostgresSmartMoneyRepository(dsn=dsn),
            backup_engine=make_mock_backup_engine(), allow_test_mode=True,
            issuers=TRIAL_ISSUERS, ingestion_only=True,
        )

    def test_production_precommit_on_stock_postgres_and_lock_loss(self, disposable_db):
        """Execute the actual non-test Step 7 guard on stock PostgreSQL."""
        pipeline = self.pipeline(disposable_db)
        pipeline._step2_claim_advisory_lock()
        pipeline.allow_test_mode = False
        try:
            # Auth is isolated here; the production SQL and transaction are real.
            with patch("analyst_dashboard.smart_money.continuous_pipeline.require_continuous_authorization"):
                _, receipt = pipeline._step7_parse_and_reconcile([], "production-guard-valid")
                assert receipt["coverage_status"] == "PARTIAL"
                with pipeline._lock_conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock_all()")
                with pytest.raises(RuntimeError, match="LOCK_LOST_BEFORE_COMMIT"):
                    pipeline._step7_parse_and_reconcile([], "production-guard-lost")
            with pipeline.repository.ingestion_transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM sm_reconciliation_receipts")
                    assert cur.fetchone()[0] == 1  # lost-lock transaction rolled back
        finally:
            pipeline._cleanup_lock()

    def test_other_session_lock_is_not_ownership(self, disposable_db):
        import psycopg2
        pipeline = self.pipeline(disposable_db)
        other = psycopg2.connect(disposable_db)
        pipeline._lock_conn = psycopg2.connect(disposable_db)
        try:
            with other.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(401007311509093371)")
            with pytest.raises(RuntimeError, match="LOCK_LOST_BEFORE_COMMIT"):
                pipeline._require_verified_owner_lock()
            pipeline._lock_conn.close()
            with pytest.raises(RuntimeError, match="LOCK_LOST_BEFORE_COMMIT"):
                pipeline._require_verified_owner_lock()
            pipeline._lock_conn = None
            with pytest.raises(RuntimeError, match="LOCK_LOST_BEFORE_COMMIT"):
                pipeline._require_verified_owner_lock()
        finally:
            other.close()
            pipeline._cleanup_lock()

    def test_reservation_survives_process_death_after_synthetic_send(self, disposable_db, tmp_path):
        """Exit without finally after a local wire send; restart retains full charge."""
        import subprocess
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class Handler(BaseHTTPRequestHandler):
            calls = 0
            def do_GET(self):
                Handler.calls += 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"synthetic SEC catalog")
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        code = '''
import os, sys, requests
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
repo = PostgresSmartMoneyRepository(dsn=sys.argv[1])
repo.reserve_trial_cycle_budget("crash-trial", "2026-11-20:SLOT_1:0915", "2026-11-20")
s = requests.Session()
s.trust_env = False
s.get(sys.argv[2], timeout=3)
os._exit(91)
'''
        try:
            result = subprocess.run([sys.executable, "-c", code, disposable_db,
                                     f"http://127.0.0.1:{server.server_port}/"],
                                    timeout=15, capture_output=True)
            assert result.returncode == 91, result.stderr.decode()
            assert Handler.calls == 1
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)
        progress = repo.get_trial_progress("crash-trial", "2026-11-20")
        assert progress["cycles_today"] == 1
        assert progress["requests_today"] == 8
        with pytest.raises(RuntimeError, match="ALREADY_RESERVED"):
            repo.reserve_trial_cycle_budget("crash-trial", "2026-11-20:SLOT_1:0915", "2026-11-20")

    def test_diagnostics_and_completion_do_not_duplicate_reserved_budget(self, disposable_db):
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)
        slot = "2026-11-20:SLOT_1:0915"
        repo.reserve_trial_cycle_budget("trial", slot, "2026-11-20")
        for _ in range(3):
            repo.record_trial_cycle_audit("trial", slot, "2026-11-20", 1, "SUCCESS", 2)
            repo.record_trial_cycle_audit("trial", "2026-11-20:UNKNOWN_SLOT_1", "2026-11-20", 0, "SKIPPED_NOT_ELIGIBLE", 0)
        progress = repo.get_trial_progress("trial", "2026-11-20")
        assert progress["cycles_trial"] == 1
        assert progress["requests_trial"] == 8

    def test_twelve_reservations_survive_restart_and_enforce_trial_ceiling(self, disposable_db):
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)
        for day in (17, 18, 19):
            ny_date = f"2026-11-{day}"
            for slot in range(1, 5):
                repo.reserve_trial_cycle_budget("trial", f"{ny_date}:SLOT_{slot}:0915", ny_date)
            with pytest.raises(RuntimeError, match="RESERVATION_EXCEEDED"):
                repo.reserve_trial_cycle_budget("trial", f"{ny_date}:EXTRA", ny_date)
        restarted = PostgresSmartMoneyRepository(dsn=disposable_db)
        assert restarted.get_trial_progress("trial", "2026-11-20")["requests_trial"] == 96
        with pytest.raises(RuntimeError, match="RESERVATION_EXCEEDED"):
            restarted.reserve_trial_cycle_budget("trial", "2026-11-20:SLOT_1:0915", "2026-11-20")

    def test_reservation_failure_prevents_egress_and_releases_owner_lock(self, disposable_db):
        pipeline = self.pipeline(disposable_db)
        pipeline.current_slot_id = "2026-11-20:SLOT_1:0915"
        with patch.object(pipeline, "_step1_preflight"), \
             patch.object(pipeline, "_step3_check_durable_date_eligibility_under_lock", return_value=(True, "eligible")), \
             patch.object(pipeline.repository, "reserve_trial_cycle_budget", side_effect=RuntimeError("LEDGER_UNAVAILABLE")), \
             patch.object(pipeline, "_step6_ingest_form4_filings") as ingest:
            with pytest.raises(RuntimeError, match="LEDGER_UNAVAILABLE"):
                pipeline.execute_cycle()
            ingest.assert_not_called()
            assert pipeline._lock_conn is None
        assert pipeline.repository.get_trial_progress(pipeline.trial_scheduler.trial_id, "2026-11-20")["cycles_trial"] == 0

    def test_repeated_scheduler_skips_and_dry_run_do_not_write_audits(self, disposable_db):
        pipeline = self.pipeline(disposable_db)
        with patch.object(pipeline, "_step1_preflight"), \
             patch.object(pipeline, "_step3_check_durable_date_eligibility_under_lock", return_value=(False, "BETWEEN_SLOTS")):
            for _ in range(6):
                assert pipeline.execute_cycle()["status"] == "SKIPPED_NOT_ELIGIBLE"
            assert pipeline.execute_cycle(dry_run=True)["status"] == "DRY_RUN_SUCCESS"
        progress = pipeline.repository.get_trial_progress(pipeline.trial_scheduler.trial_id, "2026-11-20")
        assert progress["cycles_trial"] == progress["requests_trial"] == 0
        with pipeline.repository.ingestion_transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM sm_audit_events")
                assert cur.fetchone()[0] == 0

    def test_earlier_receipt_does_not_suppress_later_trial_slot(self, disposable_db):
        pipeline = self.pipeline(disposable_db)
        # Daily receipt lookup belongs to the daily profile, not four-slot trials.
        with patch.object(pipeline.repository, "get_latest_reconciliation_receipt", side_effect=AssertionError("daily lookup prohibited")), \
             patch.object(pipeline.trial_scheduler, "get_ny_now", return_value=datetime(2026, 11, 20, 13, 5, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))):
            should_run, reason = pipeline._step3_check_durable_date_eligibility_under_lock()
            assert should_run, reason
            assert pipeline.current_slot_id == "2026-11-20:SLOT_2:1300"
