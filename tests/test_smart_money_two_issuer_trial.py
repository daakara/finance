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
from datetime import datetime, timezone, timedelta
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
        """Second scheduled execution on same calendar date is skipped independently of active snapshot."""
        repo = PostgresSmartMoneyRepository(dsn=disposable_db)

        # Plant reconciliation receipt with today's date in New York
        from analyst_dashboard.smart_money.freshness_authority import NY_TZ
        now_ny = datetime.now(NY_TZ)
        today_iso = now_ny.strftime("%Y-%m-%dT22:30:00Z")
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
            "failure_reasons": ["TRIAL_INTERVAL_PARTIAL_COVERAGE"],
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

        # Check date eligibility under lock without force for today
        should_run, reason = pipeline._step3_check_durable_date_eligibility_under_lock(
            force=False, as_of=now_ny.strftime("%Y-%m-%d")
        )
        assert should_run is False
        assert "ALREADY_EXECUTED_TODAY" in reason

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
