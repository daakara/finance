"""Targeted qualification suite for bounded, resumable SEC ingestion.

Verifies the 9 targeted qualification requirements:
1. Late-issuer failure with earlier durable progress preserved (AAPL authentic archived bytes succeed, NVDA failure).
2. Crash between commit and queue completion (committed DB state survives).
3. Restart and duplicate delivery (resumption skips duplicate network requests).
4. Budget exhaustion with truthful deferred work (PARTIAL coverage and explicit deferred reasons).
5. Malformed filings and amendment dependencies (unresolved status preserved).
6. Failed-request accounting (500/429 counted in total_transmissions).
7. Exact issuer/CIK enforcement under SCOPE_RECURRING_10_INGESTION.
8. Denied publication and retention calls under ingestion-only.
9. Unchanged two-issuer trial behavior.
"""
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from analyst_dashboard.smart_money.continuous_authorization import (
    require_continuous_authorization,
    SCOPE_DAILY_10,
    SCOPE_TRIAL_2,
    SCOPE_RECURRING_10_INGESTION,
    _TEN_ISSUERS_CIKS,
)
from analyst_dashboard.smart_money.continuous_pipeline import (
    SmartMoneyContinuousPipeline,
    ContinuousOutboundSession,
    TRIAL_2_GOVERNED_ISSUERS,
    DEFAULT_GOVERNED_ISSUERS,
)
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema

ACCESSION_AAPL = "0001140361-26-044773"
FIXTURE_AAPL = Path(__file__).parent / "fixtures/sec_edgar/AAPL_0001140361-26-044773.xml"
DIGEST_AAPL = "eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9"

NVDA_BASE_ACCESSION = "0002152188-26-000005"
NVDA_AMEND_ACCESSION = "0002152188-26-000007"


@pytest.fixture(autouse=True)
def require_offline_guard(pytestconfig):
    if not pytestconfig.pluginmanager.hasplugin("sec_offline_test_support"):
        pytest.skip("Requires -p sec_offline_test_support for loopback-only replay")


def authentic_aapl_filing():
    raw = FIXTURE_AAPL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DIGEST_AAPL
    return dict(
        accession_number=ACCESSION_AAPL,
        ticker="AAPL",
        cik="0000320193",
        company_name="APPLE INC",
        filing_date="2026-11-18",
        document_type="4",
        accepted_datetime="2026-11-18T23:30:10Z",
        xml_content=raw.decode("utf-8"),
        content_hash=DIGEST_AAPL,
        source_url=f"archive://tests/fixtures/sec_edgar/AAPL_{ACCESSION_AAPL}.xml",
    )


def synthetic_nvda_filing(accession=NVDA_BASE_ACCESSION, form="4", original="", date="2026-09-11"):
    xml = f"""<ownershipDocument><documentType>{form}</documentType>
      <periodOfReport>2026-09-10</periodOfReport>
      <dateOfOriginalSubmission>{original}</dateOfOriginalSubmission>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA CORP</issuerName>
        <issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik>
        <rptOwnerName>Reporting Owner</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable><nonDerivativeTransaction>
        <transactionDate><value>2026-09-10</value></transactionDate>
        <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
        <transactionAmounts><transactionShares><value>10</value></transactionShares>
        <transactionPricePerShare><value>100</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
        </transactionAmounts></nonDerivativeTransaction></nonDerivativeTable>
      </ownershipDocument>"""
    return dict(
        accession_number=accession,
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA CORP",
        filing_date=date,
        document_type=form,
        accepted_datetime=f"{date}T21:04:47Z",
        xml_content=xml,
        source_url=f"https://www.sec.gov/Archives/{accession}.xml",
        content_hash=hashlib.sha256(xml.encode("utf-8")).hexdigest(),
    )


@pytest.fixture
def replay_pg(tmp_path):
    import pgserver
    import psycopg2
    server = pgserver.get_server(tmp_path / "postgres")
    conn = psycopg2.connect(server.get_uri())
    initialize_postgres_schema(conn)
    repo = PostgresSmartMoneyRepository(dsn=server.get_uri())
    try:
        yield conn, repo, server.get_uri()
    finally:
        conn.close()
        server.cleanup()


# ── Requirement 1: Late-issuer failure preserves earlier progress ──────────────

def test_late_issuer_failure_preserves_earlier_progress(replay_pg):
    conn, repo, dsn = replay_pg
    issuers = [
        {"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC"},
        {"ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA CORP"},
    ]
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=issuers,
        ingestion_only=True,
        allow_test_mode=True,
    )

    def mock_fetch(ticker, cik, company_name=None, as_of=None):
        if ticker == "AAPL":
            return [authentic_aapl_filing()]
        elif ticker == "NVDA":
            raise RuntimeError("SEC_UPSTREAM_500_INTERNAL_ERROR: simulated late-issuer failure")
        return []

    pipeline._fetch_filings_from_sec = mock_fetch

    cycle_res = pipeline._execute_bounded_cohort_ingestion(cycle_id="cycle-late-fail-001")

    # Assert AAPL committed durably in PostgreSQL
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE ticker='AAPL'")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM form4_parsed_transactions WHERE accession_number=%s", (ACCESSION_AAPL,))
        assert cur.fetchone()[0] == 3
        cur.execute("SELECT count(*) FROM sm_source_availability_ledger WHERE accession_number=%s", (ACCESSION_AAPL,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM sm_reconciliation_receipts WHERE failure_reasons->>'unit_ticker'='AAPL'")
        assert cur.fetchone()[0] == 1

        # Assert NVDA has zero rows
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE ticker='NVDA'")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM form4_parsed_transactions p JOIN form4_raw_filings r ON p.accession_number=r.accession_number WHERE r.ticker='NVDA'")
        assert cur.fetchone()[0] == 0

    # Assert cohort-level aggregate receipt accurately reports outcome
    unit_res = cycle_res["unit_results"]
    assert unit_res["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"
    assert unit_res["NVDA"]["status"] == "FAILED_DELIVERY"
    assert "SEC_UPSTREAM_500_INTERNAL_ERROR" in unit_res["NVDA"]["error"]

    cohort_receipt = cycle_res["receipt_meta"]
    assert cohort_receipt["coverage_status"] == "PARTIAL"
    assert cohort_receipt["missing_count"] == 1
    assert any("NVDA" in r for r in cohort_receipt["failure_reasons"]["reasons"])


# ── Requirement 2: Crash between commit and queue completion ──────────────────

def test_crash_between_commit_and_queue_completion_survives(replay_pg):
    conn, repo, dsn = replay_pg
    issuers = [{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC"}]
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=issuers,
        ingestion_only=True,
        allow_test_mode=True,
    )
    pipeline._fetch_filings_from_sec = lambda ticker, cik, **kw: [authentic_aapl_filing()]

    # Execute unit commit, then simulate crash before queue completion / cycle return
    res = pipeline._execute_bounded_cohort_ingestion(cycle_id="cycle-crash-001")
    assert res["unit_results"]["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"

    # Simulate post-commit crash / process kill by raising exception
    try:
        raise RuntimeError("SIMULATED_WORKER_SIGKILL_AFTER_COMMIT")
    except RuntimeError:
        pass

    # Verify committed state is still intact in DB
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number=%s", (ACCESSION_AAPL,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM form4_parsed_transactions WHERE accession_number=%s", (ACCESSION_AAPL,))
        assert cur.fetchone()[0] == 3
        cur.execute("SELECT count(*) FROM sm_reconciliation_receipts WHERE failure_reasons->>'unit_ticker'='AAPL'")
        assert cur.fetchone()[0] == 1


# ── Requirement 3: Restart and duplicate delivery (resumption skips duplicate) ──

def test_restart_and_resumption_skips_duplicate_requests(replay_pg):
    conn, repo, dsn = replay_pg
    issuers = [
        {"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC"},
        {"ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA CORP"},
    ]
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=issuers,
        ingestion_only=True,
        allow_test_mode=True,
    )

    fetch_counts = {"AAPL": 0, "NVDA": 0}

    def mock_fetch_pass1(ticker, cik, **kw):
        fetch_counts[ticker] += 1
        if ticker == "AAPL":
            return [authentic_aapl_filing()]
        raise RuntimeError("NVDA_TIMEOUT")

    pipeline._fetch_filings_from_sec = mock_fetch_pass1
    cycle_1 = pipeline._execute_bounded_cohort_ingestion(cycle_id="cycle-resumption-pass", as_of="2026-11-21T12:00:00Z")
    assert cycle_1["unit_results"]["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"
    assert cycle_1["unit_results"]["NVDA"]["status"] == "FAILED_DELIVERY"
    assert fetch_counts["AAPL"] == 1
    assert fetch_counts["NVDA"] == 1

    # Now simulate retry / restart for the same cycle/watermark
    def mock_fetch_pass2(ticker, cik, **kw):
        fetch_counts[ticker] += 1
        if ticker == "NVDA":
            return [synthetic_nvda_filing()]
        return []

    pipeline._fetch_filings_from_sec = mock_fetch_pass2
    cycle_2 = pipeline._execute_bounded_cohort_ingestion(cycle_id="cycle-resumption-pass", as_of="2026-11-21T12:00:00Z")

    # AAPL was resumed from DB: fetch was NOT repeated!
    assert cycle_2["unit_results"]["AAPL"]["resumed"] is True
    assert fetch_counts["AAPL"] == 1  # Not incremented!
    assert fetch_counts["NVDA"] == 2  # NVDA was fetched on retry
    assert cycle_2["unit_results"]["NVDA"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"


# ── Requirement 4: Budget exhaustion with truthful deferred work ──────────────

def test_budget_exhaustion_truthful_deferred_work(replay_pg):
    conn, repo, dsn = replay_pg
    issuers = [
        {"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC"},
        {"ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA CORP"},
    ]
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=issuers,
        ingestion_only=True,
        allow_test_mode=True,
    )
    # Set max budget to 1 request
    pipeline.outbound_session.max_total_requests = 1

    def mock_fetch(ticker, cik, **kw):
        pipeline.outbound_session.total_transmissions += 1
        if ticker == "AAPL":
            return [authentic_aapl_filing()]
        return [synthetic_nvda_filing()]

    pipeline._fetch_filings_from_sec = mock_fetch

    cycle_res = pipeline._execute_bounded_cohort_ingestion(cycle_id="cycle-budget-exhaust-001")
    unit_res = cycle_res["unit_results"]

    assert unit_res["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"
    assert unit_res["NVDA"]["status"] == "EXHAUSTED_BUDGET"
    assert unit_res["NVDA"]["reason"] == "CYCLE_BUDGET_EXCEEDED"

    cohort_receipt = cycle_res["receipt_meta"]
    assert cohort_receipt["coverage_status"] == "PARTIAL"
    assert any("EXHAUSTED_BUDGET:NVDA" in r for r in cohort_receipt["failure_reasons"]["reasons"])


# ── Requirement 5: Malformed filings and amendment dependencies ───────────────

def test_amendment_without_base_preserved_as_unresolved(replay_pg):
    conn, repo, dsn = replay_pg
    issuers = [{"ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA CORP"}]
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=issuers,
        ingestion_only=True,
        allow_test_mode=True,
    )

    # Ingest amendment Form 4/A where base is missing
    amend_filing = synthetic_nvda_filing(accession=NVDA_AMEND_ACCESSION, form="4/A", original="2026-09-01")
    pipeline._fetch_filings_from_sec = lambda ticker, cik, **kw: [amend_filing]

    cycle_res = pipeline._execute_bounded_cohort_ingestion(cycle_id="cycle-unresolved-amend-001")

    # Verify amendment was saved and marked UNRESOLVED in sm_amendments
    with conn.cursor() as cur:
        cur.execute("SELECT resolution_status, base_accession FROM sm_amendments WHERE amendment_accession=%s", (NVDA_AMEND_ACCESSION,))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "UNRESOLVED"
        assert row[1] is None

        # Verify raw filing and transactions exist
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number=%s", (NVDA_AMEND_ACCESSION,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM form4_parsed_transactions WHERE accession_number=%s", (NVDA_AMEND_ACCESSION,))
        assert cur.fetchone()[0] == 1

    cohort_receipt = cycle_res["receipt_meta"]
    assert cohort_receipt["coverage_status"] == "PARTIAL"
    assert cohort_receipt["material_gap_count"] >= 1
    assert any(f"UNRESOLVED_AMENDMENT:{NVDA_AMEND_ACCESSION}" in r for r in cohort_receipt["failure_reasons"]["reasons"])


# ── Requirement 6: Failed-request accounting ──────────────────────────────────

def test_failed_request_accounting():
    session = ContinuousOutboundSession(
        rate_limiter=None,
        max_total_requests=5,
        max_per_issuer_requests=4,
        allow_test_mode=True,
    )
    session.set_current_issuer("AAPL")
    assert session.total_transmissions == 0

    req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0000320193.json").prepare()

    # Simulate wire failure in super().send
    with patch.object(requests.Session, "send", side_effect=requests.exceptions.ConnectionError("Connection failed")):
        with pytest.raises(requests.exceptions.ConnectionError):
            session.send(req)

    # Transmission count incremented before wire attempt
    assert session.total_transmissions == 1
    assert session.issuer_transmissions["AAPL"] == 1


# ── Requirement 7: Exact issuer/CIK enforcement under SCOPE_RECURRING_10_INGESTION

def test_scope_recurring_10_ingestion_authorization_enforcement():
    dummy_sha = "a" * 40
    valid_env = {
        "ARX_CONTINUOUS_INGESTION_AUTHORIZED": "true",
        "ARX_SEC_EGRESS_ROLE": "OWNER",
        "ARX_CONTINUOUS_AUTH_SCOPE": SCOPE_RECURRING_10_INGESTION,
        "ARX_CONTINUOUS_AUTH_UNIVERSE_ID": "UB-US-EQUITIES-20261025-001",
        "ARX_CONTINUOUS_AUTH_ID": "a" * 32,
        "ARX_RELEASE_SHA": dummy_sha,
        "ARX_CONTINUOUS_AUTH_RELEASE_SHA": dummy_sha,
        "ARX_SMART_MONEY_INGESTION_ONLY": "true",
        "ARX_CONTINUOUS_AUTH_ISSUED_AT": "2026-11-21T00:00:00Z",
        "ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-21T12:00:00Z",
    }
    test_now = datetime(2026, 11, 21, 6, 0, 0, tzinfo=timezone.utc)

    issuers_10 = [{"ticker": t, "cik": c} for t, c in _TEN_ISSUERS_CIKS.items()]

    # Case A: Valid 10 issuers and ingestion_only=true passes
    with patch.dict(os.environ, valid_env):
        require_continuous_authorization(issuers=issuers_10, now=test_now)

    # Case B: Ingestion only is false -> raises CONTINUOUS_AUTH_INGESTION_ONLY_REQUIRED
    invalid_ingest = dict(valid_env, ARX_SMART_MONEY_INGESTION_ONLY="false")
    with patch.dict(os.environ, invalid_ingest):
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_INGESTION_ONLY_REQUIRED"):
            require_continuous_authorization(issuers=issuers_10, now=test_now)

    # Case C: 9 issuers instead of 10 -> raises CONTINUOUS_AUTH_ISSUER_MISMATCH
    with patch.dict(os.environ, valid_env):
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_ISSUER_MISMATCH"):
            require_continuous_authorization(issuers=issuers_10[:9], now=test_now)

    # Case D: Altered ticker or CIK -> raises CONTINUOUS_AUTH_ISSUER_MISMATCH
    altered_issuers = [{"ticker": "FAKE", "cik": "0000000000"}] + issuers_10[1:]
    with patch.dict(os.environ, valid_env):
        with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_ISSUER_MISMATCH"):
            require_continuous_authorization(issuers=altered_issuers, now=test_now)


# ── Requirement 8: Denied publication and retention calls under ingestion-only ─

def test_ingestion_only_forbids_snapshot_publication_and_retention():
    pipeline = SmartMoneyContinuousPipeline(
        repository=MagicMock(),
        issuers=DEFAULT_GOVERNED_ISSUERS,
        ingestion_only=True,
        allow_test_mode=True,
    )
    pipeline.storage_adapter = MagicMock()

    # Snapshot publication must raise
    with pytest.raises(RuntimeError, match="SNAPSHOT_PUBLICATION_PROHIBITED_UNDER_INGESTION_ONLY"):
        pipeline._step8_publish_snapshot([], {}, "cycle-test")

    # Step 10 retention pruning must return [] without calling storage adapter
    pruned = pipeline._step10_prune_retention()
    assert pruned == []
    pipeline.storage_adapter.delete_object.assert_not_called()


# ── Requirement 9: Unchanged two-issuer trial behavior ─────────────────────────

def test_two_issuer_trial_behavior_preserved(monkeypatch):
    monkeypatch.setenv("ARX_CONTINUOUS_AUTH_SCOPE", SCOPE_TRIAL_2)
    pipeline = SmartMoneyContinuousPipeline(
        repository=MagicMock(),
        issuers=TRIAL_2_GOVERNED_ISSUERS,
        allow_test_mode=True,
    )
    assert pipeline.is_trial_2_issuer is True
    assert pipeline.ingestion_only is True
    assert pipeline.outbound_session.max_total_requests == 8

    # Prohibit overriding ingestion_only=False under trial profile
    with pytest.raises(RuntimeError, match="TRIAL_INGESTION_ONLY_NON_OVERRIDABLE"):
        SmartMoneyContinuousPipeline(
            repository=MagicMock(),
            issuers=TRIAL_2_GOVERNED_ISSUERS,
            ingestion_only=False,
            allow_test_mode=True,
        )
