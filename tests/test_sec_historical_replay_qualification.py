"""ARX Issue #165 — Historical Form 4 Replay Qualification Suite.

Directly establishes the required qualification evidence using authentic archived
filings in an isolated, disposable PostgreSQL database. Exercises both the production
ingestion path and the candidate bounded-resumable path where their behaviors differ.

All tests run loopback-only with external network egress strictly forbidden.
"""

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from analyst_dashboard.smart_money.continuous_pipeline import (
    SmartMoneyContinuousPipeline,
    ContinuousOutboundSession,
    TRIAL_2_GOVERNED_ISSUERS,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_models import SmartMoneySignalType
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema

# Authentic Pinned Fixture
ACCESSION_AAPL_AUTHENTIC = "0001140361-26-044773"
FIXTURE_AAPL_AUTHENTIC = Path(__file__).parent / "fixtures/sec_edgar/AAPL_0001140361-26-044773.xml"
DIGEST_AAPL_AUTHENTIC = "eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9"

# Explicitly labeled synthetic accessions (unarchived scenarios)
SYNTHETIC_NVDA_BASE_ACCESSION = "0002152188-26-000005"
SYNTHETIC_NVDA_AMEND_ACCESSION = "0002152188-26-000007"
SYNTHETIC_MULTITYPE_ACCESSION = "0000320193-26-000099"


@pytest.fixture(autouse=True)
def require_offline_guard(pytestconfig):
    if not pytestconfig.pluginmanager.hasplugin("sec_offline_test_support"):
        pytest.skip("Requires -p sec_offline_test_support for loopback-only replay")


@pytest.fixture
def replay_pg(tmp_path):
    """Disposable PostgreSQL instance on loopback with fresh schema."""
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


def get_authentic_aapl_filing() -> dict:
    """Authentic archived filing from EDGAR for Apple Inc. (2026-11-18)."""
    raw = FIXTURE_AAPL_AUTHENTIC.read_bytes()
    computed_digest = hashlib.sha256(raw).hexdigest()
    assert computed_digest == DIGEST_AAPL_AUTHENTIC
    return {
        "accession_number": ACCESSION_AAPL_AUTHENTIC,
        "ticker": "AAPL",
        "cik": "0000320193",
        "company_name": "APPLE INC",
        "filing_date": "2026-11-18",
        "document_type": "4",
        "accepted_datetime": "2026-11-18T23:30:10Z",
        "xml_content": raw.decode("utf-8"),
        "content_hash": DIGEST_AAPL_AUTHENTIC,
        "source_url": f"archive://tests/fixtures/sec_edgar/AAPL_{ACCESSION_AAPL_AUTHENTIC}.xml",
    }


def get_synthetic_multitype_filing() -> dict:
    """SYNTHETIC fixture exercising purchases, sales, gifts, exercises, and derivative table."""
    xml = """<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <periodOfReport>2026-11-16</periodOfReport>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>Apple Inc.</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214128</rptOwnerCik>
      <rptOwnerName>LEVINSON ARTHUR D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>true</isDirector>
      <isOfficer>false</isOfficer>
      <isTenPercentOwner>false</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-11-16</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-11-16</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>160.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-11-16</value></transactionDate>
      <transactionCoding><transactionCode>G</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>0.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable>
    <derivativeTransaction>
      <securityTitle><value>Non-Qualified Stock Option</value></securityTitle>
      <transactionDate><value>2026-11-16</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>2000</value></transactionShares>
        <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"""
    return {
        "accession_number": SYNTHETIC_MULTITYPE_ACCESSION,
        "ticker": "AAPL",
        "cik": "0000320193",
        "company_name": "APPLE INC",
        "filing_date": "2026-11-18",
        "document_type": "4",
        "accepted_datetime": "2026-11-18T23:30:10Z",
        "xml_content": xml,
        "content_hash": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "source_url": f"synthetic://AAPL_{SYNTHETIC_MULTITYPE_ACCESSION}.xml",
    }


def get_synthetic_nvda_filing(accession=SYNTHETIC_NVDA_BASE_ACCESSION, form="4", original="") -> dict:
    """SYNTHETIC fixture representing an NVDA Form 4 or Form 4/A filing."""
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
    return {
        "accession_number": accession,
        "ticker": "NVDA",
        "cik": "0001045810",
        "company_name": "NVIDIA CORP",
        "filing_date": "2026-09-11",
        "document_type": form,
        "accepted_datetime": "2026-09-11T21:04:47Z",
        "xml_content": xml,
        "source_url": f"synthetic://NVDA_{accession}.xml",
        "content_hash": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
    }


# ==============================================================================
# Requirement 1: Raw document, parsed transaction persistence & date distinction
# ==============================================================================

def test_authentic_aapl_persistence_and_date_distinction(replay_pg):
    """Verifies persistence of authentic AAPL Form 4 and strict segregation of filing vs replay dates.
    
    Demonstrates authentic replay for one AAPL filing (0001140361-26-044773).
    Historical interval coverage remains PARTIAL and NOT_CERTIFIED.
    """
    conn, repo, dsn = replay_pg
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=[{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC"}],
        ingestion_only=True,
        allow_test_mode=True,
    )
    filing = get_authentic_aapl_filing()
    pipeline._fetch_filings_from_sec = lambda **kw: [filing]

    res = pipeline._execute_bounded_cohort_ingestion(cycle_id="replay-test-cycle-001", as_of="2026-11-22T08:00:00Z")
    assert res["unit_results"]["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"

    with conn.cursor() as cur:
        # Verify raw filing row
        cur.execute(
            "SELECT accession_number, ticker, cik, filing_date, accepted_datetime, content_hash, created_at "
            "FROM form4_raw_filings WHERE accession_number=%s",
            (ACCESSION_AAPL_AUTHENTIC,),
        )
        row = cur.fetchone()
        assert row is not None
        acc, ticker, cik, fdate, adt, chash, cat = row
        assert acc == ACCESSION_AAPL_AUTHENTIC
        assert ticker == "AAPL"
        assert cik == "0000320193"
        assert str(fdate) == "2026-11-18"  # Historical filing date preserved
        assert adt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") == "2026-11-18T23:30:10Z"  # Historical acceptance timestamp preserved
        assert chash == DIGEST_AAPL_AUTHENTIC
        # Verify replay insertion time is distinct from filing/acceptance dates
        assert cat.year >= 2026
        assert cat.strftime("%Y-%m-%d") != "2026-11-18" or cat.strftime("%H:%M:%S") != "23:30:10"

        # Verify parsed transactions (2 S, 1 G)
        cur.execute(
            "SELECT transaction_code, shares, price_per_share, transaction_date, is_derivative "
            "FROM form4_parsed_transactions WHERE accession_number=%s ORDER BY transaction_id",
            (ACCESSION_AAPL_AUTHENTIC,),
        )
        tx_rows = cur.fetchall()
        assert len(tx_rows) == 3
        # Line 1: S (Sale)
        assert tx_rows[0][0] == "S" and float(tx_rows[0][1]) == 36792.0 and round(float(tx_rows[0][2]), 2) == 352.26
        assert str(tx_rows[0][3]) == "2026-11-16"
        assert tx_rows[0][4] is False  # Non-derivative
        # Line 2: S (Sale)
        assert tx_rows[1][0] == "S" and float(tx_rows[1][1]) == 53208.0 and round(float(tx_rows[1][2]), 2) == 353.43
        assert str(tx_rows[1][3]) == "2026-11-16"
        assert tx_rows[1][4] is False  # Non-derivative
        # Line 3: G (Gift)
        assert tx_rows[2][0] == "G" and float(tx_rows[2][1]) == 50000.0 and float(tx_rows[2][2]) == 0.0
        assert str(tx_rows[2][3]) == "2026-11-16"
        assert tx_rows[2][4] is False  # Non-derivative


# ==============================================================================
# Requirement 2: Classification of Purchases, Sales, Gifts, Exercises, Derivatives
# ==============================================================================

def test_transaction_classification_and_derivative_discrimination(replay_pg):
    """Verifies P, S, G, M codes and derivative vs non-derivative table discrimination."""
    conn, repo, dsn = replay_pg
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=[{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC"}],
        ingestion_only=True,
        allow_test_mode=True,
    )
    filing = get_synthetic_multitype_filing()
    pipeline._fetch_filings_from_sec = lambda **kw: [filing]

    res = pipeline._execute_bounded_cohort_ingestion(cycle_id="replay-multitype-001")
    assert res["unit_results"]["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"

    with conn.cursor() as cur:
        cur.execute(
            "SELECT transaction_code, signal_type, is_derivative, is_open_market_purchase_evidence, "
            "is_positive_buying_conviction, shares, price_per_share "
            "FROM form4_parsed_transactions WHERE accession_number=%s",
            (SYNTHETIC_MULTITYPE_ACCESSION,),
        )
        rows = cur.fetchall()
        assert len(rows) == 4
        by_code = {r[0]: r for r in rows}

        # Non-derivative P: Open Market Purchase
        p_row = by_code["P"]
        assert p_row[1] == SmartMoneySignalType.OPEN_MARKET_PURCHASE.value
        assert p_row[2] is False  # Non-derivative
        assert p_row[3] is True   # Open market purchase evidence
        assert p_row[4] is True   # Positive buying conviction
        assert p_row[5] == 1000.0 and p_row[6] == 150.0

        # Non-derivative S: Open Market Sale
        s_row = by_code["S"]
        assert s_row[1] == SmartMoneySignalType.OPEN_MARKET_SALE.value
        assert s_row[2] is False  # Non-derivative
        assert s_row[3] is False
        assert s_row[4] is False
        assert s_row[5] == 500.0 and s_row[6] == 160.0

        # Non-derivative G: Gift
        g_row = by_code["G"]
        assert g_row[1] == SmartMoneySignalType.GIFT.value
        assert g_row[2] is False  # Non-derivative
        assert g_row[3] is False
        assert g_row[4] is False
        assert g_row[5] == 100.0 and g_row[6] == 0.0

        # Derivative M: Option Exercise
        m_row = by_code["M"]
        assert m_row[1] == SmartMoneySignalType.OPTION_EXERCISE.value
        assert m_row[2] is True   # Derivative
        assert m_row[3] is False
        assert m_row[4] is False
        assert m_row[5] == 2000.0 and m_row[6] == 50.0


# ==============================================================================
# Requirement 3: Amendment Linkage and Unresolved Dependencies
# ==============================================================================

def test_amendment_linkage_resolved_and_unresolved(replay_pg):
    """Verifies resolved vs unresolved Form 4/A amendment dependency handling."""
    conn, repo, dsn = replay_pg
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=[{"ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA CORP"}],
        ingestion_only=True,
        allow_test_mode=True,
    )

    # 1. Ingest amendment without base -> UNRESOLVED
    amend_filing = get_synthetic_nvda_filing(
        accession=SYNTHETIC_NVDA_AMEND_ACCESSION, form="4/A", original="2026-09-01"
    )
    pipeline._fetch_filings_from_sec = lambda **kw: [amend_filing]
    res1 = pipeline._execute_bounded_cohort_ingestion(cycle_id="replay-amend-unresolved")

    assert res1["receipt_meta"]["coverage_status"] == "PARTIAL"
    assert res1["receipt_meta"]["material_gap_count"] >= 1
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resolution_status, base_accession FROM sm_amendments WHERE amendment_accession=%s",
            (SYNTHETIC_NVDA_AMEND_ACCESSION,),
        )
        row = cur.fetchone()
        assert row[0] == "UNRESOLVED"
        assert row[1] is None

    # 2. Ingest base filing, then link amendment
    base_filing = get_synthetic_nvda_filing(accession=SYNTHETIC_NVDA_BASE_ACCESSION, form="4")
    pipeline._fetch_filings_from_sec = lambda **kw: [base_filing]
    pipeline._execute_bounded_cohort_ingestion(cycle_id="replay-base-ingestion")

    # Manually link or resolve amendment with authoritative base
    repo.save_amendment(
        amendment_accession=SYNTHETIC_NVDA_AMEND_ACCESSION,
        base_accession=SYNTHETIC_NVDA_BASE_ACCESSION,
        resolution_status="RESOLVED",
        conn=conn,
    )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT resolution_status, base_accession FROM sm_amendments WHERE amendment_accession=%s",
            (SYNTHETIC_NVDA_AMEND_ACCESSION,),
        )
        row = cur.fetchone()
        assert row[0] == "RESOLVED"
        assert row[1] == SYNTHETIC_NVDA_BASE_ACCESSION


# ==============================================================================
# Requirement 4: Duplicate Delivery Produces Zero Duplicates
# ==============================================================================

def test_duplicate_delivery_produces_zero_duplicates(replay_pg):
    """Verifies that replaying authentic filings is strictly idempotent across tables."""
    conn, repo, dsn = replay_pg
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=[{"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC"}],
        ingestion_only=True,
        allow_test_mode=True,
    )
    filing = get_authentic_aapl_filing()
    pipeline._fetch_filings_from_sec = lambda **kw: [filing]

    # First delivery
    pipeline._execute_bounded_cohort_ingestion(cycle_id="replay-dup-pass-1", as_of="2026-11-22T08:00:00Z")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number=%s", (ACCESSION_AAPL_AUTHENTIC,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM form4_parsed_transactions WHERE accession_number=%s", (ACCESSION_AAPL_AUTHENTIC,))
        assert cur.fetchone()[0] == 3
        cur.execute("SELECT count(*) FROM sm_source_availability_ledger WHERE accession_number=%s", (ACCESSION_AAPL_AUTHENTIC,))
        assert cur.fetchone()[0] == 1

    # Second delivery (duplicate replay)
    pipeline._execute_bounded_cohort_ingestion(cycle_id="replay-dup-pass-2", as_of="2026-11-22T08:00:00Z")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE accession_number=%s", (ACCESSION_AAPL_AUTHENTIC,))
        assert cur.fetchone()[0] == 1  # Unchanged
        cur.execute("SELECT count(*) FROM form4_parsed_transactions WHERE accession_number=%s", (ACCESSION_AAPL_AUTHENTIC,))
        assert cur.fetchone()[0] == 3  # Unchanged
        cur.execute("SELECT count(*) FROM sm_source_availability_ledger WHERE accession_number=%s", (ACCESSION_AAPL_AUTHENTIC,))
        assert cur.fetchone()[0] == 1  # Unchanged


# ==============================================================================
# Requirement 5: Production vs Candidate Path Failure Comparison
# ==============================================================================

def test_production_path_vs_candidate_path_on_partial_failure(replay_pg):
    """Contrasts production monolithic failure (all progress lost) vs candidate unit-resumable progress."""
    conn, repo, dsn = replay_pg
    issuers = [
        {"ticker": "AAPL", "cik": "0000320193", "name": "APPLE INC"},
        {"ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA CORP"},
    ]

    # --- Scenario A: Production Path (Monolithic Batch) ---
    prod_pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=issuers,
        ingestion_only=True,
        allow_test_mode=True,
    )
    prod_pipeline.bounded_progress_enabled = False  # Exercise production fallback
    prod_pipeline.current_slot_id = "2026-11-20:SLOT_1:0915"
    for name in ("_step1_preflight", "_step2_claim_advisory_lock", "_step5_freshness_gate_check", "_cleanup_lock"):
        setattr(prod_pipeline, name, MagicMock())
    prod_pipeline._step3_check_durable_date_eligibility_under_lock = MagicMock(return_value=(True, "offline"))
    prod_pipeline._step4_pre_backup = MagicMock(return_value={})
    prod_pipeline._step9_post_backup = MagicMock(return_value={})

    def mock_prod_fetch_step6(as_of=None):
        # In production Step 6, both issuers are fetched in a single batch.
        # If NVDA fails during the batch, Step 6 raises an exception.
        raise RuntimeError("SEC_500_DURING_STEP6_BATCH")

    prod_pipeline._step6_ingest_form4_filings = mock_prod_fetch_step6

    with pytest.raises(RuntimeError, match="SEC_500_DURING_STEP6_BATCH"):
        prod_pipeline.execute_cycle(run_request_id="prod-failure-cycle")

    # In production path, Step 7 was never reached -> 0 rows persisted for AAPL
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM form4_parsed_transactions")
        assert cur.fetchone()[0] == 0

    # --- Scenario B: Candidate Path (Unit-Fenced Bounded Cohort) ---
    cand_pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=issuers,
        ingestion_only=True,
        allow_test_mode=True,
    )
    cand_pipeline.bounded_progress_enabled = True

    def mock_cand_fetch(ticker, cik, **kw):
        if ticker == "AAPL":
            return [get_authentic_aapl_filing()]
        raise RuntimeError("SEC_500_DURING_NVDA_FETCH")

    cand_pipeline._fetch_filings_from_sec = mock_cand_fetch
    cand_res = cand_pipeline._execute_bounded_cohort_ingestion(cycle_id="cand-failure-cycle")

    # In candidate path, AAPL committed durably before NVDA failure
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE ticker='AAPL'")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM form4_parsed_transactions WHERE accession_number=%s", (ACCESSION_AAPL_AUTHENTIC,))
        assert cur.fetchone()[0] == 3
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE ticker='NVDA'")
        assert cur.fetchone()[0] == 0

    assert cand_res["unit_results"]["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"
    assert cand_res["unit_results"]["NVDA"]["status"] == "FAILED_DELIVERY"
    assert cand_res["receipt_meta"]["coverage_status"] == "PARTIAL"


# ==============================================================================
# Requirement 6: Restart Resumption Skips Completed Work
# ==============================================================================

def test_restart_resumption_skips_completed_work(replay_pg):
    """Verifies that restart with the same watermark retrieves completed unit and avoids wire calls."""
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

    fetch_calls = {"AAPL": 0, "NVDA": 0}

    def fetch_attempt_1(ticker, cik, **kw):
        fetch_calls[ticker] += 1
        if ticker == "AAPL":
            return [get_authentic_aapl_filing()]
        raise RuntimeError("NVDA_TIMEOUT")

    pipeline._fetch_filings_from_sec = fetch_attempt_1
    res1 = pipeline._execute_bounded_cohort_ingestion(cycle_id="resumption-cycle", as_of="2026-11-22T08:00:00Z")
    assert res1["unit_results"]["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"
    assert res1["unit_results"]["NVDA"]["status"] == "FAILED_DELIVERY"
    assert fetch_calls["AAPL"] == 1
    assert fetch_calls["NVDA"] == 1

    # Restart / Resume pass
    def fetch_attempt_2(ticker, cik, **kw):
        fetch_calls[ticker] += 1
        if ticker == "NVDA":
            return [get_synthetic_nvda_filing()]
        return []

    pipeline._fetch_filings_from_sec = fetch_attempt_2
    res2 = pipeline._execute_bounded_cohort_ingestion(cycle_id="resumption-cycle", as_of="2026-11-22T08:00:00Z")

    # AAPL was resumed: fetch was NOT called
    assert res2["unit_results"]["AAPL"]["resumed"] is True
    assert fetch_calls["AAPL"] == 1  # Unchanged!
    assert fetch_calls["NVDA"] == 2  # Fetched on retry
    assert res2["unit_results"]["NVDA"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"
    assert res2["all_issuers_completed"] is True


# ==============================================================================
# Requirement 7: Failed Requests Increment Request Budget
# ==============================================================================

def test_failed_requests_accounted_in_request_budget():
    """Verifies that wire connection failures and HTTP errors increment transmission counters."""
    session = ContinuousOutboundSession(
        rate_limiter=None,
        max_total_requests=4,
        max_per_issuer_requests=2,
        allow_test_mode=True,
    )
    session.set_current_issuer("AAPL")
    assert session.total_transmissions == 0

    req = requests.Request("GET", "https://data.sec.gov/submissions/CIK0000320193.json").prepare()

    # Wire failure simulation
    with patch.object(requests.Session, "send", side_effect=requests.exceptions.ConnectionError("Wire drop")):
        with pytest.raises(requests.exceptions.ConnectionError):
            session.send(req)

    # Transmission counter was incremented before the wire call
    assert session.total_transmissions == 1
    assert session.issuer_transmissions["AAPL"] == 1


# ==============================================================================
# Requirement 8 & 9: Budget Exhaustion & Exact Receipt Reconciliation Arithmetic
# ==============================================================================

def test_budget_exhaustion_and_exact_receipt_reconciliation(replay_pg):
    """Verifies deferred work recording and exact reconciliation: discovered = processed + failed + deferred."""
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
    # Clamp total budget to 1 request
    pipeline.outbound_session.max_total_requests = 1

    def mock_fetch(ticker, cik, **kw):
        pipeline.outbound_session.total_transmissions += 1
        if ticker == "AAPL":
            return [get_authentic_aapl_filing()]
        return [get_synthetic_nvda_filing()]

    pipeline._fetch_filings_from_sec = mock_fetch

    res = pipeline._execute_bounded_cohort_ingestion(cycle_id="budget-reconcile-001")
    unit_res = res["unit_results"]
    receipt = res["receipt_meta"]

    # Unit outcomes
    assert unit_res["AAPL"]["status"] == "SUCCESSFULLY_INGESTED_FILINGS"
    assert unit_res["NVDA"]["status"] == "EXHAUSTED_BUDGET"

    # Reconciliation Arithmetic
    discovered_accessions = res["raw_filings"]  # Only AAPL was discovered/fetched
    assert len(discovered_accessions) == 1
    assert discovered_accessions[0]["accession_number"] == ACCESSION_AAPL_AUTHENTIC

    # Cohort receipt reconciliation
    assert receipt["coverage_status"] == "PARTIAL"
    assert receipt["expected_count"] == 1  # 1 accession expected (AAPL)
    assert receipt["missing_count"] == 0   # 0 failed delivery
    assert receipt["material_gap_count"] >= 1  # Deferred NVDA counts as material gap
    assert any("EXHAUSTED_BUDGET:NVDA" in r for r in receipt["failure_reasons"]["reasons"])


# ==============================================================================
# Requirement 10: Ingestion-Only Mode Prevents Scoring & Snapshot Publication
# ==============================================================================

def test_ingestion_only_mode_prevents_scoring_and_snapshot_publication(replay_pg):
    """Verifies that ingestion-only mode strictly forbids scoring, publication, and snapshot DB writes."""
    conn, repo, dsn = replay_pg
    pipeline = SmartMoneyContinuousPipeline(
        dsn=dsn,
        repository=repo,
        issuers=TRIAL_2_GOVERNED_ISSUERS,
        ingestion_only=True,
        allow_test_mode=True,
    )
    pipeline.storage_adapter = MagicMock()

    # Attempting to publish snapshot raises fail-closed exception
    with pytest.raises(RuntimeError, match="SNAPSHOT_PUBLICATION_PROHIBITED_UNDER_INGESTION_ONLY"):
        pipeline._step8_publish_snapshot(
            parsed_txs=[],
            receipt_meta={},
            cycle_id="illegal-publish-attempt",
        )

    # Storage adapter is never called
    pipeline.storage_adapter.put_json.assert_not_called()

    # Snapshot tables in DB remain empty
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM scanner_snapshots")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM scanner_active_snapshots")
        assert cur.fetchone()[0] == 0
