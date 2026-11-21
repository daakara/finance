"""
tests/test_sec_disclosure_postgres_integration.py

Comprehensive PostgreSQL Integration Test Suite for SEC Form 4 Disclosure View (ARX Issue #165).
Uses the actual production PostgreSQL 16 schema initialized via initialize_postgres_schema(conn).

Verifies:
1. Issuer Binding & Cryptographic Receipt Validation (failure_reasons JSONB, canonical hash, cutoff, zero-activity proof).
2. Public Availability Boundaries (cutoff watermark comparisons, pending availability).
3. Amendment Semantics & Ambiguity (ambiguous target, full replacement supersession).
4. Mixed Filing Aggregation Removal (different owners, instruments, codes; aggregates withheld).
5. Gift Consideration (reported 0.0 vs positive; non-finite numbers sanitized).
6. Freshness Separation (served_at vs as_of; never now()) & Zero Read-Path Side Effects (zero egress, zero writes, unheld lock).
"""

import os
import json
import math
import uuid
import datetime
import tempfile
import pytest

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extras import RealDictCursor
pgserver = pytest.importorskip("pgserver")

from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    compute_canonical_receipt_hash,
)
from analyst_dashboard.data.sec_edgar_fetcher import (
    SecEdgarFetcher,
    _clean_numeric,
    _attach_parsed_transactions_to_filings,
)
from api.routes.smart_money import get_sec_filings


@pytest.fixture(scope="module")
def postgres_test_server():
    """Module-scoped ephemeral PostgreSQL 16 server via pgserver with production schema initialized."""
    pgdata = tempfile.mkdtemp(prefix="arx_sec_disclosure_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()

    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)
    conn.close()

    yield uri

    server.cleanup()


@pytest.fixture
def clean_postgres_db(postgres_test_server):
    """Truncates mutable tables between tests while preserving production schema and triggers."""
    conn = psycopg2.connect(postgres_test_server)
    with conn.cursor() as cur:
        cur.execute(
            """
            TRUNCATE TABLE
                form4_parsed_transactions,
                sm_source_availability_ledger,
                sm_amendments,
                sm_reconciliation_receipts,
                form4_raw_filings
            CASCADE;
            """
        )
    conn.commit()
    conn.close()
    return postgres_test_server


def test_clean_numeric_sanitizes_non_finite_inputs():
    """Verify _clean_numeric sanitizes NaN, Inf, and invalid types to None."""
    assert _clean_numeric(None) is None
    assert _clean_numeric("invalid") is None
    assert _clean_numeric(float("nan")) is None
    assert _clean_numeric(float("inf")) is None
    assert _clean_numeric(float("-inf")) is None
    assert _clean_numeric(math.nan) is None
    assert _clean_numeric(math.inf) is None
    assert _clean_numeric("NaN") is None
    assert _clean_numeric("Infinity") is None
    assert _clean_numeric("-Infinity") is None

    assert _clean_numeric(0) == 0.0
    assert _clean_numeric(0.0) == 0.0
    assert _clean_numeric("0.0") == 0.0
    assert _clean_numeric(352.77) == 352.77
    assert _clean_numeric("352.77") == 352.77


def test_postgres_receipt_issuer_binding_and_canonical_hash(clean_postgres_db):
    """
    Test 1: Verify PostgreSQL governed receipt validation:
    - Receipt with wrong ticker/CIK in failure_reasons JSONB is rejected.
    - Receipt with hash mismatch is rejected.
    - Receipt with future watermark relative to cutoff is rejected.
    - Governed valid receipt with matching JSONB issuer binding and valid canonical hash is accepted.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Insert raw filing for AAPL using exact production columns
    cur.execute(
        """
        INSERT INTO form4_raw_filings (
            accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url
        ) VALUES (
            '0001140361-26-044952', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T23:30:21Z', '<xml></xml>',
            '93aa1a0a864d72ccc38dcd00a5ed6258b64634dbde4ab271ea78db8efd3841d0', 'https://sec.gov/aapl.txt'
        );
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (
            accession_number, public_availability_at, source, source_reference
        ) VALUES (
            '0001140361-26-044952', '2026-11-19T23:30:21Z', 'SEC_EDGAR_API', 'https://sec.gov/aapl.txt'
        );
        """
    )

    # 1. Receipt with wrong issuer in JSONB failure_reasons (NVDA instead of AAPL)
    rc_wrong_issuer_id = str(uuid.uuid4())
    wrong_reasons = {
        "ticker": "NVDA",
        "issuer_cik": "0001045810",
        "covered_tickers": ["NVDA"],
        "covered_ciks": ["0001045810"],
        "schema_version": 1,
    }
    cur.execute(
        """
        INSERT INTO sm_reconciliation_receipts (
            receipt_id, interval_start, interval_end, certified_watermark, receipt_version,
            expected_count, accounted_count, missing_count, material_gap_count, coverage_status,
            failure_reasons, reconciliation_hash, release_sha
        ) VALUES (
            %s, '2026-11-01T00:00:00Z', '2026-11-19T23:30:21Z', '2026-11-19T23:30:21Z', 1,
            1, 1, 0, 0, 'COMPLETE',
            %s, '0000000000000000000000000000000000000000000000000000000000000000', '868d7afc'
        );
        """,
        (rc_wrong_issuer_id, json.dumps(wrong_reasons)),
    )
    conn.commit()

    fetcher = SecEdgarFetcher()
    from unittest.mock import patch

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        env = fetcher.get_recent_filings_envelope("AAPL")
        assert env["status"] == "AVAILABLE_WITH_RESULTS"
        # Since the only receipt was for NVDA, coverage status must NOT adopt NVDA's COMPLETE!
        assert env["coverage_status"] == "SUPPORTED_SEC_ISSUER"
        assert env["certified_watermark"] is None

    # 2. Insert valid AAPL receipt with verified canonical hash
    rc_valid_id = str(uuid.uuid4())
    aapl_reasons = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["0001140361-26-044952"],
        "source_hashes": {"0001140361-26-044952": "93aa1a0a864d72ccc38dcd00a5ed6258b64634dbde4ab271ea78db8efd3841d0"},
        "material_gaps": [],
        "schema_version": 1,
    }
    canonical_hash = compute_canonical_receipt_hash(
        schema_version=1,
        ticker="AAPL",
        issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z",
        interval_end="2026-11-19T23:30:21Z",
        certified_watermark="2026-11-19T23:30:21Z",
        receipt_version=2,
        supersedes_receipt_id=None,
        accession_inventory=aapl_reasons["accession_inventory"],
        source_hashes=aapl_reasons["source_hashes"],
        material_gaps=[],
        zero_activity_proof=None,
    )
    cur.execute(
        """
        INSERT INTO sm_reconciliation_receipts (
            receipt_id, interval_start, interval_end, certified_watermark, receipt_version,
            expected_count, accounted_count, missing_count, material_gap_count, coverage_status,
            failure_reasons, reconciliation_hash, release_sha
        ) VALUES (
            %s, '2026-11-01T00:00:00Z', '2026-11-19T23:30:21Z', '2026-11-19T23:30:21Z', 2,
            1, 1, 0, 0, 'PARTIAL',
            %s, %s, '868d7afc'
        );
        """,
        (rc_valid_id, json.dumps(aapl_reasons), canonical_hash),
    )
    conn.commit()

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        env2 = fetcher.get_recent_filings_envelope("AAPL")
        assert env2["status"] == "AVAILABLE_WITH_RESULTS"
        assert env2["coverage_status"] == "PARTIAL"
        assert env2["certified_watermark"] == "2026-11-19T23:30:21Z"

    conn.close()


def test_postgres_public_availability_boundaries(clean_postgres_db):
    """
    Test 2: Verify public availability boundary enforcement:
    - Transactions with public_availability_at <= cutoff are EFFECTIVE.
    - Transactions with public_availability_at > cutoff are PENDING_PUBLIC_AVAILABILITY and is_effective = False.
    - Transactions with null public_availability_at when cutoff is active are PENDING_PUBLIC_AVAILABILITY.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor()

    cutoff = "2026-11-19T23:30:21Z"

    # 1. Filing 1: Valid availability before cutoff
    cur.execute(
        """
        INSERT INTO form4_raw_filings (
            accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url
        ) VALUES ('acc-past', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T20:00:00Z', '<xml></xml>', 'hash1', 'https://sec.gov/1');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES ('acc-past', '2026-11-19T20:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/1');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (
            transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd,
            reporting_owner_name, reporting_owner_role, ownership_type, is_officer, acquired_disposed
        ) VALUES (
            'tx-past', 'acc-past', '2026-11-18', 'S', 1000, 350.0, 350000.0, 'Tim Cook', 'CEO', 'D', true, 'D'
        );
        """
    )

    # 2. Filing 2: Future availability after cutoff
    cur.execute(
        """
        INSERT INTO form4_raw_filings (
            accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url
        ) VALUES ('acc-future', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T23:45:00Z', '<xml></xml>', 'hash2', 'https://sec.gov/2');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES ('acc-future', '2026-11-20T01:00:00Z', 'SEC_EDGAR_API', 'https://sec.gov/2');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (
            transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd,
            reporting_owner_name, reporting_owner_role, ownership_type, is_officer, acquired_disposed
        ) VALUES (
            'tx-future', 'acc-future', '2026-11-19', 'S', 500, 355.0, 177500.0, 'Tim Cook', 'CEO', 'D', true, 'D'
        );
        """
    )

    # Add receipt establishing cutoff
    rc_id = str(uuid.uuid4())
    aapl_reasons = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "accession_inventory": ["acc-past"],
        "source_hashes": {"acc-past": "hash1"},
        "material_gaps": [],
        "schema_version": 1,
    }
    canonical_hash = compute_canonical_receipt_hash(
        schema_version=1,
        ticker="AAPL",
        issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z",
        interval_end=cutoff,
        certified_watermark=cutoff,
        receipt_version=1,
        supersedes_receipt_id=None,
        accession_inventory=aapl_reasons["accession_inventory"],
        source_hashes=aapl_reasons["source_hashes"],
        material_gaps=[],
        zero_activity_proof=None,
    )
    cur.execute(
        """
        INSERT INTO sm_reconciliation_receipts (
            receipt_id, interval_start, interval_end, certified_watermark, receipt_version,
            expected_count, accounted_count, missing_count, material_gap_count, coverage_status,
            failure_reasons, reconciliation_hash, release_sha
        ) VALUES (
            %s, '2026-11-01T00:00:00Z', %s, %s, 1,
            1, 1, 0, 0, 'COMPLETE',
            %s, %s, '868d7afc'
        );
        """,
        (rc_id, cutoff, cutoff, json.dumps(aapl_reasons), canonical_hash),
    )
    conn.commit()

    fetcher = SecEdgarFetcher()
    from unittest.mock import patch

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        env = fetcher.get_recent_filings_envelope("AAPL")
        items_by_acc = {item["accession_number"]: item for item in env["items"]}

        past_item = items_by_acc["acc-past"]
        assert past_item["is_effective"] is True
        assert past_item["effective_status"] == "EFFECTIVE"

        future_item = items_by_acc["acc-future"]
        assert future_item["is_effective"] is False
        assert future_item["effective_status"] == "PENDING_PUBLIC_AVAILABILITY"

    conn.close()


def test_postgres_amendment_supersession_and_ambiguity_semantics(clean_postgres_db):
    """
    Test 3: Verify amendment resolution semantics:
    - Target of ambiguous amendment (AMENDMENT_TARGET_AMBIGUOUS or UNRESOLVED) is marked has_ambiguous_amendment=True, effective_status=AMENDMENT_EFFECT_AMBIGUOUS.
    - Base filing superseded by full replacement is marked is_superseded=True, superseded_by=amendment_accession, effective_status=SUPERSEDED_BY_AMENDMENT.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor()

    # Base filing 1 (ambiguous amendment target)
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-base-ambig', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-10', '2026-11-10T20:00:00Z', '<xml></xml>', 'hash-ambig', 'https://sec.gov/ambig');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES ('tx-base-ambig', 'acc-base-ambig', '2026-11-09', 'S', 1000, 340.0, 340000.0, 'Luca Maestri', 'CFO', 'D');
        """
    )

    # Ambiguous amendment
    cur.execute(
        """
        INSERT INTO sm_amendments (amendment_accession, base_accession, amendment_number, resolution_status)
        VALUES ('acc-amend-ambig', 'acc-base-ambig', 1, 'AMENDMENT_TARGET_AMBIGUOUS');
        """
    )

    # Base filing 2 (cleanly superseded)
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-base-superseded', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-05', '2026-11-05T20:00:00Z', '<xml></xml>', 'hash-sup', 'https://sec.gov/sup');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES ('tx-base-sup', 'acc-base-superseded', '2026-11-04', 'S', 2000, 330.0, 660000.0, 'Luca Maestri', 'CFO', 'D');
        """
    )

    # Full replacement amendment
    cur.execute(
        """
        INSERT INTO sm_amendments (amendment_accession, base_accession, amendment_number, resolution_status)
        VALUES ('acc-amend-full', 'acc-base-superseded', 1, 'FULL_REPLACEMENT');
        """
    )
    conn.commit()

    fetcher = SecEdgarFetcher()
    from unittest.mock import patch

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        env = fetcher.get_recent_filings_envelope("AAPL")
        items_by_acc = {item["accession_number"]: item for item in env["items"]}

        # 1. Ambiguous target check
        ambig_item = items_by_acc["acc-base-ambig"]
        assert ambig_item["has_ambiguous_amendment"] is True
        assert ambig_item["is_effective"] is False
        assert ambig_item["effective_status"] == "AMENDMENT_EFFECT_AMBIGUOUS"
        assert "ambiguous amendment" in ambig_item["amendment_note"].lower()

        # 2. Superseded base check
        superseded_item = items_by_acc["acc-base-superseded"]
        assert superseded_item["is_superseded"] is True
        assert superseded_item["superseded_by"] == "acc-amend-full"
        assert superseded_item["is_effective"] is False
        assert superseded_item["effective_status"] == "SUPERSEDED_BY_AMENDMENT"
        assert "superseded by amendment acc-amend-full" in superseded_item["amendment_note"].lower()

    conn.close()


def test_postgres_mixed_filing_aggregates_withheld(clean_postgres_db):
    """
    Test 4: Verify mixed filing aggregates are withheld:
    - Multi-transaction filing with different transaction codes (e.g. M option exercise and S open market sale) sets is_mixed=True, shares_traded=None, total_value_usd=None.
    - Multi-transaction filing with different reporting owners sets is_mixed=True.
    - Multi-transaction filing with identical owner, identical security title, and identical transaction code safely aggregates.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor()

    # 1. Filing with mixed codes (M exercise + S sale)
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-mixed-codes', '0001045810', 'NVDA', 'NVIDIA Corporation', '2026-11-15', '2026-11-15T21:00:00Z', '<xml></xml>', 'h-mix', 'https://sec.gov/mix');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type, is_derivative)
        VALUES
            ('tx-m', 'acc-mixed-codes', '2026-11-14', 'M', 10000, 0.0, 0.0, 'Colette Kress', 'CFO', 'D', true),
            ('tx-s', 'acc-mixed-codes', '2026-11-14', 'S', 5000, 290.0, 1450000.0, 'Colette Kress', 'CFO', 'D', false);
        """
    )

    # 2. Filing with different owners
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-multi-owners', '0001045810', 'NVDA', 'NVIDIA Corporation', '2026-11-12', '2026-11-12T21:00:00Z', '<xml></xml>', 'h-own', 'https://sec.gov/own');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES
            ('tx-own-1', 'acc-multi-owners', '2026-11-11', 'S', 1000, 290.0, 290000.0, 'Jensen Huang', 'CEO', 'D'),
            ('tx-own-2', 'acc-multi-owners', '2026-11-11', 'S', 2000, 290.0, 580000.0, 'Colette Kress', 'CFO', 'D');
        """
    )

    # 3. Filing with identical owner, identical code S, complete numbers (comparable)
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-comparable', '0001045810', 'NVDA', 'NVIDIA Corporation', '2026-11-10', '2026-11-10T21:00:00Z', '<xml></xml>', 'h-comp', 'https://sec.gov/comp');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES
            ('tx-comp-1', 'acc-comparable', '2026-11-09', 'S', 3000, 290.0, 870000.0, 'Jensen Huang', 'CEO', 'D'),
            ('tx-comp-2', 'acc-comparable', '2026-11-09', 'S', 2000, 290.0, 580000.0, 'Jensen Huang', 'CEO', 'D');
        """
    )
    conn.commit()

    fetcher = SecEdgarFetcher()
    from unittest.mock import patch

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        env = fetcher.get_recent_filings_envelope("NVDA")
        items_by_acc = {item["accession_number"]: item for item in env["items"]}

        # Mixed codes: aggregate withheld
        mixed_item = items_by_acc["acc-mixed-codes"]
        assert mixed_item["is_mixed"] is True
        assert mixed_item["shares_traded"] is None
        assert mixed_item["total_value_usd"] is None
        assert mixed_item["price_per_share"] is None
        assert len(mixed_item["transactions"]) == 2

        # Multiple owners: aggregate withheld
        owner_item = items_by_acc["acc-multi-owners"]
        assert owner_item["is_mixed"] is True
        assert owner_item["shares_traded"] is None
        assert owner_item["total_value_usd"] is None

        # Comparable: safely aggregated
        comp_item = items_by_acc["acc-comparable"]
        assert comp_item["is_mixed"] is False
        assert comp_item["shares_traded"] == 5000.0
        assert comp_item["total_value_usd"] == 1450000.0
        assert comp_item["price_per_share"] == 290.0

    conn.close()


def test_postgres_gift_consideration_and_non_finite_inputs(clean_postgres_db):
    """
    Test 5: Verify gift consideration ($0.00 vs null vs positive) and non-finite numeric sanitation:
    - Code G with 0.0 price is preserved as 0.0.
    - Code G with positive consideration ($25.00) is preserved as 25.00.
    - Missing amounts or non-finite inputs in unit helper _attach_parsed_transactions_to_filings are sanitized to None.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES
            ('acc-g0', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-18', '2026-11-18T20:00:00Z', '<xml></xml>', 'h-g0', 'https://sec.gov/g0'),
            ('acc-gpos', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-16', '2026-11-16T20:00:00Z', '<xml></xml>', 'h-gp', 'https://sec.gov/gp');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES
            ('tx-g0', 'acc-g0', '2026-11-17', 'G', 1000, 0.0, 0.0, 'Tim Cook', 'CEO', 'D'),
            ('tx-gp', 'acc-gpos', '2026-11-15', 'G', 200, 25.0, 5000.0, 'Tim Cook', 'CEO', 'D');
        """
    )
    conn.commit()

    fetcher = SecEdgarFetcher()
    from unittest.mock import patch

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        env = fetcher.get_recent_filings_envelope("AAPL")
        items_by_acc = {item["accession_number"]: item for item in env["items"]}

        # Zero consideration gift
        g0 = items_by_acc["acc-g0"]
        assert g0["is_gift"] is True
        assert g0["price_per_share"] == 0.0
        assert g0["total_value_usd"] == 0.0

        # Positive consideration gift
        gpos = items_by_acc["acc-gpos"]
        assert gpos["is_gift"] is True
        assert gpos["price_per_share"] == 25.0
        assert gpos["total_value_usd"] == 5000.0

    # Also verify non-finite numeric sanitation via helper with synthetic raw dicts
    synthetic_items = [{"accession_number": "acc-nan", "form": "4"}]
    synthetic_txs = [
        {
            "transaction_id": "tx-nan",
            "accession_number": "acc-nan",
            "transaction_date": "2026-11-10",
            "transaction_code": "S",
            "shares": float("nan"),
            "price_per_share": float("inf"),
            "total_value_usd": float("-inf"),
            "reporting_owner_name": "Tim Cook",
            "reporting_owner_role": "CEO",
            "ownership_type": "D",
        }
    ]
    _attach_parsed_transactions_to_filings(synthetic_items, synthetic_txs)
    nan_item = synthetic_items[0]
    assert nan_item["shares_traded"] is None
    assert nan_item["price_per_share"] is None
    assert nan_item["total_value_usd"] is None
    assert nan_item["transactions"][0]["shares"] is None
    assert nan_item["transactions"][0]["price_per_share"] is None
    assert nan_item["transactions"][0]["total_value_usd"] is None

    conn.close()


def test_postgres_freshness_separation_and_zero_read_path_side_effects(clean_postgres_db):
    """
    Test 6: Verify response-generation separation (served_at) and zero read-path side effects:
    - served_at is present and distinct from evidence observation as_of.
    - as_of reflects authentic evidence timestamp (or None), never now().
    - Zero network egress occurs (assert SEC HTTP requests not made).
    - Zero database mutations occur (assert table row counts before == after).
    - SEC advisory lock (401007311509093371) remains completely unheld throughout.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Insert baseline filing with distinct historical timestamp
    historical_pub_at = "2026-11-19T23:30:21Z"
    cur.execute(
        """
        INSERT INTO form4_raw_filings (
            accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url
        ) VALUES (
            'acc-aapl-hist', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T23:30:21Z', '<xml></xml>', 'h-hist', 'https://sec.gov/hist'
        );
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES ('acc-aapl-hist', %s, 'SEC_EDGAR_API', 'https://sec.gov/hist');
        """,
        (historical_pub_at,),
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (
            transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd,
            reporting_owner_name, reporting_owner_role, ownership_type
        ) VALUES (
            'tx-hist-s', 'acc-aapl-hist', '2026-11-17', 'S', 2398, 352.77, 845942.46, 'Jennifer Newstead', 'General Counsel', 'D'
        );
        """
    )
    conn.commit()

    # Capture row counts across all tables before read
    tables = [
        "form4_raw_filings",
        "form4_parsed_transactions",
        "sm_source_availability_ledger",
        "sm_amendments",
        "sm_reconciliation_receipts",
    ]
    counts_before = {}
    for tbl in tables:
        cur.execute(f"SELECT COUNT(*) AS c FROM {tbl};")
        counts_before[tbl] = cur.fetchone()["c"]

    # Verify advisory lock unheld before read
    cur.execute("SELECT pg_try_advisory_lock(401007311509093371) AS acquired;")
    lock_test = cur.fetchone()["acquired"]
    assert lock_test is True, "Advisory lock should be unheld initially"
    cur.execute("SELECT pg_advisory_unlock(401007311509093371);")
    conn.commit()

    fetcher = SecEdgarFetcher()
    from unittest.mock import patch

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False), \
         patch.object(fetcher, "_get", side_effect=AssertionError("Live SEC network egress prohibited on read path!")):

        # 1. Fetcher envelope
        env = fetcher.get_recent_filings_envelope("AAPL")
        assert env["status"] == "AVAILABLE_WITH_RESULTS"
        assert env["as_of"] == historical_pub_at
        assert env["served_at"] is not None
        assert env["served_at"] != env["as_of"]
        # Assert as_of is NOT today's response time
        assert env["as_of"].startswith("2026-11-19")

        # 2. FastAPI API route contract via get_sec_filings
        with patch("api.routes.smart_money.sec_fetcher", fetcher):
            api_resp = get_sec_filings("AAPL")
            assert api_resp["status"] == "AVAILABLE_WITH_RESULTS"
            assert api_resp["as_of"] == historical_pub_at
            assert api_resp["served_at"] is not None
            assert api_resp["served_at"] != api_resp["as_of"]
            assert api_resp["served_at"].endswith("Z")
            assert api_resp["count"] == 1

    # Verify row counts after read: STRICT ZERO WRITES
    for tbl in tables:
        cur.execute(f"SELECT COUNT(*) AS c FROM {tbl};")
        c_after = cur.fetchone()["c"]
        assert c_after == counts_before[tbl], f"Read path caused unexpected mutation in table {tbl} ({counts_before[tbl]} -> {c_after})"

    # Verify advisory lock remains completely unheld
    cur.execute("SELECT pg_try_advisory_lock(401007311509093371) AS acquired;")
    lock_after = cur.fetchone()["acquired"]
    assert lock_after is True, "Advisory lock should remain unheld after read"
    cur.execute("SELECT pg_advisory_unlock(401007311509093371);")
    conn.commit()
    conn.close()


def test_adversarial_receipt_integrity_and_conflicting_identity_rejection(clean_postgres_db):
    """
    Test 7 (Adversarial): Verify rejection of missing/tampered hashes, conflicting issuer identities,
    inverted intervals, future cutoffs, and missing supersession targets.
    Asserts fallback to explicit uncertified/partial disclosure.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    from analyst_dashboard.data.sec_edgar_fetcher import _validate_issuer_reconciliation_receipt

    # 1. Missing hash
    r_missing_hash = {
        "receipt_id": str(uuid.uuid4()),
        "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": "2026-11-19T23:30:21Z",
        "certified_watermark": "2026-11-19T23:30:21Z",
        "receipt_version": 1,
        "coverage_status": "COMPLETE",
        "failure_reasons": {
            "ticker": "AAPL",
            "issuer_cik": "0000320193",
            "zero_activity_proof": {"catalog_bytes": "{}", "source_catalog_hash": "a" * 64},
        },
        "reconciliation_hash": None,
    }
    ok, _, err = _validate_issuer_reconciliation_receipt(r_missing_hash, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "MISSING_OR_INVALID_INTEGRITY_EVIDENCE" in err

    # 2. Tampered / wrong hash
    r_tampered_hash = dict(r_missing_hash)
    r_tampered_hash["reconciliation_hash"] = "f" * 64
    ok, _, err = _validate_issuer_reconciliation_receipt(r_tampered_hash, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "INTEGRITY_HASH_MISMATCH" in err

    # 3. Conflicting issuer identity: internal receipt conflict (AAPL ticker with NVDA CIK)
    r_conflicting = dict(r_missing_hash)
    r_conflicting["failure_reasons"] = {"ticker": "AAPL", "issuer_cik": "0001045810"}
    r_conflicting["reconciliation_hash"] = "0" * 64
    ok, _, err = _validate_issuer_reconciliation_receipt(r_conflicting, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "CONFLICTING_ISSUER_IDENTITY" in err

    # 4. Conflicting requested identity: requested AAPL ticker with NVDA CIK
    ok, _, err = _validate_issuer_reconciliation_receipt(r_tampered_hash, "AAPL", "0001045810", cur=cur)
    assert ok is False
    assert "CONFLICTING_ISSUER_IDENTITY" in err

    # 5. Inverted interval (start > end)
    r_inverted = dict(r_missing_hash)
    r_inverted["interval_start"] = "2026-11-20T00:00:00Z"
    r_inverted["interval_end"] = "2026-11-10T00:00:00Z"
    r_inverted["reconciliation_hash"] = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start=r_inverted["interval_start"], interval_end=r_inverted["interval_end"],
        certified_watermark=r_inverted["certified_watermark"], receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=[], source_hashes={}, material_gaps=[],
        zero_activity_proof={"catalog_bytes": "{}", "source_catalog_hash": "a" * 64},
    )
    ok, _, err = _validate_issuer_reconciliation_receipt(r_inverted, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "INVALID_INTERVAL" in err

    # 6. Watermark postdates evaluation cutoff
    r_future_wm = dict(r_missing_hash)
    r_future_wm["certified_watermark"] = "2026-11-25T00:00:00Z"
    r_future_wm["reconciliation_hash"] = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start=r_future_wm["interval_start"], interval_end=r_future_wm["interval_end"],
        certified_watermark=r_future_wm["certified_watermark"], receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=[], source_hashes={}, material_gaps=[],
        zero_activity_proof={"catalog_bytes": "{}", "source_catalog_hash": "a" * 64},
    )
    ok, _, err = _validate_issuer_reconciliation_receipt(
        r_future_wm, "AAPL", "0000320193", evaluation_cutoff="2026-11-19T23:30:21Z", cur=cur
    )
    assert ok is False
    assert "CUTOFF_EXCEEDED" in err

    # 7. Non-existent superseded receipt
    r_missing_sup = dict(r_missing_hash)
    r_missing_sup["supersedes_receipt_id"] = str(uuid.uuid4())
    r_missing_sup["receipt_version"] = 2
    r_missing_sup["reconciliation_hash"] = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start=r_missing_sup["interval_start"], interval_end=r_missing_sup["interval_end"],
        certified_watermark=r_missing_sup["certified_watermark"], receipt_version=2,
        supersedes_receipt_id=r_missing_sup["supersedes_receipt_id"], accession_inventory=[], source_hashes={}, material_gaps=[],
        zero_activity_proof={"catalog_bytes": "{}", "source_catalog_hash": "a" * 64},
    )
    ok, _, err = _validate_issuer_reconciliation_receipt(r_missing_sup, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "SUPERSEDED_RECEIPT_NOT_FOUND" in err

    conn.close()


def test_postgres_amendment_resolved_lineage_and_future_amendments(clean_postgres_db):
    """
    Test 8: Verify amendment lineage semantics:
    - Legacy RESOLVED amendment lineage is scope-unverified; RESOLVED cannot imply verified amendment effect.
    - Base filing target of RESOLVED is flagged has_ambiguous_amendment=True, effective_status=AMENDMENT_EFFECT_AMBIGUOUS, is_effective=False.
    - Amendment filing itself with RESOLVED is effective_status=SCOPE_UNVERIFIED, is_effective=False.
    - Future amendment (public_availability > cutoff) cannot retroactively supersede base filing at cutoff watermark.
    - Partial correction marks base filing as has_partial_amendment=True without full supersession.
    - Offset timestamps are normalized (e.g. -05:00 converted to UTC Z) before cutoff comparison.
    - Missing availability fails closed to PENDING_PUBLIC_AVAILABILITY.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor()

    cutoff_wm = "2026-11-19T23:30:21Z"

    # 1. Base filing 1 & Legacy RESOLVED amendment
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES
            ('acc-base-resolved', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-10', '2026-11-10T18:00:00Z', '<xml></xml>', 'h-res-b', 'https://sec.gov/res-b'),
            ('acc-amend-resolved', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-12', '2026-11-12T18:00:00Z', '<xml></xml>', 'h-res-a', 'https://sec.gov/res-a');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES
            ('acc-base-resolved', '2026-11-10T18:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/res-b'),
            ('acc-amend-resolved', '2026-11-12T18:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/res-a');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES
            ('tx-res-b', 'acc-base-resolved', '2026-11-09', 'S', 1000, 340.0, 340000.0, 'Tim Cook', 'CEO', 'D'),
            ('tx-res-a', 'acc-amend-resolved', '2026-11-09', 'S', 1200, 340.0, 408000.0, 'Tim Cook', 'CEO', 'D');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_amendments (amendment_accession, base_accession, amendment_number, resolution_status)
        VALUES ('acc-amend-resolved', 'acc-base-resolved', 1, 'RESOLVED');
        """
    )

    # 2. Base filing 2 & Future Amendment (FULL_REPLACEMENT available AFTER cutoff)
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES
            ('acc-base-fut', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-15', '2026-11-15T18:00:00Z', '<xml></xml>', 'h-fut-b', 'https://sec.gov/fut-b'),
            ('acc-amend-fut', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-20', '2026-11-20T10:00:00Z', '<xml></xml>', 'h-fut-a', 'https://sec.gov/fut-a');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES
            ('acc-base-fut', '2026-11-15T18:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/fut-b'),
            ('acc-amend-fut', '2026-11-20T10:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/fut-a');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES
            ('tx-fut-b', 'acc-base-fut', '2026-11-14', 'S', 2000, 345.0, 690000.0, 'Tim Cook', 'CEO', 'D'),
            ('tx-fut-a', 'acc-amend-fut', '2026-11-14', 'S', 2500, 345.0, 862500.0, 'Tim Cook', 'CEO', 'D');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_amendments (amendment_accession, base_accession, amendment_number, resolution_status)
        VALUES ('acc-amend-fut', 'acc-base-fut', 1, 'FULL_REPLACEMENT');
        """
    )

    # 3. Base filing 3 & Partial Correction
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES
            ('acc-base-part', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-16', '2026-11-16T18:00:00Z', '<xml></xml>', 'h-prt-b', 'https://sec.gov/prt-b'),
            ('acc-amend-part', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-18', '2026-11-18T18:00:00Z', '<xml></xml>', 'h-prt-a', 'https://sec.gov/prt-a');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES
            ('acc-base-part', '2026-11-16T18:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/prt-b'),
            ('acc-amend-part', '2026-11-18T18:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/prt-a');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES
            ('tx-prt-b', 'acc-base-part', '2026-11-15', 'S', 3000, 342.0, 1026000.0, 'Tim Cook', 'CEO', 'D'),
            ('tx-prt-a', 'acc-amend-part', '2026-11-15', 'S', 3000, 344.0, 1032000.0, 'Tim Cook', 'CEO', 'D');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_amendments (amendment_accession, base_accession, amendment_number, resolution_status)
        VALUES ('acc-amend-part', 'acc-base-part', 1, 'PARTIAL_CORRECTION');
        """
    )

    # 4. Filing 4 with offset timestamp: 2026-11-19T18:30:21-05:00 converts to 2026-11-19T23:30:21Z (<= cutoff)
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-offset-ok', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T18:30:21-05:00', '<xml></xml>', 'h-off-ok', 'https://sec.gov/off-ok');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES ('acc-offset-ok', '2026-11-19T18:30:21-05:00', 'SEC_EDGAR_API', 'https://sec.gov/off-ok');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES ('tx-off-ok', 'acc-offset-ok', '2026-11-19', 'S', 500, 350.0, 175000.0, 'Tim Cook', 'CEO', 'D');
        """
    )

    # 5. Filing 5 with missing availability ledger entry (accepted_datetime postdates cutoff)
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-no-avail', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-20T10:00:00Z', '<xml></xml>', 'h-acc-no-avail', 'https://sec.gov/no-av');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES ('tx-no-av', 'acc-no-avail', '2026-11-19', 'S', 500, 350.0, 175000.0, 'Tim Cook', 'CEO', 'D');
        """
    )

    # Add certified reconciliation receipt certifying cutoff
    all_accs = [
        "acc-base-resolved", "acc-amend-resolved",
        "acc-base-fut",
        "acc-base-part", "acc-amend-part",
        "acc-offset-ok",
    ]
    expected_hashes = {
        "acc-base-resolved": "h-res-b",
        "acc-amend-resolved": "h-res-a",
        "acc-base-fut": "h-fut-b",
        "acc-base-part": "h-prt-b",
        "acc-amend-part": "h-prt-a",
        "acc-offset-ok": "h-off-ok",
    }
    reasons = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": all_accs,
        "source_hashes": expected_hashes,
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z", interval_end=cutoff_wm,
        certified_watermark=cutoff_wm, receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=all_accs,
        source_hashes=reasons["source_hashes"], material_gaps=[],
        zero_activity_proof=None,
    )
    cur.execute(
        """
        INSERT INTO sm_reconciliation_receipts (
            receipt_id, interval_start, interval_end, certified_watermark, receipt_version,
            expected_count, accounted_count, missing_count, material_gap_count, coverage_status,
            failure_reasons, reconciliation_hash, release_sha
        ) VALUES (
            %s, '2026-11-01T00:00:00Z', %s, %s, 1,
            %s, %s, 0, 0, 'COMPLETE',
            %s, %s, '868d7afc'
        );
        """,
        (str(uuid.uuid4()), cutoff_wm, cutoff_wm, len(all_accs), len(all_accs), json.dumps(reasons), rec_hash),
    )
    conn.commit()

    fetcher = SecEdgarFetcher()
    from unittest.mock import patch

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        env = fetcher.get_recent_filings_envelope("AAPL")
        items = {item["accession_number"]: item for item in env["items"]}

        # 1. Base filing target of legacy RESOLVED cannot imply verified amendment effect
        res_base = items["acc-base-resolved"]
        assert res_base["has_ambiguous_amendment"] is True
        assert res_base["is_effective"] is False
        assert res_base["effective_status"] == "AMENDMENT_EFFECT_AMBIGUOUS"
        assert "legacy resolved" in res_base["amendment_note"].lower()

        # Amendment itself with RESOLVED is scope-unverified
        res_am = items["acc-amend-resolved"]
        assert res_am["is_effective"] is False
        assert res_am["effective_status"] == "SCOPE_UNVERIFIED"

        # 2. Future amendment: base filing remains effective at watermark cutoff
        fut_base = items["acc-base-fut"]
        assert fut_base["is_superseded"] is False
        assert fut_base["is_effective"] is True
        assert fut_base["effective_status"] == "EFFECTIVE"
        assert "postdates cutoff" in fut_base["amendment_note"].lower()

        # The future amendment itself is pending availability
        fut_am = items["acc-amend-fut"]
        assert fut_am["is_effective"] is False
        assert fut_am["effective_status"] == "PENDING_PUBLIC_AVAILABILITY"

        # 3. Partial correction: base filing has has_partial_amendment=True without full supersession
        part_base = items["acc-base-part"]
        assert part_base["is_superseded"] is False
        assert part_base["has_partial_amendment"] is True
        assert part_base["is_effective"] is False
        assert part_base["effective_status"] == "PARTIAL_CORRECTION"

        # 4. Offset timestamp normalized properly: -05:00 converted to UTC Z <= cutoff
        offset_item = items["acc-offset-ok"]
        assert offset_item["is_effective"] is True
        assert offset_item["effective_status"] == "EFFECTIVE"

        # 5. Missing availability fails closed to PENDING_PUBLIC_AVAILABILITY
        no_avail = items["acc-no-avail"]
        assert no_avail["is_effective"] is False
        assert no_avail["effective_status"] == "PENDING_PUBLIC_AVAILABILITY"

    conn.close()


def test_adversarial_query_failures_fail_closed(clean_postgres_db):
    """
    Test 10 (Adversarial): Verify that database query failures during inventory,
    supersession, or superseding checks fail closed:
    - Never log an exception and subsequently return valid=True.
    - _validate_issuer_reconciliation_receipt returns (False, None, VERIFICATION_UNAVAILABLE: ...).
    - get_recent_filings_envelope leaves coverage_status = "SUPPORTED_SEC_ISSUER" and certified_watermark = None.
    - Historical raw filings and parsed transactions remain accessible to the caller.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    from analyst_dashboard.data.sec_edgar_fetcher import (
        _validate_issuer_reconciliation_receipt,
        SecEdgarFetcher,
    )
    from analyst_dashboard.smart_money.storage.postgres_repository import compute_canonical_receipt_hash
    from unittest.mock import MagicMock, patch

    # Insert baseline filing & availability ledger
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-fail-closed', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T20:00:00Z', '<xml></xml>', 'h-fc', 'https://sec.gov/fc');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES ('acc-fail-closed', '2026-11-19T20:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/fc');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES ('tx-fc', 'acc-fail-closed', '2026-11-18', 'S', 100, 350.0, 35000.0, 'Tim Cook', 'CEO', 'D');
        """
    )
    conn.commit()

    # Valid receipt metadata
    r_id = str(uuid.uuid4())
    sup_id = str(uuid.uuid4())
    reasons = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["acc-fail-closed"],
        "source_hashes": {"acc-fail-closed": "h-fc"},
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash = compute_canonical_receipt_hash(
        schema_version=1,
        ticker="AAPL",
        issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z",
        interval_end="2026-11-19T23:30:21Z",
        certified_watermark="2026-11-19T23:30:21Z",
        receipt_version=2,
        supersedes_receipt_id=sup_id,
        accession_inventory=reasons["accession_inventory"],
        source_hashes=reasons["source_hashes"],
        material_gaps=[],
        zero_activity_proof=None,
    )
    receipt = {
        "receipt_id": r_id,
        "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": "2026-11-19T23:30:21Z",
        "certified_watermark": "2026-11-19T23:30:21Z",
        "receipt_version": 2,
        "supersedes_receipt_id": sup_id,
        "coverage_status": "COMPLETE",
        "failure_reasons": reasons,
        "reconciliation_hash": rec_hash,
    }

    # 1. Inventory query failure (supersedes_receipt_id is None, inventory query fails)
    receipt_inv_fail = dict(receipt)
    receipt_inv_fail["supersedes_receipt_id"] = None
    receipt_inv_fail["receipt_version"] = 1
    receipt_inv_fail["reconciliation_hash"] = compute_canonical_receipt_hash(
        schema_version=1,
        ticker="AAPL",
        issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z",
        interval_end="2026-11-19T23:30:21Z",
        certified_watermark="2026-11-19T23:30:21Z",
        receipt_version=1,
        supersedes_receipt_id=None,
        accession_inventory=reasons["accession_inventory"],
        source_hashes=reasons["source_hashes"],
        material_gaps=[],
        zero_activity_proof=None,
    )
    def cur_execute_inv_fail(sql, params=None):
        if "FROM form4_raw_filings" in str(sql):
            raise psycopg2.OperationalError("database connection lost during inventory query")
        return None
    mock_cur_inv = MagicMock()
    mock_cur_inv.execute.side_effect = cur_execute_inv_fail
    mock_cur_inv.fetchone.return_value = None  # for superseding check
    ok, valid_rc, err = _validate_issuer_reconciliation_receipt(receipt_inv_fail, "AAPL", "0000320193", cur=mock_cur_inv)
    assert ok is False
    assert valid_rc is None
    assert "VERIFICATION_UNAVAILABLE: inventory query failed" in err

    # 2. Supersession query failure
    mock_cur_sup = MagicMock()
    def cur_execute_side_effect(sql, params=None):
        if "supersedes_receipt_id" in str(sql) or "receipt_version FROM sm_reconciliation_receipts" in str(sql):
            raise psycopg2.OperationalError("deadlock detected during supersession query")
        return None
    mock_cur_sup.execute.side_effect = cur_execute_side_effect
    mock_cur_sup.fetchall.return_value = [{"accession_number": "acc-fail-closed", "cik": "0000320193", "ticker": "AAPL", "content_hash": "h-fc", "public_availability_at": "2026-11-19T20:05:00Z"}]
    mock_cur_sup.fetchone.return_value = {"cnt": 0}
    ok, valid_rc, err = _validate_issuer_reconciliation_receipt(receipt, "AAPL", "0000320193", cur=mock_cur_sup)
    assert ok is False
    assert valid_rc is None
    assert "VERIFICATION_UNAVAILABLE: supersession check query failed" in err

    # 3. Superseding check query failure
    receipt_no_sup = dict(receipt)
    receipt_no_sup["supersedes_receipt_id"] = None
    receipt_no_sup["receipt_version"] = 1
    receipt_no_sup["reconciliation_hash"] = compute_canonical_receipt_hash(
        schema_version=1,
        ticker="AAPL",
        issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z",
        interval_end="2026-11-19T23:30:21Z",
        certified_watermark="2026-11-19T23:30:21Z",
        receipt_version=1,
        supersedes_receipt_id=None,
        accession_inventory=reasons["accession_inventory"],
        source_hashes=reasons["source_hashes"],
        material_gaps=[],
        zero_activity_proof=None,
    )
    def cur_execute_sup2(sql, params=None):
        if "WHERE supersedes_receipt_id =" in str(sql):
            raise psycopg2.OperationalError("timeout on superseding check query")
        return None
    mock_cur_sup2 = MagicMock()
    mock_cur_sup2.execute.side_effect = cur_execute_sup2
    mock_cur_sup2.fetchall.return_value = [{"accession_number": "acc-fail-closed", "cik": "0000320193", "ticker": "AAPL", "content_hash": "h-fc", "public_availability_at": "2026-11-19T20:05:00Z"}]
    mock_cur_sup2.fetchone.return_value = {"cnt": 0}
    ok, valid_rc, err = _validate_issuer_reconciliation_receipt(receipt_no_sup, "AAPL", "0000320193", cur=mock_cur_sup2)
    assert ok is False
    assert valid_rc is None
    assert "VERIFICATION_UNAVAILABLE: superseding check query failed" in err

    # 4. End-to-end disclosure call verification
    cur.execute(
        """
        INSERT INTO sm_reconciliation_receipts (
            receipt_id, interval_start, interval_end, certified_watermark, receipt_version,
            expected_count, accounted_count, missing_count, material_gap_count, coverage_status,
            failure_reasons, reconciliation_hash, release_sha
        ) VALUES (
            %s, '2026-11-01T00:00:00Z', '2026-11-19T23:30:21Z', '2026-11-19T23:30:21Z', 1,
            1, 1, 0, 0, 'COMPLETE',
            %s, %s, '868d7afc'
        );
        """,
        (r_id, json.dumps(receipt_no_sup["failure_reasons"]), receipt_no_sup["reconciliation_hash"]),
    )
    conn.commit()

    fetcher = SecEdgarFetcher()
    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False), \
         patch("analyst_dashboard.data.sec_edgar_fetcher._validate_issuer_reconciliation_receipt", return_value=(False, None, "VERIFICATION_UNAVAILABLE: inventory query failed: db error")):

        env = fetcher.get_recent_filings_envelope("AAPL")
        assert env["status"] == "AVAILABLE_WITH_RESULTS"
        assert env["coverage_status"] == "SUPPORTED_SEC_ISSUER"
        assert env["certified_watermark"] is None
        # Raw filings remain available
        assert len(env["items"]) >= 1
        assert any(item["accession_number"] == "acc-fail-closed" for item in env["items"])

    conn.close()


def test_adversarial_issuer_identity_cross_mismatches(clean_postgres_db):
    """
    Test 11 (Adversarial): Verify rejection of conflicting ticker/CIK identities
    even when one identifier matches under the issuer contract:
    - Matching ticker (AAPL) but accession has wrong CIK (NVDA CIK 0001045810).
    - Matching CIK (0000320193) but accession has wrong ticker (NVDA).
    - Conflicting ticker/CIK contract within receipt.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    from analyst_dashboard.data.sec_edgar_fetcher import _validate_issuer_reconciliation_receipt
    from analyst_dashboard.smart_money.storage.postgres_repository import compute_canonical_receipt_hash

    # Filing with AAPL ticker but NVDA CIK
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-aapl-wrong-cik', '0001045810', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T20:00:00Z', '<xml></xml>', 'h-awc', 'https://sec.gov/awc');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES ('acc-aapl-wrong-cik', '2026-11-19T20:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/awc');
        """
    )

    # Filing with AAPL CIK but NVDA ticker
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES ('acc-aapl-wrong-sym', '0000320193', 'NVDA', 'Apple Inc.', '2026-11-19', '2026-11-19T20:00:00Z', '<xml></xml>', 'h-aws', 'https://sec.gov/aws');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES ('acc-aapl-wrong-sym', '2026-11-19T20:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/aws');
        """
    )
    conn.commit()

    # 1. Receipt inventorying accession with matching ticker but wrong CIK
    reasons_wrong_cik = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["acc-aapl-wrong-cik"],
        "source_hashes": {"acc-aapl-wrong-cik": "h-awc"},
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash1 = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z", interval_end="2026-11-19T23:30:21Z",
        certified_watermark="2026-11-19T23:30:21Z", receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=reasons_wrong_cik["accession_inventory"],
        source_hashes=reasons_wrong_cik["source_hashes"], material_gaps=[], zero_activity_proof=None,
    )
    receipt1 = {
        "receipt_id": str(uuid.uuid4()),
        "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": "2026-11-19T23:30:21Z",
        "certified_watermark": "2026-11-19T23:30:21Z",
        "receipt_version": 1,
        "coverage_status": "COMPLETE",
        "failure_reasons": reasons_wrong_cik,
        "reconciliation_hash": rec_hash1,
    }
    ok, _, err = _validate_issuer_reconciliation_receipt(receipt1, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "INVENTORY_ISSUER_MISMATCH" in err or "CONFLICTING_ISSUER_IDENTITY" in err
    assert "0001045810" in err

    # 2. Receipt inventorying accession with matching CIK but wrong ticker
    reasons_wrong_sym = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["acc-aapl-wrong-sym"],
        "source_hashes": {"acc-aapl-wrong-sym": "h-aws"},
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash2 = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z", interval_end="2026-11-19T23:30:21Z",
        certified_watermark="2026-11-19T23:30:21Z", receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=reasons_wrong_sym["accession_inventory"],
        source_hashes=reasons_wrong_sym["source_hashes"], material_gaps=[], zero_activity_proof=None,
    )
    receipt2 = {
        "receipt_id": str(uuid.uuid4()),
        "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": "2026-11-19T23:30:21Z",
        "certified_watermark": "2026-11-19T23:30:21Z",
        "receipt_version": 1,
        "coverage_status": "COMPLETE",
        "failure_reasons": reasons_wrong_sym,
        "reconciliation_hash": rec_hash2,
    }
    ok, _, err = _validate_issuer_reconciliation_receipt(receipt2, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "INVENTORY_ISSUER_MISMATCH" in err or "CONFLICTING_ISSUER_IDENTITY" in err
    assert "NVDA" in err

    conn.close()


def test_adversarial_accession_evidence_and_availability(clean_postgres_db):
    """
    Test 12 (Adversarial): Verify that accession-level evidence is strictly enforced:
    - Missing accession source-hash binding in receipt -> rejected.
    - Mismatched content hash between receipt binding and database -> rejected.
    - Missing public availability ledger entry -> rejected.
    - Availability postdating certified watermark or cutoff -> rejected.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    from analyst_dashboard.data.sec_edgar_fetcher import _validate_issuer_reconciliation_receipt
    from analyst_dashboard.smart_money.storage.postgres_repository import compute_canonical_receipt_hash

    # Base filings
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES
            ('acc-evidence-ok', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T20:00:00Z', '<xml></xml>', 'hash-exact-match', 'https://sec.gov/e1'),
            ('acc-no-avail-entry', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T20:00:00Z', '<xml></xml>', 'hash-no-avail', 'https://sec.gov/e2'),
            ('acc-avail-late', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T20:00:00Z', '<xml></xml>', 'hash-late', 'https://sec.gov/e3');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES
            ('acc-evidence-ok', '2026-11-19T20:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/e1'),
            ('acc-avail-late', '2026-11-20T01:00:00Z', 'SEC_EDGAR_API', 'https://sec.gov/e3');
        """
    )
    conn.commit()

    watermark = "2026-11-19T23:30:21Z"

    # A. Missing accession hash binding in source_hashes
    reasons_missing_hash = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["acc-evidence-ok"],
        "source_hashes": {},
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash_a = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z", interval_end=watermark,
        certified_watermark=watermark, receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=reasons_missing_hash["accession_inventory"],
        source_hashes=reasons_missing_hash["source_hashes"], material_gaps=[], zero_activity_proof=None,
    )
    receipt_a = {
        "receipt_id": str(uuid.uuid4()), "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": watermark, "certified_watermark": watermark,
        "receipt_version": 1, "coverage_status": "COMPLETE",
        "failure_reasons": reasons_missing_hash, "reconciliation_hash": rec_hash_a,
    }
    ok, _, err = _validate_issuer_reconciliation_receipt(receipt_a, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "MISSING_ACCESSION_SOURCE_HASH" in err

    # B. Mismatched content hash
    reasons_mismatched_hash = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["acc-evidence-ok"],
        "source_hashes": {"acc-evidence-ok": "tampered-hash-does-not-match-db"},
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash_b = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z", interval_end=watermark,
        certified_watermark=watermark, receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=reasons_mismatched_hash["accession_inventory"],
        source_hashes=reasons_mismatched_hash["source_hashes"], material_gaps=[], zero_activity_proof=None,
    )
    receipt_b = {
        "receipt_id": str(uuid.uuid4()), "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": watermark, "certified_watermark": watermark,
        "receipt_version": 1, "coverage_status": "COMPLETE",
        "failure_reasons": reasons_mismatched_hash, "reconciliation_hash": rec_hash_b,
    }
    ok, _, err = _validate_issuer_reconciliation_receipt(receipt_b, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "INVENTORY_HASH_MISMATCH" in err

    # C. Missing public availability record
    reasons_missing_avail = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["acc-no-avail-entry"],
        "source_hashes": {"acc-no-avail-entry": "hash-no-avail"},
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash_c = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z", interval_end=watermark,
        certified_watermark=watermark, receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=reasons_missing_avail["accession_inventory"],
        source_hashes=reasons_missing_avail["source_hashes"], material_gaps=[], zero_activity_proof=None,
    )
    receipt_c = {
        "receipt_id": str(uuid.uuid4()), "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": watermark, "certified_watermark": watermark,
        "receipt_version": 1, "coverage_status": "COMPLETE",
        "failure_reasons": reasons_missing_avail, "reconciliation_hash": rec_hash_c,
    }
    ok, _, err = _validate_issuer_reconciliation_receipt(receipt_c, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "MISSING_PUBLIC_AVAILABILITY" in err

    # D. Availability after cutoff watermark
    reasons_late_avail = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["acc-avail-late"],
        "source_hashes": {"acc-avail-late": "hash-late"},
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash_d = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z", interval_end=watermark,
        certified_watermark=watermark, receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=reasons_late_avail["accession_inventory"],
        source_hashes=reasons_late_avail["source_hashes"], material_gaps=[], zero_activity_proof=None,
    )
    receipt_d = {
        "receipt_id": str(uuid.uuid4()), "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": watermark, "certified_watermark": watermark,
        "receipt_version": 1, "coverage_status": "COMPLETE",
        "failure_reasons": reasons_late_avail, "reconciliation_hash": rec_hash_d,
    }
    ok, _, err = _validate_issuer_reconciliation_receipt(receipt_d, "AAPL", "0000320193", cur=cur)
    assert ok is False
    assert "AVAILABILITY_AFTER_CUTOFF" in err

    conn.close()


def test_valid_receipt_contracts_and_partial_semantics(clean_postgres_db):
    """
    Test 13: Verify valid receipt acceptance across documented contracts and
    strict preservation of legitimate PARTIAL inventory semantics:
    - Current schema v1/v2 receipt accepted with verified inventory and availability.
    - Legacy receipt accepted under documented colon-delimited contract.
    - PARTIAL coverage receipt is preserved as PARTIAL (never silently converted to COMPLETE),
      even when uninventoried filings are present in the interval.
    """
    conn = psycopg2.connect(clean_postgres_db)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    import hashlib
    from analyst_dashboard.data.sec_edgar_fetcher import (
        _validate_issuer_reconciliation_receipt,
        SecEdgarFetcher,
    )
    from analyst_dashboard.smart_money.storage.postgres_repository import compute_canonical_receipt_hash, _to_iso

    watermark = "2026-11-19T23:30:21Z"

    # Insert two filings: one inventoried, one uninventoried
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, cik, ticker, company_name, filing_date, accepted_datetime, xml_content, content_hash, source_url)
        VALUES
            ('acc-valid-1', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-18', '2026-11-18T15:00:00Z', '<xml></xml>', 'h-v1', 'https://sec.gov/v1'),
            ('acc-uninv-2', '0000320193', 'AAPL', 'Apple Inc.', '2026-11-19', '2026-11-19T15:00:00Z', '<xml></xml>', 'h-u2', 'https://sec.gov/u2');
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source, source_reference)
        VALUES
            ('acc-valid-1', '2026-11-18T15:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/v1'),
            ('acc-uninv-2', '2026-11-19T15:05:00Z', 'SEC_EDGAR_API', 'https://sec.gov/u2');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (transaction_id, accession_number, transaction_date, transaction_code, shares, price_per_share, total_value_usd, reporting_owner_name, reporting_owner_role, ownership_type)
        VALUES
            ('tx-v1', 'acc-valid-1', '2026-11-17', 'S', 500, 350.0, 175000.0, 'Tim Cook', 'CEO', 'D'),
            ('tx-u2', 'acc-uninv-2', '2026-11-18', 'S', 300, 352.0, 105600.0, 'Tim Cook', 'CEO', 'D');
        """
    )
    conn.commit()

    # 1. Valid current schema v1 receipt (PARTIAL coverage with uninventoried filing in interval)
    reasons_partial = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["acc-valid-1"],
        "source_hashes": {"acc-valid-1": "h-v1"},
        "material_gaps": [],
        "schema_version": 1,
    }
    rec_hash_curr = compute_canonical_receipt_hash(
        schema_version=1, ticker="AAPL", issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z", interval_end=watermark,
        certified_watermark=watermark, receipt_version=1,
        supersedes_receipt_id=None, accession_inventory=reasons_partial["accession_inventory"],
        source_hashes=reasons_partial["source_hashes"], material_gaps=[], zero_activity_proof=None,
    )
    receipt_partial = {
        "receipt_id": str(uuid.uuid4()),
        "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": watermark,
        "certified_watermark": watermark,
        "receipt_version": 1,
        "coverage_status": "PARTIAL",
        "failure_reasons": reasons_partial,
        "reconciliation_hash": rec_hash_curr,
    }
    ok, valid_rc, err = _validate_issuer_reconciliation_receipt(receipt_partial, "AAPL", "0000320193", cur=cur)
    assert ok is True, f"Valid PARTIAL receipt should pass: {err}"
    assert valid_rc["coverage_status"] == "PARTIAL", "PARTIAL status must be preserved"
    assert valid_rc["certified_watermark"] == watermark

    # Insert partial receipt into DB to test end-to-end disclosure envelope
    cur.execute(
        """
        INSERT INTO sm_reconciliation_receipts (
            receipt_id, interval_start, interval_end, certified_watermark, receipt_version,
            expected_count, accounted_count, missing_count, material_gap_count, coverage_status,
            failure_reasons, reconciliation_hash, release_sha
        ) VALUES (
            %s, '2026-11-01T00:00:00Z', %s, %s, 1,
            1, 1, 0, 0, 'PARTIAL',
            %s, %s, '868d7afc'
        );
        """,
        (receipt_partial["receipt_id"], watermark, watermark, json.dumps(reasons_partial), rec_hash_curr),
    )
    conn.commit()

    fetcher = SecEdgarFetcher()
    from unittest.mock import patch

    with patch("analyst_dashboard.smart_money.storage.postgres_connection.get_connection", side_effect=lambda: psycopg2.connect(clean_postgres_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="postgres"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        env = fetcher.get_recent_filings_envelope("AAPL")
        assert env["status"] == "AVAILABLE_WITH_RESULTS"
        # Must preserve legitimate PARTIAL coverage semantics:
        assert env["coverage_status"] == "PARTIAL"
        assert env["certified_watermark"] == watermark

    # 2. Valid legacy receipt contract
    acc_inv_legacy = ["acc-valid-1"]
    s_hashes_legacy = {"acc-valid-1": "h-v1"}
    legacy_payload = (
        f"AAPL:0000320193:"
        f"{_to_iso('2026-11-01T00:00:00Z')}:{_to_iso(watermark)}:{_to_iso(watermark)}:"
        f"{json.dumps(sorted(list(set(acc_inv_legacy))))}:{json.dumps(s_hashes_legacy, sort_keys=True)}:0"
    )
    legacy_hash = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()
    receipt_legacy = {
        "receipt_id": str(uuid.uuid4()),
        "interval_start": "2026-11-01T00:00:00Z",
        "interval_end": watermark,
        "certified_watermark": watermark,
        "receipt_version": 1,
        "coverage_status": "PARTIAL",
        "failure_reasons": {
            "ticker": "AAPL",
            "issuer_cik": "0000320193",
            "accession_inventory": acc_inv_legacy,
            "source_hashes": s_hashes_legacy,
            "material_gaps": [],
        },
        "reconciliation_hash": legacy_hash,
    }
    ok_leg, valid_leg, err_leg = _validate_issuer_reconciliation_receipt(receipt_legacy, "AAPL", "0000320193", cur=cur)
    assert ok_leg is True, f"Valid legacy receipt should pass: {err_leg}"
    assert valid_leg["certified_watermark"] == watermark

    conn.close()

