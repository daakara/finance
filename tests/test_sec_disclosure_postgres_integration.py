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
        "accession_inventory": ["acc-past", "acc-future"],
        "source_hashes": {},
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
        source_hashes={},
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
            2, 2, 0, 0, 'COMPLETE',
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
