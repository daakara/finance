"""
Deterministic Point-in-Time Correctness Tests for Smart Money Form 4 Data.

Verifies:
- Case A: Filing not yet public (transaction_date before as_of, but acceptance after as_of -> EXCLUDED)
- Case B: Filing public (acceptance before as_of -> ELIGIBLE)
- Case C: Late filer (transaction long before as_of, but filing public after as_of -> NO LEAKAGE)
- Case D: Amendment (4/A published after original as_of -> not visible prior to amendment availability)
"""

import sqlite3
import pytest
from analyst_dashboard.smart_money.form4_ingestion import Form4IngestionManager
from analyst_dashboard.smart_money.form4_models import SmartMoneyTransaction


@pytest.fixture
def pit_ingestion_manager(tmp_path):
    """Creates a temporary isolated SQLite database with Form 4 schema."""
    db_file = str(tmp_path / "pit_test.db")
    mgr = Form4IngestionManager(db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Case A & B: Filing with transaction_date = 2026-09-01, public_availability_at = 2026-09-20T14:30:00Z
    cursor.execute("""
        INSERT INTO form4_parsed_transactions (
            record_key, accession_number, issuer_cik, issuer_symbol, issuer_name,
            reporting_owner_cik, reporting_owner_name, is_officer, officer_title,
            is_director, is_ten_percent_owner, filing_date, transaction_date,
            transaction_code, signal_type, shares, price_per_share, total_value_usd,
            acquired_disposed, ownership_nature, is_derivative,
            is_open_market_purchase_evidence, is_positive_buying_conviction,
            acceptance_datetime, public_availability_at, source_fetch_time, content_hash, parsed_at
        ) VALUES (
            'ACC-001_1', '0000000000-26-000001', '0001000001', 'TESTA', 'Test Issuer A',
            '0002000001', 'Alice Insider', 1, 'CEO',
            0, 0, '2026-09-20', '2026-09-01',
            'P', 'OPEN_MARKET_BUY', 10000.0, 50.0, 500000.0,
            'A', 'D', 0,
            1, 1,
            '2026-09-20T14:30:00Z', '2026-09-20T14:30:00Z', '2026-09-20T14:31:00Z', 'h1', '2026-09-20T14:31:00Z'
        )
    """)
    cursor.execute("""
        INSERT INTO sm_source_availability_ledger (
            accession_number, issuer_cik, public_availability_at, first_fetched_at,
            source_hash, form_type, normalization_version
        ) VALUES (
            '0000000000-26-000001', '0001000001', '2026-09-20T14:30:00Z',
            '2026-09-20T14:31:00Z', 'hash_case_a_b', '4', '2.2.0'
        )
    """)

    # Case C: Late filer - transaction on 2026-06-01, but filed 2026-09-25T10:00:00Z
    cursor.execute("""
        INSERT INTO form4_parsed_transactions (
            record_key, accession_number, issuer_cik, issuer_symbol, issuer_name,
            reporting_owner_cik, reporting_owner_name, is_officer, officer_title,
            is_director, is_ten_percent_owner, filing_date, transaction_date,
            transaction_code, signal_type, shares, price_per_share, total_value_usd,
            acquired_disposed, ownership_nature, is_derivative,
            is_open_market_purchase_evidence, is_positive_buying_conviction,
            acceptance_datetime, public_availability_at, source_fetch_time, content_hash, parsed_at
        ) VALUES (
            'ACC-002_1', '0000000000-26-000002', '0001000002', 'TESTC', 'Test Issuer C Late',
            '0002000002', 'Bob LateFiler', 1, 'CFO',
            0, 0, '2026-09-25', '2026-06-01',
            'P', 'OPEN_MARKET_BUY', 20000.0, 10.0, 200000.0,
            'A', 'D', 0,
            1, 1,
            '2026-09-25T10:00:00Z', '2026-09-25T10:00:00Z', '2026-09-25T10:01:00Z', 'h2', '2026-09-25T10:01:00Z'
        )
    """)
    cursor.execute("""
        INSERT INTO sm_source_availability_ledger (
            accession_number, issuer_cik, public_availability_at, first_fetched_at,
            source_hash, form_type, normalization_version
        ) VALUES (
            '0000000000-26-000002', '0001000002', '2026-09-25T10:00:00Z',
            '2026-09-25T10:01:00Z', 'hash_case_c', '4', '2.2.0'
        )
    """)

    # Case D: Original filed 2026-09-15T09:00:00Z, amendment (4/A) filed 2026-09-28T16:00:00Z
    cursor.execute("""
        INSERT INTO form4_parsed_transactions (
            record_key, accession_number, issuer_cik, issuer_symbol, issuer_name,
            reporting_owner_cik, reporting_owner_name, is_officer, officer_title,
            is_director, is_ten_percent_owner, filing_date, transaction_date,
            transaction_code, signal_type, shares, price_per_share, total_value_usd,
            acquired_disposed, ownership_nature, is_derivative,
            is_open_market_purchase_evidence, is_positive_buying_conviction,
            acceptance_datetime, public_availability_at, source_fetch_time, content_hash, parsed_at
        ) VALUES (
            'ACC-003_1', '0000000000-26-000003', '0001000003', 'TESTD', 'Test Issuer D',
            '0002000003', 'Charlie Director', 0, 'Director',
            1, 0, '2026-09-15', '2026-09-14',
            'P', 'OPEN_MARKET_BUY', 5000.0, 100.0, 500000.0,
            'A', 'D', 0,
            1, 1,
            '2026-09-15T09:00:00Z', '2026-09-15T09:00:00Z', '2026-09-15T09:01:00Z', 'h3', '2026-09-15T09:01:00Z'
        )
    """)
    cursor.execute("""
        INSERT INTO sm_source_availability_ledger (
            accession_number, issuer_cik, public_availability_at, first_fetched_at,
            source_hash, form_type, normalization_version
        ) VALUES (
            '0000000000-26-000003', '0001000003', '2026-09-15T09:00:00Z',
            '2026-09-15T09:01:00Z', 'hash_case_d_orig', '4', '2.2.0'
        )
    """)

    # Amendment accession
    cursor.execute("""
        INSERT INTO form4_parsed_transactions (
            record_key, accession_number, issuer_cik, issuer_symbol, issuer_name,
            reporting_owner_cik, reporting_owner_name, is_officer, officer_title,
            is_director, is_ten_percent_owner, filing_date, transaction_date,
            transaction_code, signal_type, shares, price_per_share, total_value_usd,
            acquired_disposed, ownership_nature, is_derivative,
            is_open_market_purchase_evidence, is_positive_buying_conviction,
            acceptance_datetime, public_availability_at, source_fetch_time, content_hash, parsed_at
        ) VALUES (
            'ACC-004_1', '0000000000-26-000004', '0001000003', 'TESTD', 'Test Issuer D',
            '0002000003', 'Charlie Director', 0, 'Director',
            1, 0, '2026-09-28', '2026-09-14',
            'P', 'OPEN_MARKET_BUY', 7500.0, 100.0, 750000.0,
            'A', 'D', 0,
            1, 1,
            '2026-09-28T16:00:00Z', '2026-09-28T16:00:00Z', '2026-09-28T16:01:00Z', 'h4', '2026-09-28T16:01:00Z'
        )
    """)
    cursor.execute("""
        INSERT INTO sm_source_availability_ledger (
            accession_number, issuer_cik, public_availability_at, first_fetched_at,
            source_hash, form_type, normalization_version
        ) VALUES (
            '0000000000-26-000004', '0001000003', '2026-09-28T16:00:00Z',
            '2026-09-28T16:01:00Z', 'hash_case_d_amend', '4/A', '2.2.0'
        )
    """)

    conn.commit()
    conn.close()
    return mgr


def test_case_a_filing_not_yet_public(pit_ingestion_manager):
    """Case A: transaction_date is earlier, but acceptanceDateTime is after historical_as_of -> EXCLUDED."""
    as_of = "2026-09-10T23:59:59Z"
    results = pit_ingestion_manager.get_transactions_as_of(as_of, symbol="TESTA")
    assert len(results) == 0, "Filing accepted on 2026-09-20 must be EXCLUDED as of 2026-09-10"


def test_case_b_filing_public(pit_ingestion_manager):
    """Case B: acceptanceDateTime is at or before historical_as_of -> ELIGIBLE."""
    as_of = "2026-09-21T00:00:00Z"
    results = pit_ingestion_manager.get_transactions_as_of(as_of, symbol="TESTA")
    assert len(results) == 1, "Filing accepted on 2026-09-20 must be ELIGIBLE as of 2026-09-21"
    assert results[0].issuer_symbol == "TESTA"
    assert results[0].shares == 10000.0


def test_case_c_late_filer_no_backward_leakage(pit_ingestion_manager):
    """Case C: Late filer with transaction executed 2026-06-01 must not leak into 2026-09-01 as-of."""
    as_of_prior = "2026-09-01T00:00:00Z"
    results_prior = pit_ingestion_manager.get_transactions_as_of(as_of_prior, symbol="TESTC")
    assert len(results_prior) == 0, "Late filing from 2026-09-25 must not leak into 2026-09-01 historical view"

    as_of_post = "2026-09-26T00:00:00Z"
    results_post = pit_ingestion_manager.get_transactions_as_of(as_of_post, symbol="TESTC")
    assert len(results_post) == 1, "Late filing is only available after acceptance"


def test_case_d_amendment_temporal_isolation(pit_ingestion_manager):
    """Case D: Historical query before 4/A amendment acceptance sees only original filing."""
    as_of_before_amendment = "2026-09-20T00:00:00Z"
    txs_before = pit_ingestion_manager.get_transactions_as_of(as_of_before_amendment, symbol="TESTD")
    assert len(txs_before) == 1
    assert txs_before[0].accession_number == "0000000000-26-000003"
    assert txs_before[0].shares == 5000.0

    as_of_after_amendment = "2026-09-30T00:00:00Z"
    txs_after = pit_ingestion_manager.get_transactions_as_of(as_of_after_amendment, symbol="TESTD")
    assert len(txs_after) == 2
    accessions = {t.accession_number for t in txs_after}
    assert "0000000000-26-000003" in accessions
    assert "0000000000-26-000004" in accessions


def test_naive_datetime_rejected(pit_ingestion_manager):
    """Naive datetimes lacking timezone information must be rejected."""
    with pytest.raises(ValueError, match="Naive datetime rejected"):
        pit_ingestion_manager.get_transactions_as_of("2026-09-10 12:00:00", symbol="TESTA")
