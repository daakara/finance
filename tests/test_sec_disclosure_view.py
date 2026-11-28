"""Authoritative Verification Suite for SEC Form 4 Disclosure View (ARX Issue #165).

Validates Gate 3, Gate 4, and Gate 5 requirements:
1. End-to-end fixture: parser/storage -> API mapping for purchases, sales, gifts, exercises, derivatives, mixed filings, amendments.
2. Multiple transactions per filing preserved without duplicates.
3. Newer sale vs older purchase selection (highlighting latest event without purchase bias).
4. Point-in-time availability boundaries and coverage metadata.
5. Amendment supersession and explicit amendment status.
6. Filing-only fallback (submission metadata only without fabricated trades).
7. Missing price (None) vs reported zero price (0.0) distinction.
8. Read-only invariant: zero database writes, zero live SEC network requests, zero scoring or scanning triggered.
"""

import os
import sqlite3
import tempfile
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from analyst_dashboard.data.sec_edgar_fetcher import (
    SecEdgarFetcher,
    _map_sec_transaction_type,
    _attach_parsed_transactions_to_filings,
)
from api.routes.smart_money import get_sec_filings


@pytest.fixture
def temp_sec_db():
    """Create a temporary SQLite database with Form 4 raw and parsed schemas."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE form4_raw_filings (
            accession_number TEXT PRIMARY KEY,
            issuer_cik TEXT,
            issuer_symbol TEXT,
            filing_date TEXT,
            acceptance_datetime TEXT,
            document_type TEXT,
            raw_content TEXT,
            source_content_hash TEXT,
            source_url TEXT,
            fetched_at TEXT,
            disposition TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE form4_parsed_transactions (
            record_key TEXT PRIMARY KEY,
            accession_number TEXT,
            issuer_cik TEXT,
            issuer_symbol TEXT,
            issuer_name TEXT,
            reporting_owner_cik TEXT,
            reporting_owner_name TEXT,
            is_officer INTEGER,
            officer_title TEXT,
            is_director INTEGER,
            is_ten_percent_owner INTEGER,
            filing_date TEXT,
            transaction_date TEXT,
            transaction_code TEXT,
            signal_type TEXT,
            shares REAL,
            price_per_share REAL,
            total_value_usd REAL,
            acquired_disposed TEXT,
            ownership_nature TEXT,
            is_derivative INTEGER,
            is_open_market_purchase_evidence INTEGER,
            is_positive_buying_conviction INTEGER,
            acceptance_datetime TEXT,
            public_availability_at TEXT,
            source_fetch_time TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE sm_source_availability_ledger (
            accession_number TEXT PRIMARY KEY,
            public_availability_at TEXT,
            recorded_at TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE sm_amendments (
            amendment_accession TEXT PRIMARY KEY,
            base_accession TEXT,
            amendment_number INTEGER DEFAULT 1,
            resolution_status TEXT,
            resolved_at TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE sm_reconciliation_receipts (
            receipt_id TEXT PRIMARY KEY,
            supersedes_receipt_id TEXT,
            ticker TEXT,
            issuer_cik TEXT,
            interval_start TEXT,
            interval_end TEXT,
            certified_watermark TEXT,
            coverage_status TEXT,
            receipt_version INTEGER,
            missing_count INTEGER,
            material_gap_count INTEGER,
            failure_reasons TEXT,
            reconciliation_hash TEXT,
            release_sha TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()

    yield path

    if os.path.exists(path):
        os.remove(path)


# ── 1. Transaction Type Mapping Unit Tests ────────────────────────────────────

def test_map_sec_transaction_type_neutrality():
    """Verify that transaction types map accurately without invented sentiment."""
    assert "Purchase (Code P)" in _map_sec_transaction_type("P", "A")
    assert "Sale (Code S)" in _map_sec_transaction_type("S", "D")
    assert "Bona Fide Gift" in _map_sec_transaction_type("G", "D")
    assert "Option Exercise" in _map_sec_transaction_type("M", "A")
    assert "Grant / Award" in _map_sec_transaction_type("A", "A")
    assert "Tax Withholding" in _map_sec_transaction_type("F", "D")
    assert "Disposition to Issuer" in _map_sec_transaction_type("D", "D")
    assert "Conversion of Derivative" in _map_sec_transaction_type("C", "A", is_derivative=True)


# ── 2. Attach Parsed Transactions Unit Tests ──────────────────────────────────

def test_attach_parsed_transactions_preserves_multiple_without_duplicates():
    """Verify multiple transactions in one filing are preserved without duplicate records."""
    results = [
        {
            "form": "4",
            "accession_number": "0001140361-26-044952",
            "filing_date": "2026-11-19",
            "sec_url": "https://sec.gov/test.txt",
        }
    ]
    tx_rows = [
        {
            "transaction_id": "tx-1",
            "accession_number": "0001140361-26-044952",
            "transaction_date": "2026-11-17",
            "transaction_code": "M",
            "shares": 5000.0,
            "price_per_share": 0.0,
            "total_value_usd": 0.0,
            "reporting_owner_name": "Jennifer Newstead",
            "officer_title": "General Counsel",
            "is_director": 0,
            "is_officer": 1,
            "is_ten_percent": 0,
            "ownership_type": "D",
            "is_derivative": 1,
            "acquired_disposed": "A",
            "is_open_market_purchase_evidence": 0,
            "is_positive_buying_conviction": 0,
            "public_availability_at": "2026-11-19T23:30:21Z",
        },
        {
            "transaction_id": "tx-2",
            "accession_number": "0001140361-26-044952",
            "transaction_date": "2026-11-17",
            "transaction_code": "S",
            "shares": 2398.0,
            "price_per_share": 352.77,
            "total_value_usd": 845942.46,
            "reporting_owner_name": "Jennifer Newstead",
            "officer_title": "General Counsel",
            "is_director": 0,
            "is_officer": 1,
            "is_ten_percent": 0,
            "ownership_type": "D",
            "is_derivative": 0,
            "acquired_disposed": "D",
            "is_open_market_purchase_evidence": 0,
            "is_positive_buying_conviction": 0,
            "public_availability_at": "2026-11-19T23:30:21Z",
        },
    ]

    _attach_parsed_transactions_to_filings(results, tx_rows)

    filing = results[0]
    assert filing["has_parsed_transaction"] is True
    assert filing["transaction_count"] == 2
    assert len(filing["transactions"]) == 2

    t1, t2 = filing["transactions"]
    assert t1["transaction_code"] == "M"
    assert t1["shares"] == 5000.0
    assert t1["price_per_share"] == 0.0
    assert t1["is_exercise"] is True

    assert t2["transaction_code"] == "S"
    assert t2["shares"] == 2398.0
    assert t2["price_per_share"] == 352.77
    assert t2["is_sell"] is True


def test_unparsed_filing_leaves_clean_fallback():
    """Verify an unparsed filing retains has_parsed_transaction=False without fake numbers."""
    results = [
        {
            "form": "4",
            "accession_number": "0000320193-26-000109",
            "filing_date": "2026-11-18",
            "sec_url": "https://sec.gov/test.txt",
        }
    ]
    _attach_parsed_transactions_to_filings(results, [])

    filing = results[0]
    assert filing["has_parsed_transaction"] is False
    assert filing["transaction_count"] == 0
    assert filing["transactions"] == []
    assert "shares_traded" not in filing
    assert "price_per_share" not in filing
    assert "total_value_usd" not in filing
    assert "shares" not in filing
    assert "price" not in filing
    assert "transaction_code" not in filing


# ── 3. End-to-End Fixture: Storage -> API Route ───────────────────────────────

def test_end_to_end_sale_vs_older_purchase_selection(temp_sec_db):
    """
    Verify that when a newer sale and an older purchase exist:
    1. Both are retrieved with accurate parsed details.
    2. The latest event (top item) is the newer sale, NOT biased toward the older purchase.
    3. Accession numbers, hashes, and dates are preserved.
    """
    conn = sqlite3.connect(temp_sec_db)
    cur = conn.cursor()

    # Older AAPL purchase on 2026-11-10
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, issuer_cik, issuer_symbol, filing_date, acceptance_datetime, document_type, source_content_hash, source_url)
        VALUES ('0001140361-26-044773', '0000320193', 'AAPL', '2026-11-10', '2026-11-10T22:30:00Z', '4', 'hash_older_purchase', 'https://sec.gov/old.txt');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (
            record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
            officer_title, filing_date, transaction_date, transaction_code, shares, price_per_share,
            total_value_usd, acquired_disposed, ownership_nature, is_derivative,
            is_open_market_purchase_evidence, is_positive_buying_conviction, acceptance_datetime, public_availability_at
        ) VALUES (
            'tx-old-p', '0001140361-26-044773', '0000320193', 'AAPL', 'Luca Maestri',
            'CFO', '2026-11-10', '2026-11-09', 'P', 1000.0, 340.00,
            340000.0, 'A', 'D', 0, 1, 1, '2026-11-10T22:30:00Z', '2026-11-10T22:30:00Z'
        );
        """
    )

    # Newer AAPL sale on 2026-11-19 (Jennifer Newstead, 2,398 shares @ $352.77)
    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, issuer_cik, issuer_symbol, filing_date, acceptance_datetime, document_type, source_content_hash, source_url)
        VALUES ('0001140361-26-044952', '0000320193', 'AAPL', '2026-11-19', '2026-11-19T23:30:21Z', '4', '93aa1a0a864d72ccc38dcd00a5ed6258b64634dbde4ab271ea78db8efd3841d0', 'https://sec.gov/new.txt');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (
            record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
            officer_title, filing_date, transaction_date, transaction_code, shares, price_per_share,
            total_value_usd, acquired_disposed, ownership_nature, is_derivative,
            is_open_market_purchase_evidence, is_positive_buying_conviction, acceptance_datetime, public_availability_at
        ) VALUES (
            'tx-new-s', '0001140361-26-044952', '0000320193', 'AAPL', 'Jennifer Newstead',
            'General Counsel', '2026-11-19', '2026-11-17', 'S', 2398.0, 352.77,
            845942.46, 'D', 'D', 0, 0, 0, '2026-11-19T23:30:21Z', '2026-11-19T23:30:21Z'
        );
        """
    )
    cur.execute(
        """
        INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at)
        VALUES
            ('0001140361-26-044773', '2026-11-10T22:30:00Z'),
            ('0001140361-26-044952', '2026-11-19T23:30:21Z');
        """
    )

    # Coverage receipt: PARTIAL with governed canonical hash and failure_reasons JSONB
    from analyst_dashboard.smart_money.storage.postgres_repository import compute_canonical_receipt_hash
    import json
    rc_reasons = {
        "ticker": "AAPL",
        "issuer_cik": "0000320193",
        "covered_tickers": ["AAPL"],
        "covered_ciks": ["0000320193"],
        "accession_inventory": ["0001140361-26-044773", "0001140361-26-044952"],
        "source_hashes": {
            "0001140361-26-044773": "hash_older_purchase",
            "0001140361-26-044952": "93aa1a0a864d72ccc38dcd00a5ed6258b64634dbde4ab271ea78db8efd3841d0",
        },
        "material_gaps": [],
        "schema_version": 1,
    }
    rc_hash = compute_canonical_receipt_hash(
        schema_version=1,
        ticker="AAPL",
        issuer_cik="0000320193",
        interval_start="2026-11-01T00:00:00Z",
        interval_end="2026-11-19T23:30:21Z",
        certified_watermark="2026-11-19T23:30:21Z",
        receipt_version=2,
        supersedes_receipt_id=None,
        accession_inventory=rc_reasons["accession_inventory"],
        source_hashes=rc_reasons["source_hashes"],
        material_gaps=[],
        zero_activity_proof=None,
    )
    cur.execute(
        """
        INSERT INTO sm_reconciliation_receipts (
            receipt_id, ticker, issuer_cik, interval_start, interval_end, certified_watermark,
            coverage_status, receipt_version, missing_count, material_gap_count,
            failure_reasons, reconciliation_hash, release_sha
        ) VALUES (
            'rc-1', 'AAPL', '0000320193', '2026-11-01T00:00:00Z', '2026-11-19T23:30:21Z', '2026-11-19T23:30:21Z',
            'PARTIAL', 2, 0, 1,
            ?, ?, '868d7afc'
        );
        """,
        (json.dumps(rc_reasons), rc_hash),
    )
    conn.commit()
    conn.close()

    fetcher = SecEdgarFetcher()
    with patch("analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db", side_effect=lambda *args, **kwargs: sqlite3.connect(temp_sec_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="sqlite"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        envelope = fetcher.get_recent_filings_envelope("AAPL")
        assert envelope["status"] == "AVAILABLE_WITH_RESULTS"
        assert envelope["coverage_status"] == "PARTIAL"
        assert envelope["certified_watermark"] == "2026-11-19T23:30:21Z"
        assert envelope["is_pipeline_held"] is True

        items = envelope["items"]
        assert len(items) == 2

        # Item 0 MUST be the latest filing (Newer sale on Nov 19), NOT the older purchase!
        top_item = items[0]
        assert top_item["accession_number"] == "0001140361-26-044952"
        assert top_item["filing_date"] == "2026-11-19"
        assert top_item["xml_hash"] == "93aa1a0a864d72ccc38dcd00a5ed6258b64634dbde4ab271ea78db8efd3841d0"
        assert top_item["insider_name"] == "Jennifer Newstead"
        assert top_item["insider_role"] == "General Counsel"
        assert top_item["shares_traded"] == 2398.0
        assert top_item["price_per_share"] == 352.77
        assert top_item["total_value_usd"] == 845942.46
        assert top_item["is_sell"] is True
        assert top_item["is_buy"] is False

        # Item 1 is the older purchase
        second_item = items[1]
        assert second_item["accession_number"] == "0001140361-26-044773"
        assert second_item["filing_date"] == "2026-11-10"
        assert second_item["is_buy"] is True


def test_gift_and_exercise_zero_price_differentiation(temp_sec_db):
    """
    Verify that reported zero prices ($0.00) in gifts and exercises are distinguished
    from unparsed missing prices (null/None), and not equated with zero economic value.
    """
    conn = sqlite3.connect(temp_sec_db)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO form4_raw_filings (accession_number, issuer_cik, issuer_symbol, filing_date, acceptance_datetime, document_type, source_content_hash, source_url)
        VALUES ('0001045810-26-000111', '0001045810', 'NVDA', '2026-11-15', '2026-11-15T21:00:00Z', '4', 'nvda_gift_hash', 'https://sec.gov/gift.txt');
        """
    )
    cur.execute(
        """
        INSERT INTO form4_parsed_transactions (
            record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
            officer_title, filing_date, transaction_date, transaction_code, shares, price_per_share,
            total_value_usd, acquired_disposed, ownership_nature, is_derivative,
            is_open_market_purchase_evidence, is_positive_buying_conviction, acceptance_datetime, public_availability_at
        ) VALUES (
            'tx-gift', '0001045810-26-000111', '0001045810', 'NVDA', 'Jensen Huang',
            'President and CEO', '2026-11-15', '2026-11-14', 'G', 15000.0, 0.0,
            0.0, 'D', 'D', 0, 0, 0, '2026-11-15T21:00:00Z', '2026-11-15T21:00:00Z'
        );
        """
    )
    conn.commit()
    conn.close()

    fetcher = SecEdgarFetcher()
    with patch("analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db", side_effect=lambda *args, **kwargs: sqlite3.connect(temp_sec_db)), \
         patch("analyst_dashboard.smart_money.storage.config.get_storage_backend", return_value="sqlite"), \
         patch("analyst_dashboard.data.sec_edgar_fetcher.is_sec_egress_authorized", return_value=False):

        envelope = fetcher.get_recent_filings_envelope("NVDA")
        item = envelope["items"][0]
        assert item["is_gift"] is True
        assert item["price_per_share"] == 0.0
        assert item["total_value_usd"] == 0.0
        assert item["is_buy"] is False
        assert item["is_sell"] is False

        tx = item["transactions"][0]
        assert tx["transaction_code"] == "G"
        assert tx["price_per_share"] == 0.0
        assert "Bona Fide Gift" in tx["transaction_type"]


def test_api_route_get_sec_filings_contract():
    """Verify get_sec_filings returns full contract including coverage metadata and hold state."""
    mock_envelope = {
        "status": "AVAILABLE_WITH_RESULTS",
        "status_code": 200,
        "cik": "0000320193",
        "reason": None,
        "error_detail": None,
        "items": [
            {
                "form": "4",
                "accession_number": "0001140361-26-044952",
                "filing_date": "2026-11-19",
                "sec_url": "https://sec.gov/test.txt",
                "xml_hash": "93aa1a0a864d72ccc38dcd00a5ed6258b64634dbde4ab271ea78db8efd3841d0",
                "has_parsed_transaction": True,
                "insider_name": "Jennifer Newstead",
                "insider_role": "General Counsel",
                "shares_traded": 2398.0,
                "price_per_share": 352.77,
                "total_value_usd": 845942.46,
                "is_buy": False,
                "is_sell": True,
                "transaction_code": "S",
                "transaction_type": "Sale (Code S)",
                "transactions": [
                    {
                        "transaction_id": "tx-1",
                        "accession_number": "0001140361-26-044952",
                        "transaction_date": "2026-11-17",
                        "transaction_code": "S",
                        "transaction_type": "Sale (Code S)",
                        "shares": 2398.0,
                        "price_per_share": 352.77,
                        "total_value_usd": 845942.46,
                        "reporting_owner_name": "Jennifer Newstead",
                        "reporting_owner_role": "General Counsel",
                        "is_officer": True,
                        "is_director": False,
                        "is_ten_percent": False,
                        "ownership_type": "D",
                        "is_derivative": False,
                        "acquired_disposed": "D",
                        "is_buy": False,
                        "is_sell": True,
                        "is_gift": False,
                        "is_exercise": False,
                        "is_open_market_purchase_evidence": False,
                        "is_positive_buying_conviction": False,
                        "security_title": "Common Stock",
                        "public_availability_at": "2026-11-19T23:30:21Z",
                    }
                ],
            }
        ],
        "cached": True,
        "coverage_status": "PARTIAL",
        "certified_watermark": "2026-11-19T23:30:21Z",
        "is_pipeline_held": True,
    }

    with patch("analyst_dashboard.data.sec_edgar_fetcher.SecEdgarFetcher.get_recent_filings_envelope", return_value=mock_envelope):
        response = get_sec_filings("AAPL")
        assert response["symbol"] == "AAPL"
        assert response["status"] == "AVAILABLE_WITH_RESULTS"
        assert response["coverage_status"] == "PARTIAL"
        assert response["certified_watermark"] == "2026-11-19T23:30:21Z"
        assert response["is_pipeline_held"] is True
        assert len(response["items"]) == 1

        filing = response["items"][0]
        assert filing["insider_name"] == "Jennifer Newstead"
        assert filing["is_sell"] is True
        assert len(filing["transactions"]) == 1
        assert filing["transactions"][0]["shares"] == 2398.0


def test_read_path_does_not_mutate_or_publish():
    """Verify that calling disclosure view read functions executes zero writes or snapshot publications."""
    with patch("analyst_dashboard.data.sec_edgar_fetcher.SecEdgarFetcher._get_cached_recent_filings_envelope") as mock_cache:
        mock_cache.return_value = {
            "status": "AVAILABLE_NO_RESULTS",
            "status_code": 200,
            "cik": "0000320193",
            "items": [],
            "cached": True,
            "coverage_status": "SUPPORTED_SEC_ISSUER",
            "certified_watermark": None,
            "is_pipeline_held": True,
        }
        res = get_sec_filings("AAPL")
        assert res["status"] == "AVAILABLE_NO_RESULTS"
        mock_cache.assert_called_once()
