"""
ARX Terminal — Directive SM-DATA-GOV-001 Deterministic Integrity & Evidence Lifecycle Suite

Validates the 10 Gate C governance invariants:
1. test_no_unauthorized_canonical_evidence_deletion
2. test_idempotent_filing_reprocessing
3. test_original_and_amended_filing_coexistence
4. test_transaction_level_supersession
5. test_public_information_historical_cutoff
6. test_arx_observed_historical_cutoff
7. test_parser_version_reproducibility
8. test_missing_or_ambiguous_timestamp_handling
9. test_source_coverage_versus_neutral_conviction
10. test_backup_and_recovery_evidence
"""

import os
import json
import tempfile
import sqlite3
from datetime import datetime, timezone
import pytest

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    import pgserver
except ImportError:
    pgserver = None

from fastapi import HTTPException
from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_models import SmartMoneySignalType
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine, SmartMoneyDomainStatus, SmartMoneyDirection
from api.routes.smart_money import get_smart_money_evidence


@pytest.fixture(scope="module")
def pg_db():
    """Module-scoped isolated Postgres server using pgserver."""
    if psycopg2 is None:
        pytest.skip("psycopg2 is not installed")
    if pgserver is None:
        pytest.skip("pgserver is not installed")

    pgdata = tempfile.mkdtemp(prefix="arx_sm_data_gov_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)
    conn.close()

    yield {"uri": uri, "server": server}
    try:
        server.cleanup()
    except Exception:
        pass


class TestSmDataGov001EvidenceLifecycle:
    """Deterministic validation of SM-DATA-GOV-001 requirements."""

    # 1. Destruction Protection & Authorization Gate
    def test_no_unauthorized_canonical_evidence_deletion(self, pg_db):
        uri = pg_db["uri"]
        repo = PostgresSmartMoneyRepository(dsn=uri)
        acc = "0001045810-26-900001"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Seed record
        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-01", now_iso, "<xml/>", "hash_del_1", "https://sec.gov"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (
                    accession_number, public_availability_at, source, source_reference
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc, now_iso, "SEC_INDEX", "ref_del_1"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING;
            """, (
                f"{acc}:0001045810:NON_DERIVATIVE:1", acc, "2026-10-01", "P",
                100.0, 100.0, 10000.0, "Test Executive",
                "VP", False, True, False,
                "VP", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))
        conn.commit()
        conn.close()

        # Prohibit deletion without authorization token - fails closed unconditionally
        with pytest.raises(PermissionError, match="CANONICAL_DELETION_PROHIBITED"):
            repo.delete_canonical_evidence(accession_number=acc, authorization_token=None)

        # Prohibit deletion with arbitrary token
        with pytest.raises(PermissionError, match="CANONICAL_DELETION_PROHIBITED"):
            repo.delete_canonical_evidence(accession_number=acc, authorization_token="INVALID_TOKEN_123")

        # Prohibit deletion even with forged retention prefix - fails closed unconditionally
        with pytest.raises(PermissionError, match="CANONICAL_DELETION_PROHIBITED"):
            repo.delete_canonical_evidence(
                accession_number=acc,
                authorization_token="AUTH_LEGAL_APPROVED_RETENTION_DELETION:2026-11-AUDIT-SCHEDULE",
            )

        # Verify parsed transactions, availability ledger, and raw filings remain intact
        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM form4_parsed_transactions WHERE accession_number = %s;", (acc,))
            assert cur.fetchone()[0] == 1

            cur.execute("SELECT COUNT(*) FROM sm_source_availability_ledger WHERE accession_number = %s;", (acc,))
            assert cur.fetchone()[0] == 1

            # Raw filing remains permanently immutable under abort_immutable_record trigger
            cur.execute("SELECT COUNT(*) FROM form4_raw_filings WHERE accession_number = %s;", (acc,))
            assert cur.fetchone()[0] == 1

            # Direct SQL deletion on raw filing is blocked by trigger
            with pytest.raises(psycopg2.InternalError, match="abort_immutable_record"):
                cur.execute("DELETE FROM form4_raw_filings WHERE accession_number = %s;", (acc,))
        conn.rollback()
        conn.close()

    # 2. Idempotent Reprocessing
    def test_idempotent_filing_reprocessing(self, pg_db):
        uri = pg_db["uri"]
        repo = PostgresSmartMoneyRepository(dsn=uri)
        acc = "0001045810-26-900002"
        now_iso = datetime.now(timezone.utc).isoformat()

        filing_data = {
            "accession_number": acc,
            "cik": "0001045810",
            "ticker": "NVDA",
            "company_name": "NVIDIA CORP",
            "filing_date": "2026-10-02",
            "accepted_datetime": now_iso,
            "xml_content": "<ownershipDocument><test/></ownershipDocument>",
            "content_hash": "hash_idem_002",
            "source_url": "https://sec.gov/idem2",
        }

        # First save: inserts record
        res1 = repo.save_raw_filing(filing_data)
        assert res1 is True

        # Second save with identical content: idempotent no-op returns False
        res2 = repo.save_raw_filing(filing_data)
        assert res2 is False

        # Attempted save with conflicting content hash: fails closed
        tampered_filing = dict(filing_data, content_hash="hash_tampered_999")
        with pytest.raises(RuntimeError, match="SOURCE_INTEGRITY_CONFLICT"):
            repo.save_raw_filing(tampered_filing)

    # 3. Original and Amended Filing Coexistence
    def test_original_and_amended_filing_coexistence(self, pg_db):
        uri = pg_db["uri"]
        acc_orig = "0001045810-26-900003"
        acc_amend = "0001045810-26-900003A"
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            # 1. Original Form 4
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_orig, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-03", now_iso, "<orig_xml/>", "hash_orig_3", "https://sec.gov/3"))

            # 2. Amended Form 4/A
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_amend, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-04", now_iso, "<amend_xml/>", "hash_amend_3", "https://sec.gov/3A"))

            # 3. sm_amendments linkage: amendment_accession = acc_amend, base_accession = acc_orig
            cur.execute("""
                INSERT INTO sm_amendments (
                    amendment_accession, base_accession, amendment_number, resolution_status, resolved_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (amendment_accession) DO NOTHING;
            """, (acc_amend, acc_orig, 1, "SUPERSEDED_OR_CORRECTED", now_iso))

            # Verify both coexist in form4_raw_filings
            cur.execute("SELECT COUNT(*) FROM form4_raw_filings WHERE accession_number IN (%s, %s);", (acc_orig, acc_amend))
            coexist_count = cur.fetchone()[0]
            assert coexist_count == 2
        conn.commit()
        conn.close()

    # 4. Transaction-Level Supersession (No Double Counting)
    def test_transaction_level_supersession(self, pg_db, monkeypatch):
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        monkeypatch.setenv("ARX_STORAGE_BACKEND", "postgres")

        acc_orig = "0001045810-26-900004"
        acc_amend = "0001045810-26-900004A"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            for acc, fdate in [(acc_orig, today), (acc_amend, today)]:
                cur.execute("""
                    INSERT INTO form4_raw_filings (
                        accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                        xml_content, content_hash, source_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (accession_number) DO NOTHING;
                """, (acc, "0001045810", "NVDA", "NVIDIA CORP", fdate, now_iso, "<xml/>", f"hash_{acc}", f"https://sec.gov/{acc}"))
                cur.execute("""
                    INSERT INTO sm_source_availability_ledger (
                        accession_number, public_availability_at, source, source_reference
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (accession_number) DO NOTHING;
                """, (acc, now_iso, "SEC_INDEX", f"ref_{acc}"))

            # Original transaction: 10,000 shares @ $100 ($1,000,000)
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING;
            """, (
                f"{acc_orig}:0001045810:NON_DERIVATIVE:1", acc_orig, today, "P",
                10000.0, 100.0, 1000000.0, "Superseded Executive",
                "VP", False, True, False,
                "VP", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))

            # Amended transaction: 1,000 shares @ $100 ($100,000)
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING;
            """, (
                f"{acc_amend}:0001045810:NON_DERIVATIVE:1", acc_amend, today, "P",
                1000.0, 100.0, 10000.0, "Superseded Executive",
                "VP", False, True, False,
                "VP", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))

            # sm_amendments linkage: acc_amend supersedes acc_orig
            cur.execute("""
                INSERT INTO sm_amendments (
                    amendment_accession, base_accession, amendment_number, resolution_status, resolved_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (amendment_accession) DO NOTHING;
            """, (acc_amend, acc_orig, 1, "SUPERSEDED_OR_CORRECTED", now_iso))
        conn.commit()
        conn.close()

        engine = SmartMoneyEngine()
        result = engine.evaluate_smart_money_feeds("NVDA", window_days=90)
        sec_feed = result["feeds"]["sec_form4"]
        txs = [t for t in sec_feed["observations"] if t["insider_name"] == "Superseded Executive"]

        # Only 1 observation for this transaction sequence (superseded excluded)
        assert len(txs) == 1
        assert txs[0]["id"] == acc_amend
        assert txs[0]["shares_traded"] == 1000.0
        assert txs[0]["total_value_usd"] == 10000.0

    # 5. Public Available Historical Cutoff (No Future Lookahead)
    def test_public_information_historical_cutoff(self, pg_db, monkeypatch):
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        monkeypatch.setenv("ARX_STORAGE_BACKEND", "postgres")

        acc_early = "0001045810-26-900005E"
        acc_late = "0001045810-26-900005L"

        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            # Early filing: publicly available 2026-10-01
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_early, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-01", "2026-10-01T12:00:00+00:00", "<xml/>", "hash_early_5", "https://sec.gov/e"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (
                    accession_number, public_availability_at, source, source_reference
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_early, "2026-10-01T12:00:00+00:00", "SEC_INDEX", "ref_e"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING;
            """, (
                f"{acc_early}:0001045810:NON_DERIVATIVE:1", acc_early, "2026-10-01", "P",
                500.0, 100.0, 50000.0, "Early Buyer",
                "VP", False, True, False,
                "VP", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))

            # Late filing: publicly available 2026-10-20
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_late, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-20", "2026-10-20T12:00:00+00:00", "<xml/>", "hash_late_5", "https://sec.gov/l"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (
                    accession_number, public_availability_at, source, source_reference
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_late, "2026-10-20T12:00:00+00:00", "SEC_INDEX", "ref_l"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING;
            """, (
                f"{acc_late}:0001045810:NON_DERIVATIVE:1", acc_late, "2026-10-20", "P",
                500.0, 100.0, 50000.0, "Late Buyer",
                "VP", False, True, False,
                "VP", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))
        conn.commit()
        conn.close()

        # Query point-in-time as of 2026-10-10 under PUBLIC_AVAILABLE replay mode
        engine = SmartMoneyEngine()
        result = engine.evaluate_smart_money_feeds(
            "NVDA",
            window_days=90,
            as_of="2026-10-10T00:00:00+00:00",
            replay_mode="PUBLIC_AVAILABLE",
        )
        sec_feed = result["feeds"]["sec_form4"]
        obs_ids = [o["id"] for o in sec_feed["observations"]]

        # Early filing must be present; late filing must be strictly excluded
        assert acc_early in obs_ids
        assert acc_late not in obs_ids

        # Repository bulk query method test
        repo = PostgresSmartMoneyRepository(dsn=uri)
        bulk_txs = repo.get_bulk_transactions_as_of(
            as_of_watermark="2026-10-10T00:00:00+00:00",
            symbols=["NVDA"],
            min_date="2026-09-01",
            replay_mode="PUBLIC_AVAILABLE",
        )
        bulk_accs = [tx["accession_number"] for tx in bulk_txs.get("NVDA", [])]
        assert acc_early in bulk_accs
        assert acc_late not in bulk_accs

    # 6. ARX Observed Historical Cutoff (Knowledge Cutoff)
    def test_arx_observed_historical_cutoff(self, pg_db, monkeypatch):
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        monkeypatch.setenv("ARX_STORAGE_BACKEND", "postgres")

        acc_weekend = "0001045810-26-900006W"

        # Public acceptance: Friday 2026-10-09 18:00
        # ARX first observed / recorded: Monday 2026-10-12 09:00
        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_weekend, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-09", "2026-10-09T18:00:00+00:00",
                  "<xml/>", "hash_w_6", "https://sec.gov/w", "2026-10-12T09:00:00+00:00"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (
                    accession_number, public_availability_at, source, source_reference, recorded_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_weekend, "2026-10-09T18:00:00+00:00", "SEC_INDEX", "ref_w", "2026-10-12T09:00:00+00:00"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING;
            """, (
                f"{acc_weekend}:0001045810:NON_DERIVATIVE:1", acc_weekend, "2026-10-09", "P",
                700.0, 100.0, 70000.0, "Weekend Filer",
                "Director", True, False, False,
                "Director", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))
        conn.commit()
        conn.close()

        # Watermark W: Saturday 2026-10-10 12:00
        cutoff_w = "2026-10-10T12:00:00+00:00"

        engine = SmartMoneyEngine()

        # Under PUBLIC_AVAILABLE: Filing was public on Friday -> included
        res_pub = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=cutoff_w, replay_mode="PUBLIC_AVAILABLE")
        obs_pub = [o["id"] for o in res_pub["feeds"]["sec_form4"]["observations"]]
        assert acc_weekend in obs_pub

        # Under ARX_OBSERVED: ARX did not observe until Monday -> EXCLUDED
        res_arx = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=cutoff_w, replay_mode="ARX_OBSERVED")
        obs_arx = [o["id"] for o in res_arx["feeds"]["sec_form4"]["observations"]]
        assert acc_weekend not in obs_arx

    # 7. Parser Version Reproducibility
    def test_parser_version_reproducibility(self):
        sample_xml = """<?xml version="1.0"?>
        <ownershipDocument>
            <schemaVersion>X0508</schemaVersion>
            <documentType>4</documentType>
            <periodOfReport>2026-10-15</periodOfReport>
            <issuer>
                <issuerCik>0001045810</issuerCik>
                <issuerName>NVIDIA CORP</issuerName>
                <issuerTradingSymbol>NVDA</issuerTradingSymbol>
            </issuer>
            <reportingOwner>
                <reportingOwnerId>
                    <rptOwnerCik>0001234567</rptOwnerCik>
                    <rptOwnerName>Jensen Huang</rptOwnerName>
                </reportingOwnerId>
                <reportingOwnerRelationship>
                    <isDirector>1</isDirector>
                    <isOfficer>1</isOfficer>
                    <isTenPercentOwner>0</isTenPercentOwner>
                    <officerTitle>President and CEO</officerTitle>
                </reportingOwnerRelationship>
            </reportingOwner>
            <nonDerivativeTable>
                <nonDerivativeTransaction>
                    <securityTitle><value>Common Stock</value></securityTitle>
                    <transactionDate><value>2026-10-14</value></transactionDate>
                    <transactionCoding>
                        <transactionFormType>4</transactionFormType>
                        <transactionCode>P</transactionCode>
                        <equitySwapInvolved>0</equitySwapInvolved>
                    </transactionCoding>
                    <transactionAmounts>
                        <transactionShares><value>1000</value></transactionShares>
                        <transactionPricePerShare><value>120.50</value></transactionPricePerShare>
                        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
                    </transactionAmounts>
                    <ownershipNature>
                        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
                    </ownershipNature>
                </nonDerivativeTransaction>
            </nonDerivativeTable>
        </ownershipDocument>
        """

        filing1 = Form4XmlParser.parse_filing(sample_xml, accession_number="0001045810-26-999999")
        filing2 = Form4XmlParser.parse_filing(sample_xml, accession_number="0001045810-26-999999")

        assert len(filing1.transactions) == 1
        assert len(filing2.transactions) == 1

        tx1 = filing1.transactions[0]
        tx2 = filing2.transactions[0]

        # Verify Schema V2 classification fields
        assert tx1.is_derivative is False
        assert tx1.signal_type == SmartMoneySignalType.OPEN_MARKET_PURCHASE
        assert tx1.acquired_disposed == "A"
        assert tx1.is_positive_buying_conviction is True
        assert tx1.is_open_market_purchase_evidence is True

        # Verify deterministic byte/field reproducibility
        assert tx1.record_key == tx2.record_key
        assert tx1.total_value_usd == tx2.total_value_usd
        assert tx1.shares == tx2.shares
        assert tx1.price_per_share == tx2.price_per_share

    # 8. Missing / Ambiguous Timestamp Handling (No Fabricated Timestamps)
    def test_missing_or_ambiguous_timestamp_handling(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_ambig.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE form4_parsed_transactions (
            record_key TEXT PRIMARY KEY,
            accession_number TEXT,
            issuer_cik TEXT,
            issuer_symbol TEXT,
            reporting_owner_name TEXT,
            officer_title TEXT,
            is_director INTEGER,
            is_officer INTEGER,
            is_ten_percent_owner INTEGER,
            filing_date TEXT,
            transaction_date TEXT,
            transaction_code TEXT,
            signal_type TEXT,
            shares REAL,
            price_per_share REAL,
            total_value_usd REAL,
            acquired_disposed TEXT,
            is_positive_buying_conviction INTEGER,
            is_open_market_purchase_evidence INTEGER,
            acceptance_datetime TEXT,
            is_derivative INTEGER
        )""")
        # Insert record with NULL acceptance_datetime
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_AMBIG_1', '0001045810-26-900008N', '0001045810', 'NVDA', 'Ambiguous Timestamp Filer', 'Officer', 0, 1, 0,
            '2026-11-01', '2026-11-01', 'P', 'OPEN_MARKET_PURCHASE', 100.0, 100.0, 10000.0,
            'A', 1, 1, NULL, 0
        )""")
        conn.commit()
        conn.close()

        monkeypatch.setenv("ARX_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("SMART_MONEY_DB_PATH", db_path)
        monkeypatch.setattr(
            "analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db",
            lambda: sqlite3.connect(db_path)
        )

        engine = SmartMoneyEngine()
        result = engine.evaluate_smart_money_feeds("NVDA", window_days=90)
        txs = [t for t in result["feeds"]["sec_form4"]["observations"] if t["id"] == "0001045810-26-900008N"]

        assert len(txs) == 1
        # Must record availability_uncertain: True rather than defaulting to midnight UTC or fabricated timestamp
        assert txs[0]["availability_uncertain"] is True
        assert txs[0]["acceptance_datetime"] is None

    # 9. Source Coverage vs Neutral Conviction (Epistemic Honesty)
    def test_source_coverage_versus_neutral_conviction(self):
        engine = SmartMoneyEngine()

        # Query a symbol that has no local ingested filings under HOLD_SEC_001
        res = engine.evaluate_smart_money_feeds("XYZNONEXISTENT", window_days=90)
        sec_feed = res["feeds"]["sec_form4"]

        # Must report PIPELINE_PENDING, UNVERIFIED, NONE
        assert sec_feed["status"] == "PIPELINE_PENDING"
        assert sec_feed["direction"] == "UNVERIFIED"
        assert sec_feed["confidence"] == "NONE"
        assert sec_feed["coverage_status"] in ("SUPPORTED_SEC_ISSUER", "UNVERIFIED_ISSUER")
        assert "HOLD_SEC_001" in sec_feed["explanation"]
        assert "0 insider transactions filed" not in sec_feed["explanation"]

        # In assess_symbol, PIPELINE_HELD must not be converted into neutral conviction
        assessment = engine.assess_symbol("XYZNONEXISTENT")
        sec_sub = assessment["subdomains"]["sec_form_4"]
        assert sec_sub["status"] == SmartMoneyDomainStatus.PIPELINE_HELD.value
        assert sec_sub["directional_bias"] == SmartMoneyDirection.UNASSESSED.value
        assert assessment["pillar_eligibility"] is False

    # 10. Backup & Recovery Evidence
    def test_backup_and_recovery_evidence(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        secrets_dir = os.path.join(base_dir, ".secrets")

        backup_receipt_path = os.path.join(secrets_dir, "post_ingestion_backup_receipt.json")
        restore_receipt_path = os.path.join(secrets_dir, "post_ingestion_restore_verification_receipt.json")
        rehearsal_receipt_path = os.path.join(secrets_dir, "stage2_backup_rehearsal_receipt.json")

        assert os.path.exists(backup_receipt_path), "post_ingestion_backup_receipt.json missing"
        assert os.path.exists(restore_receipt_path), "post_ingestion_restore_verification_receipt.json missing"
        assert os.path.exists(rehearsal_receipt_path), "stage2_backup_rehearsal_receipt.json missing"

        with open(backup_receipt_path, encoding="utf-8") as f:
            backup_data = json.load(f)
        assert backup_data["status"] == "PASS"
        assert backup_data["verdict"] in ("PASS", "POST_INGESTION_BACKUP_CUSTODY_VERIFIED")
        assert len(backup_data["dump_sha256"]) == 64
        assert len(backup_data["manifest_sha256"]) == 64
        assert backup_data["dump_size_bytes"] > 0

        with open(restore_receipt_path, encoding="utf-8") as f:
            restore_data = json.load(f)
        assert restore_data["status"] == "PASS"
        assert restore_data["verdict"] in ("PASS", "POST_INGESTION_RESTORE_VERIFIED_IN_ISOLATED_CONTAINMENT")
        assert len(restore_data["verified_digest"]) == 64
        assert restore_data["records_verified"]["form4_raw_filings"] > 0
        assert restore_data["records_verified"]["form4_parsed_transactions"] > 0

        with open(rehearsal_receipt_path, encoding="utf-8") as f:
            rehearsal_data = json.load(f)
        assert rehearsal_data["overall_verdict"] in ("PASS", "STAGE_2_BACKUP_RESTORE_ACCEPTANCE_REHEARSAL_VERIFIED")

    # 11. SQLite Deterministic Friday/Monday Replay Isolation
    def test_sqlite_friday_monday_replay(self, tmp_path, monkeypatch):
        """
        Validates SM-DATA-GOV-001 Section 3 in SQLite:
        Filing published Friday evening (acceptance_datetime = 2026-10-16T21:00:00+00:00).
        Filing ingested by ARX Monday morning (created_at = 2026-10-19T09:00:00+00:00).
        Replay at Saturday 2026-10-17 12:00:
        - PUBLIC_AVAILABLE: Includes filing (was public on Friday).
        - ARX_OBSERVED: Excludes filing (ARX had not yet ingested it; no lookahead substitution).
        Replay at Monday 2026-10-19 12:00:
        - Both PUBLIC_AVAILABLE and ARX_OBSERVED include the filing.
        """
        db_path = str(tmp_path / "sqlite_replay.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""CREATE TABLE form4_parsed_transactions (
            record_key TEXT PRIMARY KEY,
            accession_number TEXT,
            issuer_cik TEXT,
            issuer_symbol TEXT,
            reporting_owner_name TEXT,
            officer_title TEXT,
            is_director INTEGER,
            is_officer INTEGER,
            is_ten_percent_owner INTEGER,
            filing_date TEXT,
            transaction_date TEXT,
            transaction_code TEXT,
            signal_type TEXT,
            shares REAL,
            price_per_share REAL,
            total_value_usd REAL,
            acquired_disposed TEXT,
            is_positive_buying_conviction INTEGER,
            is_open_market_purchase_evidence INTEGER,
            acceptance_datetime TEXT,
            is_derivative INTEGER,
            created_at TEXT
        )""")

        cur.execute("""CREATE TABLE sm_amendments (
            amendment_accession TEXT,
            base_accession TEXT,
            resolution_status TEXT
        )""")

        acc = "0001045810-26-900010W"
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_WEEKEND_1', ?, '0001045810', 'NVDA', 'Weekend Executive', 'Officer', 0, 1, 0,
            '2026-10-16', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 5000.0, 120.0, 600000.0,
            'A', 1, 1, '2026-10-16T21:00:00+00:00', 0, '2026-10-19T09:00:00+00:00'
        )""", (acc,))
        conn.commit()
        conn.close()

        monkeypatch.setenv("ARX_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("SMART_MONEY_DB_PATH", db_path)
        monkeypatch.setattr(
            "analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db",
            lambda: sqlite3.connect(db_path)
        )

        engine = SmartMoneyEngine()

        # Watermark W: Saturday 2026-10-17 12:00
        w_sat = "2026-10-17T12:00:00+00:00"

        # 1. PUBLIC_AVAILABLE on Saturday -> Includes filing
        res_pub_sat = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_sat, replay_mode="PUBLIC_AVAILABLE")
        obs_pub_sat = [t["id"] for t in res_pub_sat["feeds"]["sec_form4"]["observations"]]
        assert acc in obs_pub_sat, "PUBLIC_AVAILABLE must include filing disseminated on Friday"

        # 2. ARX_OBSERVED on Saturday -> Strictly EXCLUDES filing (not observed until Monday)
        res_arx_sat = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_sat, replay_mode="ARX_OBSERVED")
        obs_arx_sat = [t["id"] for t in res_arx_sat["feeds"]["sec_form4"]["observations"]]
        assert acc not in obs_arx_sat, "ARX_OBSERVED must exclude filing not ingested by ARX until Monday"

        # Watermark W: Monday 2026-10-19 12:00
        w_mon = "2026-10-19T12:00:00+00:00"

        # 3. ARX_OBSERVED on Monday afternoon -> Includes filing
        res_arx_mon = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_mon, replay_mode="ARX_OBSERVED")
        obs_arx_mon = [t["id"] for t in res_arx_mon["feeds"]["sec_form4"]["observations"]]
        assert acc in obs_arx_mon, "ARX_OBSERVED must include filing after Monday ingestion"

        # 4. PUBLIC_AVAILABLE on Monday afternoon -> Includes filing
        res_pub_mon = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_mon, replay_mode="PUBLIC_AVAILABLE")
        obs_pub_mon = [t["id"] for t in res_pub_mon["feeds"]["sec_form4"]["observations"]]
        assert acc in obs_pub_mon

    # 12. Replay Parameter Validation (Fail-Closed Robustness)
    def test_replay_parameter_validation(self):
        """
        Validates fail-closed input validation on as_of and replay_mode
        across Engine, API endpoint, and Repository.
        """
        engine = SmartMoneyEngine()

        # Engine evaluate_smart_money_feeds validation
        with pytest.raises(ValueError, match="Invalid replay_mode"):
            engine.evaluate_smart_money_feeds("NVDA", replay_mode="FORGED_MODE")

        with pytest.raises(ValueError, match="Invalid as_of timestamp"):
            engine.evaluate_smart_money_feeds("NVDA", as_of="not-a-valid-iso-date")

        # Engine assess_symbol validation
        with pytest.raises(ValueError, match="Invalid replay_mode"):
            engine.assess_symbol("NVDA", replay_mode="FUTURE_LEAK")

        with pytest.raises(ValueError, match="Invalid as_of timestamp"):
            engine.assess_symbol("NVDA", as_of="2026/13/45 99:99")

        # API get_smart_money_evidence validation
        with pytest.raises(HTTPException) as exc_info:
            get_smart_money_evidence("NVDA", replay_mode="INVALID_REPLAY")
        assert exc_info.value.status_code == 400
        assert "Invalid replay_mode" in exc_info.value.detail

        with pytest.raises(HTTPException) as exc_info:
            get_smart_money_evidence("NVDA", as_of="bad-date-time")
        assert exc_info.value.status_code == 400
        assert "Invalid as_of timestamp" in exc_info.value.detail

        # Repository parameter validation
        repo = PostgresSmartMoneyRepository(dsn="postgresql://localhost:5432/dummy")
        with pytest.raises(ValueError, match="Invalid replay_mode"):
            repo.get_bulk_transactions_as_of("2026-10-15T00:00:00Z", ["NVDA"], "2026-01-01", replay_mode="BAD")

        with pytest.raises(ValueError, match="Invalid as_of_watermark"):
            repo.get_bulk_transactions_as_of("invalid-iso-watermark", ["NVDA"], "2026-01-01")

        with pytest.raises(ValueError, match="Invalid replay_mode"):
            repo.get_ambiguous_amendment_symbols_as_of("2026-10-15T00:00:00Z", ["NVDA"], replay_mode="BAD")

        with pytest.raises(ValueError, match="Invalid as_of_watermark"):
            repo.get_ambiguous_amendment_symbols_as_of("invalid-iso-watermark", ["NVDA"])

    # 13. Amendment Supersession & Double-Counting Prevention
    def test_amendment_supersession_adversarial(self, tmp_path, monkeypatch):
        """
        Validates Form 4/A supersession integrity in SQLite:
        - When an amendment supersedes an original filing, only the amended transaction is counted.
        - Original transaction is suppressed to prevent double-counting.
        - Ambiguous amendments are excluded from positive conviction.
        """
        db_path = str(tmp_path / "sqlite_amend.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""CREATE TABLE form4_parsed_transactions (
            record_key TEXT PRIMARY KEY,
            accession_number TEXT,
            issuer_cik TEXT,
            issuer_symbol TEXT,
            reporting_owner_name TEXT,
            officer_title TEXT,
            is_director INTEGER,
            is_officer INTEGER,
            is_ten_percent_owner INTEGER,
            filing_date TEXT,
            transaction_date TEXT,
            transaction_code TEXT,
            signal_type TEXT,
            shares REAL,
            price_per_share REAL,
            total_value_usd REAL,
            acquired_disposed TEXT,
            is_positive_buying_conviction INTEGER,
            is_open_market_purchase_evidence INTEGER,
            acceptance_datetime TEXT,
            is_derivative INTEGER,
            created_at TEXT
        )""")

        cur.execute("""CREATE TABLE sm_amendments (
            amendment_accession TEXT,
            base_accession TEXT,
            resolution_status TEXT
        )""")

        acc_orig = "0001045810-26-900011O"
        acc_amend = "0001045810-26-900011A"
        acc_ambig = "0001045810-26-900012U"

        # 1. Original filing (erroneous 10,000 shares)
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_ORIG_1', ?, '0001045810', 'NVDA', 'CEO Jensen', 'CEO', 1, 1, 0,
            '2026-10-01', '2026-10-01', 'P', 'OPEN_MARKET_PURCHASE', 10000.0, 100.0, 1000000.0,
            'A', 1, 1, '2026-10-01T18:00:00+00:00', 0, '2026-10-01T18:05:00+00:00'
        )""", (acc_orig,))

        # 2. Amended filing (corrected 5,000 shares)
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_AMEND_1', ?, '0001045810', 'NVDA', 'CEO Jensen', 'CEO', 1, 1, 0,
            '2026-10-05', '2026-10-01', 'P', 'OPEN_MARKET_PURCHASE', 5000.0, 100.0, 500000.0,
            'A', 1, 1, '2026-10-05T18:00:00+00:00', 0, '2026-10-05T18:05:00+00:00'
        )""", (acc_amend,))

        # 3. Ambiguous amendment filing
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_AMBIG_1', ?, '0001045810', 'NVDA', 'Ambiguous Insider', 'Officer', 0, 1, 0,
            '2026-10-10', '2026-10-10', 'P', 'OPEN_MARKET_PURCHASE', 2000.0, 100.0, 200000.0,
            'A', 1, 1, '2026-10-10T18:00:00+00:00', 0, '2026-10-10T18:05:00+00:00'
        )""", (acc_ambig,))

        # sm_amendments linkage: acc_orig is SUPERSEDED, acc_ambig is AMENDMENT_TARGET_AMBIGUOUS
        cur.execute("""INSERT INTO sm_amendments VALUES (?, ?, 'SUPERSEDED')""", (acc_amend, acc_orig))
        cur.execute("""INSERT INTO sm_amendments VALUES (?, '0000000000-00-000000', 'AMENDMENT_TARGET_AMBIGUOUS')""", (acc_ambig,))

        conn.commit()
        conn.close()

        monkeypatch.setenv("ARX_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("SMART_MONEY_DB_PATH", db_path)
        monkeypatch.setattr(
            "analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db",
            lambda: sqlite3.connect(db_path)
        )

        engine = SmartMoneyEngine()
        result = engine.evaluate_smart_money_feeds("NVDA", window_days=90)
        observations = result["feeds"]["sec_form4"]["observations"]
        obs_ids = [o["id"] for o in observations]

        # 1. Superseded original must be excluded
        assert acc_orig not in obs_ids, f"Original {acc_orig} must be excluded because it was SUPERSEDED"

        # 2. Corrected amendment must be included
        assert acc_amend in obs_ids, f"Amended {acc_amend} must be included"

        # 3. Ambiguous amendment must be excluded
        assert acc_ambig not in obs_ids, f"Ambiguous {acc_ambig} must be excluded"

        # 4. Total shares must be exactly 5000 (no double-counting with original 10000)
        total_shares = sum(o["shares_traded"] for o in observations)
        assert total_shares == 5000.0, f"Expected 5,000 shares, got {total_shares} (double-counting detected!)"

    # 14. Point-in-Time Amendment Supersession: Friday-Original / Monday-Amendment / Saturday-Replay
    def test_point_in_time_amendment_friday_monday_replay(self, tmp_path, monkeypatch):
        """
        Adversarial regression test for SM-DATA-GOV-001.3:
        - Original filed Friday 2026-10-16 18:00 (10,000 shares)
        - Amendment filed Monday 2026-10-19 18:00 (5,000 shares)
        - Replay on Saturday 2026-10-17 12:00:
          * PUBLIC_AVAILABLE: Must return Friday original (10,000 shs). Future Monday amendment MUST NOT leak backward.
          * ARX_OBSERVED: Must return Friday original (10,000 shs).
        - Replay on Monday night 2026-10-19 23:59:
          * PUBLIC_AVAILABLE: Must return Monday amendment (5,000 shs). Original must be superseded.
          * ARX_OBSERVED: Must return Monday amendment (5,000 shs).
        """
        db_path = str(tmp_path / "sqlite_amend_pit.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""CREATE TABLE form4_parsed_transactions (
            record_key TEXT PRIMARY KEY,
            accession_number TEXT,
            issuer_cik TEXT,
            issuer_symbol TEXT,
            reporting_owner_name TEXT,
            officer_title TEXT,
            is_director INTEGER,
            is_officer INTEGER,
            is_ten_percent_owner INTEGER,
            filing_date TEXT,
            transaction_date TEXT,
            transaction_code TEXT,
            signal_type TEXT,
            shares REAL,
            price_per_share REAL,
            total_value_usd REAL,
            acquired_disposed TEXT,
            is_positive_buying_conviction INTEGER,
            is_open_market_purchase_evidence INTEGER,
            acceptance_datetime TEXT,
            is_derivative INTEGER,
            created_at TEXT
        )""")

        cur.execute("""CREATE TABLE sm_amendments (
            amendment_accession TEXT,
            base_accession TEXT,
            amendment_number INTEGER DEFAULT 1,
            resolution_status TEXT
        )""")

        acc_orig = "0001045810-26-900030O"
        acc_amend = "0001045810-26-900030A"

        # Friday original: 10,000 shares
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_ORIG_PIT', ?, '0001045810', 'NVDA', 'CEO Jensen', 'CEO', 1, 1, 0,
            '2026-10-16', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 10000.0, 100.0, 1000000.0,
            'A', 1, 1, '2026-10-16T18:00:00+00:00', 0, '2026-10-16T18:05:00+00:00'
        )""", (acc_orig,))

        # Monday amendment: 5,000 shares
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_AMEND_PIT', ?, '0001045810', 'NVDA', 'CEO Jensen', 'CEO', 1, 1, 0,
            '2026-10-19', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 5000.0, 100.0, 500000.0,
            'A', 1, 1, '2026-10-19T18:00:00+00:00', 0, '2026-10-19T18:05:00+00:00'
        )""", (acc_amend,))

        cur.execute("""INSERT INTO sm_amendments VALUES (?, ?, 1, 'SUPERSEDED')""", (acc_amend, acc_orig))
        conn.commit()
        conn.close()

        monkeypatch.setenv("ARX_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("SMART_MONEY_DB_PATH", db_path)
        monkeypatch.setattr(
            "analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db",
            lambda: sqlite3.connect(db_path)
        )

        engine = SmartMoneyEngine()
        w_sat = "2026-10-17T12:00:00+00:00"
        w_mon = "2026-10-19T23:59:59+00:00"

        # 1. Saturday replay (PUBLIC_AVAILABLE): Original must be present, amendment absent
        res_pub_sat = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_sat, replay_mode="PUBLIC_AVAILABLE")
        obs_pub_sat = res_pub_sat["feeds"]["sec_form4"]["observations"]
        assert [o["id"] for o in obs_pub_sat] == [acc_orig], "Saturday PUBLIC_AVAILABLE must include original filing"
        assert sum(o["shares_traded"] for o in obs_pub_sat) == 10000.0

        # 2. Saturday replay (ARX_OBSERVED): Original must be present, amendment absent
        res_arx_sat = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_sat, replay_mode="ARX_OBSERVED")
        obs_arx_sat = res_arx_sat["feeds"]["sec_form4"]["observations"]
        assert [o["id"] for o in obs_arx_sat] == [acc_orig], "Saturday ARX_OBSERVED must include original filing"
        assert sum(o["shares_traded"] for o in obs_arx_sat) == 10000.0

        # 3. Monday night replay (PUBLIC_AVAILABLE): Amendment must be present, original superseded
        res_pub_mon = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_mon, replay_mode="PUBLIC_AVAILABLE")
        obs_pub_mon = res_pub_mon["feeds"]["sec_form4"]["observations"]
        assert [o["id"] for o in obs_pub_mon] == [acc_amend], "Monday night PUBLIC_AVAILABLE must include amendment"
        assert sum(o["shares_traded"] for o in obs_pub_mon) == 5000.0

        # 4. Monday night replay (ARX_OBSERVED): Amendment must be present, original superseded
        res_arx_mon = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_mon, replay_mode="ARX_OBSERVED")
        obs_arx_mon = res_arx_mon["feeds"]["sec_form4"]["observations"]
        assert [o["id"] for o in obs_arx_mon] == [acc_amend], "Monday night ARX_OBSERVED must include amendment"
        assert sum(o["shares_traded"] for o in obs_arx_mon) == 5000.0

    # 15. Point-in-Time Partial Correction Line-Item Supersession
    def test_point_in_time_partial_correction(self, tmp_path, monkeypatch):
        """
        Adversarial test for partial corrections across time:
        - Base filing has two transactions: Jensen (10,000 shs) and Colette (500 shs).
        - Monday partial amendment only corrects Jensen to 1,000 shs; Colette is intact.
        - On Saturday: Both original Jensen (10,000) + Colette (500) = 10,500 shs.
        - On Monday night: Amended Jensen (1,000) + intact original Colette (500) = 1,500 shs.
        """
        db_path = str(tmp_path / "sqlite_part_amend.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""CREATE TABLE form4_parsed_transactions (
            record_key TEXT PRIMARY KEY,
            accession_number TEXT,
            issuer_cik TEXT,
            issuer_symbol TEXT,
            reporting_owner_name TEXT,
            officer_title TEXT,
            is_director INTEGER,
            is_officer INTEGER,
            is_ten_percent_owner INTEGER,
            filing_date TEXT,
            transaction_date TEXT,
            transaction_code TEXT,
            signal_type TEXT,
            shares REAL,
            price_per_share REAL,
            total_value_usd REAL,
            acquired_disposed TEXT,
            is_positive_buying_conviction INTEGER,
            is_open_market_purchase_evidence INTEGER,
            acceptance_datetime TEXT,
            is_derivative INTEGER,
            created_at TEXT
        )""")

        cur.execute("""CREATE TABLE sm_amendments (
            amendment_accession TEXT,
            base_accession TEXT,
            amendment_number INTEGER DEFAULT 1,
            resolution_status TEXT
        )""")

        acc_base = "0001045810-26-900040B"
        acc_amend = "0001045810-26-900040A"

        # Base: Jensen 10,000 shares
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_BASE_JENSEN', ?, '0001045810', 'NVDA', 'Jensen Huang', 'CEO', 1, 1, 0,
            '2026-10-16', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 10000.0, 100.0, 1000000.0,
            'A', 1, 1, '2026-10-16T18:00:00+00:00', 0, '2026-10-16T18:05:00+00:00'
        )""", (acc_base,))

        # Base: Colette 500 shares
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_BASE_COLETTE', ?, '0001045810', 'NVDA', 'Colette Kress', 'CFO', 0, 1, 0,
            '2026-10-16', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 500.0, 100.0, 50000.0,
            'A', 1, 1, '2026-10-16T18:00:00+00:00', 0, '2026-10-16T18:05:00+00:00'
        )""", (acc_base,))

        # Monday partial amendment: corrects Jensen to 1,000 shares
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_AMEND_JENSEN', ?, '0001045810', 'NVDA', 'Jensen Huang', 'CEO', 1, 1, 0,
            '2026-10-19', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 1000.0, 100.0, 100000.0,
            'A', 1, 1, '2026-10-19T18:00:00+00:00', 0, '2026-10-19T18:05:00+00:00'
        )""", (acc_amend,))

        cur.execute("""INSERT INTO sm_amendments VALUES (?, ?, 1, 'PARTIAL_CORRECTION')""", (acc_amend, acc_base))
        conn.commit()
        conn.close()

        monkeypatch.setenv("ARX_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("SMART_MONEY_DB_PATH", db_path)
        monkeypatch.setattr(
            "analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db",
            lambda: sqlite3.connect(db_path)
        )

        engine = SmartMoneyEngine()
        w_sat = "2026-10-17T12:00:00+00:00"
        w_mon = "2026-10-19T23:59:59+00:00"

        # Saturday: both original transactions active (10,500 total)
        res_sat = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_sat)
        obs_sat = res_sat["feeds"]["sec_form4"]["observations"]
        assert len(obs_sat) == 2
        assert sum(o["shares_traded"] for o in obs_sat) == 10500.0

        # Monday night: amended Jensen (1,000) + intact original Colette (500) = 1,500 total
        res_mon = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_mon)
        obs_mon = res_mon["feeds"]["sec_form4"]["observations"]
        assert len(obs_mon) == 2
        names = {o["insider_name"]: o["shares_traded"] for o in obs_mon}
        assert names["Jensen Huang"] == 1000.0
        assert names["Colette Kress"] == 500.0
        assert sum(o["shares_traded"] for o in obs_mon) == 1500.0

    # 16. Point-in-Time Amendment Chain Integrity
    def test_point_in_time_amendment_chain(self, tmp_path, monkeypatch):
        """
        Adversarial test for 3-stage amendment chain:
        - Original: Friday Oct 16 (10,000 shs)
        - Amendment 1: Monday Oct 19 (8,000 shs, amendment_number=1)
        - Amendment 2: Wednesday Oct 21 (5,000 shs, amendment_number=2)
        - Saturday replay: Original (10,000)
        - Tuesday replay: Amendment 1 (8,000)
        - Thursday replay: Amendment 2 (5,000)
        """
        db_path = str(tmp_path / "sqlite_amend_chain.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""CREATE TABLE form4_parsed_transactions (
            record_key TEXT PRIMARY KEY,
            accession_number TEXT,
            issuer_cik TEXT,
            issuer_symbol TEXT,
            reporting_owner_name TEXT,
            officer_title TEXT,
            is_director INTEGER,
            is_officer INTEGER,
            is_ten_percent_owner INTEGER,
            filing_date TEXT,
            transaction_date TEXT,
            transaction_code TEXT,
            signal_type TEXT,
            shares REAL,
            price_per_share REAL,
            total_value_usd REAL,
            acquired_disposed TEXT,
            is_positive_buying_conviction INTEGER,
            is_open_market_purchase_evidence INTEGER,
            acceptance_datetime TEXT,
            is_derivative INTEGER,
            created_at TEXT
        )""")

        cur.execute("""CREATE TABLE sm_amendments (
            amendment_accession TEXT,
            base_accession TEXT,
            amendment_number INTEGER DEFAULT 1,
            resolution_status TEXT
        )""")

        acc_orig = "0001045810-26-900050O"
        acc_am1 = "0001045810-26-900050A1"
        acc_am2 = "0001045810-26-900050A2"

        # Original: Oct 16, 10,000 shares
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_CHAIN_0', ?, '0001045810', 'NVDA', 'CEO Jensen', 'CEO', 1, 1, 0,
            '2026-10-16', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 10000.0, 100.0, 1000000.0,
            'A', 1, 1, '2026-10-16T18:00:00+00:00', 0, '2026-10-16T18:05:00+00:00'
        )""", (acc_orig,))

        # Amendment 1: Oct 19, 8,000 shares
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_CHAIN_1', ?, '0001045810', 'NVDA', 'CEO Jensen', 'CEO', 1, 1, 0,
            '2026-10-19', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 8000.0, 100.0, 800000.0,
            'A', 1, 1, '2026-10-19T18:00:00+00:00', 0, '2026-10-19T18:05:00+00:00'
        )""", (acc_am1,))

        # Amendment 2: Oct 21, 5,000 shares
        cur.execute("""INSERT INTO form4_parsed_transactions VALUES (
            'TX_CHAIN_2', ?, '0001045810', 'NVDA', 'CEO Jensen', 'CEO', 1, 1, 0,
            '2026-10-21', '2026-10-16', 'P', 'OPEN_MARKET_PURCHASE', 5000.0, 100.0, 500000.0,
            'A', 1, 1, '2026-10-21T18:00:00+00:00', 0, '2026-10-21T18:05:00+00:00'
        )""", (acc_am2,))

        cur.execute("""INSERT INTO sm_amendments VALUES (?, ?, 1, 'SUPERSEDED')""", (acc_am1, acc_orig))
        cur.execute("""INSERT INTO sm_amendments VALUES (?, ?, 2, 'SUPERSEDED')""", (acc_am2, acc_orig))
        conn.commit()
        conn.close()

        monkeypatch.setenv("ARX_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("SMART_MONEY_DB_PATH", db_path)
        monkeypatch.setattr(
            "analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db",
            lambda: sqlite3.connect(db_path)
        )

        engine = SmartMoneyEngine()

        # 1. Saturday Oct 17: Original active (10,000)
        res_sat = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of="2026-10-17T12:00:00+00:00")
        obs_sat = res_sat["feeds"]["sec_form4"]["observations"]
        assert [o["id"] for o in obs_sat] == [acc_orig]
        assert sum(o["shares_traded"] for o in obs_sat) == 10000.0

        # 2. Tuesday Oct 20: Amendment 1 active (8,000)
        res_tue = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of="2026-10-20T12:00:00+00:00")
        obs_tue = res_tue["feeds"]["sec_form4"]["observations"]
        assert [o["id"] for o in obs_tue] == [acc_am1]
        assert sum(o["shares_traded"] for o in obs_tue) == 8000.0

        # 3. Thursday Oct 22: Amendment 2 active (5,000)
        res_thu = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of="2026-10-22T12:00:00+00:00")
        obs_thu = res_thu["feeds"]["sec_form4"]["observations"]
        assert [o["id"] for o in obs_thu] == [acc_am2]
        assert sum(o["shares_traded"] for o in obs_thu) == 5000.0

    # 17. PostgreSQL Point-in-Time Amendment Supersession (Full Lifecycle)
    def test_postgres_point_in_time_amendment_supersession(self, pg_db, monkeypatch):
        """
        Validates point-in-time amendment supersession in PostgreSQL:
        - Original filed Friday 2026-10-16 18:00 (10,000 shares)
        - Amendment filed Monday 2026-10-19 18:00 (5,000 shares)
        - Saturday replay: Original active (10,000 shs) under both PUBLIC_AVAILABLE and ARX_OBSERVED
        - Monday night replay: Amendment active (5,000 shs), original superseded
        - Repository get_bulk_transactions_as_of point-in-time consistency
        """
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        monkeypatch.setenv("ARX_STORAGE_BACKEND", "postgres")

        acc_orig = "0001045810-26-900060O"
        acc_amend = "0001045810-26-900060A"

        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            # 1. Friday Original filing
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_orig, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-16", "2026-10-16T18:00:00+00:00",
                  "<xml/>", "hash_pg_orig", "https://sec.gov/o", "2026-10-16T18:05:00+00:00"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (
                    accession_number, public_availability_at, source, recorded_at
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_orig, "2026-10-16T18:00:00+00:00", "SEC_INDEX", "2026-10-16T18:05:00+00:00"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING;
            """, (
                f"{acc_orig}:0001045810:NON_DERIVATIVE:1", acc_orig, "2026-10-16", "P",
                10000.0, 100.0, 1000000.0, "CEO Jensen",
                "CEO", True, True, False,
                "CEO", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))

            # 2. Monday Amendment filing
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_amend, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-19", "2026-10-19T18:00:00+00:00",
                  "<xml/>", "hash_pg_amend", "https://sec.gov/a", "2026-10-19T18:05:00+00:00"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (
                    accession_number, public_availability_at, source, recorded_at
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (acc_amend, "2026-10-19T18:00:00+00:00", "SEC_INDEX", "2026-10-19T18:05:00+00:00"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING;
            """, (
                f"{acc_amend}:0001045810:NON_DERIVATIVE:1", acc_amend, "2026-10-16", "P",
                5000.0, 100.0, 500000.0, "CEO Jensen",
                "CEO", True, True, False,
                "CEO", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))

            # 3. sm_amendments linkage
            cur.execute("""
                INSERT INTO sm_amendments (
                    amendment_accession, base_accession, amendment_number, resolution_status, resolved_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (amendment_accession) DO NOTHING;
            """, (acc_amend, acc_orig, 1, "SUPERSEDED", "2026-10-19T18:05:00+00:00"))
        conn.commit()
        conn.close()

        engine = SmartMoneyEngine()
        w_sat = "2026-10-17T12:00:00+00:00"
        w_mon = "2026-10-19T23:59:59+00:00"

        # Saturday replay (Engine): must return original (10,000 shs)
        res_sat_pub = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_sat, replay_mode="PUBLIC_AVAILABLE")
        obs_sat_pub = [o for o in res_sat_pub["feeds"]["sec_form4"]["observations"] if o["id"] in (acc_orig, acc_amend)]
        assert [o["id"] for o in obs_sat_pub] == [acc_orig]
        assert sum(o["shares_traded"] for o in obs_sat_pub) == 10000.0

        res_sat_arx = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_sat, replay_mode="ARX_OBSERVED")
        obs_sat_arx = [o for o in res_sat_arx["feeds"]["sec_form4"]["observations"] if o["id"] in (acc_orig, acc_amend)]
        assert [o["id"] for o in obs_sat_arx] == [acc_orig]
        assert sum(o["shares_traded"] for o in obs_sat_arx) == 10000.0

        # Monday night replay (Engine): must return amendment (5,000 shs)
        res_mon_pub = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_mon, replay_mode="PUBLIC_AVAILABLE")
        obs_mon_pub = [o for o in res_mon_pub["feeds"]["sec_form4"]["observations"] if o["id"] in (acc_orig, acc_amend)]
        assert [o["id"] for o in obs_mon_pub] == [acc_amend]
        assert sum(o["shares_traded"] for o in obs_mon_pub) == 5000.0

        res_mon_arx = engine.evaluate_smart_money_feeds("NVDA", window_days=90, as_of=w_mon, replay_mode="ARX_OBSERVED")
        obs_mon_arx = [o for o in res_mon_arx["feeds"]["sec_form4"]["observations"] if o["id"] in (acc_orig, acc_amend)]
        assert [o["id"] for o in obs_mon_arx] == [acc_amend]
        assert sum(o["shares_traded"] for o in obs_mon_arx) == 5000.0

        # Repository bulk query method test
        repo = PostgresSmartMoneyRepository(dsn=uri)
        bulk_sat = repo.get_bulk_transactions_as_of(
            as_of_watermark=w_sat,
            symbols=["NVDA"],
            min_date="2026-10-01",
            replay_mode="PUBLIC_AVAILABLE",
        )
        sat_target_accs = [t["accession_number"] for t in bulk_sat.get("NVDA", []) if t["accession_number"] in (acc_orig, acc_amend)]
        assert sat_target_accs == [acc_orig]

        bulk_mon = repo.get_bulk_transactions_as_of(
            as_of_watermark=w_mon,
            symbols=["NVDA"],
            min_date="2026-10-01",
            replay_mode="PUBLIC_AVAILABLE",
        )
        mon_target_accs = [t["accession_number"] for t in bulk_mon.get("NVDA", []) if t["accession_number"] in (acc_orig, acc_amend)]
        assert mon_target_accs == [acc_amend]
