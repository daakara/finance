"""
ARX Terminal — Directive SM-ACT-004 Deterministic Defect & Architecture Suite

Validates:
1. Worker standby mode without crash when ARX_SEC_EGRESS_ROLE != 'OWNER'
2. Table II derivative purchase exclusion (Code P derivatives do not trigger bullish conviction)
3. Form 4/A amendment deduplication (superseded filings excluded from feed observations)
4. Dual-backend storage resolution parity (PostgreSQL query path and SQLite fallback)
5. Directive SM-ACT-004R PostgreSQL classification & amendment reconciliation
"""

import os
import sqlite3
import tempfile
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

psycopg2 = pytest.importorskip("psycopg2")
pgserver = pytest.importorskip("pgserver")

from analyst_dashboard.smart_money.worker import (
    SmartMoneyIngestionWorker,
    WorkerHealthState,
    get_sec_egress_role,
)
from analyst_dashboard.smart_money.storage.config import get_storage_backend
from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine


class TestWorkerStandbyMode:
    """Verifies Worker avoids crashing in rehearsal or non-production environments."""

    @pytest.mark.parametrize("role", ["DISABLED", "REVOKED", "ANALYSIS_ONLY", "STANDBY", ""])
    def test_worker_standby_preflight_and_start(self, role, monkeypatch):
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", role)
        worker = SmartMoneyIngestionWorker()

        # Direct preflight must fail closed with UNAUTHORIZED_WORKER_ROLE per contract
        with pytest.raises(RuntimeError, match="UNAUTHORIZED_WORKER_ROLE"):
            worker.validate_preflight()
        assert worker.state == WorkerHealthState.FATAL

        # But start(single_cycle=True) must enter healthy STANDBY without raising or crashing
        worker.start(single_cycle=True)
        assert worker.state == WorkerHealthState.STANDBY

        # Health envelope must report healthy subsystem in STANDBY
        env = worker.get_health_envelope()
        assert env["worker_subsystem_healthy"] is True
        assert env["status"] == "STANDBY"

    def test_worker_start_single_cycle_exits_cleanly(self, monkeypatch):
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "DISABLED")
        worker = SmartMoneyIngestionWorker()

        # start(single_cycle=True) must exit without unhandled exception or sys.exit crash
        worker.start(single_cycle=True)
        assert worker.state == WorkerHealthState.STANDBY


class TestDerivativePurchaseExclusion:
    """Verifies Table II derivative transactions coded P are never misclassified as bullish open-market buys."""

    @pytest.fixture
    def test_db(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_smart_money.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
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
                is_positive_buying_conviction INTEGER,
                is_open_market_purchase_evidence INTEGER,
                acceptance_datetime TEXT,
                footnotes_json TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE sm_source_availability_ledger (
                accession_number TEXT PRIMARY KEY,
                public_availability_at TEXT,
                source TEXT,
                source_reference TEXT,
                recorded_at TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE sm_amendments (
                accession_number TEXT PRIMARY KEY,
                amended_accession_number TEXT,
                relationship TEXT NOT NULL,
                detected_at TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()

        monkeypatch.setenv("SMART_MONEY_DB_PATH", db_path)
        monkeypatch.setattr(
            "analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db",
            lambda: sqlite3.connect(db_path)
        )
        return db_path

    def test_table_ii_put_option_purchase_excluded_from_buys(self, test_db):
        conn = sqlite3.connect(test_db)
        cur = conn.cursor()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Authentic Table I Open-Market Stock Purchase ($100k)
        cur.execute("""
            INSERT INTO form4_parsed_transactions (
                record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
                is_officer, officer_title, is_director, is_ten_percent_owner, filing_date,
                transaction_date, transaction_code, signal_type, shares, price_per_share, total_value_usd,
                acquired_disposed, is_derivative, is_positive_buying_conviction, is_open_market_purchase_evidence,
                acceptance_datetime
            ) VALUES (
                'TX_STOCK_BUY_001', '0000000001-26-000001', '0001045810', 'NVDA', 'Jensen Huang',
                1, 'CEO', 1, 0, ?, ?, 'P', 'OPEN_MARKET_PURCHASE', 1000, 100.0, 100000.0,
                'A', 0, 1, 1, ?
            );
        """, (today, today, now_iso))

        # 2. Table II Derivative Put/Call Option Purchase ($500k) - coded 'P' but is_derivative=1
        cur.execute("""
            INSERT INTO form4_parsed_transactions (
                record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
                is_officer, officer_title, is_director, is_ten_percent_owner, filing_date,
                transaction_date, transaction_code, signal_type, shares, price_per_share, total_value_usd,
                acquired_disposed, is_derivative, is_positive_buying_conviction, is_open_market_purchase_evidence,
                acceptance_datetime
            ) VALUES (
                'TX_DERIV_BUY_002', '0000000001-26-000002', '0001045810', 'NVDA', 'Colette Kress',
                1, 'CFO', 0, 0, ?, ?, 'P', 'DERIVATIVE_PURCHASE', 5000, 100.0, 500000.0,
                'A', 1, 0, 0, ?
            );
        """, (today, today, now_iso))
        conn.commit()
        conn.close()

        engine = SmartMoneyEngine()
        result = engine.evaluate_smart_money_feeds("NVDA", window_days=90)
        sec_feed = result["feeds"]["sec_form4"]
        txs = sec_feed["observations"]

        # 2 total transactions in feed
        assert len(txs) == 2

        stock_tx = next(t for t in txs if t["insider_name"] == "Jensen Huang")
        deriv_tx = next(t for t in txs if t["insider_name"] == "Colette Kress")

        # Table I stock buy must be classified as buy
        assert stock_tx["is_buy"] is True
        assert stock_tx["is_derivative"] is False
        assert stock_tx["transaction_type"] == "Purchase (Direct Open Market)"

        # Table II derivative purchase must NOT be classified as buy
        assert deriv_tx["is_buy"] is False
        assert deriv_tx["is_derivative"] is True
        assert deriv_tx["transaction_type"] == "Derivative Purchase"

        # Summary statistics: Only $100k buy volume, NOT $600k!
        buys = [t for t in txs if t.get("is_buy")]
        assert len(buys) == 1
        assert sum(t["total_value_usd"] for t in buys) == 100000.0


class TestAmendmentDeduplication:
    """Verifies superseded Form 4 filings are excluded when Form 4/A amendments exist."""

    @pytest.fixture
    def test_db_amendments(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_amendments.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
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
                is_positive_buying_conviction INTEGER,
                is_open_market_purchase_evidence INTEGER,
                acceptance_datetime TEXT,
                footnotes_json TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE sm_amendments (
                accession_number TEXT PRIMARY KEY,
                amended_accession_number TEXT,
                relationship TEXT NOT NULL,
                detected_at TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()

        monkeypatch.setenv("SMART_MONEY_DB_PATH", db_path)
        monkeypatch.setattr(
            "analyst_dashboard.smart_money.form4_ingestion.connect_smart_money_db",
            lambda: sqlite3.connect(db_path)
        )
        return db_path

    def test_superseded_filing_excluded_from_observations(self, test_db_amendments):
        conn = sqlite3.connect(test_db_amendments)
        cur = conn.cursor()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Original Form 4 with erroneous $1,000,000 volume (10,000 shares)
        orig_acc = "0000000002-26-000001"
        cur.execute("""
            INSERT INTO form4_parsed_transactions (
                record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
                is_officer, officer_title, is_director, is_ten_percent_owner, filing_date,
                transaction_date, transaction_code, signal_type, shares, price_per_share, total_value_usd,
                acquired_disposed, is_derivative, is_positive_buying_conviction, is_open_market_purchase_evidence,
                acceptance_datetime
            ) VALUES (
                'TX_ORIG_001', ?, '0001045810', 'NVDA', 'Jensen Huang',
                1, 'CEO', 1, 0, ?, ?, 'P', 'OPEN_MARKET_PURCHASE', 10000, 100.0, 1000000.0,
                'A', 0, 1, 1, ?
            );
        """, (orig_acc, today, today, now_iso))

        # 2. Form 4/A Amendment correcting volume to $100,000 (1,000 shares)
        amend_acc = "0000000002-26-000002"
        cur.execute("""
            INSERT INTO form4_parsed_transactions (
                record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
                is_officer, officer_title, is_director, is_ten_percent_owner, filing_date,
                transaction_date, transaction_code, signal_type, shares, price_per_share, total_value_usd,
                acquired_disposed, is_derivative, is_positive_buying_conviction, is_open_market_purchase_evidence,
                acceptance_datetime
            ) VALUES (
                'TX_AMEND_002', ?, '0001045810', 'NVDA', 'Jensen Huang',
                1, 'CEO', 1, 0, ?, ?, 'P', 'OPEN_MARKET_PURCHASE', 1000, 100.0, 100000.0,
                'A', 0, 1, 1, ?
            );
        """, (amend_acc, today, today, now_iso))

        # 3. sm_amendments linkage marking original as SUPERSEDED_OR_CORRECTED
        cur.execute("""
            INSERT INTO sm_amendments (accession_number, amended_accession_number, relationship, detected_at)
            VALUES (?, ?, 'SUPERSEDED_OR_CORRECTED', ?);
        """, (orig_acc, amend_acc, now_iso))
        cur.execute("""
            INSERT INTO sm_amendments (accession_number, amended_accession_number, relationship, detected_at)
            VALUES (?, ?, 'EXPLICIT_XML_AMENDMENT', ?);
        """, (amend_acc, orig_acc, now_iso))
        conn.commit()
        conn.close()

        engine = SmartMoneyEngine()
        result = engine.evaluate_smart_money_feeds("NVDA", window_days=90)
        sec_feed = result["feeds"]["sec_form4"]
        txs = sec_feed["observations"]

        # Only 1 transaction must be present: the amendment!
        assert len(txs) == 1
        assert txs[0]["id"] == amend_acc
        assert txs[0]["total_value_usd"] == 100000.0

        # Volume must NOT double-count: $100k, NOT $1.1M
        buys = [t for t in txs if t.get("is_buy")]
        assert len(buys) == 1
        assert sum(t["total_value_usd"] for t in buys) == 100000.0


class TestDualBackendStorageParity:
    """Verifies runtime dual-backend storage resolution."""

    def test_storage_backend_selection(self, monkeypatch):
        monkeypatch.delenv("SMART_MONEY_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert get_storage_backend() == "sqlite"

        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", "postgresql://user:pass@localhost:5432/arx")
        assert get_storage_backend() == "postgres"


@pytest.fixture
def pg_db():
    """Isolated Postgres server for defect reproduction & closure verification."""
    pgdata = tempfile.mkdtemp(prefix="arx_sm_act_004r_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)
    conn.close()

    yield {"uri": uri, "server": server}
    server.cleanup()


class TestPostgreSqlDerivativeClassificationDefectAndClosure:
    """
    Directive SM-ACT-004R: Verifies that Table II derivative transactions (even with Code 'P')
    do not masquerade as common-stock purchases, and missing derivative metadata fails closed.
    """

    def test_postgres_table_ii_derivative_not_classified_as_buy(self, pg_db, monkeypatch):
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            # 1. Raw filing
            acc = "0000000001-26-000001"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc, "0001045810", "NVDA", "NVIDIA CORP", today, now_iso, "<xml/>", "h1", "https://sec.gov/1"))

            # 2. Availability ledger
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (
                    accession_number, public_availability_at, source, source_reference
                ) VALUES (%s, %s, %s, %s);
            """, (acc, now_iso, "SEC_INDEX", "ref1"))

            # 3. Table I Common Stock Purchase: $100,000 (Jensen Huang, 1000 shs @ $100)
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                f"{acc}:0001045810:NON_DERIVATIVE:1", acc, today, "P",
                1000.0, 100.0, 100000.0, "Jensen Huang",
                "CEO", True, True, False,
                "CEO", "D", False, "OPEN_MARKET_PURCHASE",
                "A", True, True
            ))

            # 4. Table II Derivative Call Option Purchase: $500,000 (Colette Kress, 5000 shs @ $100, code P)
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                f"{acc}:0001045810:DERIVATIVE:1", acc, today, "P",
                5000.0, 100.0, 500000.0, "Colette Kress",
                "CFO", False, True, False,
                "CFO", "D", True, "DERIVATIVE_PURCHASE",
                "A", False, False
            ))

            # 5. Row with unverified/missing derivative metadata and non-matching ID
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                f"{acc}:0001045810:UNKNOWN_TYPE:1", acc, today, "P",
                2000.0, 100.0, 200000.0, "Jay Puri",
                "EVP", False, True, False,
                "EVP", "D", False, None,
                None, False, False
            ))
        conn.commit()
        conn.close()

        engine = SmartMoneyEngine()
        result = engine.evaluate_smart_money_feeds("NVDA", window_days=90)
        sec_feed = result["feeds"]["sec_form4"]
        txs = sec_feed["observations"]

        assert len(txs) == 3
        huang = next(t for t in txs if t["insider_name"] == "Jensen Huang")
        kress = next(t for t in txs if t["insider_name"] == "Colette Kress")
        puri = next(t for t in txs if t["insider_name"] == "Jay Puri")

        # Table I verified common stock purchase
        assert huang["is_buy"] is True
        assert huang["is_derivative"] is False
        assert huang["transaction_type"] == "Purchase (Direct Open Market)"

        # Table II derivative purchase (Code P) must NOT be a buy
        assert kress["is_buy"] is False
        assert kress["is_derivative"] is True
        assert kress["transaction_type"] == "Derivative Purchase"

        # Missing metadata row must fail closed and NOT be treated as a buy
        assert puri["is_buy"] is False

        # Only Jensen Huang ($100k) counted in buys, NOT $600k or $800k!
        buys = [t for t in txs if t.get("is_buy")]
        assert len(buys) == 1
        assert sum(t["total_value_usd"] for t in buys) == 100000.0


class TestPostgreSqlAmendmentReconciliationAcrossScenarios:
    """
    Directive SM-ACT-004R: Verifies PostgreSQL amendment reconciliation:
    - Full supersession excludes original filing
    - Partial correction preserves valid unchanged transactions
    - Additional transactions are merged
    - Unresolved/ambiguous amendments are excluded
    - Repeated ingestion is idempotent
    """

    def test_postgres_amendment_reconciliation(self, pg_db, monkeypatch):
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            # 1. Base Filing ACC-01 with 2 transactions:
            # Tx 1: Huang $1,000,000 (10,000 shs)
            # Tx 2: Kress $50,000 (500 shs)
            acc_base = "0000000002-26-000001"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc_base, "0001045810", "NVDA", "NVIDIA CORP", today, now_iso, "<xml/>", "hb", "https://sec.gov/b"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source)
                VALUES (%s, %s, %s);
            """, (acc_base, now_iso, "SEC_INDEX"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES
                (%s, %s, %s, 'P', 10000, 100, 1000000, 'Jensen Huang', 'CEO', true, true, false, 'CEO', 'D', false, 'OPEN_MARKET_PURCHASE', 'A', true, true),
                (%s, %s, %s, 'P', 500, 100, 50000, 'Colette Kress', 'CFO', false, true, false, 'CFO', 'D', false, 'OPEN_MARKET_PURCHASE', 'A', true, true);
            """, (f"{acc_base}:0001045810:NON_DERIVATIVE:1", acc_base, today, f"{acc_base}:0001045810:NON_DERIVATIVE:2", acc_base, today))

            # 2. Partial Correction Amendment ACC-02:
            # Only corrects Huang to $100,000 (1,000 shs). Does not include Kress.
            acc_amend = "0000000002-26-000002"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc_amend, "0001045810", "NVDA", "NVIDIA CORP", today, now_iso, "<xml/>", "ha", "https://sec.gov/a"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source)
                VALUES (%s, %s, %s);
            """, (acc_amend, now_iso, "SEC_INDEX"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES
                (%s, %s, %s, 'P', 1000, 100, 100000, 'Jensen Huang', 'CEO', true, true, false, 'CEO', 'D', false, 'OPEN_MARKET_PURCHASE', 'A', true, true);
            """, (f"{acc_amend}:0001045810:NON_DERIVATIVE:1", acc_amend, today))

            # Partial amendment linkage:
            cur.execute("""
                INSERT INTO sm_amendments (amendment_accession, base_accession, amendment_number, resolution_status, resolved_at)
                VALUES (%s, %s, 1, 'PARTIAL_CORRECTION', %s);
            """, (acc_amend, acc_base, now_iso))

            # 3. Unresolved/Ambiguous Amendment ACC-03:
            # Must be excluded from observations without suppressing base filing.
            acc_ambig = "0000000002-26-000003"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc_ambig, "0001045810", "NVDA", "NVIDIA CORP", today, now_iso, "<xml/>", "hamb", "https://sec.gov/amb"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source)
                VALUES (%s, %s, %s);
            """, (acc_ambig, now_iso, "SEC_INDEX"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES
                (%s, %s, %s, 'P', 9999, 100, 999900, 'Unknown Insider', 'Director', true, false, false, 'Director', 'D', false, 'OPEN_MARKET_PURCHASE', 'A', true, true);
            """, (f"{acc_ambig}:0001045810:NON_DERIVATIVE:1", acc_ambig, today))
            cur.execute("""
                INSERT INTO sm_amendments (amendment_accession, base_accession, amendment_number, resolution_status, resolved_at)
                VALUES (%s, NULL, 1, 'AMENDMENT_TARGET_AMBIGUOUS', %s);
            """, (acc_ambig, now_iso))
        conn.commit()
        conn.close()

        engine = SmartMoneyEngine()
        result = engine.evaluate_smart_money_feeds("NVDA", window_days=90)
        sec_feed = result["feeds"]["sec_form4"]
        txs = sec_feed["observations"]

        # Expected:
        # - Ambiguous amendment ACC-03 is EXCLUDED
        # - Base Huang ($1M) is superseded by Amendment Huang ($100k)
        # - Base Kress ($50k) is NOT suppressed and remains valid!
        assert len(txs) == 2
        insider_names = [t["insider_name"] for t in txs]
        assert "Jensen Huang" in insider_names
        assert "Colette Kress" in insider_names
        assert "Unknown Insider" not in insider_names

        buys = [t for t in txs if t.get("is_buy")]
        assert len(buys) == 2
        huang_tx = next(t for t in buys if t["insider_name"] == "Jensen Huang")
        kress_tx = next(t for t in buys if t["insider_name"] == "Colette Kress")

        assert huang_tx["total_value_usd"] == 100000.0
        assert kress_tx["total_value_usd"] == 50000.0
        assert sum(t["total_value_usd"] for t in buys) == 150000.0


class TestBoundedCanaryObservationCycle:
    """
    Directive SM-ACT-004R Section 5: Prove bounded canary controls:
    Verify worker can execute exactly one authorized NVDA observation cycle
    with defined limits and stop conditions in synthetic/cache-only mode without SEC egress.
    """

    def test_bounded_nvda_observation_cycle_without_sec_egress(self, pg_db, monkeypatch):
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "DISABLED")
        monkeypatch.setenv("ARX_ENVIRONMENT", "rehearsal")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        # Seed synthetic NVDA data
        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            acc = "0000000003-26-000001"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc, "0001045810", "NVDA", "NVIDIA CORP", today, now_iso, "<xml/>", "hcanary", "https://sec.gov/canary"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source)
                VALUES (%s, %s, %s);
            """, (acc, now_iso, "SEC_INDEX"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, 'P', 2500, 120, 300000, 'Jensen Huang', 'CEO', true, true, false, 'CEO', 'D', false, 'OPEN_MARKET_PURCHASE', 'A', true, true);
            """, (f"{acc}:0001045810:NON_DERIVATIVE:1", acc, today))
        conn.commit()
        conn.close()

        # 1. Verify worker stays safely in STANDBY without live egress
        worker = SmartMoneyIngestionWorker(dsn=uri)
        worker.start(single_cycle=True)
        assert worker.state == WorkerHealthState.STANDBY

        # 2. Evaluate bounded NVDA feed in synthetic/cache-only mode
        engine = SmartMoneyEngine()
        res = engine.evaluate_smart_money_feeds("NVDA", window_days=90)

        assert "feeds" in res
        sec_feed = res["feeds"]["sec_form4"]
        assert sec_feed["status"] == "VERIFIED"
        assert sec_feed["observation_count"] == 1
        assert sec_feed["observations"][0]["insider_name"] == "Jensen Huang"
        assert sec_feed["observations"][0]["total_value_usd"] == 300000.0
        assert sec_feed["direction"] == "BULLISH"


class TestTaxWithholdingAndDispositionClassification:
    """
    Directive SM-ACT-016: Code F Tax-Withholding and Non-Discretionary Disposition Classification.
    Validates:
    1. Code F (TAX_WITHHOLDING) is preserved as verified filing evidence but excluded from discretionary insider-selling conviction.
    2. Code G (GIFT) and Code D (ISSUER_DISPOSITION) are excluded from insider sales.
    3. Code S (OPEN_MARKET_SALE) remains correctly classified as discretionary insider sales.
    4. Standalone tax withholding does not produce false-positive BEARISH feed direction.
    """

    def test_nvda_tax_withholding_and_dispositions_postgres(self, pg_db, monkeypatch):
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        monkeypatch.setenv("ARX_ENVIRONMENT", "rehearsal")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            # 1. Filing 1: Scott Gawel (Code F - Tax Withholding)
            acc_f = "0001243821-26-000007"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc_f, "0001045810", "NVDA", "NVIDIA CORP", today, now_iso, "<xml/>", "h_gawel", "https://sec.gov/f"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source)
                VALUES (%s, %s, %s);
            """, (acc_f, now_iso, "SEC_INDEX"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, 'F', 2932, 212.17, 622082.44, 'GAWEL SCOTT', 'Officer', false, true, false, 'Controller', 'D', false, 'TAX_WITHHOLDING', 'D', false, false);
            """, (f"{acc_f}:0001243821:NON_DERIVATIVE:0", acc_f, today))

            # 2. Filing 2: Tench Coxe (Code G - Gift)
            acc_g = "0001197647-26-000011"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc_g, "0001045810", "NVDA", "NVIDIA CORP", today, now_iso, "<xml/>", "h_coxe", "https://sec.gov/g"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source)
                VALUES (%s, %s, %s);
            """, (acc_g, now_iso, "SEC_INDEX"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, 'G', 500000, 0, 0, 'COXE TENCH', 'Director', true, false, false, NULL, 'D', false, 'GIFT', 'D', false, false);
            """, (f"{acc_g}:0001197647:NON_DERIVATIVE:0", acc_g, today))

            # 3. Filing 3: Timothy Teter (Code S - Open Market Sale)
            acc_s = "0001696841-26-000014"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc_s, "0001045810", "NVDA", "NVIDIA CORP", today, now_iso, "<xml/>", "h_teter", "https://sec.gov/s"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source)
                VALUES (%s, %s, %s);
            """, (acc_s, now_iso, "SEC_INDEX"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, 'S', 12483, 222.1932, 2773637.72, 'Teter Timothy S.', 'Officer', false, true, false, 'General Counsel', 'D', false, 'OPEN_MARKET_SALE', 'D', false, false);
            """, (f"{acc_s}:0001696841:NON_DERIVATIVE:0", acc_s, today))

        conn.commit()
        conn.close()

        engine = SmartMoneyEngine()
        res = engine.evaluate_smart_money_feeds("NVDA", window_days=90)

        assert "feeds" in res
        sec_feed = res["feeds"]["sec_form4"]
        assert sec_feed["status"] == "VERIFIED"
        assert sec_feed["observation_count"] == 3

        # Lookup observations by insider name
        obs_by_name = {o["insider_name"]: o for o in sec_feed["observations"]}

        # 1. Verify Scott Gawel (Tax Withholding)
        gawel = obs_by_name["GAWEL SCOTT"]
        assert gawel["transaction_type"] == "Tax Withholding", f"Expected Tax Withholding, got {gawel['transaction_type']}"
        assert gawel["transaction_code"] == "F"
        assert gawel["signal_type"] == "TAX_WITHHOLDING"
        assert gawel["is_sell"] is False, "Code F tax withholding must not be marked is_sell=True"
        assert gawel["is_buy"] is False
        assert gawel["total_value_usd"] == 622082.44

        # 2. Verify Tench Coxe (Gift)
        coxe = obs_by_name["COXE TENCH"]
        assert coxe["transaction_type"] == "Gift", f"Expected Gift, got {coxe['transaction_type']}"
        assert coxe["transaction_code"] == "G"
        assert coxe["signal_type"] == "GIFT"
        assert coxe["is_sell"] is False, "Code G gift must not be marked is_sell=True"

        # 3. Verify Timothy Teter (Open Market Sale)
        teter = obs_by_name["Teter Timothy S."]
        assert teter["transaction_type"] == "Sale (Direct)", f"Expected Sale (Direct), got {teter['transaction_type']}"
        assert teter["transaction_code"] == "S"
        assert teter["signal_type"] == "OPEN_MARKET_SALE"
        assert teter["is_sell"] is True, "Code S open-market sale must be marked is_sell=True"
        assert teter["total_value_usd"] == 2773637.72

        # 4. Confluence Inputs Synthesis: Only 1 discretionary sell (Teter), strictly excluding Gawel & Coxe
        conf_inputs = res.get("confluence_inputs")
        assert conf_inputs is not None
        assert conf_inputs["insider_sell_count_90d"] == 1, f"Expected 1 insider sell, got {conf_inputs['insider_sell_count_90d']}"
        assert conf_inputs["insider_buy_count_90d"] == 0

        # Direction remains BEARISH due to Teter's $2.77M open-market sale
        assert sec_feed["direction"] == "BEARISH"

    def test_standalone_tax_withholding_produces_neutral_direction(self, pg_db, monkeypatch):
        """Verifies that an issuer with ONLY statutory tax withholding is NEUTRAL, not falsely BEARISH."""
        uri = pg_db["uri"]
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", uri)
        monkeypatch.setenv("ARX_ENVIRONMENT", "rehearsal")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = psycopg2.connect(uri)
        with conn.cursor() as cur:
            acc = "0001243821-26-000099"
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date, accepted_datetime,
                    xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (acc, "0001045810", "AAPL", "APPLE INC", today, now_iso, "<xml/>", "h_aapl", "https://sec.gov/aapl"))
            cur.execute("""
                INSERT INTO sm_source_availability_ledger (accession_number, public_availability_at, source)
                VALUES (%s, %s, %s);
            """, (acc, now_iso, "SEC_INDEX"))
            cur.execute("""
                INSERT INTO form4_parsed_transactions (
                    transaction_id, accession_number, transaction_date, transaction_code,
                    shares, price_per_share, total_value_usd, reporting_owner_name,
                    reporting_owner_role, is_director, is_officer, is_ten_percent,
                    officer_title, ownership_type, is_derivative, signal_type,
                    acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                ) VALUES (%s, %s, %s, 'F', 5000, 230.0, 1150000.0, 'Tim Cook', 'Officer', false, true, false, 'CEO', 'D', false, 'TAX_WITHHOLDING', 'D', false, false);
            """, (f"{acc}:0001243821:NON_DERIVATIVE:0", acc, today))
        conn.commit()
        conn.close()

        engine = SmartMoneyEngine()
        res = engine.evaluate_smart_money_feeds("AAPL", window_days=90)

        assert "feeds" in res
        sec_feed = res["feeds"]["sec_form4"]
        assert sec_feed["status"] == "VERIFIED"
        assert sec_feed["observation_count"] == 1

        obs = sec_feed["observations"][0]
        assert obs["transaction_type"] == "Tax Withholding"
        assert obs["is_sell"] is False
        assert obs["is_buy"] is False

        # Direction MUST be NEUTRAL (no buys, and statutory tax withholding is not discretionary selling)
        assert sec_feed["direction"] == "NEUTRAL", f"Expected NEUTRAL, got {sec_feed['direction']}"

        conf_inputs = res.get("confluence_inputs")
        assert conf_inputs is not None
        assert conf_inputs["insider_sell_count_90d"] == 0
        assert conf_inputs["insider_buy_count_90d"] == 0

