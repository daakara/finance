"""
ARX Terminal — Directive SM-ACT-004 Deterministic Defect & Architecture Suite

Validates:
1. Worker standby mode without crash when ARX_SEC_EGRESS_ROLE != 'OWNER'
2. Table II derivative purchase exclusion (Code P derivatives do not trigger bullish conviction)
3. Form 4/A amendment deduplication (superseded filings excluded from feed observations)
4. Dual-backend storage resolution parity (PostgreSQL query path and SQLite fallback)
"""

import os
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.worker import (
    SmartMoneyIngestionWorker,
    WorkerHealthState,
    get_sec_egress_role,
)
from analyst_dashboard.smart_money.storage.config import get_storage_backend
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine


class TestWorkerStandbyMode:
    """Verifies Worker avoids crashing in rehearsal or non-production environments."""

    @pytest.mark.parametrize("role", ["DISABLED", "REVOKED", "ANALYSIS_ONLY", "STANDBY", ""])
    def test_worker_standby_preflight_does_not_raise(self, role, monkeypatch):
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", role)
        worker = SmartMoneyIngestionWorker()

        # Must return False rather than raising RuntimeError("UNAUTHORIZED_WORKER_ROLE")
        is_owner = worker.validate_preflight()
        assert is_owner is False
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
