"""Exercise real SmartMoneyEngine evaluation against a disposable SQLite fixture.

No production DB, API, SEC, or network access. Public availability and ARX
observation timestamps are intentionally different.
"""
import sqlite3
import sys
import types

from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine


def _fixture(tmp_path, monkeypatch, filing_date, observation_date):
    path = tmp_path / "isolated_form4.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE form4_parsed_transactions (
                record_key TEXT, accession_number TEXT, issuer_cik TEXT,
                issuer_symbol TEXT, reporting_owner_name TEXT, officer_title TEXT,
                is_director INTEGER, is_officer INTEGER, is_ten_percent_owner INTEGER,
                filing_date TEXT, transaction_date TEXT, transaction_code TEXT,
                signal_type TEXT, shares REAL, price_per_share REAL, total_value_usd REAL,
                acquired_disposed TEXT, is_positive_buying_conviction INTEGER,
                is_open_market_purchase_evidence INTEGER, acceptance_datetime TEXT,
                created_at TEXT, ownership_nature TEXT
            )
        """)
        conn.execute("""
            INSERT INTO form4_parsed_transactions VALUES
            ('NVDA-fixture-1', '0000000000-26-000001', '0001045810',
             'NVDA', 'Fixture Reporting Owner', 'Officer',
             0, 1, 0, ?, ?, 'F', 'TAX_WITHHOLDING', 10, 100, 1000,
             'D', 0, 0, ?, ?, 'D')
        """, (filing_date, filing_date, "2026-10-02T21:13:29Z", observation_date))
    config_mod = types.ModuleType("analyst_dashboard.smart_money.storage.config")
    config_mod.get_storage_backend = lambda: "sqlite"
    ingest_mod = types.ModuleType("analyst_dashboard.smart_money.form4_ingestion")
    ingest_mod.connect_smart_money_db = lambda: sqlite3.connect(path)
    monkeypatch.setitem(sys.modules, config_mod.__name__, config_mod)
    monkeypatch.setitem(sys.modules, ingest_mod.__name__, ingest_mod)


def _evaluate(*, as_of="2026-11-10T12:00:00Z", replay_mode="ARX_OBSERVED"):
    return SmartMoneyEngine.evaluate_smart_money_feeds(
        symbol="NVDA", window_days=90, include_curated=False,
        as_of=as_of, replay_mode=replay_mode,
    )


def test_historical_only_filing_cannot_certify_zero_recent_filings(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, "2025-01-15", "2025-01-16T12:00:00Z")
    result = _evaluate()
    sec = result["feeds"]["sec_form4"]
    assert sec["status"] == "PIPELINE_PENDING"
    assert sec["direction"] == "UNVERIFIED"
    assert sec["confidence"] == "NONE"
    assert sec["observation_count"] == 0
    assert result["overall_status"] == "PIPELINE_PENDING"
    assert result["confluence_inputs"] is None


def test_sqlite_real_engine_retains_observation_not_sec_acceptance(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, "2026-10-01", "2026-10-03T10:20:00Z")
    result = _evaluate()
    sec = result["feeds"]["sec_form4"]
    assert sec["status"] == "VERIFIED"
    assert sec["observation_count"] == 1
    record = sec["observations"][0]
    assert record["transaction_type"] == "Tax Withholding"
    assert record["is_positive_buying_conviction"] is False
    assert record["arx_first_observed_at"] == "2026-10-03T10:20:00Z"
    assert record["ingestion_timestamp"] == "2026-10-03T10:20:00Z"
    assert record["acceptance_datetime"] == "2026-10-02T21:13:29Z"
    assert sec["latest_successful_ingestion"] == "2026-10-03T10:20:00Z"
    assert sec["latest_successful_sec_source_check"] is None


def test_publicly_available_does_not_mean_already_observed_by_arx(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, "2026-10-01", "2026-10-03T10:20:00Z")
    cutoff = "2026-10-02T23:00:00Z"
    public = _evaluate(as_of=cutoff, replay_mode="PUBLIC_AVAILABLE")
    observed = _evaluate(as_of=cutoff, replay_mode="ARX_OBSERVED")
    assert public["feeds"]["sec_form4"]["status"] == "VERIFIED"
    assert public["feeds"]["sec_form4"]["observation_count"] == 1
    assert observed["feeds"]["sec_form4"]["status"] == "PIPELINE_PENDING"
    assert observed["feeds"]["sec_form4"]["observation_count"] == 0
    assert observed["overall_status"] == "PIPELINE_PENDING"
    assert observed["confluence_inputs"] is None
