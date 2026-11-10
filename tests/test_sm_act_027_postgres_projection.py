"""SM-ACT-027 real PostgreSQL SQL/ledger behavioral integration.

Uses only an isolated ephemeral PostgreSQL instance supplied by CI.
No external SEC, Railway, R2 or production credentials.
"""
import os
import sys
import types

import pytest

from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine

pytestmark = pytest.mark.skipif(
    not os.environ.get("SM_ACT_027_POSTGRES_TEST_DSN"),
    reason="Requires isolated PostgreSQL CI instance; never use production database",
)


@pytest.fixture
def pg_fixture(monkeypatch):
    import psycopg2
    from psycopg2.extras import RealDictCursor

    dsn = os.environ["SM_ACT_027_POSTGRES_TEST_DSN"]
    conn = psycopg2.connect(dsn)
    schema = "sm_act_027_fixture"
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS " + schema)
            cur.execute("SET search_path TO " + schema)
            for table in ("sm_amendments", "sm_source_availability_ledger", "form4_parsed_transactions", "form4_raw_filings"):
                cur.execute("DROP TABLE IF EXISTS " + table)
            cur.execute("""
                CREATE TABLE form4_raw_filings (
                    accession_number TEXT PRIMARY KEY, cik TEXT, ticker TEXT,
                    filing_date DATE, created_at TIMESTAMPTZ, accepted_datetime TIMESTAMPTZ
                )
            """)
            cur.execute("""
                CREATE TABLE form4_parsed_transactions (
                    transaction_id TEXT, accession_number TEXT, reporting_owner_name TEXT,
                    officer_title TEXT, is_director BOOLEAN, is_officer BOOLEAN,
                    is_ten_percent BOOLEAN, transaction_date DATE, transaction_code TEXT,
                    shares NUMERIC, price_per_share NUMERIC, total_value_usd NUMERIC,
                    ownership_type TEXT, is_derivative BOOLEAN, signal_type TEXT,
                    acquired_disposed TEXT, is_positive_buying_conviction BOOLEAN,
                    is_open_market_purchase_evidence BOOLEAN
                )
            """)
            cur.execute("""
                CREATE TABLE sm_source_availability_ledger (
                    accession_number TEXT, public_availability_at TIMESTAMPTZ,
                    recorded_at TIMESTAMPTZ
                )
            """)
            cur.execute("""
                CREATE TABLE sm_amendments (
                    amendment_accession TEXT, base_accession TEXT,
                    resolution_status TEXT, amendment_number INTEGER
                )
            """)
        conn.commit()
        config_mod = types.ModuleType("analyst_dashboard.smart_money.storage.config")
        config_mod.get_storage_backend = lambda: "postgres"
        conn_mod = types.ModuleType("analyst_dashboard.smart_money.storage.postgres_connection")

        class IsolatedConnection:
            def __enter__(self):
                with conn.cursor() as cur:
                    cur.execute("SET search_path TO " + schema)
                return conn

            def __exit__(self, *args):
                conn.rollback() if args[0] is not None else conn.commit()
                return False

            def cursor(self, *args, **kwargs):
                return conn.cursor(*args, **kwargs)

        conn_mod.get_connection = lambda: IsolatedConnection()
        monkeypatch.setitem(sys.modules, config_mod.__name__, config_mod)
        monkeypatch.setitem(sys.modules, conn_mod.__name__, conn_mod)
        yield conn
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS " + schema + " CASCADE")
        conn.commit()
        conn.close()


def _insert_filing(conn, *, filing_date="2026-10-01", public_at="2026-10-02T21:13:29+00:00", observed_at="2026-10-03T10:20:00+00:00"):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sm_act_027_fixture.form4_raw_filings
            VALUES ('0000000000-26-000001', '0001045810', 'NVDA', %s, %s, %s)
        """, (filing_date, observed_at, public_at))
        cur.execute("""
            INSERT INTO sm_act_027_fixture.form4_parsed_transactions VALUES (
                '0000000000-26-000001:0001045810:NON_DERIVATIVE:0',
                '0000000000-26-000001', 'Fixture Reporting Owner', 'Officer',
                FALSE, TRUE, FALSE, %s, 'F',
                10, 100, 1000, 'D', FALSE, 'TAX_WITHHOLDING',
                'D', FALSE, FALSE)
        """, (filing_date,))
        cur.execute("""
            INSERT INTO sm_act_027_fixture.sm_source_availability_ledger
            VALUES ('0000000000-26-000001', %s, %s)
        """, (public_at, observed_at))
    conn.commit()


def _assess(*, watermark="2026-11-10T12:00:00+00:00", replay_mode="PUBLIC_AVAILABLE"):
    return SmartMoneyEngine.evaluate_smart_money_feeds(
        "NVDA", window_days=90, include_curated=False,
        as_of=watermark, replay_mode=replay_mode,
    )


def test_postgres_real_query_preserves_distinct_timestamps_and_withholding(pg_fixture):
    _insert_filing(pg_fixture)
    sec = _assess()["feeds"]["sec_form4"]
    assert sec["status"] == "VERIFIED"
    assert sec["observation_count"] == 1
    record = sec["observations"][0]
    assert record["transaction_type"] == "Tax Withholding"
    assert not record["is_positive_buying_conviction"]
    assert str(record["arx_first_observed_at"]).startswith("2026-10-03 10:20:00")
    assert str(record["sec_accepted_at"]).startswith("2026-10-02 21:13:29")
    assert record["ingestion_timestamp"] == record["arx_first_observed_at"]
    assert sec["latest_successful_ingestion"] == record["arx_first_observed_at"]
    assert sec["latest_successful_sec_source_check"] is None


def test_postgres_public_available_precedes_arx_observation(pg_fixture):
    _insert_filing(pg_fixture)
    t = "2026-10-02T23:00:00+00:00"
    public = _assess(watermark=t, replay_mode="PUBLIC_AVAILABLE")
    observed = _assess(watermark=t, replay_mode="ARX_OBSERVED")
    assert public["feeds"]["sec_form4"]["status"] == "VERIFIED"
    assert public["feeds"]["sec_form4"]["observation_count"] == 1
    assert observed["feeds"]["sec_form4"]["status"] == "PIPELINE_PENDING"
    assert observed["feeds"]["sec_form4"]["observation_count"] == 0
    assert observed["confluence_inputs"] is None


def test_postgres_old_records_do_not_certify_current_zero(pg_fixture):
    _insert_filing(pg_fixture, filing_date="2025-01-15", public_at="2025-01-16T09:00:00+00:00", observed_at="2025-01-17T09:00:00+00:00")
    sec = _assess()["feeds"]["sec_form4"]
    assert sec["status"] == "PIPELINE_PENDING"
    assert sec["direction"] == "UNVERIFIED"
    assert sec["confidence"] == "NONE"
    assert sec["observation_count"] == 0


def test_public_assessment_postgres_pit_modes_remain_separate_from_live_hold(pg_fixture):
    """The public-assessment wrapper must invoke the real PostgreSQL PIT ledger."""
    _insert_filing(pg_fixture)
    cutoff = "2026-10-02T23:00:00+00:00"
    public = SmartMoneyEngine.assess_symbol("NVDA", as_of=cutoff, replay_mode="PUBLIC_AVAILABLE")
    observed = SmartMoneyEngine.assess_symbol("NVDA", as_of=cutoff, replay_mode="ARX_OBSERVED")
    assert public["pit_evidence"]["feeds"]["sec_form4"]["status"] == "VERIFIED"
    assert public["pit_evidence"]["feeds"]["sec_form4"]["observation_count"] == 1
    assert observed["pit_evidence"]["feeds"]["sec_form4"]["status"] == "PIPELINE_PENDING"
    assert observed["pit_evidence"]["feeds"]["sec_form4"]["observation_count"] == 0
    assert public["overall_status"] == observed["overall_status"] == "PIPELINE_HELD"
    assert public["active_holds"] == observed["active_holds"] == ["HOLD_SEC_001"]
    assert public["latest_successful_ingestion"] is None
    assert observed["latest_successful_ingestion"] is None
