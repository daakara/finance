"""Archived-byte replay and expansion blockers; no live SEC data acquisition."""
import hashlib
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from analyst_dashboard.smart_money.continuous_pipeline import (
    SmartMoneyContinuousPipeline, DEFAULT_GOVERNED_ISSUERS, TRIAL_2_GOVERNED_ISSUERS,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository

ACCESSION = "0001140361-26-044773"
FIXTURE = Path(__file__).parent / "fixtures/sec_edgar/AAPL_0001140361-26-044773.xml"
DIGEST = "eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9"


@pytest.fixture(autouse=True)
def require_offline_guard(pytestconfig):
    if not pytestconfig.pluginmanager.hasplugin("sec_offline_test_support"):
        pytest.skip("Requires -p sec_offline_test_support for loopback-only replay")


def archived_filing():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DIGEST
    return dict(accession_number=ACCESSION, ticker="AAPL", cik="0000320193",
                company_name="APPLE INC", filing_date="2026-11-18",
                document_type="4", accepted_datetime="2026-11-18T23:30:10Z",
                xml_content=raw.decode(), content_hash=DIGEST,
                source_url="archive://tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml")


@pytest.fixture
def replay_db(tmp_path):
    import pgserver
    import psycopg2
    from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
    server = pgserver.get_server(tmp_path / "postgres")
    conn = psycopg2.connect(server.get_uri())
    initialize_postgres_schema(conn)
    repo = PostgresSmartMoneyRepository(dsn=server.get_uri())
    pipeline = SmartMoneyContinuousPipeline(
        dsn=server.get_uri(), repository=repo, issuers=TRIAL_2_GOVERNED_ISSUERS,
        ingestion_only=True, allow_test_mode=True,
    )
    try:
        yield conn, repo, pipeline
    finally:
        conn.close()
        server.cleanup()


def table_counts(conn):
    with conn.cursor() as cur:
        result = []
        for table in ("form4_raw_filings", "form4_parsed_transactions",
                      "sm_source_availability_ledger", "scanner_snapshots", "scanner_active_snapshots"):
            cur.execute(f"SELECT count(*) FROM {table}")
            result.append(cur.fetchone()[0])
        return tuple(result)


def test_archived_fixture_identity_and_transaction_codes():
    item = archived_filing()
    parsed = Form4XmlParser.parse_filing(item["xml_content"], ACCESSION)
    assert (parsed.issuer_symbol, parsed.issuer_cik) == ("AAPL", "0000320193")
    assert [tx.transaction_code for tx in parsed.transactions] == ["S", "S", "G"]


def test_real_archived_replay_is_idempotent_and_partial(replay_db):
    conn, repo, pipeline = replay_db
    _, receipt = pipeline._step7_parse_and_reconcile([archived_filing()], "offline-archive-1")
    assert receipt["coverage_status"] == "PARTIAL"
    assert table_counts(conn) == (1, 3, 1, 0, 0)
    pipeline._step7_parse_and_reconcile([archived_filing()], "offline-archive-2")
    assert table_counts(conn) == (1, 3, 1, 0, 0)
    with conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM form4_raw_filings WHERE accession_number=%s", (ACCESSION,))
        assert cur.fetchone()[0] == DIGEST


def test_archived_batch_rolls_back_on_evidence_failure(replay_db):
    conn, repo, pipeline = replay_db
    with patch.object(repo, "save_availability", side_effect=RuntimeError("injected evidence failure")):
        with pytest.raises(RuntimeError, match="injected evidence failure"):
            pipeline._step7_parse_and_reconcile([archived_filing()], "offline-archive-rollback")
    assert table_counts(conn) == (0, 0, 0, 0, 0)


def test_current_ten_issuer_ingestion_only_budget_covers_catalogs(monkeypatch):
    monkeypatch.setenv("ARX_CONTINUOUS_AUTH_SCOPE", "DAILY_10_ISSUER_FORM4")
    pipeline = SmartMoneyContinuousPipeline(
        repository=MagicMock(), issuers=DEFAULT_GOVERNED_ISSUERS,
        ingestion_only=True, allow_test_mode=True,
    )
    assert not pipeline.is_trial_2_issuer
    assert len(pipeline.issuers) == 10
    assert pipeline.outbound_session.max_total_requests == 40
    assert pipeline.outbound_session.max_total_requests >= len(pipeline.issuers)


def test_offline_transport_guard_refuses_external_addresses():
    with socket.socket() as sock:
        with pytest.raises(RuntimeError, match="OFFLINE_REPLAY_EXTERNAL_EGRESS_REFUSED"):
            sock.connect(("203.0.113.1", 443))
