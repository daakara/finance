"""Offline continuous ingestion acceptance, including real isolated PG rollback."""
import hashlib
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository

BASE = '0002152188-26-000005'
AMEND = '0002152188-26-000007'


def filing(accession=BASE, form='4', owner='2152188', original='', date='2026-09-11'):
    xml = f'''<ownershipDocument><documentType>{form}</documentType>
      <periodOfReport>2026-09-10</periodOfReport>
      <dateOfOriginalSubmission>{original}</dateOfOriginalSubmission>
      <issuer><issuerCik>1045810</issuerCik><issuerName>NVIDIA</issuerName>
        <issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>{owner}</rptOwnerCik>
        <rptOwnerName>Reporting Owner</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable><nonDerivativeTransaction>
        <transactionDate><value>2026-09-10</value></transactionDate>
        <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
        <transactionAmounts><transactionShares><value>10</value></transactionShares>
        <transactionPricePerShare><value>100</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
        </transactionAmounts></nonDerivativeTransaction></nonDerivativeTable>
      </ownershipDocument>'''
    return dict(accession_number=accession, ticker='NVDA', cik='0001045810',
                company_name='NVIDIA', filing_date=date, document_type=form,
                accepted_datetime=f'{date}T21:04:47Z', xml_content=xml,
                source_url=f'https://www.sec.gov/Archives/{accession}.xml',
                content_hash=hashlib.sha256(xml.encode()).hexdigest())


@pytest.fixture
def db(tmp_path, monkeypatch):
    pgserver = pytest.importorskip('pgserver')
    psycopg2 = pytest.importorskip('psycopg2')
    from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
    from analyst_dashboard.smart_money.storage import postgres_repository as module
    try:
        server = pgserver.get_server(tmp_path / 'pg')
    except PermissionError:
        if os.getenv('ARX_REQUIRE_LOCAL_POSTGRES_TESTS') == 'true':
            raise
        pytest.skip('Runtime denies local sockets; required PostgreSQL acceptance runs in CI')
    conn = psycopg2.connect(server.get_uri())
    initialize_postgres_schema(conn)

    @contextmanager
    def connection(**kwargs):
        yield conn
    monkeypatch.setattr(module, 'get_connection', connection)
    repo = PostgresSmartMoneyRepository(dsn=server.get_uri())
    pipeline = SmartMoneyContinuousPipeline(repository=repo, allow_test_mode=True)
    try:
        yield conn, repo, pipeline
    finally:
        conn.close()
        server.cleanup()


def count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM {table}')
        return cur.fetchone()[0]


def test_parser_preserves_original_submission_date_and_xml_owner_identity():
    item = filing(AMEND, '4/A', original='2026-09-11')
    parsed = Form4XmlParser.parse_filing(item['xml_content'], AMEND)
    assert parsed.original_submission_date == '2026-09-11'
    assert parsed.reporting_owner_ciks == ['0002152188']
    assert parsed.filing_date == '2026-09-10'  # report period is NOT original filing date


def test_production_injection_cannot_restore_latest_only_fetch_path():
    pipeline = SmartMoneyContinuousPipeline(repository=MagicMock(), allow_test_mode=True)
    pipeline.allow_test_mode = False
    pipeline.issuers = pipeline.issuers[:1]
    pipeline.sec_fetcher = MagicMock()
    item = filing()
    with patch.object(pipeline, '_check_in_flight_authorization'), patch.object(
        pipeline, '_fetch_filings_from_sec', return_value=[item]
    ) as guarded:
        assert pipeline._step6_ingest_form4_filings() == [item]
    guarded.assert_called_once()
    pipeline.sec_fetcher.fetch_form4_filings.assert_not_called()
    pipeline.repository.save_raw_filing.assert_not_called()


def test_atomic_ingestion_resolves_amendment_first_and_is_idempotent(db):
    conn, repo, pipeline = db
    inputs = [filing(AMEND, '4/A', original='2026-09-11', date='2026-10-13'), filing()]
    txs, _ = pipeline._step7_parse_and_reconcile(inputs, 'cycle1')
    assert len(txs) == 2
    assert count(conn, 'form4_raw_filings') == 2
    assert count(conn, 'sm_source_availability_ledger') == 2
    assert count(conn, 'form4_parsed_transactions') == 2
    with conn.cursor() as cur:
        cur.execute('SELECT base_accession, resolution_status FROM sm_amendments')
        assert cur.fetchone() == (BASE, 'RESOLVED')
        cur.execute('SELECT public_availability_at FROM sm_source_availability_ledger WHERE accession_number=%s', (BASE,))
        assert cur.fetchone()[0].isoformat() == '2026-09-11T21:04:47+00:00'
    pipeline._step7_parse_and_reconcile(inputs, 'cycle2')
    assert count(conn, 'form4_raw_filings') == 2
    assert count(conn, 'sm_source_availability_ledger') == 2
    assert count(conn, 'form4_parsed_transactions') == 2
    assert count(conn, 'sm_amendments') == 1
    assert count(conn, 'scanner_active_snapshots') == 0


@pytest.mark.parametrize('fault', ['parse', 'availability', 'amendment', 'receipt'])
def test_failed_evidence_write_rolls_back_entire_batch(db, fault):
    conn, repo, pipeline = db
    item = filing(AMEND, '4/A', original='2026-09-11')
    if fault == 'parse':
        item['xml_content'] = '<ownershipDocument>broken'
        item['content_hash'] = hashlib.sha256(item['xml_content'].encode()).hexdigest()
        context = patch.object(repo, 'save_reconciliation_receipt', wraps=repo.save_reconciliation_receipt)
    else:
        method = {'availability': 'save_availability', 'amendment': 'save_amendment',
                  'receipt': 'save_reconciliation_receipt'}[fault]
        context = patch.object(repo, method, side_effect=RuntimeError('injected'))
    with context, pytest.raises(RuntimeError):
        pipeline._step7_parse_and_reconcile([filing(), item], 'fault')
    for table in ['form4_raw_filings', 'sm_source_availability_ledger', 'sm_amendments',
                  'form4_parsed_transactions', 'sm_reconciliation_receipts']:
        assert count(conn, table) == 0
    assert repo.get_existing_accessions_for_issuer('NVDA') == set()


def test_ambiguous_original_stays_unresolved_then_missing_base_recovers(db):
    conn, repo, pipeline = db
    amendment = filing(AMEND, '4/A', original='2026-09-11', date='2026-10-13')
    txs, receipt = pipeline._step7_parse_and_reconcile([amendment], 'no-base')
    assert receipt['coverage_status'] == 'PARTIAL'
    with pytest.raises(RuntimeError, match='Reconciliation evidence incomplete'):
        pipeline._step8_publish_snapshot(txs, receipt, 'unresolved')
    assert AMEND not in repo.get_existing_accessions_for_issuer('NVDA')
    pipeline._step7_parse_and_reconcile([amendment, filing()], 'base-arrived')
    assert AMEND in repo.get_existing_accessions_for_issuer('NVDA')
    # Distinct accession with same XML owner and date is ambiguous, never guess.
    duplicate = filing('0009999999-26-000001')
    pipeline._step7_parse_and_reconcile([duplicate], 'ambiguous')
    with repo.ingestion_transaction() as c:
        assert repo.resolve_amendment_base(amendment, '2026-09-11', ['0002152188'], c) is None
        assert repo.resolve_amendment_base(amendment, '2026-09-11', ['0000000123'], c) is None


def test_repository_transaction_propagates_failure_and_rolls_back():
    repo = PostgresSmartMoneyRepository(dsn='offline')
    conn = MagicMock()
    @contextmanager
    def connection(**kwargs):
        yield conn
    with patch('analyst_dashboard.smart_money.storage.postgres_repository.get_connection', connection):
        with pytest.raises(RuntimeError, match='write failed'):
            with repo.ingestion_transaction() as shared:
                assert shared is conn
                raise RuntimeError('write failed')
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_reconciliation_uses_one_shared_connection_and_propagates_lineage_failure():
    repo = MagicMock(spec=PostgresSmartMoneyRepository)
    conn = repo.ingestion_transaction.return_value.__enter__.return_value
    repo.resolve_amendment_base.return_value = BASE
    pipeline = SmartMoneyContinuousPipeline(repository=repo, allow_test_mode=True)
    txs, _ = pipeline._step7_parse_and_reconcile(
        [filing(AMEND, '4/A', original='2026-09-11'), filing()], 'offline'
    )
    assert len(txs) == 2
    for method in [repo.save_raw_filing, repo.save_availability,
                   repo.save_amendment, repo.save_parsed_transactions, repo.save_reconciliation_receipt]:
        assert method.called
        assert all(c.kwargs['conn'] is conn for c in method.call_args_list)
    assert txs[0].public_availability_at == '2026-09-11T21:04:47Z'
    repo.save_amendment.side_effect = RuntimeError('lineage write failed')
    with pytest.raises(RuntimeError, match='lineage write failed'):
        pipeline._step7_parse_and_reconcile([filing(AMEND, '4/A')], 'fail')
    assert repo.ingestion_transaction.return_value.__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize('xml', ['not xml', '<ownershipDocument>broken'])
def test_invalid_payload_fails_before_reconciliation_receipt(xml):
    repo = MagicMock(spec=PostgresSmartMoneyRepository)
    pipeline = SmartMoneyContinuousPipeline(repository=repo, allow_test_mode=True)
    item = filing()
    item['xml_content'] = xml
    with pytest.raises(RuntimeError, match='SEC_FORM4_PARSE_FAILED'):
        pipeline._step7_parse_and_reconcile([item], 'invalid')
    repo.save_reconciliation_receipt.assert_not_called()


def test_unresolved_amendment_cannot_publish_complete_snapshot():
    repo = MagicMock(spec=PostgresSmartMoneyRepository)
    repo.resolve_amendment_base.return_value = None
    pipeline = SmartMoneyContinuousPipeline(repository=repo, allow_test_mode=True)
    txs, receipt = pipeline._step7_parse_and_reconcile([filing(AMEND, '4/A')], 'unknown-base')
    assert receipt['material_gap_count'] == 1
    assert receipt['coverage_status'] == 'PARTIAL'
    with pytest.raises(RuntimeError, match='Reconciliation evidence incomplete'):
        pipeline._step8_publish_snapshot(txs, receipt, 'unknown-base')
    repo.publish_scanner_snapshot_atomic.assert_not_called()
