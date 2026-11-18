"""Offline continuous ingestion acceptance, including real isolated PG rollback."""
import hashlib
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.amendment_classifier import (
    classify_amendment_effect,
    SCOPE_FULL_REPLACEMENT,
    SCOPE_PARTIAL_CORRECTION,
    SCOPE_ADDITIONAL_TRANSACTION,
    SCOPE_AMBIGUOUS,
    SCOPE_UNRESOLVED,
)
from analyst_dashboard.smart_money.radar_adapter import RadarSmartMoneyAdapter

BASE = '0002152188-26-000005'
AMEND = '0002152188-26-000007'


def filing(accession=BASE, form='4', owner='2152188', original='', date='2026-09-11', remarks='', footnotes=None):
    remarks_xml = f"<remarks>{remarks}</remarks>" if remarks else ""
    fn_xml = ""
    if footnotes:
        fn_nodes = "".join([f'<footnote id="F{i}">{fn}</footnote>' for i, fn in enumerate(footnotes, 1)])
        fn_xml = f"<footnotes>{fn_nodes}</footnotes>"
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
      {remarks_xml}
      {fn_xml}
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
    inputs = [filing(AMEND, '4/A', original='2026-09-11', date='2026-10-13', remarks='This amendment is filed solely to correct the number of shares.'), filing()]
    txs, _ = pipeline._step7_parse_and_reconcile(inputs, 'cycle1')
    assert len(txs) == 2
    assert count(conn, 'form4_raw_filings') == 2
    assert count(conn, 'sm_source_availability_ledger') == 2
    assert count(conn, 'form4_parsed_transactions') == 2
    with conn.cursor() as cur:
        cur.execute('SELECT base_accession, resolution_status FROM sm_amendments')
        assert cur.fetchone() == (BASE, 'PARTIAL_CORRECTION')
        cur.execute('SELECT public_availability_at FROM sm_source_availability_ledger WHERE accession_number=%s', (BASE,))
        assert cur.fetchone()[0].astimezone(timezone.utc).isoformat() == '2026-09-11T21:04:47+00:00'
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


def historical_row(accession=BASE, public_at='2026-11-16T18:00:00Z', tx_date='2026-11-16'):
    return dict(transaction_id=f'{accession}:0002152188:NON_DERIVATIVE:0',
                accession_number=accession, issuer_cik='0001045810', symbol='NVDA',
                company_name='NVIDIA', reporting_owner_name='Reporting Owner',
                transaction_date=tx_date, public_availability_at=public_at,
                transaction_code='P', shares=1000, price_per_share=100, total_value_usd=100000,
                acquired_disposed='A', is_derivative=False, is_open_market_purchase_evidence=True,
                ownership_type='D', is_officer=True, officer_title='CEO')


def live_snapshot_pipeline(repo):
    pipeline = SmartMoneyContinuousPipeline(repository=repo, allow_test_mode=True)
    pipeline.allow_test_mode = False
    pipeline.issuers = pipeline.issuers[:1]
    pipeline._check_in_flight_authorization = MagicMock()
    return pipeline


def test_live_snapshot_uses_older_persisted_evidence_not_current_batch():
    from datetime import date, datetime, timezone
    repo = MagicMock(spec=PostgresSmartMoneyRepository)
    row = historical_row(tx_date=date(2026, 11, 16))
    row['public_availability_at'] = datetime(2026, 11, 16, 18, tzinfo=timezone.utc)
    repo.get_snapshot_evidence_as_of.return_value = {
        'transactions': {'NVDA': [row]}, 'coverage_receipt_ids': ['coverage-nvda-30d'],
    }
    pipeline = live_snapshot_pipeline(repo)
    # New batch has no buying; persistent historical evidence still qualifies.
    snapshot = pipeline._step8_publish_snapshot([], {'reconciliation_hash': 'hash'}, 'historical',
                                               as_of='2026-11-17T12:00:00Z')
    assert snapshot['matched_count'] == 1
    assert snapshot['results'][0]['scanner_evidence']['grossPurchaseValueUsd'] == 100000
    repo.get_snapshot_evidence_as_of.assert_called_once_with(
        '2026-11-17T12:00:00Z', pipeline.issuers, '2026-10-18',
    )
    assert snapshot['provenance']['historical_input']['source'] == 'POSTGRES_PUBLIC_AVAILABLE'
    assert snapshot['provenance']['historical_input']['coverage_receipt_ids'] == ['coverage-nvda-30d']
    assert snapshot['data_as_of'] == '2026-11-17T12:00:00Z'


@pytest.mark.parametrize('changes', [
    {'public_availability_at': '2026-11-18T00:00:00Z'},
    {'public_availability_at': '2026-11-16T18:00:00'},
    {'transaction_date': '2026-11-18'},
    {'issuer_cik': '0000000001'},
    {'transaction_id': 'unverified-record-key'},
])
def test_live_snapshot_rejects_ineligible_persisted_evidence(changes):
    repo = MagicMock(spec=PostgresSmartMoneyRepository)
    row = historical_row()
    row.update(changes)
    repo.get_snapshot_evidence_as_of.return_value = {'transactions': {'NVDA': [row]}, 'coverage_receipt_ids': ['coverage']}
    pipeline = live_snapshot_pipeline(repo)
    with pytest.raises(RuntimeError, match='SNAPSHOT_'):
        pipeline._step8_publish_snapshot([], {'reconciliation_hash': 'hash'}, 'invalid', as_of='2026-11-17')
    repo.publish_scanner_snapshot_atomic.assert_not_called()


@pytest.mark.parametrize('as_of, expected', [
    ('2026-11-17', '2026-11-17T23:59:59+00:00'),
    ('2026-11-17T17:00:00-05:00', '2026-11-17T22:00:00+00:00'),
])
def test_snapshot_cutoff_preserves_intraday_precision_and_date_only_contract(as_of, expected):
    assert SmartMoneyContinuousPipeline._snapshot_cutoff(as_of).isoformat() == expected


@pytest.mark.parametrize('as_of', ['2026-11-17T17:00:00', 'invalid', '2026-99-99'])
def test_invalid_snapshot_cutoff_never_falls_back_to_now(as_of):
    with pytest.raises(ValueError):
        SmartMoneyContinuousPipeline._snapshot_cutoff(as_of)


def test_read_failure_cannot_fall_back_to_current_batch():
    repo = MagicMock(spec=PostgresSmartMoneyRepository)
    repo.get_snapshot_evidence_as_of.side_effect = RuntimeError('database unavailable')
    pipeline = live_snapshot_pipeline(repo)
    with pytest.raises(RuntimeError, match='database unavailable'):
        pipeline._step8_publish_snapshot([historical_row()], {'reconciliation_hash': 'hash'}, 'offline', as_of='2026-11-17')
    repo.publish_scanner_snapshot_atomic.assert_not_called()


def test_missing_issuer_coverage_blocks_even_empty_historical_snapshot(db):
    conn, repo, pipeline = db
    with pytest.raises(RuntimeError, match='SNAPSHOT_HISTORICAL_COVERAGE_UNVERIFIED'):
        repo.get_snapshot_evidence_as_of('2026-11-17T12:00:00Z', pipeline.issuers[:1], '2026-10-18')
    assert conn.status == 1  # read-only transaction ended; no pooled transaction leak


def pit_filing(accession=BASE, form='4', public_date='2026-09-21', original='', filing_date=None):
    d = filing_date or public_date
    item = filing(accession, form, original=original, date=d)
    item['xml_content'] = item['xml_content'].replace('2026-09-10', '2026-09-20')
    item['content_hash'] = hashlib.sha256(item['xml_content'].encode()).hexdigest()
    return item


def test_postgres_historical_input_excludes_future_availability_and_future_transaction(db):
    conn, repo, pipeline = db
    future = pit_filing('0009999999-26-000002')
    future['xml_content'] = future['xml_content'].replace('2026-09-20', '2026-09-30')
    future['content_hash'] = hashlib.sha256(future['xml_content'].encode()).hexdigest()
    pipeline._step7_parse_and_reconcile([
        pit_filing(), future, pit_filing('0009999999-26-000003', public_date='2026-10-13'),
    ], 'history')
    rows = repo.get_bulk_transactions_as_of('2026-09-22T12:00:00Z', ['NVDA'], '2026-08-23')['NVDA']
    assert [row['accession_number'] for row in rows] == [BASE]


@pytest.mark.parametrize('scope', ['FULL_REPLACEMENT', 'PARTIAL_CORRECTION', 'ADDITIONAL_TRANSACTION'])
def test_postgres_amendment_effect_is_applied_only_after_publication(db, scope):
    conn, repo, pipeline = db
    pipeline._step7_parse_and_reconcile([
        pit_filing(), pit_filing(AMEND, '4/A', '2026-10-13', '2026-09-21'),
    ], 'amendment')
    # Offline fixture supplies independently reviewed scope; a RESOLVED link alone does not.
    with conn.cursor() as cur:
        cur.execute('UPDATE sm_amendments SET resolution_status=%s WHERE amendment_accession=%s', (scope, AMEND))
    conn.commit()
    before = repo.get_bulk_transactions_as_of('2026-10-12T23:59:59Z', ['NVDA'], '2026-09-12')['NVDA']
    after = repo.get_bulk_transactions_as_of('2026-10-13T23:59:59Z', ['NVDA'], '2026-09-13')['NVDA']
    assert [row['accession_number'] for row in before] == [BASE]
    expected = {BASE, AMEND} if scope == 'ADDITIONAL_TRANSACTION' else {AMEND}
    assert {row['accession_number'] for row in after} == expected


def test_resolved_lineage_without_amendment_scope_blocks_snapshot(db):
    conn, repo, pipeline = db
    pipeline._step7_parse_and_reconcile([
        pit_filing(), pit_filing(AMEND, '4/A', '2026-10-13', '2026-09-21'),
    ], 'unclassified-amendment')
    with pytest.raises(RuntimeError, match='SNAPSHOT_AMENDMENT_EFFECT_UNVERIFIED'):
        repo.get_snapshot_evidence_as_of('2026-10-14T12:00:00Z', pipeline.issuers[:1], '2026-09-14')


def test_snapshot_evidence_read_uses_repeatable_read_and_bound_coverage(db):
    conn, repo, pipeline = db
    f_base = pit_filing()
    pipeline._step7_parse_and_reconcile([f_base], 'persisted-history')
    # Explicit isolated coverage fixture; production must obtain equivalent issuer-bound evidence.
    coverage = dict(receipt_id='offline-coverage', ticker='NVDA', issuer_cik='0001045810',
                    interval_start='2026-08-01', interval_end='2026-11-17',
                    certified_watermark='2026-11-17T23:59:59Z', coverage_status='COMPLETE',
                    missing_count=0, material_gap_count=0,
                    accession_inventory=[BASE], source_hashes={BASE: f_base['content_hash']})
    with patch.object(repo, 'get_authoritative_receipt_for_interval', return_value=coverage):
        result = repo.get_snapshot_evidence_as_of('2026-09-22T12:00:00Z', pipeline.issuers[:1], '2026-08-23')
    assert [row['accession_number'] for row in result['transactions']['NVDA']] == [BASE]
    assert result['coverage_receipt_ids'] == ['offline-coverage']
    assert conn.status == 1


def test_historical_code_p_does_not_override_prohibited_buying_evidence():
    repo = MagicMock(spec=PostgresSmartMoneyRepository)
    row = historical_row()
    row['is_open_market_purchase_evidence'] = False
    repo.get_snapshot_evidence_as_of.return_value = {'transactions': {'NVDA': [row]}, 'coverage_receipt_ids': ['coverage']}
    snapshot = live_snapshot_pipeline(repo)._step8_publish_snapshot(
        [], {'reconciliation_hash': 'hash'}, 'no-inference', as_of='2026-11-17',
    )
    assert snapshot['matched_count'] == 0


def test_amendment_classifier_direct_signals():
    # 1. Full replacement
    f_full = filing(AMEND, '4/A', original='2026-09-11', remarks='This amendment amends and restates the original filing in its entirety.')
    parsed_full = Form4XmlParser.parse_filing(f_full['xml_content'], AMEND)
    scope, reason = classify_amendment_effect(parsed_full, base_filing=filing())
    assert scope == SCOPE_FULL_REPLACEMENT
    assert 'in its entirety' in reason

    # 2. Partial correction via footnotes
    f_part = filing(AMEND, '4/A', original='2026-09-11', footnotes=['This amendment is filed solely to correct the transaction price per share.'])
    parsed_part = Form4XmlParser.parse_filing(f_part['xml_content'], AMEND)
    scope, reason = classify_amendment_effect(parsed_part, base_filing=filing())
    assert scope == SCOPE_PARTIAL_CORRECTION
    assert 'solely to correct' in reason

    # 3. Additional transaction via remarks
    f_add = filing(AMEND, '4/A', original='2026-09-11', remarks='Filed to report additional transactions that were inadvertently omitted from the original Form 4.')
    parsed_add = Form4XmlParser.parse_filing(f_add['xml_content'], AMEND)
    scope, reason = classify_amendment_effect(parsed_add, base_filing=filing())
    assert scope == SCOPE_ADDITIONAL_TRANSACTION
    assert 'inadvertently omitted' in reason

    # 4. Ambiguous case without scope phrases
    f_ambig = filing(AMEND, '4/A', original='2026-09-11')
    parsed_ambig = Form4XmlParser.parse_filing(f_ambig['xml_content'], AMEND)
    scope, _ = classify_amendment_effect(parsed_ambig, base_filing=filing())
    assert scope == SCOPE_AMBIGUOUS

    # 5. Contradictory scope phrases fail closed
    f_contra = filing(AMEND, '4/A', original='2026-09-11', remarks='Filed solely to correct shares and supersedes in its entirety.')
    parsed_contra = Form4XmlParser.parse_filing(f_contra['xml_content'], AMEND)
    scope, _ = classify_amendment_effect(parsed_contra, base_filing=filing())
    assert scope == SCOPE_AMBIGUOUS

    # 6. Unresolved base filing returns UNRESOLVED
    scope, _ = classify_amendment_effect(parsed_full, base_filing=None)
    assert scope == SCOPE_UNRESOLVED


def test_end_to_end_amendment_classification_and_point_in_time_effect(db):
    conn, repo, pipeline = db
    # Base filing published 2026-09-21
    base_item = pit_filing(BASE, '4', '2026-09-21')
    # Full replacement amendment published 2026-10-13
    amend_item = pit_filing(AMEND, '4/A', '2026-10-13', original='2026-09-21')
    amend_item['xml_content'] = amend_item['xml_content'].replace(
        '</ownershipDocument>',
        '<remarks>This Form 4/A replaces the original Form 4 in its entirety.</remarks></ownershipDocument>'
    )
    amend_item['content_hash'] = hashlib.sha256(amend_item['xml_content'].encode()).hexdigest()

    txs, receipt = pipeline._step7_parse_and_reconcile([base_item, amend_item], 'amend-e2e')
    assert receipt['coverage_status'] == 'COMPLETE'

    with conn.cursor() as cur:
        cur.execute('SELECT resolution_status FROM sm_amendments WHERE amendment_accession=%s', (AMEND,))
        assert cur.fetchone()[0] == SCOPE_FULL_REPLACEMENT

    # Before amendment publication (2026-10-12): base filing transactions active
    before = repo.get_bulk_transactions_as_of('2026-10-12T23:59:59Z', ['NVDA'], '2026-09-12')['NVDA']
    assert [row['accession_number'] for row in before] == [BASE]

    # After amendment publication (2026-10-14): base filing superseded, only amendment active
    after = repo.get_bulk_transactions_as_of('2026-10-14T23:59:59Z', ['NVDA'], '2026-09-12')['NVDA']
    assert [row['accession_number'] for row in after] == [AMEND]


def test_record_issuer_coverage_receipt_and_authoritative_retrieval(db):
    conn, repo, pipeline = db
    f_base = pit_filing()
    pipeline._step7_parse_and_reconcile([f_base], 'coverage-test')

    # Record durable coverage receipt bound to NVDA CIK and interval
    r_id = repo.record_issuer_coverage_receipt(
        ticker='NVDA',
        issuer_cik='0001045810',
        interval_start='2026-08-01T00:00:00Z',
        interval_end='2026-11-17T23:59:59Z',
        certified_watermark='2026-11-17T23:59:59Z',
        accession_inventory=[BASE],
        source_hashes={BASE: f_base['content_hash']},
        material_gaps=[],
        coverage_status='COMPLETE',
    )
    assert r_id is not None

    # Retrieve with issuer_cik
    receipt = repo.get_authoritative_receipt_for_interval(
        interval_start='2026-08-23',
        interval_end='2026-09-22T12:00:00Z',
        certified_watermark='2026-09-22T12:00:00Z',
        issuer_cik='0001045810',
    )
    assert receipt is not None
    assert receipt['ticker'] == 'NVDA'
    assert receipt['issuer_cik'] == '0001045810'
    assert receipt['accession_inventory'] == [BASE]
    assert receipt['coverage_status'] == 'COMPLETE'
    assert receipt['material_gap_count'] == 0

    valid, reason = RadarSmartMoneyAdapter.validate_issuer_coverage_receipt(
        receipt, 'NVDA', '0001045810', '2026-08-23', '2026-09-22T12:00:00Z',
    )
    assert valid is True
    assert reason is None

    # Snapshot evidence read succeeds with the durable receipt
    evidence = repo.get_snapshot_evidence_as_of('2026-09-22T12:00:00Z', pipeline.issuers[:1], '2026-08-23')
    assert [row['accession_number'] for row in evidence['transactions']['NVDA']] == [BASE]
    assert r_id in evidence['coverage_receipt_ids']


def test_unbound_global_cycle_receipt_cannot_certify_issuer_historical_coverage(db):
    conn, repo, pipeline = db
    # Global cycle receipt without issuer binding
    global_receipt = {
        'receipt_id': '00000000-0000-0000-0000-000000000099',
        'interval_start': '2026-08-01T00:00:00Z',
        'interval_end': '2026-11-17T23:59:59Z',
        'certified_watermark': '2026-11-17T23:59:59Z',
        'receipt_version': 1,
        'expected_count': 5,
        'accounted_count': 5,
        'missing_count': 0,
        'material_gap_count': 0,
        'coverage_status': 'COMPLETE',
        'failure_reasons': [],
        'reconciliation_hash': 'generic-global-hash',
    }
    repo.save_reconciliation_receipt(global_receipt)

    # Query with specific issuer CIK must NOT match the unbound receipt
    receipt = repo.get_authoritative_receipt_for_interval(
        interval_start='2026-08-23',
        interval_end='2026-09-22T12:00:00Z',
        certified_watermark='2026-09-22T12:00:00Z',
        issuer_cik='0001045810',
    )
    assert receipt is None

    with pytest.raises(RuntimeError, match='SNAPSHOT_HISTORICAL_COVERAGE_UNVERIFIED'):
        repo.get_snapshot_evidence_as_of('2026-09-22T12:00:00Z', pipeline.issuers[:1], '2026-08-23')


def test_missing_coverage_blocks_snapshot_even_if_row_count_zero(db):
    conn, repo, pipeline = db
    # AAPL (CIK 0000320193) has zero rows in the DB and no coverage receipt
    aapl_issuer = [{'ticker': 'AAPL', 'cik': '0000320193', 'company_name': 'APPLE INC'}]
    with pytest.raises(RuntimeError, match='SNAPSHOT_HISTORICAL_COVERAGE_UNVERIFIED'):
        repo.get_snapshot_evidence_as_of('2026-11-17T12:00:00Z', aapl_issuer, '2026-10-18')


def test_verified_zero_activity_requires_and_accepts_authoritative_coverage_receipt(db):
    conn, repo, pipeline = db
    aapl_issuer = [{'ticker': 'AAPL', 'cik': '0000320193', 'company_name': 'APPLE INC'}]

    # Record genuine coverage receipt proving zero activity for AAPL over 30d with authoritative catalog proof
    r_id = repo.record_issuer_coverage_receipt(
        ticker='AAPL',
        issuer_cik='0000320193',
        interval_start='2026-10-01T00:00:00Z',
        interval_end='2026-11-17T23:59:59Z',
        certified_watermark='2026-11-17T23:59:59Z',
        accession_inventory=[],
        source_hashes={'sec_submissions_index_sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},
        material_gaps=[],
        coverage_status='COMPLETE',
        zero_activity_proof={
            'source_catalog_hash': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
            'catalog_watermark': '2026-11-17T23:59:59Z',
            'catalog_type': 'SEC_SUBMISSIONS_INDEX',
        },
    )
    evidence = repo.get_snapshot_evidence_as_of('2026-11-17T12:00:00Z', aapl_issuer, '2026-10-18')
    assert evidence['transactions']['AAPL'] == []
    assert r_id in evidence['coverage_receipt_ids']


def test_empty_inventory_without_catalog_proof_fails_closed(db):
    conn, repo, pipeline = db
    # Attempting to record zero-activity COMPLETE receipt without proof must raise
    with pytest.raises(RuntimeError, match='ZERO_ACTIVITY_UNPROVED'):
        repo.record_issuer_coverage_receipt(
            ticker='AAPL',
            issuer_cik='0000320193',
            interval_start='2026-10-01T00:00:00Z',
            interval_end='2026-11-17T23:59:59Z',
            certified_watermark='2026-11-17T23:59:59Z',
            accession_inventory=[],
            source_hashes={},
            material_gaps=[],
            coverage_status='COMPLETE',
        )


def test_superseding_coverage_receipt_preserves_immutable_history(db):
    conn, repo, pipeline = db
    import uuid
    r1 = str(uuid.uuid4())
    r2 = str(uuid.uuid4())

    f_base = pit_filing(BASE, public_date='2026-08-15')
    f_amend = pit_filing(AMEND, '4/A', public_date='2026-09-10', original='2026-08-15')
    pipeline._step7_parse_and_reconcile([f_base, f_amend], 'super-test')

    # Receipt v1
    repo.record_issuer_coverage_receipt(
        ticker='NVDA',
        issuer_cik='0001045810',
        interval_start='2026-08-01T00:00:00Z',
        interval_end='2026-11-17T23:59:59Z',
        certified_watermark='2026-11-17T23:59:59Z',
        accession_inventory=[BASE],
        source_hashes={BASE: f_base['content_hash']},
        coverage_status='COMPLETE',
        receipt_version=1,
        receipt_id=r1,
    )

    # Superseding receipt v2
    repo.record_issuer_coverage_receipt(
        ticker='NVDA',
        issuer_cik='0001045810',
        interval_start='2026-08-01T00:00:00Z',
        interval_end='2026-11-17T23:59:59Z',
        certified_watermark='2026-11-17T23:59:59Z',
        accession_inventory=[BASE, AMEND],
        source_hashes={BASE: f_base['content_hash'], AMEND: f_amend['content_hash']},
        coverage_status='COMPLETE',
        receipt_version=2,
        supersedes_receipt_id=r1,
        receipt_id=r2,
    )

    # Attempting to UPDATE sm_reconciliation_receipts must fail via trigger
    with conn.cursor() as cur:
        with pytest.raises(Exception, match='IMMUTABILITY_VIOLATION'):
            cur.execute("UPDATE sm_reconciliation_receipts SET coverage_status='TAMPERED' WHERE receipt_id=%s", (r1,))
    conn.rollback()

    # Query returns highest version v2
    active = repo.get_authoritative_receipt_for_interval(
        interval_start='2026-08-23',
        interval_end='2026-09-22T12:00:00Z',
        certified_watermark='2026-09-22T12:00:00Z',
        issuer_cik='0001045810',
    )
    assert active is not None
    assert str(active['receipt_id']) == r2
    assert active['receipt_version'] == 2
    assert str(active['supersedes_receipt_id']) == r1
    assert sorted(active['accession_inventory']) == sorted([BASE, AMEND])


# ── Adversarial Tests for Review Findings 1 - 5 ───────────────────────────────

def test_amendment_classifier_review_regression_cases():
    base = {'accession_number': '0001', 'ticker': 'NVDA'}

    # Case 1: "The entire filing remains unchanged except for a footnote."
    f1 = {'document_type': '4/A', 'is_amendment': True, 'remarks': 'The entire filing remains unchanged except for a footnote.'}
    st1, r1 = classify_amendment_effect(f1, base_filing=base, raw_xml='<ownershipDocument><remarks>The entire filing remains unchanged except for a footnote.</remarks></ownershipDocument>')
    assert st1 == SCOPE_PARTIAL_CORRECTION
    assert st1 != SCOPE_FULL_REPLACEMENT

    # Case 2: "The reporting person entered into an amended and restated voting agreement."
    f2 = {'document_type': '4/A', 'is_amendment': True, 'remarks': 'The reporting person entered into an amended and restated voting agreement.'}
    st2, r2 = classify_amendment_effect(f2, base_filing=base, raw_xml='<ownershipDocument><remarks>The reporting person entered into an amended and restated voting agreement.</remarks></ownershipDocument>')
    assert st2 == SCOPE_AMBIGUOUS
    assert st2 != SCOPE_FULL_REPLACEMENT

    # Case 3: Line-item subset overlap alone must NOT classify as PARTIAL_CORRECTION
    f3 = {'document_type': '4/A', 'is_amendment': True, 'remarks': 'General update', 'transactions': [{'reporting_owner_name': 'Alice'}]}
    base_txs = [{'reporting_owner_name': 'Alice'}, {'reporting_owner_name': 'Alice'}]
    st3, r3 = classify_amendment_effect(f3, base_filing=base, base_transactions=base_txs, raw_xml='<ownershipDocument><remarks>General update</remarks></ownershipDocument>')
    assert st3 == SCOPE_AMBIGUOUS


def test_partial_correction_date_and_code_changes_prevent_duplicates(db):
    conn, repo, pipeline = db
    today = datetime.now(timezone.utc).date().isoformat()
    acc_base = '0000000001-26-000010'
    acc_amend = '0000000001-26-000011'
    base_t_at = '2026-09-01T10:00:00Z'
    amend_t_at = '2026-09-15T10:00:00Z'

    # 1. Ingest base filing (transaction date 2026-08-28, code S)
    f_base = pit_filing(acc_base, public_date='2026-09-01')
    f_base['public_availability_at'] = base_t_at
    pipeline._step7_parse_and_reconcile([f_base], 'pit-base')

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO form4_parsed_transactions (
                transaction_id, accession_number, transaction_date, transaction_code,
                shares, price_per_share, total_value_usd, reporting_owner_name,
                reporting_owner_role, is_director, is_officer, is_ten_percent,
                officer_title, ownership_type, is_derivative, signal_type,
                acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            f"{acc_base}:0001045810:I:1", acc_base, "2026-08-28", "S",
            5000.0, 120.0, 600000.0, "Alice Executive",
            "VP", False, True, False, "VP", "D", False, "OPEN_MARKET_SALE",
            "D", False, False,
        ))

    # 2. Ingest partial correction amendment (corrects date to 2026-08-29, code to P)
    f_amend = pit_filing(acc_amend, '4/A', public_date='2026-09-15', original='2026-09-01')
    f_amend['public_availability_at'] = amend_t_at
    f_amend['xml_content'] = f_amend['xml_content'].replace(
        '</ownershipDocument>',
        '<remarks>This Form 4/A is filed solely to correct the transaction date and code.</remarks></ownershipDocument>'
    )
    f_amend['content_hash'] = hashlib.sha256(f_amend['xml_content'].encode()).hexdigest()
    pipeline._step7_parse_and_reconcile([f_amend], 'pit-amend')

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO form4_parsed_transactions (
                transaction_id, accession_number, transaction_date, transaction_code,
                shares, price_per_share, total_value_usd, reporting_owner_name,
                reporting_owner_role, is_director, is_officer, is_ten_percent,
                officer_title, ownership_type, is_derivative, signal_type,
                acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            f"{acc_amend}:0001045810:I:1", acc_amend, "2026-08-29", "P",
            6000.0, 125.0, 750000.0, "Alice Executive",
            "VP", False, True, False, "VP", "D", False, "OPEN_MARKET_PURCHASE",
            "A", True, True,
        ))

    # Set amendment status to PARTIAL_CORRECTION
    repo.save_amendment(acc_amend, acc_base, 1, SCOPE_PARTIAL_CORRECTION, resolved_at=amend_t_at)

    # 3. Before amendment cutoff (2026-09-10): ONLY base transaction is returned
    before_txs = repo.get_bulk_transactions_as_of('2026-09-10T23:59:59Z', ['NVDA'], '2026-08-01')['NVDA']
    assert len(before_txs) == 1
    assert before_txs[0]['accession_number'] == acc_base
    assert str(before_txs[0]['transaction_date']) == '2026-08-28'
    assert before_txs[0]['transaction_code'] == 'S'

    # 4. After amendment cutoff (2026-09-16): ONLY amended transaction is returned (never both!)
    after_txs = repo.get_bulk_transactions_as_of('2026-09-16T23:59:59Z', ['NVDA'], '2026-08-01')['NVDA']
    assert len(after_txs) == 1
    assert after_txs[0]['accession_number'] == acc_amend
    assert str(after_txs[0]['transaction_date']) == '2026-08-29'
    assert after_txs[0]['transaction_code'] == 'P'
    assert float(after_txs[0]['shares']) == 6000.0


def test_amendment_state_transition_from_resolved_and_conflict_detection(db):
    conn, repo, pipeline = db
    acc_amend = '0002152188-26-000007'
    acc_base = '0002152188-26-000005'

    # 1. Simulate Parker-style legacy RESOLVED record
    repo.save_amendment(acc_amend, acc_base, 1, 'RESOLVED')
    with conn.cursor() as cur:
        cur.execute("SELECT resolution_status FROM sm_amendments WHERE amendment_accession=%s", (acc_amend,))
        assert cur.fetchone()[0] == 'RESOLVED'

    # 2. Transition from RESOLVED to PARTIAL_CORRECTION
    repo.save_amendment(acc_amend, acc_base, 1, SCOPE_PARTIAL_CORRECTION)
    with conn.cursor() as cur:
        cur.execute("SELECT resolution_status FROM sm_amendments WHERE amendment_accession=%s", (acc_amend,))
        assert cur.fetchone()[0] == SCOPE_PARTIAL_CORRECTION

    # 3. Idempotent re-execution of PARTIAL_CORRECTION succeeds
    repo.save_amendment(acc_amend, acc_base, 1, SCOPE_PARTIAL_CORRECTION)
    with conn.cursor() as cur:
        cur.execute("SELECT resolution_status FROM sm_amendments WHERE amendment_accession=%s", (acc_amend,))
        assert cur.fetchone()[0] == SCOPE_PARTIAL_CORRECTION

    # 4. Conflicting transition (e.g. attempting to downgrade to UNRESOLVED or change base) fails closed
    with pytest.raises(RuntimeError, match='AMENDMENT_TRANSITION_CONFLICT'):
        repo.save_amendment(acc_amend, '0002152188-26-999999', 1, 'UNRESOLVED')


def test_tampered_reconciliation_hash_rejected(db):
    conn, repo, pipeline = db
    import uuid
    f_base = pit_filing(BASE, public_date='2026-08-15')
    pipeline._step7_parse_and_reconcile([f_base], 'hash-test')

    # Save a receipt with a forged/tampered reconciliation_hash
    receipt_payload = {
        'receipt_id': str(uuid.uuid4()),
        'interval_start': '2026-08-01T00:00:00Z',
        'interval_end': '2026-11-17T23:59:59Z',
        'certified_watermark': '2026-11-17T23:59:59Z',
        'receipt_version': 1,
        'expected_count': 1,
        'accounted_count': 1,
        'missing_count': 0,
        'material_gap_count': 0,
        'coverage_status': 'COMPLETE',
        'failure_reasons': {
            'reasons': [],
            'ticker': 'NVDA',
            'issuer_cik': '0001045810',
            'accession_inventory': [BASE],
            'source_hashes': {BASE: f_base['content_hash']},
            'material_gaps': [],
        },
        'reconciliation_hash': 'forged_deadbeef_hash_value',
    }
    repo.save_reconciliation_receipt(receipt_payload)

    # Retrieval must reject tampered hash
    receipt = repo.get_authoritative_receipt_for_interval(
        interval_start='2026-08-23',
        interval_end='2026-09-22T12:00:00Z',
        certified_watermark='2026-09-22T12:00:00Z',
        issuer_cik='0001045810',
    )
    assert receipt is None


def test_incomplete_inventory_with_omitted_filing_rejected(db):
    conn, repo, pipeline = db
    # Persist two filings in the interval
    f1 = pit_filing('0000000001-26-000020', public_date='2026-08-15')
    f2 = pit_filing('0000000001-26-000021', public_date='2026-08-20')
    pipeline._step7_parse_and_reconcile([f1, f2], 'omitted-test')

    # Record receipt that claims complete coverage but only includes f1 in accession_inventory (omitting f2)
    r_id = repo.record_issuer_coverage_receipt(
        ticker='NVDA',
        issuer_cik='0001045810',
        interval_start='2026-08-01T00:00:00Z',
        interval_end='2026-11-17T23:59:59Z',
        certified_watermark='2026-11-17T23:59:59Z',
        accession_inventory=['0000000001-26-000020'],
        source_hashes={'0000000001-26-000020': f1['content_hash']},
        coverage_status='COMPLETE',
    )

    # Retrieval must detect omitted filing in interval and reject receipt
    receipt = repo.get_authoritative_receipt_for_interval(
        interval_start='2026-08-01',
        interval_end='2026-11-17T23:59:59Z',
        certified_watermark='2026-11-17T23:59:59Z',
        issuer_cik='0001045810',
    )
    assert receipt is None

