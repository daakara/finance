"""Offline continuous ingestion acceptance, including real isolated PG rollback."""
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    validate_catalog_evidence,
)
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

    # Authentic SEC submissions index catalog for AAPL covering the interval with zero Form 4 filings
    aapl_catalog = {
        "cik": "0000320193",
        "entityType": "operating",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-26-000001"],
                "filingDate": ["2026-09-01"],
                "reportDate": ["2026-08-31"],
                "acceptanceDateTime": ["2026-09-01T16:30:00.000Z"],
                "form": ["8-K"],
            },
            "files": [],
        },
    }
    cat_bytes = json.dumps(aapl_catalog).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    # Record genuine coverage receipt proving zero activity for AAPL over 30d with authoritative catalog proof
    r_id = repo.record_issuer_coverage_receipt(
        ticker='AAPL',
        issuer_cik='0000320193',
        interval_start='2026-10-01T00:00:00Z',
        interval_end='2026-11-17T23:59:59Z',
        certified_watermark='2026-11-17T23:59:59Z',
        accession_inventory=[],
        source_hashes={'sec_submissions_index_sha256': cat_hash},
        material_gaps=[],
        coverage_status='COMPLETE',
        zero_activity_proof={
            'catalog_content': cat_bytes.decode('utf-8'),
            'source_catalog_hash': cat_hash,
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

    base_xml = """<ownershipDocument><documentType>4</documentType>
      <periodOfReport>2026-08-28</periodOfReport>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName>
        <issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0001045810</rptOwnerCik>
        <rptOwnerName>Alice Executive</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isOfficer>true</isOfficer><officerTitle>VP</officerTitle></reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable><nonDerivativeTransaction>
        <transactionDate><value>2026-08-28</value></transactionDate>
        <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
        <transactionAmounts><transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>120</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
        </transactionAmounts></nonDerivativeTransaction></nonDerivativeTable>
      </ownershipDocument>"""

    amend_xml = """<ownershipDocument><documentType>4/A</documentType>
      <periodOfReport>2026-08-28</periodOfReport>
      <dateOfOriginalSubmission>2026-09-01</dateOfOriginalSubmission>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName>
        <issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0001045810</rptOwnerCik>
        <rptOwnerName>Alice Executive</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isOfficer>true</isOfficer><officerTitle>VP</officerTitle></reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable><nonDerivativeTransaction>
        <transactionDate><value>2026-08-29</value></transactionDate>
        <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
        <transactionAmounts><transactionShares><value>6000</value></transactionShares>
        <transactionPricePerShare><value>125</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
        </transactionAmounts></nonDerivativeTransaction></nonDerivativeTable>
      <remarks>This Form 4/A is filed solely to correct the transaction date and code.</remarks>
      </ownershipDocument>"""

    # 1. Ingest base filing (transaction date 2026-08-28, code S)
    f_base = pit_filing(acc_base, public_date='2026-09-01')
    f_base['xml_content'] = base_xml
    f_base['content_hash'] = hashlib.sha256(base_xml.encode()).hexdigest()
    f_base['public_availability_at'] = base_t_at
    pipeline._step7_parse_and_reconcile([f_base], 'pit-base')

    # 2. Ingest partial correction amendment (corrects date to 2026-08-29, code to P)
    f_amend = pit_filing(acc_amend, '4/A', public_date='2026-09-15', original='2026-09-01')
    f_amend['public_availability_at'] = amend_t_at
    f_amend['xml_content'] = amend_xml
    f_amend['content_hash'] = hashlib.sha256(amend_xml.encode()).hexdigest()
    pipeline._step7_parse_and_reconcile([f_amend], 'pit-amend')

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


def test_reproduction_gate1a_false_verified_zero_coverage_rejected(db):
    """
    GATE 1A: False verified-zero coverage reproduction.
    Caller supplies empty accession inventory with source_hashes={"catalog": ""}.
    Rejection is required: record_issuer_coverage_receipt must refuse to certify COMPLETE coverage
    without valid, verified catalog artifact bytes and SHA-256 digest, and get_authoritative_receipt_for_interval
    must reject any such receipt.
    """
    conn, repo, pipeline = db

    # Rejection is required at writer level
    with pytest.raises(RuntimeError, match="ZERO_ACTIVITY_UNPROVED|AUTHORITATIVE_CATALOG_PROOF"):
        repo.record_issuer_coverage_receipt(
            ticker='NVDA',
            issuer_cik='0001045810',
            interval_start='2026-08-01T00:00:00Z',
            interval_end='2026-11-17T23:59:59Z',
            certified_watermark='2026-11-17T23:59:59Z',
            accession_inventory=[],
            source_hashes={"catalog": ""},
            coverage_status='COMPLETE',
        )


def test_reproduction_gate1b_incorrect_partial_correction_positional_mapping_rejected(db):
    """
    GATE 1B: Incorrect partial-correction mapping reproduction.
    Base filing contains transaction A at sequence 0 ($100) and B at sequence 1 ($200).
    Amendment contains only corrected B, indexed locally at sequence 0 ($250).
    Positional sequence matching erroneously suppresses A and retains both original B and corrected B.
    Rejection is required: positional matching is invalid; the system must retain A ($100) and replace ONLY B ($200 -> $250)
    when explicit evidence-backed mapping is provided, or block if unproven.
    """
    conn, repo, pipeline = db

    base_xml = """<ownershipDocument><documentType>4</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>100</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>200</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    f_base = dict(accession_number="0002152188-26-000005", ticker="NVDA", cik="0001045810",
                  company_name="NVIDIA", filing_date="2026-09-01", document_type="4",
                  accepted_datetime="2026-09-01T21:00:00Z", xml_content=base_xml,
                  source_url="https://www.sec.gov/Archives/0002152188-26-000005.xml",
                  content_hash=hashlib.sha256(base_xml.encode()).hexdigest())

    amend_xml = """<ownershipDocument><documentType>4/A</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <dateOfOriginalSubmission>2026-09-01</dateOfOriginalSubmission>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>250</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <remarks>This Form 4/A is filed solely to correct the transaction price of the second transaction.</remarks>
    </ownershipDocument>"""

    f_amend = dict(accession_number="0002152188-26-000007", ticker="NVDA", cik="0001045810",
                   company_name="NVIDIA", filing_date="2026-09-05", document_type="4/A",
                   accepted_datetime="2026-09-05T21:00:00Z", xml_content=amend_xml,
                   source_url="https://www.sec.gov/Archives/0002152188-26-000007.xml",
                   content_hash=hashlib.sha256(amend_xml.encode()).hexdigest())

    pipeline._step7_parse_and_reconcile([f_base], "cycle1")
    pipeline._step7_parse_and_reconcile([f_amend], "cycle2")

    tx_map = repo.get_bulk_transactions_as_of("2026-09-10T00:00:00Z", ["NVDA"], "2026-08-01")
    nvda_txs = tx_map.get("NVDA", [])

    prices = sorted([float(t["price_per_share"]) for t in nvda_txs])
    assert prices == [100.0, 250.0], f"Positional matching error: expected [100.0, 250.0] but got {prices}"


def test_catalog_evidence_digest_tampering_and_mismatch_rejected():
    """GATE 4: Digest tampering and invalid format rejection."""
    valid_cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000001"],
                "filingDate": ["2026-07-15"],
                "reportDate": ["2026-07-15"],
                "acceptanceDateTime": ["2026-07-15T16:00:00.000Z"],
                "form": ["8-K"],
            },
            "files": [],
        },
    }
    cat_bytes = json.dumps(valid_cat).encode("utf-8")
    actual_hash = hashlib.sha256(cat_bytes).hexdigest()

    # Tampered / mismatched hash
    with pytest.raises(ValueError, match="CATALOG_DIGEST_MISMATCH"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256="0" * 64,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
        )

    # Malformed digest (not 64 lowercase hex)
    with pytest.raises(ValueError, match="INVALID_CATALOG_DIGEST"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256="not_a_valid_hex",
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
        )

    # Empty bytes
    with pytest.raises(ValueError, match="EMPTY_CATALOG_BYTES"):
        validate_catalog_evidence(
            catalog_bytes=b"",
            catalog_sha256=actual_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
        )


def test_catalog_evidence_wrong_issuer_cik_and_ticker_rejected():
    """GATE 4: Rejection of catalog evidence when CIK or ticker does not match."""
    cat = {
        "cik": "0000320193",  # AAPL CIK
        "entityType": "operating",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-26-000001"],
                "filingDate": ["2026-07-15"],
                "reportDate": ["2026-07-15"],
                "acceptanceDateTime": ["2026-07-15T16:00:00.000Z"],
                "form": ["8-K"],
            },
            "files": [],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    # Wrong CIK (passing NVDA CIK 0001045810 for AAPL catalog)
    with pytest.raises(ValueError, match="WRONG_ISSUER_CIK"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="AAPL",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
        )

    # Wrong Ticker (passing MSFT for AAPL catalog)
    with pytest.raises(ValueError, match="WRONG_ISSUER_TICKER"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="MSFT",
            issuer_cik="0000320193",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
        )


def test_catalog_evidence_incomplete_interval_rejected():
    """GATE 4: Rejection of catalog that does not exhaustively cover the observation interval."""
    # Earliest date in catalog is 2026-10-15, but interval starts 2026-08-01
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000010"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00.000Z"],
                "form": ["8-K"],
            },
            "files": [{"name": "CIK0001045810-submissions-001.json", "filingCount": 1000}],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    with pytest.raises(ValueError, match="CATALOG_INTERVAL_INCOMPLETE"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
        )


def test_catalog_evidence_caller_inventory_omission_rejected():
    """GATE 4: Rejection when caller inventory omits an eligible filing found in catalog."""
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000005", "0001045810-26-000001"],
                "filingDate": ["2026-09-01", "2026-07-01"],
                "reportDate": ["2026-09-01", "2026-07-01"],
                "acceptanceDateTime": ["2026-09-01T16:00:00.000Z", "2026-07-01T16:00:00.000Z"],
                "form": ["4", "8-K"],
            },
            "files": [],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    # Caller claims empty inventory, but catalog has Form 4 0001045810-26-000005
    with pytest.raises(ValueError, match="CALLER_INVENTORY_OMISSION"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            accession_inventory=[],
        )


def test_record_amendment_mappings_validation_wrong_base_duplicate_target(db):
    """GATE 4: Validation rules in record_amendment_transaction_mappings."""
    conn, repo, pipeline = db

    base_acc = "0001045810-26-000100"
    amend_acc = "0001045810-26-000101"

    base_xml = """<ownershipDocument><documentType>4</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>100</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>200</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    amend_xml = """<ownershipDocument><documentType>4/A</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <dateOfOriginalSubmission>2026-09-01</dateOfOriginalSubmission>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>250</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <derivativeTable>
        <derivativeTransaction>
          <securityTitle><value>Option</value></securityTitle>
          <conversionOrExercisePrice><value>100</value></conversionOrExercisePrice>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>50</value></transactionShares><transactionPricePerShare><value>0</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
          <exerciseDate><value>2026-09-01</value></exerciseDate>
          <expirationDate><value>2028-09-01</value></expirationDate>
          <underlyingSecurity><underlyingSecurityTitle><value>Common</value></underlyingSecurityTitle><underlyingSecurityShares><value>50</value></underlyingSecurityShares></underlyingSecurity>
        </derivativeTransaction>
      </derivativeTable>
      <remarks>Solely to correct.</remarks>
    </ownershipDocument>"""

    f_base = dict(accession_number=base_acc, ticker="NVDA", cik="0001045810",
                  company_name="NVIDIA", filing_date="2026-09-01", document_type="4",
                  accepted_datetime="2026-09-01T21:00:00Z", xml_content=base_xml,
                  source_url=f"https://www.sec.gov/Archives/{base_acc}.xml",
                  content_hash=hashlib.sha256(base_xml.encode()).hexdigest())

    f_amend = dict(accession_number=amend_acc, ticker="NVDA", cik="0001045810",
                   company_name="NVIDIA", filing_date="2026-09-05", document_type="4/A",
                   accepted_datetime="2026-09-05T21:00:00Z", xml_content=amend_xml,
                   source_url=f"https://www.sec.gov/Archives/{amend_acc}.xml",
                   content_hash=hashlib.sha256(amend_xml.encode()).hexdigest())

    pipeline._step7_parse_and_reconcile([f_base], "m-base")
    pipeline._step7_parse_and_reconcile([f_amend], "m-amend")

    b_txs = repo.get_parsed_transactions_for_accession(base_acc)
    a_txs = repo.get_parsed_transactions_for_accession(amend_acc)
    b_tx_ids = sorted([t["transaction_id"] for t in b_txs])

    deriv_a_id = [t["transaction_id"] for t in a_txs if t.get("is_derivative")][0]
    non_deriv_a_id = [t["transaction_id"] for t in a_txs if not t.get("is_derivative")][0]

    # Table mismatch
    with pytest.raises(ValueError, match="WRONG_TABLE_MAPPING"):
        repo.record_amendment_transaction_mappings(
            base_accession=base_acc,
            amendment_accession=amend_acc,
            mappings=[
                {"amendment_transaction_id": deriv_a_id, "base_transaction_id": b_tx_ids[0]},
                {"amendment_transaction_id": non_deriv_a_id, "base_transaction_id": b_tx_ids[1]},
            ],
        )

    # Duplicate target mapping
    with pytest.raises(ValueError, match="DUPLICATE_TARGET_MAPPING"):
        repo.record_amendment_transaction_mappings(
            base_accession=base_acc,
            amendment_accession=amend_acc,
            mappings=[
                {"amendment_transaction_id": non_deriv_a_id, "base_transaction_id": b_tx_ids[0]},
                {"amendment_transaction_id": deriv_a_id, "base_transaction_id": b_tx_ids[0]},
            ],
        )


def test_unproven_mapping_blocks_snapshot_publication_fail_closed(db):
    """GATE 4: Unproven mapping for multi-transaction base blocks snapshot publication fail-closed."""
    conn, repo, pipeline = db

    base_acc = "0001045810-26-000200"
    amend_acc = "0001045810-26-000201"

    base_xml = """<ownershipDocument><documentType>4</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>100</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>200</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    amend_xml = """<ownershipDocument><documentType>4/A</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <dateOfOriginalSubmission>2026-09-01</dateOfOriginalSubmission>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>250</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <remarks>This Form 4/A is filed solely to correct the transaction price.</remarks>
    </ownershipDocument>"""

    f_base = dict(accession_number=base_acc, ticker="NVDA", cik="0001045810",
                  company_name="NVIDIA", filing_date="2026-09-01", document_type="4",
                  accepted_datetime="2026-09-01T21:00:00Z", xml_content=base_xml,
                  source_url=f"https://www.sec.gov/Archives/{base_acc}.xml",
                  content_hash=hashlib.sha256(base_xml.encode()).hexdigest())

    f_amend = dict(accession_number=amend_acc, ticker="NVDA", cik="0001045810",
                   company_name="NVIDIA", filing_date="2026-09-05", document_type="4/A",
                   accepted_datetime="2026-09-05T21:00:00Z", xml_content=amend_xml,
                   source_url=f"https://www.sec.gov/Archives/{amend_acc}.xml",
                   content_hash=hashlib.sha256(amend_xml.encode()).hexdigest())

    pipeline._step7_parse_and_reconcile([f_base], "unprov-base")
    pipeline._step7_parse_and_reconcile([f_amend], "unprov-amend")

    ambig_symbols = repo.get_ambiguous_amendment_symbols_as_of("2026-09-10T00:00:00Z", ["NVDA"])
    assert "NVDA" in ambig_symbols

    tx_map = repo.get_bulk_transactions_as_of("2026-09-10T00:00:00Z", ["NVDA"], "2026-08-01")
    accs = [t["accession_number"] for t in tx_map["NVDA"]]
    assert amend_acc not in accs
    assert len(accs) == 2


def test_amendment_chain_point_in_time_supersession_retains_unamended_lines(db):
    """GATE 4: Amendment chain supersession: Base -> Amend1 (corrects line 2) -> Amend2 (corrects line 2 again).
    Verify Amend2 supersedes Amend1 at Amend2 cutoff while retaining unamended base line 1 throughout."""
    conn, repo, pipeline = db

    base_acc = "0001045810-26-000300"
    amend1_acc = "0001045810-26-000301"
    amend2_acc = "0001045810-26-000302"

    t_base = "2026-09-01T10:00:00Z"
    t_amend1 = "2026-09-05T10:00:00Z"
    t_amend2 = "2026-09-10T10:00:00Z"

    base_xml = """<ownershipDocument><documentType>4</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>100</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>200</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    amend1_xml = """<ownershipDocument><documentType>4/A</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <dateOfOriginalSubmission>2026-09-01</dateOfOriginalSubmission>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>220</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <remarks>Solely to correct the second transaction price.</remarks>
    </ownershipDocument>"""

    amend2_xml = """<ownershipDocument><documentType>4/A</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <dateOfOriginalSubmission>2026-09-01</dateOfOriginalSubmission>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>250</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <remarks>Solely to correct the second transaction price.</remarks>
    </ownershipDocument>"""

    f_base = dict(accession_number=base_acc, ticker="NVDA", cik="0001045810",
                  company_name="NVIDIA", filing_date="2026-09-01", document_type="4",
                  accepted_datetime=t_base, xml_content=base_xml,
                  source_url=f"https://www.sec.gov/Archives/{base_acc}.xml",
                  content_hash=hashlib.sha256(base_xml.encode()).hexdigest(),
                  public_availability_at=t_base)

    f_amend1 = dict(accession_number=amend1_acc, ticker="NVDA", cik="0001045810",
                    company_name="NVIDIA", filing_date="2026-09-05", document_type="4/A",
                    accepted_datetime=t_amend1, xml_content=amend1_xml,
                    source_url=f"https://www.sec.gov/Archives/{amend1_acc}.xml",
                    content_hash=hashlib.sha256(amend1_xml.encode()).hexdigest(),
                    public_availability_at=t_amend1)

    f_amend2 = dict(accession_number=amend2_acc, ticker="NVDA", cik="0001045810",
                    company_name="NVIDIA", filing_date="2026-09-10", document_type="4/A",
                    accepted_datetime=t_amend2, xml_content=amend2_xml,
                    source_url=f"https://www.sec.gov/Archives/{amend2_acc}.xml",
                    content_hash=hashlib.sha256(amend2_xml.encode()).hexdigest(),
                    public_availability_at=t_amend2)

    pipeline._step7_parse_and_reconcile([f_base], "chain-base")
    pipeline._step7_parse_and_reconcile([f_amend1], "chain-amend1")

    # Set amendment_number = 2 for amend2 to reflect sequence
    repo.save_amendment(amend2_acc, base_acc, 2, "PARTIAL_CORRECTION", resolved_at=t_amend2)
    pipeline._step7_parse_and_reconcile([f_amend2], "chain-amend2")
    repo.save_amendment(amend2_acc, base_acc, 2, "PARTIAL_CORRECTION", resolved_at=t_amend2)

    # Point 1: Before Amend 1 (2026-09-03): Base A ($100) and Base B ($200)
    txs_p1 = repo.get_bulk_transactions_as_of("2026-09-03T00:00:00Z", ["NVDA"], "2026-08-01")["NVDA"]
    prices_p1 = sorted([float(t["price_per_share"]) for t in txs_p1])
    assert prices_p1 == [100.0, 200.0]

    # Point 2: Between Amend 1 and Amend 2 (2026-09-07): Base A ($100) and Amend1 B ($220)
    txs_p2 = repo.get_bulk_transactions_as_of("2026-09-07T00:00:00Z", ["NVDA"], "2026-08-01")["NVDA"]
    prices_p2 = sorted([float(t["price_per_share"]) for t in txs_p2])
    assert prices_p2 == [100.0, 220.0]

    # Point 3: After Amend 2 (2026-09-12): Base A ($100) and Amend2 B ($250) (Amend1 B is superseded!)
    txs_p3 = repo.get_bulk_transactions_as_of("2026-09-12T00:00:00Z", ["NVDA"], "2026-08-01")["NVDA"]
    prices_p3 = sorted([float(t["price_per_share"]) for t in txs_p3])
    assert prices_p3 == [100.0, 250.0]


def test_reproduction_gate2a_empty_recent_catalog_with_overlapping_files_rejected():
    """
    REPRODUCTION GATE 1:
    validate_catalog_evidence must NOT accept an empty filings.recent catalog with an
    overlapping unread filings.files entry as COMPLETE / NO_ELIGIBLE_FILINGS.
    It must reject incomplete catalog evidence.
    """
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": [],
                "filingDate": [],
                "reportDate": [],
                "acceptanceDateTime": [],
                "form": [],
            },
            "files": [{
                "name": "CIK0001045810-submissions-001.json",
                "filingCount": 500,
                "filingFrom": "2026-08-01",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    with pytest.raises(ValueError, match="CATALOG_INTERVAL_INCOMPLETE|HISTORICAL_FILES_UNREAD"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            accession_inventory=[],
        )


def test_reproduction_gate2b_unrelated_ordinal_second_transaction_not_first():
    """
    REPRODUCTION GATE 2:
    classify_amendment_effect must NOT map:
    'This amendment is filed solely to correct the price of the second transaction. The first quarter award is otherwise unchanged.'
    to the FIRST base transaction. It must contextually identify the second transaction.
    """
    amend_filing = {
        "document_type": "4/A",
        "is_amendment": True,
        "transactions": [{"record_key": "amend_acc:0001045810:NON_DERIVATIVE:0", "is_derivative": False}],
    }
    base_filing = {"accession_number": "base_acc", "content_hash": "h1"}
    base_txs = [
        {"transaction_id": "base_acc:0001045810:NON_DERIVATIVE:0", "is_derivative": False},
        {"transaction_id": "base_acc:0001045810:NON_DERIVATIVE:1", "is_derivative": False},
    ]
    narrative = (
        "This amendment is filed solely to correct the price of the second transaction. "
        "The first quarter award is otherwise unchanged."
    )
    raw_xml = f"<ownershipDocument><remarks>{narrative}</remarks></ownershipDocument>"

    res = classify_amendment_effect(
        amendment_filing=amend_filing,
        base_filing=base_filing,
        base_transactions=base_txs,
        raw_xml=raw_xml,
    )
    assert res.status == "PARTIAL_CORRECTION"
    assert len(res.mappings) == 1
    # Must map to the SECOND transaction (index 1 / seq 1), NOT the first transaction
    assert res.mappings[0]["base_transaction_id"] == "base_acc:0001045810:NON_DERIVATIVE:1"


def test_catalog_evidence_missing_pagination_pages_rejected():
    """QUALITY GATE: Rejection when overlapping historical pagination files are missing from pagination_artifacts."""
    cat_pages = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [
                {"name": "page-001.json", "filingCount": 100, "filingFrom": "2026-09-01", "filingTo": "2026-10-01"},
                {"name": "page-002.json", "filingCount": 100, "filingFrom": "2026-08-01", "filingTo": "2026-08-31"},
            ],
        },
    }
    cat_bytes = json.dumps(cat_pages).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page1_content = {
        "accessionNumber": ["0001045810-26-000040"],
        "filingDate": ["2026-09-15"],
        "acceptanceDateTime": ["2026-09-15T16:00:00Z"],
        "form": ["4"],
    }

    with pytest.raises(ValueError, match="CATALOG_INTERVAL_INCOMPLETE.*page-002.json"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            pagination_artifacts={"page-001.json": page1_content},  # page-002 is MISSING
        )


def test_catalog_evidence_valid_complete_with_verified_pagination_artifacts():
    """QUALITY GATE: Acceptance of complete evidence across recent and historical pagination files."""
    cat_pages = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [
                {"name": "page-001.json", "filingCount": 100, "filingFrom": "2026-09-01", "filingTo": "2026-10-01"},
                {"name": "page-002.json", "filingCount": 100, "filingFrom": "2026-08-01", "filingTo": "2026-08-31"},
            ],
        },
    }
    cat_bytes = json.dumps(cat_pages).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page1_content = {
        "accessionNumber": ["0001045810-26-000040"],
        "filingDate": ["2026-09-15"],
        "acceptanceDateTime": ["2026-09-15T16:00:00Z"],
        "form": ["4"],
    }
    page2_content = {
        "accessionNumber": ["0001045810-26-000030"],
        "filingDate": ["2026-08-01"],
        "acceptanceDateTime": ["2026-08-01T16:00:00Z"],
        "form": ["4"],
    }
    page1_bytes = json.dumps(page1_content).encode("utf-8")
    page2_bytes = json.dumps(page2_content).encode("utf-8")
    page1_hash = hashlib.sha256(page1_bytes).hexdigest()
    page2_hash = hashlib.sha256(page2_bytes).hexdigest()

    res = validate_catalog_evidence(
        catalog_bytes=cat_bytes,
        catalog_sha256=cat_hash,
        ticker="NVDA",
        issuer_cik="0001045810",
        interval_start="2026-08-01T00:00:00Z",
        interval_end="2026-11-17T23:59:59Z",
        certified_watermark="2026-11-17T23:59:59Z",
        pagination_artifacts={
            "page-001.json": page1_bytes,
            "page-002.json": page2_bytes,
        },
        accession_inventory=["0001045810-26-000030", "0001045810-26-000040"],
        source_hashes={
            "0001045810-26-000030": "a" * 64,
            "0001045810-26-000040": "b" * 64,
            "page-001.json": page1_hash,
            "page-002.json": page2_hash,
        },
    )
    assert res["coverage_status"] == "COMPLETE"
    assert res["derived_inventory"] == ["0001045810-26-000030", "0001045810-26-000040"]
    assert res["zero_activity_reason"] is None
    assert res["historical_manifest"] == {
        "page-001.json": page1_hash,
        "page-002.json": page2_hash,
    }


def test_catalog_evidence_genuine_verified_zero_activity_accepted():
    """QUALITY GATE: Genuine verified-zero activity accepted when empty recent has zero historical files."""
    cat_zero = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": [],
                "filingDate": [],
                "reportDate": [],
                "acceptanceDateTime": [],
                "form": [],
            },
            "files": [],
        },
    }
    cat_bytes = json.dumps(cat_zero).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    res = validate_catalog_evidence(
        catalog_bytes=cat_bytes,
        catalog_sha256=cat_hash,
        ticker="NVDA",
        issuer_cik="0001045810",
        interval_start="2026-08-01T00:00:00Z",
        interval_end="2026-11-17T23:59:59Z",
        certified_watermark="2026-11-17T23:59:59Z",
        accession_inventory=[],
    )
    assert res["coverage_status"] == "COMPLETE"
    assert res["derived_inventory"] == []
    assert res["zero_activity_reason"] == "NO_ELIGIBLE_FILINGS"


def test_classify_amendment_effect_unrelated_ordinals_ignored():
    """QUALITY GATE: Unrelated ordinal words (first quarter, first grant, first half) must not map to line 1."""
    amend_filing = {
        "document_type": "4/A",
        "is_amendment": True,
        "transactions": [{"record_key": "amend_acc:0001045810:NON_DERIVATIVE:0", "is_derivative": False}],
    }
    base_filing = {"accession_number": "base_acc", "content_hash": "h1"}
    base_txs = [
        {"transaction_id": "base_acc:0001045810:NON_DERIVATIVE:0", "is_derivative": False},
        {"transaction_id": "base_acc:0001045810:NON_DERIVATIVE:1", "is_derivative": False},
    ]

    # 1. First quarter award with no transaction ordinal -> must fail closed to AMENDMENT_SCOPE_AMBIGUOUS
    xml_no_tx = (
        "<ownershipDocument><remarks>"
        "This amendment is filed solely to correct an error. The first quarter award is otherwise unchanged."
        "</remarks></ownershipDocument>"
    )
    res_no_tx = classify_amendment_effect(
        amendment_filing=amend_filing,
        base_filing=base_filing,
        base_transactions=base_txs,
        raw_xml=xml_no_tx,
    )
    assert res_no_tx.status == "AMENDMENT_SCOPE_AMBIGUOUS"
    assert res_no_tx.mappings == []

    # 2. First grant and first half with second transaction -> maps to second transaction
    xml_second = (
        "<ownershipDocument><remarks>"
        "This amendment is filed solely to correct the second transaction price. "
        "The first grant from the first half of the year is unchanged."
        "</remarks></ownershipDocument>"
    )
    res_second = classify_amendment_effect(
        amendment_filing=amend_filing,
        base_filing=base_filing,
        base_transactions=base_txs,
        raw_xml=xml_second,
    )
    assert res_second.status == "PARTIAL_CORRECTION"
    assert len(res_second.mappings) == 1
    assert res_second.mappings[0]["base_transaction_id"] == "base_acc:0001045810:NON_DERIVATIVE:1"


def test_classify_amendment_effect_conflicting_ordinals_fails_closed():
    """QUALITY GATE: Conflicting or ambiguous transaction ordinal references in narrative must fail closed."""
    amend_filing = {
        "document_type": "4/A",
        "is_amendment": True,
        "transactions": [{"record_key": "amend_acc:0001045810:NON_DERIVATIVE:0", "is_derivative": False}],
    }
    base_filing = {"accession_number": "base_acc", "content_hash": "h1"}
    base_txs = [
        {"transaction_id": "base_acc:0001045810:NON_DERIVATIVE:0", "is_derivative": False},
        {"transaction_id": "base_acc:0001045810:NON_DERIVATIVE:1", "is_derivative": False},
    ]
    # Narrative refers to both the first transaction and the second transaction
    xml_conflict = (
        "<ownershipDocument><remarks>"
        "This amendment is filed solely to correct the first transaction and the second transaction."
        "</remarks></ownershipDocument>"
    )
    res = classify_amendment_effect(
        amendment_filing=amend_filing,
        base_filing=base_filing,
        base_transactions=base_txs,
        raw_xml=xml_conflict,
    )
    assert res.status == "AMENDMENT_SCOPE_AMBIGUOUS"
    assert res.mappings == []


def test_classify_amendment_effect_reordered_base_transactions():
    """QUALITY GATE: Resolving source ordinals against source sequence independently of database result order."""
    amend_filing = {
        "document_type": "4/A",
        "is_amendment": True,
        "transactions": [{"record_key": "amend_acc:0001045810:NON_DERIVATIVE:0", "is_derivative": False}],
    }
    base_filing = {"accession_number": "base_acc", "content_hash": "h1"}
    # Base transactions provided in REVERSED order (seq 1 first, seq 0 second)
    base_txs_reversed = [
        {"transaction_id": "base_acc:0001045810:NON_DERIVATIVE:1", "is_derivative": False, "price": 200},
        {"transaction_id": "base_acc:0001045810:NON_DERIVATIVE:0", "is_derivative": False, "price": 100},
    ]
    narrative = (
        "This amendment is filed solely to correct the price of the second transaction. "
        "The first quarter award is otherwise unchanged."
    )
    raw_xml = f"<ownershipDocument><remarks>{narrative}</remarks></ownershipDocument>"

    res = classify_amendment_effect(
        amendment_filing=amend_filing,
        base_filing=base_filing,
        base_transactions=base_txs_reversed,
        raw_xml=raw_xml,
    )
    assert res.status == "PARTIAL_CORRECTION"
    assert len(res.mappings) == 1
    # Must correctly map to seq 1 (base_acc:0001045810:NON_DERIVATIVE:1) despite list order
    assert res.mappings[0]["base_transaction_id"] == "base_acc:0001045810:NON_DERIVATIVE:1"


def test_amendment_reordered_base_transactions_preserves_unaffected_transactions_in_db(db):
    """
    QUALITY GATE: End-to-end database replay test on disposable PostgreSQL.
    Proves that correcting the second transaction preserves unaffected base transaction (seq 0) active.
    """
    conn, repo, pipeline = db

    base_acc = "0002152188-26-000088"
    amend_acc = "0002152188-26-000089"

    base_xml = """<ownershipDocument><documentType>4</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>100</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>200</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    amend_xml = """<ownershipDocument><documentType>4/A</documentType>
      <periodOfReport>2026-09-01</periodOfReport>
      <dateOfOriginalSubmission>2026-09-01</dateOfOriginalSubmission>
      <issuer><issuerCik>0001045810</issuerCik><issuerName>NVIDIA</issuerName><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>0002152188</rptOwnerCik><rptOwnerName>Parker</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-09-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts><transactionShares><value>20</value></transactionShares><transactionPricePerShare><value>250</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <remarks>This amendment is filed solely to correct the price of the second transaction. The first quarter award is otherwise unchanged.</remarks>
    </ownershipDocument>"""

    t_base = "2026-09-01T21:00:00Z"
    t_amend = "2026-09-05T21:00:00Z"

    f_base = dict(accession_number=base_acc, ticker="NVDA", cik="0001045810",
                  company_name="NVIDIA", filing_date="2026-09-01", document_type="4",
                  accepted_datetime=t_base, xml_content=base_xml,
                  source_url=f"https://www.sec.gov/Archives/{base_acc}.xml",
                  content_hash=hashlib.sha256(base_xml.encode()).hexdigest(),
                  public_availability_at=t_base)

    f_amend = dict(accession_number=amend_acc, ticker="NVDA", cik="0001045810",
                   company_name="NVIDIA", filing_date="2026-09-05", document_type="4/A",
                   accepted_datetime=t_amend, xml_content=amend_xml,
                   source_url=f"https://www.sec.gov/Archives/{amend_acc}.xml",
                   content_hash=hashlib.sha256(amend_xml.encode()).hexdigest(),
                   public_availability_at=t_amend)

    pipeline._step7_parse_and_reconcile([f_base], "reordered-test-base")
    pipeline._step7_parse_and_reconcile([f_amend], "reordered-test-amend")

    # Query before amendment: Base A ($100) and Base B ($200)
    txs_before = repo.get_bulk_transactions_as_of("2026-09-03T00:00:00Z", ["NVDA"], "2026-08-01")["NVDA"]
    prices_before = sorted([float(t["price_per_share"]) for t in txs_before])
    assert prices_before == [100.0, 200.0]

    # Query after amendment: Base A ($100) remains ACTIVE; Base B ($200) replaced by Amendment B ($250)
    txs_after = repo.get_bulk_transactions_as_of("2026-09-10T00:00:00Z", ["NVDA"], "2026-08-01")["NVDA"]
    prices_after = sorted([float(t["price_per_share"]) for t in txs_after])
    assert prices_after == [100.0, 250.0]


def test_reproduction_unhashed_substitute_page_cannot_certify_complete_rejected():
    """
    REPRODUCTION:
    An unhashed substitute historical page must NOT certify COMPLETE / NO_ELIGIBLE_FILINGS.
    validate_catalog_evidence must require a pinned, valid 64-char SHA-256 digest for every
    historical page used to certify coverage.
    """
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [{
                "name": "page-001.json",
                "filingCount": 100,
                "filingFrom": "2026-08-01",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    # Unhashed substitute page with zero Form 4 filings
    substitute_page = {
        "accessionNumber": ["0001045810-26-000099"],
        "filingDate": ["2026-08-01"],
        "acceptanceDateTime": ["2026-08-01T16:00:00Z"],
        "form": ["8-K"],
    }

    with pytest.raises(ValueError, match="MISSING_PAGINATION_DIGEST|UNHASHED_PAGINATION_PAGE"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            pagination_artifacts={"page-001.json": substitute_page},
            accession_inventory=[],
            source_hashes={},  # No hash provided for page-001.json!
        )


def test_catalog_evidence_substituted_page_digest_mismatch_rejected():
    """
    QUALITY GATE:
    Changed page bytes or substituted historical artifacts must be rejected with PAGINATION_DIGEST_MISMATCH.
    """
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [{
                "name": "page-001.json",
                "filingCount": 100,
                "filingFrom": "2026-08-01",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page_bytes = b'{"accessionNumber": ["0001045810-26-000099"], "filingDate": ["2026-08-01"], "acceptanceDateTime": ["2026-08-01T16:00:00Z"], "form": ["8-K"]}'
    fake_hash = "f" * 64

    with pytest.raises(ValueError, match="PAGINATION_DIGEST_MISMATCH"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-01T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            pagination_artifacts={"page-001.json": page_bytes},
            accession_inventory=[],
            source_hashes={"page-001.json": fake_hash},
        )


def test_coverage_receipt_binds_historical_manifest_and_revalidates_on_read(db):
    """
    QUALITY GATE:
    record_issuer_coverage_receipt binds the historical artifact manifest into the receipt envelope,
    and get_authoritative_receipt_for_interval revalidates it on read.
    """
    conn, repo, pipeline = db

    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [{
                "name": "page-001.json",
                "filingCount": 100,
                "filingFrom": "2026-08-01",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page_content = {
        "accessionNumber": ["0001045810-26-000099"],
        "filingDate": ["2026-08-01"],
        "acceptanceDateTime": ["2026-08-01T16:00:00Z"],
        "form": ["8-K"],
    }
    page_bytes = json.dumps(page_content).encode("utf-8")
    page_hash = hashlib.sha256(page_bytes).hexdigest()

    # 1. Record receipt with valid pinned historical manifest
    r_id = repo.record_issuer_coverage_receipt(
        ticker="NVDA",
        issuer_cik="0001045810",
        interval_start="2026-08-01T00:00:00Z",
        interval_end="2026-11-17T23:59:59Z",
        certified_watermark="2026-11-17T23:59:59Z",
        accession_inventory=[],
        source_hashes={"page-001.json": page_hash},
        coverage_status="COMPLETE",
        zero_activity_proof={
            "catalog_bytes": cat_bytes,
            "catalog_sha256": cat_hash,
            "pagination_artifacts": {"page-001.json": page_bytes},
            "historical_manifest": {"page-001.json": page_hash},
        },
        conn=conn,
    )
    assert r_id is not None

    # 2. Read authoritative receipt -> verifies historical_manifest revalidation on read
    auth_receipt = repo.get_authoritative_receipt_for_interval(
        ticker="NVDA",
        interval_start="2026-08-01T00:00:00Z",
        interval_end="2026-11-17T23:59:59Z",
        certified_watermark="2026-11-17T23:59:59Z",
        conn=conn,
    )
    assert auth_receipt is not None
    assert auth_receipt["receipt_id"] == r_id
    z_proof = auth_receipt.get("structured_metadata", {}).get("zero_activity_proof", {})
    assert z_proof.get("historical_manifest") == {"page-001.json": page_hash}


def test_artifact_conflicting_bytes_and_content_rejected():
    """
    QUALITY GATE:
    An artifact supplying verified bytes containing a Form 4, but supplying an alternate
    'content' field omitting that Form 4, must be rejected with ARTIFACT_PAYLOAD_MISMATCH.
    """
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [{
                "name": "page-001.json",
                "filingCount": 100,
                "filingFrom": "2026-08-15",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page_content = {
        "accessionNumber": ["0001045810-26-000099"],
        "filingDate": ["2026-08-15"],
        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
        "form": ["4"],
    }
    page_bytes = json.dumps(page_content).encode("utf-8")
    page_hash = hashlib.sha256(page_bytes).hexdigest()

    deceptive_content = {
        "accessionNumber": ["0001045810-26-000088"],
        "filingDate": ["2026-08-15"],
        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
        "form": ["8-K"],
    }

    artifact = {
        "bytes": page_bytes,
        "sha256": page_hash,
        "content": deceptive_content,
    }

    with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD_MISMATCH.*conflicting 'content' field"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-15T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            pagination_artifacts={"page-001.json": artifact},
            accession_inventory=[],
        )


def test_artifact_conflicting_bytes_and_data_rejected():
    """
    QUALITY GATE:
    An artifact supplying verified bytes containing a Form 4, but supplying an alternate
    'data' field omitting that Form 4, must be rejected with ARTIFACT_PAYLOAD_MISMATCH.
    """
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [{
                "name": "page-001.json",
                "filingCount": 100,
                "filingFrom": "2026-08-15",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page_content = {
        "accessionNumber": ["0001045810-26-000099"],
        "filingDate": ["2026-08-15"],
        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
        "form": ["4"],
    }
    page_bytes = json.dumps(page_content).encode("utf-8")
    page_hash = hashlib.sha256(page_bytes).hexdigest()

    deceptive_data = {
        "accessionNumber": ["0001045810-26-000088"],
        "filingDate": ["2026-08-15"],
        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
        "form": ["8-K"],
    }

    artifact = {
        "bytes": page_bytes,
        "sha256": page_hash,
        "data": deceptive_data,
    }

    with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD_MISMATCH.*conflicting 'data' field"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-15T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            pagination_artifacts={"page-001.json": artifact},
            accession_inventory=[],
        )


def test_artifact_conflicting_bytes_and_recent_rejected():
    """
    QUALITY GATE:
    An artifact supplying verified bytes containing a Form 4, but supplying an alternate
    'recent' field omitting that Form 4, must be rejected with ARTIFACT_PAYLOAD_MISMATCH.
    """
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [{
                "name": "page-001.json",
                "filingCount": 100,
                "filingFrom": "2026-08-15",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page_content = {
        "accessionNumber": ["0001045810-26-000099"],
        "filingDate": ["2026-08-15"],
        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
        "form": ["4"],
    }
    page_bytes = json.dumps(page_content).encode("utf-8")
    page_hash = hashlib.sha256(page_bytes).hexdigest()

    deceptive_recent = {
        "accessionNumber": ["0001045810-26-000088"],
        "filingDate": ["2026-08-15"],
        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
        "form": ["8-K"],
    }

    artifact = {
        "bytes": page_bytes,
        "sha256": page_hash,
        "recent": deceptive_recent,
    }

    with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD_MISMATCH.*conflicting 'recent' field"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-15T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            pagination_artifacts={"page-001.json": artifact},
            accession_inventory=[],
        )


def test_artifact_bytes_parsed_exclusively_eligible_filing_cannot_disappear():
    """
    QUALITY GATE:
    Eligible filings in verified artifact bytes are parsed exclusively from those bytes
    and cannot disappear. An empty caller inventory is rejected with CALLER_INVENTORY_OMISSION.
    """
    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [{
                "name": "page-001.json",
                "filingCount": 100,
                "filingFrom": "2026-08-15",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page_content = {
        "accessionNumber": ["0001045810-26-000099"],
        "filingDate": ["2026-08-15"],
        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
        "form": ["4"],
    }
    page_bytes = json.dumps(page_content).encode("utf-8")
    page_hash = hashlib.sha256(page_bytes).hexdigest()

    artifact = {
        "bytes": page_bytes,
        "sha256": page_hash,
    }

    with pytest.raises(ValueError, match="CALLER_INVENTORY_OMISSION.*0001045810-26-000099"):
        validate_catalog_evidence(
            catalog_bytes=cat_bytes,
            catalog_sha256=cat_hash,
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-15T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            pagination_artifacts={"page-001.json": artifact},
            accession_inventory=[],
        )


def test_receipt_read_revalidation_rejects_conflicting_artifact_payload(db):
    """
    QUALITY GATE:
    Receipt read revalidation in get_authoritative_receipt_for_interval applies the exact
    same artifact payload integrity rule, failing closed on conflicting fields.
    """
    conn, repo, pipeline = db

    cat = {
        "cik": "0001045810",
        "entityType": "operating",
        "name": "NVIDIA CORP",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000050"],
                "filingDate": ["2026-10-15"],
                "reportDate": ["2026-10-15"],
                "acceptanceDateTime": ["2026-10-15T16:00:00Z"],
                "form": ["8-K"],
            },
            "files": [{
                "name": "page-001.json",
                "filingCount": 100,
                "filingFrom": "2026-08-15",
                "filingTo": "2026-10-31",
            }],
        },
    }
    cat_bytes = json.dumps(cat).encode("utf-8")
    cat_hash = hashlib.sha256(cat_bytes).hexdigest()

    page_content = {
        "accessionNumber": ["0001045810-26-000099"],
        "filingDate": ["2026-08-15"],
        "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
        "form": ["8-K"],
    }
    page_bytes = json.dumps(page_content).encode("utf-8")
    page_hash = hashlib.sha256(page_bytes).hexdigest()

    conflicting_artifact = {
        "bytes": page_bytes,
        "sha256": page_hash,
        "content": {
            "accessionNumber": ["0001045810-26-000099"],
            "filingDate": ["2026-08-15"],
            "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
            "form": ["4"],
        },
    }

    # Recording with conflicting artifact fails closed on write:
    with pytest.raises(RuntimeError, match="AUTHORITATIVE_CATALOG_PROOF_FAILED.*ARTIFACT_PAYLOAD_MISMATCH"):
        repo.record_issuer_coverage_receipt(
            ticker="NVDA",
            issuer_cik="0001045810",
            interval_start="2026-08-15T00:00:00Z",
            interval_end="2026-11-17T23:59:59Z",
            certified_watermark="2026-11-17T23:59:59Z",
            accession_inventory=[],
            source_hashes={"page-001.json": page_hash},
            coverage_status="COMPLETE",
            zero_activity_proof={
                "catalog_bytes": cat_bytes,
                "catalog_sha256": cat_hash,
                "pagination_artifacts": {"page-001.json": conflicting_artifact},
                "historical_manifest": {"page-001.json": page_hash},
            },
            conn=conn,
        )

    # Read revalidation of a corrupt record in DB fails closed:
    from psycopg2.extras import Json
    fake_id = str(uuid.uuid4())
    conflicting_artifact_str = {
        "bytes": page_bytes.decode("utf-8"),
        "sha256": page_hash,
        "content": {
            "accessionNumber": ["0001045810-26-000099"],
            "filingDate": ["2026-08-15"],
            "acceptanceDateTime": ["2026-08-15T16:00:00Z"],
            "form": ["4"],
        },
    }
    fake_reasons = {
        "ticker": "NVDA",
        "issuer_cik": "0001045810",
        "covered_tickers": ["NVDA"],
        "covered_ciks": ["0001045810"],
        "accession_inventory": [],
        "source_hashes": {"page-001.json": page_hash},
        "material_gaps": [],
        "public_availability_cutoff": "2026-11-17T23:59:59Z",
        "zero_activity_proof": {
            "catalog_bytes": cat_bytes.decode("utf-8"),
            "catalog_sha256": cat_hash,
            "pagination_artifacts": {"page-001.json": conflicting_artifact_str},
            "historical_manifest": {"page-001.json": page_hash},
        },
    }
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sm_reconciliation_receipts (
                receipt_id, interval_start, interval_end, certified_watermark,
                receipt_version, supersedes_receipt_id, expected_count, accounted_count,
                missing_count, material_gap_count, coverage_status, failure_reasons,
                reconciliation_hash, release_sha
            ) VALUES (%s, %s, %s, %s, 1, NULL, 0, 0, 0, 0, 'COMPLETE', %s, 'fakehash', 'release')
        """, (
            fake_id,
            "2026-08-15T00:00:00Z",
            "2026-11-17T23:59:59Z",
            "2026-11-17T23:59:59Z",
            Json(fake_reasons),
        ))

    auth_receipt = repo.get_authoritative_receipt_for_interval(
        ticker="NVDA",
        interval_start="2026-08-15T00:00:00Z",
        interval_end="2026-11-17T23:59:59Z",
        certified_watermark="2026-11-17T23:59:59Z",
        conn=conn,
    )
    assert auth_receipt is None






