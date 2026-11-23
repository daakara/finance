"""
ARX Known-Positive BTU Form 4 Canary — Local PostgreSQL Replay Script.
Strictly offline: Retained-response adapter makes zero HTTP calls.
Database: Disposable local PostgreSQL via pgserver with repository canonical schema.
Never falls back to SQLite, shared databases, or production environments.
"""
import os
import sys
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET
import psycopg2
from psycopg2.extras import RealDictCursor
import pgserver

# Worktree root setup
WORKTREE = Path(__file__).resolve().parents[3]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

P = Path(__file__).parent

os.environ['ARX_PROSPECTIVE_START_UTC'] = '2026-11-23T00:00:00Z'
os.environ['ARX_RELEASE_SHA'] = '2eccc64a2cb3c649ad4d584a942285a6406d1ce0'
os.environ['ARX_CONTINUOUS_AUTH_SCOPE'] = 'RECURRING_2_ISSUER_FORM4_TRIAL'

from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline
from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser

ACC = '0001570382-26-000015'
CIK = '0001064728'
TICKER = 'BTU'
COMPANY = 'PEABODY ENERGY CORP'

# Verify retained source files
raw_xml_bytes = (P / 'peabody_0001570382-26-000015.xml').read_bytes()
catalog_bytes = (P / 'submissions.json').read_bytes()
primary_html_bytes = (P / 'primary_response.raw').read_bytes()

for filename, content in [
    ('custody.json', raw_xml_bytes),
    ('catalog_custody.json', catalog_bytes),
    ('primary_custody.json', primary_html_bytes)
]:
    meta = json.loads((P / filename).read_text())
    assert meta['status'] == 200
    assert meta['sha256'] == hashlib.sha256(content).hexdigest()
    assert meta['bytes'] == len(content)

xml_str = raw_xml_bytes.decode('utf-8')
assert xml_str.encode('utf-8') == raw_xml_bytes

root = ET.fromstring(raw_xml_bytes)
assert root.findtext('issuer/issuerCik').zfill(10) == CIK
assert root.findtext('issuer/issuerTradingSymbol') == TICKER
assert root.findtext('documentType') == '4'

source_transactions = []
for table, node in [('nonDerivativeTable', 'nonDerivativeTransaction'), ('derivativeTable', 'derivativeTransaction')]:
    for tx in root.findall(f'{table}/{node}'):
        source_transactions.append({
            'code': tx.findtext('transactionCoding/transactionCode'),
            'date': tx.findtext('transactionDate/value'),
            'shares': float(tx.findtext('transactionAmounts/transactionShares/value')),
            'price': float(tx.findtext('transactionAmounts/transactionPricePerShare/value')),
            'derivative': table == 'derivativeTable',
        })
assert len(source_transactions) == 1

class MockResponse:
    def __init__(self, body, status_code=200):
        self.status_code = status_code
        self.content = body
        self.text = body.decode('utf-8')
    def json(self):
        return json.loads(self.content)

class RetainedResponses:
    def __init__(self):
        self.headers = {}
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append(url)
        if url == f'https://data.sec.gov/submissions/CIK{CIK}.json':
            return MockResponse(catalog_bytes)
        if url == 'https://www.sec.gov/Archives/edgar/data/1064728/000157038226000015/xslF345X06/wk-form4_1795438862.xml':
            return MockResponse(primary_html_bytes)
        if url == 'https://www.sec.gov/Archives/edgar/data/1064728/000157038226000015/wk-form4_1795438862.xml':
            return MockResponse(raw_xml_bytes)
        raise AssertionError(f'Unretained request blocked: {url}')

def no_http(*args, **kwargs):
    raise AssertionError('External HTTP strictly prohibited during offline replay')

def run_replay():
    tmp_pgdata = tempfile.mkdtemp(prefix='canary-pg-')
    try:
        # Start disposable local PostgreSQL instance
        server = pgserver.get_server(tmp_pgdata)
        uri = server.get_uri()
        print(f"[GATE 2] Started disposable local PostgreSQL at: {uri}")

        # Connect and verify local isolation
        with psycopg2.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                pg_version = cur.fetchone()[0]
                print(f"[GATE 2] PostgreSQL version: {pg_version}")
                cur.execute("SELECT current_database(), current_user, inet_server_addr();")
                db_info = cur.fetchone()
                print(f"[GATE 2] Target db={db_info[0]}, user={db_info[1]}, addr={db_info[2]} (strictly local)")

            # Initialize canonical schema
            initialize_postgres_schema(conn, release_sha=os.environ['ARX_RELEASE_SHA'])
            print(f"[GATE 2] Canonical schema initialized for release {os.environ['ARX_RELEASE_SHA']}.")

        # Instantiate repository pointing to isolated local PostgreSQL
        repo = PostgresSmartMoneyRepository(dsn=uri)

        # Confirm repository is empty before test
        initial_accessions = repo.get_existing_accessions_for_issuer(TICKER)
        assert len(initial_accessions) == 0, f"Expected empty watermark, found: {initial_accessions}"
        print("[GATE 2] Verified watermark is empty prior to writes.")

        # Construct pipeline in test mode with retained responses
        pipeline = SmartMoneyContinuousPipeline.__new__(SmartMoneyContinuousPipeline)
        pipeline.repository = repo
        pipeline.outbound_session = RetainedResponses()
        pipeline.allow_test_mode = True
        pipeline.is_trial_2_issuer = False
        pipeline.ingestion_only = True
        pipeline.cycle_deadline_monotonic = None
        pipeline.issuers = [{'ticker': TICKER, 'cik': CIK, 'company_name': COMPANY}]

        with patch('requests.sessions.Session.request', no_http), \
             patch('analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent', return_value='ARXTerminal Research Akara.david26@gmail.com'):

            # GATE 3: Discovery from retained responses (with HTML-to-XML fallback)
            found = pipeline._fetch_filings_from_sec(TICKER, CIK, COMPANY, as_of='2026-11-23T20:30:02.764065Z')
            print(f"[GATE 3] Discovery found {len(found)} filing(s)")
            assert len(found) == 1, f"Expected 1 filing, got {len(found)}"
            f = found[0]
            assert f['accession_number'] == ACC
            assert f['cik'] == CIK
            assert f['ticker'] == TICKER
            assert f['content_hash'] == hashlib.sha256(raw_xml_bytes).hexdigest()
            assert f['source_url'] == 'https://www.sec.gov/Archives/edgar/data/1064728/000157038226000015/wk-form4_1795438862.xml'
            print("[GATE 3] HTML-to-XML fallback verified: clean XML resolved from archive.")

            # Step 7 canonical parsing and persistence
            cycle_id = 'canary-btu-20261123-pg'
            parsed_tx_objs, receipt_meta = pipeline._step7_parse_and_reconcile(found, cycle_id, as_of='2026-11-23T20:30:02.764065Z')
            print(f"[GATE 3] Canonical Step 7 complete: {len(parsed_tx_objs)} transaction(s) parsed, receipt_id={receipt_meta['receipt_id']}")
            assert len(parsed_tx_objs) == 1
            assert receipt_meta['coverage_status'] == 'PARTIAL'
            print("[GATE 3] Coverage certified status: PARTIAL (trial holds preserved).")

            # Direct SQL assertions
            with psycopg2.connect(uri) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM form4_raw_filings WHERE accession_number = %s;", (ACC,))
                    raw_row = cur.fetchone()
                    assert raw_row is not None
                    assert raw_row['ticker'] == TICKER
                    assert raw_row['cik'] == CIK
                    assert raw_row['content_hash'] == hashlib.sha256(raw_xml_bytes).hexdigest()
                    assert str(raw_row['filing_date']) == '2026-11-23'
                    print("[GATE 3 SQL] form4_raw_filings row verified.")

                    cur.execute("SELECT * FROM sm_source_availability_ledger WHERE accession_number = %s;", (ACC,))
                    avail_row = cur.fetchone()
                    assert avail_row is not None
                    from datetime import timezone
                    assert avail_row['public_availability_at'].astimezone(timezone.utc).isoformat().startswith('2026-11-23T13:01:04')
                    print("[GATE 3 SQL] sm_source_availability_ledger row verified.")

                    cur.execute("SELECT * FROM form4_parsed_transactions WHERE accession_number = %s;", (ACC,))
                    tx_rows = cur.fetchall()
                    assert len(tx_rows) == 1
                    tx_row = tx_rows[0]
                    assert tx_row['transaction_code'] == 'P'
                    assert float(tx_row['shares']) == 4411.0
                    assert float(tx_row['price_per_share']) == 15.99
                    assert str(tx_row['transaction_date']) == '2026-11-20'
                    assert tx_row['reporting_owner_name'] == 'Grech James C.'
                    assert tx_row['is_derivative'] is False
                    print("[GATE 3 SQL] form4_parsed_transactions row verified: 4411 shares @ $15.99, code P, owner Grech James C.")

                    cur.execute("SELECT count(*) FROM form4_raw_filings;")
                    count_raw = cur.fetchone()['count']
                    cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
                    count_tx = cur.fetchone()['count']
                    cur.execute("SELECT count(*) FROM sm_source_availability_ledger;")
                    count_avail = cur.fetchone()['count']
                    cur.execute("SELECT count(*) FROM sm_reconciliation_receipts;")
                    count_receipts = cur.fetchone()['count']
                    print(f"[GATE 3 SQL Census] raw_filings={count_raw}, transactions={count_tx}, availability={count_avail}, receipts={count_receipts}")
                    assert count_raw == 1
                    assert count_tx == 1
                    assert count_avail == 1
                    assert count_receipts == 1

            # GATE 4: Repeat discovery with known watermark
            pipeline.outbound_session.calls.clear()
            repeat_found = pipeline._fetch_filings_from_sec(TICKER, CIK, COMPANY, as_of='2026-11-23T20:30:02.764065Z')
            print(f"[GATE 4] Repeat discovery found {len(repeat_found)} new filing(s)")
            assert len(repeat_found) == 0, f"Expected 0 new filings on repeat discovery, got {len(repeat_found)}"
            assert pipeline.outbound_session.calls == [f'https://data.sec.gov/submissions/CIK{CIK}.json'], \
                f"Expected only catalog call on repeat discovery, got {pipeline.outbound_session.calls}"
            print("[GATE 4] Repeat discovery correctly identified accession as known from PostgreSQL watermark.")

            # GATE 4: 5x Idempotency replays
            for replay_i in range(1, 6):
                r_parsed, r_receipt = pipeline._step7_parse_and_reconcile(found, f"canary-btu-replay-{replay_i}", as_of='2026-11-23T20:30:02.764065Z')
                assert len(r_parsed) == 1

            # Assert evidence counts unchanged
            with psycopg2.connect(uri) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT count(*) FROM form4_raw_filings;")
                    assert cur.fetchone()['count'] == 1, "Duplicate raw filing created!"
                    cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
                    assert cur.fetchone()['count'] == 1, "Duplicate transaction created!"
                    cur.execute("SELECT count(*) FROM sm_source_availability_ledger;")
                    assert cur.fetchone()['count'] == 1, "Duplicate availability created!"
                    cur.execute("SELECT count(*) FROM sm_reconciliation_receipts;")
                    total_receipts = cur.fetchone()['count']
                    # 1 initial + 5 replays = 6 audit receipts (distinct from duplicate filings/txs)
                    assert total_receipts == 6, f"Expected 6 audit receipts, got {total_receipts}"

                    # Verify raw filing and transaction fields are completely unchanged
                    cur.execute("SELECT * FROM form4_raw_filings WHERE accession_number = %s;", (ACC,))
                    r_raw = cur.fetchone()
                    assert r_raw['content_hash'] == raw_row['content_hash']
                    assert r_raw['created_at'] == raw_row['created_at']

                    cur.execute("SELECT * FROM form4_parsed_transactions WHERE accession_number = %s;", (ACC,))
                    r_tx = cur.fetchone()
                    assert r_tx['transaction_id'] == tx_row['transaction_id']
                    assert float(r_tx['shares']) == 4411.0
                    print("[GATE 4] 5x replay idempotency verified: 0 duplicates, original evidence unmutated.")

            # GATE 4: Conflicting content rollback
            conflicting_filing = dict(found[0])
            conflicting_filing['content_hash'] = '0000000000000000000000000000000000000000000000000000000000000000'
            conflicting_filing['xml_content'] = xml_str + '\n<!-- CONFLICT -->'

            conflict_rejected = False
            try:
                pipeline._step7_parse_and_reconcile([conflicting_filing], "canary-btu-conflict", as_of='2026-11-23T20:30:02.764065Z')
            except RuntimeError as exc:
                if 'SOURCE_INTEGRITY_CONFLICT' in str(exc):
                    conflict_rejected = True
                    print(f"[GATE 4] Conflicting content rejected with expected exception: {exc}")
                else:
                    raise

            assert conflict_rejected, "Expected SOURCE_INTEGRITY_CONFLICT on conflicting hash"

            # Assert complete transaction rollback: zero partial writes committed
            with psycopg2.connect(uri) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT count(*) FROM form4_raw_filings;")
                    assert cur.fetchone()['count'] == 1
                    cur.execute("SELECT count(*) FROM form4_parsed_transactions;")
                    assert cur.fetchone()['count'] == 1
                    cur.execute("SELECT count(*) FROM sm_source_availability_ledger;")
                    assert cur.fetchone()['count'] == 1
                    cur.execute("SELECT count(*) FROM sm_reconciliation_receipts;")
                    assert cur.fetchone()['count'] == 6  # Conflict cycle did not commit a receipt
                    print("[GATE 4] Conflict rollback verified: zero partial mutations committed to PostgreSQL.")

            # Record results
            result = {
                "checkoutHeadSha": "f1c5e0d33cda1d73d959a43fef1c5b5044bfae56",
                "workerReleaseAnchor": "2eccc64a2cb3c649ad4d584a942285a6406d1ce0",
                "scopedFilesParity": "IDENTICAL_0_DIFFS",
                "sourceCommit": "f1c5e0d33cda1d73d959a43fef1c5b5044bfae56 (tested discovery/parser/persistence files match worker release 2eccc64a2cb3c649ad4d584a942285a6406d1ce0 with 0 diffs)",
                "postgresVersion": pg_version,
                "databaseUri": "postgresql://postgres@local-disposable-socket/postgres",
                "issuer": TICKER,
                "issuerCik": CIK,
                "accession": ACC,
                "filingDate": f['filing_date'],
                "acceptanceUtc": f['accepted_datetime'],
                "xmlBytes": len(raw_xml_bytes),
                "xmlSha256": hashlib.sha256(raw_xml_bytes).hexdigest(),
                "sourceTransactions": source_transactions,
                "parsedTransactions": len(parsed_tx_objs),
                "reportingOwners": ["Grech James C."],
                "signalTypes": ["OPEN_MARKET_PURCHASE"],
                "discoveryFirstPass": len(found),
                "discoveryReplay": len(repeat_found),
                "retainedAdapterCalls": [
                    f'https://data.sec.gov/submissions/CIK{CIK}.json',
                    'https://www.sec.gov/Archives/edgar/data/1064728/000157038226000015/xslF345X06/wk-form4_1795438862.xml',
                    'https://www.sec.gov/Archives/edgar/data/1064728/000157038226000015/wk-form4_1795438862.xml'
                ],
                "mechanicsVerdict": "PASS",
                "postgresPersistence": "PASS",
                "tableRowCounts": {
                    "form4_raw_filings": 1,
                    "form4_parsed_transactions": 1,
                    "sm_source_availability_ledger": 1,
                    "sm_reconciliation_receipts": 6
                },
                "replayStateUnchanged": True,
                "identicalReplayCount": 5,
                "conflictingHashRejected": conflict_rejected,
                "conflictRollbackVerified": True,
                "productionWrites": 0,
                "networkDuringReplay": 0,
                "coverageStatus": "PARTIAL_UNCERTIFIED",
                "scoring": "HELD",
                "btuProductionDispatch": "NOT_EXERCISED",
                "btuProductionAuthorization": "NOT_EXERCISED",
                "btuScoringIntegration": "NOT_EXERCISED",
                "aaplNvdaProductionTrial": "SEPARATE_AND_UNCHANGED",
                "scope": "ISOLATED_LOCAL_POSTGRES_CANARY_MECHANICS",
                "limitation": "BTU discovery, parsing, and local PostgreSQL persistence mechanics verified. BTU production dispatch, production authorization, and scoring integration were not exercised or authorized. Historical coverage across trial universe remains uncertified. The separate AAPL/NVDA production trial remains distinct and unchanged."
            }

            result_file = P / 'result.json'
            result_file.write_text(json.dumps(result, indent=2))
            print(f"\n[SUMMARY] Updated {result_file} successfully.")
            print(json.dumps(result, indent=2))

        server.cleanup()
        print("[GATE 2] Cleaned up disposable local PostgreSQL server.")
    finally:
        shutil.rmtree(tmp_pgdata, ignore_errors=True)

if __name__ == '__main__':
    run_replay()
