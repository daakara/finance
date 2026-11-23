"""Isolated source-backed canary. Retained-response adapter makes no HTTP calls."""
import os,json,hashlib,sqlite3,tempfile
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
import xml.etree.ElementTree as ET
import sqlite3


P=Path(__file__).parent
DB=Path(tempfile.mkdtemp(prefix='isolated-replay-',dir=P))/'canary.sqlite'
ROOT=Path('/workspace/scratch/5f4195e7ee3f/arx-asset-discovery')
os.environ['ARX_PROSPECTIVE_START_UTC']='2026-11-23T00:00:00Z'
os.environ['ARX_RELEASE_SHA']='62c19395836de83aecb92da2381fbbf392d9f046'
os.environ['ARX_SMART_MONEY_DB']=str(P/'unused_local_cache.db')
from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline
from analyst_dashboard.smart_money.form4_ingestion import Form4IngestionManager


from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
ACC='0001570382-26-000015'; CIK='0001064728'
raw=(P/'peabody_0001570382-26-000015.xml').read_bytes();catalog=(P/'submissions.json').read_bytes()
for filename,content in [('custody.json',raw),('catalog_custody.json',catalog)]:
 meta=json.loads((P/filename).read_text());assert meta['status']==200 and meta['sha256']==hashlib.sha256(content).hexdigest() and meta['bytes']==len(content)
xml=raw.decode('utf-8'); assert xml.encode('utf-8')==raw
root=ET.fromstring(raw)
assert root.findtext('issuer/issuerCik').zfill(10)==CIK
assert root.findtext('issuer/issuerTradingSymbol')=='BTU'
assert root.findtext('documentType')=='4'
source_transactions=[]
for table,node in [('nonDerivativeTable','nonDerivativeTransaction'),('derivativeTable','derivativeTransaction')]:
 for tx in root.findall(f'{table}/{node}'):
  source_transactions.append({'code':tx.findtext('transactionCoding/transactionCode'),'date':tx.findtext('transactionDate/value'),'shares':float(tx.findtext('transactionAmounts/transactionShares/value')),'price':float(tx.findtext('transactionAmounts/transactionPricePerShare/value')),'derivative':table=='derivativeTable'})
assert source_transactions
class Response:
 status_code=200
 def __init__(self,body):self.text=body.decode('utf-8');self.content=body
 def json(self):return json.loads(self.content)
class RetainedResponses:
 def __init__(self):self.headers={};self.calls=[]
 def get(self,url,**kwargs):
  self.calls.append(url)
  if url=='https://data.sec.gov/submissions/CIK0001064728.json':return Response(catalog)
  if url=='https://www.sec.gov/Archives/edgar/data/1064728/000157038226000015/xslF345X06/wk-form4_1795438862.xml':return Response((P/'primary_response.raw').read_bytes())
  if url=='https://www.sec.gov/Archives/edgar/data/1064728/000157038226000015/wk-form4_1795438862.xml':return Response(raw)
  raise AssertionError('Unretained request blocked: '+url)

def no_http(*args,**kwargs):raise AssertionError('HTTP prohibited during replay')
from unittest.mock import MagicMock
with patch('requests.sessions.Session.request',no_http):
 manager=Form4IngestionManager(db_path=str(DB),sec_fetcher=MagicMock(),rate_limiter=MagicMock())
 class LocalWatermark:
  def get_existing_accessions_for_issuer(self,ticker):
   with sqlite3.connect(DB) as cx:return [row[0] for row in cx.execute('SELECT accession_number FROM form4_raw_filings WHERE issuer_symbol=?',(ticker,))]
 pipeline=SmartMoneyContinuousPipeline.__new__(SmartMoneyContinuousPipeline)
 pipeline.repository=LocalWatermark();pipeline.outbound_session=RetainedResponses();pipeline.allow_test_mode=True;pipeline.is_trial_2_issuer=False
 with patch('analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent',return_value='ARXTerminal Research Akara.david26@gmail.com'):
  found=pipeline._fetch_filings_from_sec('BTU',CIK,'PEABODY ENERGY CORP',as_of='2026-11-23T20:30:02.764065Z')
  assert len(found)==1 and found[0]['accession_number']==ACC
  assert found[0]['content_hash']==hashlib.sha256(raw).hexdigest()
  f=found[0]
  parsed=Form4XmlParser.parse_filing(xml,ACC,filing_date_hint=f['filing_date'],acceptance_datetime=f['accepted_datetime'],source_url=f['source_url'])
  assert parsed.issuer_cik==CIK and parsed.issuer_symbol=='BTU'
  assert len(parsed.transactions)==len(source_transactions)>0
  expected_owners={x.findtext('reportingOwnerId/rptOwnerName') for x in root.findall('reportingOwner')}
  assert {tx.reporting_owner_name for tx in parsed.transactions}==expected_owners
  for a,b in zip(parsed.transactions,source_transactions):
   assert a.transaction_code==b['code'] and a.transaction_date==b['date'] and a.shares==b['shares'] and a.price_per_share==b['price'] and bool(a.is_derivative)==b['derivative']
  def ingest(content):return manager.ingest_accession(ACC,CIK,filing_date=f['filing_date'],acceptance_datetime=f['accepted_datetime'],raw_content=content,source_url=f['source_url'])
  first=ingest(xml);assert first['status']=='INGESTED_NEW'
  def census():
   with sqlite3.connect(DB) as cx:
    return {t:cx.execute('SELECT * FROM '+t+' ORDER BY 1').fetchall() for t in ['form4_raw_filings','form4_parsed_transactions','sm_source_availability_ledger','sm_amendments']}
  before=census();assert len(before['form4_raw_filings'])==1 and len(before['form4_parsed_transactions'])==len(source_transactions)
  replay_found=pipeline._fetch_filings_from_sec('BTU',CIK,'PEABODY ENERGY CORP',as_of='2026-11-23T20:30:02.764065Z');assert replay_found==[]
  replay_results=[ingest(xml)['status'] for _ in range(5)];assert set(replay_results)=={'KNOWN_IDEMPOTENT_NOOP'}
  assert census()==before
  rejected=False
  try:ingest(xml+'\n')
  except RuntimeError as e:rejected='SOURCE_INTEGRITY_CONFLICT' in str(e)
  assert rejected and census()==before
  result={'sourceCommit':os.environ['ARX_RELEASE_SHA'],'issuer':'BTU','issuerCik':CIK,'accession':ACC,'filingDate':f['filing_date'],'acceptanceUtc':f['accepted_datetime'],'xmlBytes':len(raw),'xmlSha256':hashlib.sha256(raw).hexdigest(),'sourceTransactions':source_transactions,'parsedTransactions':len(parsed.transactions),'reportingOwners':sorted({tx.reporting_owner_name for tx in parsed.transactions}),'signalTypes':sorted({str(tx.signal_type.value) for tx in parsed.transactions}),'discoveryFirstPass':len(found),'discoveryReplay':len(replay_found),'retainedAdapterCalls':pipeline.outbound_session.calls,'storageStateSha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'replayStateUnchanged':True,'identicalReplayCount':5,'conflictingHashRejected':rejected,'productionWrites':0,'networkDuringReplay':0,'postgresPersistence':'NOT_CHECKED_LOCAL_SOCKET_PERMISSION_DENIED','coverageStatus':'UNCERTIFIED','scoring':'HELD','scope':'ISOLATED_LOCAL_KNOWN_POSITIVE_NOT_PRODUCTION_ACTIVATION','limitation':'Actual pipeline discovery method replayed against retained HTTP responses and a local SQLite watermark adapter. Actual parser and SQLite ingestion primitive tested. PostgreSQL persistence and production HTTP authorization/scheduler are not exercised.'}
  (P/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
