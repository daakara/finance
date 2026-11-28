"""Regressions for the independently demonstrated bounded-ingestion defects."""
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from test_sec_bounded_resumable_ingestion import replay_pg, authentic_aapl_filing, synthetic_nvda_filing
from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline, DEFAULT_GOVERNED_ISSUERS

ROOT = Path(__file__).resolve().parents[1]
AAPL = {'ticker':'AAPL','cik':'0000320193','company_name':'APPLE INC'}
NVDA = {'ticker':'NVDA','cik':'0001045810','company_name':'NVIDIA CORP'}
MSFT = {'ticker':'MSFT','cik':'0000789019','company_name':'MICROSOFT CORP'}

@pytest.fixture(autouse=True)
def offline_env(monkeypatch, pytestconfig):
    if not pytestconfig.pluginmanager.hasplugin('sec_offline_test_support'):
        pytest.skip('Requires loopback-only offline support')
    monkeypatch.setenv('ARX_SEC_EGRESS_ROLE','OWNER')
    monkeypatch.delenv('ARX_CONTINUOUS_AUTH_SCOPE',raising=False)
    monkeypatch.delenv('ARX_ENVIRONMENT',raising=False)

def pipeline(db, issuers=None):
    return SmartMoneyContinuousPipeline(dsn=db[2],repository=db[1],issuers=issuers or [AAPL],ingestion_only=True,allow_test_mode=True)

@pytest.mark.parametrize('failure',['deadline','owner','grant','revocation','receipt_delay'])
def test_every_unit_commit_is_fenced(replay_pg,monkeypatch,failure):
    p=pipeline(replay_pg)
    p.cycle_deadline_monotonic=time.monotonic()+60
    def fetch(**kw):
        if failure=='deadline':p.cycle_deadline_monotonic=time.monotonic()-1
        if failure=='owner':
            p.allow_test_mode=False
            monkeypatch.setattr('analyst_dashboard.smart_money.continuous_pipeline.require_continuous_authorization',lambda **kw:None)
            p._require_verified_owner_lock=MagicMock(side_effect=RuntimeError('OWNER_LOST'))
        if failure=='grant':
            p.allow_test_mode=False
            monkeypatch.setattr('analyst_dashboard.smart_money.continuous_pipeline.require_continuous_authorization',MagicMock(side_effect=RuntimeError('GRANT_EXPIRED')))
        if failure=='revocation':monkeypatch.setenv('ARX_SEC_EGRESS_ROLE','DISABLED')
        return [authentic_aapl_filing()]
    p._fetch_filings_from_sec=fetch
    if failure=='receipt_delay':
        original=replay_pg[1].save_reconciliation_receipt
        def save(*a,**kw):
            result=original(*a,**kw)
            p.cycle_deadline_monotonic=time.monotonic()-1
            return result
        monkeypatch.setattr(replay_pg[1],'save_reconciliation_receipt',save)
    with pytest.raises(RuntimeError):p._execute_bounded_cohort_ingestion('fenced-unit')
    with replay_pg[0].cursor() as cur:
        for table in ('form4_raw_filings','form4_parsed_transactions','sm_source_availability_ledger','sm_reconciliation_receipts'):
            cur.execute(f'SELECT count(*) FROM {table}')
            assert cur.fetchone()[0]==0

@pytest.mark.parametrize('changed',['trial','slot','execution','binding','watermark','cik','release'])
def test_resume_requires_exact_identity(replay_pg,changed):
    p=pipeline(replay_pg);p.current_slot_id='slot-1';p.trial_scheduler=SimpleNamespace(trial_id='trial-1')
    p._fetch_filings_from_sec=lambda **kw:[authentic_aapl_filing()]
    p._execute_bounded_cohort_ingestion('cycle-1',as_of='2026-11-21T12:00:00Z')
    from analyst_dashboard.smart_money.continuous_authorization import resolve_release_identity
    args=dict(ticker='AAPL',trial_id='trial-1',slot_id='slot-1',execution_id='cycle-1',binding_hash=p._ingestion_binding_hash(),issuer_cik=AAPL['cik'],release_sha=resolve_release_identity(),watermark='2026-11-21T12:00:00Z')
    assert replay_pg[1].get_completed_unit_receipt(**args)
    key={'trial':'trial_id','slot':'slot_id','execution':'execution_id','binding':'binding_hash','watermark':'watermark','cik':'issuer_cik','release':'release_sha'}[changed]
    args[key]={'watermark':'2026-11-22T12:00:00Z','release':'a'*40}.get(changed,'different')
    assert replay_pg[1].get_completed_unit_receipt(**args) is None

def test_resume_lookup_error_prevents_egress(replay_pg,monkeypatch):
    p=pipeline(replay_pg);p._fetch_filings_from_sec=MagicMock()
    monkeypatch.setattr(replay_pg[1],'get_completed_unit_receipt',MagicMock(side_effect=RuntimeError('database unavailable')))
    with pytest.raises(RuntimeError,match='UNIT_RESUME_LOOKUP_FAILED'):p._execute_bounded_cohort_ingestion('db-failure')
    p._fetch_filings_from_sec.assert_not_called()

def test_unresolved_amendment_gaps_survive_resume(replay_pg):
    p=pipeline(replay_pg,[NVDA])
    p._fetch_filings_from_sec=lambda **kw:[synthetic_nvda_filing(form='4/A',original='2026-09-11')]
    first=p._execute_bounded_cohort_ingestion('gap-cycle',as_of='2026-11-21T12:00:00Z')
    p._fetch_filings_from_sec=MagicMock(side_effect=AssertionError('duplicate wire'))
    second=p._execute_bounded_cohort_ingestion('gap-cycle',as_of='2026-11-21T12:00:00Z')
    assert not first['all_issuers_completed'] and not second['all_issuers_completed']
    assert second['receipt_meta']['material_gap_count']>=1
    assert second['receipt_meta']['expected_count']==1
    p._fetch_filings_from_sec.assert_not_called()

def test_full_cycle_preserves_partial_and_queue_failure(replay_pg):
    conn,repo,dsn=replay_pg
    request=str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sm_run_requests(request_id,request_type,requested_by_subject_hash,status,lease_token) VALUES (%s,'OPERATOR_INGESTION','offline','RUNNING','lease')",(request,))
    conn.commit()
    p=pipeline(replay_pg,[AAPL,NVDA,MSFT])
    for name in ('_step1_preflight','_step2_claim_advisory_lock','_step5_freshness_gate_check','_cleanup_lock'):
        setattr(p,name,MagicMock())
    p._step3_check_durable_date_eligibility_under_lock=MagicMock(return_value=(True,'offline'))
    p._step4_pre_backup=MagicMock(return_value={});p._step9_post_backup=MagicMock(return_value={})
    def fetch(ticker,**kw):
        if ticker=='AAPL':return []
        raise RuntimeError('upstream failure')
    p._fetch_filings_from_sec=fetch
    result=p.execute_cycle(run_request_id=request,lease_token='lease')
    assert result['status']=='PARTIAL'
    row=repo.get_run_request(request)
    assert row['status']=='FAILED' and row['last_failure_code']=='PARTIAL_INGESTION'

def test_expanded_budget_survives_new_repository_and_renewal(replay_pg):
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
    repo=replay_pg[1];day=datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    r=repo.reserve_expanded_ingestion_budget('OFFLINE_PHASE','execution-1','binding',day,'2026-11-21T12:00:00Z')
    assert r['total_intents']==0
    for _ in range(4):repo.reserve_expanded_transmission('OFFLINE_PHASE','execution-1','binding','AAPL')
    fresh=PostgresSmartMoneyRepository(dsn=replay_pg[2])
    resumed=fresh.reserve_expanded_ingestion_budget('OFFLINE_PHASE','execution-1','binding',day,'2026-11-21T13:00:00Z')
    assert resumed['total_intents']==4 and resumed['issuer_intents']['AAPL']==4
    assert resumed['watermark']=='2026-11-21T12:00:00Z'
    with pytest.raises(RuntimeError,match='TRANSMISSION_LIMIT'):
        fresh.reserve_expanded_transmission('OFFLINE_PHASE','execution-1','binding','AAPL')
    with pytest.raises(RuntimeError,match='QUOTA_EXCEEDED'):
        fresh.reserve_expanded_ingestion_budget('OFFLINE_PHASE','execution-2','binding',day,'2026-11-21T13:00:00Z')
    fresh.reserve_expanded_ingestion_budget('OFFLINE_PHASE','execution-2','binding','2099-01-01','2099-01-01T12:00:00Z')
    with pytest.raises(RuntimeError,match='QUOTA_EXCEEDED'):
        fresh.reserve_expanded_ingestion_budget('OFFLINE_PHASE','execution-3','binding','2099-01-02','2099-01-02T12:00:00Z')


def test_real_worker_process_exit_and_fresh_worker_resume(replay_pg,tmp_path):
    """Actual process death, real queue lease recovery, fresh worker and durable sends."""
    conn,repo,dsn=replay_pg
    request=str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sm_run_requests(request_id,request_type,requested_by_subject_hash,status,max_attempts) VALUES (%s,'OPERATOR_INGESTION','crash-recovery','PENDING',3)",(request,))
    conn.commit()
    trace=tmp_path/'fetches.txt'
    child=tmp_path/'worker_child.py'
    child.write_text(r'''
import os,sys,time,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import MagicMock
import sec_offline_test_support
sec_offline_test_support.pytest_configure(None)
import psycopg2
from analyst_dashboard.smart_money import worker as module
from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline,DEFAULT_GOVERNED_ISSUERS
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.storage.postgres_coordination import acquire_sec_owner_lock
from test_sec_bounded_resumable_ingestion import authentic_aapl_filing
now=datetime.now(timezone.utc)
os.environ.update(ARX_SEC_EGRESS_ROLE='OWNER',ARX_CONTINUOUS_INGESTION_AUTHORIZED='true',
    ARX_SMART_MONEY_INGESTION_ONLY='true',ARX_INGESTION_PHASE_ID='OFFLINE_PROCESS_PHASE',
    ARX_INGESTION_DATES='2026-11-25,2026-11-27',
    ARX_CONTINUOUS_AUTH_SCOPE='RECURRING_10_ISSUER_FORM4_INGESTION_ONLY',
    ARX_CONTINUOUS_AUTH_UNIVERSE_ID='UB-US-EQUITIES-20261025-001',
    ARX_RELEASE_SHA='a'*40,RAILWAY_GIT_COMMIT_SHA='a'*40,ARX_CONTINUOUS_AUTH_RELEASE_SHA='a'*40,
    ARX_CONTINUOUS_AUTH_ID=('b' if sys.argv[2]=='crash' else 'c')*32,
    ARX_CONTINUOUS_AUTH_ISSUED_AT=(now-timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
    ARX_CONTINUOUS_AUTH_EXPIRES_AT=(now+timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ'))
cohort=sorted(DEFAULT_GOVERNED_ISSUERS,key=lambda i:(i['ticker']!='AAPL',i['ticker']!='NVDA',i['ticker']))
w=module.SmartMoneyIngestionWorker.__new__(module.SmartMoneyIngestionWorker)
w.repository=PostgresSmartMoneyRepository(dsn=sys.argv[1]);w.dsn=sys.argv[1]
w._owner_conn=psycopg2.connect(w.dsn)
assert acquire_sec_owner_lock(w._owner_conn)
w._owner_lock_acquired=True;w.worker_id='offline-'+sys.argv[2];w.lease_duration_sec=60
w._last_stale_recovery_at=time.time();w.stale_recovery_interval_sec=30
w._shutdown_requested=False;w._is_executing_cycle=False
w.containment_gate=MagicMock();w.containment_gate.check_ingestion_eligibility.return_value=True
w.governed_issuers=cohort;w.ingestion_only=True

def build(**kw):
    p=SmartMoneyContinuousPipeline(**kw)
    p._step1_preflight=MagicMock()
    p._step3_check_durable_date_eligibility_under_lock=MagicMock(return_value=(True,'OFFLINE_SCHEDULE'))
    p._step4_pre_backup=MagicMock(return_value={});p._step5_freshness_gate_check=MagicMock()
    p._step9_post_backup=MagicMock(return_value={})
    def fetch(ticker,**kwargs):
        with open(sys.argv[3],'a') as f:f.write(ticker+'\n')
        # Deliberately simulated network delivery; durable admission and commit are real.
        for _ in range(2 if ticker=='AAPL' else 1):
            p._reserve_expanded_send(ticker)
            p.outbound_session.total_transmissions+=1
            p.outbound_session.issuer_transmissions[ticker]=p.outbound_session.issuer_transmissions.get(ticker,0)+1
        if ticker=='NVDA' and sys.argv[2]=='crash':os._exit(73)
        return [authentic_aapl_filing()] if ticker=='AAPL' else []
    p._fetch_filings_from_sec=fetch
    return p
module.SmartMoneyContinuousPipeline=build
try:
    assert w.process_run_requests()==1
finally:
    w._owner_conn.close()
''')
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'tests'),str(ROOT/'tools/qualification'),str(ROOT)]))
    first=subprocess.run([sys.executable,str(child),dsn,'crash',str(trace)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert first.returncode==73,first.stdout+first.stderr
    row=repo.get_run_request(request)
    assert row['status']=='RUNNING' and row['attempt_count']==1
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_raw_filings WHERE ticker='AAPL'");assert cur.fetchone()[0]==1
        cur.execute("UPDATE sm_run_requests SET lease_expires_at=NOW()-interval '1 second' WHERE request_id=%s",(request,))
    conn.commit()
    assert repo.recover_stale_run_requests(backoff_sec=0)==1
    second=subprocess.run([sys.executable,str(child),dsn,'resume',str(trace)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert second.returncode==0,second.stdout+second.stderr
    row=repo.get_run_request(request)
    assert row['status']=='COMPLETED' and row['attempt_count']==2
    calls=trace.read_text().splitlines()
    assert calls.count('AAPL')==1 and calls.count('NVDA')==2
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM form4_parsed_transactions WHERE accession_number='0001140361-26-044773'");assert cur.fetchone()[0]==3
        cur.execute("SELECT payload FROM sm_audit_events WHERE event_type='EXPANDED_INGESTION_BUDGET'")
        events=[r[0] for r in cur.fetchall()]
    assert len([e for e in events if e['kind']=='RESERVATION'])==1
    assert sum(e.get('requests_reserved',0) for e in events)==40
    assert len([e for e in events if e['kind']=='SEND_INTENT' and e['issuer']=='AAPL'])==2
    assert len([e for e in events if e['kind']=='SEND_INTENT'])==12


def test_failed_wire_attempt_is_durably_charged(replay_pg,monkeypatch):
    import requests
    from unittest.mock import patch
    monkeypatch.setenv('ARX_CONTINUOUS_AUTH_SCOPE','RECURRING_10_ISSUER_FORM4_INGESTION_ONLY')
    monkeypatch.setenv('ARX_INGESTION_PHASE_ID','OFFLINE_WIRE_PHASE')
    p=pipeline(replay_pg,DEFAULT_GOVERNED_ISSUERS);p.execution_id='wire-execution'
    day=datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    repo=replay_pg[1];binding=p._ingestion_binding_hash()
    repo.reserve_expanded_ingestion_budget(p.expanded_phase_id,p.execution_id,binding,day,'2026-11-21T12:00:00Z')
    session=p.outbound_session;session.before_transmission=p._reserve_expanded_send;session.set_current_issuer('AAPL')
    request=requests.Request('GET','https://data.sec.gov/submissions/CIK0000320193.json').prepare()
    with patch.object(requests.Session,'send',side_effect=requests.exceptions.ConnectionError('offline injected failure')):
        with pytest.raises(requests.exceptions.ConnectionError):session.send(request)
    restored=repo.reserve_expanded_ingestion_budget(p.expanded_phase_id,p.execution_id,binding,day,'2026-11-21T13:00:00Z')
    assert restored['total_intents']==1 and restored['issuer_intents']=={'AAPL':1}
    assert session.total_transmissions==1


def test_tampered_resume_metadata_is_rejected(replay_pg):
    p=pipeline(replay_pg);p._fetch_filings_from_sec=lambda **kw:[authentic_aapl_filing()]
    p._execute_bounded_cohort_ingestion('sealed-cycle',as_of='2026-11-21T12:00:00Z')
    conn,repo,dsn=replay_pg
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM sm_reconciliation_receipts WHERE failure_reasons->>'unit_ticker'='AAPL'")
        names=[d[0] for d in cur.description];forged=dict(zip(names,cur.fetchone()))
    forged['receipt_id']=str(uuid.uuid4());forged['failure_reasons']['parsed_transaction_count']=999
    repo.save_reconciliation_receipt(forged)
    p._fetch_filings_from_sec=MagicMock(side_effect=AssertionError('must not transmit'))
    with pytest.raises(RuntimeError,match='UNIT_RESUME_LOOKUP_FAILED'):
        p._execute_bounded_cohort_ingestion('sealed-cycle',as_of='2026-11-21T12:00:00Z')
    p._fetch_filings_from_sec.assert_not_called()


def test_terminal_db_overrun_is_not_success(replay_pg,monkeypatch):
    conn,repo,dsn=replay_pg;request=str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sm_run_requests(request_id,request_type,requested_by_subject_hash,status,lease_token) VALUES (%s,'OPERATOR_INGESTION','deadline','RUNNING','lease')",(request,))
    conn.commit()
    p=pipeline(replay_pg,[AAPL,NVDA,MSFT])
    for name in ('_step1_preflight','_step2_claim_advisory_lock','_step5_freshness_gate_check','_cleanup_lock'):
        setattr(p,name,MagicMock())
    p._step3_check_durable_date_eligibility_under_lock=MagicMock(return_value=(True,'offline'))
    p._step4_pre_backup=MagicMock(return_value={});p._step9_post_backup=MagicMock(return_value={})
    p._fetch_filings_from_sec=lambda **kw:[]
    original=repo.complete_run_request
    def slow_complete(*a,**kw):
        original(*a,**kw)
        p.cycle_deadline_monotonic=time.monotonic()-1
    monkeypatch.setattr(repo,'complete_run_request',slow_complete)
    result=p.execute_cycle(run_request_id=request,lease_token='lease')
    assert result['status']=='DEGRADED_DEADLINE_EXCEEDED' and not result['deadline_compliant']
    assert repo.get_run_request(request)['status']=='FAILED'


@pytest.mark.parametrize('dry_run',[False,True])
def test_skip_and_dry_run_do_not_double_complete(replay_pg,dry_run):
    conn,repo,dsn=replay_pg;request=str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sm_run_requests(request_id,request_type,requested_by_subject_hash,status,lease_token) VALUES (%s,'OPERATOR_INGESTION','skip','RUNNING','lease')",(request,))
    conn.commit()
    p=pipeline(replay_pg,[AAPL,NVDA,MSFT])
    p._step1_preflight=MagicMock();p._step2_claim_advisory_lock=MagicMock();p._cleanup_lock=MagicMock()
    p._step3_check_durable_date_eligibility_under_lock=MagicMock(return_value=(False,'OUTSIDE_SCHEDULE'))
    result=p.execute_cycle(run_request_id=request,lease_token='lease',dry_run=dry_run)
    assert result['status']==('DRY_RUN_SUCCESS' if dry_run else 'SKIPPED_NOT_ELIGIBLE')
    assert repo.get_run_request(request)['status']==('RUNNING' if dry_run else 'SKIPPED_NOT_ELIGIBLE')
