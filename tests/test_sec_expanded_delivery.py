"""Bounded scheduled delivery, deduplication and consumption-time fencing."""
from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest
from test_sec_bounded_resumable_ingestion import replay_pg
from analyst_dashboard.smart_money.smart_money_scheduler import ExpandedIngestionSchedule
from analyst_dashboard.smart_money.worker import SmartMoneyIngestionWorker
from analyst_dashboard.smart_money.continuous_pipeline import DEFAULT_GOVERNED_ISSUERS
from analyst_dashboard.smart_money.continuous_authorization import require_continuous_authorization

CONTRACT = {'ARX_INGESTION_PHASE_ID':'OFFLINE_DELIVERY_PHASE', 'ARX_INGESTION_DATES':'2026-11-25,2026-11-27',
            'ARX_CONTINUOUS_AUTH_RELEASE_SHA':'a'*40, 'ARX_INGESTION_SCHEDULE_ENABLED':'true'}
DUE=datetime(2026,11,25,14,15,tzinfo=timezone.utc)

@pytest.fixture(autouse=True)
def offline(pytestconfig,monkeypatch):
    if not pytestconfig.pluginmanager.hasplugin('sec_offline_test_support'):
        pytest.skip('Requires loopback-only offline support')
    for k,v in CONTRACT.items():monkeypatch.setenv(k,v)

@pytest.mark.parametrize('raw',['','2026-11-25','2026-11-25,2026-11-25','2026-11-27,2026-11-25','2026-11-25,2026-11-26','2026-11-21,2026-11-23'])
def test_only_two_explicit_post_trial_dissemination_dates(raw):
    with pytest.raises(RuntimeError):ExpandedIngestionSchedule.from_environment(dict(CONTRACT,ARX_INGESTION_DATES=raw))

@pytest.mark.parametrize('stamp,due',[
 ('2026-11-25T14:14:59+00:00',False),('2026-11-25T14:15:00+00:00',True),
 ('2026-11-25T14:19:59+00:00',True),('2026-11-25T14:20:00+00:00',False),
 ('2026-11-26T14:15:00+00:00',False),('2026-11-27T14:15:00+00:00',True),
 ('2026-11-28T14:15:00+00:00',False)])
def test_window_is_narrow_and_has_no_catchup(stamp,due):
    assert ExpandedIngestionSchedule.from_environment(CONTRACT).evaluate(datetime.fromisoformat(stamp))[0] is due


def worker(db):
    w=SmartMoneyIngestionWorker.__new__(SmartMoneyIngestionWorker)
    w.repository=db[1];w.is_expanded_10_ingestion=True;w.is_trial_2_issuer=False
    w.governed_issuers=DEFAULT_GOVERNED_ISSUERS
    w._has_verified_owner_lock=MagicMock(return_value=True)
    w._last_stale_recovery_at=0;w.stale_recovery_interval_sec=30;w.worker_id='offline-delivery';w.lease_duration_sec=60
    w.containment_gate=MagicMock();w.containment_gate.check_ingestion_eligibility.return_value=True
    return w


def test_duplicate_polls_restart_and_terminal_failure_keep_one_job(replay_pg,monkeypatch):
    monkeypatch.setattr('analyst_dashboard.smart_money.worker.require_continuous_authorization',lambda **kw:None)
    monkeypatch.setattr(ExpandedIngestionSchedule,'get_ny_now',lambda self,now=None:DUE.astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')))
    w=worker(replay_pg);first=w.dispatch_expanded_slot()
    assert first and w.dispatch_expanded_slot()==first
    w2=worker(replay_pg);assert w2.dispatch_expanded_slot()==first
    row=replay_pg[1].claim_next_run_request('offline',60)
    replay_pg[1].complete_run_request(first,lease_token=row['lease_token'],status='FAILED',failure_code='INJECTED')
    assert w2.dispatch_expanded_slot()==first
    with replay_pg[0].cursor() as cur:
        cur.execute("SELECT count(*) FROM sm_run_requests");assert cur.fetchone()[0]==1
        cur.execute("SELECT count(*) FROM sm_audit_events WHERE event_type='EXPANDED_INGESTION_BUDGET'");assert cur.fetchone()[0]==0

@pytest.mark.parametrize('control',['disabled','owner_lost','expired_grant','outside_window'])
def test_dispatch_cannot_bypass_controls(replay_pg,monkeypatch,control):
    w=worker(replay_pg)
    monkeypatch.setattr('analyst_dashboard.smart_money.worker.require_continuous_authorization',lambda **kw:None)
    monkeypatch.setattr(ExpandedIngestionSchedule,'get_ny_now',lambda self,now=None:DUE)
    if control=='disabled':monkeypatch.setenv('ARX_INGESTION_SCHEDULE_ENABLED','false')
    if control=='owner_lost':w._has_verified_owner_lock.return_value=False
    if control=='expired_grant':
        monkeypatch.setattr('analyst_dashboard.smart_money.worker.require_continuous_authorization',MagicMock(side_effect=RuntimeError('EXPIRED')))
    if control=='outside_window':monkeypatch.setattr(ExpandedIngestionSchedule,'get_ny_now',lambda self,now=None:datetime(2026,11,25,16,tzinfo=timezone.utc))
    if control in ('expired_grant','owner_lost'):
        with pytest.raises(RuntimeError):w.dispatch_expanded_slot()
    else:assert w.dispatch_expanded_slot() is None
    with replay_pg[0].cursor() as cur:
        cur.execute('SELECT count(*) FROM sm_run_requests');assert cur.fetchone()[0]==0


def test_delayed_delivery_is_terminal_without_ingestion(replay_pg,monkeypatch):
    policy=ExpandedIngestionSchedule.from_environment(CONTRACT)
    request=replay_pg[1].enqueue_expanded_daily_request(policy.subject_for(DUE.date()))
    monkeypatch.setattr(ExpandedIngestionSchedule,'get_ny_now',lambda self,now=None:datetime(2026,11,25,16,tzinfo=timezone.utc))
    w=worker(replay_pg)
    count=w.process_run_requests()
    assert count==1
    row=replay_pg[1].get_run_request(request)
    assert row['status']=='SKIPPED_NOT_ELIGIBLE' and row['last_failure_code']=='EXPANDED_DELIVERY_EXPIRED'
    with replay_pg[0].cursor() as cur:
        cur.execute("SELECT count(*) FROM sm_reconciliation_receipts");assert cur.fetchone()[0]==0
        cur.execute("SELECT count(*) FROM sm_audit_events WHERE event_type='EXPANDED_INGESTION_BUDGET'");assert cur.fetchone()[0]==0


def test_grant_cannot_extend_past_the_last_delivery_plus_cycle():
    env=dict(CONTRACT,ARX_CONTINUOUS_INGESTION_AUTHORIZED='true',ARX_SEC_EGRESS_ROLE='OWNER',
             ARX_CONTINUOUS_AUTH_SCOPE='RECURRING_10_ISSUER_FORM4_INGESTION_ONLY',
             ARX_CONTINUOUS_AUTH_UNIVERSE_ID='UB-US-EQUITIES-20261025-001',ARX_CONTINUOUS_AUTH_ID='b'*32,
             ARX_RELEASE_SHA='a'*40,RAILWAY_GIT_COMMIT_SHA='a'*40,ARX_SMART_MONEY_INGESTION_ONLY='true',
             ARX_CONTINUOUS_AUTH_ISSUED_AT='2026-11-27T13:30:00Z',ARX_CONTINUOUS_AUTH_EXPIRES_AT='2026-11-27T14:21:01Z')
    with pytest.raises(RuntimeError,match='PHASE_END_EXCEEDED'):
        require_continuous_authorization(env=env,issuers=DEFAULT_GOVERNED_ISSUERS,now=datetime(2026,11,27,14,tzinfo=timezone.utc))
    env['ARX_CONTINUOUS_AUTH_EXPIRES_AT']='2026-11-27T14:21:00Z'
    require_continuous_authorization(env=env,issuers=DEFAULT_GOVERNED_ISSUERS,now=datetime(2026,11,27,14,tzinfo=timezone.utc))
