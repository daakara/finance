"""Offline PT-03 settlement mechanics; no production or source acquisition."""
import copy
import json
import multiprocessing
from datetime import datetime, timezone

import exchange_calendars as xcals
import pytest

from analyst_dashboard.governance.forward_capture import capture_batch, UNIVERSE_PATH
from analyst_dashboard.governance.forward_settlement import settle_batch

COHORT = 'COHORT_FORWARD_PILOT_SETTLEMENT_TEST'
SYMBOLS = json.loads(UNIVERSE_PATH.read_bytes())['symbols']
CAL = xcals.get_calendar('XNYS')
NOW = datetime(2027, 2, 1, 22, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def local(monkeypatch):
    for k in ('ARX_FORCE_PRODUCTION_STORAGE_VALIDATION', 'RAILWAY_ENVIRONMENT_NAME',
              'RAILWAY_ENVIRONMENT', 'ENVIRONMENT'):
        monkeypatch.delenv(k, raising=False)


def capture(root, sessions=1, missing=False):
    labels = CAL.sessions_in_range('2026-11-23', '2026-12-23')[:sessions]
    for i, label in enumerate(labels):
        day = label.date().isoformat()
        close = CAL.session_close(label).to_pydatetime()
        evaluations = {s: {'decisionState': 'VALID_SETUP', 'sourceTimestampUtc': close.isoformat(),
                          'observedAtUtc': close.isoformat(),
                          'plan': {'spot': 100., 'corridorMin': 98., 'corridorMax': 102.,
                                   'stopLoss': 95., 'takeProfit1': 110., 'confluenceScore': 70.}}
                       for s in SYMBOLS}
        if missing:
            evaluations.pop('NVDA')
        batch = {'schemaVersion': '1.0', 'samplingMode': 'SYSTEMATIC', 'sessionDate': day,
                 'engineSha': 'a' * 40, 'evaluations': evaluations}
        from datetime import timedelta
        capture_batch(json.dumps(batch).encode(), cohort_id=COHORT, start_session='2026-11-23',
                      forward_root=root, initialize=i == 0, now=close + timedelta(minutes=20))


def prices(end='2026-12-29', *, collision=False, no_entry=False):
    days = [s.date().isoformat() for s in CAL.sessions_in_range('2026-11-24', end)]
    bars = [{'date': d, 'Open': 100., 'High': 103., 'Low': 97., 'Close': 101., 'Volume': 1000}
            for d in days]
    if collision:
        bars[0]['Low'] = 94.
    if no_entry:
        for b in bars:
            b.update(Open=105., Low=104., High=106., Close=105.)
    return {'schemaVersion': '1.0', 'cohortId': COHORT, 'asOfSession': end,
            'series': {s: {'provider': 'OFFLINE_TEST', 'bars': copy.deepcopy(bars),
                          'adjustmentEvidence': {'status': 'VERIFIED', 'priceBasis': 'SPLIT_ADJUSTED_ONLY',
                            'cashDividendsAdjusted': False, 'splitEvents': [],
                            'sourceSha256': 'b' * 64, 'sourceDescription': 'Synthetic offline metadata',
                            'coverageStart': days[0], 'coverageEnd': end}}
                       for s in SYMBOLS}}


def settle(root, batch, **kw):
    return settle_batch(json.dumps(batch).encode(), cohort_id=COHORT, forward_root=root,
                        initialize=True, now=NOW, **kw)


def ledger(root):
    return root / COHORT / 'settlement_ledger.json'


def test_repeat_five_times_is_byte_identical_and_keeps_capture_frozen(tmp_path):
    capture(tmp_path, missing=True)
    capture_path = tmp_path / COHORT / 'capture_stream.jsonl'
    before = capture_path.read_bytes()
    batch = prices()
    assert settle(tmp_path, batch)['written'] == 10
    original = ledger(tmp_path).read_bytes()
    for _ in range(5):
        assert settle(tmp_path, batch)['written'] == 0
        assert ledger(tmp_path).read_bytes() == original
    state = json.loads(original)
    assert state['triggeredEntries'] == 9
    outcomes = [r['result'] for r in state['results'].values() if r['result']]
    assert all(r['outcome'] == 'NEUTRAL' and r['exit_reason'] == 'HORIZON_EXIT' for r in outcomes)
    assert all(r['net_simulated_return_pct'] == pytest.approx(.75) for r in outcomes)
    assert capture_path.read_bytes() == before
    assert not state['eligibleForOpportunityCapture']


def test_capacity_is_chronological_and_not_selected_by_returns(tmp_path):
    capture(tmp_path, sessions=3)
    batch = prices()
    settle(tmp_path, batch)
    state = json.loads(ledger(tmp_path).read_bytes())
    assert state['reservedEntrySlots'] == state['triggeredEntries'] == 20
    assert sum(r['status'] == 'SKIPPED_CAP_REACHED' for r in state['results'].values()) == 10
    assert len(state['results']) == 30


def test_twenty_capture_sessions_keep_all_two_hundred_decisions(tmp_path):
    capture(tmp_path, sessions=20)
    settle(tmp_path, prices('2027-01-27'))
    state = json.loads(ledger(tmp_path).read_bytes())
    assert len(state['results']) == 200
    assert state['triggeredEntries'] == state['reservedEntrySlots'] == 20
    assert sum(r['status'] == 'SKIPPED_CAP_REACHED' for r in state['results'].values()) == 180
    assert len({e['eventId'] for e in state['events']}) == len(state['events'])


def test_daily_collision_reserves_capacity_without_fabricating_fill(tmp_path):
    capture(tmp_path)
    settle(tmp_path, prices(collision=True))
    state = json.loads(ledger(tmp_path).read_bytes())
    assert state['triggeredEntries'] == 0
    assert state['uncertainEntryReservations'] == state['reservedEntrySlots'] == 10
    assert all(r['status'] == 'UNRESOLVED_INTRABAR_SEQUENCE' and r['result']['entry_price'] is None
               for r in state['results'].values())


def test_gap_above_limit_without_touch_never_fills(tmp_path):
    capture(tmp_path)
    settle(tmp_path, prices(no_entry=True))
    state = json.loads(ledger(tmp_path).read_bytes())
    assert state['triggeredEntries'] == 0
    assert all(r['status'] == 'ENTRY_NOT_TRIGGERED' for r in state['results'].values())


def test_incremental_replay_and_historical_revision_rejection(tmp_path):
    capture(tmp_path)
    settle(tmp_path, prices('2026-11-24'))
    first = json.loads(ledger(tmp_path).read_bytes())
    assert all(r['status'] == 'UNMATURED' for r in first['results'].values())
    settle(tmp_path, prices())
    final = ledger(tmp_path).read_bytes()
    state = json.loads(final)
    assert len(state['events']) == 20 and len(state['runs']) == 2
    revised = prices()
    revised['series']['VOO']['bars'][0]['Close'] = 100.5
    with pytest.raises(ValueError, match='revision'):
        settle(tmp_path, revised)
    assert ledger(tmp_path).read_bytes() == final
    with pytest.raises(ValueError, match='regression'):
        settle(tmp_path, prices('2026-11-24'))


@pytest.mark.parametrize('kind', ['missing_session', 'duplicate', 'future', 'provider', 'adjustment', 'split', 'dividend', 'nan'])
def test_bad_evidence_blocks_before_ledger_write(tmp_path, kind):
    capture(tmp_path)
    batch = prices()
    series = batch['series']['VOO']
    if kind == 'missing_session': series['bars'].pop(1)
    if kind == 'duplicate': series['bars'].append(series['bars'][-1])
    if kind == 'future': series['bars'][-1]['date'] = '2027-01-04'
    if kind == 'provider': series.pop('provider')
    if kind == 'adjustment': series['adjustmentEvidence']['status'] = 'UNKNOWN'
    if kind == 'split': series['adjustmentEvidence']['splitEvents'] = [{'factor': 2}]
    if kind == 'dividend': series['adjustmentEvidence']['cashDividendsAdjusted'] = True
    if kind == 'nan': series['bars'][0]['Close'] = float('nan')
    with pytest.raises((ValueError, KeyError)):
        settle(tmp_path, batch)
    assert not ledger(tmp_path).exists()


@pytest.mark.parametrize('kind', ['missing_ledger', 'corrupt_ledger', 'missing_input', 'corrupt_capture'])
def test_restart_corruption_never_resets_state(tmp_path, kind):
    capture(tmp_path)
    batch = prices()
    settle(tmp_path, batch)
    if kind == 'missing_ledger': ledger(tmp_path).unlink()
    if kind == 'corrupt_ledger': ledger(tmp_path).write_text('{}')
    if kind == 'missing_input': next((tmp_path / COHORT / 'settlement_inputs').iterdir()).unlink()
    if kind == 'corrupt_capture': (tmp_path / COHORT / 'capture_stream.jsonl').write_text('broken')
    with pytest.raises((ValueError, KeyError, OSError)):
        settle(tmp_path, batch)


def test_production_and_future_close_blocked(tmp_path, monkeypatch):
    capture(tmp_path)
    monkeypatch.setenv('ARX_FORCE_PRODUCTION_STORAGE_VALIDATION', '1')
    with pytest.raises(ValueError, match='Production'):
        settle(tmp_path, prices())
    monkeypatch.delenv('ARX_FORCE_PRODUCTION_STORAGE_VALIDATION')
    with pytest.raises(ValueError, match='completed'):
        settle_batch(json.dumps(prices()).encode(), cohort_id=COHORT, forward_root=tmp_path,
                     initialize=True, now=datetime(2026, 12, 29, 12, tzinfo=timezone.utc))


def test_atomic_commit_failure_preserves_previous_ledger(tmp_path, monkeypatch):
    capture(tmp_path)
    settle(tmp_path, prices('2026-11-24'))
    original = ledger(tmp_path).read_bytes()
    import analyst_dashboard.governance.forward_settlement as module
    writer = module.safe_write_forward_pilot_file
    def fail_ledger(path, *args):
        if str(path).endswith('settlement_ledger.json'):
            raise OSError('Injected failure before ledger replacement')
        return writer(path, *args)
    monkeypatch.setattr(module, 'safe_write_forward_pilot_file', fail_ledger)
    with pytest.raises(OSError, match='Injected'):
        settle(tmp_path, prices())
    assert ledger(tmp_path).read_bytes() == original
    monkeypatch.setattr(module, 'safe_write_forward_pilot_file', writer)
    assert settle(tmp_path, prices())['written'] == 10


def _worker(root, batch, queue):
    try:
        queue.put(settle(root, batch)['written'])
    except Exception as e:
        queue.put(str(e))


def test_concurrent_replays_create_one_event_set(tmp_path):
    capture(tmp_path)
    ctx = multiprocessing.get_context('fork')
    q = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(tmp_path, prices(), q)) for _ in range(6)]
    for p in workers: p.start()
    for p in workers:
        p.join(15)
        assert p.exitcode == 0
    assert sorted(q.get(timeout=2) for _ in workers) == [0, 0, 0, 0, 0, 10]
    assert len(json.loads(ledger(tmp_path).read_bytes())['events']) == 10
