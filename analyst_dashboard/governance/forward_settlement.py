"""Offline daily-bar settlement; no source fetching, production or broker adapter.

The journal is replayable and input assertions are never promoted to independently
authenticated market evidence. Ambiguous entries reserve capacity conservatively.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from analyst_dashboard.governance.forward_capture import (
    CONTRACT_PATH, UNIVERSE_PATH, canonical, digest, normalize_evaluation, utc, session_window,
    compute_decision_id, compute_evaluation_cycle_id, DECISION_SCHEMA_VERSION,
)
from analyst_dashboard.governance.paper_trading_outcome_evaluator import PaperTradingOutcomeEvaluator
from analyst_dashboard.governance.storage import (
    is_production_runtime, safe_write_forward_pilot_file, validate_forward_write_path,
)


def read_capture(root, cohort_id):
    """Verify retained inputs and recompute each captured decision before use."""
    directory = root / cohort_id
    def read(relative):
        p = directory / relative
        validate_forward_write_path(str(p), cohort_id, str(root))
        return p.read_bytes()
    binding = json.loads(read('cohort.json'))
    universe = json.loads(UNIVERSE_PATH.read_bytes())
    if (binding.get('cohortId') != cohort_id or binding.get('universeSha256') != digest(universe)
            or binding.get('contractSha256') != hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()):
        raise ValueError('Frozen cohort/contract binding mismatch')
    raw = read('capture_stream.jsonl')
    lines = raw.splitlines()
    symbols = universe['symbols']
    if not raw.endswith(b'\n') or not lines or len(lines) % 10 or len(lines) > 200:
        raise ValueError('Missing, truncated or invalid capture stream')
    records = [json.loads(line) for line in lines]
    ids = set()
    cal = xcals.get_calendar('XNYS')
    for offset in range(0, len(records), 10):
        group = records[offset:offset + 10]
        if [r['symbol'] for r in group] != symbols:
            raise ValueError('Incomplete capture cycle')
        first = group[0]
        if not re.fullmatch(r'[a-f0-9]{64}', str(first['inputSha256'])):
            raise ValueError('Invalid capture input identity')
        source = read('inputs/' + first['inputSha256'] + '.json')
        if hashlib.sha256(source).hexdigest() != first['inputSha256']:
            raise ValueError('Retained capture input hash mismatch')
        batch = json.loads(source)
        session = first['sessionDate']
        close = cal.session_close(session).to_pydatetime()
        cycle = compute_evaluation_cycle_id('FORWARD_MANUAL_BATCH', close.isoformat(), digest(binding))
        if offset and session <= records[offset - 1]['sessionDate']:
            raise ValueError('Out-of-order capture cycles')
        for r in group:
            _, next_open, ordinal = session_window(session, binding['startSession'], utc(r['capturedAtUtc']))
            if (any(r.get(k) != v for k, v in binding.items()) or r['decisionId'] in ids
                    or r['cycleId'] != cycle or r['sessionDate'] != session
                    or r['decisionId'] != compute_decision_id(r['symbol'], cycle, binding['engineSha'], DECISION_SCHEMA_VERSION)
                    or r['recordSha256'] != digest({k: v for k, v in r.items() if k != 'recordSha256'})
                    or r['payloadSha256'] != digest(batch) or r['inputSha256'] != first['inputSha256']
                    or batch['sessionDate'] != session or batch['engineSha'] != binding['engineSha']
                    or batch.get('samplingMode') != 'SYSTEMATIC' or batch.get('schemaVersion') != '1.0'
                    or r['sessionOrdinal'] != ordinal or r['nextSessionOpenUtc'] != next_open.isoformat()
                    or r['evidenceOrigin'] != 'OFFLINE_INPUT' or r['eligibleForOpportunityCapture'] is not False):
                raise ValueError('Capture decision/source integrity mismatch')
            normalized = normalize_evaluation(r['symbol'], batch['evaluations'].get(r['symbol']), close, utc(r['capturedAtUtc']))
            if any(r.get(k) != v for k, v in normalized.items()):
                raise ValueError('Captured plan differs from retained source')
            ids.add(r['decisionId'])
    return binding, records


def validate_series(item, expected, as_of):
    """Validate supplied, retained provenance assertions; no external certification."""
    if not isinstance(item, dict) or not item.get('provider'):
        raise ValueError('Missing price provider')
    proof = item.get('adjustmentEvidence', {})
    # Support no-split intervals first. Split-factor application needs separate qualification.
    if (proof.get('status') != 'VERIFIED' or proof.get('priceBasis') != 'SPLIT_ADJUSTED_ONLY'
            or proof.get('cashDividendsAdjusted') is not False or proof.get('splitEvents') != []
            or not re.fullmatch(r'[a-f0-9]{64}', str(proof.get('sourceSha256', '')))
            or not proof.get('sourceDescription')):
        raise ValueError('UNRESOLVED_ADJUSTMENT_PROVENANCE')
    if expected and (proof.get('coverageStart', '9999') > min(expected)
                     or proof.get('coverageEnd', '') < max(expected)):
        raise ValueError('Adjustment evidence does not cover required interval')
    bars = item.get('bars')
    if not isinstance(bars, list):
        raise ValueError('Missing retained daily bars')
    dates = []
    for b in bars:
        d = b['date']
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', d) or d > as_of:
            raise ValueError('Future or invalid daily bar')
        if not xcals.get_calendar('XNYS').is_session(d):
            raise ValueError('Non-session daily bar')
        vals = [b[k] for k in ('Open', 'High', 'Low', 'Close', 'Volume')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
            raise ValueError('Nonfinite OHLCV')
        o, h, l, c, v = vals
        if not (0 < l <= min(o, c) <= max(o, c) <= h and v >= 0):
            raise ValueError('Invalid OHLCV bounds')
        dates.append(d)
    if dates != sorted(set(dates)) or not expected.issubset(set(dates)):
        raise ValueError('Duplicate, unordered or missing required sessions')
    return bars


def settle_batch(raw_batch: bytes, *, cohort_id: str, forward_root: Path,
                 initialize=False, now=None):
    """Replay a complete retained prefix and atomically record changed results.

    No clock override is exposed by the CLI. Repeated identical state is a no-op.
    """
    if is_production_runtime():
        raise ValueError('Production settlement is not activated')
    if not re.fullmatch(r'COHORT_FORWARD_PILOT_[A-Za-z0-9_-]+', cohort_id):
        raise ValueError('Invalid forward cohort namespace')
    root = Path(forward_root).absolute()
    if not root.is_dir():
        raise ValueError('Explicit existing forward root required')
    batch = json.loads(raw_batch)
    canonical(batch)
    if batch.get('schemaVersion') != '1.0' or batch.get('cohortId') != cohort_id:
        raise ValueError('Settlement batch schema/cohort mismatch')
    as_of = batch['asOfSession']
    now = now or datetime.now(timezone.utc)
    cal = xcals.get_calendar('XNYS')
    if now.tzinfo is None or not cal.is_session(as_of) or cal.session_close(as_of).to_pydatetime() > now:
        raise ValueError('Settlement requires a completed XNYS session')
    lock_path = root / '.capture.lock'  # Same lock as capture: stable stream snapshot.
    validate_forward_write_path(str(lock_path), cohort_id, str(root))
    with os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600), 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        binding, records = read_capture(root, cohort_id)
        directory = root / cohort_id
        ledger_path = directory / 'settlement_ledger.json'
        inputs_dir = directory / 'settlement_inputs'
        validate_forward_write_path(str(ledger_path), cohort_id, str(root))
        validate_forward_write_path(str(inputs_dir), cohort_id, str(root))
        previous = None
        if ledger_path.exists():
            previous = json.loads(ledger_path.read_bytes())
            if previous['stateSha256'] != digest({k: v for k, v in previous.items() if k != 'stateSha256'}):
                raise ValueError('Settlement ledger integrity mismatch')
            if previous['binding'] != binding or as_of < previous['asOfSession']:
                raise ValueError('Settlement binding mismatch or clock regression')
            for receipt in previous['runs']:
                raw_path = inputs_dir / (receipt['inputSha256'] + '.json')
                validate_forward_write_path(str(raw_path), cohort_id, str(root))
                if hashlib.sha256(raw_path.read_bytes()).hexdigest() != receipt['inputSha256']:
                    raise ValueError('Retained settlement input integrity mismatch')
        elif not initialize or inputs_dir.exists():
            raise ValueError('Missing settlement ledger; explicit first initialization required, no reset')
        if any(r['sessionDate'] > as_of for r in records):
            raise ValueError('Settlement predates captured decisions')
        expected = {}
        for r in records:
            if r['evaluationStatus'] == 'RECORDED' and r['decisionState'] in {'VALID_SETUP', 'ACTIONABLE_SETUP'}:
                start = cal.next_session(r['sessionDate'])
                end = (start.to_pydatetime() + timedelta(days=60)).date().isoformat()
                horizon = [s.date().isoformat() for s in cal.sessions_in_range(start, end)[:24]]
                expected.setdefault(r['symbol'], set()).update(d for d in horizon if d <= as_of)
        series = batch.get('series', {})
        if not isinstance(series, dict) or set(series) - set(json.loads(UNIVERSE_PATH.read_bytes())['symbols']):
            raise ValueError('Invalid price series universe')
        prices = {s: validate_series(series.get(s), dates, as_of) for s, dates in expected.items()}
        if previous:
            # An expanded snapshot may add bars, but cannot revise any retained history/provider.
            for s, old in previous['priceHistory'].items():
                new = series.get(s, {})
                if new.get('provider') != old['provider']:
                    raise ValueError('Price provider changed')
                by_date = {b['date']: b for b in prices.get(s, [])}
                if any(by_date.get(b['date']) != b for b in old['bars']):
                    raise ValueError('Historical price revision rejected')
        results = {}
        reservations = []
        rank = {s: i for i, s in enumerate(json.loads(UNIVERSE_PATH.read_bytes())['symbols'])}
        for r in records:
            result = {'status': 'NOT_ELIGIBLE', 'result': None}
            if r['symbol'] in expected and r['evaluationStatus'] == 'RECORDED' and r['decisionState'] in {'VALID_SETUP', 'ACTIONABLE_SETUP'}:
                bars = prices[r['symbol']]
                frame = pd.DataFrame(bars).set_index('date') if bars else pd.DataFrame()
                if len(frame):
                    frame.index = pd.to_datetime(frame.index)
                signal = {**r['plan'], 'symbol': r['symbol'], 'signalDate': r['sessionDate'], 'engineVersion': binding['engineSha']}
                value = PaperTradingOutcomeEvaluator.evaluate_signal(signal, frame, provider=series[r['symbol']]['provider']).to_dict()
                result = {'status': value['outcome'], 'result': value}
                entry_day = value['entry_date']
                uncertain = value['outcome'] == 'UNRESOLVED_INTRABAR_SEQUENCE' and not entry_day
                if entry_day or uncertain:
                    reservations.append((entry_day or value['exit_date'], r['sessionDate'], rank[r['symbol']], r['decisionId'], uncertain))
            results[r['decisionId']] = result
        admitted = set()
        uncertain_count = 0
        for n, (_, _, _, decision_id, uncertain) in enumerate(sorted(reservations)):
            if n < 20:
                admitted.add(decision_id)
                uncertain_count += int(uncertain)
            else:
                results[decision_id] = {'status': 'SKIPPED_CAP_REACHED', 'result': None}
        state = {'schemaVersion': '1.0', 'binding': binding, 'asOfSession': as_of,
                 'evidenceOrigin': 'OFFLINE_INPUT', 'eligibleForOpportunityCapture': False,
                 'adjustmentEvidenceAuthority': 'RETAINED_INPUT_ASSERTION_NOT_INDEPENDENTLY_AUTHENTICATED',
                 'results': results, 'reservedEntrySlots': len(admitted),
                 'uncertainEntryReservations': uncertain_count,
                 'triggeredEntries': len(admitted) - uncertain_count,
                 'priceHistory': {s: {'provider': series[s]['provider'], 'bars': bars} for s, bars in prices.items()}}
        if previous and all(previous.get(k) == v for k, v in state.items()):
            return {'written': 0, 'totalCandidates': len(records), 'reservedEntrySlots': len(admitted)}
        old_results = previous['results'] if previous else {}
        events = list(previous['events']) if previous else []
        for decision_id, result in results.items():
            if old_results.get(decision_id) != result:
                events.append({'eventId': digest([decision_id, result]), 'decisionId': decision_id, **result})
        raw_sha = hashlib.sha256(raw_batch).hexdigest()
        state['events'] = events
        state['runs'] = (previous['runs'] if previous else []) + [{'asOfSession': as_of, 'inputSha256': raw_sha}]
        state['stateSha256'] = digest(state)
        raw_path = inputs_dir / (raw_sha + '.json')
        safe_write_forward_pilot_file(str(raw_path), cohort_id, raw_batch.decode('utf-8'), str(root))
        safe_write_forward_pilot_file(str(ledger_path), cohort_id, canonical(state) + '\n', str(root))
        return {'written': len(events) - (len(previous['events']) if previous else 0),
                'totalCandidates': len(records), 'reservedEntrySlots': len(admitted)}
