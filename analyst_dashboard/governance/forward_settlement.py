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
from analyst_dashboard.governance.market_data_adapter import (
    verify_raw_response_bytes,
    verify_corporate_action_evidence,
    normalize_yahoo_chart_bars,
    RawBytesHashMismatchError,
    UnverifiedAdjustmentEvidenceError,
    UnsupportedSplitAdjustmentError,
    DividendAdjustedDataError,
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


def validate_series(item, expected, as_of, symbol=None, artifacts_dir=None):
    """Validate supplied, retained provenance assertions and market data integrity."""
    if not isinstance(item, dict) or not item.get('provider'):
        raise ValueError('Missing price provider')

    item_symbol = symbol or item.get('symbol')
    if symbol and item.get('symbol') and item['symbol'].upper() != symbol.upper():
        raise ValueError(f"Symbol mismatch in series item: expected {symbol}, got {item['symbol']}")

    # If raw response bytes are provided, cryptographically verify and normalize them
    if 'rawResponseBytes' in item or 'rawBytes' in item:
        raw_b = item.get('rawResponseBytes') or item.get('rawBytes')
        if isinstance(raw_b, str):
            raw_b = raw_b.encode('utf-8')
        raw_sha = item.get('rawResponseSha256')
        if not raw_sha:
            raise ValueError('Missing rawResponseSha256 for provided raw response bytes')
        verify_raw_response_bytes(raw_b, raw_sha)
        if item.get('provider') == 'YAHOO_FINANCE' and item_symbol:
            derived_bars = normalize_yahoo_chart_bars(raw_b, item_symbol, as_of=as_of)
            if item.get('bars') and derived_bars != item['bars']:
                raise ValueError('NORMALIZATION_DESYNC: normalized bars do not match raw response bytes')
    elif 'rawResponseSha256' in item:
        raw_sha = str(item['rawResponseSha256'])
        if not re.fullmatch(r'[a-f0-9]{64}', raw_sha):
            raise ValueError('Invalid rawResponseSha256 format')

    proof = item.get('adjustmentEvidence', {})
    require_auth = (item.get('evidenceOrigin') == 'AUTHENTIC_FORWARD' or 'corporateActionReference' in proof)
    raw_ca = (
        item.get('corporateActionRawBytes')
        or item.get('corporateActionBytes')
        or (proof.get('corporateActionReference') and (
            proof['corporateActionReference'].get('rawArtifactBytes')
            or proof['corporateActionReference'].get('rawBytes')
        ))
    )
    try:
        verify_corporate_action_evidence(
            proof,
            expected,
            require_authoritative_ref=require_auth,
            symbol=item_symbol,
            raw_artifact_bytes=raw_ca,
            artifacts_dir=artifacts_dir,
        )
    except (UnverifiedAdjustmentEvidenceError, UnsupportedSplitAdjustmentError, DividendAdjustedDataError) as e:
        raise ValueError(str(e))
    except ValueError as e:
        if 'UNSUPPORTED_ADJUSTMENT_BASIS' in str(e) or 'cash dividends' in str(e) or 'UNRESOLVED_ADJUSTMENT_PROVENANCE' in str(e):
            raise ValueError('UNRESOLVED_ADJUSTMENT_PROVENANCE')
        raise

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
        raw_responses_dir = directory / 'raw_responses'
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
        prices = {
            s: validate_series(series.get(s), dates, as_of, symbol=s, artifacts_dir=raw_responses_dir)
            for s, dates in expected.items()
        }
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
        price_history = {}
        for s, bars in prices.items():
            entry = {'provider': series[s]['provider'], 'bars': bars}
            if 'rawResponseSha256' in series[s]:
                entry['rawResponseSha256'] = series[s]['rawResponseSha256']
            if 'retrievalTimestampUtc' in series[s]:
                entry['retrievalTimestampUtc'] = series[s]['retrievalTimestampUtc']
            if 'timezone' in series[s]:
                entry['timezone'] = series[s]['timezone']
            price_history[s] = entry

        candidate_origin = records[0]['evidenceOrigin'] if records else 'OFFLINE_INPUT'
        if batch.get('evidenceOrigin') == 'AUTHENTIC_FORWARD' and candidate_origin == 'OFFLINE_INPUT':
            raise ValueError(
                'CANNOT_UPGRADE_OFFLINE_DECISIONS: Authentic prices cannot upgrade OFFLINE_INPUT candidate decisions to authentic forward observations'
            )

        all_series_authentic_prices = True
        all_series_authentic_ca = True

        for s in expected:
            s_data = series.get(s, {})
            proof = s_data.get('adjustmentEvidence', {})
            corp_ref = proof.get('corporateActionReference') or s_data.get('corporateActionReference')
            has_raw_prices = (
                ('rawResponseBytes' in s_data or 'rawBytes' in s_data or (raw_responses_dir / f"{s_data.get('rawResponseSha256')}.raw").exists())
                and bool(s_data.get('rawResponseSha256'))
            )
            if not (s_data.get('provider') == 'YAHOO_FINANCE' and has_raw_prices):
                all_series_authentic_prices = False

            ca_bytes_present = bool(
                corp_ref and (
                    corp_ref.get('rawArtifactBytes') or corp_ref.get('rawBytes') or
                    s_data.get('corporateActionRawBytes') or s_data.get('corporateActionBytes') or
                    (raw_responses_dir / f"{corp_ref.get('sourceSha256')}.ca.raw").exists() or
                    (raw_responses_dir / f"{corp_ref.get('sourceSha256')}.raw").exists()
                )
            )
            ca_acq = (corp_ref.get('acquisitionProvenance') if corp_ref else None) or proof.get('acquisitionProvenance')
            ca_has_acquisition = bool(
                isinstance(ca_acq, dict) and
                ca_acq.get('provider') in ('SEC_EDGAR_SUBMISSIONS', 'SEC_EDGAR', 'OFFICIAL_EXCHANGE_DISCLOSURE', 'OFFICIAL_EXCHANGE_NOTICE') and
                ca_acq.get('httpStatus') == 200 and
                bool(ca_acq.get('targetUrl')) and
                bool(ca_acq.get('retrievalTimestampUtc'))
            )
            if not (corp_ref and ca_bytes_present and ca_has_acquisition):
                all_series_authentic_ca = False

        price_source_provenance = 'AUTHENTIC_RETAINED_SOURCE' if all_series_authentic_prices else 'OFFLINE_FIXTURE'
        adj_authority = 'AUTHENTIC_CORPORATE_ACTION_VERIFIED' if all_series_authentic_ca else 'RETAINED_INPUT_ASSERTION_NOT_INDEPENDENTLY_AUTHENTICATED'

        if batch.get('priceSourceProvenance') == 'AUTHENTIC_RETAINED_SOURCE' and not all_series_authentic_prices:
            raise ValueError('AUTHENTIC_PRICE_PROVENANCE_MISMATCH: Batch asserted AUTHENTIC_RETAINED_SOURCE but price series lacked verified raw response bytes')

        if batch.get('adjustmentEvidenceAuthority') == 'AUTHENTIC_CORPORATE_ACTION_VERIFIED' and not all_series_authentic_ca:
            raise ValueError('UNSUPPORTED_AUTHENTICITY_CERTIFICATION: Batch claimed AUTHENTIC_CORPORATE_ACTION_VERIFIED but corporate-action evidence lacks verified external acquisition provenance')

        state = {
            'schemaVersion': '1.0',
            'binding': binding,
            'asOfSession': as_of,
            'candidateEvidenceOrigin': candidate_origin,
            'candidateOpportunityEligible': False,
            'priceSourceProvenance': price_source_provenance,
            'adjustmentEvidenceAuthority': adj_authority,
            'activationState': 'DISABLED',
            'eligibleForOpportunityCapture': False,
            'evidenceOrigin': candidate_origin,
            'results': results,
            'reservedEntrySlots': len(admitted),
            'uncertainEntryReservations': uncertain_count,
            'triggeredEntries': len(admitted) - uncertain_count,
            'priceHistory': price_history,
        }
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

        # Retain raw response bytes into raw_responses/ if provided
        for s, s_data in series.items():
            if isinstance(s_data, dict):
                if ('rawResponseBytes' in s_data or 'rawBytes' in s_data) and 'rawResponseSha256' in s_data:
                    r_sha = s_data['rawResponseSha256']
                    r_b = s_data.get('rawResponseBytes') or s_data.get('rawBytes')
                    if isinstance(r_b, str):
                        r_b = r_b.encode('utf-8')
                    raw_file = raw_responses_dir / f"{r_sha}.raw"
                    meta_file = raw_responses_dir / f"{r_sha}.meta.json"
                    validate_forward_write_path(str(raw_file), cohort_id, str(root))
                    validate_forward_write_path(str(meta_file), cohort_id, str(root))
                    safe_write_forward_pilot_file(str(raw_file), cohort_id, r_b, str(root))
                    meta_payload = {
                        'symbol': s,
                        'provider': s_data.get('provider'),
                        'rawSha256': r_sha,
                        'retrievalTimestampUtc': s_data.get('retrievalTimestampUtc'),
                        'timezone': s_data.get('timezone', 'America/New_York'),
                        'byteLength': len(r_b),
                        'targetUrl': s_data.get('targetUrl'),
                    }
                    safe_write_forward_pilot_file(str(meta_file), cohort_id, canonical(meta_payload) + '\n', str(root))

                proof = s_data.get('adjustmentEvidence') or {}
                corp_ref = proof.get('corporateActionReference') or s_data.get('corporateActionReference')
                ca_bytes = (
                    (corp_ref and (corp_ref.get('rawArtifactBytes') or corp_ref.get('rawBytes')))
                    or s_data.get('corporateActionRawBytes')
                    or s_data.get('corporateActionBytes')
                )
                if corp_ref and ca_bytes and 'sourceSha256' in corp_ref:
                    ca_sha = corp_ref['sourceSha256']
                    if isinstance(ca_bytes, str):
                        ca_bytes = ca_bytes.encode('utf-8')
                    ca_file = raw_responses_dir / f"{ca_sha}.ca.raw"
                    ca_meta_file = raw_responses_dir / f"{ca_sha}.ca.meta.json"
                    validate_forward_write_path(str(ca_file), cohort_id, str(root))
                    validate_forward_write_path(str(ca_meta_file), cohort_id, str(root))
                    safe_write_forward_pilot_file(str(ca_file), cohort_id, ca_bytes, str(root))
                    ca_meta_payload = {
                        'symbol': s,
                        'sourceType': corp_ref.get('sourceType'),
                        'sourceSha256': ca_sha,
                        'coverageStart': corp_ref.get('coverageStart'),
                        'coverageEnd': corp_ref.get('coverageEnd'),
                        'byteLength': len(ca_bytes),
                    }
                    if corp_ref.get('acquisitionProvenance'):
                        ca_meta_payload['acquisitionProvenance'] = corp_ref['acquisitionProvenance']
                    safe_write_forward_pilot_file(str(ca_meta_file), cohort_id, canonical(ca_meta_payload) + '\n', str(root))

        raw_path = inputs_dir / (raw_sha + '.json')
        safe_write_forward_pilot_file(str(raw_path), cohort_id, raw_batch.decode('utf-8'), str(root))
        safe_write_forward_pilot_file(str(ledger_path), cohort_id, canonical(state) + '\n', str(root))
        return {'written': len(events) - (len(previous['events']) if previous else 0),
                'totalCandidates': len(records), 'reservedEntrySlots': len(admitted)}
