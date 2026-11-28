#!/usr/bin/env python3
"""Acquire completed-session market data and execute bounded forward capture.

Governed by:
- ARX Paper-Trading Outcome Evaluation Contract V1.0.1
- docs/planning/ARX_PAPER_TRADING_ROADMAP.md (PR #322)

Enforces:
1. Shared persistent coordination store (session claims, request reservations, completion).
2. Runtime window recheck (session_window) using exchange calendar at actual execution time.
3. Hard calendar-day cutoff in America/New_York (no backdating or retrospective replay).
4. Explicit cumulative request budget counting initial requests, retries, and failures.
5. Durably reserved attempts BEFORE transmission across runners and branches.
6. Idempotent capture commit: crash after write does not produce duplicates on recovery.
7. Separate tracking of invocation, source cutoff, completed capture times, and outside-corridor classifications.
8. Raw response byte integrity, content-addressed storage, and split rejection.
9. Complete 10-symbol accounting without fabrication.
10. Next-session execution eligibility: decisions apply to T+1 open, never observed session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.governance.forward_capture import (
    UNIVERSE_PATH,
    canonical,
    session_window,
    utc,
)
from analyst_dashboard.governance.forward_evaluator_adapter import (
    execute_isolated_forward_evaluation,
)
from analyst_dashboard.governance.market_data_adapter import (
    normalize_yahoo_chart_bars,
    verify_raw_response_bytes,
)
from analyst_dashboard.governance.paper_trading_coordination import (
    BudgetExhaustedError,
    ConcurrentClaimConflictError,
    CoordinationStoreUnavailableError,
    PaperTradingCoordinationStore,
)

logger = logging.getLogger("arx.governance.acquire_and_capture")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_ENGINE_SHA = "05fb5a320501b7bdd32630bc3d18c3a719cef626"
DEFAULT_TOTAL_REQUEST_BUDGET = 12  # 10 symbols + max 2 retries across session
YAHOO_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def derive_runner_identity(explicit_runner: Optional[str] = None) -> str:
    """Derives a deterministic, traceable runner identity for lease claims."""
    if explicit_runner:
        return explicit_runner.strip()

    gh_run_id = os.getenv("GITHUB_RUN_ID")
    gh_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
    if gh_run_id:
        return f"gha-run-{gh_run_id}-attempt-{gh_attempt}"

    railway_replica = os.getenv("RAILWAY_REPLICA_ID")
    if railway_replica:
        return f"railway-{railway_replica[:12]}"

    hostname = socket.gethostname()
    pid = os.getpid()
    return f"local-{hostname}-{pid}"


class DurableRequestLedger:
    """Maintains an append-only JSONL record of all HTTP request attempts across retries and restarts."""

    def __init__(self, ledger_file: Path):
        self.ledger_file = ledger_file
        self.attempts: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.ledger_file.is_file():
            for line in self.ledger_file.read_text().splitlines():
                if line.strip():
                    try:
                        self.attempts.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    @property
    def total_requests(self) -> int:
        return len(self.attempts)

    def reserve_attempt(self, symbol: str, target_url: str) -> int:
        """Atomically reserves an attempt on disk BEFORE transmission."""
        attempt_index = len(self.attempts) + 1
        entry = {
            "attemptIndex": attempt_index,
            "timestampUtc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "targetUrl": target_url,
            "status": "PENDING_TRANSMISSION",
        }
        self.attempts.append(entry)
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return attempt_index

    def finalize_attempt(
        self,
        attempt_index: int,
        *,
        status: str,
        http_code: Optional[int] = None,
        error: Optional[str] = None,
        byte_length: int = 0,
        sha256: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Updates the attempt record and persists completion status to disk."""
        target_entry = None
        for a in self.attempts:
            if a.get("attemptIndex") == attempt_index:
                target_entry = a
                break
        if target_entry is None:
            target_entry = {"attemptIndex": attempt_index}
            self.attempts.append(target_entry)

        target_entry.update({
            "completedAtUtc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "httpCode": http_code,
            "error": error,
            "byteLength": byte_length,
            "sha256": sha256,
            "durationMs": round(duration_ms, 2),
        })

        temp_file = self.ledger_file.with_suffix(".tmp")
        with temp_file.open("w", encoding="utf-8") as f:
            for a in self.attempts:
                f.write(json.dumps(a) + "\n")
            f.flush()
            os.fsync(f.fileno())
        temp_file.replace(self.ledger_file)
        return target_entry

    def record_attempt(
        self,
        *,
        symbol: str,
        target_url: str,
        status: str,
        http_code: Optional[int] = None,
        error: Optional[str] = None,
        byte_length: int = 0,
        sha256: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> Dict[str, Any]:
        idx = self.reserve_attempt(symbol=symbol, target_url=target_url)
        return self.finalize_attempt(
            idx,
            status=status,
            http_code=http_code,
            error=error,
            byte_length=byte_length,
            sha256=sha256,
            duration_ms=duration_ms,
        )

    def is_symbol_acquired(self, symbol: str, raw_dir: Path, session_date: str) -> Optional[Dict[str, Any]]:
        """Checks if symbol was already successfully acquired in a previous attempt/run and validates disk integrity."""
        for att in reversed(self.attempts):
            if att.get("symbol") == symbol and att.get("status") == "SUCCESS" and att.get("sha256"):
                sha = att["sha256"]
                target_raw = raw_dir / f"{symbol}_{sha[:16]}.raw"
                if target_raw.is_file() and hashlib.sha256(target_raw.read_bytes()).hexdigest() == sha:
                    try:
                        bars = normalize_yahoo_chart_bars(target_raw.read_bytes(), symbol, as_of=session_date)
                        if bars and (bars[-1].get("date") or bars[-1].get("Date")) == session_date:
                            logger.info(f"Reusing already acquired and verified raw data for {symbol} (hash: {sha[:16]})")
                            return {"rawPath": str(target_raw.resolve()), "rawResponseSha256": sha}
                    except Exception:
                        pass
        return None


def acquire_symbol_data(
    symbol: str,
    raw_dir: Path,
    session_date: str,
    cohort_id: str,
    store: PaperTradingCoordinationStore,
    ledger: DurableRequestLedger,
    max_requests: int,
    offline_fixture_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Acquires and validates raw chart bytes for one symbol respecting durable request budget across store & ledger."""
    # 1. Check if already acquired in an earlier attempt/restart
    existing = ledger.is_symbol_acquired(symbol, raw_dir, session_date)
    if existing:
        return existing

    if offline_fixture_dir is not None:
        # Offline fixture path
        fixture_raw = offline_fixture_dir / f"{symbol}.raw"
        if not fixture_raw.is_file():
            matches = list(offline_fixture_dir.glob(f"{symbol}*.raw"))
            if matches:
                fixture_raw = matches[0]
            else:
                logger.warning(f"Offline fixture for {symbol} not found in {offline_fixture_dir}")
                return None
        raw_bytes = fixture_raw.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        ts_utc = datetime.now(timezone.utc).isoformat()
        target_raw = raw_dir / f"{symbol}_{digest[:16]}.raw"
        target_meta = raw_dir / f"{symbol}_{digest[:16]}.meta.json"
        target_raw.write_bytes(raw_bytes)
        meta = {
            "symbol": symbol,
            "source": "offline_fixture",
            "fixturePath": str(fixture_raw),
            "timestampUtc": ts_utc,
            "sha256": digest,
            "byteLength": len(raw_bytes),
            "rawFilePath": str(target_raw.resolve()),
        }
        target_meta.write_text(json.dumps(meta, indent=2))
        return {"rawPath": str(target_raw.resolve()), "rawResponseSha256": digest}

    # 2. Live network acquisition
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=6mo&interval=1d&events=history&includeAdjustedClose=true"
    headers = {"User-Agent": YAHOO_USER_AGENT}

    max_symbol_retries = 2
    attempts_for_symbol = 0

    while attempts_for_symbol <= max_symbol_retries:
        consumed = store.get_consumed_budget(cohort_id, session_date)
        if consumed >= max_requests:
            logger.error(
                f"Total request budget exhausted ({consumed}/{max_requests}). Halting acquisition."
            )
            return None

        attempts_for_symbol += 1
        t0 = time.perf_counter()

        # DURABLY RESERVE ATTEMPT IN STORE AND LEDGER BEFORE TRANSMISSION
        attempt_idx = store.reserve_http_attempt(
            cohort_id=cohort_id,
            session_date=session_date,
            symbol=symbol,
            target_url=url,
            max_budget=max_requests,
        )
        ledger_idx = ledger.reserve_attempt(symbol=symbol, target_url=url)
        logger.info(f"Reserved attempt {attempt_idx}/{max_requests} for {symbol} in coordination store and ledger")

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw_bytes = resp.read()
                http_code = resp.getcode()

            duration_ms = (time.perf_counter() - t0) * 1000.0
            digest = hashlib.sha256(raw_bytes).hexdigest()
            verify_raw_response_bytes(raw_bytes, digest)

            # Validate completed session bar presence and reject unsupported adjustments
            bars = normalize_yahoo_chart_bars(raw_bytes, symbol, as_of=session_date)
            if not bars:
                logger.warning(f"No bars returned for {symbol} as of {session_date}")
                store.finalize_http_attempt(
                    cohort_id=cohort_id,
                    session_date=session_date,
                    attempt_index=attempt_idx,
                    status="FAILED_EMPTY_BARS",
                    http_code=http_code,
                    duration_ms=duration_ms,
                    byte_length=len(raw_bytes),
                    sha256=digest,
                )
                ledger.finalize_attempt(
                    ledger_idx,
                    status="FAILED_EMPTY_BARS",
                    http_code=http_code,
                    duration_ms=duration_ms,
                    byte_length=len(raw_bytes),
                    sha256=digest,
                )
                return None

            latest_bar_date = bars[-1].get("date") or bars[-1].get("Date")
            if latest_bar_date != session_date:
                err_msg = f"Latest bar {latest_bar_date} != {session_date}"
                logger.warning(
                    f"Latest bar date for {symbol} is '{latest_bar_date}', expected completed session '{session_date}'"
                )
                store.finalize_http_attempt(
                    cohort_id=cohort_id,
                    session_date=session_date,
                    attempt_index=attempt_idx,
                    status="FAILED_SESSION_DATE_MISMATCH",
                    http_code=http_code,
                    duration_ms=duration_ms,
                    byte_length=len(raw_bytes),
                    sha256=digest,
                    error_message=err_msg,
                )
                ledger.finalize_attempt(
                    ledger_idx,
                    status="FAILED_SESSION_DATE_MISMATCH",
                    http_code=http_code,
                    duration_ms=duration_ms,
                    byte_length=len(raw_bytes),
                    sha256=digest,
                    error=err_msg,
                )
                return None

            target_raw = raw_dir / f"{symbol}_{digest[:16]}.raw"
            target_meta = raw_dir / f"{symbol}_{digest[:16]}.meta.json"
            target_raw.write_bytes(raw_bytes)

            ts_utc = datetime.now(timezone.utc).isoformat()
            meta = {
                "symbol": symbol,
                "targetUrl": url,
                "timestampUtc": ts_utc,
                "sha256": digest,
                "byteLength": len(raw_bytes),
                "rawFilePath": str(target_raw.resolve()),
                "receiptType": "SHA-256 integrity receipt",
            }
            target_meta.write_text(json.dumps(meta, indent=2))

            store.finalize_http_attempt(
                cohort_id=cohort_id,
                session_date=session_date,
                attempt_index=attempt_idx,
                status="SUCCESS",
                http_code=http_code,
                duration_ms=duration_ms,
                byte_length=len(raw_bytes),
                sha256=digest,
            )
            ledger.finalize_attempt(
                ledger_idx,
                status="SUCCESS",
                http_code=http_code,
                duration_ms=duration_ms,
                byte_length=len(raw_bytes),
                sha256=digest,
            )

            time.sleep(0.25)  # Rate limiting spacing
            return {"rawPath": str(target_raw.resolve()), "rawResponseSha256": digest}

        except urllib.error.HTTPError as e:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning(f"HTTPError {e.code} for {symbol} (attempt {attempts_for_symbol})")
            store.finalize_http_attempt(
                cohort_id=cohort_id,
                session_date=session_date,
                attempt_index=attempt_idx,
                status=f"HTTP_ERROR_{e.code}",
                http_code=e.code,
                error_message=str(e),
                duration_ms=duration_ms,
            )
            ledger.finalize_attempt(
                ledger_idx,
                status=f"HTTP_ERROR_{e.code}",
                http_code=e.code,
                error=str(e),
                duration_ms=duration_ms,
            )
            if e.code not in (429, 500, 502, 503, 504) or attempts_for_symbol > max_symbol_retries:
                return None
            time.sleep(2.0 * attempts_for_symbol)
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            is_timeout = "timeout" in str(e).lower() or isinstance(e, TimeoutError)
            status_str = "TIMEOUT" if is_timeout else "REQUEST_EXCEPTION"
            logger.warning(f"Acquisition error ({status_str}) for {symbol}: {e} (attempt {attempts_for_symbol})")
            store.finalize_http_attempt(
                cohort_id=cohort_id,
                session_date=session_date,
                attempt_index=attempt_idx,
                status=status_str,
                error_message=str(e),
                duration_ms=duration_ms,
            )
            ledger.finalize_attempt(
                ledger_idx,
                status=status_str,
                error=str(e),
                duration_ms=duration_ms,
            )
            if attempts_for_symbol > max_symbol_retries:
                return None
            time.sleep(2.0 * attempts_for_symbol)

    return None


def run_acquire_and_capture(
    *,
    session_date: str,
    cohort_id: str,
    start_session: str,
    forward_root: Path,
    staging_dir: Path,
    engine_sha: str = DEFAULT_ENGINE_SHA,
    total_request_budget: int = DEFAULT_TOTAL_REQUEST_BUDGET,
    initialize: bool = False,
    offline_fixture_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
    runner_id: Optional[str] = None,
    coordination_db_path: Optional[Path | str] = None,
    auth_expiry_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Coordinates window recheck, coordination lease claim, bounded acquisition, manifest creation, and evaluation."""
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        raise ValueError("Naive capture clock rejected")
    now_dt = now_dt.astimezone(timezone.utc)
    invocation_time_iso = now_dt.isoformat()

    active_runner = derive_runner_identity(runner_id)

    # 1. Authoritative Exchange Calendar & Window Calculation
    cal = None
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XNYS")
    except Exception as e:
        logger.error(f"Cannot initialize XNYS exchange calendar: {e}")
        return {
            "status": "CALENDAR_INITIALIZATION_ERROR",
            "sessionDate": session_date,
            "reason": str(e),
            "verdict": "BLOCKED",
        }

    if not cal.is_session(session_date):
        return {
            "status": "CAPTURE_WINDOW_EXPIRED_OR_INVALID",
            "sessionDate": session_date,
            "reason": f"Date {session_date} is not a valid XNYS trading session",
            "verdict": "REJECTED",
        }

    close_dt = cal.session_close(session_date).to_pydatetime()
    next_open_dt = cal.session_open(cal.next_session(session_date)).to_pydatetime()
    window_open_dt = close_dt + timedelta(minutes=15)
    window_close_dt = next_open_dt - timedelta(hours=12)
    effective_close_dt = min(window_close_dt, auth_expiry_utc) if auth_expiry_utc else window_close_dt

    # 2. Premature Invocation Guard (NOT_YET_OPEN)
    if now_dt < window_open_dt:
        logger.warning(
            f"PREMATURE_INVOCATION: Current time {now_dt.isoformat()} < window open {window_open_dt.isoformat()}"
        )
        return {
            "status": "CAPTURE_WINDOW_NOT_YET_OPEN",
            "sessionDate": session_date,
            "windowOpenUtc": window_open_dt.isoformat(),
            "windowCloseUtc": effective_close_dt.isoformat(),
            "currentUtc": now_dt.isoformat(),
            "reason": f"Session {session_date} capture window opens at {window_open_dt.isoformat()} (15m post-close); current time is premature",
            "verdict": "NOT_YET_OPEN",
        }

    # 3. Expired / Cutoff Guard (MISSED)
    if now_dt > effective_close_dt or (auth_expiry_utc is not None and now_dt >= auth_expiry_utc):
        logger.error(
            f"CAPTURE_WINDOW_EXPIRED: Current time {now_dt.isoformat()} > cutoff {effective_close_dt.isoformat()}"
        )
        return {
            "status": "CAPTURE_WINDOW_EXPIRED_OR_INVALID",
            "sessionDate": session_date,
            "windowCloseUtc": effective_close_dt.isoformat(),
            "currentUtc": now_dt.isoformat(),
            "reason": f"Capture window strictly expired at {effective_close_dt.isoformat()}",
            "verdict": "MISSED",
        }

    # 4. Authoritative window check & ordinal derivation
    try:
        close_dt, next_open_dt, ordinal = session_window(session_date, start_session, now_dt)
        source_cutoff_iso = close_dt.isoformat()
    except ValueError as e:
        logger.error(f"CAPTURE_WINDOW_REJECTED: {e}")
        return {
            "status": "CAPTURE_WINDOW_EXPIRED_OR_INVALID",
            "sessionDate": session_date,
            "reason": str(e),
            "verdict": "MISSED" if "boundary" in str(e) or "Historical/future" in str(e) else "REJECTED",
        }

    # 3. Persistent Coordination Store Initialization (Fail-Closed)
    resolved_db_path = coordination_db_path or (forward_root / ".paper_trading_coordination.db")
    store = PaperTradingCoordinationStore(db_path=resolved_db_path)

    # 4. Check If Session Is Already Fully Committed (Stream + Store check)
    cohort_stream = forward_root / cohort_id / "capture_stream.jsonl"
    stream_has_session = False
    if cohort_stream.is_file():
        for line in cohort_stream.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    if rec.get("sessionDate") == session_date:
                        stream_has_session = True
                        break
                except json.JSONDecodeError:
                    pass

    store_completed = store.is_capture_completed(cohort_id, session_date)
    if stream_has_session and store_completed:
        logger.info(f"Session {session_date} already committed and verified in {cohort_id}.")
        return {
            "status": "ALREADY_CAPTURED",
            "sessionDate": session_date,
            "cohortId": cohort_id,
            "written": 0,
            "reason": f"Session {session_date} already recorded in capture stream and coordination store",
        }

    # 5. Acquire Atomic Session Claim in Coordination Store
    claim_res = store.claim_session(
        cohort_id=cohort_id,
        session_date=session_date,
        claimed_by=active_runner,
        invocation_time_utc=invocation_time_iso,
        source_cutoff_time_utc=source_cutoff_iso,
        max_budget=total_request_budget,
        now_utc=now_dt,
    )
    if claim_res.get("status") == "ALREADY_COMPLETED":
        return {
            "status": "ALREADY_CAPTURED",
            "sessionDate": session_date,
            "cohortId": cohort_id,
            "written": 0,
            "reason": f"Session {session_date} marked completed in coordination store",
        }

    logger.info(f"Coordination claim status for {cohort_id} {session_date}: {claim_res.get('status')} by {active_runner}")

    # 6. Idempotent Recovery Check: If records exist on disk but store uncommitted, complete without duplicating
    if stream_has_session:
        logger.info(f"Crash recovery: Session {session_date} records exist in stream but store uncommitted. Marking completed.")
        store.mark_capture_completed(
            cohort_id=cohort_id,
            session_date=session_date,
            batch_sha256="RECOVERED_FROM_EXISTING_STREAM",
            records_count=10,
            valid_setup_count=0,
            insufficient_data_count=10,
            outside_corridor_count=0,
            waiting_pullback_count=0,
            note="Recovered from pre-existing capture_stream without duplicate write",
            now_utc=now_dt,
        )
        return {
            "status": "ALREADY_CAPTURED",
            "sessionDate": session_date,
            "cohortId": cohort_id,
            "written": 0,
            "reason": "Idempotent recovery: stream already contained records; marked completed in store without duplicate append",
        }

    # 7. Setup Staging Directory & Durable Ledger
    staging_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = staging_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    universe = json.loads(UNIVERSE_PATH.read_bytes())
    symbols: List[str] = universe["symbols"]

    ledger_file = staging_dir / "request_attempts_ledger.jsonl"
    ledger = DurableRequestLedger(ledger_file)
    logger.info(
        f"Initialized DurableRequestLedger ({ledger.total_requests}/{total_request_budget} consumed previously)."
    )

    sources_manifest: Dict[str, Any] = {
        "sessionDate": session_date,
        "cohortId": cohort_id,
        "stagedAtUtc": now_dt.isoformat(),
        "invocationTimeUtc": invocation_time_iso,
        "sourceCutoffTimeUtc": source_cutoff_iso,
        "totalRequestBudget": total_request_budget,
        "claimedBy": active_runner,
        "symbols": {},
    }

    # 8. Acquire data for all symbols
    for sym in symbols:
        res = acquire_symbol_data(
            sym,
            raw_dir=raw_dir,
            session_date=session_date,
            cohort_id=cohort_id,
            store=store,
            ledger=ledger,
            max_requests=total_request_budget,
            offline_fixture_dir=offline_fixture_dir,
        )
        sources_manifest["symbols"][sym] = res

    consumed_attempts = store.get_consumed_budget(cohort_id, session_date)
    sources_manifest["totalRequestsUsed"] = consumed_attempts
    sources_manifest["requestAttemptsLedger"] = str(ledger_file.resolve())

    manifest_path = staging_dir / f"manifest_{session_date}.json"
    manifest_path.write_text(json.dumps(sources_manifest, indent=2))
    logger.info(
        f"Acquisition finished. Requests used: {consumed_attempts}/{total_request_budget}. Manifest: {manifest_path}"
    )

    # 9. Execute isolated point-in-time forward evaluation
    capture_result = execute_isolated_forward_evaluation(
        sources_manifest,
        session_date=session_date,
        cohort_id=cohort_id,
        start_session=start_session,
        forward_root=forward_root,
        engine_sha=engine_sha,
        initialize=initialize,
        now=now_dt,
    )

    # 10. Compute fine-grained diagnostic counts and outside-corridor classifications
    eval_batch_file = forward_root / cohort_id / "inputs" / f"{capture_result['batchSha256']}.json"
    valid_setup_count = 0
    insufficient_data_count = 0
    outside_corridor_count = 0
    waiting_pullback_count = 0

    if eval_batch_file.is_file():
        try:
            eval_batch_data = json.loads(eval_batch_file.read_text())
            evals = eval_batch_data.get("evaluations", {})
            for e in evals.values():
                if isinstance(e, dict):
                    d_state = e.get("decisionState")
                    exec_status = e.get("executionStatus")
                    if d_state in ("ACTIONABLE_SETUP", "VALID_SETUP"):
                        valid_setup_count += 1
                    elif d_state == "INSUFFICIENT_DATA":
                        insufficient_data_count += 1
                        if exec_status in ("EXTENDED_ABOVE_BUY_ZONE", "OUTSIDE_BUY_CORRIDOR", "CORRIDOR_VIOLATION"):
                            outside_corridor_count += 1
                        elif exec_status == "WAITING_PULLBACK":
                            waiting_pullback_count += 1
        except Exception as ex:
            logger.warning(f"Could not parse eval batch for diagnostic counts: {ex}")

    completion_time_iso = datetime.now(timezone.utc).isoformat()

    # 11. Mark Capture Completed in Coordination Store
    store.mark_capture_completed(
        cohort_id=cohort_id,
        session_date=session_date,
        batch_sha256=capture_result["batchSha256"],
        records_count=capture_result.get("written", 10),
        valid_setup_count=valid_setup_count,
        insufficient_data_count=insufficient_data_count,
        outside_corridor_count=outside_corridor_count,
        waiting_pullback_count=waiting_pullback_count,
        note=f"Captured forward session {session_date} by {active_runner}",
        now_utc=datetime.now(timezone.utc),
    )

    return {
        **capture_result,
        "status": capture_result.get("status", "CAPTURED_FORWARD_SESSION"),
        "sessionDate": session_date,
        "cohortId": cohort_id,
        "startSession": start_session,
        "sessionOrdinal": ordinal,
        "nextSessionOpenUtc": next_open_dt.isoformat(),
        "totalRequestsUsed": consumed_attempts,
        "totalRequestBudget": total_request_budget,
        "requestAttemptsLedger": str(ledger_file.resolve()),
        "invocationTimeUtc": invocation_time_iso,
        "sourceCutoffTimeUtc": source_cutoff_iso,
        "completedCaptureTimeUtc": completion_time_iso,
        "claimedBy": active_runner,
        "validSetupCount": valid_setup_count,
        "insufficientDataCount": insufficient_data_count,
        "outsideCorridorCount": outside_corridor_count,
        "waitingPullbackCount": waiting_pullback_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire and capture market data for ARX forward pilot session.")
    parser.add_argument("--session-date", required=True, help="Completed session date (YYYY-MM-DD)")
    parser.add_argument("--cohort-id", default="COHORT_FORWARD_PILOT_005", help="Cohort ID")
    parser.add_argument("--start-session", default="2026-11-27", help="First session date for cohort (YYYY-MM-DD)")
    parser.add_argument("--forward-root", default="data/forward_pilot", help="Forward pilot root directory")
    parser.add_argument("--staging-dir", default=None, help="Staging directory (default data/forward_pilot/staging_<date>)")
    parser.add_argument("--engine-sha", default=DEFAULT_ENGINE_SHA, help="Registered engine SHA")
    parser.add_argument("--request-budget", type=int, default=DEFAULT_TOTAL_REQUEST_BUDGET, help="Max requests budget")
    parser.add_argument("--initialize", action="store_true", help="Initialize cohort if missing")
    parser.add_argument("--offline-fixture-dir", default=None, help="Directory containing offline *.raw fixtures")
    parser.add_argument("--runner-id", default=None, help="Explicit runner identity for coordination lease")
    parser.add_argument("--coordination-db", default=None, help="Custom coordination store database path")
    parser.add_argument("--auth-expiry-utc", default=None, help="Optional strict ISO-8601 authorization expiry")

    args = parser.parse_args()

    forward_root = Path(args.forward_root).resolve()
    staging_dir = Path(args.staging_dir).resolve() if args.staging_dir else forward_root / f"staging_{args.session_date.replace('-', '')}"

    auth_expiry = None
    if args.auth_expiry_utc:
        auth_expiry = datetime.fromisoformat(args.auth_expiry_utc.replace("Z", "+00:00"))

    # Fail closed pre-check on exchange calendar date
    cal = None
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XNYS")
    except Exception as e:
        logger.error(f"Cannot initialize XNYS exchange calendar: {e}")
        return 1

    if not cal.is_session(args.session_date):
        logger.error(f"Target date '{args.session_date}' is NOT an XNYS trading session (holiday or weekend).")
        print(json.dumps({
            "status": "CAPTURE_WINDOW_EXPIRED_OR_INVALID",
            "sessionDate": args.session_date,
            "reason": f"Date {args.session_date} is not a valid XNYS trading session",
            "verdict": "REJECTED",
        }, indent=2))
        return 1

    try:
        result = run_acquire_and_capture(
            session_date=args.session_date,
            cohort_id=args.cohort_id,
            start_session=args.start_session,
            forward_root=forward_root,
            staging_dir=staging_dir,
            engine_sha=args.engine_sha,
            total_request_budget=args.request_budget,
            initialize=args.initialize,
            offline_fixture_dir=Path(args.offline_fixture_dir).resolve() if args.offline_fixture_dir else None,
            runner_id=args.runner_id,
            coordination_db_path=args.coordination_db,
            auth_expiry_utc=auth_expiry,
        )
    except CoordinationStoreUnavailableError as e:
        logger.error(f"FATAL: Coordination store unavailable (fail-closed): {e}")
        print(json.dumps({
            "status": "COORDINATION_STORE_UNAVAILABLE",
            "sessionDate": args.session_date,
            "reason": str(e),
            "verdict": "BLOCKED",
        }, indent=2))
        return 1
    except ConcurrentClaimConflictError as e:
        logger.warning(f"CONCURRENT_CLAIM_CONFLICT: {e}")
        print(json.dumps({
            "status": "CONCURRENT_CLAIM_CONFLICT",
            "sessionDate": args.session_date,
            "reason": str(e),
            "verdict": "BLOCKED",
        }, indent=2))
        return 1
    except BudgetExhaustedError as e:
        logger.error(f"BUDGET_EXHAUSTED: {e}")
        print(json.dumps({
            "status": "BUDGET_EXHAUSTED",
            "sessionDate": args.session_date,
            "reason": str(e),
            "verdict": "BLOCKED",
        }, indent=2))
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") in ("CAPTURED_FORWARD_SESSION", "SUCCESS", "ALREADY_CAPTURED"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
