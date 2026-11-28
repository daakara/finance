#!/usr/bin/env python3
"""Acquire completed-session market data and execute bounded forward capture.

Governed by:
- ARX Paper-Trading Outcome Evaluation Contract V1.0.1
- docs/planning/ARX_PAPER_TRADING_ROADMAP.md (PR #322)

Enforces:
1. Runtime window recheck (session_window) at actual execution time.
2. Hard calendar-day cutoff in America/New_York (no backdating or retrospective replay).
3. Explicit total request budget counting initial requests, retries, and failures.
4. Raw response byte integrity, content-addressed storage, and split rejection.
5. Non-destructive staging (preserves prior staging evidence).
6. Complete 10-symbol accounting: records missing/failed inputs without fabrication.
7. Next-session execution eligibility: decisions apply to T+1 open, never observed session.
8. Duplicate capture protection under POSIX locking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
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

logger = logging.getLogger("arx.governance.acquire_and_capture")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_ENGINE_SHA = "05fb5a320501b7bdd32630bc3d18c3a719cef626"
DEFAULT_TOTAL_REQUEST_BUDGET = 12  # 10 symbols + max 2 retries across session
YAHOO_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


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
    ledger: DurableRequestLedger,
    max_requests: int,
    offline_fixture_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Acquires and validates raw chart bytes for one symbol respecting durable request budget."""
    # Check if already acquired in an earlier attempt/restart
    existing = ledger.is_symbol_acquired(symbol, raw_dir, session_date)
    if existing:
        return existing

    if offline_fixture_dir is not None:
        # Offline test / replay path
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

    # Live network acquisition
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=6mo&interval=1d&events=history&includeAdjustedClose=true"
    headers = {"User-Agent": YAHOO_USER_AGENT}

    max_symbol_retries = 2
    attempts_for_symbol = 0

    while attempts_for_symbol <= max_symbol_retries:
        if ledger.total_requests >= max_requests:
            logger.error(
                f"Total request budget exhausted ({ledger.total_requests}/{max_requests}). Halting acquisition."
            )
            return None

        attempts_for_symbol += 1
        t0 = time.perf_counter()

        # DURABLY RESERVE ATTEMPT BEFORE TRANSMISSION
        attempt_idx = ledger.reserve_attempt(symbol=symbol, target_url=url)
        logger.info(f"Reserved attempt {attempt_idx}/{max_requests} for {symbol} on disk before transmission")

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
                ledger.finalize_attempt(
                    attempt_idx,
                    status="FAILED_EMPTY_BARS",
                    http_code=http_code,
                    duration_ms=duration_ms,
                    byte_length=len(raw_bytes),
                    sha256=digest,
                )
                return None

            latest_bar_date = bars[-1].get("date") or bars[-1].get("Date")
            if latest_bar_date != session_date:
                logger.warning(
                    f"Latest bar date for {symbol} is '{latest_bar_date}', expected completed session '{session_date}'"
                )
                ledger.finalize_attempt(
                    attempt_idx,
                    status="FAILED_SESSION_DATE_MISMATCH",
                    http_code=http_code,
                    duration_ms=duration_ms,
                    byte_length=len(raw_bytes),
                    sha256=digest,
                    error=f"Latest bar {latest_bar_date} != {session_date}",
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

            ledger.finalize_attempt(
                attempt_idx,
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
            ledger.finalize_attempt(
                attempt_idx,
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
            logger.warning(f"Acquisition error for {symbol}: {e} (attempt {attempts_for_symbol})")
            ledger.finalize_attempt(
                attempt_idx,
                status="REQUEST_EXCEPTION",
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
) -> Dict[str, Any]:
    """Coordinates window recheck, bounded acquisition, manifest creation, and evaluation."""
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        raise ValueError("Naive capture clock rejected")
    now_dt = now_dt.astimezone(timezone.utc)

    AUTH_EXPIRY_UTC = datetime.fromisoformat("2026-11-28T05:00:00+00:00")
    if now_dt >= AUTH_EXPIRY_UTC:
        logger.error(
            f"AUTHORIZATION_EXPIRED: Current time {now_dt.isoformat()} >= expiry {AUTH_EXPIRY_UTC.isoformat()}"
        )
        return {
            "status": "CAPTURE_WINDOW_EXPIRED_OR_INVALID",
            "sessionDate": session_date,
            "reason": f"Authorization strictly expired at {AUTH_EXPIRY_UTC.isoformat()}",
            "verdict": "MISSED",
        }

    # 1. Recheck permitted capture window at runtime
    try:
        close_dt, next_open_dt, ordinal = session_window(session_date, start_session, now_dt)
    except ValueError as e:
        logger.error(f"CAPTURE_WINDOW_REJECTED: {e}")
        return {
            "status": "CAPTURE_WINDOW_EXPIRED_OR_INVALID",
            "sessionDate": session_date,
            "reason": str(e),
            "verdict": "MISSED" if "Historical/future" in str(e) or "boundary" in str(e) else "REJECTED",
        }

    # 2. Check duplicate capture in existing stream
    cohort_stream = forward_root / cohort_id / "capture_stream.jsonl"
    if cohort_stream.is_file():
        for line in cohort_stream.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    if rec.get("sessionDate") == session_date:
                        logger.info(f"Session {session_date} already committed in {cohort_id}.")
                        return {
                            "status": "ALREADY_CAPTURED",
                            "sessionDate": session_date,
                            "cohortId": cohort_id,
                            "written": 0,
                            "reason": f"Session {session_date} already recorded in capture stream",
                        }
                except json.JSONDecodeError:
                    pass

    # 3. Setup non-destructive staging directory
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
        "totalRequestBudget": total_request_budget,
        "symbols": {},
    }

    # 4. Acquire data for all symbols
    for sym in symbols:
        res = acquire_symbol_data(
            sym,
            raw_dir=raw_dir,
            session_date=session_date,
            ledger=ledger,
            max_requests=total_request_budget,
            offline_fixture_dir=offline_fixture_dir,
        )
        sources_manifest["symbols"][sym] = res

    sources_manifest["totalRequestsUsed"] = ledger.total_requests
    sources_manifest["requestAttemptsLedger"] = str(ledger_file.resolve())

    manifest_path = staging_dir / f"manifest_{session_date}.json"
    manifest_path.write_text(json.dumps(sources_manifest, indent=2))
    logger.info(
        f"Acquisition finished. Requests used: {ledger.total_requests}/{total_request_budget}. Manifest: {manifest_path}"
    )

    # 5. Execute isolated point-in-time forward evaluation
    capture_result = execute_isolated_forward_evaluation(
        sources_manifest,
        cohort_id=cohort_id,
        start_session=start_session,
        forward_root=forward_root,
        session_date=session_date,
        engine_sha=engine_sha,
        initialize=initialize,
        now=now_dt,
    )

    return {
        **capture_result,
        "status": capture_result.get("status", "CAPTURED_FORWARD_SESSION"),
        "sessionDate": session_date,
        "cohortId": cohort_id,
        "startSession": start_session,
        "sessionOrdinal": ordinal,
        "nextSessionOpenUtc": next_open_dt.isoformat(),
        "totalRequestsUsed": ledger.total_requests,
        "totalRequestBudget": total_request_budget,
        "requestAttemptsLedger": str(ledger_file.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-date", required=True, help="Completed session date (YYYY-MM-DD)")
    parser.add_argument("--cohort-id", required=True, help="Forward cohort ID (COHORT_FORWARD_PILOT_*)")
    parser.add_argument("--start-session", required=True, help="Cohort start session date (YYYY-MM-DD)")
    parser.add_argument("--forward-root", type=Path, default=REPO_ROOT / "data" / "forward_pilot")
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument("--engine-sha", default=DEFAULT_ENGINE_SHA)
    parser.add_argument("--request-budget", type=int, default=DEFAULT_TOTAL_REQUEST_BUDGET)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--offline-fixture-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-check and exit without writes")

    args = parser.parse_args()

    staging_dir = args.staging_dir or (
        args.forward_root / f"staging_{args.session_date.replace('-', '')}_completed"
    )

    if args.dry_run:
        now_dt = datetime.now(timezone.utc)
        AUTH_EXPIRY_UTC = datetime.fromisoformat("2026-11-28T05:00:00+00:00")
        if now_dt >= AUTH_EXPIRY_UTC:
            print(json.dumps({
                "status": "DRY_RUN_WINDOW_CLOSED",
                "sessionDate": args.session_date,
                "reason": f"Authorization strictly expired at {AUTH_EXPIRY_UTC.isoformat()}",
                "verdict": "MISSED",
            }, indent=2))
            return 1
        try:
            close_dt, next_open_dt, ordinal = session_window(args.session_date, args.start_session, now_dt)
            print(json.dumps({
                "status": "DRY_RUN_WINDOW_OPEN",
                "sessionDate": args.session_date,
                "startSession": args.start_session,
                "sessionOrdinal": ordinal,
                "sessionCloseUtc": close_dt.isoformat(),
                "nextSessionOpenUtc": next_open_dt.isoformat(),
                "currentUtc": now_dt.isoformat(),
                "verdict": "READY_FOR_BOUNDED_CAPTURE",
            }, indent=2))
            return 0
        except ValueError as e:
            print(json.dumps({
                "status": "DRY_RUN_WINDOW_CLOSED",
                "sessionDate": args.session_date,
                "reason": str(e),
                "verdict": "MISSED" if "Historical/future" in str(e) or "boundary" in str(e) else "REJECTED",
            }, indent=2))
            return 1

    result = run_acquire_and_capture(
        session_date=args.session_date,
        cohort_id=args.cohort_id,
        start_session=args.start_session,
        forward_root=args.forward_root,
        staging_dir=staging_dir,
        engine_sha=args.engine_sha,
        total_request_budget=args.request_budget,
        initialize=args.initialize,
        offline_fixture_dir=args.offline_fixture_dir,
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") in ("CAPTURED_FORWARD_SESSION", "SUCCESS", "ALREADY_CAPTURED"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
