#!/usr/bin/env python3
"""Bounded AAPL/NVDA no-work rehearsal with explicit authority and audit receipts.

Historical operator-reported source digest (not independently attested here):
2df294a727467452931eedbd4c781492554189c99f6db3447707f300d957c6e3 (7782 bytes).
This version is distinct from the historical executed wrapper. The fixed historical
plan expects preservation, not ingestion additions; any row-content delta fails
acceptance. A future write trial requires its own accession-bound acceptance plan.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyst_dashboard.data.sec_edgar_fetcher import SecRateLimiter
from analyst_dashboard.smart_money.canary_runner import (
    NvdaCanaryRunner, CanaryDataIntegrityError, CanarySecurityError,
    CanaryBudgetExceededError,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    LOCK_SEC_EGRESS_OWNER, LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID,
)
import psycopg2
from analyst_dashboard.smart_money.storage.postgres_preservation import (
    compute_database_preservation_fingerprint, diff_preservation_fingerprints,
    CANONICAL_PROTECTED_TABLES,
)

logger = logging.getLogger("sequential_canary_pair")
HISTORICAL_EXECUTED_WRAPPER_SHA256 = "2df294a727467452931eedbd4c781492554189c99f6db3447707f300d957c6e3"
HISTORICAL_EXECUTED_WRAPPER_SIZE_BYTES = 7782
PAIR_MAX_REQUESTS_CEILING = 8
PAIR_MAX_DURATION_SEC_CEILING = 60.0
PAIR_MAX_RPS_CEILING = 5.0
INTER_ISSUER_MIN_GAP_SEC = 1.0


def build_pair_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog or "execute_sequential_canary_pair.py")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Read-only checks; no SEC calls or writes.")
    mode.add_argument("--live", action="store_true", help="Explicitly acknowledge scoped live execution.")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--auth-issued-at", required=True, help="Explicit timezone-aware authorization timestamp.")
    parser.add_argument("--auth-expires-at", required=True, help="Explicit expiry, at most 15 minutes after issue.")
    parser.add_argument("--timeout", dest="max_duration_sec", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _validate_authority(dry_run, live, issued, expires, budget):
    if not isinstance(dry_run, bool) or not isinstance(live, bool) or dry_run == live:
        raise CanarySecurityError("EXPLICIT_MODE_REQUIRED: select exactly one of dry_run or live")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or not 0 < budget <= 60.0:
        raise CanaryBudgetExceededError("INVALID_AGGREGATE_BUDGET: require finite 0 < timeout <= 60")
    if not all(isinstance(t, datetime) and t.tzinfo is not None and t.utcoffset() is not None for t in (issued, expires)):
        raise CanarySecurityError("EXPLICIT_AUTHORIZATION_REQUIRED: timezone-aware issue and expiry required")
    now = datetime.now(timezone.utc)
    if not issued <= now < expires or not 0 < (expires - issued).total_seconds() <= 900:
        raise CanarySecurityError("INVALID_AUTHORIZATION_WINDOW")


def _inspect_database(dsn, remaining, scope):
    """Read a consistent state and real lock state; never infer successful cleanup."""
    if remaining <= 0:
        raise CanaryBudgetExceededError("Aggregate deadline expired before database inspection")
    # Connection and SQL operations are bounded; any overrun still fails final acceptance.
    conn = psycopg2.connect(dsn=dsn, connect_timeout=max(1, math.ceil(min(remaining, 10))))
    try:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('statement_timeout', %s, true)", (str(max(1, int(min(remaining, 10) * 1000))),))
            cur.execute("""SELECT count(*) FROM pg_locks WHERE locktype='advisory'
                AND classid=%s AND objid=%s AND objsubid=1 AND granted""",
                (LOCK_SEC_EGRESS_OWNER_CLASSID, LOCK_SEC_EGRESS_OWNER_OBJID))
            held = cur.fetchone()[0] != 0
        fingerprint = compute_database_preservation_fingerprint(conn, scope=scope)
        missing = set(CANONICAL_PROTECTED_TABLES) - set(fingerprint.get("tables", {}))
        if missing:
            raise CanarySecurityError("PRESERVATION_TABLES_MISSING: " + ",".join(sorted(missing)))
        return fingerprint, held
    finally:
        conn.close()


def execute_sequential_pair(
    dsn: str, dry_run: bool = False,
    auth_issued_at: Optional[datetime] = None,
    auth_expires_at: Optional[datetime] = None,
    sessions: Optional[Dict[str, Any]] = None,
    shared_limiter=None, max_duration_sec: float = 60.0, *, live: bool = False,
) -> Dict[str, Any]:
    """No implicit execution or authorization. Failures carry a forensic pair_receipt.

    Each issuer commits independently. Postflight failure cannot undo an earlier
    issuer commit and never claims that it did. Cleanup/evidence collection may
    extend beyond the execution deadline; such overruns cannot return SUCCESS.
    """
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    runners, results, errors = [], [], []
    before = after = difference = None
    lock_held = None
    failure = None
    status = "FAILED"
    pair_receipt = None

    def remaining():
        return max_duration_sec - (time.monotonic() - started)

    _validate_authority(dry_run, live, auth_issued_at, auth_expires_at, max_duration_sec)
    try:
        before, initially_held = _inspect_database(dsn, remaining(), "SEQUENTIAL_PAIR_PRE_EXECUTION")
        if initially_held:
            raise CanarySecurityError("ADVISORY_LOCK_BUSY")
        if remaining() <= 0:
            raise CanaryBudgetExceededError("Aggregate deadline expired during preflight")
        limiter = shared_limiter or SecRateLimiter(max_per_second=5.0)
        plan = (("AAPL", "0000320193", "0001140361-26-044773"), ("NVDA", "0001045810", None))
        for idx, (ticker, cik, accession) in enumerate(plan):
            if idx:
                if remaining() <= INTER_ISSUER_MIN_GAP_SEC:
                    raise CanaryBudgetExceededError("Insufficient aggregate budget for inter-issuer gap")
                time.sleep(INTER_ISSUER_MIN_GAP_SEC)
            if remaining() <= 0:
                raise CanaryBudgetExceededError("Sequential pair aggregate deadline exceeded")
            # Use the SAME supplied authorization for both issuers; never renew it.
            os.environ["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"
            runner = NvdaCanaryRunner(
                dsn=dsn, session=sessions.get(ticker) if sessions else None,
                ticker=ticker, cik=cik, accession=accession,
                interval_start="2026-11-17T00:00:00Z", interval_end="2026-11-18T23:59:59Z",
                public_availability_cutoff="2026-11-19T00:00:00Z",
                auth_issued_at=auth_issued_at, auth_expires_at=auth_expires_at,
                max_requests=4, max_rps=5.0, max_unseen_docs=3,
                max_duration_sec=min(30.0, remaining()),
            )
            runner.rate_limiter = limiter
            runners.append(runner)
            try:
                result = runner.execute_dry_run() if dry_run else runner.execute()
            except CanaryDataIntegrityError as exc:
                if not str(exc).startswith("NO_UNSEEN_ELIGIBLE_FILINGS:"):
                    raise
                result = {"issuer": ticker, "cik": cik, "accession": accession,
                          "verdict": "NO_ELIGIBLE_WORK", "reason": "NO_UNSEEN_ELIGIBLE_FILINGS",
                          "requests_made": runner.request_count, "transactions_persisted": 0,
                          "detail": str(exc)}
            except BaseException as exc:
                results.append({"issuer": ticker, "cik": cik, "verdict": "FAILED", "error": str(exc)})
                raise
            results.append(result)
            if dry_run and (result.get("verdict") != "SUCCESS" or result.get("preflight_verdict") != "PASSED"):
                raise CanarySecurityError(ticker + " dry-run preflight failed")
            if sum(r.request_count for r in runners) > 8:
                raise CanaryBudgetExceededError("Aggregate request limit exceeded")
            if remaining() <= 0:
                raise CanaryBudgetExceededError("Sequential pair aggregate deadline exceeded")
        status = "SUCCESS"
    except BaseException as exc:
        failure = exc
        errors.append(str(exc))
    finally:
        os.environ["ARX_SEC_EGRESS_ROLE"] = "STANDBY"
        os.environ["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "false"
        # Always attempt evidence collection, including after SIGINT and issuer errors.
        # Forensic cleanup gets a bounded inspection even after the execution deadline.
        try:
            after, lock_held = _inspect_database(dsn, 10.0, "SEQUENTIAL_PAIR_POST_EXECUTION")
            if lock_held:
                errors.append("ADVISORY_LOCK_STILL_HELD")
            if before is not None:
                difference = diff_preservation_fingerprints(before, after)
                if not difference["matched"]:
                    errors.append("PRESERVATION_MISMATCH: historical no-work plan changed protected contents")
            else:
                errors.append("PRE_EXECUTION_FINGERPRINT_UNAVAILABLE")
        except BaseException as exc:
            errors.append("POST_EXECUTION_VERIFICATION_FAILED: " + str(exc))
        duration = time.monotonic() - started
        valid_budget = isinstance(max_duration_sec, (int, float)) and not isinstance(max_duration_sec, bool) and math.isfinite(max_duration_sec)
        if not valid_budget or duration > max_duration_sec:
            errors.append("AGGREGATE_DEADLINE_EXCEEDED: total operation includes postflight")
        if errors:
            status = "FAILED"
        pair_receipt = {
            "receipt_type": "ARX_TWO_ISSUER_SEQUENTIAL_CANARY_RECEIPT", "version": "1.2.0",
            "status": status, "mode": "DRY_RUN" if dry_run else "LIVE" if live else "UNSPECIFIED",
            "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
            "aggregate_duration_seconds": round(duration, 6),
            "aggregate_duration_ceiling_seconds": max_duration_sec if valid_budget else None,
            "aggregate_requests_made": sum(r.request_count for r in runners),
            "aggregate_requests_ceiling": 8, "shared_rate_limiter_max_rps": 5.0,
            "issuer_results": results, "issuers_executed": [r["issuer"] for r in results],
            "fatal_error": str(failure) if failure else None, "verification_errors": errors,
            "transaction_scope": "PER_ISSUER; prior commits are not rolled back by later failure",
            "safety_invariants": {"worker_role": os.environ.get("ARX_SEC_EGRESS_ROLE"),
                "containment_authorized": os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"),
                "advisory_lock_id": LOCK_SEC_EGRESS_OWNER, "advisory_lock_held": lock_held,
                "lock_verification": "NOT_CHECKED" if lock_held is None else "FAIL" if lock_held else "PASS"},
            "preservation_fingerprints": {"before": before, "after": after, "diff": difference},
        }
    if failure is not None:
        failure.pair_receipt = pair_receipt
        raise failure
    return pair_receipt


def main(argv=None) -> int:
    args = build_pair_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    dsn = args.dsn or os.environ.get("SMART_MONEY_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("FATAL: PostgreSQL DSN is required", file=sys.stderr)
        return 1
    try:
        issued = datetime.fromisoformat(args.auth_issued_at.replace("Z", "+00:00"))
        expires = datetime.fromisoformat(args.auth_expires_at.replace("Z", "+00:00"))
        receipt = execute_sequential_pair(dsn, dry_run=args.dry_run, live=args.live,
            auth_issued_at=issued, auth_expires_at=expires, max_duration_sec=args.max_duration_sec)
    except BaseException as exc:
        receipt = getattr(exc, "pair_receipt", None)
        if receipt is not None:
            print(json.dumps(receipt, indent=2, default=str))
        print("FATAL: " + str(exc), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    print(json.dumps(receipt, indent=2, default=str))
    return 0 if receipt["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
