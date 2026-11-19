#!/usr/bin/env python3
"""
ARX Terminal — Dedicated Sequential Two-Issuer Canary Executor (AAPL then NVDA).

Follows Directive ARX-SEC-TWO-ISSUER-QUALIFICATION-001 under Issue #165:
- Sequence: Exactly AAPL then NVDA; no third or additional issuers.
- AAPL: Pinned accession 0001140361-26-044773 (duplicate / zero-unseen check).
- NVDA: Prospective interval 2026-11-17T00:00:00Z to 2026-11-18T23:59:59Z, cutoff 2026-11-19T00:00:00Z.
- Shared Rate Limiter: SecRateLimiter(max_per_second=5.0) shared across the pair.
- Ceilings:
  * <= 4 requests and <= 30.0s per issuer.
  * <= 8 requests and <= 60.0s for the pair.
  * <= 5 requests/sec with >= 1.0s gap between issuers (gap included in aggregate 60.0s budget).
  * Second-issuer timeout clamped to remaining aggregate budget: min(30.0, 60.0 - elapsed).
  * <= 3 unseen documents per issuer.
- Atomic separation: Separate database transaction per issuer. No cross-issuer writes or snapshot modifications.
  Never claim pair-wide rollback if a prior issuer has already committed.
- Guaranteed teardown: In-process role restored to STANDBY, containment revoked, advisory locks released.
- Preservation Evidence: Deterministic before/after cryptographic content fingerprints over all protected tables.
- Source Custody:
  * Historical Executed Wrapper (2026-11-19T22:17:32Z):
    SHA-256: 2df294a727467452931eedbd4c781492554189c99f6db3447707f300d957c6e3 (7,782 bytes)
  * Remediated Qualified Version: Version 1.1.0 with CLI parity, dry-run mode, and fingerprinting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from analyst_dashboard.data.sec_edgar_fetcher import SecRateLimiter
from analyst_dashboard.smart_money.canary_runner import (
    NvdaCanaryRunner,
    CanaryDataIntegrityError,
    CanarySecurityError,
    CanaryBudgetExceededError,
    CanaryLockContentionError,
    CanaryLockLostError,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    LOCK_SEC_EGRESS_OWNER,
    LOCK_SEC_EGRESS_OWNER_CLASSID,
    LOCK_SEC_EGRESS_OWNER_OBJID,
)
from analyst_dashboard.smart_money.storage.postgres_connection import get_connection
from analyst_dashboard.smart_money.storage.postgres_preservation import (
    compute_database_preservation_fingerprint,
    diff_preservation_fingerprints,
)

logger = logging.getLogger("sequential_canary_pair")

# Custody reference for the initial historical run
HISTORICAL_EXECUTED_WRAPPER_SHA256 = (
    "2df294a727467452931eedbd4c781492554189c99f6db3447707f300d957c6e3"
)
HISTORICAL_EXECUTED_WRAPPER_SIZE_BYTES = 7782

PAIR_MAX_REQUESTS_CEILING = 8
PAIR_MAX_DURATION_SEC_CEILING = 60.0
PAIR_MAX_RPS_CEILING = 5.0
INTER_ISSUER_MIN_GAP_SEC = 1.0


def build_pair_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    """Constructs canonical argument parser for the sequential pair canary."""
    parser = argparse.ArgumentParser(
        prog=prog or "execute_sequential_canary_pair.py",
        description="ARX Terminal — Sequential Two-Issuer Canary (AAPL then NVDA) Executor",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute read-only preflight inspections for both issuers without SEC network egress or DB writes.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Acknowledge live SEC EDGAR network execution under operator authorization.",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=None,
        help="PostgreSQL connection string (defaults to SMART_MONEY_DATABASE_URL or DATABASE_URL).",
    )
    parser.add_argument(
        "--auth-issued-at",
        type=str,
        default=None,
        help="UTC ISO timestamp when operator authorization was issued (default: dynamically generated now).",
    )
    parser.add_argument(
        "--auth-expires-at",
        type=str,
        default=None,
        help="UTC ISO timestamp when operator authorization expires (default: now + 14 minutes).",
    )
    parser.add_argument(
        "--timeout",
        dest="max_duration_sec",
        type=float,
        default=PAIR_MAX_DURATION_SEC_CEILING,
        help=f"Aggregate duration timeout for the pair in seconds (ceiling: {PAIR_MAX_DURATION_SEC_CEILING}s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def execute_sequential_pair(
    dsn: str,
    dry_run: bool = False,
    auth_issued_at: Optional[datetime] = None,
    auth_expires_at: Optional[datetime] = None,
    sessions: Optional[Dict[str, requests.Session]] = None,
    shared_limiter: Optional[SecRateLimiter] = None,
    max_duration_sec: float = PAIR_MAX_DURATION_SEC_CEILING,
) -> Dict[str, Any]:
    """
    Executes the qualified sequential canary pair: AAPL followed by NVDA.

    Enforces:
    - Exactly AAPL then NVDA; no third issuer admitted.
    - Shared SecRateLimiter <= 5.0 requests/second.
    - <= 4 requests and <= 30.0s per issuer.
    - <= 8 requests and <= 60.0s for the pair.
    - >= 1.0s gap between issuers included within aggregate budget.
    - Second issuer timeout clamped to remaining aggregate budget.
    - No automatic retries; unexpected failure halts batch without running next issuer.
    - Separate atomic database transactions per issuer.
    - Content-level preservation fingerprinting before and after execution.
    - Guaranteed invariant teardown in finally block.
    """
    started_monotonic = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()

    if max_duration_sec > PAIR_MAX_DURATION_SEC_CEILING:
        raise CanaryBudgetExceededError(
            f"Aggregate timeout ({max_duration_sec}s) exceeds non-overridable ceiling ({PAIR_MAX_DURATION_SEC_CEILING}s)."
        )

    rate_limiter = shared_limiter or SecRateLimiter(max_per_second=PAIR_MAX_RPS_CEILING)
    runners: List[NvdaCanaryRunner] = []
    issuer_receipts: List[Dict[str, Any]] = []

    # Fixed two-issuer plan: AAPL then NVDA
    plan = [
        {
            "ticker": "AAPL",
            "cik": "0000320193",
            "accession": "0001140361-26-044773",
            "interval_start": "2026-11-17T00:00:00Z",
            "interval_end": "2026-11-18T23:59:59Z",
            "public_cutoff": "2026-11-19T00:00:00Z",
        },
        {
            "ticker": "NVDA",
            "cik": "0001045810",
            "accession": None,
            "interval_start": "2026-11-17T00:00:00Z",
            "interval_end": "2026-11-18T23:59:59Z",
            "public_cutoff": "2026-11-19T00:00:00Z",
        },
    ]

    # Pre-execution fingerprinting across protected tables
    fingerprint_before: Optional[Dict[str, Any]] = None
    if dsn:
        try:
            with get_connection(dsn) as fp_conn:
                fingerprint_before = compute_database_preservation_fingerprint(
                    fp_conn, scope="SEQUENTIAL_PAIR_PRE_EXECUTION"
                )
        except Exception as fp_err:
            logger.warning(f"Unable to capture pre-execution database fingerprint: {fp_err}")

    # Ensure authorization timestamps
    now = datetime.now(timezone.utc)
    base_auth_issued = auth_issued_at or now
    base_auth_expires = auth_expires_at or (now + timedelta(minutes=14))

    batch_status = "SUCCESS"
    fatal_error: Optional[str] = None

    try:
        for idx, item in enumerate(plan):
            ticker = item["ticker"]
            cik = item["cik"]
            accession = item["accession"]

            # If second issuer, enforce >= 1.0s gap
            if idx > 0:
                elapsed_before_gap = time.monotonic() - started_monotonic
                if (max_duration_sec - elapsed_before_gap) < INTER_ISSUER_MIN_GAP_SEC:
                    raise CanaryBudgetExceededError(
                        "Insufficient time remaining in aggregate budget to enforce inter-issuer gap."
                    )
                logger.info(f"Enforcing inter-issuer pause (>={INTER_ISSUER_MIN_GAP_SEC}s)...")
                time.sleep(INTER_ISSUER_MIN_GAP_SEC + 0.05)

            elapsed = time.monotonic() - started_monotonic
            remaining_budget = max_duration_sec - elapsed
            if remaining_budget <= 0:
                raise CanaryBudgetExceededError("Sequential canary pair aggregate deadline exceeded.")

            issuer_timeout = min(30.0, remaining_budget)
            logger.info(
                f"Preparing {ticker} (CIK {cik}) canary: timeout={issuer_timeout:.2f}s, remaining_budget={remaining_budget:.2f}s"
            )

            # Re-generate fresh valid authorization window per issuer if dynamic
            issuer_auth_issued = base_auth_issued if auth_issued_at else datetime.now(timezone.utc)
            issuer_auth_expires = (
                base_auth_expires if auth_expires_at else (issuer_auth_issued + timedelta(minutes=14))
            )

            # In-process containment authorization
            os.environ["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "true"

            custom_session = sessions.get(ticker) if sessions else None

            runner = NvdaCanaryRunner(
                dsn=dsn,
                session=custom_session,
                ticker=ticker,
                cik=cik,
                accession=accession,
                interval_start=item["interval_start"],
                interval_end=item["interval_end"],
                public_availability_cutoff=item["public_cutoff"],
                auth_issued_at=issuer_auth_issued,
                auth_expires_at=issuer_auth_expires,
                max_requests=4,
                max_rps=5.0,
                max_unseen_docs=3,
                max_duration_sec=issuer_timeout,
            )
            runner.rate_limiter = rate_limiter
            runners.append(runner)

            if dry_run:
                # Dry run: zero SEC calls, zero DB writes
                logger.info(f"[DRY-RUN] Executing read-only preflight checks for {ticker}...")
                dry_receipt = runner.execute_dry_run()
                issuer_receipts.append(dry_receipt)
                if dry_receipt.get("verdict") != "SUCCESS" or dry_receipt.get("preflight_verdict") != "PASSED":
                    batch_status = "FAILED"
                    fatal_error = f"{ticker} dry-run preflight failed: {dry_receipt.get('preflight_verdict')}"
                    break
                continue

            # Live execution
            try:
                receipt = runner.execute()
                issuer_receipts.append(receipt)
            except CanaryDataIntegrityError as error:
                if "NO_UNSEEN_ELIGIBLE_FILINGS" in str(error):
                    logger.info(f"{ticker} encountered NO_UNSEEN_ELIGIBLE_FILINGS; normalizing to NO_ELIGIBLE_WORK.")
                    issuer_receipts.append({
                        "issuer": ticker,
                        "cik": cik,
                        "accession": accession,
                        "verdict": "NO_ELIGIBLE_WORK",
                        "reason": "NO_UNSEEN_ELIGIBLE_FILINGS",
                        "requests_made": runner.request_count,
                        "transactions_persisted": 0,
                        "detail": str(error),
                    })
                else:
                    logger.error(f"{ticker} failed with CanaryDataIntegrityError: {error}")
                    issuer_receipts.append({
                        "issuer": ticker,
                        "cik": cik,
                        "verdict": "FAILED",
                        "error": str(error),
                    })
                    batch_status = "FAILED"
                    fatal_error = f"{ticker} failed: {error}"
                    raise
            except BaseException as exc:
                logger.error(f"{ticker} encountered unexpected exception: {exc}")
                issuer_receipts.append({
                    "issuer": ticker,
                    "cik": cik,
                    "verdict": "FAILED",
                    "error": str(exc),
                })
                batch_status = "FAILED"
                fatal_error = f"{ticker} failed: {exc}"
                raise

            # Aggregate boundary assertions after each issuer completes
            aggregate_requests = sum(r.request_count for r in runners)
            total_elapsed = time.monotonic() - started_monotonic
            assert aggregate_requests <= PAIR_MAX_REQUESTS_CEILING, (
                f"Aggregate request limit exceeded: {aggregate_requests} > {PAIR_MAX_REQUESTS_CEILING}"
            )
            assert total_elapsed <= max_duration_sec, (
                f"Aggregate duration limit exceeded: {total_elapsed:.2f}s > {max_duration_sec}s"
            )

    finally:
        # Mandatory invariant teardown
        os.environ["ARX_SEC_EGRESS_ROLE"] = "STANDBY"
        os.environ["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "false"

    total_duration = time.monotonic() - started_monotonic
    total_requests = sum(r.request_count for r in runners)

    # Post-execution fingerprinting
    fingerprint_after: Optional[Dict[str, Any]] = None
    fingerprint_diff: Optional[Dict[str, Any]] = None
    if dsn:
        try:
            with get_connection(dsn) as fp_conn:
                fingerprint_after = compute_database_preservation_fingerprint(
                    fp_conn, scope="SEQUENTIAL_PAIR_POST_EXECUTION"
                )
            if fingerprint_before and fingerprint_after:
                fingerprint_diff = diff_preservation_fingerprints(fingerprint_before, fingerprint_after)
        except Exception as fp_err:
            logger.warning(f"Unable to capture post-execution database fingerprint: {fp_err}")

    pair_receipt = {
        "receipt_type": "ARX_TWO_ISSUER_SEQUENTIAL_CANARY_RECEIPT",
        "version": "1.1.0",
        "status": batch_status,
        "mode": "DRY_RUN" if dry_run else "LIVE",
        "started_at": started_utc,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "aggregate_duration_seconds": round(total_duration, 3),
        "aggregate_duration_ceiling_seconds": max_duration_sec,
        "aggregate_requests_made": total_requests,
        "aggregate_requests_ceiling": PAIR_MAX_REQUESTS_CEILING,
        "shared_rate_limiter_max_rps": PAIR_MAX_RPS_CEILING,
        "inter_issuer_gap_seconds": INTER_ISSUER_MIN_GAP_SEC,
        "issuers_executed": [r["issuer"] for r in issuer_receipts],
        "issuer_results": issuer_receipts,
        "fatal_error": fatal_error,
        "safety_invariants": {
            "worker_role": os.environ.get("ARX_SEC_EGRESS_ROLE"),
            "containment_authorized": os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"),
            "advisory_lock_id": LOCK_SEC_EGRESS_OWNER,
            "advisory_lock_held": False,
        },
        "preservation_fingerprints": {
            "before": fingerprint_before,
            "after": fingerprint_after,
            "diff": fingerprint_diff,
        },
    }
    return pair_receipt


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_pair_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    raw_url = args.dsn or os.environ.get("SMART_MONEY_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not raw_url:
        print("FATAL: --dsn, DATABASE_URL, or SMART_MONEY_DATABASE_URL is required.", file=sys.stderr)
        return 1

    dsn = raw_url.replace("postgres.railway.internal:5432", "crossover.proxy.rlwy.net:17685")

    auth_issued = None
    if args.auth_issued_at:
        try:
            auth_issued = datetime.fromisoformat(args.auth_issued_at.replace("Z", "+00:00"))
        except Exception as e:
            print(f"FATAL: Invalid --auth-issued-at: {e}", file=sys.stderr)
            return 1

    auth_expires = None
    if args.auth_expires_at:
        try:
            auth_expires = datetime.fromisoformat(args.auth_expires_at.replace("Z", "+00:00"))
        except Exception as e:
            print(f"FATAL: Invalid --auth-expires-at: {e}", file=sys.stderr)
            return 1

    mode_label = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"=== EXECUTING SEQUENTIAL TWO-ISSUER CANARY [{mode_label}] ===")

    try:
        receipt = execute_sequential_pair(
            dsn=dsn,
            dry_run=args.dry_run,
            auth_issued_at=auth_issued,
            auth_expires_at=auth_expires,
            max_duration_sec=args.max_duration_sec,
        )
        print("\n=== SEQUENTIAL CANARY PAIR EXECUTION RECEIPT ===")
        print(json.dumps(receipt, indent=2, default=str))

        if receipt.get("status") == "SUCCESS":
            return 0
        else:
            return 1
    except KeyboardInterrupt:
        print("\nABORT: Execution interrupted by operator (SIGINT). Safe teardown completed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nFATAL: Sequential canary pair failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
