#!/usr/bin/env python3
"""
ARX Terminal — Dedicated NVDA-Only Bounded SEC Ingestion One-Shot Executor.

Enforces surgical one-shot execution following deployed PR #305 (SM-ACT-014R):
- Target Scope: Strictly NVDA (CIK 0001045810) hardcoded in code path
- Transmission Limits: <= 5 requests/sec, <= 40 requests/cycle, <= 4 requests/issuer, <= 3 unseen XML documents/issuer
- Duration Timeout: <= 30.0s wall-clock ceiling (enforces socket-level deadlines)
- Pinned UTC Boundaries: interval_start, interval_end, public-availability cutoff, and auth_expires_at
- Interval Specification: Strict distinction between prospective post-baseline (2026-11-17 onward)
  and historical baseline reconciliation (2026-08-18 to 2026-11-16). Zero silent substitution.
- Process-Scoped Egress Role: ARX_SEC_EGRESS_ROLE=OWNER in-process only; reverted to STANDBY in finally:
- Nonblocking Advisory Lock: pg_try_advisory_lock(401007311509093371) on protected execution session
  (decomposes in pg_locks to classid=93366790, objid=1926593531, objsubid=1)
- Queue Isolation: sm_run_requests verified 0 active rows ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE')
- Fail-Closed Aborts: budget exhaustion, HTTP 429, missing availability, parse failure, issuer mismatch, lock loss, ambiguous amendment
- Interruption Cleanup: Explicit transaction rollback and invariant teardown on SIGINT / KeyboardInterrupt
- Atomic Persistence: database writes commit atomically or roll back completely
- Snapshot Protection: Never modifies scanner_active_snapshots; active 10-issuer snapshot preserved
- Partial Coverage Certification: Certifies coverage_status = "PARTIAL" in sm_reconciliation_receipts
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.smart_money.canary_runner import (
    NvdaCanaryRunner,
    CANARY_TICKER,
    CANARY_CIK,
    CANARY_MAX_REQUESTS,
    CANARY_MAX_DURATION_SEC,
    CANARY_MAX_RPS,
    CANARY_MAX_UNSEEN_DOCS,
    CanarySecurityError,
    build_canary_parser,
    run_canary_cli,
    execute_oneshot as canonical_execute_oneshot,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    LOCK_SEC_EGRESS_OWNER,
    LOCK_SEC_EGRESS_OWNER_CLASSID,
    LOCK_SEC_EGRESS_OWNER_OBJID,
)

# Authoritative Constants for Lock Verification
assert LOCK_SEC_EGRESS_OWNER == 401007311509093371
assert LOCK_SEC_EGRESS_OWNER_CLASSID == 93366790
assert LOCK_SEC_EGRESS_OWNER_OBJID == 1926593531


def parse_args() -> argparse.Namespace:
    return build_canary_parser(prog="execute_nvda_oneshot_cycle.py").parse_args()


def execute_oneshot(args: argparse.Namespace) -> Dict[str, Any]:
    """Programmatic execution sharing the canonical validation and execution path."""
    return canonical_execute_oneshot(args)


def main():
    sys.exit(run_canary_cli(sys.argv[1:], prog="execute_nvda_oneshot_cycle.py"))


if __name__ == "__main__":
    main()
