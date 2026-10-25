#!/usr/bin/env python3
"""
ARX Terminal — Smart Money Daily Continuous Ingestion CLI Entrypoint.

Executes the authoritative 10-step daily ingestion cycle:
Pre-Backup -> Verify -> Lock -> Ingest -> Reconcile -> Publish -> Post-Backup -> Release.

Supports:
  --dry-run      Run preflight and structural validations without mutating DB/storage.
  --force        Bypass America/New_York 22:30 schedule and dissemination calendar check.
  --as-of        Set point-in-time evaluation date / watermark (YYYY-MM-DD).
  --request-id   Optional sm_run_requests operator request ID to claim and complete.
  --verbose      Enable verbose DEBUG logging.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.smart_money.smart_money_scheduler import SmartMoneyTimezoneScheduler
from analyst_dashboard.smart_money.continuous_pipeline import (
    SmartMoneyContinuousPipeline,
    GOVERNED_UNIVERSE_ID,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ARX Terminal — Smart Money Daily Continuous Ingestion Runner"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute preflight validations without database mutations or snapshot publication.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass America/New_York 22:30 schedule window and calendar checks.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Point-in-time watermark date (YYYY-MM-DD). Defaults to current UTC date.",
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default=None,
        help="Operator run request ID from sm_run_requests queue.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("smart_money.daily_runner")

    print("================================================================================")
    print("  ARX TERMINAL: SMART MONEY CONTINUOUS INGESTION PIPELINE (PHASE 1)")
    print("================================================================================")
    print(f"Timestamp:       {datetime.now(timezone.utc).isoformat()}")
    print(f"Governed Cohort: {GOVERNED_UNIVERSE_ID} (10 US Equities)")
    print(f"Mode:            {'DRY-RUN' if args.dry_run else 'LIVE EXECUTION'}")
    print(f"Force Schedule:  {args.force}")
    if args.as_of:
        print(f"As-Of Watermark: {args.as_of}")
    if args.request_id:
        print(f"Request ID:      {args.request_id}")
    print("--------------------------------------------------------------------------------\n")

    # 1. Evaluate schedule unless force flag is provided
    scheduler = SmartMoneyTimezoneScheduler()
    last_executed_date = None
    try:
        from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
        from analyst_dashboard.smart_money.storage.config import get_database_url
        dsn = get_database_url()
        if dsn and not args.dry_run:
            repo = PostgresSmartMoneyRepository(dsn=dsn)
            snap = repo.get_latest_active_snapshot()
            if snap and snap.get("data_as_of"):
                as_of_val = snap["data_as_of"]
                if isinstance(as_of_val, str):
                    last_executed_date = datetime.strptime(as_of_val[:10], "%Y-%m-%d").date()
                elif hasattr(as_of_val, "date"):
                    last_executed_date = as_of_val.date()
    except Exception as e:
        logger.debug(f"Could not query last executed date from repository: {e}")

    should_run, reason = scheduler.should_execute_now(
        last_executed_ny_date=last_executed_date,
        force=args.force,
    )

    if not should_run and not args.dry_run:
        logger.warning(f"SCHEDULE_CHECK_DECLINED: {reason}")
        print(f"[SCHEDULE] Cycle declined: {reason}")
        print("[SCHEDULE] To execute outside scheduled window, supply --force or --dry-run.")
        sys.exit(0)

    print(f"[SCHEDULE] Schedule check verified: {reason}")

    # 2. Execute 10-step continuous pipeline
    pipeline = SmartMoneyContinuousPipeline(allow_test_mode=args.dry_run)
    try:
        result = pipeline.execute_cycle(
            run_request_id=args.request_id,
            force=args.force,
            dry_run=args.dry_run,
            as_of=args.as_of,
            trigger_type="OPERATOR_CLI" if args.force else "SCHEDULED_CLI",
        )

        print("\n================================================================================")
        print("  CYCLE EXECUTION SUMMARY")
        print("================================================================================")
        print(f"Cycle ID:           {result.get('cycle_id')}")
        print(f"Status:             {result.get('status')}")
        print(f"Snapshot ID:        {result.get('snapshot_id')}")
        print(f"Matched Count:      {result.get('matched_count')}")
        print(f"Universe Size:      {result.get('universe_size')}")
        print(f"Receipt ID:         {result.get('receipt_id')}")
        print(f"Pre-Backup ID:      {result.get('pre_backup_id')}")
        print(f"Post-Backup ID:     {result.get('post_backup_id')}")
        print(f"Steps Completed:    {len(result.get('steps_completed', []))}/10")
        for s in result.get("steps_completed", []):
            print(f"  - {s}")

        if result.get("errors"):
            print("Errors / Warnings:")
            for e in result.get("errors"):
                print(f"  ! {e}")

        print("================================================================================\n")

        if result.get("status") in ("SUCCESS", "DRY_RUN_SUCCESS"):
            sys.exit(0)
        elif result.get("status") == "DEGRADED_POST_BACKUP_FAILED":
            logger.warning("Pipeline completed in DEGRADED_POST_BACKUP_FAILED state.")
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.critical(f"FATAL: Continuous ingestion pipeline failed: {e}", exc_info=True)
        print(f"\n[FATAL] Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
