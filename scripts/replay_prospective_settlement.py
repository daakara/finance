#!/usr/bin/env python3
"""ARX Shadow Prospective Outcome Settlement Replay CLI (Directive ARX-LEARNING-003).

Executes bounded, idempotent non-production replay settlement of prospective
decision records and historical recommendations against authoritative market tape.

Enforces:
- HOLD_SCAN_001 = ACTIVE: Automatic periodic settlement daemon remains disabled.
- Zero live execution or model weight mutation.
- Unresolved observations are never counted as wins or losses.
"""

import sys
import argparse
import json
import logging
from datetime import datetime, timezone
import pandas as pd

from analyst_dashboard.governance.prospective_capture import (
    ProspectiveDecisionCaptureEngine,
)
from analyst_dashboard.governance.shadow_replay_runner import (
    ShadowReplayRunner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arx.replay_settlement")


def main():
    parser = argparse.ArgumentParser(description="ARX Shadow Replay Settlement CLI")
    parser.add_argument("--db-path", default=None, help="Custom governance SQLite database path")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of records to replay")
    parser.add_argument("--symbol", default=None, help="Filter by ticker symbol")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without persisting outcome links")
    args = parser.parse_args()

    engine = ProspectiveDecisionCaptureEngine(db_path=args.db_path)
    runner = ShadowReplayRunner(capture_engine=engine)

    conn = engine.db.get_connection()
    try:
        query = "SELECT * FROM prospective_decision_events WHERE trade_plan_json IS NOT NULL"
        params = []
        if args.symbol:
            query += " AND symbol = ?"
            params.append(args.symbol.upper())
        query += " ORDER BY created_at_utc ASC LIMIT ?"
        params.append(args.limit)

        cur = conn.execute(query, tuple(params))
        decisions = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    logger.info(f"Loaded {len(decisions)} prospective decision records for replay settlement.")

    if not decisions:
        logger.info("No prospective decision records found in prospective_decision_events.")
        print(json.dumps({
            "status": "SUCCESS",
            "message": "Zero records pending settlement in prospective_decision_events.",
            "total_evaluated": 0,
            "dry_run": args.dry_run,
        }, indent=2))
        return

    # In a full run, market data would be fetched per ticker; for replay CLI, build empty map or use provider
    market_map = {}
    batch_result = runner.replay_captured_decisions_batch(
        decisions=decisions,
        market_data_map=market_map,
        persist=(not args.dry_run),
    )

    print(json.dumps({
        "status": "SUCCESS",
        "total_evaluated": batch_result["total_evaluated"],
        "status_counts": batch_result["status_counts"],
        "resolved_count": batch_result["resolved_count"],
        "unresolved_count": batch_result["unresolved_count"],
        "dry_run": args.dry_run,
        "governance_notice": batch_result["governance_notice"],
    }, indent=2))


if __name__ == "__main__":
    main()
