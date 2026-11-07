#!/usr/bin/env python3
"""ARX Shadow Prospective Outcome Settlement Replay CLI (Directive ARX-LEARNING-003 / ARX-LEARNING-004).

Executes bounded, idempotent non-production replay settlement of prospective
decision records and historical recommendations against authoritative market tape.

Enforces:
- HOLD_SCAN_001 = ACTIVE: Automatic periodic settlement daemon remains disabled.
- Zero live execution or model weight mutation.
- Unresolved observations are never counted as wins or losses.
- Authoritative market data requirement: Empty market data maps cannot settle real outcomes.
"""

import sys
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
import pandas as pd

from analyst_dashboard.governance.prospective_capture import (
    ProspectiveDecisionCaptureEngine,
)
from analyst_dashboard.governance.shadow_replay_runner import (
    ShadowReplayRunner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arx.replay_settlement")


def load_authoritative_market_data(file_path: str) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Loads authoritative price bars from JSON or CSV.
    Computes and returns SHA256 along with symbol-keyed DataFrame map.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Authoritative market data file not found: {file_path}")

    with open(file_path, "rb") as f:
        file_bytes = f.read()
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()

    market_map = {}
    if file_path.endswith(".json"):
        raw = json.loads(file_bytes.decode("utf-8"))
        if isinstance(raw, dict):
            for sym, bars in raw.items():
                if isinstance(bars, list) and bars:
                    df = pd.DataFrame(bars)
                    date_col = next((c for c in df.columns if c.lower() in ("date", "timestamp")), None)
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df.set_index(date_col, inplace=True)
                    rename_map = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
                    df.rename(columns=rename_map, inplace=True)
                    market_map[sym.upper()] = df
        elif isinstance(raw, list):
            df = pd.DataFrame(raw)
            sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
            date_col = next((c for c in df.columns if c.lower() in ("date", "timestamp")), None)
            if sym_col and date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                rename_map = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
                df.rename(columns=rename_map, inplace=True)
                for sym, group in df.groupby(sym_col):
                    gdf = group.drop(columns=[sym_col]).set_index(date_col)
                    market_map[str(sym).upper()] = gdf
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
        date_col = next((c for c in df.columns if c.lower() in ("date", "timestamp")), None)
        if sym_col and date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            rename_map = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
            df.rename(columns=rename_map, inplace=True)
            for sym, group in df.groupby(sym_col):
                gdf = group.drop(columns=[sym_col]).set_index(date_col)
                market_map[str(sym).upper()] = gdf
        elif date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            rename_map = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
            df.rename(columns=rename_map, inplace=True)
            df.set_index(date_col, inplace=True)
            market_map["_DEFAULT_"] = df

    return market_map, file_sha256


def main():
    parser = argparse.ArgumentParser(description="ARX Shadow Replay Settlement CLI")
    parser.add_argument("--db-path", default=None, help="Custom governance SQLite database path")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of records to replay")
    parser.add_argument("--symbol", default=None, help="Filter by ticker symbol")
    parser.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help="Explicitly permit committing outcome links to database. Default is False (dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Explicitly run in non-persisting dry-run mode (default).",
    )
    parser.add_argument("--market-data-file", default=None, help="Path to authoritative market data file (JSON or CSV)")
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

    market_map = {}
    market_data_sha256 = None
    if args.market_data_file:
        try:
            market_map, market_data_sha256 = load_authoritative_market_data(args.market_data_file)
            logger.info(
                f"Loaded authoritative market data for {len(market_map)} symbols (SHA-256: {market_data_sha256})."
            )
        except Exception as e:
            logger.error(f"Failed to load authoritative market data from {args.market_data_file}: {e}")
            print(json.dumps({
                "status": "ERROR",
                "error": f"Failed to load market data file: {str(e)}",
                "settlement_eligibility": "INELIGIBLE_FOR_REAL_SETTLEMENT",
                "settlement_qualification": "UNQUALIFIED",
            }, indent=2))
            sys.exit(1)

    has_authoritative_data = bool(
        args.market_data_file
        and market_map
        and any(isinstance(df, pd.DataFrame) and not df.empty for df in market_map.values())
    )

    # Directive ARX-LEARNING-004:
    # 1. Default CLI to non-persisting dry-run.
    # 2. Require explicit authoritative timestamped market-data inputs AND
    #    separately qualified persistence flag before committing outcomes.
    if args.persist and not args.dry_run:
        if not has_authoritative_data:
            logger.error(
                "PERSISTENCE_REJECTED: Committing outcomes strictly requires explicit authoritative "
                "timestamped market data (--market-data-file). Reverting to dry-run."
            )
            effective_persist = False
        else:
            effective_persist = True
    else:
        effective_persist = False

    effective_dry_run = not effective_persist

    if not decisions:
        logger.info("No prospective decision records found in prospective_decision_events.")
        print(json.dumps({
            "status": "SUCCESS",
            "message": "Zero records pending settlement in prospective_decision_events.",
            "total_evaluated": 0,
            "dry_run": effective_dry_run,
            "persistence_committed": effective_persist,
            "market_data_authority": "NOT_APPLICABLE (No decisions to evaluate)",
            "settlement_eligibility": "INELIGIBLE_FOR_REAL_SETTLEMENT",
            "settlement_qualification": "UNQUALIFIED",
        }, indent=2))
        return

    # If single-symbol market data was loaded under '_DEFAULT_' and --symbol was specified, map it
    if "_DEFAULT_" in market_map and args.symbol:
        market_map[args.symbol.upper()] = market_map["_DEFAULT_"]

    batch_result = runner.replay_captured_decisions_batch(
        decisions=decisions,
        market_data_map=market_map,
        persist=effective_persist,
    )

    is_qualified = batch_result.get("settlement_qualification") == "OUTCOME_SETTLEMENT_QUALIFIED"

    output = {
        "status": "SUCCESS" if is_qualified else "UNRESOLVED_ONLY",
        "market_data_authority": batch_result["market_data_authority"],
        "market_data_file": args.market_data_file,
        "market_data_sha256": market_data_sha256,
        "settlement_eligibility": batch_result["settlement_eligibility"],
        "settlement_qualification": batch_result["settlement_qualification"],
        "total_evaluated": batch_result["total_evaluated"],
        "status_counts": batch_result["status_counts"],
        "resolved_count": batch_result["resolved_count"],
        "unresolved_count": batch_result["unresolved_count"],
        "dry_run": effective_dry_run,
        "persistence_committed": effective_persist,
        "governance_notice": batch_result["governance_notice"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
