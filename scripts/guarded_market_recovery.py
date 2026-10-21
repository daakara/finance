#!/usr/bin/env python3
"""
scripts/guarded_market_recovery.py

Guarded Disaster Recovery CLI Tool (ARX Closure Model v2).
Executes the non-destructive, evidence-preserving 9-step SQLite rollback runbook.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.data.guarded_recovery import GuardedRecoveryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arx.guarded_recovery_cli")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guarded Disaster Recovery Tool for ARX Market / Identity SQLite Stores"
    )
    parser.add_argument("--target-db", required=True, help="Path to target SQLite database")
    parser.add_argument("--backup-file", required=True, help="Path to verified SQLite backup file")
    parser.add_argument("--expected-sha256", required=True, help="Expected SHA-256 of backup file")
    parser.add_argument("--incident-id", required=True, help="Incident reference identifier (e.g. INC-G-S2-...)")
    parser.add_argument("--operator-auth", required=True, help="Formal operator authorization token (e.g. G-S2-ACT-...)")
    parser.add_argument("--maintenance-token", required=True, help="Incident-specific maintenance mode authorization token (e.g. MAINT-INC-...)")
    parser.add_argument("--quarantine-dir", default="/root/incident_quarantine", help="Base quarantine directory")
    parser.add_argument("--receipt-output", default=None, help="Path to save recovery receipt JSON")

    args = parser.parse_args()

    engine = GuardedRecoveryEngine(
        target_db_path=args.target_db,
        backup_db_path=args.backup_file,
        expected_backup_sha256=args.expected_sha256,
        incident_id=args.incident_id,
        operator_authorization=args.operator_auth,
        maintenance_mode_token=args.maintenance_token,
        quarantine_base_dir=args.quarantine_dir,
    )

    try:
        logger.info("Initiating 9-step guarded disaster recovery sequence...")
        receipt = engine.execute_guarded_recovery()
        receipt_dict = receipt.to_dict()
        receipt_json = json.dumps(receipt_dict, indent=2)

        if args.receipt_output:
            out_p = Path(args.receipt_output).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(receipt_json, encoding="utf-8")
            logger.info(f"Recovery receipt written to '{out_p}'.")

        print(receipt_json)
        logger.info("Guarded disaster recovery completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Guarded disaster recovery FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
