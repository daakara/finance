"""
ARX Terminal — Dedicated Smart Money Automated Backup Runner.

Execution CLI and scheduler for continuous PostgreSQL logical backups:
- Enforces session-level advisory locking (LOCK_BACKUP_EXECUTION);
- Separate execution identity from SEC ingestion worker;
- Does not acquire or interfere with SEC_EGRESS_OWNER lock;
- Safe retention pruning after verified upload.
"""

import sys
import time
import signal
import logging
import argparse
from typing import Optional

from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager
from analyst_dashboard.smart_money.storage.backup.storage_adapter import get_backup_storage_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smart_money.backup_runner")


def run_backup_cycle(
    engine: SmartMoneyBackupEngine,
    retention: BackupRetentionManager,
) -> bool:
    """Executes a single backup and safe retention pruning pass."""
    try:
        logger.info("Initiating automated backup execution cycle...")
        manifest = engine.create_backup()
        logger.info(f"Backup {manifest['backup_id']} completed successfully.")

        pruned = retention.prune_backups()
        if pruned:
            logger.info(f"Pruned {len(pruned)} expired backup sets: {pruned}")
        return True
    except Exception as e:
        logger.error(f"Backup execution failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="ARX Smart Money Backup Runner")
    parser.add_argument("--single-run", action="store_true", help="Execute single backup pass and exit")
    parser.add_argument("--interval-sec", type=int, default=86400, help="Interval between scheduled executions (default: 86400 / 24h)")
    parser.add_argument("--retention-count", type=int, default=10, help="Maximum number of backups to retain")
    parser.add_argument("--retention-days", type=int, default=7, help="Maximum retention period in days")
    args = parser.parse_args()

    storage_adapter = get_backup_storage_adapter()
    engine = SmartMoneyBackupEngine(storage_adapter=storage_adapter)
    retention = BackupRetentionManager(
        storage_adapter=storage_adapter,
        retention_count=args.retention_count,
        retention_days=args.retention_days,
    )

    if args.single_run:
        success = run_backup_cycle(engine, retention)
        sys.exit(0 if success else 1)

    running = True

    def _sig_handler(signum, frame):
        nonlocal running
        logger.info(f"Received signal {signum}. Terminating backup scheduler gracefully...")
        running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    logger.info(f"Smart Money backup scheduler started (interval={args.interval_sec}s).")
    while running:
        run_backup_cycle(engine, retention)
        # Sleep in small slices to respond promptly to signals
        elapsed = 0
        while running and elapsed < args.interval_sec:
            time.sleep(min(1.0, args.interval_sec - elapsed))
            elapsed += 1

    logger.info("Backup scheduler terminated cleanly.")


if __name__ == "__main__":
    main()
