"""
analyst_dashboard/research/maintenance/cli.py

CLI runner for the ARX Research-Maintenance Pilot.
Provides manually executable event-check commands, returning distinct documented outcomes:
- Exit Code 0: SUCCESS_PACKET_GENERATED or ALREADY_COMPLETED
- Exit Code 1: PROCESSING_FAILED
- Exit Code 2: REJECTED_NON_PERIODIC (e.g. Form 8-K)
- Exit Code 3: NO_NEW_FILING

Governed by ARX Closure Model v2 and ARX-DECISION-RESEARCH-PROCESS-V1.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .contracts import EventProcessingState
from .service import ResearchMaintenanceService

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("arx_maintenance_cli")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARX Terminal — Research Maintenance Event Checker & Packet Generator."
    )
    parser.add_argument("--symbol", type=str, required=True, help="Target published asset symbol (e.g. FIX, DUOL, ULTA, MEDP, ISRG, DECK)")
    parser.add_argument("--accession", type=str, default=None, help="Target specific SEC accession number")
    parser.add_argument("--cutoff-date", type=str, default="2026-10-16", help="Evidence cutoff date (YYYY-MM-DD)")
    parser.add_argument("--replay", action="store_true", help="Explicitly label run as replay (preserves freshness boundaries)")
    parser.add_argument("--force-refresh", action="store_true", help="Force fresh retrieval bypassing local cache")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode using local cache")
    parser.add_argument("--register-path", type=str, default=None, help="Custom path for event_register.json")
    parser.add_argument("--packets-dir", type=str, default=None, help="Custom directory for generated review packets")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom directory for raw SEC caches")

    args = parser.parse_args()

    service = ResearchMaintenanceService(
        register_path=args.register_path,
        packets_dir=args.packets_dir,
        cache_dir=args.cache_dir,
        offline_mode=args.offline,
    )

    symbol = args.symbol.strip().upper()
    logger.info(f"Checking maintenance events for {symbol} (cutoff: {args.cutoff_date}, replay: {args.replay})...")

    packet, event, warnings = service.run_maintenance_pipeline(
        symbol=symbol,
        target_accession=args.accession,
        cutoff_date=args.cutoff_date,
        is_replay=args.replay,
        force_refresh=args.force_refresh,
    )

    if warnings:
        for w in warnings:
            logger.warning(f"  {w}")

    if packet:
        print("\n" + "=" * 80)
        print(f"OUTCOME: SUCCESS_PACKET_GENERATED for {symbol}")
        print(f"Packet ID:           {packet.packet_id}")
        print(f"Accepted Baseline:   {packet.accepted_baseline_id}")
        print(f"Draft Successor:     {packet.draft_successor_id}")
        print(f"Filing Form:         {packet.filing_metadata.get('form')} ({packet.filing_metadata.get('filing_date')})")
        print(f"Reporting Period:    {packet.filing_metadata.get('reporting_period')}")
        print(f"Proposed Action:     {packet.proposed_disposition.value}")
        print(f"Changes Reconciled:  {len(packet.material_financial_changes)}")
        print("=" * 80 + "\n")
        return 0

    if event:
        if event.processing_state == EventProcessingState.REJECTED_MISIDENTIFIED:
            print("\n" + "=" * 80)
            print(f"OUTCOME: REJECTED_NON_PERIODIC for {symbol} (accession {event.accession_number})")
            print(f"Detail: {event.error_message}")
            print("=" * 80 + "\n")
            return 2
        elif event.processing_state in (EventProcessingState.REVIEW_PENDING, EventProcessingState.REVIEW_ACCEPTED):
            print("\n" + "=" * 80)
            print(f"OUTCOME: ALREADY_COMPLETED for {symbol} (accession {event.accession_number})")
            print(f"State: {event.processing_state.value}")
            print("=" * 80 + "\n")
            return 0
        elif event.processing_state == EventProcessingState.FAILED:
            print("\n" + "=" * 80)
            print(f"OUTCOME: PROCESSING_FAILED for {symbol} (accession {event.accession_number})")
            print(f"Failed Stage: {event.failed_stage}")
            print(f"Error: {event.error_message}")
            print("=" * 80 + "\n")
            return 1

    if any("NO_NEW_FILING" in w for w in warnings):
        print(f"\nOUTCOME: NO_NEW_FILING for {symbol}.")
        return 3

    print(f"\nOUTCOME: PROCESSING_FAILED for {symbol} (no event or packet returned).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
