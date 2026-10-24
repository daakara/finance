"""
scripts/research/acquire_listing_directories.py

Explicit diagnostic command to acquire public listing directory feeds,
preserve raw response bytes and metadata sidecars, and replay through
offline parsers for source-population accounting.

Usage:
    python scripts/research/acquire_listing_directories.py --output-dir data/raw_feeds/2026-10-16
    python scripts/research/acquire_listing_directories.py --source sec_company_tickers_exchange
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.universe.acquisition import (
    CANONICAL_SOURCES,
    ListingAcquisitionRunner,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARX Terminal — Listing Directory Acquisition & Accounting Tool"
    )
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    default_output = Path("data/raw_feeds") / today_str

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory to store raw feed bytes and metadata (default: {default_output})",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=list(CANONICAL_SOURCES.keys()) + ["all"],
        default="all",
        help="Specific source to acquire, or 'all' (default: all)",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Optional path to write full machine-readable JSON report",
    )
    parser.add_argument(
        "--offline-dir",
        type=Path,
        default=None,
        help="Optional directory containing offline test fixtures to bypass network fetch",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save machine-readable JSON and Markdown accounting reports to output directory",
    )
    parser.add_argument(
        "--match",
        action="store_true",
        default=True,
        help="Run non-binding diagnostic cross-source matching (default: True)",
    )
    parser.add_argument(
        "--no-match",
        action="store_false",
        dest="match",
        help="Disable diagnostic cross-source matching",
    )

    args = parser.parse_args()

    # Determine target sources
    if args.source == "all":
        sources = list(CANONICAL_SOURCES.values())
    else:
        sources = [CANONICAL_SOURCES[args.source]]

    offline_bytes: dict[str, bytes] | None = None
    if args.offline_dir and args.offline_dir.exists():
        offline_bytes = {}
        for s in sources:
            for ext in [".json", ".txt"]:
                fixture_file = args.offline_dir / f"{s.source_id}{ext}"
                if fixture_file.exists():
                    offline_bytes[s.source_id] = fixture_file.read_bytes()
                    break

    print(f"Executing listing acquisition for {len(sources)} source(s)...")
    print(f"Target local output directory: {args.output_dir.resolve()}")
    if offline_bytes:
        print(f"Replaying from offline fixtures: {list(offline_bytes.keys())}")

    runner = ListingAcquisitionRunner(output_dir=args.output_dir)
    report = runner.run(
        sources=sources,
        run_matching=args.match,
        offline_bytes=offline_bytes,
    )

    # Print human-readable report
    print("\n" + report.format_human_readable() + "\n")

    # Optional JSON output
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Machine-readable report written to: {args.json_report.resolve()}")

    if args.save_report:
        from analyst_dashboard.universe.acquisition import save_accounting_reports
        json_path, md_path = save_accounting_reports(report, args.output_dir)
        print(f"Machine-readable JSON report: {json_path}")
        print(f"Human-readable Markdown report: {md_path}")

    return 0 if report.overall_outcome in ("SUCCESS", "PARTIAL_ACQUISITION", "COMPLETE_SUCCESS", "PARTIAL_SUCCESS") else 1


if __name__ == "__main__":
    sys.exit(main())
