#!/usr/bin/env python3
"""
scripts/diagnostics/real_source_admission_diagnostic.py

Sprint 2: Bounded Real-Source Evidence Handoff & Admission Diagnostic CLI.
Executes the reproducible admission diagnostic for the 7-candidate sample:
AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU.

Usage:
  python scripts/diagnostics/real_source_admission_diagnostic.py \
    --output-dir output/diagnostics/real_source_diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analyst_dashboard.universe.real_source_diagnostic import (
    RealSourceAdmissionDiagnostic,
    format_evidence_gap_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARX Terminal Sprint 2: Bounded Real-Source Admission Diagnostic CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write diagnostic JSON and Markdown reports",
    )
    parser.add_argument(
        "--evaluation-cutoff",
        type=str,
        default="2026-10-16",
        help="Target session cutoff date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--market-db-path",
        type=str,
        default=None,
        help="Explicit path to market SQLite database for read-only inspection",
    )

    args = parser.parse_args()

    db_path = Path(args.market_db_path) if args.market_db_path else None
    result = RealSourceAdmissionDiagnostic.run(
        market_db_path=db_path,
        evaluation_cutoff=args.evaluation_cutoff,
    )

    report_md = format_evidence_gap_report(result)
    print(report_md)

    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / "real_source_diagnostic_report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nWrote JSON report to: {json_file}")

        md_file = out_path / "REAL_SOURCE_EVIDENCE_GAP_REPORT.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Wrote Markdown report to: {md_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
