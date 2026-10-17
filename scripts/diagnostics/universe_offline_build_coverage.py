#!/usr/bin/env python3
"""
scripts/diagnostics/universe_offline_build_coverage.py

ARX Terminal — Sprint 2: Bounded Offline Universe Build & Coverage Integration CLI.

Demonstrates the complete end-to-end offline path connecting:
attributable candidate records
→ identity/listing association checks
→ input validation and atomic conversion
→ Universe Builder eligibility evaluation (Rules R01–R07)
→ market-data readiness diagnostics
→ structured outcome report

Usage:
  # Run all three scenarios (A: Convertible, B: Atomic Fail-Closed, C: Real-Source Replay):
  python scripts/diagnostics/universe_offline_build_coverage.py --scenario all --output-dir output/diagnostics/offline_build_coverage

  # Run Scenario A only:
  python scripts/diagnostics/universe_offline_build_coverage.py --scenario A --output-dir output/diagnostics/scenario_a

  # Run Scenario B only:
  python scripts/diagnostics/universe_offline_build_coverage.py --scenario B --output-dir output/diagnostics/scenario_b

  # Run Scenario C only:
  python scripts/diagnostics/universe_offline_build_coverage.py --scenario C --output-dir output/diagnostics/scenario_c
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.universe.offline_build_coverage import (
    OfflineBuildCoverageConfig,
    OfflineBuildCoverageResult,
    build_scenario_a_candidates,
    build_scenario_b_candidates,
    build_scenario_b2_conversion_failure_candidates,
    build_scenario_d_identity_conflict_candidates,
    build_scenario_c_candidates,
    format_offline_build_coverage_markdown,
    run_offline_build_and_coverage,
    save_offline_build_coverage_artifacts,
)
from analyst_dashboard.universe.coverage_diagnostics import ReadOnlyMarketStore


def create_demo_market_db(db_path: str, as_of_date: str = "2026-10-16") -> str:
    """Creates a local SQLite market store seeded with representative test histories."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS asset_ohlcv_daily (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        )
    """)

    base_date = date(2025, 1, 1)

    # 1. AAPL: 252 bars, fresh session date (2026-10-16) -> DATA_READY
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else as_of_date
        cur.execute(
            "INSERT OR REPLACE INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("AAPL", d, 150.0, 155.0, 149.0, 152.0, 1000000),
        )

    # 2. MSFT: 200 bars, stale session date (2026-10-12) -> STALE_QUOTE
    for i in range(200):
        d = (base_date + timedelta(days=i)).isoformat() if i < 199 else "2026-10-12"
        cur.execute(
            "INSERT OR REPLACE INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("MSFT", d, 300.0, 305.0, 298.0, 302.0, 500000),
        )

    # 3. GLX: 47 bars (< 50 bars) -> INSUFFICIENT_HISTORY
    for i in range(47):
        d = (base_date + timedelta(days=i)).isoformat()
        cur.execute(
            "INSERT OR REPLACE INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("GLX", d, 10.0, 11.0, 9.5, 10.5, 50000),
        )

    # 4. SPY: 252 bars, fresh session date (2026-10-16) -> DATA_READY in market data (ineligible in builder R01)
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else as_of_date
        cur.execute(
            "INSERT OR REPLACE INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("SPY", d, 500.0, 505.0, 498.0, 502.0, 5000000),
        )

    # 5. SEC1: 252 bars, fresh session date (2026-10-16) -> DATA_READY in market data (ineligible in builder R06)
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else as_of_date
        cur.execute(
            "INSERT OR REPLACE INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("SEC1", d, 75.0, 77.0, 74.0, 76.0, 300000),
        )

    # 6. DISP1: 252 bars, fresh session date (2026-10-16)
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else as_of_date
        cur.execute(
            "INSERT OR REPLACE INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("DISP1", d, 25.0, 26.0, 24.5, 25.5, 100000),
        )

    # 7. UNSUP1: 252 bars, fresh session date (2026-10-16) -> DATA_READY in market data (ineligible in builder R04)
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else as_of_date
        cur.execute(
            "INSERT OR REPLACE INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("UNSUP1", d, 45.0, 46.0, 44.0, 45.5, 200000),
        )

    # Note: NOMKT is intentionally omitted to demonstrate MISSING_MARKET_DATA

    conn.commit()
    conn.close()
    return db_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARX Terminal Sprint 2: Bounded Offline Universe Build & Coverage Integration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["A", "B", "B2", "D", "C", "all"],
        default="all",
        help="Offline scenario to execute: A (Convertible Batch), B (Atomic Fail-Closed), B2 (Conversion Boundary Failure), D (Identity Conflict), C (Real-Source Replay), or all",
    )
    parser.add_argument(
        "--market-db",
        type=str,
        default=None,
        help="Path to read-only SQLite market store (defaults to temporary demo store if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/diagnostics/offline_build_coverage",
        help="Directory to write structured JSON and Markdown outcome reports",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default="2026-10-16",
        help="Target session as-of date (YYYY-MM-DD)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve or create market store
    temp_db_dir = None
    db_path = args.market_db
    if not db_path:
        default_candidate = REPO_ROOT / "market_data.db"
        if default_candidate.exists():
            db_path = str(default_candidate)
        else:
            temp_db_dir = tempfile.mkdtemp(prefix="demo_market_db_")
            db_path = os.path.join(temp_db_dir, "demo_market.db")
            create_demo_market_db(db_path, as_of_date=args.as_of)

    results: List[OfflineBuildCoverageResult] = []

    try:
        store = ReadOnlyMarketStore(db_path)

        # ── SCENARIO A: Fully Convertible Batch ──────────────────────────────
        if args.scenario in ("A", "all"):
            print("==================================================================")
            print("EXECUTING SCENARIO A: Fully Convertible Batch Reaching Builder & Readiness")
            print("==================================================================")
            cands_a = build_scenario_a_candidates()
            config_a = OfflineBuildCoverageConfig(
                candidate_records=cands_a,
                as_of_date=args.as_of,
                market_store=store,
                scenario_label="SCENARIO_A_CONVERTIBLE_BATCH",
                is_synthetic=True,
                output_dir=args.output_dir,
            )
            res_a = run_offline_build_and_coverage(config_a)
            results.append(res_a)
            print(f"Scenario A Outcome: {res_a.overall_outcome}")
            print(f"  - Candidates: {res_a.stage_1_identity_association.total_candidates}")
            print(f"  - Admitted: {res_a.stage_2_validation_and_conversion.converted_security_count}")
            print(f"  - Builder Evaluated: {res_a.stage_3_builder_evaluation.evaluated_security_count} "
                  f"(Eligible: {res_a.stage_3_builder_evaluation.eligible_count}, Ineligible: {res_a.stage_3_builder_evaluation.ineligible_count})")
            print(f"  - Diagnostics: Data Ready: {res_a.stage_4_coverage_diagnostics.data_ready_count}, "
                  f"Stale: {res_a.stage_4_coverage_diagnostics.stale_quote_count}, "
                  f"Insufficient: {res_a.stage_4_coverage_diagnostics.insufficient_history_count}, "
                  f"Missing: {res_a.stage_4_coverage_diagnostics.missing_market_data_count}")
            print()

        # ── SCENARIO B: Atomic Fail-Closed Batch ──────────────────────────────
        if args.scenario in ("B", "all"):
            print("==================================================================")
            print("EXECUTING SCENARIO B: Atomic Fail-Closed Batch (Zero Partial Handoff)")
            print("==================================================================")
            cands_b = build_scenario_b_candidates()
            config_b = OfflineBuildCoverageConfig(
                candidate_records=cands_b,
                as_of_date=args.as_of,
                market_store=store,
                scenario_label="SCENARIO_B_ATOMIC_FAIL_CLOSED",
                is_synthetic=True,
                output_dir=args.output_dir,
            )
            res_b = run_offline_build_and_coverage(config_b)
            results.append(res_b)
            print(f"Scenario B Outcome: {res_b.overall_outcome}")
            print(f"  - Candidates: {res_b.stage_1_identity_association.total_candidates}")
            print(f"  - Admission: {res_b.stage_2_validation_and_conversion.status} "
                  f"(Converted: {res_b.stage_2_validation_and_conversion.converted_security_count})")
            print(f"  - Builder Status: {res_b.stage_3_builder_evaluation.status} "
                  f"(Evaluated: {res_b.stage_3_builder_evaluation.evaluated_security_count})")
            print(f"  - Zero Partial Handoff: {res_b.reconciliation_closure.get('zero_partial_handoff_verified')}")
            print()

        # ── SCENARIO B2: Missing Conversion-Boundary Failure (Unknown Primary Listing)
        if args.scenario in ("B2", "all"):
            print("==================================================================")
            print("EXECUTING SCENARIO B2: Conversion Boundary Failure on Unknown Primary Listing")
            print("==================================================================")
            cands_b2 = build_scenario_b2_conversion_failure_candidates()
            config_b2 = OfflineBuildCoverageConfig(
                candidate_records=cands_b2,
                as_of_date=args.as_of,
                market_store=store,
                scenario_label="SCENARIO_B2_CONVERSION_FAILURE",
                is_synthetic=True,
                output_dir=args.output_dir,
            )
            res_b2 = run_offline_build_and_coverage(config_b2)
            results.append(res_b2)
            print(f"Scenario B2 Outcome: {res_b2.overall_outcome}")
            print(f"  - Candidates: {res_b2.stage_1_identity_association.total_candidates}")
            print(f"  - Validator Component: {res_b2.stage_2_validation_and_conversion.component_validation_status} "
                  f"(is_batch_valid: {res_b2.stage_2_validation_and_conversion.is_batch_valid})")
            print(f"  - Conversion Boundary: {res_b2.stage_2_validation_and_conversion.conversion_status} "
                  f"(Converted: {res_b2.stage_2_validation_and_conversion.converted_security_count})")
            print(f"  - Builder Status: {res_b2.stage_3_builder_evaluation.status} "
                  f"(Evaluated: {res_b2.stage_3_builder_evaluation.evaluated_security_count})")
            print(f"  - Zero Partial Handoff: {res_b2.reconciliation_closure.get('zero_partial_handoff_verified')}")
            print()

        # ── SCENARIO D: Unresolved Identity Conflict Batch ───────────────────
        if args.scenario in ("D", "all"):
            print("==================================================================")
            print("EXECUTING SCENARIO D: Unresolved Identity Conflict Batch")
            print("==================================================================")
            cands_d = build_scenario_d_identity_conflict_candidates()
            config_d = OfflineBuildCoverageConfig(
                candidate_records=cands_d,
                as_of_date=args.as_of,
                market_store=store,
                scenario_label="SCENARIO_D_IDENTITY_CONFLICT",
                is_synthetic=True,
                output_dir=args.output_dir,
            )
            res_d = run_offline_build_and_coverage(config_d)
            results.append(res_d)
            print(f"Scenario D Outcome: {res_d.overall_outcome}")
            print(f"  - Candidates: {res_d.stage_1_identity_association.total_candidates}")
            print(f"  - Stage 1 Identity: Resolved: {res_d.stage_1_identity_association.resolved_identities_count}, "
                  f"Blocked: {res_d.stage_1_identity_association.blocked_identities_count}")
            print(f"  - Admission: {res_d.stage_2_validation_and_conversion.status} "
                  f"(Converted: {res_d.stage_2_validation_and_conversion.converted_security_count})")
            print(f"  - Builder Status: {res_d.stage_3_builder_evaluation.status} "
                  f"(Evaluated: {res_d.stage_3_builder_evaluation.evaluated_security_count})")
            print(f"  - Zero Partial Handoff: {res_d.reconciliation_closure.get('zero_partial_handoff_verified')}")
            print()

        # ── SCENARIO C: Real-Source Retained Evidence Replay ──────────────────
        if args.scenario in ("C", "all"):
            print("==================================================================")
            print("EXECUTING SCENARIO C: Real-Source Retained Evidence Replay")
            print("==================================================================")
            cands_c = build_scenario_c_candidates()
            config_c = OfflineBuildCoverageConfig(
                candidate_records=cands_c,
                as_of_date=args.as_of,
                market_store=store,
                scenario_label="SCENARIO_C_REAL_SOURCE_REPLAY",
                is_synthetic=False,
                output_dir=args.output_dir,
            )
            res_c = run_offline_build_and_coverage(config_c)
            results.append(res_c)
            print(f"Scenario C Outcome: {res_c.overall_outcome}")
            print(f"  - Real Candidates: {res_c.stage_1_identity_association.total_candidates}")
            print(f"  - Admission: {res_c.stage_2_validation_and_conversion.status} "
                  f"(Fail-closed without fabricated currency/identity)")
            print(f"  - Builder Status: {res_c.stage_3_builder_evaluation.status}")
            print(f"  - Venue Disagreements Preserved: {res_c.stage_1_identity_association.venue_conflict_count}")
            print()

        store.close()

    finally:
        if temp_db_dir and os.path.exists(temp_db_dir):
            import shutil
            shutil.rmtree(temp_db_dir, ignore_errors=True)

    print("==================================================================")
    print(f"ALL SCENARIOS COMPLETE. Artifacts written to: {args.output_dir}")
    print("STOP POINT: OFFLINE_BUILD_COVERAGE_INTEGRATION_LOCAL_VERIFIED")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
