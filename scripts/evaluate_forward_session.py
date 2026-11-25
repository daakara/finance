#!/usr/bin/env python3
"""Execute one isolated point-in-time recommendation evaluation session.

Ingests point-in-time completed session bars for the frozen 10-symbol universe,
evaluates decisions via Confluence/VCP without invoking background schedulers or
modifying production databases, and commits to isolated forward pilot storage.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyst_dashboard.governance.forward_evaluator_adapter import (
    execute_isolated_forward_evaluation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        required=True,
        type=Path,
        help="JSON file defining point-in-time source paths or bars per symbol",
    )
    parser.add_argument(
        "--session-date",
        required=True,
        help="Completed XNYS session date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cohort-id",
        required=True,
        help="Forward pilot cohort namespace (COHORT_FORWARD_PILOT_*)",
    )
    parser.add_argument(
        "--start-session",
        required=True,
        help="Cohort start session date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--forward-root",
        required=True,
        type=Path,
        help="Path to isolated forward pilot root directory",
    )
    parser.add_argument(
        "--engine-sha",
        required=True,
        help="40-character lowercase engine commit SHA",
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Explicitly initialize a new cohort namespace",
    )

    args = parser.parse_args()

    try:
        manifest_data = json.loads(args.input_manifest.read_text())
        result = execute_isolated_forward_evaluation(
            manifest_data,
            cohort_id=args.cohort_id,
            start_session=args.start_session,
            forward_root=args.forward_root,
            session_date=args.session_date,
            engine_sha=args.engine_sha,
            initialize=args.initialize,
        )
    except Exception as e:
        print(json.dumps({"status": "BLOCKED", "reason": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
