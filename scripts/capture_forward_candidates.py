#!/usr/bin/env python3
"""Capture one local retained evaluation batch. No scheduler or live evaluation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyst_dashboard.governance.forward_capture import capture_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Retained systematic evaluation JSON")
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--start-session", required=True, help="First XNYS capture session, YYYY-MM-DD")
    parser.add_argument("--forward-root", required=True, type=Path, help="Existing isolated local directory")
    parser.add_argument("--initialize", action="store_true", help="Explicitly initialize a new cohort once")
    args = parser.parse_args()
    try:
        result = capture_batch(args.input.read_bytes(), cohort_id=args.cohort_id,
                               start_session=args.start_session, forward_root=args.forward_root,
                               initialize=args.initialize)
    except (ValueError, KeyError, OSError) as e:
        print(json.dumps({"status": "BLOCKED", "reason": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "CAPTURED_OFFLINE_INPUT", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
