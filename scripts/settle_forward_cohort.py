#!/usr/bin/env python3
"""Replay retained daily prices into isolated paper-pilot settlement storage."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyst_dashboard.governance.forward_settlement import settle_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--cohort-id', required=True)
    parser.add_argument('--forward-root', required=True, type=Path)
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    try:
        result = settle_batch(args.input.read_bytes(), cohort_id=args.cohort_id,
                              forward_root=args.forward_root, initialize=args.initialize)
    except (ValueError, KeyError, TypeError, OSError) as e:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({'status': 'SETTLED_OFFLINE_INPUT', **result}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
