#!/usr/bin/env python3
"""
scripts/research/check_maintenance_events.py

Manually executable event-check command for ARX Terminal research maintenance.
Suitable for later cron/scheduler integration.

Usage:
  python scripts/research/check_maintenance_events.py --symbol FIX
  python scripts/research/check_maintenance_events.py --symbol FIX --accession 0001104659-26-118159
  python scripts/research/check_maintenance_events.py --symbol FIX --replay --accession 0001104659-26-086258
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.research.maintenance.cli import main

if __name__ == "__main__":
    sys.exit(main())
