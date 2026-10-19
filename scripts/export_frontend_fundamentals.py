"""
scripts/export_frontend_fundamentals.py

Generates and validates static snapshot JSON for frontend SSG and hydration
from the authoritative sourced fundamentals ledger.

Usage:
    python scripts/export_frontend_fundamentals.py          # Export snapshot
    python scripts/export_frontend_fundamentals.py --check  # Verify snapshot parity (fails with code 1 if drift)
"""

import sys
import json
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.research.sourced_fundamentals.service import get_sourced_fundamentals_service


def generate_snapshot() -> dict:
    service = get_sourced_fundamentals_service()
    meta = service.get_snapshot_metadata()
    companies_catalog = service.list_available_companies()

    snapshot_export = {
        "metadata": meta,
        "companies": {},
    }

    for c in companies_catalog:
        sym = c["symbol"]
        data = service.get_company_snapshot(sym)
        if data:
            snapshot_export["companies"][sym] = data

    return snapshot_export


def main():
    target = Path("frontend/lib/sourcedFundamentalsSnapshot.json")
    check_mode = "--check" in sys.argv

    current_data = generate_snapshot()

    if check_mode:
        if not target.exists():
            print(f"[FAIL] Target snapshot does not exist at {target}", file=sys.stderr)
            sys.exit(1)

        with open(target, "r", encoding="utf-8") as f:
            disk_data = json.load(f)

        if current_data != disk_data:
            print(
                f"[FAIL] Stale snapshot detected! '{target}' is out of sync with service ledger.\n"
                f"Run 'python scripts/export_frontend_fundamentals.py' to update.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[PASS] Snapshot parity verified: '{target}' exactly matches service ledger.")
        sys.exit(0)

    # Export mode
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2)

    print(f"[OK] Exported {len(current_data['companies'])} companies to {target}")


if __name__ == "__main__":
    main()
