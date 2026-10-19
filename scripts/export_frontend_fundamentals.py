"""
scripts/export_frontend_fundamentals.py

Generates static snapshot JSON for frontend SSG and hydration from the sourced fundamentals ledger.
"""

import json
from pathlib import Path
from analyst_dashboard.research.sourced_fundamentals.service import get_sourced_fundamentals_service

def main():
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

    target = Path("frontend/lib/sourcedFundamentalsSnapshot.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(snapshot_export, f, indent=2)

    print(f"Wrote {len(snapshot_export['companies'])} companies to {target}")

if __name__ == "__main__":
    main()
