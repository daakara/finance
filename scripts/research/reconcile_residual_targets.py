"""ARX Terminal — Residual Target Reconciliation Script (Sections 2, 3).
Freezes and prints the exact 16 residual targets and failure mode proofs.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.statutory_filing_selector import StatutoryFilingSelector
from scripts.research.series_prospectus_mapper import SeriesMetadata

INPUT_MANIFEST_PATH = Path("docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json")
CACHE_DIR = Path("data/research/cache")
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"


def main():
    manifest = json.load(open(INPUT_MANIFEST_PATH, encoding="utf-8"))["records"]
    cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()

    submission_cache = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        submission_cache[cik_str] = json.load(open(p, encoding="utf-8"))

    file_cache = {}
    by_cik = defaultdict(list)
    for r in manifest:
        target = SeriesMetadata(
            symbol=r["symbol"],
            cik=str(r["cik"]).lstrip("0") or "0",
            series_id=r["series_id"],
            class_id=r["class_id"],
            legal_name=r["legal_name"]
        )
        sub_json = submission_cache.get(target.cik, {})
        res = StatutoryFilingSelector.select_statutory_filing(
            target, sub_json, CACHE_DIR, file_cache=file_cache, cached_filenames=cached_filenames
        )
        if res.selection_outcome == "SOURCE_CACHE_MISS":
            by_cik[target.cik].append({
                "symbol": target.symbol,
                "cik": target.cik,
                "series_id": target.series_id,
                "class_id": target.class_id,
                "legal_name": target.legal_name,
                "candidate_accession": res.selected_accession,
                "document_filename": res.document_filename,
                "filing_date": res.filing_date,
            })

    total_misses = sum(len(ts) for ts in by_cik.values())
    print(f"TOTAL RESIDUAL TARGETS: {total_misses}")
    print("=" * 100)

    for cik, targets in sorted(by_cik.items()):
        c_name = submission_cache.get(cik, {}).get("name", "Unknown")
        print(f"\nREGISTRANT: {c_name} (CIK {cik}) — {len(targets)} targets")
        print("-" * 100)
        for t in targets:
            print(
                f"Symbol: {t['symbol']:<5} | Series: {t['series_id']:<10} | Class: {t['class_id']:<10} | "
                f"Acc: {t['candidate_accession']} | File: {t['document_filename']:<25} | "
                f"Date: {t['filing_date']} | Name: {t['legal_name']}"
            )

    # Save to JSON
    out_path = Path("docs/research/FROZEN_RESIDUAL_16_TARGETS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_residual_targets": total_misses,
            "by_cik": {
                cik: {
                    "registrant_name": submission_cache.get(cik, {}).get("name", "Unknown"),
                    "targets": ts,
                }
                for cik, ts in by_cik.items()
            }
        }, f, indent=2)
    print(f"\nSaved frozen residual targets to {out_path}")


if __name__ == "__main__":
    main()
