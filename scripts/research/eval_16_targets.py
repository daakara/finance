"""ARX Terminal — Evaluation of Residual 16 Targets under STATUTORY_FILING_SELECTOR_V1_1_0.
"""

import sys
import json
from pathlib import Path

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
    manifest = {r["symbol"]: r for r in json.load(open(INPUT_MANIFEST_PATH, encoding="utf-8"))["records"]}
    cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()

    submission_cache = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        submission_cache[cik_str] = json.load(open(p, encoding="utf-8"))

    file_cache = {}

    targets_16 = [
        "GEM", "GGUS", "GIND", "GPIX", "GSEE", "GSEU", "GSID", "GSIE", "GSJY", "GSLC", "GSUS", "GVUS",
        "XLC", "XLFI", "XOEX", "VT"
    ]

    print("=== Section 12: Re-evaluation of the 16 Targets ===")
    print("-" * 120)
    print(f"{'Symbol':<6} | {'CIK':<8} | {'Outcome':<36} | {'Form':<8} | {'Accession':<22} | {'Document File':<28} | {'Date'}")
    print("-" * 120)

    results = []
    for sym in targets_16:
        r = manifest[sym]
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
        results.append({
            "symbol": sym,
            "cik": target.cik,
            "outcome": res.selection_outcome,
            "form": res.selected_form,
            "accession": res.selected_accession,
            "doc": res.document_filename,
            "date": res.filing_date,
            "candidate_count": res.candidate_count,
        })
        print(f"{sym:<6} | {target.cik:<8} | {res.selection_outcome:<36} | {res.selected_form:<8} | {res.selected_accession:<22} | {res.document_filename:<28} | {res.filing_date}")

    out_json = Path("docs/research/RESIDUAL_16_EVALUATION.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation to {out_json}")


if __name__ == "__main__":
    main()
