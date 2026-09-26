"""ARX Terminal — Classify Residual 46 Cache Misses (Sections 4 & 5).

Audits each residual cache miss for:
1. Accession known
2. Document filename known
3. Pre-boundary eligibility
4. Local source presence
5. Provenance ledger presence
6. Target-specific evidence verification (Section 5)
7. Root cause assignment
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.statutory_filing_selector import StatutoryFilingSelector
from scripts.research.series_prospectus_mapper import SeriesMetadata

FROZEN_PATH = Path("docs/research/FROZEN_RESIDUAL_46_CACHE_MISSES.json")
LEDGER_PATH = Path("docs/research/SOURCE_PROVENANCE_LEDGER.json")
PROSPECTUS_DIR = Path("data/research/cache/sec_prospectus")
OUTPUT_CLASSIFICATION_PATH = Path("docs/research/RESIDUAL_46_CLASSIFICATION.json")


def main():
    print("=" * 80)
    print("ARX TERMINAL — CLASSIFY RESIDUAL 46 CACHE MISSES")
    print("=" * 80)

    data = json.load(open(FROZEN_PATH, encoding="utf-8"))
    records = data["records"]
    ledger = json.load(open(LEDGER_PATH, encoding="utf-8"))
    ledger_files = ledger.get("files", {})

    classified: List[Dict[str, Any]] = []

    unique_files = {}

    for r in records:
        sym = r["symbol"]
        cik = r["cik"]
        sid = r["series_id"]
        cid = r["class_id"]
        name = r["legal_name"]
        acc = r["candidate_accession"]
        form = r["candidate_form"]
        fdate = r["candidate_filing_date"]
        doc = r["candidate_document_filename"]
        fname = r["expected_filename"]

        target = SeriesMetadata(symbol=sym, cik=cik, series_id=sid, class_id=cid, legal_name=name)

        # 1. Existence / metadata checks
        acc_known = bool(acc and acc != "NONE")
        doc_known = bool(doc and doc != "NONE")
        eligible_preboundary = bool(fdate and fdate <= "2026-09-24")
        local_present = (PROSPECTUS_DIR / fname).exists()
        ledger_present = fname in ledger_files

        # 2. Section 5: Target-specific evidence verification
        target_evidence = StatutoryFilingSelector.match_target_metadata(target, doc, r.get("selection_reason", ""))

        # 3. Root cause assignment
        if not acc_known or not doc_known:
            root_cause = "SELECTOR_FALSE_POSITIVE_CACHE_MISS"
        elif not eligible_preboundary:
            root_cause = "POST_BOUNDARY_DISQUALIFIED"
        elif local_present and ledger_present:
            root_cause = "STALE_CACHE_INDEX"
        elif local_present and not ledger_present:
            root_cause = "LEDGER_MISSING_BUT_FILE_PRESENT"
        elif not local_present and ledger_present:
            root_cause = "LOCAL_FILE_MISSING_BUT_LEDGER_PRESENT"
        elif not target_evidence:
            root_cause = "METADATA_ONLY_FALSE_POSITIVE"
        else:
            root_cause = "KNOWN_ELIGIBLE_DOCUMENT_NOT_CACHED"

        item = {
            "symbol": sym,
            "cik": cik,
            "series_id": sid,
            "class_id": cid,
            "legal_name": name,
            "candidate_accession": acc,
            "candidate_form": form,
            "candidate_filing_date": fdate,
            "candidate_document_filename": doc,
            "expected_filename": fname,
            "accession_known": "YES" if acc_known else "NO",
            "document_filename_known": "YES" if doc_known else "NO",
            "document_eligible_preboundary": "YES" if eligible_preboundary else "NO",
            "local_source_present": "YES" if local_present else "NO",
            "provenance_ledger_entry_present": "YES" if ledger_present else "NO",
            "target_specific_evidence_confirmed": "YES" if target_evidence else "NO",
            "root_cause": root_cause,
        }
        classified.append(item)

        if root_cause == "KNOWN_ELIGIBLE_DOCUMENT_NOT_CACHED":
            if fname not in unique_files:
                unique_files[fname] = {
                    "cik": cik,
                    "accession": acc,
                    "form": form,
                    "filing_date": fdate,
                    "document_filename": doc,
                    "covered_targets": []
                }
            unique_files[fname]["covered_targets"].append(sym)

    output = {
        "total_residual_records": len(classified),
        "root_cause_counts": {},
        "unique_files_to_acquire_count": len(unique_files),
        "unique_files_to_acquire": unique_files,
        "records": classified
    }

    from collections import Counter
    rc_counts = Counter(r["root_cause"] for r in classified)
    output["root_cause_counts"] = dict(rc_counts)

    with open(OUTPUT_CLASSIFICATION_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nRoot Cause Breakdown:")
    for rc, count in rc_counts.items():
        print(f"  {rc}: {count}")

    print(f"\nUnique Files to Acquire: {len(unique_files)}")
    for fn, info in unique_files.items():
        print(f"  {fn} (CIK {info['cik']}, Form {info['form']}, Date {info['filing_date']}) -> covers {len(info['covered_targets'])} targets: {info['covered_targets']}")

    print(f"\nSaved classification to: {OUTPUT_CLASSIFICATION_PATH}")


if __name__ == "__main__":
    main()
