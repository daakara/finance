"""ARX Terminal — Inspect Candidate Metadata for the 12 Unique Files."""

import json
from pathlib import Path

FROZEN_PATH = Path("docs/research/FROZEN_RESIDUAL_46_CACHE_MISSES.json")
SUBMISSIONS_DIR = Path("data/research/cache/sec_submissions")


def main():
    data = json.load(open(FROZEN_PATH, encoding="utf-8"))["records"]
    unique = {}
    for r in data:
        fn = r["expected_filename"]
        unique.setdefault(fn, []).append(r)

    print(f"Total unique files: {len(unique)}\n")

    for fn, targets in unique.items():
        first = targets[0]
        cik = str(first["cik"]).zfill(10)
        acc = first["candidate_accession"]
        form = first["candidate_form"]
        doc = first["candidate_document_filename"]
        fdate = first["candidate_filing_date"]

        sub_path = SUBMISSIONS_DIR / f"CIK{cik}.json"
        desc = "N/A"
        if sub_path.exists():
            sub = json.load(open(sub_path, encoding="utf-8"))
            recent = sub.get("filings", {}).get("recent", {})
            accs = recent.get("accessionNumber", [])
            if acc in accs:
                idx = accs.index(acc)
                desc = recent.get("primaryDocDescription", [])[idx]

        syms = [t["symbol"] for t in targets]
        names = [t["legal_name"] for t in targets]
        print(f"File: {fn}")
        print(f"  CIK: {cik} | Form: {form} | Date: {fdate} | Doc: {doc}")
        print(f"  SEC Description: {desc}")
        print(f"  Covers {len(syms)} targets: {syms}")
        print(f"  First legal name: {names[0]}")
        print("-" * 80)


if __name__ == "__main__":
    main()
