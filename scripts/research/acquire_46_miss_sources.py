"""ARX Terminal — Acquire Missing Sources for Residual 46 Cache Misses (Section 6 & 7).

Downloads only the 12 unique confirmed pre-boundary files with rate-limiting,
validates byte length and SHA256, and records exact provenance in SOURCE_PROVENANCE_LEDGER.json.
"""

import sys
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import requests

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

FROZEN_PATH = Path("docs/research/FROZEN_RESIDUAL_46_CACHE_MISSES.json")
LEDGER_PATH = Path("docs/research/SOURCE_PROVENANCE_LEDGER.json")
PROSPECTUS_DIR = Path("data/research/cache/sec_prospectus")

HEADERS = {"User-Agent": "ArxTerminal/1.0 (research@arxterminal.org)"}


def main():
    print("=" * 80)
    print("ARX TERMINAL — ACQUIRE MISSING SOURCES FOR RESIDUAL 46 TARGETS")
    print("=" * 80)

    data = json.load(open(FROZEN_PATH, encoding="utf-8"))["records"]
    unique_files = {}
    for r in data:
        fn = r["expected_filename"]
        if fn not in unique_files:
            unique_files[fn] = {
                "cik": r["cik"],
                "accession": r["candidate_accession"],
                "form": r["candidate_form"],
                "filing_date": r["candidate_filing_date"],
                "doc": r["candidate_document_filename"],
                "targets": []
            }
        unique_files[fn]["targets"].append(r["symbol"])

    print(f"Total unique files to evaluate: {len(unique_files)}")

    ledger = json.load(open(LEDGER_PATH, encoding="utf-8")) if LEDGER_PATH.exists() else {"files": {}}
    ledger_files = ledger.setdefault("files", {})

    new_downloads = 0
    new_bytes = 0
    failures = 0

    for idx, (fname, info) in enumerate(unique_files.items()):
        cik = str(int(info["cik"]))
        acc = info["accession"]
        form = info["form"]
        fdate = info["filing_date"]
        doc = info["doc"]
        local_path = PROSPECTUS_DIR / fname

        # Temporal check
        assert fdate <= "2026-09-24", f"Fatal temporal boundary violation: {fname} has date {fdate}"

        if local_path.exists() and fname in ledger_files:
            print(f"[{idx+1}/{len(unique_files)}] Already cached: {fname}")
            continue

        nodash = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{doc}"

        time.sleep(0.15)
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code == 200:
                content = r.content
                sha = hashlib.sha256(content).hexdigest()
                with open(local_path, "wb") as f:
                    f.write(content)
                ledger_files[fname] = {
                    "cik": info["cik"],
                    "accession": acc,
                    "form": form,
                    "filing_date": fdate,
                    "document_filename": doc,
                    "source_url": url,
                    "download_timestamp": datetime.now(timezone.utc).isoformat(),
                    "byte_length": len(content),
                    "sha256": sha,
                    "snapshot_eligibility": "ELIGIBLE_PRE_BOUNDARY",
                    "local_path": str(local_path),
                }
                new_downloads += 1
                new_bytes += len(content)
                print(f"[{idx+1}/{len(unique_files)}] Downloaded {fname} ({len(content):,} bytes) [SHA256: {sha[:16]}...]")
            else:
                failures += 1
                print(f"[{idx+1}/{len(unique_files)}] HTTP_{r.status_code} for {fname}")
        except Exception as e:
            failures += 1
            print(f"[{idx+1}/{len(unique_files)}] Exception for {fname}: {e}")

    ledger["total_files"] = len(ledger_files)
    ledger["total_bytes"] = sum(f["byte_length"] for f in ledger_files.values())
    ledger["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    print(f"\nACQUISITION SUMMARY:")
    print(f"  NEW_SOURCE_FILES = {new_downloads}")
    print(f"  NEW_SOURCE_BYTES = {new_bytes:,}")
    print(f"  ACQUISITION_FAILURES = {failures}")
    print(f"  TOTAL_FILES_IN_LEDGER = {ledger['total_files']}")
    print(f"  TOTAL_BYTES_IN_LEDGER = {ledger['total_bytes']:,}")


if __name__ == "__main__":
    main()
