"""ARX Terminal — One-pass download of the final 70 missing statutory files.
Closes all remaining 115 SOURCE_CACHE_MISS targets to 0.
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

from scripts.research.statutory_filing_selector import StatutoryFilingSelector
from scripts.research.series_prospectus_mapper import SeriesMetadata

INPUT_MANIFEST_PATH = Path("docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json")
CACHE_DIR = Path("data/research/cache")
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"
LEDGER_PATH = Path("docs/research/SOURCE_PROVENANCE_LEDGER.json")

HEADERS = {"User-Agent": "ArxTerminal/1.0 (research@arxterminal.org)"}


def main():
    manifest = json.load(open(INPUT_MANIFEST_PATH, encoding="utf-8"))["records"]
    cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()

    submission_cache = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        submission_cache[cik_str] = json.load(open(p, encoding="utf-8"))

    file_cache = {}
    files_to_download = {}

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
            fname = f"{res.selected_accession}_{res.document_filename}"
            if fname not in cached_filenames and fname not in files_to_download:
                files_to_download[fname] = {
                    "cik": target.cik,
                    "accession": res.selected_accession,
                    "doc": res.document_filename,
                    "form": res.selected_form,
                    "date": res.filing_date,
                }

    print(f"Total unique missing files to download: {len(files_to_download)}")
    ledger = json.load(open(LEDGER_PATH, encoding="utf-8"))

    new_downloads = 0
    failures = 0

    for idx, (fname, info) in enumerate(files_to_download.items()):
        cik = info["cik"]
        acc = info["accession"]
        doc = info["doc"]
        form = info["form"]
        fdate = info["date"]

        local_path = PROSPECTUS_DIR / fname
        if local_path.exists() and fname in ledger.get("files", {}):
            continue

        nodash = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/{doc}"

        time.sleep(0.12)
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                content = r.content
                sha = hashlib.sha256(content).hexdigest()
                with open(local_path, "wb") as f:
                    f.write(content)
                ledger["files"][fname] = {
                    "cik": cik,
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
                if (idx + 1) % 10 == 0 or idx == len(files_to_download) - 1:
                    print(f"[{idx+1}/{len(files_to_download)}] Downloaded {fname} ({len(content):,} bytes)")
            else:
                failures += 1
                print(f"[{idx+1}/{len(files_to_download)}] HTTP_{r.status_code} for {fname}")
        except Exception as e:
            failures += 1
            print(f"[{idx+1}/{len(files_to_download)}] Error for {fname}: {e}")

    ledger["total_files"] = len(ledger["files"])
    ledger["total_bytes"] = sum(f["byte_length"] for f in ledger["files"].values())
    ledger["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    print(f"Downloaded {new_downloads} files, {failures} failures. Total files in ledger: {ledger['total_files']}")


if __name__ == "__main__":
    main()
