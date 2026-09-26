"""ARX Terminal — Batch Acquisition for Frozen V1.2.0 Cache Misses.

Acquires the unique SEC statutory documents corresponding to the 267 target-level
cache misses identified under frozen STATUTORY_FILING_SELECTOR_V1_2_0.

Strict constraints:
- User-Agent compliant with SEC EDGAR policy
- Rate-limited to <= 8 requests/second
- SHA256 hashed and recorded in provenance ledger
- Rejects post-boundary documents (filingDate > 2026-09-24)
- Zero duplicate downloads
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "research" / "cache"
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"
MISS_SET_PATH = REPO_ROOT / "docs" / "research" / "SOURCE_CACHE_MISS_POPULATION_V1_2_0.json"
LEDGER_PATH = REPO_ROOT / "docs" / "research" / "V1_2_0_ACQUISITION_PROVENANCE_LEDGER.json"

USER_AGENT = "ARX Research research@arxterminal.com"
SNAPSHOT_BOUNDARY_DATE = "2026-09-24"


def build_accession_to_cik_and_date_map() -> Dict[str, Dict[str, str]]:
    """Index accession to registrant CIK, filing date, and form across submissions."""
    acc_map = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            cik_val = data.get("cik")
            if not cik_val:
                cik_val = p.stem.replace("CIK", "").split("-")[0]
            cik_int = str(int(cik_val))
            
            # Check recent
            if "filings" in data and "recent" in data["filings"]:
                rec = data["filings"]["recent"]
                accs = rec.get("accessionNumber", [])
                fdates = rec.get("filingDate", [])
                forms = rec.get("form", [])
                for i, a in enumerate(accs):
                    if a not in acc_map:
                        acc_map[a] = {
                            "cik": cik_int,
                            "filing_date": fdates[i] if i < len(fdates) else "",
                            "form": forms[i] if i < len(forms) else "",
                        }
            elif "accessionNumber" in data:
                accs = data.get("accessionNumber", [])
                fdates = data.get("filingDate", [])
                forms = data.get("form", [])
                for i, a in enumerate(accs):
                    if a not in acc_map:
                        acc_map[a] = {
                            "cik": cik_int,
                            "filing_date": fdates[i] if i < len(fdates) else "",
                            "form": forms[i] if i < len(forms) else "",
                        }
        except Exception:
            pass
    return acc_map


def acquire_missing_documents():
    print("=" * 80)
    print("ARX TERMINAL — ACQUIRING UNIQUE CACHE MISS DOCUMENTS (SECTION 7)")
    print("=" * 80)

    assert MISS_SET_PATH.exists(), f"Missing miss set: {MISS_SET_PATH}"
    with open(MISS_SET_PATH, "r", encoding="utf-8") as f:
        miss_data = json.load(f)

    unique_docs = miss_data.get("unique_documents", [])
    print(f"Loaded {len(unique_docs)} unique documents to acquire.")

    PROSPECTUS_DIR.mkdir(parents=True, exist_ok=True)
    acc_map = build_accession_to_cik_and_date_map()
    print(f"Indexed {len(acc_map)} accessions for registrant CIK and metadata resolution.")

    new_files = 0
    new_bytes = 0
    failures = 0
    duplicates = 0
    post_boundary_rejected = 0
    provenance_entries = []

    for idx, item in enumerate(unique_docs):
        cik_str = str(int(item["cik"]))
        acc = item["accession"]
        doc = item["document_filename"]
        target_fname = f"{acc}_{doc}"
        local_path = PROSPECTUS_DIR / target_fname

        # Check metadata
        meta = acc_map.get(acc, {})
        reg_cik = meta.get("cik", cik_str)
        filing_date = meta.get("filing_date", "")
        form = meta.get("form", "")

        # Strict temporal boundary check
        if filing_date and filing_date > SNAPSHOT_BOUNDARY_DATE:
            print(f"[{idx+1}/{len(unique_docs)}] REJECTED POST-BOUNDARY: {target_fname} (filingDate {filing_date} > {SNAPSHOT_BOUNDARY_DATE})")
            post_boundary_rejected += 1
            continue

        if local_path.exists():
            # Already acquired
            duplicates += 1
            print(f"[{idx+1}/{len(unique_docs)}] ALREADY PRESENT: {target_fname}")
            raw_bytes = local_path.read_bytes()
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            provenance_entries.append({
                "cik": reg_cik,
                "accession": acc,
                "form": form,
                "filing_date": filing_date,
                "document_filename": doc,
                "local_filename": target_fname,
                "source_url": f"https://www.sec.gov/Archives/edgar/data/{reg_cik}/{acc.replace('-', '')}/{doc}",
                "byte_length": len(raw_bytes),
                "sha256": sha256,
                "status": "ALREADY_CACHED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            continue

        # Download from SEC EDGAR
        # Download from SEC EDGAR with retry logic
        acc_no_hyphen = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{reg_cik}/{acc_no_hyphen}/{doc}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        data = None
        last_error = None
        for attempt in range(3):
            try:
                time.sleep(0.15)  # <= 6.6 req/sec to be well within 10 req/s limit
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                break
            except Exception as e:
                last_error = e
                time.sleep(1.0 * (attempt + 1))

        if data is None and reg_cik != cik_str:
            # Fallback to cik_str
            url_fallback = f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_no_hyphen}/{doc}"
            req_fallback = urllib.request.Request(url_fallback, headers={"User-Agent": USER_AGENT})
            for attempt in range(3):
                try:
                    time.sleep(0.15)
                    with urllib.request.urlopen(req_fallback, timeout=15) as resp:
                        data = resp.read()
                    url = url_fallback
                    reg_cik = cik_str
                    break
                except Exception as e_fb:
                    last_error = e_fb
                    time.sleep(1.0 * (attempt + 1))

        if data is not None:
            sha256 = hashlib.sha256(data).hexdigest()
            local_path.write_bytes(data)
            new_files += 1
            new_bytes += len(data)

            provenance_entries.append({
                "cik": reg_cik,
                "accession": acc,
                "form": form,
                "filing_date": filing_date,
                "document_filename": doc,
                "local_filename": target_fname,
                "source_url": url,
                "byte_length": len(data),
                "sha256": sha256,
                "status": "ACQUIRED_SUCCESS",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[{idx+1}/{len(unique_docs)}] ACQUIRED: {target_fname} ({len(data):,} bytes)")
        else:
            failures += 1
            print(f"[{idx+1}/{len(unique_docs)}] FAILURE: {target_fname} from {url} - {str(last_error)}")
            provenance_entries.append({
                "cik": reg_cik,
                "accession": acc,
                "form": form,
                "filing_date": filing_date,
                "document_filename": doc,
                "local_filename": target_fname,
                "source_url": url,
                "byte_length": 0,
                "sha256": "NONE",
                "status": f"ACQUISITION_FAILURE: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # Save provenance ledger
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total_documents_attempted": len(unique_docs),
            "new_source_files": new_files,
            "new_source_bytes": new_bytes,
            "acquisition_failures": failures,
            "duplicate_downloads": duplicates,
            "post_boundary_rejected": post_boundary_rejected,
            "entries": provenance_entries,
        }, f, indent=2)

    print("\nACQUISITION SUMMARY:")
    print(f"NEW_SOURCE_FILES = {new_files}")
    print(f"NEW_SOURCE_BYTES = {new_bytes:,}")
    print(f"ACQUISITION_FAILURES = {failures}")
    print(f"DUPLICATE_DOWNLOADS = {duplicates}")
    print(f"POST_BOUNDARY_DOCUMENTS_ACQUIRED_FOR_SELECTION = {post_boundary_rejected}")
    print(f"Saved provenance ledger to: {LEDGER_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    acquire_missing_documents()
