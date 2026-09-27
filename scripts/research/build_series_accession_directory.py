"""ARX Terminal — Authoritative Series-Accession Directory Builder & Cache Acquisition.

Queries SEC EDGAR atom feeds for each ETF Series ID to obtain the authoritative
pre-boundary Form 497K (or Form 485BPOS) filing accession, strictly respecting:
1. SNAPSHOT_BOUNDARY = 2026-09-24T23:59:59Z (dateb=20260924).
2. Rate-limiting: max 8 req/sec (0.12s sleep).
3. Primary document lookup via local sec_submissions cache.
4. Single-pass pre-boundary prospectus acquisition with SHA-256 provenance.
"""

import sys
import json
import time
import hashlib
import requests
import re
from pathlib import Path
from typing import Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.acquire_prospectus import download_filing

CACHE_DIR = REPO_ROOT / "data" / "research" / "cache"
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"
FAILURE_CORPUS_PATH = REPO_ROOT / "docs" / "research" / "ETF_MANDATE_FAILURE_CORPUS_PRE_V1_3_0.json"
DIRECTORY_OUTPUT_PATH = REPO_ROOT / "data" / "research" / "sec_series_accession_directory_v1.json"
PROVENANCE_OUTPUT_PATH = REPO_ROOT / "docs" / "research" / "V1_3_0_ACQUISITION_PROVENANCE_LEDGER.json"

HEADERS = {"User-Agent": "ArxTerminal/1.0 (research@arxterminal.org)"}
SNAPSHOT_BOUNDARY_DATE = "2026-09-24"


def build_submission_accession_map():
    """Build fast lookup map: (cik_10digit, accession) -> (primary_document, form, filing_date)."""
    acc_map = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_cik = data.get("cik") or p.stem.replace("CIK", "").split("-")[0]
            cik_10 = str(int(raw_cik)).zfill(10)
            
            # recent
            if "filings" in data and "recent" in data["filings"]:
                rec = data["filings"]["recent"]
                for a, doc, fdate, form in zip(
                    rec.get("accessionNumber", []),
                    rec.get("primaryDocument", []),
                    rec.get("filingDate", []),
                    rec.get("form", [])
                ):
                    if a:
                        acc_map[(cik_10, a)] = (doc, form, fdate)
            # historical files
            if "accessionNumber" in data:
                for a, doc, fdate, form in zip(
                    data.get("accessionNumber", []),
                    data.get("primaryDocument", []),
                    data.get("filingDate", []),
                    data.get("form", [])
                ):
                    if a:
                        acc_map[(cik_10, a)] = (doc, form, fdate)
        except Exception:
            pass
    return acc_map


def query_edgar_series(series_id: str, form_type: str = "497K") -> Optional[Dict[str, str]]:
    """Query EDGAR atom feed for a specific series ID pre-boundary."""
    url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={series_id}&type={form_type}"
        f"&dateb=20260924&owner=exclude&count=5&output=atom"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL)
        for e in entries:
            acc_m = re.search(r"accession-number>(.*?)</", e)
            up_m = re.search(r"<updated>(.*?)</updated>", e)
            title_m = re.search(r"<title>(.*?)</title>", e)
            if acc_m and up_m:
                acc = acc_m.group(1).strip()
                fdate = up_m.group(1)[:10].strip()
                form = title_m.group(1).split()[0].strip() if title_m else form_type
                # Enforce strict pre-boundary check
                if fdate <= SNAPSHOT_BOUNDARY_DATE:
                    return {"accession": acc, "filing_date": fdate, "form": form}
    except Exception as exc:
        print(f"Error querying EDGAR for {series_id} {form_type}: {exc}")
    return None


def main():
    print("=" * 80)
    print("ARX TERMINAL — SEC SERIES ACCESSION DIRECTORY BUILDER & ACQUISITION")
    print(f"Snapshot Boundary: {SNAPSHOT_BOUNDARY_DATE} 23:59:59Z")
    print("=" * 80)

    with open(FAILURE_CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} pre-V1.3.0 failure targets.")
    acc_map = build_submission_accession_map()
    print(f"Indexed {len(acc_map)} accession records from local submissions.")

    # Load existing directory if present to allow restartability
    directory = {}
    if DIRECTORY_OUTPUT_PATH.exists():
        try:
            with open(DIRECTORY_OUTPUT_PATH, "r", encoding="utf-8") as f:
                directory = json.load(f)
            print(f"Loaded {len(directory)} existing series directory records.")
        except Exception:
            directory = {}

    provenance_entries = []
    acquired_count = 0
    cached_count = 0
    missing_doc_count = 0

    total = len(corpus)
    for i, item in enumerate(corpus, 1):
        sid = item["series_id"]
        sym = item["symbol"]
        raw_cik = item["cik"]
        cik_10 = str(int(raw_cik)).zfill(10)

        if sid in directory:
            rec = directory[sid]
        else:
            # Step 1: Query 497K
            rec = query_edgar_series(sid, "497K")
            time.sleep(0.12)
            if not rec:
                # Step 2: Query 485BPOS if 497K not found
                rec = query_edgar_series(sid, "485BPOS")
                time.sleep(0.12)

            if rec:
                acc = rec["accession"]
                # Lookup primary document
                lookup = acc_map.get((cik_10, acc))
                if lookup:
                    rec["primary_document"] = lookup[0]
                    rec["form"] = lookup[1]
                    rec["filing_date"] = lookup[2]
                else:
                    # Check if accession exists in any CIK
                    found = False
                    for (c, a), val in acc_map.items():
                        if a == acc:
                            rec["primary_document"] = val[0]
                            rec["form"] = val[1]
                            rec["filing_date"] = val[2]
                            found = True
                            break
                    if not found:
                        rec["primary_document"] = "UNKNOWN"

                rec["symbol"] = sym
                rec["cik"] = cik_10
                directory[sid] = rec

        if rec:
            acc = rec.get("accession")
            pdoc = rec.get("primary_document")
            fdate = rec.get("filing_date")
            form = rec.get("form")

            if pdoc and pdoc != "UNKNOWN":
                local_fname = f"{acc}_{pdoc}"
                local_path = PROSPECTUS_DIR / local_fname
                if local_path.exists():
                    cached_count += 1
                else:
                    # Download once with rate limit
                    ok, fpath, sha = download_filing(cik_10, acc, pdoc)
                    if ok:
                        acquired_count += 1
                        provenance_entries.append({
                            "symbol": sym,
                            "series_id": sid,
                            "cik": cik_10,
                            "accession": acc,
                            "document_filename": pdoc,
                            "form": form,
                            "filing_date": fdate,
                            "source_sha256": sha,
                            "status": "ACQUIRED_SUCCESS"
                        })
                    else:
                        missing_doc_count += 1
            else:
                missing_doc_count += 1

        if i % 50 == 0 or i == total:
            print(f"Progress: [{i}/{total}] | Directory entries: {len(directory)} | Cached: {cached_count} | Acquired: {acquired_count} | Unresolved: {missing_doc_count}")
            # Persist checkpoint
            with open(DIRECTORY_OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(directory, f, indent=2)

    # Final persist
    with open(DIRECTORY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(directory, f, indent=2)

    # Persist acquisition provenance
    prov_ledger = {
        "ledger_version": "V1_3_0_ACQUISITION_PROVENANCE_LEDGER",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot_boundary": f"{SNAPSHOT_BOUNDARY_DATE}T23:59:59Z",
        "total_series_attempted": total,
        "authoritative_series_mapped": len(directory),
        "newly_acquired_documents": acquired_count,
        "entries": provenance_entries
    }
    with open(PROVENANCE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prov_ledger, f, indent=2)

    print("\n" + "=" * 80)
    print("DIRECTORY BUILD & ACQUISITION COMPLETE")
    print(f"Total Series Attempted: {total}")
    print(f"Authoritative Series Mapped: {len(directory)}")
    print(f"Newly Acquired Files: {acquired_count}")
    print(f"Already Cached Files: {cached_count}")
    print(f"Directory written to: {DIRECTORY_OUTPUT_PATH}")
    print(f"Provenance written to: {PROVENANCE_OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
