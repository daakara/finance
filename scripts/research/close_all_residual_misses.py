"""ARX Terminal — Close All Residual Cache Misses Iteratively.

Iterates over the frozen 267 cache miss targets, identifying cascading missing documents,
acquiring them from SEC EDGAR with rate limiting, retry backoff, and SHA-256 recording,
until SOURCE_CACHE_MISS drops to exactly 0 across the entire target set.
"""

import sys
import json
import time
import hashlib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.statutory_filing_selector import StatutoryFilingSelector
from scripts.research.series_prospectus_mapper import SeriesMetadata

CACHE_DIR = REPO_ROOT / "data" / "research" / "cache"
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"
MISS_SET_PATH = REPO_ROOT / "docs" / "research" / "SOURCE_CACHE_MISS_POPULATION_V1_2_0.json"
LEDGER_PATH = REPO_ROOT / "docs" / "research" / "V1_2_0_ACQUISITION_PROVENANCE_LEDGER.json"
EVAL_PATH = REPO_ROOT / "docs" / "research" / "V1_2_0_RESIDUAL_267_EVALUATION.json"

USER_AGENT = "ARX Research research@arxterminal.com"
SNAPSHOT_BOUNDARY_DATE = "2026-09-24"


def build_accession_meta_map():
    acc_map = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            cik_val = data.get("cik") or p.stem.replace("CIK", "").split("-")[0]
            cik_int = str(int(cik_val))
            if "filings" in data and "recent" in data["filings"]:
                rec = data["filings"]["recent"]
                for a, doc, fdate, form in zip(rec.get("accessionNumber", []), rec.get("primaryDocument", []), rec.get("filingDate", []), rec.get("form", [])):
                    if a and a not in acc_map:
                        acc_map[a] = {"cik": cik_int, "doc": doc, "fdate": fdate, "form": form}
            elif "accessionNumber" in data:
                for a, doc, fdate, form in zip(data.get("accessionNumber", []), data.get("primaryDocument", []), data.get("filingDate", []), data.get("form", [])):
                    if a and a not in acc_map:
                        acc_map[a] = {"cik": cik_int, "doc": doc, "fdate": fdate, "form": form}
        except Exception:
            pass
    return acc_map


def run_closure_loop():
    print("=" * 80)
    print("ARX TERMINAL — CLOSING RESIDUAL CACHE MISSES ITERATIVELY")
    print("=" * 80)

    miss_data = json.load(open(MISS_SET_PATH, encoding="utf-8"))
    targets = miss_data["targets"]

    sub_cache = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        if "-submissions-" in p.name:
            continue
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        try:
            sub_cache[cik_str] = json.load(open(p, encoding="utf-8"))
        except Exception:
            pass

    acc_map = build_accession_meta_map()
    ledger = json.load(open(LEDGER_PATH, encoding="utf-8"))
    existing_entries = ledger.get("entries", [])
    already_acquired_files = {e["local_filename"] for e in existing_entries if e.get("status") == "ACQUIRED_SUCCESS"}

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- EVALUATION ITERATION {iteration} ---")

        cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()
        file_cache = {}

        outcomes = Counter()
        results = []
        pending_misses = []

        for t in targets:
            cik = str(t["cik"]).lstrip("0") or "0"
            target = SeriesMetadata(
                symbol=t["symbol"],
                cik=cik,
                series_id=t["series_id"],
                class_id=t["class_id"],
                legal_name=t["legal_name"],
            )
            sub_json = sub_cache.get(cik, {})
            res = StatutoryFilingSelector.select_statutory_filing(
                target, sub_json, CACHE_DIR, file_cache=file_cache, cached_filenames=cached_filenames
            )
            outcomes[res.selection_outcome] += 1
            entry = {
                "symbol": t["symbol"],
                "outcome": res.selection_outcome,
                "accession": res.selected_accession,
                "doc": res.document_filename,
                "form": res.selected_form,
            }
            results.append(entry)
            if res.selection_outcome == "SOURCE_CACHE_MISS":
                pending_misses.append(entry)

        print(f"Iteration {iteration} Outcomes across {len(targets)} targets:")
        for k, v in outcomes.items():
            print(f"  {k}: {v}")

        # Save evaluation snapshot
        with open(EVAL_PATH, "w", encoding="utf-8") as f:
            json.dump({"total": len(targets), "outcomes": dict(outcomes), "results": results}, f, indent=2)

        if not pending_misses:
            print("\nSUCCESS: All targets resolved! SOURCE_CACHE_MISS = 0.")
            break

        # Collect unique docs to acquire
        unique_to_acquire = sorted({(r["accession"], r["doc"], r["form"]) for r in pending_misses if r["doc"]})
        print(f"Found {len(unique_to_acquire)} unique files needed for {len(pending_misses)} pending misses.")

        new_downloads = 0
        for idx, (acc, doc, form) in enumerate(unique_to_acquire):
            target_fname = f"{acc}_{doc}"
            local_path = PROSPECTUS_DIR / target_fname

            if local_path.exists():
                continue

            meta = acc_map.get(acc, {})
            reg_cik = meta.get("cik", "")
            fdate = meta.get("fdate", "")

            acc_no_hyphen = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{reg_cik}/{acc_no_hyphen}/{doc}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

            data = None
            last_err = None
            for attempt in range(3):
                try:
                    time.sleep(0.15)
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = resp.read()
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1.0 * (attempt + 1))

            if data is not None:
                sha = hashlib.sha256(data).hexdigest()
                local_path.write_bytes(data)
                new_downloads += 1
                ledger["new_source_files"] += 1
                ledger["new_source_bytes"] += len(data)
                already_acquired_files.add(target_fname)
                existing_entries.append({
                    "cik": reg_cik,
                    "accession": acc,
                    "form": form,
                    "filing_date": fdate,
                    "document_filename": doc,
                    "local_filename": target_fname,
                    "source_url": url,
                    "byte_length": len(data),
                    "sha256": sha,
                    "status": "ACQUIRED_SUCCESS",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                print(f"  [{idx+1}/{len(unique_to_acquire)}] ACQUIRED: {target_fname} ({len(data):,} bytes)")
            else:
                ledger["acquisition_failures"] += 1
                print(f"  [{idx+1}/{len(unique_to_acquire)}] FAILURE: {target_fname} from {url} - {last_err}")

        ledger["total_documents_attempted"] += len(unique_to_acquire)
        ledger["entries"] = existing_entries
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)

        if new_downloads == 0:
            print("No new documents could be downloaded. Breaking loop.")
            break

    print("=" * 80)


if __name__ == "__main__":
    run_closure_loop()
