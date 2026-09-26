"""ARX Terminal — Resumable Statutory Source Acquisition Engine.

Closes the population-scale statutory source coverage gap for ETF mandate research.
Enforces:
1. Strict pre-boundary eligibility (filingDate <= 2026-09-24).
2. Rate-limiting <= 10 req/sec (0.12s inter-request delay).
3. Resumable checkpointing with SHA256 verification (zero duplicate downloads on resume).
4. Provenance tracking for every acquired file.
5. Loop iteration until SOURCE_CACHE_MISS == 0 across the entire 2,884 target population.
"""

import sys
import os
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Set, Tuple
import requests

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.statutory_filing_selector import (
    StatutoryFilingSelector,
    OUTCOME_SELECTED_STATUTORY_PROSPECTUS,
    OUTCOME_SELECTED_SUMMARY_PROSPECTUS,
    OUTCOME_NO_PREBOUNDARY_CANDIDATE,
    OUTCOME_TARGET_ABSENT_FROM_ALL,
    OUTCOME_SOURCE_CACHE_MISS,
    OUTCOME_AMBIGUOUS_MAPPING,
    OUTCOME_CONFLICTING_DOCUMENTS,
)
from scripts.research.series_prospectus_mapper import SeriesMetadata

INPUT_MANIFEST_PATH = Path("docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json")
EXPECTED_MANIFEST_SHA = "764363abedf51dd40365cf26d17d429fe4596619bd7e8e648cca17502286635a"
CACHE_DIR = Path("data/research/cache")
SUBMISSIONS_DIR = CACHE_DIR / "sec_submissions"
PROSPECTUS_DIR = CACHE_DIR / "sec_prospectus"
PROVENANCE_LEDGER_PATH = Path("docs/research/SOURCE_PROVENANCE_LEDGER.json")

SNAPSHOT_BOUNDARY = "2026-09-24"

HEADERS = {
    "User-Agent": "ArxTerminal/1.0 (research@arxterminal.org)",
    "Accept-Encoding": "gzip, deflate",
}


def load_provenance_ledger() -> Dict[str, Dict[str, Any]]:
    """Load existing provenance records indexed by filename."""
    if not PROVENANCE_LEDGER_PATH.exists():
        return {}
    try:
        with open(PROVENANCE_LEDGER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("files", {})
    except Exception:
        return {}


def save_provenance_ledger(ledger: Dict[str, Dict[str, Any]]):
    """Save provenance records to ledger."""
    PROVENANCE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "ledger_version": "SOURCE_PROVENANCE_V1_0_0",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_files": len(ledger),
        "total_bytes": sum(f.get("byte_length", 0) for f in ledger.values()),
        "files": ledger,
    }
    with open(PROVENANCE_LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def download_single_filing(
    cik: str,
    accession: str,
    doc_filename: str,
    form: str,
    filing_date: str,
    ledger: Dict[str, Dict[str, Any]]
) -> Tuple[bool, str, int, str]:
    """Download a single pre-boundary filing with rate limiting, retries, and ledger check.
    
    Returns: (success, sha256, byte_len, status_msg)
    """
    PROSPECTUS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{accession}_{doc_filename}"
    local_path = PROSPECTUS_DIR / filename

    # Strict snapshot boundary verification (Section 7)
    if filing_date > SNAPSHOT_BOUNDARY:
        raise ValueError(f"CRITICAL: Attempted download of post-boundary filing: {filing_date} > {SNAPSHOT_BOUNDARY}")

    # Check if already cached and verified in ledger (Section 21: Checkpoint / Resume)
    if local_path.exists() and filename in ledger:
        raw_bytes = local_path.read_bytes()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        if sha == ledger[filename].get("sha256"):
            return True, sha, len(raw_bytes), "ALREADY_CACHED_AND_VERIFIED"

    acc_nodash = accession.replace("-", "")
    cik_int = int(cik)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc_filename}"

    max_retries = 4
    for attempt in range(max_retries):
        time.sleep(0.12)  # Respect SEC rate limit (< 10 req/s)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                content_bytes = resp.content
                if not content_bytes:
                    return False, "", 0, "EMPTY_RESPONSE"

                sha = hashlib.sha256(content_bytes).hexdigest()
                byte_len = len(content_bytes)

                # Write to disk
                with open(local_path, "wb") as f:
                    f.write(content_bytes)

                # Record in ledger (Section 22: Source Provenance Completeness)
                ledger[filename] = {
                    "cik": cik,
                    "accession": accession,
                    "form": form,
                    "filing_date": filing_date,
                    "document_filename": doc_filename,
                    "source_url": url,
                    "download_timestamp": datetime.now(timezone.utc).isoformat(),
                    "byte_length": byte_len,
                    "sha256": sha,
                    "snapshot_eligibility": "ELIGIBLE_PRE_BOUNDARY",
                    "local_path": str(local_path),
                }
                return True, sha, byte_len, "DOWNLOAD_SUCCESS"
            elif resp.status_code == 429:
                wait_sec = 2.0 * (attempt + 1)
                print(f"[429 Rate Limited] Backing off {wait_sec}s...")
                time.sleep(wait_sec)
            elif resp.status_code == 404:
                return False, "", 0, f"HTTP_404_NOT_FOUND"
            else:
                if attempt == max_retries - 1:
                    return False, "", 0, f"HTTP_{resp.status_code}"
                time.sleep(1.0)
        except Exception as e:
            if attempt == max_retries - 1:
                return False, "", 0, f"EXCEPTION: {str(e)}"
            time.sleep(1.0)

    return False, "", 0, "MAX_RETRIES_EXCEEDED"


def run_acquisition():
    print("=" * 80)
    print("ARX TERMINAL — STATUTORY SOURCE ACQUISITION ENGINE (SECTION 7, 8, 21, 22)")
    print("=" * 80)

    # 1. Verify manifest
    assert INPUT_MANIFEST_PATH.exists()
    raw_manifest = INPUT_MANIFEST_PATH.read_bytes()
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    assert manifest_sha == EXPECTED_MANIFEST_SHA
    records = json.loads(raw_manifest.decode("utf-8")).get("records", [])
    total_targets = len(records)
    print(f"Verified input manifest SHA256: {manifest_sha} ({total_targets} targets)")

    # 2. Load submission metadata
    submission_cache: Dict[str, dict] = {}
    for p in SUBMISSIONS_DIR.glob("CIK*.json"):
        cik_str = p.stem.replace("CIK", "").lstrip("0") or "0"
        try:
            with open(p, "r", encoding="utf-8") as f:
                submission_cache[cik_str] = json.load(f)
        except Exception:
            pass
    print(f"Loaded {len(submission_cache)} CIK submission files.")

    # 3. Load provenance ledger
    ledger = load_provenance_ledger()
    print(f"Loaded existing provenance ledger: {len(ledger)} files tracked.")

    # 4. Acquisition loop: iterate until SOURCE_CACHE_MISS == 0
    total_new_files = 0
    total_new_bytes = 0
    total_failures = 0
    iteration = 0

    while True:
        iteration += 1
        print(f"\n--- Acquisition Iteration {iteration} ---")
        cached_filenames = {p.name for p in PROSPECTUS_DIR.iterdir()} if PROSPECTUS_DIR.exists() else set()
        print(f"Current local cache files: {len(cached_filenames)}")

        # Evaluate selector across all targets to identify needed missing sources
        miss_files_to_acquire: Dict[str, Dict[str, str]] = {}
        outcome_counter = Counter()

        for r in records:
            sym = r.get("symbol", "")
            cik = str(r.get("cik", "")).lstrip("0") or "0"
            sid = r.get("series_id", "")
            cid = r.get("class_id", "")
            name = r.get("legal_name", "")

            target = SeriesMetadata(symbol=sym, cik=cik, series_id=sid, class_id=cid, legal_name=name)
            sub_json = submission_cache.get(cik, {})
            sel_res = StatutoryFilingSelector.select_statutory_filing(
                target, sub_json, CACHE_DIR, cached_filenames=cached_filenames
            )
            outcome_counter[sel_res.selection_outcome] += 1

            if sel_res.selection_outcome == OUTCOME_SOURCE_CACHE_MISS:
                acc = sel_res.selected_accession
                doc = sel_res.document_filename
                form = sel_res.selected_form
                fdate = sel_res.filing_date
                fname = f"{acc}_{doc}"
                if fname not in cached_filenames and fname not in miss_files_to_acquire:
                    miss_files_to_acquire[fname] = {
                        "cik": cik,
                        "accession": acc,
                        "document_filename": doc,
                        "form": form,
                        "filing_date": fdate,
                    }

        print(f"Current Selector Outcomes: {dict(outcome_counter)}")
        print(f"Remaining SOURCE_CACHE_MISS targets: {outcome_counter[OUTCOME_SOURCE_CACHE_MISS]}")
        print(f"Unique missing files to acquire this round: {len(miss_files_to_acquire)}")

        if outcome_counter[OUTCOME_SOURCE_CACHE_MISS] == 0 or len(miss_files_to_acquire) == 0:
            print("No remaining source cache misses! Acquisition complete.")
            break

        # Acquire files for this iteration
        round_acquired = 0
        round_bytes = 0
        round_failures = 0

        items = list(miss_files_to_acquire.values())
        for idx, item in enumerate(items):
            if (idx + 1) % 50 == 0 or idx == 0 or idx == len(items) - 1:
                print(f"Acquiring [{idx+1}/{len(items)}] {item['accession']}_{item['document_filename']} (CIK {item['cik']}, {item['form']})...")

            success, sha, b_len, msg = download_single_filing(
                cik=item["cik"],
                accession=item["accession"],
                doc_filename=item["document_filename"],
                form=item["form"],
                filing_date=item["filing_date"],
                ledger=ledger,
            )

            if success:
                if msg == "DOWNLOAD_SUCCESS":
                    round_acquired += 1
                    round_bytes += b_len
            else:
                round_failures += 1
                print(f"FAILED to download {item['accession']}_{item['document_filename']}: {msg}")

            # Periodic checkpoint save every 25 files
            if (idx + 1) % 25 == 0:
                save_provenance_ledger(ledger)

        save_provenance_ledger(ledger)
        total_new_files += round_acquired
        total_new_bytes += round_bytes
        total_failures += round_failures

        print(f"Round {iteration} finished: {round_acquired} newly acquired ({round_bytes:,} bytes), {round_failures} failures.")

        if round_acquired == 0 and round_failures > 0:
            print("Stopping due to repeated failures without progress.")
            break

    print("\n" + "=" * 80)
    print("ACQUISITION RUN COMPLETE")
    print(f"NEW_SOURCE_FILES_ACQUIRED = {total_new_files}")
    print(f"NEW_SOURCE_BYTES = {total_new_bytes:,}")
    print(f"ACQUISITION_FAILURES = {total_failures}")
    print(f"TOTAL_FILES_IN_LEDGER = {len(ledger)}")
    print(f"LEDGER_PATH = {PROVENANCE_LEDGER_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    run_acquisition()
