"""ARX Terminal — Bounded Residual 2-Target Source Cache Acquisition.

Acquires exactly the 2 residual pre-boundary Form 497K Summary Prospectuses
for FDLO and SEPQ identified during V1.3.0 reconciliation.

Strict constraints:
- User-Agent compliant with SEC EDGAR policy
- Rate-limited (sleep >= 0.12s)
- Binary write mode (zero newline conversion)
- Cryptographic provenance recorded in SOURCE_PROVENANCE_LEDGER.json
  and V1_3_0_ACQUISITION_PROVENANCE_LEDGER.json
- Strictly pre-boundary (2017-01-13 and 2023-07-31 <= 2026-09-24)
"""

import sys
import json
import time
import hashlib
import requests
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROSPECTUS_DIR = REPO_ROOT / "data" / "research" / "cache" / "sec_prospectus"
SOURCE_PROV_PATH = REPO_ROOT / "docs" / "research" / "SOURCE_PROVENANCE_LEDGER.json"
V1_3_0_PROV_PATH = REPO_ROOT / "docs" / "research" / "V1_3_0_ACQUISITION_PROVENANCE_LEDGER.json"

HEADERS = {"User-Agent": "ArxTerminal/1.0 (research@arxterminal.org)"}
SNAPSHOT_BOUNDARY = "2026-09-24T23:59:59Z"

TARGETS_TO_ACQUIRE = [
    {
        "symbol": "FDLO",
        "cik": "0000945908",
        "cik_int": 945908,
        "series_id": "S000054751",
        "class_id": "C000171932",
        "accession": "0001193125-17-009627",
        "form": "497K",
        "filing_date": "2017-01-13",
        "document_filename": "d287805d497k.htm",
        "url": "https://www.sec.gov/Archives/edgar/data/945908/000119312517009627/d287805d497k.htm"
    },
    {
        "symbol": "SEPQ",
        "cik": "0001683471",
        "cik_int": 1683471,
        "series_id": "S000076366",
        "class_id": "C000236165",
        "accession": "0000894189-23-005235",
        "form": "497K",
        "filing_date": "2023-07-31",
        "document_filename": "stftacticalgrowthtugsummar.htm",
        "url": "https://www.sec.gov/Archives/edgar/data/1683471/000089418923005235/stftacticalgrowthtugsummar.htm"
    }
]


def acquire_sources():
    print("=" * 80)
    print("ARX TERMINAL — RESIDUAL 2-TARGET SOURCE CACHE ACQUISITION")
    print("=" * 80)

    requested_count = len(TARGETS_TO_ACQUIRE)
    acquired_count = 0
    failure_count = 0
    duplicate_count = 0

    acquired_records = []

    for t in TARGETS_TO_ACQUIRE:
        sym = t["symbol"]
        acc = t["accession"]
        doc = t["document_filename"]
        local_fname = f"{acc}_{doc}"
        local_path = PROSPECTUS_DIR / local_fname

        if local_path.exists():
            print(f"[{sym}] Local file already exists: {local_fname}")
            duplicate_count += 1
            content = local_path.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            acquired_records.append((t, local_fname, local_path, len(content), sha, sha))
            continue

        print(f"[{sym}] Downloading from {t['url']}...")
        time.sleep(0.15)  # SEC rate-limit compliance
        resp = requests.get(t["url"], headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"[{sym}] ERROR: HTTP {resp.status_code}")
            failure_count += 1
            continue

        raw_bytes = resp.content
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()

        # Binary write mode prevents newline modification
        with open(local_path, "wb") as f:
            f.write(raw_bytes)

        disk_bytes = local_path.read_bytes()
        disk_sha = hashlib.sha256(disk_bytes).hexdigest()
        assert disk_sha == raw_sha, f"Local write corruption: {disk_sha} != {raw_sha}"

        print(f"[{sym}] Successfully saved: {local_fname} ({len(raw_bytes)} bytes, SHA256: {raw_sha})")
        acquired_count += 1
        acquired_records.append((t, local_fname, local_path, len(raw_bytes), raw_sha, disk_sha))

    print(f"\nREQUESTED_DOCUMENTS = {requested_count}")
    print(f"ACQUIRED_DOCUMENTS = {acquired_count}")
    print(f"ACQUISITION_FAILURES = {failure_count}")
    print(f"DUPLICATE_DOWNLOADS = {duplicate_count}")

    assert failure_count == 0, f"Acquisition failures encountered: {failure_count}"
    assert acquired_count == 2, f"Expected 2 acquired documents, got {acquired_count}"

    # Update SOURCE_PROVENANCE_LEDGER.json
    with open(SOURCE_PROV_PATH, "r", encoding="utf-8") as f:
        source_prov = json.load(f)

    files_dict = source_prov.get("files", {})
    ts_now = datetime.now(timezone.utc).isoformat()

    for t, local_fname, local_path, byte_len, raw_sha, disk_sha in acquired_records:
        files_dict[local_fname] = {
            "cik": t["cik"].lstrip("0"),
            "accession": t["accession"],
            "form": t["form"],
            "filing_date": t["filing_date"],
            "document_filename": t["document_filename"],
            "source_url": t["url"],
            "download_timestamp": ts_now,
            "byte_length": byte_len,
            "sha256": raw_sha,
            "snapshot_eligibility": "ELIGIBLE_PRE_BOUNDARY",
            "local_path": str(local_path.relative_to(REPO_ROOT))
        }

    source_prov["total_files"] = len(files_dict)
    source_prov["total_bytes"] = sum(e["byte_length"] for e in files_dict.values())
    source_prov["last_updated"] = ts_now

    with open(SOURCE_PROV_PATH, "w", encoding="utf-8") as f:
        json.dump(source_prov, f, indent=2)
    print(f"Updated {SOURCE_PROV_PATH} (total_files: {source_prov['total_files']}, total_bytes: {source_prov['total_bytes']})")

    # Update V1_3_0_ACQUISITION_PROVENANCE_LEDGER.json
    if V1_3_0_PROV_PATH.exists():
        with open(V1_3_0_PROV_PATH, "r", encoding="utf-8") as f:
            v1_3_prov = json.load(f)
    else:
        v1_3_prov = {
            "ledger_version": "V1_3_0_ACQUISITION_PROVENANCE_LEDGER",
            "generated_at": ts_now,
            "snapshot_boundary": SNAPSHOT_BOUNDARY,
            "entries": []
        }

    existing_keys = {f"{e['accession']}_{e['document_filename']}" for e in v1_3_prov.get("entries", [])}
    for t, local_fname, local_path, byte_len, raw_sha, disk_sha in acquired_records:
        if local_fname not in existing_keys:
            v1_3_prov["entries"].append({
                "symbol": t["symbol"],
                "series_id": t["series_id"],
                "cik": t["cik"],
                "accession": t["accession"],
                "document_filename": t["document_filename"],
                "form": t["form"],
                "filing_date": t["filing_date"],
                "source_sha256": raw_sha,
                "status": "ACQUIRED_SUCCESS"
            })

    v1_3_prov["newly_acquired_documents"] = len(v1_3_prov["entries"])
    with open(V1_3_0_PROV_PATH, "w", encoding="utf-8") as f:
        json.dump(v1_3_prov, f, indent=2)
    print(f"Updated {V1_3_0_PROV_PATH} (entries: {len(v1_3_prov['entries'])})")


if __name__ == "__main__":
    acquire_sources()
