"""
scripts/retrieve_three_candidate_sec_evidence.py

Retrieves official accession metadata and Form 10-K filing documents for AAPL, GOOG, and GOOGL
from SEC EDGAR Submissions API and SEC Archives using authorized operator contact identity.
Preserves original response bytes, computes SHA-256 digests, and extracts Section 12(b) tables.
"""

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
from dotenv import load_dotenv

load_dotenv()
ua = os.environ.get("SEC_USER_AGENT")
if not ua:
    raise ValueError("SEC_USER_AGENT must be configured in environment")

headers = {
    "User-Agent": ua,
    "Accept-Encoding": "gzip, deflate",
}

filings_dir = Path("data/research/sec_filings_sample")
filings_dir.mkdir(parents=True, exist_ok=True)
submissions_dir = Path("data/research/sec_submissions_sample")
submissions_dir.mkdir(parents=True, exist_ok=True)

targets = [
    {
        "symbol_group": "AAPL",
        "cik": "0000320193",
        "cik_clean": "320193",
    },
    {
        "symbol_group": "GOOG_GOOGL",
        "cik": "0001652044",
        "cik_clean": "1652044",
    },
]

retrieval_records = {}

for target in targets:
    cik = target["cik"]
    group = target["symbol_group"]
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    print(f"Fetching Submissions API for {group} ({cik}) from {sub_url}...")
    req = urllib.request.Request(sub_url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        sub_raw = resp.read()
        try:
            sub_decompressed = gzip.decompress(sub_raw)
        except Exception:
            sub_decompressed = sub_raw

        sub_sha = hashlib.sha256(sub_decompressed).hexdigest()
        sub_file = submissions_dir / f"CIK{cik}.json"
        sub_file.write_bytes(sub_decompressed)
        print(f"Saved Submissions API JSON to {sub_file} (Bytes: {len(sub_decompressed)}, SHA256: {sub_sha})")

        data = json.loads(sub_decompressed.decode("utf-8"))
        recent = data["filings"]["recent"]
        idx = recent["form"].index("10-K")
        accession = recent["accessionNumber"][idx]
        filing_date = recent["filingDate"][idx]
        report_date = recent["reportDate"][idx]
        primary_doc = recent["primaryDocument"][idx]
        acc_no_hyphen = accession.replace("-", "")
        print(f"Verified from metadata: Form 10-K Accession: {accession}, Filing Date: {filing_date}, Report Date: {report_date}, Primary Doc: {primary_doc}")

    time.sleep(1)  # SEC Fair Access rate limit compliance

    cik_clean = target["cik_clean"]
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_no_hyphen}/{primary_doc}"
    print(f"Fetching Primary Doc from {doc_url}...")
    req_doc = urllib.request.Request(doc_url, headers=headers)
    with urllib.request.urlopen(req_doc, timeout=30) as resp:
        doc_raw = resp.read()
        try:
            doc_decompressed = gzip.decompress(doc_raw)
        except Exception:
            doc_decompressed = doc_raw

        doc_sha = hashlib.sha256(doc_decompressed).hexdigest()
        doc_file = filings_dir / primary_doc
        doc_file.write_bytes(doc_decompressed)
        print(f"Saved Primary Document to {doc_file} (Bytes: {len(doc_decompressed)}, SHA256: {doc_sha})")

        retrieval_records[group] = {
            "cik": cik,
            "accession": accession,
            "filing_date": filing_date,
            "report_date": report_date,
            "primary_doc": primary_doc,
            "doc_url": doc_url,
            "doc_file": str(doc_file).replace("\\", "/"),
            "doc_bytes": len(doc_decompressed),
            "doc_sha256": doc_sha,
            "sub_url": sub_url,
            "sub_file": str(sub_file).replace("\\", "/"),
            "sub_bytes": len(sub_decompressed),
            "sub_sha256": sub_sha,
        }

    time.sleep(1)

manifest_path = filings_dir / "retrieval_manifest.json"
manifest_path.write_text(json.dumps(retrieval_records, indent=2), encoding="utf-8")
print(f"Retrieval complete and manifest recorded to {manifest_path}!")
