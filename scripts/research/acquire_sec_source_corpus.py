#!/usr/bin/env python3
"""
scripts/research/acquire_sec_source_corpus.py

ARX Terminal — ETF Research Pipeline V2
SEC Target Source Corpus Acquisition Engine.

Performs idempotent, rate-compliant, byte-exact acquisition and provenance certification
of the frozen 857-document SEC source corpus defined by Acquisition Manifest V2.
"""

import sys
import os
import time
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Frozen authority paths and expected hashes
ACQUISITION_MANIFEST_V2_PATH = REPO_ROOT / "docs" / "research" / "ETF_V2_ACQUISITION_MANIFEST_V2.json"
EXPECTED_MANIFEST_V2_SHA256 = "2fd9f04d9ca21601a373d465e3253d124596947413e51e699a84f45c7ed017c0"

SNAPSHOT_BOUNDARY_ISO = "2026-09-24T23:59:59Z"

PROSPECTUS_CACHE_DIR = REPO_ROOT / "data" / "research" / "cache" / "sec_prospectus"
SUBMISSIONS_CACHE_DIR = REPO_ROOT / "data" / "research" / "cache" / "sec_submissions"

OUTPUT_PROVENANCE_LEDGER_PATH = REPO_ROOT / "docs" / "research" / "ETF_V2_SEC_SOURCE_ACQUISITION_PROVENANCE_LEDGER.json"
OUTPUT_CORPUS_MANIFEST_PATH = REPO_ROOT / "docs" / "research" / "ETF_V2_SEC_SOURCE_CORPUS_MANIFEST.json"

USER_AGENT = "ArxTerminal/1.0 (research@arxterminal.org)"
SEC_HEADERS = {"User-Agent": USER_AGENT}

# Rate limiting: max 8-9 req/sec to stay safely below SEC 10 req/sec limit
REQUEST_DELAY_SECONDS = 0.12

INTERSTITIAL_ERROR_PATTERNS = [
    b"429 Too Many Requests",
    b"Your Request Couldn't Be Processed",
    b"SEC.gov | Page Not Found",
    b"Access Denied",
    b"Request Rate Exceeded",
    b"<title>Access Denied</title>",
]


def format_local_path(path: Path) -> str:
    """Formats path relative to REPO_ROOT if possible, else standard forward-slash path."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def compute_sha256_bytes(data: bytes) -> str:
    """Computes SHA256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: Path) -> str:
    """Computes SHA256 of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sec_interstitial_or_error(data: bytes) -> Tuple[bool, str]:
    """Validates that response bytes do not represent an SEC error or interstitial page."""
    if len(data) == 0:
        return True, "Empty response body"
    for pat in INTERSTITIAL_ERROR_PATTERNS:
        if pat.lower() in data.lower():
            return True, f"Matched error pattern: {pat.decode('ascii', errors='ignore')}"
    return False, ""


def construct_sec_url(cik: str, accession: str, primary_document: str, expected_location: str = "") -> str:
    """Constructs the canonical SEC EDGAR archive URL."""
    if expected_location and expected_location.startswith("data/"):
        return f"https://www.sec.gov/Archives/edgar/{expected_location}"
    cik_int = str(int(str(cik).lstrip("0") or "0"))
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_document}"


def load_and_deduplicate_manifest(manifest_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses Acquisition Manifest V2 and deduplicates records into unique documents,
    aggregating targets, series IDs, class IDs, and roles.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in records:
        acc = r["accession"]
        pdoc = r["primary_document"]
        loc = r.get("expected_sec_source_location", "")
        raw_cik = str(r.get("cik", "")).strip()
        if not raw_cik and loc.startswith("data/"):
            parts = loc.split("/")
            if len(parts) >= 2:
                raw_cik = parts[1]

        key = (acc, pdoc)

        if key not in dedup:
            dedup[key] = {
                "accession": acc,
                "primary_document": pdoc,
                "form": r.get("form", ""),
                "cik": raw_cik,
                "acceptance_timestamp": r.get("acceptance_timestamp", ""),
                "filing_date": r.get("filing_date", ""),
                "expected_sec_source_location": loc,
                "series_ids": set(),
                "class_ids": set(),
                "target_symbols": set(),
                "authority_chain_roles": set(),
            }

        if r.get("series_id"):
            dedup[key]["series_ids"].add(r["series_id"])
        if r.get("class_id"):
            dedup[key]["class_ids"].add(r["class_id"])
        if r.get("applicable_target"):
            dedup[key]["target_symbols"].add(r["applicable_target"])
        if r.get("authority_chain_role"):
            dedup[key]["authority_chain_roles"].add(r["authority_chain_role"])

    unique_items = []
    # Sort deterministically by accession, primary_document
    for key in sorted(dedup.keys()):
        item = dedup[key]
        unique_items.append({
            "accession": item["accession"],
            "primary_document": item["primary_document"],
            "form": item["form"],
            "cik": item["cik"],
            "acceptance_timestamp": item["acceptance_timestamp"],
            "filing_date": item["filing_date"],
            "expected_sec_source_location": item["expected_sec_source_location"],
            "series_ids": sorted(list(item["series_ids"])),
            "class_ids": sorted(list(item["class_ids"])),
            "target_symbols": sorted(list(item["target_symbols"])),
            "authority_chain_roles": sorted(list(item["authority_chain_roles"])),
        })

    return unique_items, data


def acquire_or_verify_document(
    doc_item: Dict[str, Any],
    cache_dir: Path,
    session: requests.Session,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """
    Idempotently acquires or verifies a single document.
    Returns a complete provenance record.
    """
    acc = doc_item["accession"]
    pdoc = doc_item["primary_document"]
    cik = doc_item["cik"]
    fn = f"{acc}_{pdoc}"
    local_path = cache_dir / fn

    expected_url = construct_sec_url(cik, acc, pdoc, doc_item.get("expected_sec_source_location", ""))

    # 1. Check if valid local file already exists
    if local_path.exists():
        raw_bytes = local_path.read_bytes()
        is_err, err_msg = is_sec_interstitial_or_error(raw_bytes)
        if not is_err:
            raw_sha = compute_sha256_bytes(raw_bytes)
            return {
                "accession": acc,
                "primary_document": pdoc,
                "form": doc_item["form"],
                "cik": cik,
                "series_ids": doc_item["series_ids"],
                "class_ids": doc_item["class_ids"],
                "target_symbols": doc_item["target_symbols"],
                "authority_chain_roles": doc_item["authority_chain_roles"],
                "snapshot_boundary": SNAPSHOT_BOUNDARY_ISO,
                "acceptance_timestamp": doc_item["acceptance_timestamp"],
                "sec_source_url": expected_url,
                "http_status": 200,
                "acquisition_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(local_path.stat().st_mtime)),
                "byte_length": len(raw_bytes),
                "raw_sha256": raw_sha,
                "local_path": format_local_path(local_path),
                "acquisition_method": "REUSED_EXISTING_VERIFIED",
                "retry_count": 0,
                "validation_status": "VALID",
                "error_message": "",
            }

    # 2. Acquire from SEC EDGAR
    retries = 0
    backoffs = [1.0, 2.0, 4.0, 8.0]
    last_err = ""
    http_status = 0
    raw_bytes = b""

    for attempt in range(max_retries):
        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            resp = session.get(expected_url, headers=SEC_HEADERS, timeout=15)
            http_status = resp.status_code
            if resp.status_code == 200:
                content = resp.content
                is_err, err_msg = is_sec_interstitial_or_error(content)
                if not is_err:
                    raw_bytes = content
                    # Persist atomically
                    temp_path = cache_dir / f"{fn}.tmp_{os.getpid()}"
                    temp_path.write_bytes(raw_bytes)
                    temp_path.replace(local_path)

                    raw_sha = compute_sha256_bytes(raw_bytes)
                    # Verify read-back matches
                    read_back_sha = compute_sha256_bytes(local_path.read_bytes())
                    assert raw_sha == read_back_sha, f"Read-back hash mismatch for {fn}"

                    return {
                        "accession": acc,
                        "primary_document": pdoc,
                        "form": doc_item["form"],
                        "cik": cik,
                        "series_ids": doc_item["series_ids"],
                        "class_ids": doc_item["class_ids"],
                        "target_symbols": doc_item["target_symbols"],
                        "authority_chain_roles": doc_item["authority_chain_roles"],
                        "snapshot_boundary": SNAPSHOT_BOUNDARY_ISO,
                        "acceptance_timestamp": doc_item["acceptance_timestamp"],
                        "sec_source_url": expected_url,
                        "http_status": 200,
                        "acquisition_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "byte_length": len(raw_bytes),
                        "raw_sha256": raw_sha,
                        "local_path": format_local_path(local_path),
                        "acquisition_method": "ACQUIRED_NEW",
                        "retry_count": retries,
                        "validation_status": "VALID",
                        "error_message": "",
                    }
                else:
                    last_err = f"SEC content validation failed: {err_msg}"
            elif resp.status_code == 404:
                last_err = "HTTP 404 Not Found"
                break
            elif resp.status_code == 429:
                last_err = "HTTP 429 Rate limit exceeded"
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = f"Exception: {str(e)}"

        retries += 1
        if attempt < len(backoffs):
            time.sleep(backoffs[attempt])

    return {
        "accession": acc,
        "primary_document": pdoc,
        "form": doc_item["form"],
        "cik": cik,
        "series_ids": doc_item["series_ids"],
        "class_ids": doc_item["class_ids"],
        "target_symbols": doc_item["target_symbols"],
        "authority_chain_roles": doc_item["authority_chain_roles"],
        "snapshot_boundary": SNAPSHOT_BOUNDARY_ISO,
        "acceptance_timestamp": doc_item["acceptance_timestamp"],
        "sec_source_url": expected_url,
        "http_status": http_status,
        "acquisition_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "byte_length": len(raw_bytes),
        "raw_sha256": compute_sha256_bytes(raw_bytes) if raw_bytes else "",
        "local_path": format_local_path(local_path),
        "acquisition_method": "FAILED",
        "retry_count": retries,
        "validation_status": "FAILED",
        "error_message": last_err,
    }


def compute_corpus_aggregate_identity(provenance_records: List[Dict[str, Any]]) -> str:
    """
    Computes a deterministic aggregate hash representing the identity of the entire acquired corpus.
    Formula: SHA256 of sorted newline-delimited lines: '{accession}_{primary_document}:{raw_sha256}'.
    """
    lines = []
    for r in sorted(provenance_records, key=lambda x: (x["accession"], x["primary_document"])):
        lines.append(f"{r['accession']}_{r['primary_document']}:{r['raw_sha256']}")
    manifest_bytes = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(manifest_bytes).hexdigest()


def execute_acquisition():
    """Main execution function for the SEC target source acquisition gate."""
    print("=" * 80, flush=True)
    print("ARX TERMINAL — ETF RESEARCH PIPELINE V2", flush=True)
    print("SEC TARGET SOURCE ACQUISITION EXECUTION ENGINE", flush=True)
    print(f"Snapshot Boundary: {SNAPSHOT_BOUNDARY_ISO}", flush=True)
    print("=" * 80, flush=True)

    # 1. Verify Manifest V2 Hash
    actual_manifest_sha = compute_sha256_file(ACQUISITION_MANIFEST_V2_PATH)
    print(f"Acquisition Manifest V2 Path: {ACQUISITION_MANIFEST_V2_PATH}", flush=True)
    print(f"Manifest V2 Actual SHA256:   {actual_manifest_sha}", flush=True)
    print(f"Manifest V2 Expected SHA256: {EXPECTED_MANIFEST_V2_SHA256}", flush=True)
    assert actual_manifest_sha == EXPECTED_MANIFEST_V2_SHA256, "Manifest V2 SHA256 mismatch! Aborting."
    print("Manifest V2 integrity verified: PASS", flush=True)

    # 2. Parse and Deduplicate
    unique_items, raw_manifest = load_and_deduplicate_manifest(ACQUISITION_MANIFEST_V2_PATH)
    print(f"Total target mappings in manifest: {len(raw_manifest.get('records', []))}", flush=True)
    print(f"Unique documents required:         {len(unique_items)}", flush=True)
    assert len(unique_items) == 857, f"Expected 857 unique documents, got {len(unique_items)}"
    print("Unique acquisition universe verified: 857 documents", flush=True)

    PROSPECTUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Execute Acquisition Loop
    session = requests.Session()
    provenance_records = []
    acquired_new = 0
    reused_existing = 0
    failed_count = 0

    print(f"\nBeginning acquisition / verification across {len(unique_items)} unique documents...", flush=True)
    start_time = time.time()

    for idx, item in enumerate(unique_items, 1):
        rec = acquire_or_verify_document(item, PROSPECTUS_CACHE_DIR, session)
        provenance_records.append(rec)

        if rec["acquisition_method"] == "ACQUIRED_NEW":
            acquired_new += 1
        elif rec["acquisition_method"] == "REUSED_EXISTING_VERIFIED":
            reused_existing += 1
        else:
            failed_count += 1
            print(f"  [FAILED] {rec['accession']}_{rec['primary_document']}: {rec['error_message']}", flush=True)

        if idx % 25 == 0 or idx == len(unique_items):
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  Progress: {idx:3d}/{len(unique_items)} | Reused: {reused_existing:3d} | New: {acquired_new:3d} | Failed: {failed_count:2d} ({rate:.1f} docs/sec)", flush=True)

    elapsed_total = time.time() - start_time
    print(f"\nAcquisition pass completed in {elapsed_total:.1f} seconds.", flush=True)
    print(f"ACQUIRED_NEW             = {acquired_new}", flush=True)
    print(f"REUSED_EXISTING_VERIFIED = {reused_existing}", flush=True)
    print(f"FAILED                   = {failed_count}", flush=True)
    print(f"TOTAL_PROCESSED          = {len(provenance_records)}", flush=True)

    assert len(provenance_records) == 857
    assert acquired_new + reused_existing + failed_count == 857

    # 4. Generate Immutable Source-Provenance Ledger
    ledger_doc = {
        "ledger_version": "1.0.0",
        "ledger_authority": "SEC_TARGET_SOURCE_ACQUISITION_EXECUTION_GATE",
        "source_manifest_path": str(ACQUISITION_MANIFEST_V2_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_manifest_sha256": actual_manifest_sha,
        "snapshot_boundary": SNAPSHOT_BOUNDARY_ISO,
        "expected_unique_documents": 857,
        "successful_unique_documents": acquired_new + reused_existing,
        "failed_unique_documents": failed_count,
        "acquired_new": acquired_new,
        "reused_existing_verified": reused_existing,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance_records": provenance_records,
    }

    with open(OUTPUT_PROVENANCE_LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger_doc, f, indent=2)
    ledger_sha = compute_sha256_file(OUTPUT_PROVENANCE_LEDGER_PATH)
    print(f"Wrote Provenance Ledger: {OUTPUT_PROVENANCE_LEDGER_PATH}")
    print(f"PROVENANCE_LEDGER_SHA256 = {ledger_sha}")

    # 5. Generate Corpus Manifest
    corpus_aggregate_id = compute_corpus_aggregate_identity(provenance_records)
    corpus_manifest_records = []
    for r in sorted(provenance_records, key=lambda x: (x["accession"], x["primary_document"])):
        corpus_manifest_records.append({
            "accession": r["accession"],
            "primary_document": r["primary_document"],
            "form": r["form"],
            "cik": r["cik"],
            "series_ids": r["series_ids"],
            "class_ids": r["class_ids"],
            "target_symbols": r["target_symbols"],
            "authority_chain_roles": r["authority_chain_roles"],
            "acceptance_timestamp": r["acceptance_timestamp"],
            "byte_length": r["byte_length"],
            "raw_sha256": r["raw_sha256"],
            "local_path": r["local_path"],
            "validation_status": r["validation_status"],
        })

    corpus_manifest_doc = {
        "manifest_version": "1.0.0",
        "manifest_authority": "SEC_TARGET_SOURCE_ACQUISITION_EXECUTION_GATE",
        "snapshot_boundary": SNAPSHOT_BOUNDARY_ISO,
        "corpus_document_count": len(corpus_manifest_records),
        "corpus_aggregate_identity_algorithm": "SHA256(join_newline('{accession}_{primary_document}:{raw_sha256}'))",
        "corpus_aggregate_identity": corpus_aggregate_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": corpus_manifest_records,
    }

    with open(OUTPUT_CORPUS_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus_manifest_doc, f, indent=2)
    corpus_manifest_sha = compute_sha256_file(OUTPUT_CORPUS_MANIFEST_PATH)
    print(f"Wrote Corpus Manifest:   {OUTPUT_CORPUS_MANIFEST_PATH}")
    print(f"CORPUS_MANIFEST_SHA256   = {corpus_manifest_sha}")
    print(f"CORPUS_AGGREGATE_IDENTITY = {corpus_aggregate_id}")

    return {
        "acquired_new": acquired_new,
        "reused_existing": reused_existing,
        "failed": failed_count,
        "provenance_ledger_sha": ledger_sha,
        "corpus_manifest_sha": corpus_manifest_sha,
        "corpus_aggregate_id": corpus_aggregate_id,
    }


if __name__ == "__main__":
    execute_acquisition()
