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


# Execution Modes and Version Constants
EXECUTION_MODE_INITIAL_FULL = "INITIAL_FULL_CORPUS_ACQUISITION"
EXECUTION_MODE_INCREMENTAL = "INCREMENTAL_AUTHORIZED_SOURCE_ACQUISITION"
CORPUS_AGGREGATE_IDENTITY_VERSION = "1.0.0-append-ordered"

CANONICAL_857_AGGREGATE_SHA256 = "525155e195eb285624433bdd472d7a97c4a68ae724e7325dc88f5bc31519531c"
CANONICAL_858_AGGREGATE_SHA256 = "7a78fd45a04250b0a7e3c4745b2f3d93e7907ae0ab6fc00c1042bddadbecd612"

HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
CIK_DIGITS_RE = re.compile(r"^\d{1,10}$")
DATE_YYYY_MM_DD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PRIMARY_DOC_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(?:htm|html|txt|xml)$", re.IGNORECASE)


class DestructiveCorpusOverwriteError(RuntimeError):
    """Raised when INITIAL_FULL_CORPUS_ACQUISITION is invoked against an 858+ canonical corpus."""


class InvalidAuthorizedSourceSpecError(ValueError):
    """Raised when an AuthorizedSourceSpec is malformed, incomplete, path-unsafe, or post-boundary."""


class CanonicalHashConflictError(RuntimeError):
    """Raised fail-closed when an existing canonical record or cache file has a conflicting SHA-256."""


class SECPayloadValidationError(RuntimeError):
    """Raised when an acquired or cached SEC payload fails byte, interstitial, or SHA-256 verification."""


class LegacyProvenanceReconciliationRequiredError(RuntimeError):
    """Raised when the existing corpus manifest and provenance ledger are out of sync before incremental acquisition."""

    def __init__(self, message: str, deficit_info: Dict[str, Any]):
        super().__init__(message)
        self.status = "LEGACY_PROVENANCE_RECONCILIATION_REQUIRED"
        self.deficit_info = deficit_info


class InvalidProvenanceReconciliationError(ValueError):
    """Raised when provenance reconciliation inputs omit required fields or mismatch canonical corpus state."""


def compute_corpus_aggregate_identity(provenance_records: List[Dict[str, Any]]) -> str:
    """
    Computes a deterministic aggregate hash representing the identity of the acquired corpus
    in canonical append-ordered sequence order:
    Formula: SHA256(UTF8("\\n".join(f"{r['accession']}_{r['primary_document']}:{r['raw_sha256']}" for r in records))).
    Do not globally resort an existing canonical manifest after extension.
    """
    lines = [
        f"{r['accession']}_{r['primary_document']}:{r['raw_sha256']}"
        for r in provenance_records
    ]
    manifest_bytes = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(manifest_bytes).hexdigest()


def verify_manifest_aggregate_compatibility(manifest_doc: Dict[str, Any]) -> bool:
    """
    Verifies that a manifest's stored corpus_aggregate_identity matches its records
    under append-ordered canonical semantics, supporting both versioned manifests
    and legacy manifests without corpus_aggregate_identity_version.
    """
    records = manifest_doc.get("records", [])
    stored_agg = manifest_doc.get("corpus_aggregate_identity", "")
    version = manifest_doc.get("corpus_aggregate_identity_version")
    if version is not None and version != CORPUS_AGGREGATE_IDENTITY_VERSION:
        return False
    recomputed = compute_corpus_aggregate_identity(records)
    return bool(stored_agg and recomputed == stored_agg)


def _validate_single_filename(primary_document: str) -> str:
    """
    Validates that primary_document is strictly a single safe SEC document filename
    and rejects any path traversal, separators, drive letters, UNC paths, schemes, or percent encoding.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    if not isinstance(primary_document, str):
        raise InvalidAuthorizedSourceSpecError("primary_document must be a string.")
    raw = primary_document.strip()
    if not raw or raw != primary_document:
        # Reject empty or leading/trailing whitespace smuggling
        if not raw:
            raise InvalidAuthorizedSourceSpecError("primary_document must not be empty.")

    forbidden_substrings = ("/", "\\", "..", ":", "%", "?", "#", "\x00")
    for bad in forbidden_substrings:
        if bad in raw:
            raise InvalidAuthorizedSourceSpecError(
                f"Path separator, traversal, or unsafe token {bad!r} is prohibited in primary_document: {primary_document!r}"
            )

    win_p = PureWindowsPath(raw)
    posix_p = PurePosixPath(raw)
    if (
        win_p.is_absolute()
        or posix_p.is_absolute()
        or bool(win_p.drive)
        or bool(win_p.root)
        or len(win_p.parts) != 1
        or len(posix_p.parts) != 1
        or win_p.name != raw
        or posix_p.name != raw
    ):
        raise InvalidAuthorizedSourceSpecError(
            f"primary_document must be a single filename without directory components: {primary_document!r}"
        )

    if not PRIMARY_DOC_FILENAME_RE.match(raw):
        raise InvalidAuthorizedSourceSpecError(
            f"primary_document {primary_document!r} does not match canonical SEC filename syntax."
        )

    return raw


def resolve_contained_cache_path(cache_dir: Path, accession: str, primary_document: str) -> Path:
    """
    Constructs the target cache file path for (accession, primary_document) and enforces
    defense-in-depth containment inside cache_dir.
    """
    if not ACCESSION_RE.match(accession):
        raise InvalidAuthorizedSourceSpecError(
            f"Invalid accession format {accession!r}; expected ^\\d{{10}}-\\d{{2}}-\\d{{6}}$."
        )
    safe_pdoc = _validate_single_filename(primary_document)
    fn = f"{accession}_{safe_pdoc}"
    resolved_cache_dir = cache_dir.resolve()
    candidate_path = cache_dir / fn
    resolved_target = candidate_path.resolve()

    if (
        resolved_target.parent != resolved_cache_dir
        or not resolved_target.is_relative_to(resolved_cache_dir)
        or resolved_target.name != fn
    ):
        raise InvalidAuthorizedSourceSpecError(
            f"Cache path containment violation: {resolved_target} escapes {resolved_cache_dir}"
        )
    return candidate_path


def validate_authorized_source_spec(
    spec: Dict[str, Any],
    cache_dir: Path = PROSPECTUS_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Validates an incremental AuthorizedSourceSpec before any network or filesystem mutation.
    """
    if not isinstance(spec, dict):
        raise InvalidAuthorizedSourceSpecError("AuthorizedSourceSpec must be a dictionary.")

    required_strings = [
        "accession",
        "cik",
        "form",
        "primary_document",
        "expected_raw_sha256",
        "filing_date",
        "acceptance_timestamp",
    ]
    for field in required_strings:
        val = spec.get(field)
        if not isinstance(val, str) or not val.strip():
            raise InvalidAuthorizedSourceSpecError(f"Required non-empty string field '{field}' is missing or empty.")

    acc = spec["accession"].strip()
    if not ACCESSION_RE.match(acc):
        raise InvalidAuthorizedSourceSpecError(
            f"Field 'accession' must match ^\\d{{10}}-\\d{{2}}-\\d{{6}}$: {spec['accession']!r}"
        )

    cik = spec["cik"].strip()
    if not CIK_DIGITS_RE.match(cik) or int(cik) == 0:
        raise InvalidAuthorizedSourceSpecError(
            f"Field 'cik' must be a 1-to-10 digit non-zero SEC CIK: {spec['cik']!r}"
        )

    pdoc = _validate_single_filename(spec["primary_document"])
    # Enforce defense-in-depth cache path containment during validation
    resolve_contained_cache_path(cache_dir, acc, pdoc)

    filing_date = spec["filing_date"].strip()
    if not DATE_YYYY_MM_DD_RE.match(filing_date):
        raise InvalidAuthorizedSourceSpecError(
            f"Field 'filing_date' must match YYYY-MM-DD: {spec['filing_date']!r}"
        )

    expected_sha = spec["expected_raw_sha256"].strip().lower()
    if not HEX64_RE.match(expected_sha):
        raise InvalidAuthorizedSourceSpecError(
            f"Field 'expected_raw_sha256' must be a valid 64-character hex SHA-256: {spec['expected_raw_sha256']!r}"
        )

    if spec["acceptance_timestamp"].strip() > SNAPSHOT_BOUNDARY_ISO:
        raise InvalidAuthorizedSourceSpecError(
            f"Post-boundary acceptance_timestamp {spec['acceptance_timestamp']!r} > {SNAPSHOT_BOUNDARY_ISO!r} rejected."
        )

    if spec.get("expected_sec_source_location"):
        raise InvalidAuthorizedSourceSpecError(
            "Field 'expected_sec_source_location' is prohibited in incremental specs; URL must be constructed from validated (cik, accession, primary_document)."
        )

    required_lists = [
        "series_ids",
        "class_ids",
        "target_symbols",
        "authority_chain_roles",
    ]
    for lfield in required_lists:
        lval = spec.get(lfield)
        if not isinstance(lval, list) or len(lval) == 0 or not all(isinstance(x, str) and x.strip() for x in lval):
            raise InvalidAuthorizedSourceSpecError(
                f"Required non-empty list field '{lfield}' is missing or contains empty elements."
            )

    normalized = {
        "accession": acc,
        "cik": cik,
        "form": spec["form"].strip(),
        "primary_document": pdoc,
        "expected_raw_sha256": expected_sha,
        "filing_date": filing_date,
        "acceptance_timestamp": spec["acceptance_timestamp"].strip(),
        "series_ids": [s.strip() for s in spec["series_ids"]],
        "class_ids": [c.strip() for c in spec["class_ids"]],
        "target_symbols": [t.strip() for t in spec["target_symbols"]],
        "authority_chain_roles": [r.strip() for r in spec["authority_chain_roles"]],
        "expected_sec_source_location": "",
    }
    for opt_field in ("predecessor_symbol", "effective_date", "legal_name", "document_role"):
        if spec.get(opt_field) is not None:
            normalized[opt_field] = str(spec[opt_field]).strip()
    return normalized


def detect_legacy_provenance_deficit(
    corpus_manifest: Dict[str, Any],
    provenance_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inspects corpus_manifest and provenance_ledger for membership or hash asymmetry.
    Never modifies either structure.
    """
    corpus_records = corpus_manifest.get("records", [])
    prov_records = provenance_ledger.get("provenance_records", [])

    corpus_map = {(r["accession"], r["primary_document"]): r["raw_sha256"] for r in corpus_records}
    prov_map = {(r["accession"], r["primary_document"]): r["raw_sha256"] for r in prov_records}

    missing_in_prov = [k for k in corpus_map if k not in prov_map]
    extra_in_prov = [k for k in prov_map if k not in corpus_map]
    hash_mismatches = [
        k for k in corpus_map if k in prov_map and corpus_map[k] != prov_map[k]
    ]

    deficit_detected = bool(
        len(corpus_records) != len(prov_records)
        or missing_in_prov
        or extra_in_prov
        or hash_mismatches
    )
    return {
        "deficit_detected": deficit_detected,
        "corpus_count": len(corpus_records),
        "provenance_count": len(prov_records),
        "missing_in_provenance": missing_in_prov,
        "extra_in_provenance": extra_in_prov,
        "hash_mismatches": hash_mismatches,
    }


# Note: retry_count is required for new live acquisitions (produced by runtime),
# but NOT required for historical provenance reconciliation when not recorded in historical evidence.
REQUIRED_PROVENANCE_RECORD_FIELDS = (
    "accession",
    "primary_document",
    "form",
    "cik",
    "series_ids",
    "class_ids",
    "target_symbols",
    "authority_chain_roles",
    "snapshot_boundary",
    "acceptance_timestamp",
    "sec_source_url",
    "http_status",
    "acquisition_timestamp",
    "byte_length",
    "raw_sha256",
    "local_path",
    "acquisition_method",
    "validation_status",
    "error_message",
)


def reconcile_missing_provenance(
    corpus_manifest_path: Path,
    provenance_ledger_path: Path,
    authorized_reconciliation_records: List[Dict[str, Any]],
    cache_dir: Path = PROSPECTUS_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Explicitly reconciles missing provenance records for sources already present in the canonical corpus.
    - Requires explicit authorization and complete provenance metadata (never fabricates unknown fields,
      and does not require fabricating retry_count when unestablished in historical evidence).
    - Never mutates corpus_manifest_path, source bytes, or corpus_aggregate_identity.
    - Independently idempotent.
    """
    if not isinstance(authorized_reconciliation_records, list) or len(authorized_reconciliation_records) == 0:
        raise InvalidProvenanceReconciliationError("authorized_reconciliation_records must be a non-empty list.")

    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    provenance_ledger = json.loads(provenance_ledger_path.read_text(encoding="utf-8"))

    corpus_records = corpus_manifest.get("records", [])
    prov_records = provenance_ledger.get("provenance_records", [])
    corpus_by_key = {(r["accession"], r["primary_document"]): r for r in corpus_records}
    prov_by_key = {(r["accession"], r["primary_document"]): r for r in prov_records}

    records_to_append: List[Dict[str, Any]] = []
    for rec in authorized_reconciliation_records:
        if not isinstance(rec, dict):
            raise InvalidProvenanceReconciliationError("Each reconciliation record must be a dict.")
        for field in REQUIRED_PROVENANCE_RECORD_FIELDS:
            if field not in rec or rec[field] is None:
                raise InvalidProvenanceReconciliationError(
                    f"Missing or null required provenance field '{field}'; fabricating unknown fields is prohibited."
                )
            if field != "error_message" and isinstance(rec[field], str) and not rec[field].strip():
                raise InvalidProvenanceReconciliationError(
                    f"Empty required provenance string field '{field}'; fabricating unknown fields is prohibited."
                )
            if field in ("series_ids", "class_ids", "target_symbols", "authority_chain_roles"):
                if not isinstance(rec[field], list) or len(rec[field]) == 0:
                    raise InvalidProvenanceReconciliationError(
                        f"Empty required provenance list field '{field}'."
                    )

        key = (rec["accession"], rec["primary_document"])
        if key not in corpus_by_key:
            raise InvalidProvenanceReconciliationError(
                f"Cannot reconcile provenance for {key}: not present in canonical corpus manifest."
            )

        corpus_entry = corpus_by_key[key]
        if rec["raw_sha256"] != corpus_entry["raw_sha256"]:
            raise CanonicalHashConflictError(
                f"Provenance raw_sha256 {rec['raw_sha256']} conflicts with corpus raw_sha256 {corpus_entry['raw_sha256']} for {key}."
            )
        for id_field in ("cik", "form", "acceptance_timestamp"):
            if id_field in corpus_entry and rec.get(id_field) != corpus_entry.get(id_field):
                raise InvalidProvenanceReconciliationError(
                    f"Provenance field '{id_field}' ({rec.get(id_field)!r}) conflicts with canonical corpus record ({corpus_entry.get(id_field)!r}) for {key}."
                )
        corpus_bytes = corpus_entry.get("byte_size", corpus_entry.get("byte_length"))
        if corpus_bytes is not None and int(rec["byte_length"]) != int(corpus_bytes):
            raise InvalidProvenanceReconciliationError(
                f"Provenance byte_length {rec['byte_length']} conflicts with canonical corpus byte_size {corpus_bytes} for {key}."
            )
        if "filing_date" in rec and "filing_date" in corpus_entry and rec["filing_date"] != corpus_entry["filing_date"]:
            raise InvalidProvenanceReconciliationError(
                f"Provenance filing_date {rec['filing_date']!r} conflicts with canonical corpus filing_date {corpus_entry['filing_date']!r} for {key}."
            )

        if key in prov_by_key:
            existing_prov = prov_by_key[key]
            if existing_prov["raw_sha256"] != rec["raw_sha256"]:
                raise CanonicalHashConflictError(
                    f"Conflicting existing provenance record for {key}."
                )
            for check_field in ("cik", "form", "acceptance_timestamp", "sec_source_url", "byte_length"):
                if existing_prov.get(check_field) != rec.get(check_field):
                    raise InvalidProvenanceReconciliationError(
                        f"Conflicting existing provenance field '{check_field}' for {key}."
                    )
            continue

        # Verify local file bytes if present in cache_dir or repo root
        candidate_file = resolve_contained_cache_path(cache_dir, rec["accession"], rec["primary_document"])
        if not candidate_file.exists():
            alt_file = REPO_ROOT / rec["local_path"]
            if alt_file.exists():
                candidate_file = alt_file
        if candidate_file.exists():
            disk_sha = compute_sha256_file(candidate_file)
            if disk_sha != rec["raw_sha256"]:
                raise CanonicalHashConflictError(
                    f"Disk file SHA-256 {disk_sha} conflicts with expected {rec['raw_sha256']} for {key}."
                )

        records_to_append.append(dict(rec))

    if not records_to_append:
        return {
            "status": "IDEMPOTENT_NOOP",
            "provenance_count_before": len(prov_records),
            "provenance_count_after": len(prov_records),
            "reconciled_count": 0,
            "corpus_manifest_mutated": False,
        }

    new_prov_records = list(prov_records) + records_to_append
    assert new_prov_records[: len(prov_records)] == prov_records

    initial_run_count = provenance_ledger.get(
        "initial_full_acquisition_document_count",
        provenance_ledger.get("acquired_new", 802) + provenance_ledger.get("reused_existing_verified", 55),
    )
    reconciled_hist_count = sum(
        1
        for r in new_prov_records
        if r.get("acquisition_method") == "HISTORICAL_PROVENANCE_RECONCILIATION"
        or r.get("reconciliation_type") == "HISTORICAL_PROVENANCE_RECONCILIATION"
    )

    new_ledger = dict(provenance_ledger)
    new_ledger["initial_full_acquisition_document_count"] = initial_run_count
    new_ledger["historical_provenance_reconciled_count"] = reconciled_hist_count
    new_ledger["current_canonical_provenance_membership_count"] = len(new_prov_records)
    new_ledger["expected_unique_documents"] = len(new_prov_records)
    new_ledger["successful_unique_documents"] = sum(
        1 for r in new_prov_records if r.get("validation_status") == "VALID"
    )
    new_ledger["provenance_records"] = new_prov_records

    tmp_ledger = provenance_ledger_path.parent / f"{provenance_ledger_path.name}.tmp_{os.getpid()}"
    try:
        tmp_ledger.write_text(json.dumps(new_ledger, indent=2) + "\n", encoding="utf-8")
        tmp_ledger.replace(provenance_ledger_path)
    finally:
        if tmp_ledger.exists():
            tmp_ledger.unlink()

    return {
        "status": "RECONCILED",
        "provenance_count_before": len(prov_records),
        "provenance_count_after": len(new_prov_records),
        "reconciled_count": len(records_to_append),
        "corpus_manifest_mutated": False,
    }


def acquire_authorized_documents(
    authorized_documents: List[Dict[str, Any]],
    corpus_manifest_path: Path = OUTPUT_CORPUS_MANIFEST_PATH,
    provenance_ledger_path: Path = OUTPUT_PROVENANCE_LEDGER_PATH,
    cache_dir: Path = PROSPECTUS_CACHE_DIR,
    session: Optional[requests.Session] = None,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """
    Generic incremental SEC-source acquisition API (INCREMENTAL_AUTHORIZED_SOURCE_ACQUISITION).
    Extends an existing internally consistent canonical corpus in append-only order with
    explicitly authorized sources, enforcing raw SHA-256 verification, idempotency,
    fail-closed hash-conflict rejection, and atomic rollback semantics.
    """
    if not isinstance(authorized_documents, list) or len(authorized_documents) == 0:
        raise InvalidAuthorizedSourceSpecError("authorized_documents must be a non-empty list.")

    raw_validated_specs = [validate_authorized_source_spec(s, cache_dir=cache_dir) for s in authorized_documents]

    # Pre-normalize and deduplicate by (accession, primary_document) before any network or file operations.
    # - Conflicting expected_raw_sha256 -> CanonicalHashConflictError
    # - Conflicting metadata -> InvalidAuthorizedSourceSpecError
    # - Identical duplicate specification -> DEDUPE_TO_ONE
    deduped_specs_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    validated_specs: List[Dict[str, Any]] = []
    for spec in raw_validated_specs:
        k = (spec["accession"], spec["primary_document"])
        if k in deduped_specs_by_key:
            first_spec = deduped_specs_by_key[k]
            if first_spec["expected_raw_sha256"] != spec["expected_raw_sha256"]:
                raise CanonicalHashConflictError(
                    f"Conflicting expected_raw_sha256 in duplicate request for {k}: "
                    f"{first_spec['expected_raw_sha256']} vs {spec['expected_raw_sha256']}"
                )
            if first_spec != spec:
                raise InvalidAuthorizedSourceSpecError(
                    f"Conflicting metadata in duplicate request specification for {k}."
                )
            # Identical specification: DEDUPE_TO_ONE
            continue
        deduped_specs_by_key[k] = spec
        validated_specs.append(spec)

    if not corpus_manifest_path.exists() or not provenance_ledger_path.exists():
        raise FileNotFoundError("Canonical corpus manifest and provenance ledger must exist before incremental acquisition.")

    orig_corpus_bytes = corpus_manifest_path.read_bytes()
    orig_ledger_bytes = provenance_ledger_path.read_bytes()

    corpus_manifest = json.loads(orig_corpus_bytes.decode("utf-8"))
    provenance_ledger = json.loads(orig_ledger_bytes.decode("utf-8"))

    old_corpus_records = corpus_manifest.get("records", [])
    old_prov_records = provenance_ledger.get("provenance_records", [])
    prior_agg = corpus_manifest.get("corpus_aggregate_identity", "")

    if not verify_manifest_aggregate_compatibility(corpus_manifest):
        raise RuntimeError("Existing canonical corpus manifest aggregate hash does not match its records.")

    # Do not silently repair legacy provenance deficits during incremental acquisition
    deficit = detect_legacy_provenance_deficit(corpus_manifest, provenance_ledger)
    if deficit["deficit_detected"]:
        raise LegacyProvenanceReconciliationRequiredError(
            f"LEGACY_PROVENANCE_RECONCILIATION_REQUIRED: corpus={deficit['corpus_count']} vs provenance={deficit['provenance_count']}",
            deficit,
        )

    corpus_by_key = {(r["accession"], r["primary_document"]): r for r in old_corpus_records}
    prov_by_key = {(r["accession"], r["primary_document"]): r for r in old_prov_records}

    cache_dir.mkdir(parents=True, exist_ok=True)
    own_session = False
    if session is None:
        session = requests.Session()
        own_session = True

    staged_cache_files: List[Tuple[Path, Path]] = []
    new_corpus_entries: List[Dict[str, Any]] = []
    new_prov_entries: List[Dict[str, Any]] = []
    acquired_summary: List[Dict[str, Any]] = []

    try:
        for spec in validated_specs:
            acc = spec["accession"]
            pdoc = spec["primary_document"]
            expected_sha = spec["expected_raw_sha256"]
            key = (acc, pdoc)
            fn = f"{acc}_{pdoc}"
            local_path = resolve_contained_cache_path(cache_dir, acc, pdoc)

            # 1. Check if already in canonical corpus
            if key in corpus_by_key:
                existing_corpus_sha = corpus_by_key[key]["raw_sha256"].lower()
                if existing_corpus_sha != expected_sha:
                    raise CanonicalHashConflictError(
                        f"Conflicting SHA-256 for existing canonical source {key}: existing={existing_corpus_sha}, expected={expected_sha}"
                    )
                if key in prov_by_key and prov_by_key[key]["raw_sha256"].lower() != expected_sha:
                    raise CanonicalHashConflictError(
                        f"Conflicting SHA-256 in provenance ledger for {key}."
                    )
                if not local_path.exists():
                    raise SECPayloadValidationError(
                        f"Canonical record {key} exists in manifest but cache file {local_path} is missing."
                    )
                disk_bytes = local_path.read_bytes()
                is_err, err_msg = is_sec_interstitial_or_error(disk_bytes)
                if is_err:
                    raise SECPayloadValidationError(f"Existing cache file invalid for {key}: {err_msg}")
                disk_sha = compute_sha256_bytes(disk_bytes).lower()
                if disk_sha != expected_sha:
                    raise CanonicalHashConflictError(
                        f"Existing cache file SHA-256 {disk_sha} conflicts with canonical SHA-256 {expected_sha} for {key}."
                    )
                continue

            # 2. Not in canonical corpus yet: check if cache file already exists on disk
            expected_url = construct_sec_url(
                spec["cik"], acc, pdoc, spec.get("expected_sec_source_location", "")
            )
            raw_bytes = b""
            acq_method = "ACQUIRED_NEW"
            acq_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            retries = 0

            if local_path.exists():
                disk_bytes = local_path.read_bytes()
                is_err, err_msg = is_sec_interstitial_or_error(disk_bytes)
                if is_err:
                    raise SECPayloadValidationError(
                        f"Existing uncertified cache file for {key} failed validation: {err_msg}"
                    )
                disk_sha = compute_sha256_bytes(disk_bytes).lower()
                if disk_sha != expected_sha:
                    raise CanonicalHashConflictError(
                        f"Existing cache file SHA-256 {disk_sha} conflicts with expected {expected_sha} for {key}."
                    )
                raw_bytes = disk_bytes
                acq_method = "REUSED_EXISTING_VERIFIED"
                acq_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(local_path.stat().st_mtime))
            else:
                # Fetch from SEC EDGAR into a staged temporary file (do not touch local_path until atomic promotion)
                backoffs = [1.0, 2.0, 4.0, 8.0]
                last_err = ""
                fetched_ok = False
                for attempt in range(max_retries):
                    try:
                        time.sleep(REQUEST_DELAY_SECONDS)
                        resp = session.get(expected_url, headers=SEC_HEADERS, timeout=15)
                        if resp.status_code == 200:
                            content = resp.content
                            is_err, err_msg = is_sec_interstitial_or_error(content)
                            if is_err:
                                last_err = f"SEC content validation failed: {err_msg}"
                            else:
                                content_sha = compute_sha256_bytes(content).lower()
                                if content_sha != expected_sha:
                                    raise CanonicalHashConflictError(
                                        f"Acquired raw SHA-256 {content_sha} does not match expected_raw_sha256 {expected_sha} for {key}."
                                    )
                                raw_bytes = content
                                fetched_ok = True
                                break
                        elif resp.status_code == 404:
                            last_err = "HTTP 404 Not Found"
                            break
                        else:
                            last_err = f"HTTP {resp.status_code}"
                    except CanonicalHashConflictError:
                        raise
                    except Exception as exc:
                        last_err = f"Exception: {str(exc)}"
                    retries += 1
                    if attempt < len(backoffs):
                        time.sleep(backoffs[attempt])

                if not fetched_ok:
                    raise SECPayloadValidationError(
                        f"Failed to acquire valid SEC source for {key}: {last_err}"
                    )

                tmp_cache = cache_dir / f"{fn}.tmp_{os.getpid()}_{len(staged_cache_files)}"
                tmp_cache.write_bytes(raw_bytes)
                if compute_sha256_file(tmp_cache).lower() != expected_sha:
                    raise SECPayloadValidationError(f"Staged cache file hash mismatch for {key}.")
                staged_cache_files.append((tmp_cache, local_path))

            rel_local_path = format_local_path(local_path)
            prov_entry: Dict[str, Any] = {
                "accession": acc,
                "primary_document": pdoc,
                "form": spec["form"],
                "cik": spec["cik"],
                "series_ids": spec["series_ids"],
                "class_ids": spec["class_ids"],
                "target_symbols": spec["target_symbols"],
                "authority_chain_roles": spec["authority_chain_roles"],
                "snapshot_boundary": SNAPSHOT_BOUNDARY_ISO,
                "acceptance_timestamp": spec["acceptance_timestamp"],
                "sec_source_url": expected_url,
                "http_status": 200,
                "acquisition_timestamp": acq_ts,
                "byte_length": len(raw_bytes),
                "raw_sha256": expected_sha,
                "local_path": rel_local_path,
                "acquisition_method": acq_method,
                "retry_count": retries,
                "validation_status": "VALID",
                "error_message": "",
            }
            corpus_entry: Dict[str, Any] = {
                "accession": acc,
                "primary_document": pdoc,
                "form": spec["form"],
                "cik": spec["cik"],
                "filing_date": spec["filing_date"],
                "series_ids": spec["series_ids"],
                "class_ids": spec["class_ids"],
                "target_symbols": spec["target_symbols"],
                "authority_chain_roles": spec["authority_chain_roles"],
                "acceptance_timestamp": spec["acceptance_timestamp"],
                "byte_length": len(raw_bytes),
                "byte_size": len(raw_bytes),
                "raw_sha256": expected_sha,
                "local_path": rel_local_path,
                "document_role": spec.get("document_role", spec["authority_chain_roles"][0]),
                "validation_status": "VALID",
            }
            for opt_field in ("predecessor_symbol", "effective_date", "legal_name"):
                if opt_field in spec:
                    prov_entry[opt_field] = spec[opt_field]
                    corpus_entry[opt_field] = spec[opt_field]

            new_prov_entries.append(prov_entry)
            new_corpus_entries.append(corpus_entry)
            acquired_summary.append(corpus_entry)
            corpus_by_key[key] = corpus_entry
            prov_by_key[key] = prov_entry

        # 3. If no new entries were required, return IDEMPOTENT_NOOP with 0 file mutations
        if not new_corpus_entries:
            return {
                "status": "IDEMPOTENT_NOOP",
                "execution_mode": EXECUTION_MODE_INCREMENTAL,
                "corpus_document_count_before": len(old_corpus_records),
                "corpus_document_count_after": len(old_corpus_records),
                "provenance_record_count_before": len(old_prov_records),
                "provenance_record_count_after": len(old_prov_records),
                "new_documents_added": 0,
                "files_mutated": 0,
                "prior_corpus_aggregate_sha256": prior_agg,
                "new_corpus_aggregate_sha256": prior_agg,
                "acquired_records": [],
            }

        # 4. Construct candidate append-only corpus manifest and provenance ledger
        candidate_corpus_records = list(old_corpus_records) + new_corpus_entries
        candidate_prov_records = list(old_prov_records) + new_prov_entries

        # Prefix preservation assertions
        assert candidate_corpus_records[: len(old_corpus_records)] == old_corpus_records
        assert candidate_prov_records[: len(old_prov_records)] == old_prov_records
        assert len(candidate_corpus_records) == len(candidate_prov_records)

        new_agg = compute_corpus_aggregate_identity(candidate_corpus_records)
        assert HEX64_RE.match(new_agg)

        candidate_corpus_doc = dict(corpus_manifest)
        candidate_corpus_doc["corpus_document_count"] = len(candidate_corpus_records)
        candidate_corpus_doc["total_sources"] = len(candidate_corpus_records)
        candidate_corpus_doc["corpus_aggregate_identity_version"] = CORPUS_AGGREGATE_IDENTITY_VERSION
        candidate_corpus_doc["corpus_aggregate_identity"] = new_agg
        candidate_corpus_doc["records"] = candidate_corpus_records

        candidate_ledger_doc = dict(provenance_ledger)
        candidate_ledger_doc["expected_unique_documents"] = len(candidate_prov_records)
        candidate_ledger_doc["successful_unique_documents"] = sum(
            1 for r in candidate_prov_records if r.get("validation_status") == "VALID"
        )
        candidate_ledger_doc["acquired_new"] = sum(
            1 for r in candidate_prov_records if r.get("acquisition_method") == "ACQUIRED_NEW"
        )
        candidate_ledger_doc["reused_existing_verified"] = sum(
            1 for r in candidate_prov_records if r.get("acquisition_method") == "REUSED_EXISTING_VERIFIED"
        )
        candidate_ledger_doc["provenance_records"] = candidate_prov_records

        tmp_corpus_path = corpus_manifest_path.parent / f"{corpus_manifest_path.name}.tmp_{os.getpid()}"
        tmp_ledger_path = provenance_ledger_path.parent / f"{provenance_ledger_path.name}.tmp_{os.getpid()}"

        tmp_corpus_path.write_text(json.dumps(candidate_corpus_doc, indent=2) + "\n", encoding="utf-8")
        tmp_ledger_path.write_text(json.dumps(candidate_ledger_doc, indent=2) + "\n", encoding="utf-8")

        # Atomic promotion with rollback protection
        promoted_cache_paths: List[Path] = []
        try:
            for tmp_c, final_c in staged_cache_files:
                tmp_c.replace(final_c)
                promoted_cache_paths.append(final_c)
            tmp_corpus_path.replace(corpus_manifest_path)
            tmp_ledger_path.replace(provenance_ledger_path)
        except Exception:
            # Rollback any promoted artifacts so canonical state is never partially mutated
            for prom_c in promoted_cache_paths:
                if prom_c.exists():
                    prom_c.unlink()
            corpus_manifest_path.write_bytes(orig_corpus_bytes)
            provenance_ledger_path.write_bytes(orig_ledger_bytes)
            raise
        finally:
            if tmp_corpus_path.exists():
                tmp_corpus_path.unlink()
            if tmp_ledger_path.exists():
                tmp_ledger_path.unlink()

        return {
            "status": "EXTENDED",
            "execution_mode": EXECUTION_MODE_INCREMENTAL,
            "corpus_document_count_before": len(old_corpus_records),
            "corpus_document_count_after": len(candidate_corpus_records),
            "provenance_record_count_before": len(old_prov_records),
            "provenance_record_count_after": len(candidate_prov_records),
            "new_documents_added": len(new_corpus_entries),
            "files_mutated": 2 + len(promoted_cache_paths),
            "prior_corpus_aggregate_sha256": prior_agg,
            "new_corpus_aggregate_sha256": new_agg,
            "acquired_records": acquired_summary,
        }
    finally:
        for tmp_c, _ in staged_cache_files:
            if tmp_c.exists():
                tmp_c.unlink()
        if own_session:
            session.close()


def execute_acquisition(
    output_corpus_manifest_path: Path = OUTPUT_CORPUS_MANIFEST_PATH,
    output_provenance_ledger_path: Path = OUTPUT_PROVENANCE_LEDGER_PATH,
    cache_dir: Path = PROSPECTUS_CACHE_DIR,
):
    """
    Main execution function for INITIAL_FULL_CORPUS_ACQUISITION (the historical 857-document baseline).
    Refuses destructive overwrite if the target corpus manifest already contains > 857 canonical documents.
    """
    print("=" * 80, flush=True)
    print("ARX TERMINAL — ETF RESEARCH PIPELINE V2", flush=True)
    print(f"SEC TARGET SOURCE ACQUISITION ENGINE ({EXECUTION_MODE_INITIAL_FULL})", flush=True)
    print(f"Snapshot Boundary: {SNAPSHOT_BOUNDARY_ISO}", flush=True)
    print("=" * 80, flush=True)

    # Guard against truncating an 858+ canonical corpus
    if output_corpus_manifest_path.exists():
        existing_manifest = json.loads(output_corpus_manifest_path.read_text(encoding="utf-8"))
        existing_count = max(
            int(existing_manifest.get("corpus_document_count", 0)),
            len(existing_manifest.get("records", [])),
        )
        if existing_count > 857:
            raise DestructiveCorpusOverwriteError(
                f"REFUSE_DESTRUCTIVE_OVERWRITE: Target corpus manifest {output_corpus_manifest_path} "
                f"contains {existing_count} documents (> 857 historical baseline). "
                f"Use {EXECUTION_MODE_INCREMENTAL} (acquire_authorized_documents) instead."
            )

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

    cache_dir.mkdir(parents=True, exist_ok=True)

    # 3. Execute Acquisition Loop
    session = requests.Session()
    provenance_records = []
    acquired_new = 0
    reused_existing = 0
    failed_count = 0

    print(f"\nBeginning acquisition / verification across {len(unique_items)} unique documents...", flush=True)
    start_time = time.time()

    for idx, item in enumerate(unique_items, 1):
        rec = acquire_or_verify_document(item, cache_dir, session)
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

    with open(output_provenance_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger_doc, f, indent=2)
    ledger_sha = compute_sha256_file(output_provenance_ledger_path)
    print(f"Wrote Provenance Ledger: {output_provenance_ledger_path}")
    print(f"PROVENANCE_LEDGER_SHA256 = {ledger_sha}")

    # 5. Generate Corpus Manifest (records are already sorted by (accession, primary_document) from unique_items)
    corpus_manifest_records = []
    for r in provenance_records:
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

    corpus_aggregate_id = compute_corpus_aggregate_identity(corpus_manifest_records)
    corpus_manifest_doc = {
        "manifest_version": "1.0.0",
        "manifest_authority": "SEC_TARGET_SOURCE_ACQUISITION_EXECUTION_GATE",
        "snapshot_boundary": SNAPSHOT_BOUNDARY_ISO,
        "corpus_document_count": len(corpus_manifest_records),
        "corpus_aggregate_identity_algorithm": "SHA256(join_newline('{accession}_{primary_document}:{raw_sha256}'))",
        "corpus_aggregate_identity_version": CORPUS_AGGREGATE_IDENTITY_VERSION,
        "corpus_aggregate_identity": corpus_aggregate_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": corpus_manifest_records,
    }

    with open(output_corpus_manifest_path, "w", encoding="utf-8") as f:
        json.dump(corpus_manifest_doc, f, indent=2)
    corpus_manifest_sha = compute_sha256_file(output_corpus_manifest_path)
    print(f"Wrote Corpus Manifest:   {output_corpus_manifest_path}")
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
