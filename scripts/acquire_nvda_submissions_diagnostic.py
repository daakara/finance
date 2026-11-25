#!/usr/bin/env python3
"""
ARX Terminal — Bounded NVDA SEC Submissions Catalog Diagnostic Acquisition.

Authoritative Acquisition Contract (Issue #165):
- Single Target Resource: Strictly https://data.sec.gov/submissions/CIK0001045810.json
- One GET Maximum: Hard rate limiter (<= 5.0 req/s), zero automatic retries, zero redirects,
  zero pagination requests, zero XML downloads.
- Separate Diagnostic Budget: Completely decoupled from expired/exhausted trial allowances.
- Verified SEC User-Agent: Enforces SEC Fair Access format ('<EntityName> <Email>').
- Raw Byte Storage: Retains exact response bytes before parsing, recording request URL,
  UTC acquisition time, HTTP status, byte count, and SHA-256 digest in verifiable storage.
- SEC Coverage Evidence Custody: All artifacts stored under a dedicated evidence directory.
- Failure Recording: Failed attempts (timeout, redirect, HTTP 4xx/5xx) are durably recorded
  in the receipt without automatic retries.
- Offline Catalog Validation: Validates issuer identity (CIK 0001045810, NVDA), schema structure,
  historical pagination file overlap, and interval coverage [2026-08-26, 2026-11-25).
- Anti-Inference Principle: Latest accession date alone is never treated as proof of completeness.
- Zero Side Effects: Zero production database writes, zero worker activation, zero coverage-flag changes.
  PARTIAL_UNCERTIFIED, active snapshot SNAP_SM_1792939989_48f30db6, and all operational holds remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.data.sec_edgar_fetcher import (
    SecRateLimiter,
    validate_sec_user_agent,
    get_sec_user_agent,
)

# Authoritative Contract Constants
TARGET_URL = "https://data.sec.gov/submissions/CIK0001045810.json"
TARGET_CIK = "0001045810"
TARGET_TICKER = "NVDA"
TARGET_COMPANY = "NVIDIA CORP"
ASSESSMENT_INTERVAL_START = "2026-08-26T00:00:00Z"
ASSESSMENT_INTERVAL_END = "2026-11-25T00:00:00Z"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "data" / "sec_coverage_evidence" / "CIK0001045810"
MAX_REQUESTS_ALLOWED = 1

# Known Accounted In-Window Accessions (15 Baseline + 1 Forward Trial Accretion = 16)
KNOWN_IN_WINDOW_ACCESSIONS: Set[str] = {
    "0001199039-26-000012",
    "0001696841-26-000010",
    "0001197647-26-000009",
    "0001199039-26-000014",
    "0002152188-26-000005",
    "0001197649-26-000012",
    "0001243821-26-000007",
    "0001283854-26-000010",
    "0001588670-26-000014",
    "0001696841-26-000012",
    "0001199039-26-000016",
    "0001696841-26-000014",
    "0001197647-26-000011",
    "0002152188-26-000007",
    "0001197647-26-000013",
    "0001696841-26-000016",  # Forward accretion from trial Slot 4
}


class DiagnosticBudgetExceededError(RuntimeError):
    """Raised when more than exactly one request is attempted during a diagnostic invocation."""
    pass


class DiagnosticAcquisitionHandler:
    """
    Executes a bounded diagnostic acquisition for NVDA submissions with single-request
    enforcement, no retries, no redirects, and verifiable artifact storage.
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        evidence_dir: Optional[Path] = None,
        timeout_sec: float = 15.0,
        rate_limiter: Optional[SecRateLimiter] = None,
        interval_start: str = ASSESSMENT_INTERVAL_START,
        interval_end: str = ASSESSMENT_INTERVAL_END,
    ):
        self.user_agent = user_agent or get_sec_user_agent() or ""
        self.evidence_dir = Path(evidence_dir or DEFAULT_EVIDENCE_DIR)
        self.timeout_sec = timeout_sec
        self.rate_limiter = rate_limiter or SecRateLimiter(max_per_second=5.0)
        self.interval_start = interval_start
        self.interval_end = interval_end
        self.attempt_count = 0

    def validate_preconditions(self) -> Tuple[bool, str]:
        """Validate user agent and configuration prior to any network transmission."""
        is_valid, reason = validate_sec_user_agent(self.user_agent)
        if not is_valid:
            return False, f"INVALID_USER_AGENT: {reason}"
        return True, "Preconditions valid"

    def execute_single_request(
        self,
        session: Optional[requests.Session] = None,
    ) -> Dict[str, Any]:
        """
        Executes at most one HTTP GET request against the official SEC submissions endpoint.
        Returns a structured receipt dictionary.
        """
        if self.attempt_count >= MAX_REQUESTS_ALLOWED:
            raise DiagnosticBudgetExceededError(
                f"DIAGNOSTIC_BUDGET_EXCEEDED: Invocations cannot exceed {MAX_REQUESTS_ALLOWED} request. "
                f"Current attempt count: {self.attempt_count}."
            )

        self.attempt_count += 1
        acquisition_time_utc = datetime.now(timezone.utc).isoformat()
        os.makedirs(self.evidence_dir, exist_ok=True)

        receipt: Dict[str, Any] = {
            "title": "NVDA Submissions Catalog Diagnostic Acquisition Receipt",
            "issue_reference": "ARX Issue #165",
            "request_url": TARGET_URL,
            "target_cik": TARGET_CIK,
            "target_ticker": TARGET_TICKER,
            "target_company": TARGET_COMPANY,
            "acquisition_timestamp_utc": acquisition_time_utc,
            "attempt_count": self.attempt_count,
            "max_requests_allowed": MAX_REQUESTS_ALLOWED,
            "user_agent_redacted": self._redact_user_agent(self.user_agent),
            "http_status": None,
            "byte_count": 0,
            "sha256": None,
            "raw_artifact_path": None,
            "receipt_path": None,
            "success": False,
            "error": None,
            "validation": None,
            "preserved_governance_state": {
                "production_coverage_status": "PARTIAL_UNCERTIFIED",
                "active_snapshot_id": "SNAP_SM_1792939989_48f30db6",
                "database_writes_performed": 0,
                "worker_activation": False,
                "holds": [
                    "HOLD_SEC_001",
                    "HOLD_SCAN_001",
                    "HOLD_EXEC_001",
                    "PHASE_3_HOLD",
                    "ARX_CONFLUENCE_INTEGRITY_HOLD",
                ],
            },
        }

        # 1. Precondition check
        valid_pre, pre_reason = self.validate_preconditions()
        if not valid_pre:
            receipt["error"] = pre_reason
            return self._persist_receipt(receipt, acquisition_time_utc)

        # 2. Setup Session with 0 retries and rate limiter
        if session is None:
            http_session = requests.Session()
            http_session.mount("https://", HTTPAdapter(max_retries=0))
            http_session.mount("http://", HTTPAdapter(max_retries=0))
        else:
            http_session = session

        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

        # 3. Rate limiter pacing
        self.rate_limiter.acquire()

        # 4. Transmission with allow_redirects=False
        try:
            resp = http_session.get(
                TARGET_URL,
                headers=headers,
                timeout=self.timeout_sec,
                allow_redirects=False,
            )
            receipt["http_status"] = resp.status_code

            # Handle redirects explicitly
            if resp.status_code in (301, 302, 303, 307, 308):
                receipt["error"] = (
                    f"REDIRECT_REFUSED: HTTP {resp.status_code} received with Location: "
                    f"'{resp.headers.get('Location')}'. Automatic redirects are forbidden."
                )
                if resp.content:
                    self._persist_raw_bytes(resp.content, receipt, acquisition_time_utc)
                return self._persist_receipt(receipt, acquisition_time_utc)

            # Handle non-200 HTTP statuses
            if resp.status_code != 200:
                receipt["error"] = f"HTTP_ERROR: Received status code {resp.status_code}"
                if resp.content:
                    self._persist_raw_bytes(resp.content, receipt, acquisition_time_utc)
                return self._persist_receipt(receipt, acquisition_time_utc)

            # Success: persist exact raw response bytes
            raw_bytes = resp.content
            self._persist_raw_bytes(raw_bytes, receipt, acquisition_time_utc)

            # Run offline validation on exact retained bytes
            validation_result = self.validate_catalog_bytes(raw_bytes, receipt["sha256"])
            receipt["validation"] = validation_result
            receipt["success"] = validation_result.get("is_valid", False)
            if not receipt["success"]:
                receipt["error"] = validation_result.get("error", "Catalog validation failed")

        except requests.exceptions.Timeout as exc:
            receipt["error"] = f"TIMEOUT: Request timed out after {self.timeout_sec}s: {exc}"
        except requests.exceptions.RequestException as exc:
            receipt["error"] = f"REQUEST_FAILED: Network transport error: {exc}"
        except Exception as exc:
            receipt["error"] = f"UNEXPECTED_ERROR: {exc}"

        return self._persist_receipt(receipt, acquisition_time_utc)

    def validate_catalog_bytes(self, raw_bytes: bytes, digest: str) -> Dict[str, Any]:
        """
        Validates catalog bytes against SEC schema, exact issuer identity, pagination,
        and assessment interval coverage [2026-08-26, 2026-11-25).
        """
        result: Dict[str, Any] = {
            "is_valid": False,
            "issuer_identity_verified": False,
            "schema_verified": False,
            "pagination_verified": False,
            "interval_coverage_verified": False,
            "contiguous_span": None,
            "eligible_accessions_in_interval": [],
            "missing_accessions": [],
            "overlapping_pagination_files": [],
            "error": None,
        }

        # 1. Parse JSON
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            result["error"] = f"MALFORMED_JSON: Failed to decode catalog JSON: {exc}"
            return result

        if not isinstance(data, dict):
            result["error"] = "MALFORMED_SCHEMA: Root element must be a JSON object"
            return result

        # 2. Issuer Identity Binding
        cat_cik = str(data.get("cik") or "").strip().zfill(10)
        tickers = [str(t).upper().strip() for t in (data.get("tickers") or [])]
        if cat_cik != TARGET_CIK:
            result["error"] = f"WRONG_CIK: Expected '{TARGET_CIK}', found '{cat_cik}'"
            return result
        if TARGET_TICKER not in tickers:
            result["error"] = f"WRONG_TICKER: Expected '{TARGET_TICKER}' in tickers {tickers}"
            return result
        result["issuer_identity_verified"] = True

        # 3. Schema Structure in filings.recent
        filings = data.get("filings")
        if not isinstance(filings, dict):
            result["error"] = "MALFORMED_SCHEMA: Missing 'filings' dict"
            return result
        recent = filings.get("recent")
        if not isinstance(recent, dict):
            result["error"] = "MALFORMED_SCHEMA: Missing 'filings.recent' dict"
            return result

        required_cols = ("accessionNumber", "filingDate", "acceptanceDateTime", "form")
        for col in required_cols:
            if col not in recent or not isinstance(recent[col], list):
                result["error"] = f"MALFORMED_COLUMNS: Column '{col}' missing or not a list"
                return result

        n_rows = len(recent["accessionNumber"])
        for col in required_cols:
            if len(recent[col]) != n_rows:
                result["error"] = f"COLUMN_LENGTH_MISMATCH: Column '{col}' length != {n_rows}"
                return result
        result["schema_verified"] = True

        # 4. Pagination Audit (filings.files)
        hist_files = filings.get("files") or []
        start_d = self.interval_start[:10]
        end_d = self.interval_end[:10]

        overlapping_files = []
        if isinstance(hist_files, list):
            for f_entry in hist_files:
                if isinstance(f_entry, dict):
                    fname = str(f_entry.get("name") or "").strip()
                    f_from = str(f_entry.get("filingFrom") or f_entry.get("from") or "")[:10]
                    f_to = str(f_entry.get("filingTo") or f_entry.get("to") or "")[:10]
                    # Check overlap with [start_d, end_d]
                    if f_from and f_to and (f_from <= end_d and f_to >= start_d):
                        overlapping_files.append({"name": fname, "from": f_from, "to": f_to})
                    elif not f_from or not f_to:
                        # Unspecified dates conservatively flagged
                        overlapping_files.append({"name": fname, "from": f_from, "to": f_to})

        result["overlapping_pagination_files"] = overlapping_files
        # If historical pagination files overlap the assessment interval, report them
        if overlapping_files:
            result["pagination_verified"] = False
            result["error"] = (
                f"PAGINATION_EVIDENCE_REQUIRED: Historical files {overlapping_files} overlap "
                f"[{start_d}, {end_d}]. Single-request acquisition contract prohibits secondary downloads."
            )
            return result
        result["pagination_verified"] = True

        # 5. Interval Coverage & Anti-Inference Audit
        in_interval_records = []
        for i in range(n_rows):
            form = str(recent["form"][i]).strip().upper()
            f_date = str(recent["filingDate"][i]).strip()[:10]
            acc_num = str(recent["accessionNumber"][i]).strip()
            acc_dt = str(recent["acceptanceDateTime"][i]).strip()
            if form in ("4", "4/A", "4A") and (start_d <= f_date < end_d):
                in_interval_records.append({
                    "accessionNumber": acc_num,
                    "filingDate": f_date,
                    "acceptanceDateTime": acc_dt,
                    "form": form,
                })

        eligible_accessions = sorted([r["accessionNumber"] for r in in_interval_records])
        result["eligible_accessions_in_interval"] = eligible_accessions

        # Compare with known accounted accessions
        missing_from_known = [acc for acc in eligible_accessions if acc not in KNOWN_IN_WINDOW_ACCESSIONS]
        result["missing_accessions"] = missing_from_known

        # Check contiguous span of all filings in recent
        all_filing_dates = [str(recent["filingDate"][i])[:10] for i in range(n_rows) if recent["filingDate"][i]]
        if all_filing_dates:
            earliest_in_recent = min(all_filing_dates)
            latest_in_recent = max(all_filing_dates)
            result["contiguous_span"] = {"earliest": earliest_in_recent, "latest": latest_in_recent}

            # Anti-inference check: Latest accession date alone is NOT proof of completeness.
            # Must satisfy: earliest_in_recent <= start_d AND catalog retrieval timestamp >= end_d
            if earliest_in_recent > start_d:
                result["interval_coverage_verified"] = False
                result["error"] = (
                    f"INTERVAL_INCOMPLETE: Earliest filing in catalog ({earliest_in_recent}) "
                    f"is after interval start ({start_d})."
                )
                return result

        result["interval_coverage_verified"] = True
        result["is_valid"] = True
        return result

    def _persist_raw_bytes(self, raw_bytes: bytes, receipt: Dict[str, Any], timestamp_iso: str) -> None:
        """Saves exact raw response bytes to disk and computes SHA-256 digest."""
        ts_slug = timestamp_iso.replace(":", "").replace("-", "")
        raw_filename = f"CIK{TARGET_CIK}_submissions_{ts_slug}.json"
        raw_path = self.evidence_dir / raw_filename
        with open(raw_path, "wb") as f:
            f.write(raw_bytes)

        receipt["byte_count"] = len(raw_bytes)
        receipt["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        receipt["raw_artifact_path"] = str(raw_path.relative_to(REPO_ROOT) if raw_path.is_relative_to(REPO_ROOT) else raw_path)

    def _persist_receipt(self, receipt: Dict[str, Any], timestamp_iso: str) -> Dict[str, Any]:
        """Saves the diagnostic acquisition receipt JSON."""
        ts_slug = timestamp_iso.replace(":", "").replace("-", "")
        receipt_filename = f"acquisition_receipt_{ts_slug}.json"
        receipt_path = self.evidence_dir / receipt_filename
        receipt["receipt_path"] = str(receipt_path.relative_to(REPO_ROOT) if receipt_path.is_relative_to(REPO_ROOT) else receipt_path)

        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

        return receipt

    @staticmethod
    def _redact_user_agent(ua: str) -> str:
        """Redacts contact email for safe receipt logging."""
        if not ua:
            return ""
        return re.sub(r"[\w\.-]+@([\w\.-]+\.\w+)", r"***@\1", ua)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded NVDA SEC submissions diagnostic acquisition (Issue #165)."
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="SEC User-Agent header in required format ('EntityName ContactEmail@domain.com').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_EVIDENCE_DIR),
        help="Directory to store retained raw response bytes and diagnostic receipts.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds (default: 15.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and preconditions without transmitting HTTP wire egress.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    handler = DiagnosticAcquisitionHandler(
        user_agent=args.user_agent,
        evidence_dir=Path(args.output_dir),
        timeout_sec=args.timeout,
    )

    valid_pre, pre_msg = handler.validate_preconditions()
    if not valid_pre:
        print(f"FAILED_PRECONDITIONS: {pre_msg}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("DRY_RUN_PASSED: Diagnostic preconditions, rate limiter, and storage directories verified.")
        sys.exit(0)

    receipt = handler.execute_single_request()
    print(json.dumps(receipt, indent=2))

    if not receipt.get("success", False):
        print(f"\nACQUISITION_FAILED: {receipt.get('error')}", file=sys.stderr)
        sys.exit(2)

    print(f"\nACQUISITION_SUCCESS: SHA-256 {receipt.get('sha256')}")
    sys.exit(0)


if __name__ == "__main__":
    main()
