"""
ARX Terminal — Smart Money Filing Reconciliation Engine (Gate B1 & Phase D).

Implements authoritative reconciliation against SEC daily filing indexes:
- Answers: Did we miss any ownership filings that belong to this observation interval?
- Detects omissions in low-latency fast path.
- Verifies local store completeness (raw filing and parsed transaction integrity).
- Logs and tracks source gaps in sm_source_gaps.
- Computes mathematical coverage status (COMPLETE vs PARTIAL/UNKNOWN).
- Enforces canonical connection factory (connect_smart_money_db) and shared rate authority.
"""

import os
import re
import time
import uuid
import hashlib
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple

import requests

from analyst_dashboard.data.sec_edgar_fetcher import (
    SecRateLimiter,
    RateLimiter,
    DEFAULT_SEC_RATE_LIMITER,
    validate_sec_user_agent,
    get_sec_user_agent,
)
from analyst_dashboard.smart_money.form4_ingestion import (
    DEFAULT_SMART_MONEY_DB,
    connect_smart_money_db,
)
from analyst_dashboard.smart_money.incremental_models import GapType, GapStatus

logger = logging.getLogger(__name__)

ACCESSION_EXTRACTION_REGEX = re.compile(r"(\d{10}-\d{2}-\d{6})")


@dataclass(frozen=True)
class IndexFilingRecord:
    """Single filing entry extracted from SEC daily master or form index."""
    accession_number: str
    issuer_cik: str
    company_name: str
    form_type: str
    filing_date: str  # YYYY-MM-DD
    source_filename: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconciliationReport:
    """Authoritative reconciliation result comparing daily index vs local store."""
    reconciliation_id: str
    reconciled_at: str
    interval_start: str
    interval_end: str
    expected_index_count: int
    accounted_local_count: int
    missing_accessions: List[str]
    unparsed_accessions: List[str]
    newly_discovered_accessions: List[str]
    open_gap_count: int
    coverage_status: str  # COMPLETE, PARTIAL, DEGRADED, UNKNOWN
    failure_reasons: List[str]
    reconciliation_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FilingReconciliationEngine:
    """
    Authoritative reconciliation engine auditing SEC Form 4 completeness
    against official daily index files.
    """

    DAILY_INDEX_BASE_URL = "https://www.sec.gov/Archives/edgar/daily-index"

    def __init__(
        self,
        db_path: str = DEFAULT_SMART_MONEY_DB,
        rate_limiter: Optional[SecRateLimiter] = None,
        session: Optional[requests.Session] = None,
    ):
        self.db_path = db_path
        self.rate_limiter = rate_limiter or DEFAULT_SEC_RATE_LIMITER
        self.session = session or requests.Session()
        ua = get_sec_user_agent()
        self.is_configured, _ = validate_sec_user_agent(ua)
        if self.is_configured and ua:
            self.session.headers.update({"User-Agent": ua.strip()})

        self._init_reconciliation_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Centralized connection factory enforcing WAL & busy_timeout."""
        return connect_smart_money_db(self.db_path)

    def _init_reconciliation_table(self):
        """Initializes sm_reconciliation_reports and sm_source_gaps tables if needed."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sm_reconciliation_reports (
                    reconciliation_id TEXT PRIMARY KEY,
                    reconciled_at TEXT NOT NULL,
                    interval_start TEXT NOT NULL,
                    interval_end TEXT NOT NULL,
                    expected_index_count INTEGER NOT NULL,
                    accounted_local_count INTEGER NOT NULL,
                    missing_count INTEGER NOT NULL,
                    open_gap_count INTEGER NOT NULL,
                    coverage_status TEXT NOT NULL,
                    failure_reasons_json TEXT NOT NULL,
                    reconciliation_hash TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sm_source_gaps (
                    gap_id TEXT PRIMARY KEY,
                    incremental_cycle_id TEXT,
                    partition_id TEXT,
                    issuer_cik TEXT,
                    universe_build_id TEXT,
                    gap_type TEXT,
                    window_start TEXT,
                    window_end TEXT,
                    accession_number TEXT,
                    first_detected_at TEXT,
                    last_attempted_at TEXT,
                    attempt_count INTEGER,
                    status TEXT,
                    source_event_time TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def fetch_daily_master_index(self, date_str: str) -> Optional[str]:
        """
        Availability-driven retrieval of SEC daily master index.
        Returns index content string if available, or None if upstream has not yet published it.
        Format: YYYY-MM-DD
        """
        if not self.is_configured:
            logger.warning("FilingReconciliationEngine blocked: Valid SEC_USER_AGENT required.")
            return None

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year = dt.year
        quarter = f"QTR{(dt.month - 1) // 3 + 1}"
        file_date = dt.strftime("%Y%m%d")
        url = f"{self.DAILY_INDEX_BASE_URL}/{year}/{quarter}/master.{file_date}.idx"

        if self.rate_limiter:
            self.rate_limiter.acquire()

        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                logger.info(f"Daily master index not yet available for {date_str} (HTTP 404 at {url}).")
                return None
            else:
                logger.warning(f"Failed to fetch daily index for {date_str}: HTTP {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching daily master index for {date_str}: {e}")
            return None

    @classmethod
    def parse_daily_master_index(cls, index_content: str) -> List[IndexFilingRecord]:
        """
        Parses SEC daily master.idx or form.idx content.
        Filters strictly for Form 4 and Form 4/A.
        """
        records: List[IndexFilingRecord] = []
        if not index_content or not index_content.strip():
            return records

        lines = index_content.splitlines()
        header_passed = False

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Detect header transition
            if not header_passed:
                if trimmed.startswith("---") or trimmed.startswith("CIK|Company Name"):
                    header_passed = True
                continue

            # Pipe-delimited parsing
            if "|" in trimmed:
                parts = [p.strip() for p in trimmed.split("|")]
                if len(parts) >= 5:
                    raw_cik, comp_name, form_type, date_filed, filename = parts[:5]
                    if form_type.strip() in ("4", "4/A"):
                        acc_match = ACCESSION_EXTRACTION_REGEX.search(filename)
                        accession = acc_match.group(1) if acc_match else ""
                        if accession:
                            records.append(
                                IndexFilingRecord(
                                    accession_number=accession,
                                    issuer_cik=raw_cik.zfill(10),
                                    company_name=comp_name,
                                    form_type=form_type,
                                    filing_date=date_filed,
                                    source_filename=filename,
                                )
                            )
            else:
                acc_match = ACCESSION_EXTRACTION_REGEX.search(trimmed)
                if acc_match:
                    accession = acc_match.group(1)
                    if re.search(r"\b4\b|\b4/A\b", trimmed):
                        form = "4/A" if "4/A" in trimmed else "4"
                        records.append(
                            IndexFilingRecord(
                                accession_number=accession,
                                issuer_cik="0000000000",
                                company_name="INDEX_RECORD",
                                form_type=form,
                                filing_date=datetime.utcnow().strftime("%Y-%m-%d"),
                                source_filename=trimmed,
                            )
                        )

        return records

    def reconcile_interval(
        self,
        index_records: List[IndexFilingRecord],
        interval_start: str,
        interval_end: str,
        governed_ciks: Optional[Set[str]] = None,
        auto_record_gaps: bool = True,
    ) -> ReconciliationReport:
        """
        Performs mathematical set reconciliation between daily index filings
        and the local database store.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        reconcile_id = f"RECON_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Filter index records to governed universe if specified
        target_index_records = index_records
        if governed_ciks is not None:
            norm_ciks = {str(c).zfill(10) for c in governed_ciks}
            target_index_records = [r for r in index_records if r.issuer_cik in norm_ciks]

        expected_accessions = {r.accession_number for r in target_index_records}
        expected_count = len(expected_accessions)

        conn = self._get_connection()
        try:
            cur = conn.cursor()

            # Retrieve local accessions recorded in raw filings
            cur.execute("SELECT accession_number, disposition FROM form4_raw_filings")
            raw_rows = {row[0]: row[1] for row in cur.fetchall()}

            # Retrieve accessions with parsed transactions
            cur.execute("SELECT DISTINCT accession_number FROM form4_parsed_transactions")
            parsed_accessions = {row[0] for row in cur.fetchall()}

            # Compute set differences
            accounted_accessions = set(raw_rows.keys())
            missing_from_local = sorted(list(expected_accessions - accounted_accessions))

            # Unparsed filings (in raw filings, but 0 parsed transactions and not marked no-transactions)
            unparsed = []
            for acc in expected_accessions.intersection(accounted_accessions):
                disp = raw_rows.get(acc, "")
                if acc not in parsed_accessions and disp != "FETCHED_NO_RELEVANT_TRANSACTIONS":
                    unparsed.append(acc)

            # Newly discovered accessions (in index but previously missing)
            newly_discovered = missing_from_local

            # Failure reasons
            failures: List[str] = []
            if missing_from_local:
                failures.append(f"MISSING_INDEX_ACCESSIONS_IN_LOCAL_STORE: {len(missing_from_local)} filings")
            if unparsed:
                failures.append(f"UNPARSED_INDEX_ACCESSIONS_IN_LOCAL_STORE: {len(unparsed)} filings")

            # Record gaps into sm_source_gaps if requested
            open_gaps_count = 0
            if auto_record_gaps and (missing_from_local or unparsed):
                for acc in missing_from_local:
                    rec = next((r for r in target_index_records if r.accession_number == acc), None)
                    cik = rec.issuer_cik if rec else "UNKNOWN"
                    gap_key = f"{reconcile_id}:GAP_RECON_MISSING:{acc}"
                    gap_id = f"GAP-{hashlib.sha256(gap_key.encode('utf-8')).hexdigest()[:16]}"

                    cur.execute("""
                        INSERT OR REPLACE INTO sm_source_gaps (
                            gap_id, incremental_cycle_id, partition_id, issuer_cik, universe_build_id,
                            gap_type, window_start, window_end, accession_number, first_detected_at,
                            last_attempted_at, attempt_count, status, source_event_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """, (
                        gap_id, reconcile_id, f"PART-{cik}", cik, "GLOBAL_MARKET",
                        GapType.FETCH_FAILURE.value, interval_start, interval_end,
                        acc, now_str, now_str, GapStatus.OPEN.value, now_str
                    ))
                    open_gaps_count += 1

                for acc in unparsed:
                    rec = next((r for r in target_index_records if r.accession_number == acc), None)
                    cik = rec.issuer_cik if rec else "UNKNOWN"
                    gap_key = f"{reconcile_id}:GAP_RECON_UNPARSED:{acc}"
                    gap_id = f"GAP-{hashlib.sha256(gap_key.encode('utf-8')).hexdigest()[:16]}"

                    cur.execute("""
                        INSERT OR REPLACE INTO sm_source_gaps (
                            gap_id, incremental_cycle_id, partition_id, issuer_cik, universe_build_id,
                            gap_type, window_start, window_end, accession_number, first_detected_at,
                            last_attempted_at, attempt_count, status, source_event_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """, (
                        gap_id, reconcile_id, f"PART-{cik}", cik, "GLOBAL_MARKET",
                        GapType.PARSE_FAILURE.value, interval_start, interval_end,
                        acc, now_str, now_str, GapStatus.OPEN.value, now_str
                    ))
                    open_gaps_count += 1

                conn.commit()

            # Coverage status determination
            if not target_index_records and expected_count == 0:
                coverage_status = "COMPLETE" if len(raw_rows) > 0 else "UNKNOWN"
            elif len(failures) == 0:
                coverage_status = "COMPLETE"
            else:
                coverage_status = "PARTIAL"

            # Compute deterministic reconciliation digest
            summary_payload = {
                "reconciliation_id": reconcile_id,
                "interval_start": interval_start,
                "interval_end": interval_end,
                "expected_index_count": expected_count,
                "accounted_local_count": len(expected_accessions - set(missing_from_local)),
                "missing_count": len(missing_from_local),
                "unparsed_count": len(unparsed),
                "coverage_status": coverage_status,
                "failures": failures,
            }
            dumped = str(sorted(summary_payload.items()))
            recon_hash = hashlib.sha256(dumped.encode("utf-8")).hexdigest()

            report = ReconciliationReport(
                reconciliation_id=reconcile_id,
                reconciled_at=now_str,
                interval_start=interval_start,
                interval_end=interval_end,
                expected_index_count=expected_count,
                accounted_local_count=len(expected_accessions - set(missing_from_local)),
                missing_accessions=missing_from_local,
                unparsed_accessions=unparsed,
                newly_discovered_accessions=newly_discovered,
                open_gap_count=open_gaps_count,
                coverage_status=coverage_status,
                failure_reasons=failures,
                reconciliation_hash=recon_hash,
            )

            # Persist report row
            cur.execute("""
                INSERT OR REPLACE INTO sm_reconciliation_reports (
                    reconciliation_id, reconciled_at, interval_start, interval_end,
                    expected_index_count, accounted_local_count, missing_count, open_gap_count,
                    coverage_status, failure_reasons_json, reconciliation_hash, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.reconciliation_id,
                report.reconciled_at,
                report.interval_start,
                report.interval_end,
                report.expected_index_count,
                report.accounted_local_count,
                len(report.missing_accessions),
                report.open_gap_count,
                report.coverage_status,
                str(report.failure_reasons),
                report.reconciliation_hash,
                str(report.to_dict()),
            ))
            conn.commit()
            return report
        finally:
            conn.close()

    def reconcile_daily_index(
        self,
        date_str: str,
        governed_ciks: Optional[Set[str]] = None,
        auto_record_gaps: bool = True,
    ) -> Dict[str, Any]:
        """
        Reconciles filings for a specific calendar date against the daily master index.
        If the index is not yet published by the SEC, returns status RECONCILIATION_PENDING.
        """
        index_content = self.fetch_daily_master_index(date_str)
        if index_content is None:
            return {
                "status": "RECONCILIATION_PENDING",
                "date": date_str,
                "coverage_status": "RECONCILIATION_PENDING",
                "message": f"Daily master index not yet published or available for {date_str}.",
            }

        records = self.parse_daily_master_index(index_content)
        interval_start = f"{date_str}T00:00:00Z"
        interval_end = f"{date_str}T23:59:59Z"
        report = self.reconcile_interval(
            index_records=records,
            interval_start=interval_start,
            interval_end=interval_end,
            governed_ciks=governed_ciks,
            auto_record_gaps=auto_record_gaps,
        )
        return {
            "status": "COMPLETED",
            "date": date_str,
            "coverage_status": report.coverage_status,
            "report": report.to_dict(),
        }

    def get_latest_certified_watermark(self) -> Optional[str]:
        """Returns the latest interval_end where coverage_status == 'COMPLETE'."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT interval_end FROM sm_reconciliation_reports
                WHERE coverage_status = 'COMPLETE'
                ORDER BY interval_end DESC LIMIT 1
            """)
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def is_watermark_certified(self, watermark: str) -> bool:
        """Checks if there is an authoritative reconciliation report certifying completeness up to watermark."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT 1 FROM sm_reconciliation_reports
                WHERE coverage_status = 'COMPLETE' AND (interval_end >= ? OR reconciled_at >= ?)
                LIMIT 1
            """, (watermark, watermark))
            return cur.fetchone() is not None
        finally:
            conn.close()
