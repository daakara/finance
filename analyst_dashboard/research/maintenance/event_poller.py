"""
analyst_dashboard/research/maintenance/event_poller.py

SEC EDGAR Event Poller & Qualifying Filing Selector for Research Maintenance.
Polls primary-source submissions, validates filing forms, guards against misidentified
announcements, and manages durable event registration.
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import urllib.request
import urllib.error

from .contracts import (
    EventProcessingState,
    MaintenanceEvent,
    EventRegister,
)

logger = logging.getLogger(__name__)

SEC_USER_AGENT = "ARXTerminal Research Akara.david26@gmail.com"
QUALIFYING_PERIODIC_FORMS = {"10-Q", "10-K"}
MISIDENTIFIED_NON_PERIODIC_FORMS = {
    "8-K", "4", "144", "SCHEDULE 13G", "SCHEDULE 13G/A",
    "SC 13G", "SC 13D", "DEF 14A", "10-Q/A", "10-K/A"
}

SEC_EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_CACHE_TTL_SEC = 3600  # 1 hour bounded cache


def format_sec_primary_url(cik: str, accession_number: str, primary_doc: str) -> str:
    """Formats direct primary document URL on SEC EDGAR."""
    clean_cik = str(int(cik.strip()))
    acc_clean = accession_number.strip().replace("-", "")
    return f"{SEC_EDGAR_ARCHIVE_BASE}/{clean_cik}/{acc_clean}/{primary_doc}"


def format_sec_index_url(cik: str, accession_number: str) -> str:
    """Formats SEC EDGAR filing index URL."""
    clean_cik = str(int(cik.strip()))
    clean_acc = accession_number.strip()
    acc_clean = clean_acc.replace("-", "")
    return f"{SEC_EDGAR_ARCHIVE_BASE}/{clean_cik}/{acc_clean}/{clean_acc}-index.htm"


class SecEventPoller:
    """
    Manages SEC EDGAR event polling, form qualification, and event registration.
    Adheres strictly to fair-access rules, cache freshness policies, and fail-closed form validation.
    """

    def __init__(
        self,
        register: EventRegister,
        cache_dir: Optional[Union[str, Path]] = None,
        offline_mode: bool = False,
        min_request_interval_sec: float = 0.25,
        cache_ttl_sec: int = DEFAULT_CACHE_TTL_SEC,
    ):
        self.register = register
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.offline_mode = offline_mode
        self.min_request_interval_sec = min_request_interval_sec
        self.cache_ttl_sec = cache_ttl_sec
        self.last_request_time = 0.0
        self.user_agent = SEC_USER_AGENT
        self.last_submissions_receipt: Optional[Dict[str, Any]] = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)
        self.last_request_time = time.time()

    def fetch_submissions(self, cik: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch SEC submissions metadata for an entity.
        Enforces explicit bounded cache policy (TTL) and records cache freshness.
        """
        cik_clean = cik.strip().zfill(10)
        cache_file = self.cache_dir / f"CIK{cik_clean}_submissions.json" if self.cache_dir else None
        url = f"https://data.sec.gov/submissions/CIK{cik_clean}.json"

        # Check cache freshness
        if not force_refresh and cache_file and cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < self.cache_ttl_sec or self.offline_mode:
                try:
                    raw = cache_file.read_bytes()
                    data = json.loads(raw.decode("utf-8"))
                    self.last_submissions_receipt = {
                        "object_type": "SUBMISSIONS_JSON",
                        "url": url,
                        "retrieval_time_utc": datetime.utcfromtimestamp(cache_file.stat().st_mtime).isoformat() + "Z",
                        "byte_count": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "content_type": "application/json",
                        "retained_path": str(cache_file),
                        "retrieval_mode": "CACHED",
                    }
                    return data
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_file}: {e}. Refetching.")
        elif self.offline_mode and cache_file and cache_file.exists():
            try:
                raw = cache_file.read_bytes()
                data = json.loads(raw.decode("utf-8"))
                self.last_submissions_receipt = {
                    "object_type": "SUBMISSIONS_JSON",
                    "url": url,
                    "retrieval_time_utc": datetime.utcfromtimestamp(cache_file.stat().st_mtime).isoformat() + "Z",
                    "byte_count": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "content_type": "application/json",
                    "retained_path": str(cache_file),
                    "retrieval_mode": "CACHED",
                }
                return data
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}.")

        if self.offline_mode:
            raise FileNotFoundError(f"Offline mode enabled, but cache file {cache_file} was not found or invalid.")

        self._rate_limit()
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        req = urllib.request.Request(url, headers=headers)
        import gzip
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                if resp.info().get("Content-Encoding") == "gzip" or (len(raw) > 2 and raw[:2] == b"\x1f\x8b"):
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
                self.last_submissions_receipt = {
                    "object_type": "SUBMISSIONS_JSON",
                    "url": url,
                    "retrieval_time_utc": datetime.utcnow().isoformat() + "Z",
                    "byte_count": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "content_type": "application/json",
                    "retained_path": str(cache_file) if cache_file else None,
                    "retrieval_mode": "LIVE",
                }
        except Exception as e:
            logger.error(f"Failed to fetch submissions for CIK {cik_clean} from SEC: {e}")
            raise

        if cache_file:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return data

    def retrieve_filing_artifacts(
        self,
        cik: str,
        accession_number: str,
        primary_doc: str,
        force_refresh: bool = False,
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Dict[str, Any]]]:
        """
        Retrieves actual primary-source bytes for filing index and primary HTML document,
        computes SHA-256 digests directly from retrieved bytes, and returns:
            (index_sha256, primary_doc_sha256, source_receipts)
        If an artifact cannot be retrieved (e.g. offline mode without cache, or network error),
        records the limitation honestly without fabricating metadata.
        """
        clean_cik = str(int(cik.strip()))
        clean_acc = accession_number.strip().replace("-", "")
        receipts: Dict[str, Dict[str, Any]] = {}

        # 1. Submissions receipt
        if self.last_submissions_receipt:
            receipts["SUBMISSIONS_JSON"] = self.last_submissions_receipt

        # 2. Index HTML
        index_url = format_sec_index_url(cik, accession_number)
        index_cache = self.cache_dir / f"CIK{clean_cik.zfill(10)}_{clean_acc}_index.htm" if self.cache_dir else None
        index_bytes: Optional[bytes] = None
        index_mode = "LIVE"

        if not force_refresh and index_cache and index_cache.exists():
            file_age = time.time() - index_cache.stat().st_mtime
            if file_age < self.cache_ttl_sec or self.offline_mode:
                try:
                    index_bytes = index_cache.read_bytes()
                    index_mode = "CACHED"
                except Exception as e:
                    logger.warning(f"Error reading index cache {index_cache}: {e}")

        if index_bytes is None and not self.offline_mode:
            try:
                self._rate_limit()
                req = urllib.request.Request(index_url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})
                import gzip
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read()
                    if resp.info().get("Content-Encoding") == "gzip" or (len(raw) > 2 and raw[:2] == b"\x1f\x8b"):
                        raw = gzip.decompress(raw)
                    index_bytes = raw
                    index_mode = "LIVE"
                if index_cache:
                    index_cache.write_bytes(index_bytes)
            except Exception as e:
                logger.warning(f"Could not fetch index HTML from {index_url}: {e}")

        # Fallback to local cache if live fetch failed
        if index_bytes is None and index_cache and index_cache.exists():
            try:
                index_bytes = index_cache.read_bytes()
                index_mode = "CACHED"
            except Exception as e:
                logger.warning(f"Error reading index cache fallback {index_cache}: {e}")

        index_sha256: Optional[str] = None
        if index_bytes is not None:
            index_sha256 = hashlib.sha256(index_bytes).hexdigest()
            receipts["INDEX_HTML"] = {
                "object_type": "INDEX_HTML",
                "url": index_url,
                "retrieval_time_utc": datetime.utcnow().isoformat() + "Z",
                "byte_count": len(index_bytes),
                "sha256": index_sha256,
                "content_type": "text/html",
                "retained_path": str(index_cache) if index_cache else None,
                "retrieval_mode": index_mode,
            }
        else:
            receipts["INDEX_HTML"] = {
                "object_type": "INDEX_HTML",
                "url": index_url,
                "retrieval_time_utc": datetime.utcnow().isoformat() + "Z",
                "byte_count": 0,
                "sha256": None,
                "content_type": "text/html",
                "retrieval_mode": "UNAVAILABLE",
                "limitation": "Filing index HTML bytes unretrieved (offline mode or network error).",
            }

        # 3. Primary Doc HTML
        doc_url = format_sec_primary_url(cik, accession_number, primary_doc)
        doc_cache = self.cache_dir / f"CIK{clean_cik.zfill(10)}_{clean_acc}_{primary_doc}" if self.cache_dir else None
        doc_bytes: Optional[bytes] = None
        doc_mode = "LIVE"

        if not force_refresh and doc_cache and doc_cache.exists():
            file_age = time.time() - doc_cache.stat().st_mtime
            if file_age < self.cache_ttl_sec or self.offline_mode:
                try:
                    doc_bytes = doc_cache.read_bytes()
                    doc_mode = "CACHED"
                except Exception as e:
                    logger.warning(f"Error reading doc cache {doc_cache}: {e}")

        if doc_bytes is None and not self.offline_mode:
            try:
                self._rate_limit()
                req = urllib.request.Request(doc_url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})
                import gzip
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read()
                    if resp.info().get("Content-Encoding") == "gzip" or (len(raw) > 2 and raw[:2] == b"\x1f\x8b"):
                        raw = gzip.decompress(raw)
                    doc_bytes = raw
                    doc_mode = "LIVE"
                if doc_cache:
                    doc_cache.write_bytes(doc_bytes)
            except Exception as e:
                logger.warning(f"Could not fetch primary doc HTML from {doc_url}: {e}")

        # Fallback to local cache if live fetch failed
        if doc_bytes is None and doc_cache and doc_cache.exists():
            try:
                doc_bytes = doc_cache.read_bytes()
                doc_mode = "CACHED"
            except Exception as e:
                logger.warning(f"Error reading doc cache fallback {doc_cache}: {e}")

        doc_sha256: Optional[str] = None
        if doc_bytes is not None:
            doc_sha256 = hashlib.sha256(doc_bytes).hexdigest()
            receipts["PRIMARY_DOC_HTML"] = {
                "object_type": "PRIMARY_DOC_HTML",
                "url": doc_url,
                "retrieval_time_utc": datetime.utcnow().isoformat() + "Z",
                "byte_count": len(doc_bytes),
                "sha256": doc_sha256,
                "content_type": "text/html",
                "retained_path": str(doc_cache) if doc_cache else None,
                "retrieval_mode": doc_mode,
            }
        else:
            receipts["PRIMARY_DOC_HTML"] = {
                "object_type": "PRIMARY_DOC_HTML",
                "url": doc_url,
                "retrieval_time_utc": datetime.utcnow().isoformat() + "Z",
                "byte_count": 0,
                "sha256": None,
                "content_type": "text/html",
                "retrieval_mode": "UNAVAILABLE",
                "limitation": "Primary document HTML bytes unretrieved (offline mode or network error).",
            }

        return index_sha256, doc_sha256, receipts

    def discover_filing_event(
        self,
        symbol: str,
        cik: str,
        company_name: str,
        cutoff_date: str,
        target_accession: Optional[str] = None,
        is_replay: bool = False,
        force_refresh: bool = False,
    ) -> Tuple[Optional[MaintenanceEvent], List[str]]:
        """
        Scans filings for a qualifying periodic report (10-Q / 10-K).
        - Continues past completed duplicate events to locate another eligible event.
        - Keeps replay records separate from live events so replays cannot consume live slots.
        - Fails closed on non-qualifying announcements (8-K, Form 4) with clear reasons.
        """
        warnings: List[str] = []
        try:
            subs = self.fetch_submissions(cik, force_refresh=force_refresh)
        except Exception as e:
            return None, [f"RETRIEVAL_FAILED: Could not fetch submissions from SEC for {symbol} ({cik}): {e}"]

        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        acceptance_dts = recent.get("acceptanceDateTime", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        cik_clean = cik.strip().zfill(10)
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik_clean}.json"
        company_facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_clean}.json"

        # If targeting a specific accession number
        if target_accession:
            clean_acc = target_accession.strip()
            for i in range(len(accessions)):
                if accessions[i] == clean_acc:
                    form = forms[i].strip()
                    filing_date = filing_dates[i]
                    report_date = report_dates[i] if i < len(report_dates) else None
                    acc_dt = acceptance_dts[i] if i < len(acceptance_dts) else None
                    doc = primary_docs[i]

                    # Validate form type
                    if form in MISIDENTIFIED_NON_PERIODIC_FORMS:
                        msg = f"Rejected non-periodic filing: Form {form} (accn: {clean_acc}) is an earnings release or disclosure, not a Form 10-Q or 10-K."
                        logger.warning(msg)
                        rejected_event = MaintenanceEvent(
                            event_id=f"{'REPLAY-' if is_replay else 'EVT-'}{symbol.upper()}-{clean_acc}",
                            symbol=symbol.upper(),
                            cik=cik_clean,
                            company_name=company_name,
                            form=form,
                            accession_number=clean_acc,
                            filing_date=filing_date,
                            reporting_period=report_date or "UNAVAILABLE",
                            retrieval_time_utc=datetime.utcnow().isoformat() + "Z",
                            source_url=format_sec_primary_url(cik, clean_acc, doc),
                            primary_document=doc,
                            acceptance_datetime_utc=acc_dt,
                            index_url=format_sec_index_url(cik, clean_acc),
                            submissions_url=submissions_url,
                            company_facts_url=company_facts_url,
                            processing_state=EventProcessingState.REJECTED_MISIDENTIFIED,
                            is_replay=is_replay,
                            error_message=msg,
                        )
                        return rejected_event, [msg]

                    if form not in QUALIFYING_PERIODIC_FORMS:
                        msg = f"Filing {clean_acc} has non-qualifying form '{form}'."
                        return None, [msg]

                    # Valid periodic report found
                    url = format_sec_primary_url(cik, clean_acc, doc)
                    idx_hash, doc_hash, receipts = self.retrieve_filing_artifacts(cik, clean_acc, doc, force_refresh=force_refresh)
                    event = MaintenanceEvent(
                        event_id=f"{'REPLAY-' if is_replay else 'EVT-'}{symbol.upper()}-{clean_acc}",
                        symbol=symbol.upper(),
                        cik=cik_clean,
                        company_name=company_name,
                        form=form,
                        accession_number=clean_acc,
                        filing_date=filing_date,
                        reporting_period=report_date or filing_date,
                        retrieval_time_utc=datetime.utcnow().isoformat() + "Z",
                        source_url=url,
                        primary_document=doc,
                        acceptance_datetime_utc=acc_dt,
                        index_url=format_sec_index_url(cik, clean_acc),
                        index_sha256=idx_hash,
                        submissions_url=submissions_url,
                        company_facts_url=company_facts_url,
                        processing_state=EventProcessingState.REGISTERED,
                        is_replay=is_replay,
                        sha256=doc_hash,
                        source_receipts=receipts,
                    )
                    return event, warnings

            return None, [f"Target accession {target_accession} not found in submissions."]

        # Auto-scan for newer periodic filings postdating cutoff
        for i in range(len(forms)):
            form = forms[i].strip()
            filing_date = filing_dates[i]
            acc = accessions[i]
            report_date = report_dates[i] if i < len(report_dates) else None
            acc_dt = acceptance_dts[i] if i < len(acceptance_dts) else None
            doc = primary_docs[i]

            if filing_date <= cutoff_date and not is_replay:
                continue

            # If this accession is already completed in register, continue past it to find any other eligible filings!
            if not force_refresh and self.register.is_completed(symbol, acc, is_replay=is_replay):
                logger.info(f"Filing {acc} is already completed; checking next candidate filing.")
                continue

            if form in QUALIFYING_PERIODIC_FORMS:
                url = format_sec_primary_url(cik, acc, doc)
                idx_hash, doc_hash, receipts = self.retrieve_filing_artifacts(cik, acc, doc, force_refresh=force_refresh)
                event = MaintenanceEvent(
                    event_id=f"{'REPLAY-' if is_replay else 'EVT-'}{symbol.upper()}-{acc}",
                    symbol=symbol.upper(),
                    cik=cik_clean,
                    company_name=company_name,
                    form=form,
                    accession_number=acc,
                    filing_date=filing_date,
                    reporting_period=report_date or filing_date,
                    retrieval_time_utc=datetime.utcnow().isoformat() + "Z",
                    source_url=url,
                    primary_document=doc,
                    acceptance_datetime_utc=acc_dt,
                    index_url=format_sec_index_url(cik, acc),
                    index_sha256=idx_hash,
                    submissions_url=submissions_url,
                    company_facts_url=company_facts_url,
                    processing_state=EventProcessingState.REGISTERED,
                    is_replay=is_replay,
                    sha256=doc_hash,
                    source_receipts=receipts,
                )
                return event, warnings

        return None, [f"NO_NEW_FILING: No qualifying periodic filing (10-Q/10-K) postdating {cutoff_date} found for {symbol}."]
