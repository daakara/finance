"""
analyst_dashboard/research/maintenance/event_poller.py

SEC EDGAR Event Poller & Qualifying Filing Selector for Research Maintenance.
Polls primary-source submissions, validates filing forms, guards against misidentified
announcements, and manages durable event registration.
"""

from datetime import datetime
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

        # Check cache freshness
        if not force_refresh and cache_file and cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < self.cache_ttl_sec or self.offline_mode:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_file}: {e}. Refetching.")

        if self.offline_mode:
            raise FileNotFoundError(f"Offline mode enabled, but cache file {cache_file} was not found or invalid.")

        url = f"https://data.sec.gov/submissions/CIK{cik_clean}.json"
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
        except Exception as e:
            logger.error(f"Failed to fetch submissions for CIK {cik_clean} from SEC: {e}")
            raise

        if cache_file:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return data

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
                        submissions_url=submissions_url,
                        company_facts_url=company_facts_url,
                        processing_state=EventProcessingState.REGISTERED,
                        is_replay=is_replay,
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
            if self.register.is_completed(symbol, acc, is_replay=is_replay):
                logger.info(f"Filing {acc} is already completed; checking next candidate filing.")
                continue

            if form in QUALIFYING_PERIODIC_FORMS:
                url = format_sec_primary_url(cik, acc, doc)
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
                    submissions_url=submissions_url,
                    company_facts_url=company_facts_url,
                    processing_state=EventProcessingState.REGISTERED,
                    is_replay=is_replay,
                )
                return event, warnings

        return None, [f"NO_NEW_FILING: No qualifying periodic filing (10-Q/10-K) postdating {cutoff_date} found for {symbol}."]
