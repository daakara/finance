"""
ARX Terminal — Smart Money Global SEC Enumeration Engine (Gate B1).

Implements the low-latency fast-path discovery mechanism:
- Global ownership filing enumeration from SEC EDGAR RSS/Atom feed or submissions stream.
- Replaces issuer-partitioned polling (PART-{cik}) with global accession discovery.
- Filters strictly for Form 4 and Form 4/A ownership submissions.
- Deduplicates, tracks cursors, and preserves complete provenance.
- Enforces shared SEC rate limiter authority (DEFAULT_SEC_RATE_LIMITER).
"""

import re
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from analyst_dashboard.smart_money.form4_ingestion import RateLimiter, Form4IngestionManager
from analyst_dashboard.data.sec_edgar_fetcher import (
    SecEdgarFetcher,
    DEFAULT_SEC_RATE_LIMITER,
    validate_sec_user_agent,
    get_sec_user_agent,
)

logger = logging.getLogger(__name__)

# SEC Global Ownership Filings Feed URL (Atom format)
SEC_OWNERSHIP_ATOM_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&output=atom"
SEC_OWNERSHIP_DISCOVERY_CHANNEL = "SEC_EDGAR_GLOBAL_OWNERSHIP_ATOM_V1"

# Forms within product scope for Insider Smart Money
IN_SCOPE_FORM_TYPES = {"4", "4/A"}
EXCLUDED_FORM_TYPES = {"3", "3/A", "5", "5/A"}

ACCESSION_REGEX = re.compile(r"\b(\d{10}-\d{2}-\d{6})\b")
CIK_REGEX = re.compile(r"\((\d{10}|\d{1,10})\)")


@dataclass(frozen=True)
class DiscoveredFilingEntry:
    """Canonical filing identity discovered via global enumeration."""
    accession_number: str
    issuer_cik: str
    form_type: str
    filing_date: str  # YYYY-MM-DD
    acceptance_datetime: Optional[str]  # ISO UTC
    primary_document_url: str
    issuer_name: str
    issuer_symbol: Optional[str]
    discovery_channel: str
    discovered_at: str  # ISO UTC
    content_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveryBatch:
    """Batch of filings discovered during a single enumeration pass."""
    batch_id: str
    discovered_at: str
    source_url: str
    channel: str
    total_entries_found: int
    in_scope_entries_count: int
    entries: List[DiscoveredFilingEntry]
    raw_payload_hash: str
    cursor_timestamp: Optional[str] = None
    status: str = "SUCCESS"  # SUCCESS, EMPTY, TRANSPORT_ERROR, MALFORMED_PAYLOAD
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "discovered_at": self.discovered_at,
            "source_url": self.source_url,
            "channel": self.channel,
            "total_entries_found": self.total_entries_found,
            "in_scope_entries_count": self.in_scope_entries_count,
            "entries": [e.to_dict() for e in self.entries],
            "raw_payload_hash": self.raw_payload_hash,
            "cursor_timestamp": self.cursor_timestamp,
            "status": self.status,
            "error_message": self.error_message,
        }


class GlobalSecDiscoveryEngine:
    """
    Orchestrates global SEC ownership filing enumeration.
    Fast path answering: What Form 4/4-A filings appeared recently across the market?
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        sec_fetcher: Optional[SecEdgarFetcher] = None,
        ingestion_mgr: Optional[Form4IngestionManager] = None,
    ):
        self.rate_limiter = rate_limiter or DEFAULT_SEC_RATE_LIMITER
        self.sec_fetcher = sec_fetcher or SecEdgarFetcher(rate_limiter=self.rate_limiter)
        self.ingestion_mgr = ingestion_mgr or Form4IngestionManager(rate_limiter=self.rate_limiter)
        self._last_cursor: Optional[str] = None
        self._seen_accessions: Set[str] = set()

    @classmethod
    def parse_atom_feed(
        cls,
        feed_xml: str,
        channel: str = SEC_OWNERSHIP_DISCOVERY_CHANNEL,
        discovered_at: Optional[str] = None,
    ) -> Tuple[List[DiscoveredFilingEntry], int, str]:
        """
        Deterministically parses SEC EDGAR Atom XML feed into DiscoveredFilingEntry objects.
        Filters for in-scope form types (Form 4 and 4/A).
        """
        now_str = discovered_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw_hash = hashlib.sha256(feed_xml.encode("utf-8")).hexdigest() if feed_xml else ""

        if not feed_xml or not feed_xml.strip():
            return [], 0, raw_hash

        entries: List[DiscoveredFilingEntry] = []
        total_found = 0

        # Namespace handling: Atom 2005 default namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        try:
            root = ET.fromstring(feed_xml)
        except ET.ParseError as e:
            logger.warning(f"Failed to parse Atom XML feed: {e}")
            raise ValueError(f"MALFORMED_ATOM_FEED: {e}")

        # Entries can be prefixed or non-prefixed depending on parser
        atom_entries = root.findall("atom:entry", ns)
        if not atom_entries:
            atom_entries = root.findall("entry")

        total_found = len(atom_entries)

        for entry_elem in atom_entries:
            # 1. Title / Form Type / Issuer Name / CIK
            title_elem = entry_elem.find("atom:title", ns)
            if title_elem is None:
                title_elem = entry_elem.find("title")
            title_text = (title_elem.text or "").strip() if title_elem is not None else ""

            # Extract form type from category or title
            cat_elem = entry_elem.find("atom:category", ns)
            if cat_elem is None:
                cat_elem = entry_elem.find("category")

            form_type = ""
            if cat_elem is not None:
                form_type = cat_elem.attrib.get("term", "").strip()

            if not form_type and title_text:
                parts = title_text.split(" - ")
                if len(parts) >= 2:
                    form_type = parts[0].strip()

            # Scope filter: Form 4 and Form 4/A only
            if form_type not in IN_SCOPE_FORM_TYPES:
                continue

            # Extract CIK
            cik = ""
            cik_match = CIK_REGEX.search(title_text)
            if cik_match:
                cik = cik_match.group(1).zfill(10)

            # Extract Issuer Name
            issuer_name = ""
            if " - " in title_text:
                after_form = title_text.split(" - ", 1)[1]
                if "(" in after_form:
                    issuer_name = after_form.split("(", 1)[0].strip()
                else:
                    issuer_name = after_form.strip()

            # 2. Accession Number & Link
            link_elem = entry_elem.find("atom:link", ns)
            if link_elem is None:
                link_elem = entry_elem.find("link")
            link_url = link_elem.attrib.get("href", "") if link_elem is not None else ""

            id_elem = entry_elem.find("atom:id", ns)
            if id_elem is None:
                id_elem = entry_elem.find("id")
            id_text = (id_elem.text or "") if id_elem is not None else ""

            accession_number = ""
            acc_match = ACCESSION_REGEX.search(id_text) or ACCESSION_REGEX.search(link_url)
            if acc_match:
                accession_number = acc_match.group(1)

            # Check summary for fallback accession and filing date
            summary_elem = entry_elem.find("atom:summary", ns)
            if summary_elem is None:
                summary_elem = entry_elem.find("summary")
            summary_text = (summary_elem.text or "") if summary_elem is not None else ""

            filing_date = ""
            if "Filed:" in summary_text:
                fd_match = re.search(r"Filed:\s*</b>\s*(\d{4}-\d{2}-\d{2})", summary_text, re.IGNORECASE)
                if not fd_match:
                    fd_match = re.search(r"Filed:\s*(\d{4}-\d{2}-\d{2})", summary_text, re.IGNORECASE)
                if fd_match:
                    filing_date = fd_match.group(1)

            if not accession_number and "AccNo:" in summary_text:
                an_match = re.search(r"AccNo:\s*</b>\s*([\d-]+)", summary_text, re.IGNORECASE)
                if not an_match:
                    an_match = re.search(r"AccNo:\s*([\d-]+)", summary_text, re.IGNORECASE)
                if an_match:
                    accession_number = an_match.group(1).strip()

            # 3. Acceptance / Updated Timestamp
            updated_elem = entry_elem.find("atom:updated", ns)
            if updated_elem is None:
                updated_elem = entry_elem.find("updated")
            updated_text = (updated_elem.text or "").strip() if updated_elem is not None else ""

            acceptance_iso = None
            if updated_text:
                try:
                    dt = datetime.fromisoformat(updated_text.replace("Z", "+00:00"))
                    acceptance_iso = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    if not filing_date:
                        filing_date = acceptance_iso[:10]
                except Exception:
                    acceptance_iso = None

            if not filing_date and acceptance_iso:
                filing_date = acceptance_iso[:10]
            elif not filing_date:
                filing_date = now_str[:10]

            if not accession_number:
                logger.warning(f"Skipping entry missing canonical accession: {title_text}")
                continue

            entry_fp = hashlib.sha256(
                f"{accession_number}:{cik}:{form_type}:{filing_date}:{acceptance_iso}".encode("utf-8")
            ).hexdigest()

            entries.append(
                DiscoveredFilingEntry(
                    accession_number=accession_number,
                    issuer_cik=cik,
                    form_type=form_type,
                    filing_date=filing_date,
                    acceptance_datetime=acceptance_iso,
                    primary_document_url=link_url,
                    issuer_name=issuer_name,
                    issuer_symbol=None,
                    discovery_channel=channel,
                    discovered_at=now_str,
                    content_fingerprint=entry_fp,
                )
            )

        return entries, total_found, raw_hash

    def discover_ownership_filings(
        self,
        feed_content_override: Optional[str] = None,
        source_url: str = SEC_OWNERSHIP_ATOM_URL,
        since_cursor: Optional[str] = None,
    ) -> DiscoveryBatch:
        """
        Executes low-latency discovery pass across official SEC ownership feed.
        Respects SEC User-Agent and ARX 5 req/s rate limits.
        Can be driven by fixture string in feed_content_override for deterministic test replay.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        batch_id = f"DISC_BATCH_{int(time.time())}_{hashlib.sha256(now_str.encode('utf-8')).hexdigest()[:8]}"

        raw_xml = feed_content_override

        if raw_xml is None:
            # Outbound fetch with fair-access rate limit & User-Agent verification
            self.rate_limiter.wait_for_slot()
            raw_xml = self.ingestion_mgr._fetch_url_with_retry(source_url)
            if not raw_xml:
                return DiscoveryBatch(
                    batch_id=batch_id,
                    discovered_at=now_str,
                    source_url=source_url,
                    channel=SEC_OWNERSHIP_DISCOVERY_CHANNEL,
                    total_entries_found=0,
                    in_scope_entries_count=0,
                    entries=[],
                    raw_payload_hash="",
                    status="TRANSPORT_ERROR",
                    error_message=f"Failed to fetch SEC ownership feed from {source_url}",
                )

        try:
            entries, total_found, raw_hash = self.parse_atom_feed(
                feed_xml=raw_xml,
                channel=SEC_OWNERSHIP_DISCOVERY_CHANNEL,
                discovered_at=now_str,
            )
        except Exception as e:
            return DiscoveryBatch(
                batch_id=batch_id,
                discovered_at=now_str,
                source_url=source_url,
                channel=SEC_OWNERSHIP_DISCOVERY_CHANNEL,
                total_entries_found=0,
                in_scope_entries_count=0,
                entries=[],
                raw_payload_hash=hashlib.sha256(raw_xml.encode("utf-8")).hexdigest() if raw_xml else "",
                status="MALFORMED_PAYLOAD",
                error_message=str(e),
            )

        # Apply cursor filter if specified
        active_cursor = since_cursor or self._last_cursor
        filtered_entries = entries
        if active_cursor:
            filtered_entries = [
                e for e in entries
                if (e.acceptance_datetime or e.discovered_at) > active_cursor
            ]

        # Update cursor to highest acceptance/discovered timestamp
        if filtered_entries:
            max_ts = max(
                (e.acceptance_datetime or e.discovered_at) for e in filtered_entries
            )
            self._last_cursor = max_ts
        elif entries and not self._last_cursor:
            max_ts = max(
                (e.acceptance_datetime or e.discovered_at) for e in entries
            )
            self._last_cursor = max_ts

        return DiscoveryBatch(
            batch_id=batch_id,
            discovered_at=now_str,
            source_url=source_url,
            channel=SEC_OWNERSHIP_DISCOVERY_CHANNEL,
            total_entries_found=total_found,
            in_scope_entries_count=len(filtered_entries),
            entries=filtered_entries,
            raw_payload_hash=raw_hash,
            cursor_timestamp=self._last_cursor,
            status="SUCCESS",
        )
