"""
analyst_dashboard/universe/acquisition/runner.py

Orchestrates listing directory acquisition, storage preservation,
and offline parser replay across supported public exchange sources.

Strict Non-Inference Invariants:
- Acquisition is explicitly invoked; zero background daemons, hooks, or schedulers.
- Preserves raw response bytes before any normalization.
- Replays through existing offline parsers without manufacturing missing fields.
- Reports malformed rows and parsing anomalies explicitly.
- PRODUCTION_POPULATION_COMPLETENESS remains NOT_ESTABLISHED.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

from ..adapters.matching import match_sources_with_accounting
from ..adapters.models import ParsedSourceFile
from ..adapters.nasdaq_trader import parse_nasdaq_trader_directory
from ..adapters.sec_edgar import parse_sec_company_tickers_exchange
from .config import CANONICAL_SOURCES, SourceConfig
from .fetcher import ListingSourceFetcher
from .models import (
    AcquisitionAttempt,
    AcquisitionOutcome,
    SourceAccountingReport,
    SourceParsedSummary,
)
from .storage import ArtifactStorage

ACQUISITION_VERSION = "1.0.0"

STANDARD_COVERAGE_LIMITATIONS = [
    "Directory presence does not establish global instrument primary listing or trading suspension status.",
    "SEC company_tickers_exchange is issuer-level (CIK-based); multiple share classes share one CIK.",
    "OTC and unlisted securities are excluded from exchange directory feeds.",
    "Foreign issuers listed on US exchanges (ADRs, ordinary shares) have distinct regulatory reporting requirements.",
    "Market coverage denominator across full US-listed population is NOT_ESTABLISHED by directory join alone.",
]

STANDARD_UNRESOLVED_DECISIONS = [
    "Authoritative venue allowlist definition (NYSE, NASDAQ, AMEX, ARCA, BATS vs OTC/regional).",
    "Handling of share classes, dual listings, and warrants/units in scanner eligibility.",
    "Authoritative source for primary listing verification and active trading status.",
    "OpenFIGI integration model (batch licensing vs free public tier bounds).",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_acquisition_and_accounting(
    storage_dir: Path | str,
    sources: Mapping[str, SourceConfig] | None = None,
    session: requests.Session | None = None,
    fetch_fn: Callable[[SourceConfig], tuple[bytes | None, AcquisitionAttempt]] | None = None,
    offline_bytes: Mapping[str, bytes] | None = None,
    run_matching: bool = False,
) -> tuple[SourceAccountingReport, dict[str, ParsedSourceFile]]:
    """
    Executes bounded acquisition and parser replay across configured sources.

    Args:
        storage_dir: Local directory for raw artifact and sidecar metadata persistence.
        sources: Mapping of source_id to SourceConfig. Defaults to CANONICAL_SOURCES.
        session: Optional pre-configured requests.Session (e.g. for mocked testing).
        fetch_fn: Optional custom fetch function (takes SourceConfig, returns bytes, attempt).
        offline_bytes: Optional pre-loaded mapping of source_id -> raw bytes to bypass network.
        run_matching: Whether to run diagnostic cross-source matching if multi-source is parsed.

    Returns:
        tuple[SourceAccountingReport, dict[str, ParsedSourceFile]]
    """
    active_sources = sources or CANONICAL_SOURCES
    storage = ArtifactStorage(storage_dir)
    fetcher = ListingSourceFetcher(session=session) if fetch_fn is None else None

    attempts: list[AcquisitionAttempt] = []
    parsed_summaries: dict[str, SourceParsedSummary] = {}
    parsed_files: dict[str, ParsedSourceFile] = {}

    for source_id, config in active_sources.items():
        raw_bytes: bytes | None = None
        attempt: AcquisitionAttempt

        # 1. Fetching or offline replay
        if offline_bytes is not None and source_id in offline_bytes:
            raw_bytes = offline_bytes[source_id]
            start_utc = _utc_now_iso()
            attempt = AcquisitionAttempt(
                source_id=source_id,
                requested_url=config.url,
                final_url=config.url,
                retrieval_start_utc=start_utc,
                retrieval_end_utc=start_utc,
                http_status=200,
                content_type="application/json" if "json" in source_id else "text/plain",
                byte_length=len(raw_bytes),
                content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                acquisition_version=ACQUISITION_VERSION,
                outcome=AcquisitionOutcome.SUCCESS,
            )
        elif fetch_fn is not None:
            raw_bytes, attempt = fetch_fn(config)
        else:
            assert fetcher is not None
            raw_bytes, attempt = fetcher.fetch(config)

        # 2. Persist artifact atomically
        attempt = storage.save_artifact(attempt, raw_bytes)
        attempts.append(attempt)

        # 3. Offline Parser Replay
        if attempt.outcome == AcquisitionOutcome.SUCCESS and raw_bytes is not None:
            parsed_file: ParsedSourceFile

            if "sec" in source_id.lower():
                parsed_file = parse_sec_company_tickers_exchange(
                    content=raw_bytes,
                    source_identifier=config.url,
                    retrieved_at_utc=attempt.retrieval_start_utc,
                )
            else:
                # Nasdaq Trader directory (nasdaqlisted or otherlisted)
                parsed_file = parse_nasdaq_trader_directory(
                    content=raw_bytes,
                    filename_hint=config.url,
                    source_identifier=config.url,
                    retrieved_at_utc=attempt.retrieval_start_utc,
                )

            parsed_files[source_id] = parsed_file

            # Attribute publisher timestamp from footer metadata if present
            footer = parsed_file.metadata_fields.get("footer", {})
            footer_ts = None
            if isinstance(footer, dict):
                footer_ts = footer.get("file_creation_time_raw")
            elif "file_creation_time" in parsed_file.metadata_fields:
                footer_ts = parsed_file.metadata_fields.get("file_creation_time")

            if footer_ts:
                attempt.publisher_timestamp = footer_ts
                parsed_file.metadata_fields["file_creation_time"] = footer_ts

            is_usable = (parsed_file.data_row_count > 0 and parsed_file.malformed_row_count == 0)

            malformed_reasons = [
                r.malformed_reason
                for r in parsed_file.rows
                if r.is_malformed and r.malformed_reason
            ]

            parsed_summaries[source_id] = SourceParsedSummary(
                source_id=source_id,
                feed_type=parsed_file.feed_type.value,
                content_sha256=parsed_file.content_sha256,
                physical_line_count=parsed_file.physical_line_count,
                metadata_row_count=parsed_file.metadata_row_count,
                data_row_count=parsed_file.data_row_count,
                malformed_row_count=parsed_file.malformed_row_count,
                unique_record_count=parsed_file.unique_record_count,
                is_usable=is_usable,
                metadata_fields=parsed_file.metadata_fields,
                sample_malformed_reasons=malformed_reasons[:10],
            )

    # 4. Diagnostic Cross-Source Matching
    match_diagnostics: dict[str, Any] | None = None
    if run_matching:
        sec_file = next((f for k, f in parsed_files.items() if "sec" in k.lower()), None)
        dir_files = [f for k, f in parsed_files.items() if "sec" not in k.lower()]

        if sec_file is not None and dir_files:
            try:
                candidates, accounting = match_sources_with_accounting(
                    sec_file=sec_file,
                    directory_files=dir_files,
                )
                match_diagnostics = accounting.to_dict()
            except Exception as e:
                match_diagnostics = {"error": f"Match accounting failed: {e}"}

    # 5. Overall Outcome Determination
    success_count = sum(1 for a in attempts if a.outcome == AcquisitionOutcome.SUCCESS)
    usable_count = sum(1 for s in parsed_summaries.values() if s.is_usable)

    if success_count == len(active_sources) and usable_count == len(active_sources):
        overall_outcome = "SUCCESS"
    elif usable_count > 0:
        overall_outcome = "PARTIAL_ACQUISITION"
    else:
        overall_outcome = "FAILED"

    report = SourceAccountingReport(
        generated_at_utc=_utc_now_iso(),
        acquisition_version=ACQUISITION_VERSION,
        attempts=attempts,
        parsed_summaries=parsed_summaries,
        match_diagnostics=match_diagnostics,
        overall_outcome=overall_outcome,
        production_population_completeness="NOT_ESTABLISHED",
        coverage_limitations=STANDARD_COVERAGE_LIMITATIONS,
        unresolved_decisions=STANDARD_UNRESOLVED_DECISIONS,
    )

    return report, parsed_files


class ListingAcquisitionRunner:
    """Class wrapper orchestrating listing directory acquisition and accounting."""

    def __init__(
        self,
        output_dir: Path | str,
        fetcher: ListingSourceFetcher | None = None,
        session: requests.Session | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.fetcher = fetcher
        self.session = session

    def run(
        self,
        sources: Sequence[SourceConfig] | Mapping[str, SourceConfig] | None = None,
        run_matching: bool = False,
        offline_bytes: Mapping[str, bytes] | None = None,
    ) -> SourceAccountingReport:
        if sources is None:
            source_map = CANONICAL_SOURCES
        elif isinstance(sources, Mapping):
            source_map = dict(sources)
        else:
            source_map = {s.source_id: s for s in sources}

        fetch_fn = self.fetcher.fetch if self.fetcher else None
        sess = self.session or (getattr(self.fetcher, "_session", None) if self.fetcher else None)

        report, _ = run_acquisition_and_accounting(
            storage_dir=self.output_dir,
            sources=source_map,
            session=sess,
            fetch_fn=fetch_fn,
            offline_bytes=offline_bytes,
            run_matching=run_matching,
        )
        return report

