"""
analyst_dashboard/universe/acquisition

Bounded listing directory acquisition and source-population accounting package.

Exports:
- SourceConfig, CANONICAL_SOURCES, DEFAULT_USER_AGENT
- ListingSourceFetcher
- ArtifactStorage
- AcquisitionAttempt, AcquisitionOutcome, SourceParsedSummary, SourceAccountingReport
- run_acquisition_and_accounting
- save_accounting_reports, format_markdown_report
"""

from .config import (
    CANONICAL_SOURCES,
    DEFAULT_USER_AGENT,
    SourceConfig,
)
from .fetcher import ListingSourceFetcher
from .models import (
    AcquisitionAttempt,
    AcquisitionOutcome,
    SourceAccountingReport,
    SourceParsedSummary,
)
from .report import format_markdown_report, save_accounting_reports
from .runner import ListingAcquisitionRunner, run_acquisition_and_accounting
from .storage import ArtifactStorage

__all__ = [
    "SourceConfig",
    "CANONICAL_SOURCES",
    "DEFAULT_USER_AGENT",
    "ListingSourceFetcher",
    "ListingAcquisitionRunner",
    "ArtifactStorage",
    "AcquisitionAttempt",
    "AcquisitionOutcome",
    "SourceParsedSummary",
    "SourceAccountingReport",
    "run_acquisition_and_accounting",
    "save_accounting_reports",
    "format_markdown_report",
]
