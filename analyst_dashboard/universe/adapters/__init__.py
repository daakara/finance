"""
analyst_dashboard/universe/adapters

Offline Source Parsers and Adapter-to-Validator Bridge for ARX Terminal universe inputs.
"""

from .bridge import AdapterValidatorBridge
from .matching import match_sources
from .models import (
    AdapterValidationReport,
    FieldCertificationStatus,
    MatchedUniverseCandidate,
    OpenFigiMatchStatus,
    ParsedOpenFigiResult,
    ParsedSourceFile,
    ParsedSourceRow,
    SourceFeedType,
)
from .nasdaq_trader import (
    NASDAQ_LISTED_HEADER,
    OTHER_LISTED_HEADER,
    parse_nasdaq_trader_directory,
)
from .openfigi import parse_openfigi_response
from .sec_edgar import (
    format_cik,
    parse_sec_company_tickers_exchange,
    parse_sec_submissions,
)

__all__ = [
    "NASDAQ_LISTED_HEADER",
    "OTHER_LISTED_HEADER",
    "AdapterValidationReport",
    "AdapterValidatorBridge",
    "FieldCertificationStatus",
    "MatchedUniverseCandidate",
    "OpenFigiMatchStatus",
    "ParsedOpenFigiResult",
    "ParsedSourceFile",
    "ParsedSourceRow",
    "SourceFeedType",
    "format_cik",
    "match_sources",
    "parse_nasdaq_trader_directory",
    "parse_openfigi_response",
    "parse_sec_company_tickers_exchange",
    "parse_sec_submissions",
]
