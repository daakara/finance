"""
scripts/research/etf_v2/ucits_acquisition_models.py

Domain models, outcome classifications, denominator conservation accounting,
and temporal semantics for Wave 4 UCITS Authority Acquisition.

Enforces:
- Closed 11-member AcquisitionOutcome enumeration
- Explicit failure-versus-absence invariants (failure != NO_MATCH)
- Denominator accounting and conservation equation
- Separation of DiscoveryCandidate from AuthoritySource
- Strict three-level identity attribute alignment
- Zero product-specific branching
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .global_identity_models import (
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    IdentifierType,
    IdentityStatus,
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)
from .ucits_provenance_models import (
    AUTHORIZED_DOCUMENT_CLASSES,
    UCITSSourceProvenanceRecord,
    WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS,
)


class AcquisitionOutcome(str, Enum):
    """
    Closed 11-member acquisition outcome enumeration.
    Distinguishes successful acquisition, absence, and distinct failure modes.
    """
    ACQUIRED = "ACQUIRED"
    NO_MATCH = "NO_MATCH"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_REQUEST = "INVALID_REQUEST"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    ACCESS_DENIED = "ACCESS_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    CONTENT_INVALID = "CONTENT_INVALID"
    PARSER_FAILURE = "PARSER_FAILURE"
    PROVENANCE_FAILURE = "PROVENANCE_FAILURE"


# Non-absence failure outcomes (must never be collapsed into NO_MATCH)
FAILURE_OUTCOMES: frozenset[AcquisitionOutcome] = frozenset({
    AcquisitionOutcome.UNSUPPORTED,
    AcquisitionOutcome.INVALID_REQUEST,
    AcquisitionOutcome.AUTHORITY_UNAVAILABLE,
    AcquisitionOutcome.ACCESS_DENIED,
    AcquisitionOutcome.RATE_LIMITED,
    AcquisitionOutcome.RETRIEVAL_FAILURE,
    AcquisitionOutcome.CONTENT_INVALID,
    AcquisitionOutcome.PARSER_FAILURE,
    AcquisitionOutcome.PROVENANCE_FAILURE,
})

RETRYABLE_OUTCOMES: frozenset[AcquisitionOutcome] = frozenset({
    AcquisitionOutcome.AUTHORITY_UNAVAILABLE,
    AcquisitionOutcome.RATE_LIMITED,
    AcquisitionOutcome.RETRIEVAL_FAILURE,
})


def assert_failure_is_not_no_match(outcome: AcquisitionOutcome) -> None:
    """
    Enforces the core semantic invariant that operational or transport failures
    must never be reported as absence of authority or fund non-existence.
    """
    if outcome in FAILURE_OUTCOMES and outcome == AcquisitionOutcome.NO_MATCH:
        raise ValueError(f"CRITICAL INVARIANT VIOLATION: Outcome {outcome} conflated with NO_MATCH.")


class TemporalScope(str, Enum):
    """Temporal classification for statutory documents and identity evidence."""
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    EFFECTIVE_FROM = "EFFECTIVE_FROM"
    EFFECTIVE_TO = "EFFECTIVE_TO"
    UNKNOWN_TEMPORAL_SCOPE = "UNKNOWN_TEMPORAL_SCOPE"


@dataclass(frozen=True)
class TemporalMetadata:
    """Immutable representation of authority-supported temporal evidence."""
    effective_date: Optional[str] = None          # YYYY-MM-DD
    publication_date: Optional[str] = None        # YYYY-MM-DD
    temporal_scope: TemporalScope = TemporalScope.UNKNOWN_TEMPORAL_SCOPE

    def __post_init__(self) -> None:
        if self.effective_date is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", self.effective_date):
            raise ValueError(f"effective_date must be YYYY-MM-DD, got {self.effective_date!r}")
        if self.publication_date is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", self.publication_date):
            raise ValueError(f"publication_date must be YYYY-MM-DD, got {self.publication_date!r}")


@dataclass(frozen=True)
class DiscoveryCandidate:
    """
    Represents an uncertified fund/share-class candidate from search or index seeds.
    Strictly segregated from authoritative evidence; cannot populate canonical models directly.
    """
    raw_identifier: str
    identifier_type_hint: Optional[str] = None
    indicative_domicile: Optional[str] = None
    discovery_source: str = "DISCOVERY_ONLY"
    discovery_timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    is_authoritative: bool = field(default=False, init=False)


@dataclass(frozen=True)
class AuthorityLocator:
    """Authoritative address specification for statutory document retrieval."""
    authority_id: str
    jurisdiction: str                            # "IE" or "LU"
    source_url: str
    document_type: str                           # In AUTHORIZED_DOCUMENT_CLASSES
    expected_mime: str = "application/pdf"

    def __post_init__(self) -> None:
        dom = self.jurisdiction.upper().strip()
        if dom not in WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS:
            raise UnsupportedJurisdictionError(f"Unsupported UCITS jurisdiction: {self.jurisdiction!r}")
        if self.document_type not in AUTHORIZED_DOCUMENT_CLASSES:
            raise ValueError(f"Unsupported document type: {self.document_type!r}")


@dataclass(frozen=True)
class AuthorityRequest:
    """Executable request specification dispatched to an authority adapter."""
    request_id: str
    share_class_isin: str
    locator: AuthorityLocator
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30


@dataclass(frozen=True)
class RawArtifact:
    """Immutable representation of raw bytes retrieved from an authority source."""
    artifact_id: str
    raw_bytes: bytes
    byte_length: int
    raw_sha256: str
    media_type: str
    http_status: int
    retrieval_timestamp: str                     # ISO 8601 UTC
    source_url: str
    content_encoding_decoded: bool = True


@dataclass(frozen=True)
class AcquisitionResult:
    """Result of executing an acquisition attempt against an authority source."""
    attempt_id: str
    outcome: AcquisitionOutcome
    request: AuthorityRequest
    artifact: Optional[RawArtifact] = None
    provenance_record: Optional[UCITSSourceProvenanceRecord] = None
    error_message: Optional[str] = None

    def is_success(self) -> bool:
        return self.outcome == AcquisitionOutcome.ACQUIRED and self.artifact is not None


@dataclass
class UCITSPopulationUniverseCounters:
    """
    Mathematical denominator conservation accounting for UCITS acquisition runs.
    Enforces:
      attempted = acquired + no_match + unsupported + invalid_request +
                  authority_unavailable + access_denied + rate_limited +
                  retrieval_failure + content_invalid + parser_failure + provenance_failure
    """
    known_discovery_universe: int = 0
    eligible_authority_query_universe: int = 0
    attempted_acquisition_universe: int = 0
    successfully_acquired_universe: int = 0
    canonicalized_universe: int = 0
    resolvable_universe: int = 0

    no_match_count: int = 0
    unsupported_count: int = 0
    invalid_request_count: int = 0
    authority_unavailable_count: int = 0
    access_denied_count: int = 0
    rate_limited_count: int = 0
    retrieval_failure_count: int = 0
    content_invalid_count: int = 0
    parser_failure_count: int = 0
    provenance_failure_count: int = 0

    def record_outcome(self, outcome: AcquisitionOutcome) -> None:
        """Increments the appropriate outcome counter and maintains totals."""
        self.attempted_acquisition_universe += 1
        if outcome == AcquisitionOutcome.ACQUIRED:
            self.successfully_acquired_universe += 1
        elif outcome == AcquisitionOutcome.NO_MATCH:
            self.no_match_count += 1
        elif outcome == AcquisitionOutcome.UNSUPPORTED:
            self.unsupported_count += 1
        elif outcome == AcquisitionOutcome.INVALID_REQUEST:
            self.invalid_request_count += 1
        elif outcome == AcquisitionOutcome.AUTHORITY_UNAVAILABLE:
            self.authority_unavailable_count += 1
        elif outcome == AcquisitionOutcome.ACCESS_DENIED:
            self.access_denied_count += 1
        elif outcome == AcquisitionOutcome.RATE_LIMITED:
            self.rate_limited_count += 1
        elif outcome == AcquisitionOutcome.RETRIEVAL_FAILURE:
            self.retrieval_failure_count += 1
        elif outcome == AcquisitionOutcome.CONTENT_INVALID:
            self.content_invalid_count += 1
        elif outcome == AcquisitionOutcome.PARSER_FAILURE:
            self.parser_failure_count += 1
        elif outcome == AcquisitionOutcome.PROVENANCE_FAILURE:
            self.provenance_failure_count += 1
        else:
            raise ValueError(f"Unknown acquisition outcome: {outcome}")

    def validate_conservation(self) -> Tuple[bool, str]:
        """
        Validates the strict conservation equation across all universe categories.
        Returns (True, 'CONSERVED') or raises AssertionError on mismatch.
        """
        for attr, val in self.__dict__.items():
            if isinstance(val, int) and val < 0:
                raise ValueError(f"Negative universe counter detected for {attr}: {val}")

        reconciled_sum = (
            self.successfully_acquired_universe
            + self.no_match_count
            + self.unsupported_count
            + self.invalid_request_count
            + self.authority_unavailable_count
            + self.access_denied_count
            + self.rate_limited_count
            + self.retrieval_failure_count
            + self.content_invalid_count
            + self.parser_failure_count
            + self.provenance_failure_count
        )

        if self.attempted_acquisition_universe != reconciled_sum:
            diff = self.attempted_acquisition_universe - reconciled_sum
            raise AssertionError(
                f"Denominator conservation violation: attempted ({self.attempted_acquisition_universe}) "
                f"!= sum of outcomes ({reconciled_sum}), delta={diff}"
            )

        if self.eligible_authority_query_universe > self.known_discovery_universe:
            raise AssertionError(
                f"Eligible universe ({self.eligible_authority_query_universe}) exceeds discovery universe ({self.known_discovery_universe})"
            )

        return (True, "CONSERVED")


@dataclass(frozen=True)
class ExtractedListingEvidence:
    """Venue listing evidence extracted from regulated market records."""
    ticker: str
    mic: str
    trading_currency: str
    is_primary_listing: bool = False

    def __post_init__(self) -> None:
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("Listing ticker must be a non-empty string.")
        if not self.mic or len(self.mic) != 4:
            raise ValueError(f"Venue MIC must be exactly 4 uppercase chars, got: {self.mic!r}")


@dataclass(frozen=True)
class ExtractedUCITSEvidence:
    """Structured identity evidence extracted from statutory fund documentation."""
    legal_umbrella_name: str
    sub_fund_legal_name: str
    legal_domicile: str                          # "IE" or "LU"
    regulatory_regime: str                       # "EU_UCITS"
    management_company: str
    share_class_legal_name: str
    share_class_isin: str
    distribution_policy: str                     # "ACCUMULATING" or "DISTRIBUTING"
    share_class_currency: str                    # ISO 4217
    wkn: Optional[str] = None
    listings: Tuple[ExtractedListingEvidence, ...] = field(default_factory=tuple)
    source_provenance_sha256: str = ""
    effective_date: str = ""
    temporal_scope: TemporalScope = TemporalScope.CURRENT

    # Invariants enforced across levels
    TICKER_IS_GLOBAL_CANONICAL_ID: bool = False
    WKN_GLOBAL_CANONICAL_ID: bool = False
    BROKER_ALIAS_IS_CANONICAL_AUTHORITY: bool = False
