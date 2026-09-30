"""
scripts/research/etf_v2/global_identity_resolver_models.py

Domain models, closed input/output schemas, deterministic query normalization,
identifier validation, precedence ranking, and provenance models for the
Wave 3 GlobalETFIdentityResolver foundation.

Enforces:
- Three-tier identity separation (ETFInstrument != ETFShareClass != ETFListing)
- Closed 7-field ETFIdentityQuery input schema
- Closed 12-field ETFIdentityResolution output schema
- Closed 7-member ResolutionStatus enum (including first-class AUTHORITY_FAILURE)
- Closed 26-member ResolutionReason enum (zero free-text authority)
- Deterministic NFKC, whitespace, case-folding, and separator normalization
- ISO 6166 Mod-10 ISIN validation and WKN / MIC validation reuse from Wave 1
- Explicit identifier precedence and listing-context specificity ranking
- Zero timestamps, zero randomness, zero network/disk side effects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
import unicodedata

from .global_identifier_authority import (
    ISO_3166_1_ALPHA_2_CODES,
    WKN_GLOBAL_CANONICAL_ID,
    calculate_isin_check_digit,
    validate_isin,
    validate_mic,
    validate_wkn,
)
from .global_identity_models import (
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    IdentifierType as Wave1IdentifierType,
    InvalidIdentifierError,
    Jurisdiction,
)
from .identity_authority import IdentityAuthority

# Architectural Invariant Constants
WAVE_3_SCOPE: str = "GLOBAL_IDENTITY_RESOLUTION_DOMAIN_FOUNDATION"
CANONICAL_GLOBAL_IDENTITY_MODEL: str = "NORMALIZED_THREE_TIER_CORE_WITH_JURISDICTION_ADAPTERS"
INPUT_SCHEMA_CLOSED: bool = True
OUTPUT_SCHEMA_CLOSED: bool = True
RESOLUTION_CONFIDENCE_MODEL: str = "NOT_USED"
DETERMINISTIC_REASON_CODE_REQUIRED: bool = True
REASON_CODE_SCHEMA_CLOSED: bool = True
FREE_TEXT_REASON_AS_AUTHORITY: bool = False
FIRST_MATCH_WINS: bool = False
AMBIGUITY_FAILS_CLOSED: bool = True
NOT_FOUND_MEANS_GLOBAL_ABSENCE: bool = False
AMBIGUOUS_RESULT_SELECTS_FIRST_CANDIDATE: bool = False
ADAPTER_FAILURE_EQUALS_NOT_FOUND: bool = False
PARTIAL_AUTHORITY_FAILURE_FAILS_CLOSED: bool = True
RESOLVER_MUTATION_MODEL: str = "READ_ONLY"
RESOLUTION_IDEMPOTENT_FOR_IDENTICAL_AUTHORITY_STATE: bool = True
TIMESTAMP_IN_AUTHORITATIVE_RESOLUTION_OUTPUT: bool = False


class ResolverQueryClass(str, Enum):
    """Supported Wave 3 resolver query classes (Section 5)."""
    CANONICAL_INTERNAL_ID = "CANONICAL_INTERNAL_ID"
    ISIN = "ISIN"
    WKN = "WKN"
    TICKER_WITH_LISTING_CONTEXT = "TICKER_WITH_LISTING_CONTEXT"
    LEGAL_SHARE_CLASS_NAME = "LEGAL_SHARE_CLASS_NAME"
    NORMALIZED_SHARE_CLASS_NAME = "NORMALIZED_SHARE_CLASS_NAME"
    BROKER_ALIAS = "BROKER_ALIAS"


class ResolverIdentifierType(str, Enum):
    """
    Closed identifier types supported by ETFIdentityQuery and ETFIdentityResolution.
    Interoperates cleanly with Wave 1 IdentifierType without mutating Wave 1 contracts.
    """
    CANONICAL_INTERNAL_ID = "CANONICAL_INTERNAL_ID"
    ISIN = "ISIN"
    WKN = "WKN"
    TICKER = "TICKER"
    TICKER_WITH_LISTING_CONTEXT = "TICKER_WITH_LISTING_CONTEXT"
    LEGAL_SHARE_CLASS_NAME = "LEGAL_SHARE_CLASS_NAME"
    NORMALIZED_SHARE_CLASS_NAME = "NORMALIZED_SHARE_CLASS_NAME"
    BROKER_ALIAS = "BROKER_ALIAS"


# Export aliases matching readiness contract
IdentifierType = ResolverIdentifierType
QueryIdentifierType = ResolverIdentifierType


class NormalizationOutcome(str, Enum):
    """Deterministic normalization outcomes before authority resolution (Section 6.3)."""
    NORMALIZED = "NORMALIZED"
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    UNSUPPORTED_QUERY_TYPE = "UNSUPPORTED_QUERY_TYPE"


class ResolutionStatus(str, Enum):
    """
    Closed 7-member primary resolution status enum (Section 8.1 & Section 14).
    Separates semantic resolution outcomes from first-class authority execution failure.
    """
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    AUTHORITY_CONFLICT = "AUTHORITY_CONFLICT"
    UNSUPPORTED_QUERY_TYPE = "UNSUPPORTED_QUERY_TYPE"
    AUTHORITY_FAILURE = "AUTHORITY_FAILURE"


class ResolutionReason(str, Enum):
    """Closed 26-member deterministic reason-code enum (Section 19)."""
    EXACT_CANONICAL_ID_MATCH = "EXACT_CANONICAL_ID_MATCH"
    EXACT_ISIN_MATCH = "EXACT_ISIN_MATCH"
    EXACT_WKN_MATCH = "EXACT_WKN_MATCH"
    UNIQUE_TICKER_LISTING_MATCH = "UNIQUE_TICKER_LISTING_MATCH"
    UNIQUE_NAME_MATCH = "UNIQUE_NAME_MATCH"
    UNIQUE_BROKER_ALIAS_MATCH = "UNIQUE_BROKER_ALIAS_MATCH"
    MULTIPLE_SHARE_CLASSES = "MULTIPLE_SHARE_CLASSES"
    MULTIPLE_JURISDICTIONS = "MULTIPLE_JURISDICTIONS"
    MULTIPLE_LISTINGS_UNRESOLVED = "MULTIPLE_LISTINGS_UNRESOLVED"
    MULTIPLE_NAME_MATCHES = "MULTIPLE_NAME_MATCHES"
    MULTIPLE_ALIAS_MATCHES = "MULTIPLE_ALIAS_MATCHES"
    INVALID_ISIN_FORMAT = "INVALID_ISIN_FORMAT"
    INVALID_ISIN_CHECK_DIGIT = "INVALID_ISIN_CHECK_DIGIT"
    INVALID_WKN_FORMAT = "INVALID_WKN_FORMAT"
    INVALID_MIC = "INVALID_MIC"
    INVALID_QUERY = "INVALID_QUERY"
    NO_APPLICABLE_ADAPTER = "NO_APPLICABLE_ADAPTER"
    NO_AUTHORITY_MATCH = "NO_AUTHORITY_MATCH"
    CONFLICTING_AUTHORITATIVE_IDENTIFIERS = "CONFLICTING_AUTHORITATIVE_IDENTIFIERS"
    CONFLICTING_ADAPTER_IDENTITIES = "CONFLICTING_ADAPTER_IDENTITIES"
    CONFLICTING_LISTING_CONTEXT = "CONFLICTING_LISTING_CONTEXT"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    AUTHORITY_EXECUTION_FAILURE = "AUTHORITY_EXECUTION_FAILURE"
    AUTHORITY_RESPONSE_INVALID = "AUTHORITY_RESPONSE_INVALID"
    AUTHORITY_CAPABILITY_MISMATCH = "AUTHORITY_CAPABILITY_MISMATCH"
    INCOMPLETE_AUTHORITY_SET = "INCOMPLETE_AUTHORITY_SET"


class AdapterExecutionOutcome(str, Enum):
    """Explicit authority adapter execution outcomes (Section 13 & Section 14)."""
    COMPLETED_MATCH = "COMPLETED_MATCH"
    COMPLETED_NO_MATCH = "COMPLETED_NO_MATCH"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


@dataclass(frozen=True)
class ETFIdentityQuery:
    """
    Canonical closed 7-field resolver input schema (Section 10).
    No undeclared convenience fields are permitted.
    """
    raw_query: str
    identifier_type_hint: Optional[Any] = None
    jurisdiction_hint: Optional[str] = None
    mic_hint: Optional[str] = None
    venue_hint: Optional[str] = None
    currency_hint: Optional[str] = None
    broker_source_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        hint_val = (
            self.identifier_type_hint.value
            if isinstance(self.identifier_type_hint, Enum)
            else self.identifier_type_hint
        )
        return {
            "raw_query": self.raw_query,
            "identifier_type_hint": hint_val,
            "jurisdiction_hint": self.jurisdiction_hint,
            "mic_hint": self.mic_hint,
            "venue_hint": self.venue_hint,
            "currency_hint": self.currency_hint,
            "broker_source_hint": self.broker_source_hint,
        }


@dataclass(frozen=True)
class NormalizedETFIdentityQuery:
    """
    Internal normalized request representation retaining both raw_query and normalized_query.
    """
    outcome: NormalizationOutcome
    raw_query: str
    normalized_query: str
    inferred_identifier_type: Optional[ResolverIdentifierType]
    query_class: Optional[ResolverQueryClass]
    identifier_type_hint: Optional[ResolverIdentifierType]
    jurisdiction_hint: Optional[str]
    mic_hint: Optional[str]
    venue_hint: Optional[str]
    currency_hint: Optional[str]
    broker_source_hint: Optional[str]
    parsed_fields: Tuple[Tuple[ResolverIdentifierType, str], ...] = field(default_factory=tuple)
    failure_reason: Optional[ResolutionReason] = None


@dataclass(frozen=True)
class ProvenanceReference:
    """
    Immutable resolver-output provenance reference (Section 20).
    Contains zero timestamps to guarantee deterministic idempotency.
    """
    adapter_id: str
    authority_jurisdiction: str
    authority_source: str
    matched_identifier: str
    matched_identifier_type: str
    source_record_id: str
    source_document_hash: Optional[str] = None
    outcome: str = "COMPLETED_MATCH"

    def __post_init__(self) -> None:
        if not self.adapter_id or not isinstance(self.adapter_id, str):
            raise ValueError("ProvenanceReference.adapter_id must be a non-empty string.")
        if not self.authority_jurisdiction or not isinstance(self.authority_jurisdiction, str):
            raise ValueError("ProvenanceReference.authority_jurisdiction must be a non-empty string.")
        if not self.authority_source or not isinstance(self.authority_source, str):
            raise ValueError("ProvenanceReference.authority_source must be a non-empty string.")
        if not self.source_record_id or not isinstance(self.source_record_id, str):
            raise ValueError("ProvenanceReference.source_record_id must be a non-empty string.")

    def sort_key(self) -> Tuple[str, str, str, str, str, str, str]:
        return (
            self.adapter_id,
            self.authority_jurisdiction,
            self.matched_identifier_type,
            self.matched_identifier,
            self.source_record_id,
            self.source_document_hash or "",
            self.outcome,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "authority_jurisdiction": self.authority_jurisdiction,
            "authority_source": self.authority_source,
            "matched_identifier": self.matched_identifier,
            "matched_identifier_type": self.matched_identifier_type,
            "source_record_id": self.source_record_id,
            "source_document_hash": self.source_document_hash,
            "outcome": self.outcome,
        }


def sort_listings_deterministically(listings: Tuple[ETFListing, ...] | List[ETFListing]) -> Tuple[ETFListing, ...]:
    """
    Deduplicates and sorts ETFListing records deterministically by
    (venue_mic, trading_currency, ticker, listing_id).
    Raises ValueError on conflicting listings sharing a deduplication key.
    """
    by_dedup: Dict[Tuple[str, str, str], ETFListing] = {}
    for listing in listings:
        dkey = listing.deduplication_key()
        if dkey in by_dedup:
            existing = by_dedup[dkey]
            if existing.listing_id != listing.listing_id or existing.ticker != listing.ticker:
                raise ValueError(
                    f"Conflicting ETFListing records for deduplication key {dkey}: "
                    f"{existing.listing_id} vs {listing.listing_id}"
                )
        else:
            by_dedup[dkey] = listing

    return tuple(
        sorted(
            by_dedup.values(),
            key=lambda item: (item.venue_mic, item.trading_currency, item.ticker, item.listing_id),
        )
    )


def sort_provenance_deterministically(
    refs: Tuple[ProvenanceReference, ...] | List[ProvenanceReference],
) -> Tuple[ProvenanceReference, ...]:
    """Deduplicates and sorts ProvenanceReference records deterministically."""
    unique_map: Dict[Tuple[str, str, str, str, str, str, str], ProvenanceReference] = {}
    for ref in refs:
        unique_map[ref.sort_key()] = ref
    return tuple(unique_map[k] for k in sorted(unique_map.keys()))


@dataclass(frozen=True)
class ETFIdentityCandidate:
    """
    Represents a single authority-backed share-class resolution candidate.
    """
    instrument_identity: ETFInstrument
    share_class_identity: ETFShareClass
    listing_identities: Tuple[ETFListing, ...]
    matched_identifier: str
    matched_identifier_type: ResolverIdentifierType
    authority_adapter_id: str
    authority_jurisdiction: str
    provenance_references: Tuple[ProvenanceReference, ...]

    def __post_init__(self) -> None:
        if not self.provenance_references:
            raise ValueError("ETFIdentityCandidate requires at least one ProvenanceReference.")
        if self.share_class_identity.instrument_id != self.instrument_identity.canonical_instrument_id:
            raise ValueError(
                f"Share class instrument_id '{self.share_class_identity.instrument_id}' does not match "
                f"instrument '{self.instrument_identity.canonical_instrument_id}'."
            )

    def candidate_sort_key(self) -> Tuple[str, str, str, Tuple[bool, str], Tuple[bool, str], str]:
        """
        Exact 6-tuple deterministic ordering for AMBIGUOUS and AUTHORITY_CONFLICT candidates (Section 8.3):
        1. canonical internal ID (instrument_id / share_class_id), ascending
        2. jurisdiction, ascending
        3. canonical share-class identity, ascending
        4. MIC, ascending, with missing MIC after present MIC
        5. currency, ascending, with missing currency after present currency
        6. listing identity, ascending
        """
        sorted_listings = sort_listings_deterministically(self.listing_identities)
        first_listing = sorted_listings[0] if sorted_listings else None
        mic = first_listing.venue_mic if first_listing and first_listing.venue_mic else ""
        ccy = first_listing.trading_currency if first_listing and first_listing.trading_currency else ""
        lid = first_listing.listing_id if first_listing and first_listing.listing_id else ""
        return (
            self.instrument_identity.canonical_instrument_id,
            self.authority_jurisdiction,
            self.share_class_identity.share_class_id,
            (mic == "", mic),
            (ccy == "", ccy),
            lid,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_identity": self.instrument_identity.to_dict(),
            "share_class_identity": self.share_class_identity.to_dict(),
            "listing_identities": [l.to_dict() for l in self.listing_identities],
            "matched_identifier": self.matched_identifier,
            "matched_identifier_type": self.matched_identifier_type.value,
            "authority_adapter_id": self.authority_adapter_id,
            "authority_jurisdiction": self.authority_jurisdiction,
            "provenance_references": [p.to_dict() for p in self.provenance_references],
        }


def sort_candidates_deterministically(
    candidates: Tuple[ETFIdentityCandidate, ...] | List[ETFIdentityCandidate],
) -> Tuple[ETFIdentityCandidate, ...]:
    """Sorts ETFIdentityCandidate records by the exact 6-tuple Section 8.3 key."""
    return tuple(sorted(candidates, key=lambda c: c.candidate_sort_key()))


@dataclass(frozen=True)
class ETFIdentityResolution:
    """
    Canonical closed 12-field resolver output schema (Section 11).
    Enforces strict status-specific field and provenance invariants on construction.
    """
    resolution_status: ResolutionStatus
    canonical_internal_id: Optional[str]
    instrument_identity: Optional[ETFInstrument]
    share_class_identity: Optional[ETFShareClass]
    listing_identities: Tuple[ETFListing, ...]
    matched_identifier: Optional[str]
    matched_identifier_type: Optional[ResolverIdentifierType]
    authority_adapter_ids: Tuple[str, ...]
    authority_jurisdictions: Tuple[str, ...]
    ambiguity_candidates: Tuple[ETFIdentityCandidate, ...]
    provenance_references: Tuple[ProvenanceReference, ...]
    reason_code: ResolutionReason

    def __post_init__(self) -> None:
        if not isinstance(self.resolution_status, ResolutionStatus):
            raise ValueError(f"Invalid resolution_status: {self.resolution_status!r}")
        if not isinstance(self.reason_code, ResolutionReason):
            raise ValueError(f"Invalid reason_code: {self.reason_code!r}")

        if self.resolution_status == ResolutionStatus.RESOLVED:
            if not self.canonical_internal_id:
                raise ValueError("RESOLVED status requires non-null canonical_internal_id.")
            if self.instrument_identity is None:
                raise ValueError("RESOLVED status requires non-null instrument_identity.")
            if self.share_class_identity is None:
                raise ValueError("RESOLVED status requires non-null share_class_identity.")
            if self.canonical_internal_id != self.share_class_identity.share_class_id:
                raise ValueError(
                    "RESOLVED canonical_internal_id must equal share_class_identity.share_class_id."
                )
            if self.ambiguity_candidates:
                raise ValueError("RESOLVED status requires empty ambiguity_candidates.")
            if not self.matched_identifier or self.matched_identifier_type is None:
                raise ValueError("RESOLVED status requires matched_identifier and matched_identifier_type.")
            if not self.provenance_references:
                raise ValueError("RESOLVED status without authority provenance is prohibited.")
            for listing in self.listing_identities:
                if listing.share_class_id != self.share_class_identity.share_class_id:
                    raise ValueError(
                        f"Listing '{listing.listing_id}' does not belong to resolved share class "
                        f"'{self.share_class_identity.share_class_id}'."
                    )

        elif self.resolution_status == ResolutionStatus.AMBIGUOUS:
            if self.canonical_internal_id is not None or self.share_class_identity is not None:
                raise ValueError("AMBIGUOUS status requires null canonical_internal_id and share_class_identity.")
            if self.listing_identities:
                raise ValueError("AMBIGUOUS status requires empty listing_identities.")
            if len(self.ambiguity_candidates) < 2:
                raise ValueError("AMBIGUOUS status requires at least two ambiguity_candidates.")
            if not self.provenance_references:
                raise ValueError("AMBIGUOUS status without authority provenance is prohibited.")
            if self.instrument_identity is not None:
                inst_ids = {c.instrument_identity.canonical_instrument_id for c in self.ambiguity_candidates}
                if inst_ids != {self.instrument_identity.canonical_instrument_id}:
                    raise ValueError(
                        "AMBIGUOUS instrument_identity must be null unless all candidates share the same instrument."
                    )

        elif self.resolution_status == ResolutionStatus.NOT_FOUND:
            if (
                self.canonical_internal_id is not None
                or self.instrument_identity is not None
                or self.share_class_identity is not None
            ):
                raise ValueError("NOT_FOUND status requires null canonical identity fields.")
            if self.listing_identities or self.ambiguity_candidates:
                raise ValueError("NOT_FOUND status requires empty listing_identities and ambiguity_candidates.")

        elif self.resolution_status in (
            ResolutionStatus.INVALID_IDENTIFIER,
            ResolutionStatus.UNSUPPORTED_QUERY_TYPE,
            ResolutionStatus.AUTHORITY_FAILURE,
        ):
            if (
                self.canonical_internal_id is not None
                or self.instrument_identity is not None
                or self.share_class_identity is not None
            ):
                raise ValueError(f"{self.resolution_status.value} requires null canonical identity fields.")
            if self.listing_identities or self.ambiguity_candidates:
                raise ValueError(
                    f"{self.resolution_status.value} requires empty listing_identities and ambiguity_candidates."
                )

        elif self.resolution_status == ResolutionStatus.AUTHORITY_CONFLICT:
            if (
                self.canonical_internal_id is not None
                or self.instrument_identity is not None
                or self.share_class_identity is not None
            ):
                raise ValueError("AUTHORITY_CONFLICT requires null canonical identity fields.")
            if self.listing_identities:
                raise ValueError("AUTHORITY_CONFLICT requires empty listing_identities.")
            if not self.ambiguity_candidates:
                raise ValueError("AUTHORITY_CONFLICT requires non-empty conflicting ambiguity_candidates.")
            if not self.provenance_references:
                raise ValueError("AUTHORITY_CONFLICT without provenance is prohibited.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution_status": self.resolution_status.value,
            "canonical_internal_id": self.canonical_internal_id,
            "instrument_identity": self.instrument_identity.to_dict() if self.instrument_identity else None,
            "share_class_identity": self.share_class_identity.to_dict() if self.share_class_identity else None,
            "listing_identities": [l.to_dict() for l in self.listing_identities],
            "matched_identifier": self.matched_identifier,
            "matched_identifier_type": (
                self.matched_identifier_type.value if self.matched_identifier_type else None
            ),
            "authority_adapter_ids": list(self.authority_adapter_ids),
            "authority_jurisdictions": list(self.authority_jurisdictions),
            "ambiguity_candidates": [c.to_dict() for c in self.ambiguity_candidates],
            "provenance_references": [p.to_dict() for p in self.provenance_references],
            "reason_code": self.reason_code.value,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


@dataclass(frozen=True)
class AdapterCapabilities:
    """Declarative capability metadata advertised by an authority adapter (Section 9.1)."""
    adapter_id: str
    supported_jurisdictions: FrozenSet[str]
    supported_query_classes: FrozenSet[ResolverQueryClass]
    supported_identifier_namespaces: FrozenSet[str]
    supported_mics: FrozenSet[str] = field(default_factory=frozenset)
    supported_venues: FrozenSet[str] = field(default_factory=frozenset)
    supported_currencies: FrozenSet[str] = field(default_factory=frozenset)
    supported_broker_namespaces: FrozenSet[str] = field(default_factory=frozenset)
    authority_precedence: int = 1
    supports_provenance: bool = True


@dataclass(frozen=True)
class AdapterResolutionResult:
    """Result returned by an authority adapter's resolve() operation (Section 13)."""
    adapter_id: str
    outcome: AdapterExecutionOutcome
    candidates: Tuple[ETFIdentityCandidate, ...] = field(default_factory=tuple)
    provenance_references: Tuple[ProvenanceReference, ...] = field(default_factory=tuple)
    failure_reason: Optional[ResolutionReason] = None
    diagnostic_detail: str = ""


@dataclass(frozen=True)
class BrokerAliasRecord:
    """
    Generic namespace-scoped broker alias record (Section 15).
    Never treated as a global canonical identifier without a registered broker namespace.
    """
    broker_source_namespace: str
    raw_broker_alias: str
    normalized_broker_alias: str
    target_share_class_id: str
    target_listing_id: Optional[str]
    provenance_reference: ProvenanceReference
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    active: bool = True

    def __post_init__(self) -> None:
        if not self.broker_source_namespace or not self.broker_source_namespace.strip():
            raise ValueError("BrokerAliasRecord requires a non-empty broker_source_namespace.")
        if not self.raw_broker_alias or not self.raw_broker_alias.strip():
            raise ValueError("BrokerAliasRecord requires a non-empty raw_broker_alias.")
        if not self.target_share_class_id or not self.target_share_class_id.startswith("etfs:v1:"):
            raise ValueError("BrokerAliasRecord target_share_class_id must be a canonical etfs:v1: share-class ID.")


# ==============================================================================
# NORMALIZATION, VALIDATION & PRECEDENCE PRIMITIVES (SECTIONS 6, 7, 8)
# ==============================================================================

def normalize_text_nfkc(value: str) -> str:
    """
    Applies Section 6.1 general textual normalization:
    1. Unicode NFKC normalization
    2. Strip leading and trailing Unicode whitespace
    3. Collapse internal whitespace runs to a single ASCII space
    4. Unicode case folding
    """
    nfkc = unicodedata.normalize("NFKC", value)
    collapsed = re.sub(r"\s+", " ", nfkc.strip())
    return collapsed.casefold()


def normalize_name_for_share_class(value: str, normalized_mode: bool = False) -> str:
    """
    Normalizes legal or normalized share-class names without removing tranche-distinguishing
    qualifiers such as accumulating, distributing, hedged, unhedged, or currency tokens.
    """
    base = normalize_text_nfkc(value)
    if normalized_mode:
        # Canonicalize typographical dashes/quotes and strip only trademark symbols (®™©)
        cleaned = IdentityAuthority.normalize_for_matching(base)
        return normalize_text_nfkc(cleaned)
    return base


def normalize_broker_alias_text(value: str) -> str:
    """
    Normalizes broker alias text while preserving broker-specific separators and tokens.
    Never performs heuristic conversion to ISIN, WKN, or ticker.
    """
    return normalize_text_nfkc(value).upper()


def validate_and_normalize_isin_for_resolver(raw_isin: str) -> Tuple[Optional[str], Optional[ResolutionReason]]:
    """
    Validates and normalizes an ISIN string per Section 6.2 & Section 8:
    - Apply NFKC, trim, case-fold, remove only ASCII spaces and hyphens, uppercase.
    - Require 12 alphanumeric chars matching ^[A-Z]{2}[A-Z0-9]{9}\\d$ and ISO-3166 prefix.
    - Verify ISO 6166 Mod-10 Double-Add-Double check digit.
    Returns (normalized_isin, None) or (None, failure_reason).
    """
    folded = normalize_text_nfkc(raw_isin)
    # Remove only ASCII spaces and ASCII hyphens
    stripped = folded.replace(" ", "").replace("-", "").upper()
    if len(stripped) != 12 or not re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", stripped):
        return (None, ResolutionReason.INVALID_ISIN_FORMAT)

    country_prefix = stripped[:2]
    if country_prefix not in ISO_3166_1_ALPHA_2_CODES:
        return (None, ResolutionReason.INVALID_ISIN_FORMAT)

    expected_digit = calculate_isin_check_digit(stripped[:11])
    actual_digit = int(stripped[11])
    if actual_digit != expected_digit:
        return (None, ResolutionReason.INVALID_ISIN_CHECK_DIGIT)

    # Confirm Wave 1 validator agreement
    validate_isin(stripped, strict=True)
    return (stripped, None)


def validate_and_normalize_wkn_for_resolver(raw_wkn: str) -> Tuple[Optional[str], Optional[ResolutionReason]]:
    """
    Validates and normalizes a WKN string per Section 6.2 & Section 8:
    - Apply NFKC, trim, case-fold, remove only ASCII spaces and hyphens, uppercase.
    - Require exactly 6 alphanumeric characters.
    - WKN is never a global canonical ID (WKN_GLOBAL_CANONICAL_ID == False).
    """
    assert WKN_GLOBAL_CANONICAL_ID is False
    folded = normalize_text_nfkc(raw_wkn)
    stripped = folded.replace(" ", "").replace("-", "").upper()
    if len(stripped) != 6 or not re.match(r"^[A-Z0-9]{6}$", stripped):
        return (None, ResolutionReason.INVALID_WKN_FORMAT)
    validate_wkn(stripped, strict=True)
    return (stripped, None)


def validate_and_normalize_mic_for_resolver(raw_mic: str) -> Tuple[Optional[str], Optional[ResolutionReason]]:
    """Validates and normalizes a 4-character ISO 10383 MIC."""
    folded = normalize_text_nfkc(raw_mic).upper()
    if len(folded) != 4 or not re.match(r"^[A-Z0-9]{4}$", folded):
        return (None, ResolutionReason.INVALID_MIC)
    validate_mic(folded, strict=True)
    return (folded, None)


def validate_and_normalize_canonical_id(raw_cid: str) -> Tuple[Optional[str], Optional[ResolutionReason]]:
    """
    Validates and normalizes a Wave 1 canonical internal ID (etfi:v1:..., etfs:v1:..., etfl:v1:...).
    Preserves all colon and underscore separators.
    """
    folded = normalize_text_nfkc(raw_cid)
    parts = folded.split(":")
    if len(parts) < 4:
        return (None, ResolutionReason.INVALID_QUERY)

    prefix = parts[0].lower()
    version = parts[1].lower()
    if prefix not in ("etfi", "etfs", "etfl") or version != "v1":
        return (None, ResolutionReason.INVALID_QUERY)

    if prefix == "etfi":
        if len(parts) != 5:
            return (None, ResolutionReason.INVALID_QUERY)
        jur_str, dom_str, root_str = parts[2].upper(), parts[3].upper(), parts[4].upper()
        try:
            jur = Jurisdiction(jur_str)
            if not jur.is_supported():
                return (None, ResolutionReason.INVALID_QUERY)
        except ValueError:
            return (None, ResolutionReason.INVALID_QUERY)
        if dom_str not in ISO_3166_1_ALPHA_2_CODES or not re.match(r"^[A-Z0-9_-]+$", root_str):
            return (None, ResolutionReason.INVALID_QUERY)
        return (f"etfi:v1:{jur_str}:{dom_str}:{root_str}", None)

    if prefix == "etfs":
        if len(parts) != 4:
            return (None, ResolutionReason.INVALID_QUERY)
        scheme_str, id_str = parts[2].upper(), parts[3].upper()
        if scheme_str == "ISIN":
            norm_isin, err = validate_and_normalize_isin_for_resolver(id_str)
            if err is not None or norm_isin is None:
                return (None, err or ResolutionReason.INVALID_QUERY)
            return (f"etfs:v1:ISIN:{norm_isin}", None)
        if scheme_str == "SEC_CLASS_ID":
            if not re.match(r"^C\d{9}$", id_str):
                return (None, ResolutionReason.INVALID_QUERY)
            return (f"etfs:v1:SEC_CLASS_ID:{id_str}", None)
        if not re.match(r"^[A-Z0-9_-]+$", scheme_str) or not re.match(r"^[A-Z0-9_-]+$", id_str):
            return (None, ResolutionReason.INVALID_QUERY)
        return (f"etfs:v1:{scheme_str}:{id_str}", None)

    # prefix == "etfl"
    if len(parts) != 5:
        return (None, ResolutionReason.INVALID_QUERY)
    mic_str, ticker_str, ccy_str = parts[2].upper(), parts[3].upper(), parts[4].upper()
    norm_mic, mic_err = validate_and_normalize_mic_for_resolver(mic_str)
    if mic_err is not None or norm_mic is None:
        return (None, mic_err or ResolutionReason.INVALID_MIC)
    if not re.match(r"^[A-Z0-9\.\-_]+$", ticker_str) or not re.match(r"^[A-Z]{3}$", ccy_str):
        return (None, ResolutionReason.INVALID_QUERY)
    return (f"etfl:v1:{norm_mic}:{ticker_str}:{ccy_str}", None)


def parse_ticker_with_context(
    raw_ticker: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[ResolutionReason]]:
    """
    Parses a ticker string into (normalized_ticker, parsed_mic, parsed_currency, error_reason).
    - Preserves exchange/dot suffixes unless explicit composite delimiter (@ or :) is used.
    - Bare tickers retain zero inferred listing context.
    """
    folded = normalize_text_nfkc(raw_ticker).upper()
    if not folded:
        return (None, None, None, ResolutionReason.INVALID_QUERY)

    # Explicit TICKER@MIC syntax
    if "@" in folded:
        pieces = folded.split("@")
        if len(pieces) != 2 or not pieces[0] or not pieces[1]:
            return (None, None, None, ResolutionReason.INVALID_QUERY)
        t_part, mic_part = pieces[0].strip(), pieces[1].strip()
        norm_mic, mic_err = validate_and_normalize_mic_for_resolver(mic_part)
        if mic_err is not None:
            return (None, None, None, mic_err)
        if not re.match(r"^[A-Z0-9]+(?:[\.\-/][A-Z0-9]+)?$", t_part):
            return (None, None, None, ResolutionReason.INVALID_QUERY)
        return (t_part, norm_mic, None, None)

    # Explicit MIC:TICKER or MIC:TICKER:CURRENCY syntax
    if ":" in folded:
        pieces = [p.strip() for p in folded.split(":")]
        if len(pieces) == 2 and all(pieces):
            norm_mic, mic_err = validate_and_normalize_mic_for_resolver(pieces[0])
            if mic_err is not None:
                return (None, None, None, mic_err)
            if not re.match(r"^[A-Z0-9]+(?:[\.\-/][A-Z0-9]+)?$", pieces[1]):
                return (None, None, None, ResolutionReason.INVALID_QUERY)
            return (pieces[1], norm_mic, None, None)
        if len(pieces) == 3 and all(pieces):
            norm_mic, mic_err = validate_and_normalize_mic_for_resolver(pieces[0])
            if mic_err is not None:
                return (None, None, None, mic_err)
            if not re.match(r"^[A-Z0-9]+(?:[\.\-/][A-Z0-9]+)?$", pieces[1]):
                return (None, None, None, ResolutionReason.INVALID_QUERY)
            if not re.match(r"^[A-Z]{3}$", pieces[2]):
                return (None, None, None, ResolutionReason.INVALID_QUERY)
            return (pieces[1], norm_mic, pieces[2], None)
        return (None, None, None, ResolutionReason.INVALID_QUERY)

    # Standard ticker (bare or single internal separator such as BRK.B or ticker.suffix)
    if not re.match(r"^[A-Z0-9]{1,12}(?:[\.\-/][A-Z0-9]{1,6})?$", folded):
        return (None, None, None, ResolutionReason.INVALID_QUERY)

    return (folded, None, None, None)


def coerce_identifier_type_hint(hint: Any) -> Optional[ResolverIdentifierType]:
    """Coerces an optional identifier_type_hint to ResolverIdentifierType or raises InvalidIdentifierError."""
    if hint is None:
        return None
    if isinstance(hint, ResolverIdentifierType):
        if hint == ResolverIdentifierType.TICKER_WITH_LISTING_CONTEXT:
            return ResolverIdentifierType.TICKER
        return hint
    if isinstance(hint, Wave1IdentifierType):
        mapping = {
            Wave1IdentifierType.INTERNAL_INSTRUMENT_ID: ResolverIdentifierType.CANONICAL_INTERNAL_ID,
            Wave1IdentifierType.INTERNAL_SHARE_CLASS_ID: ResolverIdentifierType.CANONICAL_INTERNAL_ID,
            Wave1IdentifierType.INTERNAL_LISTING_ID: ResolverIdentifierType.CANONICAL_INTERNAL_ID,
            Wave1IdentifierType.ISIN: ResolverIdentifierType.ISIN,
            Wave1IdentifierType.WKN: ResolverIdentifierType.WKN,
            Wave1IdentifierType.TICKER: ResolverIdentifierType.TICKER,
        }
        if hint in mapping:
            return mapping[hint]
        raise InvalidIdentifierError(f"Unsupported Wave 1 IdentifierType hint for resolver: {hint}")
    if isinstance(hint, str):
        clean = normalize_text_nfkc(hint).upper()
        if clean == "TICKER_WITH_LISTING_CONTEXT":
            return ResolverIdentifierType.TICKER
        try:
            return ResolverIdentifierType(clean)
        except ValueError as exc:
            raise InvalidIdentifierError(f"Invalid identifier_type_hint: {hint!r}") from exc
    raise InvalidIdentifierError(f"Invalid identifier_type_hint type: {type(hint)}")


def precedence_rank_for_field(
    id_type: ResolverIdentifierType,
    has_mic: bool = False,
    has_venue: bool = False,
    has_currency: bool = False,
) -> int:
    """
    Returns deterministic precedence rank (lower integer = higher authority precedence)
    implementing Section 7:
      1. CANONICAL_INTERNAL_ID
      2. ISIN
      3. WKN
      4. ticker + MIC + venue + currency
      5. ticker + MIC + venue
      6. ticker + MIC (including ticker + MIC + currency)
      7. ticker + venue (including ticker + venue + currency)
      8. ticker + currency
      9. LEGAL_SHARE_CLASS_NAME
     10. NORMALIZED_SHARE_CLASS_NAME
     11. BROKER_ALIAS
     12. BARE_TICKER
    """
    if id_type == ResolverIdentifierType.CANONICAL_INTERNAL_ID:
        return 1
    if id_type == ResolverIdentifierType.ISIN:
        return 2
    if id_type == ResolverIdentifierType.WKN:
        return 3
    if id_type in (ResolverIdentifierType.TICKER, ResolverIdentifierType.TICKER_WITH_LISTING_CONTEXT):
        if has_mic and has_venue and has_currency:
            return 4
        if has_mic and has_venue:
            return 5
        if has_mic:
            return 6
        if has_venue:
            return 7
        if has_currency:
            return 8
        return 12  # BARE_TICKER
    if id_type == ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME:
        return 9
    if id_type == ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME:
        return 10
    if id_type == ResolverIdentifierType.BROKER_ALIAS:
        return 11
    return 99


def is_authoritative_identifier_type(id_type: ResolverIdentifierType) -> bool:
    """Returns True if id_type is an authoritative identifier (CANONICAL_INTERNAL_ID, ISIN, WKN)."""
    return id_type in (
        ResolverIdentifierType.CANONICAL_INTERNAL_ID,
        ResolverIdentifierType.ISIN,
        ResolverIdentifierType.WKN,
    )


_COMPOSITE_CLAUSE_KEYS: FrozenSet[str] = frozenset({
    "CANONICAL_INTERNAL_ID",
    "ISIN",
    "WKN",
    "TICKER",
    "LEGAL_SHARE_CLASS_NAME",
    "NORMALIZED_SHARE_CLASS_NAME",
    "BROKER_ALIAS",
    "MIC",
    "VENUE",
    "CURRENCY",
    "JURISDICTION",
    "BROKER_SOURCE",
})


def _looks_like_isin_candidate(text: str) -> bool:
    """
    Detects whether an unhinted query is attempting to supply an ISIN so that invalid
    ISIN format or check-digit errors fail closed instead of falling through to name/ticker.
    """
    stripped = text.replace(" ", "").replace("-", "").upper()
    if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", stripped) and stripped[:2] in ISO_3166_1_ALPHA_2_CODES:
        return True
    if (
        " " not in text
        and len(stripped) in (11, 13)
        and stripped[:2] in ISO_3166_1_ALPHA_2_CODES
        and any(ch.isdigit() for ch in stripped[2:])
        and re.match(r"^[A-Z]{2}[A-Z0-9]+$", stripped)
    ):
        return True
    return False


def normalize_identity_query(query: ETFIdentityQuery) -> NormalizedETFIdentityQuery:
    """
    Normalizes and validates an ETFIdentityQuery into a NormalizedETFIdentityQuery (Section 6 & 10).
    Never raises on malformed input; returns deterministic NormalizationOutcome and ResolutionReason.
    """
    if not isinstance(query, ETFIdentityQuery) or not isinstance(query.raw_query, str):
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.INVALID_IDENTIFIER,
            raw_query=str(getattr(query, "raw_query", "")),
            normalized_query="",
            inferred_identifier_type=None,
            query_class=None,
            identifier_type_hint=None,
            jurisdiction_hint=None,
            mic_hint=None,
            venue_hint=None,
            currency_hint=None,
            broker_source_hint=None,
            failure_reason=ResolutionReason.INVALID_QUERY,
        )

    raw_q = query.raw_query
    folded_q = normalize_text_nfkc(raw_q)
    if not folded_q:
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.INVALID_IDENTIFIER,
            raw_query=raw_q,
            normalized_query="",
            inferred_identifier_type=None,
            query_class=None,
            identifier_type_hint=None,
            jurisdiction_hint=None,
            mic_hint=None,
            venue_hint=None,
            currency_hint=None,
            broker_source_hint=None,
            failure_reason=ResolutionReason.INVALID_QUERY,
        )

    # 1. Coerce identifier_type_hint
    try:
        coerced_hint = coerce_identifier_type_hint(query.identifier_type_hint)
    except InvalidIdentifierError:
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.INVALID_IDENTIFIER,
            raw_query=raw_q,
            normalized_query=folded_q,
            inferred_identifier_type=None,
            query_class=None,
            identifier_type_hint=None,
            jurisdiction_hint=None,
            mic_hint=None,
            venue_hint=None,
            currency_hint=None,
            broker_source_hint=None,
            failure_reason=ResolutionReason.INVALID_QUERY,
        )

    # 2. Validate and normalize optional hints
    norm_jur: Optional[str] = None
    if query.jurisdiction_hint is not None:
        norm_jur = normalize_text_nfkc(query.jurisdiction_hint).upper()
        if not norm_jur:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=coerced_hint,
                query_class=None,
                identifier_type_hint=coerced_hint,
                jurisdiction_hint=None,
                mic_hint=None,
                venue_hint=None,
                currency_hint=None,
                broker_source_hint=None,
                failure_reason=ResolutionReason.INVALID_QUERY,
            )

    norm_mic: Optional[str] = None
    if query.mic_hint is not None:
        norm_mic, mic_err = validate_and_normalize_mic_for_resolver(query.mic_hint)
        if mic_err is not None:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=coerced_hint,
                query_class=None,
                identifier_type_hint=coerced_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=None,
                venue_hint=None,
                currency_hint=None,
                broker_source_hint=None,
                failure_reason=mic_err,
            )

    norm_venue: Optional[str] = None
    if query.venue_hint is not None:
        norm_venue = normalize_text_nfkc(query.venue_hint)
        if not norm_venue:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=coerced_hint,
                query_class=None,
                identifier_type_hint=coerced_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=None,
                currency_hint=None,
                broker_source_hint=None,
                failure_reason=ResolutionReason.INVALID_QUERY,
            )

    norm_ccy: Optional[str] = None
    if query.currency_hint is not None:
        norm_ccy = normalize_text_nfkc(query.currency_hint).upper()
        if not re.match(r"^[A-Z]{3}$", norm_ccy):
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=coerced_hint,
                query_class=None,
                identifier_type_hint=coerced_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=norm_venue,
                currency_hint=None,
                broker_source_hint=None,
                failure_reason=ResolutionReason.INVALID_QUERY,
            )

    norm_broker_source: Optional[str] = None
    if query.broker_source_hint is not None:
        norm_broker_source = normalize_text_nfkc(query.broker_source_hint).upper()
        if not norm_broker_source:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=coerced_hint,
                query_class=None,
                identifier_type_hint=coerced_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=norm_venue,
                currency_hint=norm_ccy,
                broker_source_hint=None,
                failure_reason=ResolutionReason.INVALID_QUERY,
            )

    # 3. Check for structured multi-field clause syntax (KEY1=VAL1;KEY2=VAL2) when unhinted
    if coerced_hint is None and ";" in raw_q and "=" in raw_q:
        raw_clauses = [c.strip() for c in raw_q.split(";") if c.strip()]
        clause_pairs: List[Tuple[str, str]] = []
        all_valid_keys = True
        for clause in raw_clauses:
            if "=" not in clause:
                all_valid_keys = False
                break
            k, v = clause.split("=", 1)
            k_up = normalize_text_nfkc(k).upper()
            if k_up not in _COMPOSITE_CLAUSE_KEYS or not v.strip():
                all_valid_keys = False
                break
            clause_pairs.append((k_up, v.strip()))

        if all_valid_keys and clause_pairs:
            parsed_id_fields: List[Tuple[ResolverIdentifierType, str]] = []
            for k_up, v_raw in clause_pairs:
                if k_up == "MIC":
                    m_val, m_err = validate_and_normalize_mic_for_resolver(v_raw)
                    if m_err is not None:
                        return NormalizedETFIdentityQuery(
                            outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                            raw_query=raw_q,
                            normalized_query=folded_q,
                            inferred_identifier_type=None,
                            query_class=None,
                            identifier_type_hint=None,
                            jurisdiction_hint=norm_jur,
                            mic_hint=None,
                            venue_hint=norm_venue,
                            currency_hint=norm_ccy,
                            broker_source_hint=norm_broker_source,
                            failure_reason=m_err,
                        )
                    norm_mic = m_val
                elif k_up == "VENUE":
                    norm_venue = normalize_text_nfkc(v_raw)
                elif k_up == "CURRENCY":
                    c_val = normalize_text_nfkc(v_raw).upper()
                    if not re.match(r"^[A-Z]{3}$", c_val):
                        return NormalizedETFIdentityQuery(
                            outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                            raw_query=raw_q,
                            normalized_query=folded_q,
                            inferred_identifier_type=None,
                            query_class=None,
                            identifier_type_hint=None,
                            jurisdiction_hint=norm_jur,
                            mic_hint=norm_mic,
                            venue_hint=norm_venue,
                            currency_hint=None,
                            broker_source_hint=norm_broker_source,
                            failure_reason=ResolutionReason.INVALID_QUERY,
                        )
                    norm_ccy = c_val
                elif k_up == "JURISDICTION":
                    norm_jur = normalize_text_nfkc(v_raw).upper()
                elif k_up == "BROKER_SOURCE":
                    norm_broker_source = normalize_text_nfkc(v_raw).upper()
                else:
                    sub_hint = ResolverIdentifierType(k_up)
                    sub_norm = _normalize_single_field_by_type(
                        raw_q=v_raw,
                        target_type=sub_hint,
                        norm_jur=norm_jur,
                        norm_mic=norm_mic,
                        norm_venue=norm_venue,
                        norm_ccy=norm_ccy,
                        norm_broker_source=norm_broker_source,
                    )
                    if sub_norm.outcome != NormalizationOutcome.NORMALIZED or sub_norm.inferred_identifier_type is None:
                        return NormalizedETFIdentityQuery(
                            outcome=sub_norm.outcome,
                            raw_query=raw_q,
                            normalized_query=folded_q,
                            inferred_identifier_type=sub_hint,
                            query_class=sub_norm.query_class,
                            identifier_type_hint=None,
                            jurisdiction_hint=norm_jur,
                            mic_hint=norm_mic,
                            venue_hint=norm_venue,
                            currency_hint=norm_ccy,
                            broker_source_hint=norm_broker_source,
                            failure_reason=sub_norm.failure_reason,
                        )
                    parsed_id_fields.append((sub_norm.inferred_identifier_type, sub_norm.normalized_query))

            if not parsed_id_fields:
                return NormalizedETFIdentityQuery(
                    outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                    raw_query=raw_q,
                    normalized_query=folded_q,
                    inferred_identifier_type=None,
                    query_class=None,
                    identifier_type_hint=None,
                    jurisdiction_hint=norm_jur,
                    mic_hint=norm_mic,
                    venue_hint=norm_venue,
                    currency_hint=norm_ccy,
                    broker_source_hint=norm_broker_source,
                    failure_reason=ResolutionReason.INVALID_QUERY,
                )

            # Sort parsed fields by precedence rank ascending
            sorted_fields = tuple(
                sorted(
                    parsed_id_fields,
                    key=lambda item: (
                        precedence_rank_for_field(
                            item[0],
                            has_mic=norm_mic is not None,
                            has_venue=norm_venue is not None,
                            has_currency=norm_ccy is not None,
                        ),
                        item[0].value,
                        item[1],
                    ),
                )
            )
            primary_type, primary_val = sorted_fields[0]
            primary_class = (
                ResolverQueryClass.TICKER_WITH_LISTING_CONTEXT
                if primary_type in (ResolverIdentifierType.TICKER, ResolverIdentifierType.TICKER_WITH_LISTING_CONTEXT)
                else ResolverQueryClass(primary_type.value)
            )
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.NORMALIZED,
                raw_query=raw_q,
                normalized_query=primary_val,
                inferred_identifier_type=primary_type,
                query_class=primary_class,
                identifier_type_hint=None,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=norm_venue,
                currency_hint=norm_ccy,
                broker_source_hint=norm_broker_source,
                parsed_fields=sorted_fields,
                failure_reason=None,
            )

    # 4. If explicit identifier_type_hint is supplied, validate strictly under that type (zero fallthrough)
    if coerced_hint is not None:
        return _normalize_single_field_by_type(
            raw_q=raw_q,
            target_type=coerced_hint,
            norm_jur=norm_jur,
            norm_mic=norm_mic,
            norm_venue=norm_venue,
            norm_ccy=norm_ccy,
            norm_broker_source=norm_broker_source,
            explicit_hint=coerced_hint,
        )

    # 5. Deterministic unhinted identifier-type inference
    # 5a. Canonical internal ID prefix (etfi:v1:, etfs:v1:, etfl:v1:)
    if re.match(r"^(?:etfi|etfs|etfl):v\d+:", folded_q, re.IGNORECASE):
        return _normalize_single_field_by_type(
            raw_q=raw_q,
            target_type=ResolverIdentifierType.CANONICAL_INTERNAL_ID,
            norm_jur=norm_jur,
            norm_mic=norm_mic,
            norm_venue=norm_venue,
            norm_ccy=norm_ccy,
            norm_broker_source=norm_broker_source,
        )

    # 5b. ISIN candidate detection (12-char ISO-3166 or malformed 11/13-char ISIN attempt)
    if _looks_like_isin_candidate(folded_q):
        return _normalize_single_field_by_type(
            raw_q=raw_q,
            target_type=ResolverIdentifierType.ISIN,
            norm_jur=norm_jur,
            norm_mic=norm_mic,
            norm_venue=norm_venue,
            norm_ccy=norm_ccy,
            norm_broker_source=norm_broker_source,
        )

    # 5c. Broker alias when broker_source_hint is explicitly supplied
    if norm_broker_source is not None:
        return _normalize_single_field_by_type(
            raw_q=raw_q,
            target_type=ResolverIdentifierType.BROKER_ALIAS,
            norm_jur=norm_jur,
            norm_mic=norm_mic,
            norm_venue=norm_venue,
            norm_ccy=norm_ccy,
            norm_broker_source=norm_broker_source,
        )

    # 5d. WKN candidate (exact 6 alphanumeric chars after stripping ASCII space/hyphen, containing at least one digit, when no listing hint forces ticker)
    stripped_alnum = folded_q.replace(" ", "").replace("-", "").upper()
    if (
        len(stripped_alnum) == 6
        and re.match(r"^[A-Z0-9]{6}$", stripped_alnum)
        and any(c.isdigit() for c in stripped_alnum)
        and norm_mic is None
        and norm_venue is None
        and norm_ccy is None
    ):
        return _normalize_single_field_by_type(
            raw_q=raw_q,
            target_type=ResolverIdentifierType.WKN,
            norm_jur=norm_jur,
            norm_mic=norm_mic,
            norm_venue=norm_venue,
            norm_ccy=norm_ccy,
            norm_broker_source=norm_broker_source,
        )

    # 5e. Multi-word share-class / fund name
    if " " in folded_q:
        return _normalize_single_field_by_type(
            raw_q=raw_q,
            target_type=ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME,
            norm_jur=norm_jur,
            norm_mic=norm_mic,
            norm_venue=norm_venue,
            norm_ccy=norm_ccy,
            norm_broker_source=norm_broker_source,
        )

    # 5f. Single-token or composite-delimited ticker
    return _normalize_single_field_by_type(
        raw_q=raw_q,
        target_type=ResolverIdentifierType.TICKER,
        norm_jur=norm_jur,
        norm_mic=norm_mic,
        norm_venue=norm_venue,
        norm_ccy=norm_ccy,
        norm_broker_source=norm_broker_source,
    )


def _normalize_single_field_by_type(
    raw_q: str,
    target_type: ResolverIdentifierType,
    norm_jur: Optional[str],
    norm_mic: Optional[str],
    norm_venue: Optional[str],
    norm_ccy: Optional[str],
    norm_broker_source: Optional[str],
    explicit_hint: Optional[ResolverIdentifierType] = None,
) -> NormalizedETFIdentityQuery:
    """Validates and normalizes a single query string under the specified target_type."""
    folded_q = normalize_text_nfkc(raw_q)

    if target_type == ResolverIdentifierType.CANONICAL_INTERNAL_ID:
        norm_cid, err = validate_and_normalize_canonical_id(raw_q)
        if err is not None or norm_cid is None:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=ResolverIdentifierType.CANONICAL_INTERNAL_ID,
                query_class=ResolverQueryClass.CANONICAL_INTERNAL_ID,
                identifier_type_hint=explicit_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=norm_venue,
                currency_hint=norm_ccy,
                broker_source_hint=norm_broker_source,
                failure_reason=err or ResolutionReason.INVALID_QUERY,
            )
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.NORMALIZED,
            raw_query=raw_q,
            normalized_query=norm_cid,
            inferred_identifier_type=ResolverIdentifierType.CANONICAL_INTERNAL_ID,
            query_class=ResolverQueryClass.CANONICAL_INTERNAL_ID,
            identifier_type_hint=explicit_hint,
            jurisdiction_hint=norm_jur,
            mic_hint=norm_mic,
            venue_hint=norm_venue,
            currency_hint=norm_ccy,
            broker_source_hint=norm_broker_source,
            parsed_fields=((ResolverIdentifierType.CANONICAL_INTERNAL_ID, norm_cid),),
        )

    if target_type == ResolverIdentifierType.ISIN:
        norm_isin, err = validate_and_normalize_isin_for_resolver(raw_q)
        if err is not None or norm_isin is None:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=ResolverIdentifierType.ISIN,
                query_class=ResolverQueryClass.ISIN,
                identifier_type_hint=explicit_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=norm_venue,
                currency_hint=norm_ccy,
                broker_source_hint=norm_broker_source,
                failure_reason=err or ResolutionReason.INVALID_ISIN_FORMAT,
            )
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.NORMALIZED,
            raw_query=raw_q,
            normalized_query=norm_isin,
            inferred_identifier_type=ResolverIdentifierType.ISIN,
            query_class=ResolverQueryClass.ISIN,
            identifier_type_hint=explicit_hint,
            jurisdiction_hint=norm_jur,
            mic_hint=norm_mic,
            venue_hint=norm_venue,
            currency_hint=norm_ccy,
            broker_source_hint=norm_broker_source,
            parsed_fields=((ResolverIdentifierType.ISIN, norm_isin),),
        )

    if target_type == ResolverIdentifierType.WKN:
        norm_wkn, err = validate_and_normalize_wkn_for_resolver(raw_q)
        if err is not None or norm_wkn is None:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=ResolverIdentifierType.WKN,
                query_class=ResolverQueryClass.WKN,
                identifier_type_hint=explicit_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=norm_venue,
                currency_hint=norm_ccy,
                broker_source_hint=norm_broker_source,
                failure_reason=err or ResolutionReason.INVALID_WKN_FORMAT,
            )
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.NORMALIZED,
            raw_query=raw_q,
            normalized_query=norm_wkn,
            inferred_identifier_type=ResolverIdentifierType.WKN,
            query_class=ResolverQueryClass.WKN,
            identifier_type_hint=explicit_hint,
            jurisdiction_hint=norm_jur,
            mic_hint=norm_mic,
            venue_hint=norm_venue,
            currency_hint=norm_ccy,
            broker_source_hint=norm_broker_source,
            parsed_fields=((ResolverIdentifierType.WKN, norm_wkn),),
        )

    if target_type in (ResolverIdentifierType.TICKER, ResolverIdentifierType.TICKER_WITH_LISTING_CONTEXT):
        t_val, parsed_mic, parsed_ccy, t_err = parse_ticker_with_context(raw_q)
        if t_err is not None or t_val is None:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.INVALID_IDENTIFIER,
                raw_query=raw_q,
                normalized_query=folded_q,
                inferred_identifier_type=ResolverIdentifierType.TICKER,
                query_class=ResolverQueryClass.TICKER_WITH_LISTING_CONTEXT,
                identifier_type_hint=explicit_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=norm_venue,
                currency_hint=norm_ccy,
                broker_source_hint=norm_broker_source,
                failure_reason=t_err or ResolutionReason.INVALID_QUERY,
            )
        effective_mic = norm_mic or parsed_mic
        effective_ccy = norm_ccy or parsed_ccy
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.NORMALIZED,
            raw_query=raw_q,
            normalized_query=t_val,
            inferred_identifier_type=ResolverIdentifierType.TICKER,
            query_class=ResolverQueryClass.TICKER_WITH_LISTING_CONTEXT,
            identifier_type_hint=explicit_hint,
            jurisdiction_hint=norm_jur,
            mic_hint=effective_mic,
            venue_hint=norm_venue,
            currency_hint=effective_ccy,
            broker_source_hint=norm_broker_source,
            parsed_fields=((ResolverIdentifierType.TICKER, t_val),),
        )

    if target_type == ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME:
        norm_name = normalize_name_for_share_class(raw_q, normalized_mode=False)
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.NORMALIZED,
            raw_query=raw_q,
            normalized_query=norm_name,
            inferred_identifier_type=ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME,
            query_class=ResolverQueryClass.LEGAL_SHARE_CLASS_NAME,
            identifier_type_hint=explicit_hint,
            jurisdiction_hint=norm_jur,
            mic_hint=norm_mic,
            venue_hint=norm_venue,
            currency_hint=norm_ccy,
            broker_source_hint=norm_broker_source,
            parsed_fields=((ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME, norm_name),),
        )

    if target_type == ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME:
        norm_name = normalize_name_for_share_class(raw_q, normalized_mode=True)
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.NORMALIZED,
            raw_query=raw_q,
            normalized_query=norm_name,
            inferred_identifier_type=ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME,
            query_class=ResolverQueryClass.NORMALIZED_SHARE_CLASS_NAME,
            identifier_type_hint=explicit_hint,
            jurisdiction_hint=norm_jur,
            mic_hint=norm_mic,
            venue_hint=norm_venue,
            currency_hint=norm_ccy,
            broker_source_hint=norm_broker_source,
            parsed_fields=((ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME, norm_name),),
        )

    if target_type == ResolverIdentifierType.BROKER_ALIAS:
        norm_alias = normalize_broker_alias_text(raw_q)
        if not norm_broker_source:
            return NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.UNSUPPORTED_QUERY_TYPE,
                raw_query=raw_q,
                normalized_query=norm_alias,
                inferred_identifier_type=ResolverIdentifierType.BROKER_ALIAS,
                query_class=ResolverQueryClass.BROKER_ALIAS,
                identifier_type_hint=explicit_hint,
                jurisdiction_hint=norm_jur,
                mic_hint=norm_mic,
                venue_hint=norm_venue,
                currency_hint=norm_ccy,
                broker_source_hint=None,
                failure_reason=ResolutionReason.NO_APPLICABLE_ADAPTER,
            )
        return NormalizedETFIdentityQuery(
            outcome=NormalizationOutcome.NORMALIZED,
            raw_query=raw_q,
            normalized_query=norm_alias,
            inferred_identifier_type=ResolverIdentifierType.BROKER_ALIAS,
            query_class=ResolverQueryClass.BROKER_ALIAS,
            identifier_type_hint=explicit_hint,
            jurisdiction_hint=norm_jur,
            mic_hint=norm_mic,
            venue_hint=norm_venue,
            currency_hint=norm_ccy,
            broker_source_hint=norm_broker_source,
            parsed_fields=((ResolverIdentifierType.BROKER_ALIAS, norm_alias),),
        )

    return NormalizedETFIdentityQuery(
        outcome=NormalizationOutcome.UNSUPPORTED_QUERY_TYPE,
        raw_query=raw_q,
        normalized_query=folded_q,
        inferred_identifier_type=target_type,
        query_class=None,
        identifier_type_hint=explicit_hint,
        jurisdiction_hint=norm_jur,
        mic_hint=norm_mic,
        venue_hint=norm_venue,
        currency_hint=norm_ccy,
        broker_source_hint=norm_broker_source,
        failure_reason=ResolutionReason.NO_APPLICABLE_ADAPTER,
    )
