"""
scripts/research/etf_v2/global_identity_models.py

Jurisdiction-neutral ETF identity domain models for Pipeline V2.
Enforces the strict invariant:
    instrument != share_class != listing
Provides deterministic serialization, immutable canonical internal IDs,
and explicit typing across jurisdictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Tuple


class UnsupportedJurisdictionError(Exception):
    """Raised when authority routing or resolution is requested for an unsupported or unknown jurisdiction."""
    pass


class CanonicalIdCollisionError(Exception):
    """Raised when a canonical internal ID collision occurs with conflicting attributes."""
    pass


class InvalidIdentifierError(Exception):
    """Raised when an identifier string fails syntactic or cryptographic validation."""
    pass


class Jurisdiction(str, Enum):
    """Supported regulatory and operational jurisdictions."""
    US_SEC = "US_SEC"
    EU_UCITS = "EU_UCITS"
    UK_UCITS = "UK_UCITS"
    UNKNOWN = "UNKNOWN"

    def is_supported(self) -> bool:
        """Returns True if this jurisdiction has an authoritative regulatory framework."""
        return self in (Jurisdiction.US_SEC, Jurisdiction.EU_UCITS, Jurisdiction.UK_UCITS)

    def assert_supported(self) -> None:
        """Fails closed if the jurisdiction is unknown or unsupported. Never silently defaults to US."""
        if not self.is_supported():
            raise UnsupportedJurisdictionError(
                f"Unsupported or unknown regulatory jurisdiction: {self.value}. "
                "Authority routing cannot default to US_SEC."
            )


class IdentifierType(str, Enum):
    """Explicit identifier classifications preventing untyped string conflation."""
    INTERNAL_INSTRUMENT_ID = "INTERNAL_INSTRUMENT_ID"
    INTERNAL_SHARE_CLASS_ID = "INTERNAL_SHARE_CLASS_ID"
    INTERNAL_LISTING_ID = "INTERNAL_LISTING_ID"
    ISIN = "ISIN"
    WKN = "WKN"
    CIK = "CIK"
    SEC_SERIES_ID = "SEC_SERIES_ID"
    SEC_CLASS_ID = "SEC_CLASS_ID"
    TICKER = "TICKER"
    MIC = "MIC"
    SEDOL = "SEDOL"
    CUSIP = "CUSIP"


class IdentityStatus(str, Enum):
    """
    Pure identity-domain resolution state.
    Does NOT imply mandate established, classification established, or predictive validity.
    """
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ETFListing:
    """
    Represents a specific venue listing of an ETF share class.
    Ticker is strictly listing metadata and NOT global ETF identity.
    Deduplication key: (share_class_id, venue_mic, trading_currency).
    """
    listing_id: str
    share_class_id: str
    venue_mic: str
    ticker: str
    trading_currency: str
    venue_name: str = ""
    local_code: Optional[str] = None  # e.g. WKN for German venues (TGAT/XETR)
    broker_aliases: Tuple[str, ...] = field(default_factory=tuple)
    identity_status: IdentityStatus = IdentityStatus.RESOLVED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def deduplication_key(self) -> Tuple[str, str, str]:
        """Canonical tuple key for listing deduplication: (share_class_id, venue_mic, trading_currency)."""
        return (self.share_class_id, self.venue_mic, self.trading_currency)

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic dictionary serialization with sorted keys."""
        return {
            "listing_id": self.listing_id,
            "share_class_id": self.share_class_id,
            "venue_mic": self.venue_mic,
            "ticker": self.ticker,
            "trading_currency": self.trading_currency,
            "venue_name": self.venue_name,
            "local_code": self.local_code,
            "broker_aliases": list(self.broker_aliases),
            "identity_status": self.identity_status.value,
            "metadata": dict(sorted(self.metadata.items())),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ETFListing:
        """Reconstruct ETFListing from deterministic dictionary."""
        return cls(
            listing_id=data["listing_id"],
            share_class_id=data["share_class_id"],
            venue_mic=data["venue_mic"],
            ticker=data["ticker"],
            trading_currency=data["trading_currency"],
            venue_name=data.get("venue_name", ""),
            local_code=data.get("local_code"),
            broker_aliases=tuple(data.get("broker_aliases", [])),
            identity_status=IdentityStatus(data.get("identity_status", IdentityStatus.RESOLVED.value)),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class ETFShareClass:
    """
    Represents a distinct share class / tranche of an ETF instrument.
    Distinct legal share classes (e.g. accumulating vs. distributing, currency hedged)
    must never be merged solely because legal fund names match.
    """
    share_class_id: str
    instrument_id: str
    isin: Optional[str] = None
    share_class_name: str = ""
    distribution_policy: str = ""  # ACCUMULATING, DISTRIBUTING
    base_currency: str = ""        # USD, EUR, GBP
    hedging_policy: str = ""       # UNHEDGED, HEDGED
    sec_class_id: Optional[str] = None  # Native US SEC Class ID if applicable
    identity_status: IdentityStatus = IdentityStatus.RESOLVED
    listings: Tuple[ETFListing, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic dictionary serialization with sorted keys."""
        return {
            "share_class_id": self.share_class_id,
            "instrument_id": self.instrument_id,
            "isin": self.isin,
            "share_class_name": self.share_class_name,
            "distribution_policy": self.distribution_policy,
            "base_currency": self.base_currency,
            "hedging_policy": self.hedging_policy,
            "sec_class_id": self.sec_class_id,
            "identity_status": self.identity_status.value,
            "listings": [listing.to_dict() for listing in self.listings],
            "metadata": dict(sorted(self.metadata.items())),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ETFShareClass:
        """Reconstruct ETFShareClass from deterministic dictionary."""
        listings = tuple(ETFListing.from_dict(item) for item in data.get("listings", []))
        return cls(
            share_class_id=data["share_class_id"],
            instrument_id=data["instrument_id"],
            isin=data.get("isin"),
            share_class_name=data.get("share_class_name", ""),
            distribution_policy=data.get("distribution_policy", ""),
            base_currency=data.get("base_currency", ""),
            hedging_policy=data.get("hedging_policy", ""),
            sec_class_id=data.get("sec_class_id"),
            identity_status=IdentityStatus(data.get("identity_status", IdentityStatus.RESOLVED.value)),
            listings=listings,
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class ETFInstrument:
    """
    Represents a legal or canonical fund-level ETF instrument.
    Independent of ticker, exchange, broker symbol, or individual listings.
    Supports 1 instrument -> N share classes -> M listings.
    """
    canonical_instrument_id: str
    legal_fund_name: str
    domicile_iso2: str
    regulatory_jurisdiction: Jurisdiction
    fund_family: str = ""
    issuer: str = ""
    fund_structure: str = ""
    identity_status: IdentityStatus = IdentityStatus.RESOLVED
    share_classes: Tuple[ETFShareClass, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Deterministic dictionary serialization with sorted keys."""
        return {
            "canonical_instrument_id": self.canonical_instrument_id,
            "legal_fund_name": self.legal_fund_name,
            "domicile_iso2": self.domicile_iso2,
            "regulatory_jurisdiction": self.regulatory_jurisdiction.value,
            "fund_family": self.fund_family,
            "issuer": self.issuer,
            "fund_structure": self.fund_structure,
            "identity_status": self.identity_status.value,
            "share_classes": [sc.to_dict() for sc in self.share_classes],
            "metadata": dict(sorted(self.metadata.items())),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Deterministic canonical JSON serialization with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ETFInstrument:
        """Reconstruct ETFInstrument from deterministic dictionary."""
        share_classes = tuple(ETFShareClass.from_dict(item) for item in data.get("share_classes", []))
        return cls(
            canonical_instrument_id=data["canonical_instrument_id"],
            legal_fund_name=data["legal_fund_name"],
            domicile_iso2=data["domicile_iso2"],
            regulatory_jurisdiction=Jurisdiction(data["regulatory_jurisdiction"]),
            fund_family=data.get("fund_family", ""),
            issuer=data.get("issuer", ""),
            fund_structure=data.get("fund_structure", ""),
            identity_status=IdentityStatus(data.get("identity_status", IdentityStatus.RESOLVED.value)),
            share_classes=share_classes,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ETFInstrument:
        """Reconstruct ETFInstrument from JSON string."""
        return cls.from_dict(json.loads(json_str))
